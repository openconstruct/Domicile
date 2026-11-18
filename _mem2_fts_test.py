#!/usr/bin/env python3
"""mem2 (full memory) store: FTS5 + vectors + cursor in one sqlite file.

Run through _test_runner.py: every path resolves under a disposable root, and
the embedder is a deterministic fake, so nothing is downloaded or loaded.
"""
import hashlib
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

import mem2_core as m

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def fake_embed(texts):
    out = []
    for t in texts:
        v = np.zeros(m.EMBED_DIMS, dtype=np.float32)
        for tok in t.lower().split():
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % m.EMBED_DIMS] += 1.0
        out.append(v)
    return m._unit(out)


m._embed = fake_embed

# Sandbox guard: this test writes logs and a db; refuse to touch a live tree.
assert os.environ.get("DOMICILE_TEST_SANDBOX"), "run via _test_runner.py"
assert os.environ["DOMICILE_TEST_SANDBOX"] in str(m.DB_PATH), m.DB_PATH

# --- 1. match-query builder ------------------------------------------------
check("match: tokens",
      m._fts_match_query("openrouter.py:153 AsyncOpenAI timeout")
      == '"openrouter" OR "py" OR "153" OR "asyncopenai" OR "timeout"')
check("match: stopwords only", m._fts_match_query("the a an and or of") is None)
check("match: no tokens", m._fts_match_query("!!! ???") is None)
check("match: question", m._fts_match_query("how did we fix the compaction ceiling")
      == '"fix" OR "compaction" OR "ceiling"')

# --- 2. index real-shaped logs ----------------------------------------------
ing = m.INGESTED_DIR
ing.mkdir(parents=True, exist_ok=True)
day1 = ing / "2026-09-01.log"
day1.write_text(
    "=== compacted from chat 'abc' ('Main Chat') at 2026-09-01T01:00:00 ===\n\n"
    "We raised the compaction ceiling to 190k.\n\n"
    "=== compacted from chat 'abc' ('Main Chat') at 2026-09-01T05:00:00 ===\n\n"
    "COMPACT_BASE_PREAMBLE lives in context_diet.py.\n")
day2 = ing / "2026-10-01.log"
day2.write_text(
    "=== compacted from chat 'abc' ('Main Chat') at 2026-10-01T02:00:00 ===\n\n"
    "Crows bent wire into hooks in captivity.\n")
m._index_logs()
db = m._get_db()
nv = db.execute("SELECT count(*) FROM vecs").fetchone()[0]
nc = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
check("index: 3 chunks, text and vectors agree", nv == nc == 3, f"vecs={nv} chunks={nc}")
check("index: stamped", m.get_meta(db, "embedder") == m.EMBED_MODEL)
check("index: cursor rows", db.execute("SELECT count(*) FROM files").fetchone()[0] == 2)

# --- 3. search ---------------------------------------------------------------
kw = m._fts_search("COMPACT_BASE_PREAMBLE", 5, None, None)
check("kw: exact string", kw and "context_diet" in kw[0]["text"])
sem = m._semantic_search("crows bent wire into hooks in captivity.", 3, None, None)
check("sem: exact text ranks first", sem and sem[0]["date"] == "2026-10-01"
      and abs(sem[0]["_native"] - 1.0) < 1e-5, str(sem[:1]))
check("sem: point id is the fts rowid",
      sem and sem[0]["point_id"] == m._point_id(str(day2), 0))
only_sep = m._semantic_search("crows wire hooks", 5, "2026-09-01", "2026-09-30")
check("sem: date filter", only_sep and all(h["date"].startswith("2026-09") for h in only_sep),
      str([h["date"] for h in only_sep]))
check("sem: empty window", m._semantic_search("x", 5, "2030-01-01", None) == [])
out = m._search("compaction ceiling", 3)
check("hybrid: sem+kw fused", "via=sem+kw" in out and "190k" in out, out[:200])

# --- 4. get_chunk ------------------------------------------------------------
got = m._get_chunk(str(day1), 1)
check("get_chunk: by ref", "COMPACT_BASE_PREAMBLE" in got and "Main Chat" in got, got)
check("get_chunk: missing", "not found" in m._get_chunk(str(day1), 99))

# --- 5. reindex is a rewrite, not an append; removed files are pruned --------
day1.write_text(day1.read_text() + "\n=== compacted from chat 'abc' ('Main Chat') at "
                "2026-09-01T09:00:00 ===\n\nA third segment.\n")
os.utime(day1, (day1.stat().st_atime, day1.stat().st_mtime + 5))
m._index_logs()
n_day1 = db.execute("SELECT count(*) FROM vecs WHERE source_file=?", (str(day1),)).fetchone()[0]
check("reindex: day1 rewritten to 3", n_day1 == 3, str(n_day1))
check("reindex: matrix reloaded", len(m._load_matrix()[0]) == 4)
day2.unlink()
m._index_logs()
nv = db.execute("SELECT count(*) FROM vecs").fetchone()[0]
nc = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
check("prune: gone file leaves both tables", nv == nc == 3, f"vecs={nv} chunks={nc}")
check("prune: gone from search", "Crows bent" not in m._search("crows wire hooks", 5))

# --- 6. embedder stamp -------------------------------------------------------
try:
    m.open_db(m.DB_PATH, "test/not-the-stamped-model")
    check("stamp: mismatch raises", False)
except RuntimeError as exc:
    check("stamp: mismatch raises", "--reembed" in str(exc), str(exc))

# --- 7. reembed swaps vectors and stamp together ------------------------------
m._db.close(); m._db = None
old_model = m.EMBED_MODEL
m.EMBED_MODEL = "test/other-model"
m._embed = lambda texts: fake_embed(["x " + t for t in texts])
m._reembed()
db2 = m.open_db(m.DB_PATH, "test/other-model")
check("reembed: restamped", m.get_meta(db2, "embedder") == "test/other-model")
check("reembed: row count kept", db2.execute("SELECT count(*) FROM vecs").fetchone()[0] == 3)
check("reembed: side table gone", db2.execute(
    "SELECT count(*) FROM sqlite_master WHERE name='vecs_new'").fetchone()[0] == 0)
db2.close()
m.EMBED_MODEL = old_model
m._embed = fake_embed

# --- 10. RRF merge shape ----------------------------------------------------
fused = {}
for retr, hits in (("sem", [{"point_id": i} for i in (1, 2, 3, 4)]),
                   ("kw", [{"point_id": i} for i in (3, 5, 1)])):
    for rank, hit in enumerate(hits):
        slot = fused.setdefault(hit["point_id"], {"score": 0.0, "via": []})
        slot["score"] += 1.0 / (m.RRF_K + rank)
        slot["via"].append(retr)
top = max(fused.items(), key=lambda kv: kv[1]["score"])
check("rrf: both-retriever hit wins", top[0] in (1, 3) and set(top[1]["via"]) == {"sem", "kw"})

print("\nALL PASS" if not failures else f"\n{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
