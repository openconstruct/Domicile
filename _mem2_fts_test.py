#!/usr/bin/env python3
"""mem2 (full memory) store: FTS5 + vectors + cursor in one sqlite file.

Run through _test_runner.py: every path resolves under a disposable root, and
the embedder is a deterministic fake, so nothing is downloaded or loaded.
"""
import hashlib
import os
import pickle
import sqlite3
import sys
import types
from pathlib import Path

import numpy as np

import mem2_server as m

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
    "Rooks bent wire into hooks in captivity.\n")
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
sem = m._semantic_search("rooks bent wire into hooks in captivity.", 3, None, None)
check("sem: exact text ranks first", sem and sem[0]["date"] == "2026-10-01"
      and abs(sem[0]["_native"] - 1.0) < 1e-5, str(sem[:1]))
check("sem: point id is the fts rowid",
      sem and sem[0]["point_id"] == m._point_id(str(day2), 0))
only_sep = m._semantic_search("rooks wire hooks", 5, "2026-09-01", "2026-09-30")
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
check("prune: gone from search", "Rooks bent" not in m._search("rooks wire hooks", 5))

# --- 6. embedder stamp -------------------------------------------------------
try:
    m.open_db(m.DB_PATH, "sentence-transformers/all-MiniLM-L6-v2")
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

# --- 8. legacy qdrant dir and no mem2.db -> loud, not a silent empty store ----
for p in (m.DB_PATH, Path(str(m.DB_PATH) + "-wal"), Path(str(m.DB_PATH) + "-shm")):
    p.unlink(missing_ok=True)
(m.LEGACY_QDRANT / "collection").mkdir(parents=True)
try:
    m._get_db()
    check("legacy: refuses without migration", False)
except RuntimeError as exc:
    check("legacy: refuses without migration", "mem2_migrate.py" in str(exc), str(exc))

# --- 9. migration from a qdrant storage.sqlite, without qdrant_client --------
# Pickle stand-ins under qdrant's real module path, then hide the module, so
# the migrator has to decode them the way it would on an install without it.
mod = types.ModuleType("qdrant_client.http.models.models")


class PointStruct:
    def __init__(self, **kw):
        self.__dict__.update(kw)


PointStruct.__module__ = mod.__name__
PointStruct.__qualname__ = "PointStruct"
mod.PointStruct = PointStruct
saved = {n: sys.modules.get(n) for n in (
    "qdrant_client", "qdrant_client.http", "qdrant_client.http.models", mod.__name__)}
for name in list(saved)[:-1]:
    sys.modules[name] = types.ModuleType(name)
sys.modules[mod.__name__] = mod

import mem2_migrate as mg  # noqa: E402

day2.write_text("=== compacted from chat 'abc' ('Main Chat') at 2026-10-01T02:00:00 ===\n\n"
                "Rooks bent wire into hooks in captivity.\n")
store = m.LEGACY_QDRANT / "collection" / "cchat-mem2" / "storage.sqlite"
store.parent.mkdir(parents=True, exist_ok=True)
q = sqlite3.connect(str(store))
q.execute("CREATE TABLE points (id TEXT PRIMARY KEY, point BLOB)")
marker = np.zeros(m.EMBED_DIMS, dtype=np.float32); marker[7] = 3.0  # un-normalised on purpose
old_path = "/old/home/.mem0/logs/ingested/" + day2.name  # pre-move path: id won't match
pts = [PointStruct(id=1, vector=[float(x) for x in marker], payload={
           "text": "Rooks bent wire into hooks in captivity.", "source_file": old_path,
           "chunk_index": 0, "date": "2026-10-01", "chat_id": "abc", "chat_title": "Main Chat"}),
       PointStruct(id=2, vector=[float(x) for x in marker], payload={
           "text": "stale chunk from a file that was later re-chunked", "source_file": old_path,
           "chunk_index": 5, "date": "2026-10-01", "chat_id": "abc", "chat_title": "Main Chat"})]
q.executemany("INSERT INTO points VALUES (?, ?)", [(str(p.id), pickle.dumps(p)) for p in pts])
q.commit(); q.close()
for name, was in saved.items():
    if was is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = was

embedded = []
m._embed = lambda texts: (embedded.extend(texts), fake_embed(texts))[1]
stats = mg.migrate(store, m.DB_PATH)
check("migrate: stats", stats == {"files": 2, "chunks": 4, "reused": 1, "embedded": 3}, str(stats))
check("migrate: reused by text, not id", "Rooks bent wire into hooks in captivity." not in embedded)
db3 = m.open_db(m.DB_PATH, m.EMBED_MODEL)
blob = db3.execute("SELECT embedding FROM vecs WHERE point_id=?",
                   (m._point_id(str(day2), 0),)).fetchone()[0]
v = np.frombuffer(blob, dtype=np.float32)
check("migrate: legacy vector carried over, normalised",
      abs(v[7] - 1.0) < 1e-6 and abs(np.linalg.norm(v) - 1) < 1e-6)
check("migrate: stale point dropped", db3.execute(
    "SELECT count(*) FROM chunks WHERE chunks MATCH 'stale'").fetchone()[0] == 0)
check("migrate: stamped bge", m.get_meta(db3, "embedder") == mg.LEGACY_EMBED_MODEL)
db3.close()
try:
    mg.migrate(store, m.DB_PATH)
    check("migrate: refuses to overwrite", False)
except SystemExit as exc:
    check("migrate: refuses to overwrite", "refusing" in str(exc))
check("migrate: legacy store untouched", store.exists())

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
