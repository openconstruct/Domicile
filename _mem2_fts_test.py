#!/usr/bin/env python3
"""Focused test for the mem2 FTS5 hybrid layer. Does not touch qdrant."""
import sys, time
from pathlib import Path
import mem2_server as m

# --- 1. match-query builder --------------------------------------------------
cases = {
    "how did we fix the compaction ceiling": '"did"... ',  # loose check below
    "openrouter.py:153 AsyncOpenAI timeout": None,
    "the a an and or of": None,   # all stopwords -> None
    "!!! ??? ...": None,          # no tokens -> None
}
q = m._fts_match_query("openrouter.py:153 AsyncOpenAI timeout")
assert q == '"openrouter" OR "py" OR "153" OR "asyncopenai" OR "timeout"', q
assert m._fts_match_query("the a an and or of") is None
assert m._fts_match_query("!!! ???") is None
assert m._fts_match_query("how did we fix the compaction ceiling") == \
    '"fix" OR "compaction" OR "ceiling"'
print("1. match-query builder OK")

# --- 2. build FTS index for real -------------------------------------------
t0 = time.time()
db = m._get_fts()
assert db is not None, "FTS5 not available"
state = m._load_fts_state()
log_files = sorted(m.LOG_DIR.glob("*.log"))
files = []
for p in log_files:
    mt = p.stat().st_mtime
    if state.get(str(p), 0) < mt:
        files.append((str(p), m._parse_log(p)))
        state[str(p)] = mt
if files:
    m._fts_write(files)
    m._save_fts_state()
n = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
print(f"2. FTS build OK: {n} rows from {len(files)} changed files in {time.time()-t0:.1f}s")

# --- 3. exact-string keyword search --------------------------------------
for probe in ["compaction ceiling", "COMPACT_BASE_PREAMBLE", "qdrant",
              "core-memories.md", "ddgs"]:
    hits = m._fts_search(probe, 5, None, None)
    print(f"   kw {probe!r:32} -> {len(hits)} hits"
          + (f"  top: {hits[0]['date']} {hits[0]['chat_title'][:40]!r}" if hits else ""))

# --- 4. idempotent rebuild (no dupes) -----------------------------------
sf, chunks = files[0] if files else (str(log_files[-1]), m._parse_log(log_files[-1]))
before = db.execute("SELECT count(*) FROM chunks WHERE source_file=?", (sf,)).fetchone()[0]
m._fts_write([(sf, m._parse_log(Path(sf)))])
after = db.execute("SELECT count(*) FROM chunks WHERE source_file=?", (sf,)).fetchone()[0]
assert before == after, f"dupe: {before} -> {after}"
print(f"4. idempotent rewrite OK ({sf.split('/')[-1]}: {after} rows stable)")

# --- 5. RRF merge shape --------------------------------------------------
semantic = [{"point_id": i, "text": f"s{i}", "date": "2026-08-01",
             "chat_title": "x", "source_file": "f", "chunk_index": i, "_native": 0.9}
            for i in [1, 2, 3, 4]]
keyword = [{"point_id": i, "text": f"k{i}", "date": "2026-08-01",
            "chat_title": "x", "source_file": "f", "chunk_index": i, "_native": -1.0}
           for i in [3, 5, 1]]
fused = {}
for retr, hits in (("sem", semantic), ("kw", keyword)):
    for rank, hit in enumerate(hits):
        pid = hit["point_id"]
        slot = fused.setdefault(pid, {"hit": hit, "score": 0.0, "via": []})
        slot["score"] += 1.0 / (m.RRF_K + rank)
        slot["via"].append(retr)
ranked = sorted(fused.values(), key=lambda s: s["score"], reverse=True)
top = ranked[0]
assert top["hit"]["point_id"] in (1, 3), top["hit"]["point_id"]
assert set(top["via"]) == {"sem", "kw"}
print(f"5. RRF merge OK: winner pid={top['hit']['point_id']} via={top['via']} "
      f"score={top['score']:.4f}")

print("\nALL PASS")
