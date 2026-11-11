#!/usr/bin/env python3
"""Transcript search through mem2 MCP tools."""

from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import sys
import threading
from datetime import date
from pathlib import Path

os.environ.setdefault("MEM0_TELEMETRY", "false")

from mcp.server import MCPServer

import numpy as np

from paths import MEM2_DB_PATH as DB_PATH, QDRANT_MEM2_PATH as LEGACY_QDRANT, \
    MEM0_LOG_DIR as LOG_DIR, INGESTED_DIR


def _log_paths() -> list[Path]:
    """The mem2 corpus: settled, already-ingested transcripts only."""
    if not INGESTED_DIR.exists():
        return []
    return sorted(INGESTED_DIR.glob("*.log"))
# The model that made the stored vectors is stamped in the db and checked on
# open; changing this needs `mem2_server.py --reembed`, never a plain restart.
EMBED_MODEL = os.environ.get("MEM2_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIMS = 384
CHUNK_TARGET_TOKENS = 800  # ~3200 chars, natural split at compaction boundaries
_EMBED_BATCH = 256
RRF_K = 60  # reciprocal-rank-fusion damping; 60 is the standard default

RECENCY_BONUS = 0.6 / RRF_K
RECENCY_DAYS = 64


def _recency_bonus(stamp: str) -> float:
    try:
        age = (date.today() - date.fromisoformat((stamp or "")[:10])).days
    except ValueError:
        return 0.0
    if age <= 0:
        return RECENCY_BONUS
    if age >= RECENCY_DAYS:
        return 0.0
    step = RECENCY_BONUS / max(1.0, math.log2(RECENCY_DAYS))
    return max(0.0, RECENCY_BONUS - step * math.floor(math.log2(age)))

_FTS_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "it", "its", "for", "on", "with", "that", "this", "as", "at", "be", "by",
    "from", "we", "i", "you", "he", "she", "they", "do", "did", "does", "not",
    "what", "when", "where", "why", "how", "which", "there", "here", "about",
}

mcp = MCPServer("mem2")

# indexing

_embedder = None
_db: sqlite3.Connection | None = None
_lock = threading.RLock()
_matrix = None  # (point_ids int64[n], dates str[n], vectors float32[n, d]) or None = reload


def _point_id(source_file: str, chunk_index: int) -> int:
    digest = hashlib.md5(f"{source_file}:{chunk_index}".encode()).hexdigest()
    return int(digest[:15], 16)  # 60 bits: fits sqlite's signed int64 rowid


# One sqlite file holds everything. `chunks` is the FTS5 keyword index and the
# text store; its rowid IS the point id, so lookups by id never scan. `vecs`
# holds one unit-normalised float32 vector per chunk. `files` is the index
# cursor, written in the same transaction as the rows it describes, so a crash
# cannot leave text and vectors disagreeing about what was indexed.
_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
    text, chat_title,
    date UNINDEXED, chat_id UNINDEXED,
    source_file UNINDEXED, chunk_index UNINDEXED,
    tokenize='unicode61');
CREATE TABLE IF NOT EXISTS vecs (
    point_id INTEGER PRIMARY KEY,
    source_file TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    date TEXT NOT NULL,
    embedding BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS idx_vecs_file ON vecs(source_file);
CREATE TABLE IF NOT EXISTS files (source_file TEXT PRIMARY KEY, mtime REAL NOT NULL);
CREATE TABLE IF NOT EXISTS store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def open_db(path: Path, embed_model: str | None = None) -> sqlite3.Connection:
    """Open (or create) a mem2 db and check its embedder stamp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    try:
        db.executescript(_SCHEMA)
    except sqlite3.OperationalError as exc:
        db.close()
        raise RuntimeError(f"mem2 needs SQLite with FTS5: {exc}") from exc
    if embed_model is not None:
        check_embedder(db, embed_model)
    return db


def get_meta(db: sqlite3.Connection, key: str) -> str | None:
    row = db.execute("SELECT value FROM store_meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(db: sqlite3.Connection, key: str, value: str) -> None:
    db.execute("INSERT INTO store_meta(key, value) VALUES (?, ?) "
               "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def check_embedder(db: sqlite3.Connection, name: str) -> None:
    # Dim cannot tell bge-small from MiniLM-L6 (both 384), so the model name is
    # the only guard against ranking one model's query against another's rows.
    have = get_meta(db, "embedder")
    if have == name:
        return
    n = db.execute("SELECT count(*) FROM vecs").fetchone()[0]
    if have is None and n == 0:
        with db:
            set_meta(db, "embedder", name)  # fresh store: first embedder wins
        return
    raise RuntimeError(
        f"mem2 embedder mismatch: {DB_PATH.name} vectors were made by {have!r}, "
        f"this process embeds with {name!r}. Run `python3 mem2_server.py --reembed`.")


def _get_db() -> sqlite3.Connection:
    global _db
    if _db is None:
        if not DB_PATH.exists() and (LEGACY_QDRANT / "collection").exists():
            raise RuntimeError(
                f"mem2 has a legacy qdrant store at {LEGACY_QDRANT} and no {DB_PATH.name}. "
                f"Run `python3 mem2_migrate.py` once to carry its vectors over.")
        _db = open_db(DB_PATH, EMBED_MODEL)
    return _db


def _fts_match_query(query: str) -> str | None:
    """Turn a natural-language query into an FTS5 MATCH expression."""
    toks = re.findall(r"[A-Za-z0-9_]+", query.lower())
    seen, uniq = set(), []
    for t in toks:
        if len(t) < 2 or t in _FTS_STOPWORDS or t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    if not uniq:
        return None
    return " OR ".join(f'"{t}"' for t in uniq)


def _get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        n = os.cpu_count() or 4
        _embedder = TextEmbedding(model_name=EMBED_MODEL, threads=n)
    return _embedder


def _unit(vectors) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def _embed(texts: list[str]) -> np.ndarray:
    return _unit(list(_get_embedder().embed(texts)))


def _load_matrix():
    global _matrix
    with _lock:
        if _matrix is None:
            rows = _get_db().execute(
                "SELECT point_id, date, embedding FROM vecs ORDER BY point_id").fetchall()
            ids = np.array([r[0] for r in rows], dtype=np.int64)
            dates = np.array([r[1] for r in rows], dtype=object)
            vecs = (np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
                    if rows else np.empty((0, 0), dtype=np.float32))
            _matrix = (ids, dates, vecs)
        return _matrix


def write_file_chunks(db: sqlite3.Connection, source_file: str, mtime: float,
                      chunks: list[dict], vectors: np.ndarray) -> None:
    """Replace one file's rows (text + vectors + cursor) in one transaction."""
    assert len(chunks) == len(vectors), (source_file, len(chunks), len(vectors))
    with db:
        delete_file(db, source_file)
        db.executemany(
            "INSERT INTO chunks (rowid, text, chat_title, date, chat_id, source_file, chunk_index)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(_point_id(c["source_file"], c["chunk_index"]), c["text"], c["chat_title"],
              c["date"], c["chat_id"], c["source_file"], c["chunk_index"]) for c in chunks])
        db.executemany(
            "INSERT INTO vecs (point_id, source_file, chunk_index, date, embedding)"
            " VALUES (?, ?, ?, ?, ?)",
            [(_point_id(c["source_file"], c["chunk_index"]), c["source_file"],
              c["chunk_index"], c["date"], np.asarray(v, dtype=np.float32).tobytes())
             for c, v in zip(chunks, vectors)])
        db.execute("INSERT INTO files(source_file, mtime) VALUES (?, ?) "
                   "ON CONFLICT(source_file) DO UPDATE SET mtime=excluded.mtime",
                   (source_file, mtime))


def delete_file(db: sqlite3.Connection, source_file: str) -> None:
    # chunks.source_file is UNINDEXED, so go through vecs' index to the rowids.
    db.execute("DELETE FROM chunks WHERE rowid IN "
               "(SELECT point_id FROM vecs WHERE source_file=?)", (source_file,))
    db.execute("DELETE FROM vecs WHERE source_file=?", (source_file,))
    db.execute("DELETE FROM files WHERE source_file=?", (source_file,))


def _split_text(text: str, target_chars: int = CHUNK_TARGET_TOKENS * 4) -> list[str]:
    """Split a segment that exceeds the embedder's window."""
    if len(text) <= target_chars:
        return [text]
    for pattern in (r"(?m)^(?=### \[)", r"\n\n"):
        pieces = re.split(pattern, text)
        if len(pieces) < 2:
            continue
        out, buf = [], ""
        for piece in pieces:
            candidate = piece if not buf else buf + ("" if pattern.startswith("(?m)") else "\n\n") + piece
            if len(candidate) > target_chars and buf:
                out.append(buf)
                buf = piece
            else:
                buf = candidate
        if buf:
            out.append(buf)
        result = []
        for chunk in out:
            if len(chunk) > target_chars:
                result.extend(chunk[i:i + target_chars]
                              for i in range(0, len(chunk), target_chars))
            else:
                result.append(chunk)
        return result
    return [text[i:i + target_chars] for i in range(0, len(text), target_chars)]


def _parse_log(path: Path) -> list[dict]:
    """Parse a log file into chunks, one per compaction event."""
    text = path.read_text(encoding="utf-8")
    chunks = []
    parts = re.split(r"(?m)^(=== compacted from chat .* ===)$", text)

    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        content = parts[i + 1].strip()
        if not content:
            continue
        marker = parts[i]
        m = re.search(r"chat '([^']+)' \('([^']+)'\)", marker)
        chat_id = m.group(1) if m else "unknown"
        chat_title = m.group(2) if m else "unknown"
        m2 = re.search(r"at (\d{4}-\d{2}-\d{2})", marker)
        date = m2.group(1) if m2 else path.stem

        for piece in _split_text(content):
            chunks.append({
                "text": piece,
                "date": date,
                "chat_id": chat_id,
                "chat_title": chat_title,
                "source_file": str(path),
                "chunk_index": len(chunks),
            })

    if not chunks and text.strip():
        for piece in _split_text(text.strip()):
            chunks.append({
                "text": piece,
                "date": path.stem,
                "chat_id": "unknown",
                "chat_title": path.stem,
                "source_file": str(path),
                "chunk_index": len(chunks),
            })

    return chunks


def _index_logs():
    global _matrix
    db = _get_db()
    corpus = {str(p): p for p in _log_paths()}
    seen = dict(db.execute("SELECT source_file, mtime FROM files"))

    gone = [f for f in seen if f not in corpus]
    pending = []
    for key, path in corpus.items():
        mtime = path.stat().st_mtime
        if seen.get(key, 0) < mtime:
            pending.append((key, mtime, _parse_log(path)))
    if not gone and not pending:
        return

    with _lock:
        if gone:
            with db:
                for f in gone:
                    delete_file(db, f)
        total = sum(len(c) for _, _, c in pending)
        done = 0
        for key, mtime, chunks in pending:
            vectors = (np.vstack([_embed([c["text"] for c in chunks[k:k + _EMBED_BATCH]])
                                  for k in range(0, len(chunks), _EMBED_BATCH)])
                       if chunks else np.empty((0, EMBED_DIMS), dtype=np.float32))
            write_file_chunks(db, key, mtime, chunks, vectors)
            done += len(chunks)
            if total >= _EMBED_BATCH:
                print(f"  mem2 index: {done}/{total} chunks", flush=True)
        _matrix = None


# search

def _semantic_search(query: str, limit: int, date_from: str | None,
                     date_to: str | None) -> list[dict]:
    """Vector pass: exact cosine over the in-RAM matrix. Best first."""
    ids, dates, vecs = _load_matrix()
    if not len(ids):
        return []
    scores = vecs @ _embed([query])[0]
    mask = np.ones(len(ids), dtype=bool)
    if date_from:
        mask &= dates >= date_from
    if date_to:
        mask &= dates <= date_to
    scores = np.where(mask, scores, -np.inf)
    k = min(int(limit), int(mask.sum()))
    if k <= 0:
        return []
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    db = _get_db()
    out = []
    for idx in top:
        pid = int(ids[idx])
        row = db.execute("SELECT text, date, chat_id, chat_title, source_file, chunk_index"
                         " FROM chunks WHERE rowid=?", (pid,)).fetchone()
        if row is None:
            continue
        text, date_, chat_id, chat_title, source_file, chunk_index = row
        out.append({"point_id": pid, "text": text, "date": date_, "chat_id": chat_id,
                    "chat_title": chat_title, "source_file": source_file,
                    "chunk_index": chunk_index, "_native": float(scores[idx])})
    return out


def _fts_search(query: str, limit: int, date_from: str | None,
                date_to: str | None) -> list[dict]:
    match = _fts_match_query(query)
    if not match:
        return []

    sql = ("SELECT rowid, text, date, chat_id, chat_title, source_file,"
           " chunk_index, bm25(chunks) AS rank FROM chunks"
           " WHERE chunks MATCH ?")
    params: list = [match]
    if date_from:
        sql += " AND date >= ?"
        params.append(date_from)
    if date_to:
        sql += " AND date <= ?"
        params.append(date_to)
    sql += " ORDER BY rank LIMIT ?"
    params.append(int(limit))

    out = []
    for row in _get_db().execute(sql, params):
        pid, text, date, chat_id, chat_title, source_file, chunk_index, rank = row
        out.append({
            "point_id": pid,
            "text": text,
            "date": date,
            "chat_id": chat_id,
            "chat_title": chat_title,
            "source_file": source_file,
            "chunk_index": chunk_index,
            "_native": float(rank),  # bm25: lower is better
        })
    return out


def _search(query: str, limit: int = 5, date_from: str = None,
            date_to: str = None) -> str:
    _index_logs()

    limit = int(limit)
    pool = max(limit * 3, 12)  # candidates per retriever before fusion
    semantic = _semantic_search(query, pool, date_from, date_to)
    keyword = _fts_search(query, pool, date_from, date_to)

    fused: dict = {}
    for retriever, hits in (("sem", semantic), ("kw", keyword)):
        for rank, hit in enumerate(hits):
            pid = hit["point_id"]
            slot = fused.get(pid)
            if slot is None:
                slot = fused[pid] = {"hit": hit, "score": 0.0, "via": []}
            slot["score"] += 1.0 / (RRF_K + rank)
            slot["via"].append(retriever)
            if not slot["hit"].get("text") and hit.get("text"):
                slot["hit"] = hit

    if not fused:
        return "No matching conversations found."

    for slot in fused.values():
        slot["score"] += _recency_bonus(slot["hit"].get("date"))

    ranked = sorted(fused.values(), key=lambda s: s["score"], reverse=True)[:limit]

    lines = []
    for i, slot in enumerate(ranked, 1):
        p = slot["hit"]
        text = p.get("text", "")
        if len(text) > 1500:
            text = text[:750] + f"\n... [{len(text) - 1500} chars trimmed] ...\n" + text[-750:]
        via = "+".join(dict.fromkeys(slot["via"]))  # 'sem', 'kw', or 'sem+kw'
        lines.append(
            f"[{i}] via={via} score={slot['score']:.4f} date={p.get('date', '?')} "
            f"chat={p.get('chat_title', '?')} "
            f"ref={p.get('source_file', '?')}#{p.get('chunk_index', '?')}\n{text}"
        )

    return "\n\n".join(lines)


def _list_dates() -> str:
    """List dates with available logs."""
    logs = _log_paths()
    if not logs:
        return "No log files found."
    lines = []
    for path in logs:
        size = path.stat().st_size
        lines.append(f"{path.stem}  ({size:,} bytes)")
    return "\n".join(lines)


def _get_chunk(source_file: str, chunk_index: int) -> str:
    """Fetch a specific chunk by source file and index."""
    _index_logs()
    row = _get_db().execute(
        "SELECT date, chat_title, text FROM chunks WHERE rowid=?",
        (_point_id(source_file, int(chunk_index)),)).fetchone()
    if row is None:
        return f"Chunk not found: {source_file}#{chunk_index}"
    date_, chat_title, text = row
    return f"date: {date_}\nchat: {chat_title}\n---\n{text}"


# MCP tools

def _safe(fn):
    """Return tool failures as text instead of raising."""
    import functools
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            return f"mem2 error: {type(exc).__name__}: {exc}"
    return wrapper


@mcp.tool()
@_safe
def mem2_search(query: str, limit: int = 5, date_from: str = None,
                date_to: str = None) -> str:
    """Search past conversations. Hybrid retrieval: a semantic pass (matches
    meaning / paraphrase) and a keyword pass (matches exact strings — error
    text, PIDs, paths, config values) are merged. Use when mem0 hints aren't
    enough and you need the full context of what was said.

    Args:
        query: Search query. Works for natural-language questions and for
            literal strings like an error message or a file path.
        limit: Max results to return (default 5).
        date_from: Only search logs from this date onward (YYYY-MM-DD).
        date_to: Only search logs up to this date (YYYY-MM-DD).
    """
    return _search(query.strip(), int(limit), date_from, date_to)


@mcp.tool()
@_safe
def mem2_list_dates() -> str:
    """List dates with available conversation logs."""
    return _list_dates()


@mcp.tool()
@_safe
def mem2_get_chunk(source_file: str, chunk_index: int) -> str:
    """Get the full text of a specific conversation chunk. Use for follow-up
    detail after mem2_search returns a result you want to read in full.

    Args:
        source_file: Path before the '#' in a result's ref= field.
        chunk_index: Integer after the '#' in a result's ref= field.
    """
    return _get_chunk(source_file, int(chunk_index))


# selftest / maintenance

def _reembed() -> int:
    """Rewrite every vector with EMBED_MODEL, then restamp, atomically.

    New vectors go to a side table; the swap and the restamp are one
    transaction, so a crash leaves the old vectors and the old stamp intact.
    """
    global _matrix
    db = open_db(DB_PATH)  # no stamp check: the stamp is what we're changing
    old = get_meta(db, "embedder")
    print(f"mem2 reembed: {old!r} -> {EMBED_MODEL!r} in {DB_PATH}")
    db.execute("DROP TABLE IF EXISTS vecs_new")
    db.execute("CREATE TABLE vecs_new AS SELECT * FROM vecs WHERE 0")
    rows = db.execute("SELECT v.point_id, v.source_file, v.chunk_index, v.date, c.text"
                      " FROM vecs v JOIN chunks c ON c.rowid = v.point_id").fetchall()
    for k in range(0, len(rows), _EMBED_BATCH):
        batch = rows[k:k + _EMBED_BATCH]
        vecs = _embed([r[4] for r in batch])
        db.executemany("INSERT INTO vecs_new VALUES (?, ?, ?, ?, ?)",
                       [(r[0], r[1], r[2], r[3], v.tobytes()) for r, v in zip(batch, vecs)])
        db.commit()
        print(f"  {min(k + _EMBED_BATCH, len(rows))}/{len(rows)}", flush=True)
    with db:
        db.execute("DELETE FROM vecs")
        db.execute("INSERT INTO vecs SELECT * FROM vecs_new")
        db.execute("DROP TABLE vecs_new")
        set_meta(db, "embedder", EMBED_MODEL)
    db.close()
    _matrix = None
    print("mem2 reembed: done")
    return 0


def _selftest() -> int:
    """Direct-mode smoke test. Indexes real logs, searches, prints results."""
    print("mem2 selftest")
    print(f"  log dir: {INGESTED_DIR}")
    print(f"  db:      {DB_PATH}")

    logs = _log_paths()
    print(f"  log files: {len(logs)}")
    if not logs:
        print("  SKIP: no log files")
        return 0

    print("  indexing...")
    _index_logs()
    db = _get_db()
    print(f"  embedder: {get_meta(db, 'embedder')}")
    print(f"  vectors: {db.execute('SELECT count(*) FROM vecs').fetchone()[0]}")
    print(f"  fts rows: {db.execute('SELECT count(*) FROM chunks').fetchone()[0]}")
    q = "compaction ceiling"
    print(f"  fts match expr for {q!r}: {_fts_match_query(q)}")

    print("  listing dates:")
    print("  " + _list_dates().replace("\n", "\n  "))

    print("\n  hybrid search: 'compaction ceiling'")
    print(_search("compaction ceiling", limit=3))

    print("\nselftest passed")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    if "--reembed" in sys.argv:
        raise SystemExit(_reembed())
    mcp.run()
