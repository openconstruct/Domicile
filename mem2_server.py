#!/usr/bin/env python3
"""Transcript search through mem2 MCP tools."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("MEM0_TELEMETRY", "false")

from mcp.server import MCPServer

from paths import MEM0_DIR as MEM2_DIR, QDRANT_MEM2_PATH as QDRANT_PATH, \
    MEM2_FTS_PATH as FTS_PATH, MEM2_INDEX_STATE as INDEX_STATE, \
    MEM2_FTS_STATE as FTS_STATE, MEM0_LOG_DIR as LOG_DIR, \
    INGESTED_DIR


def _log_paths() -> list[Path]:
    """The mem2 corpus: settled, already-ingested transcripts only."""
    if not INGESTED_DIR.exists():
        return []
    return sorted(INGESTED_DIR.glob("*.log"))
COLLECTION = "cchat-mem2"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
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
_client = None
_indexed_files: dict[str, float] | None = None  # path -> mtime when last indexed
_collection_ready = False

_fts_db: sqlite3.Connection | None = None
_fts_ok: bool | None = None            # None = untried, True/False = FTS5 usable
_fts_indexed: dict[str, float] | None = None  # path -> mtime when last FTS-indexed


def _load_index_state() -> dict[str, float]:
    global _indexed_files
    if _indexed_files is None:
        try:
            _indexed_files = json.loads(INDEX_STATE.read_text())
        except Exception:
            _indexed_files = {}
    return _indexed_files


def _save_index_state() -> None:
    try:
        INDEX_STATE.parent.mkdir(parents=True, exist_ok=True)
        tmp = INDEX_STATE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(_indexed_files or {}, indent=1))
        tmp.replace(INDEX_STATE)
    except Exception:
        pass  # a lost cursor costs a re-index, not data


def _point_id(source_file: str, chunk_index: int) -> int:
    digest = hashlib.md5(f"{source_file}:{chunk_index}".encode()).hexdigest()
    return int(digest[:15], 16)


def _load_fts_state() -> dict[str, float]:
    global _fts_indexed
    if _fts_indexed is None:
        try:
            _fts_indexed = json.loads(FTS_STATE.read_text())
        except Exception:
            _fts_indexed = {}
    return _fts_indexed


def _save_fts_state() -> None:
    try:
        FTS_STATE.parent.mkdir(parents=True, exist_ok=True)
        tmp = FTS_STATE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(_fts_indexed or {}, indent=1))
        tmp.replace(FTS_STATE)
    except Exception:
        pass


def _get_fts() -> sqlite3.Connection | None:
    global _fts_db, _fts_ok
    if _fts_ok is False:
        return None
    if _fts_db is not None:
        return _fts_db
    try:
        FTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(FTS_PATH), check_same_thread=False)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
            "  text, chat_title,"
            "  date UNINDEXED, chat_id UNINDEXED,"
            "  source_file UNINDEXED, chunk_index UNINDEXED, point_id UNINDEXED,"
            "  tokenize='unicode61'"
            ")"
        )
        db.commit()
        _fts_db, _fts_ok = db, True
        return db
    except sqlite3.OperationalError as exc:
        sys.stderr.write(f"mem2: FTS5 unavailable ({exc}); semantic-only\n")
        _fts_ok = False
        return None


def _fts_write(files: list[tuple[str, list[dict]]]) -> None:
    """Rewrite FTS rows for the given (source_file, chunks) pairs."""
    db = _get_fts()
    if db is None:
        return
    with db:
        for source_file, chunks in files:
            db.execute("DELETE FROM chunks WHERE source_file = ?", (source_file,))
            db.executemany(
                "INSERT INTO chunks"
                " (text, chat_title, date, chat_id, source_file, chunk_index, point_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (c["text"], c["chat_title"], c["date"], c["chat_id"],
                     c["source_file"], c["chunk_index"],
                     _point_id(c["source_file"], c["chunk_index"]))
                    for c in chunks
                ],
            )


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


def _get_client():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        _client = QdrantClient(path=str(QDRANT_PATH))
    return _client


def _ensure_collection():
    global _collection_ready
    if _collection_ready:
        return
    client = _get_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={"size": EMBED_DIMS, "distance": "Cosine"},
        )
    _collection_ready = True


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
    _ensure_collection()

    q_state = _load_index_state()
    fts_state = _load_fts_state()
    fts_available = _get_fts() is not None
    log_files = _log_paths()

    q_pending: list[tuple[str, float, list[dict]]] = []   # (key, mtime, chunks)
    fts_files: list[tuple[str, list[dict]]] = []          # per-file chunks needing FTS rewrite
    for path in log_files:
        mtime = path.stat().st_mtime
        key = str(path)
        q_stale = q_state.get(key, 0) < mtime
        fts_stale = fts_available and fts_state.get(key, 0) < mtime
        if not q_stale and not fts_stale:
            continue
        chunks = _parse_log(path)
        if q_stale:
            q_pending.append((key, mtime, chunks))
        if fts_stale:
            fts_files.append((key, chunks))
            fts_state[key] = mtime

    if q_pending:
        from qdrant_client.models import PointStruct
        embedder = _get_embedder()
        client = _get_client()
        total = sum(len(c) for _, _, c in q_pending)
        done = 0
        group: list[tuple[str, float, list[dict]]] = []
        group_n = 0

        def _flush(grp):
            nonlocal done
            if not grp:
                return
            chunks = [c for _, _, cs in grp for c in cs]
            vectors = list(embedder.embed([c["text"] for c in chunks]))
            points = [
                PointStruct(
                    id=_point_id(ch["source_file"], ch["chunk_index"]),
                    vector=vec,
                    payload={
                        "text": ch["text"], "date": ch["date"],
                        "chat_id": ch["chat_id"], "chat_title": ch["chat_title"],
                        "source_file": ch["source_file"],
                        "chunk_index": ch["chunk_index"],
                    },
                )
                for ch, vec in zip(chunks, vectors)
            ]
            client.upsert(collection_name=COLLECTION, points=points)
            for k, mt, cs in grp:
                q_state[k] = mt
            _save_index_state()
            done += len(chunks)
            print(f"  mem2 index: {done}/{total} chunks", flush=True)

        for item in q_pending:
            group.append(item)
            group_n += len(item[2])
            if group_n >= _EMBED_BATCH:
                _flush(group)
                group, group_n = [], 0
        _flush(group)

    if fts_files:
        _fts_write(fts_files)
        _save_fts_state()


# search

def _semantic_search(query: str, limit: int, date_from: str | None,
                     date_to: str | None) -> list[dict]:
    """qdrant / bge pass. Returns normalised chunk dicts, best first."""
    embedder = _get_embedder()
    client = _get_client()
    q_vector = list(embedder.embed([query]))[0]

    must = []
    if date_from:
        must.append({"key": "date", "range": {"gte": date_from}})
    if date_to:
        must.append({"key": "date", "range": {"lte": date_to}})
    from qdrant_client.models import Filter
    query_filter = Filter(must=must) if must else None

    results = client.query_points(
        collection_name=COLLECTION,
        query=q_vector,
        limit=int(limit),
        query_filter=query_filter,
    )
    out = []
    for hit in results.points:
        p = dict(hit.payload or {})
        p["point_id"] = hit.id
        p["_native"] = float(hit.score)
        out.append(p)
    return out


def _fts_search(query: str, limit: int, date_from: str | None,
                date_to: str | None) -> list[dict]:
    db = _get_fts()
    if db is None:
        return []
    match = _fts_match_query(query)
    if not match:
        return []

    sql = ("SELECT point_id, text, date, chat_id, chat_title, source_file,"
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
    for row in db.execute(sql, params):
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
    client = _get_client()

    from qdrant_client.models import Filter, FieldCondition, MatchValue
    results, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key="source_file", match=MatchValue(value=source_file)),
            FieldCondition(key="chunk_index", match=MatchValue(value=int(chunk_index))),
        ]),
        limit=1,
    )

    if not results:
        return f"Chunk not found: {source_file}#{chunk_index}"

    point = results[0]
    p = point.payload
    return (
        f"date: {p.get('date', '?')}\n"
        f"chat: {p.get('chat_title', '?')}\n"
        f"---\n"
        f"{p.get('text', '')}"
    )


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


# selftest

def _selftest() -> int:
    """Direct-mode smoke test. Indexes real logs, searches, prints results."""
    print("mem2 selftest")
    print(f"  log dir: {INGESTED_DIR}")
    print(f"  qdrant:  {QDRANT_PATH}")
    print(f"  fts:     {FTS_PATH}")

    logs = _log_paths()
    print(f"  log files: {len(logs)}")
    if not logs:
        print("  SKIP: no log files")
        return 0

    print("  indexing...")
    _index_logs()
    client = _get_client()
    info = client.get_collection(COLLECTION)
    print(f"  qdrant points: {info.points_count}")

    db = _get_fts()
    if db is None:
        print("  FTS5: UNAVAILABLE (semantic-only fallback)")
    else:
        n = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        print(f"  fts rows: {n}")
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
    mcp.run()
