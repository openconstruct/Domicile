"""The memory fact store ("flash memory"): one SQLite file via domicile_mem0.

Public API: hint, search, add, add_verbatim, list_memories, list_facts,
delete, mark_superseded, store. Scoring:
  semantic = top-20 by cosine (>= 0.1) over the user's facts, active only
  keyword  = FTS5 over active facts
  fused by RRF (k=60), then a salience adjustment on last_used / created
  date / use_count (domicile_mem0.retrieval).

Delete policy (owner decision, 2026-11-08): delete is for extraction/integration
only. The chat-facing `delete` op retires a fact (supersede with no
replacement, recoverable); a real removal is purge(), never dispatched.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(1, os.path.expanduser("~/domicile"))

import config_loader as CL  # noqa: E402
from paths import FLASH_DB_PATH, MEM0_DIR  # noqa: E402

from domicile_mem0.core import MemoryCore  # noqa: E402
from domicile_mem0.embedders import EMBED_DIMS, EMBED_MODEL, SharedEmbedder  # noqa: E402,F401
from domicile_mem0.fts import _fts_match_query  # noqa: E402
from domicile_mem0.vectors import normalize  # noqa: E402
from domicile_mem0.retrieval import (  # noqa: E402,F401 -- one copy, shared with mem2
    RRF_K, HINT_LIMIT, HINT_RECENCY_BONUS, HINT_RECENCY_DAYS, HYBRID_RECENCY_BONUS,
    HYBRID_RECENCY_DAYS, HINT_STALE_FLOOR, HINT_STALE_DAYS, HINT_FREQ_CAP, HINT_FREQ_K,
    HYBRID_STALE_FLOOR, HYBRID_STALE_DAYS, HYBRID_FREQ_CAP, HYBRID_FREQ_K, EVENT_WEIGHT,
    chunk_digest, recency_bonus as _recency_bonus, salience_adjust as _salience_adjust)

DB_PATH = Path(os.environ.get("MEM0_V2_DB") or FLASH_DB_PATH)
USER_ID = CL.load_identity()["mem0_user_id"]

HINT_THRESHOLD = 0.50  # MiniLM (2026-11-11); was 0.66 for bge; matched hint volume

class StoreBusy(RuntimeError):
    """Never raised by SQLite (WAL, many readers); kept so callers can `except StoreBusy`."""


# Lessons: corrections and hard-won rules ("Lesson: don't X, because Y"). They
# are a different kind of memory from facts: rarely retrieved, never stale, and
# worth more than a fact at the moment they apply. So they clear the hint bar
# a little lower, are never faded for being old or unused, and get a small
# lift in search.
LESSON_PREFIX = "Lesson:"
LESSON_HINT_MARGIN = 0.05                 # cosine: how far under the hint bar a lesson may sit
LESSON_SEARCH_BONUS = HYBRID_RECENCY_BONUS   # RRF domain: same size as "made today"


def is_lesson(text: str, md: dict | None = None) -> bool:
    return bool((md or {}).get("lesson")) or (text or "").lstrip().startswith(LESSON_PREFIX)


SEM_TOP_K = 20        # mem0 Memory.search default top_k
SEM_THRESHOLD = 0.1   # mem0 Memory.search default threshold

_store_lock = threading.RLock()
_core: MemoryCore | None = None
_llm_ready = False
_seen_keys: set[str] = set()


# --- store ----------------------------------------------------------------

def core(embedder=None) -> MemoryCore:
    global _core
    if _core is None:
        with _store_lock:
            if _core is None:
                DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                _core = MemoryCore(str(DB_PATH), embedder or SharedEmbedder())
    return _core


def store() -> MemoryCore:
    """Open (warm) the store."""
    return core()


def _ensure_llm() -> None:
    """Wire the extraction provider chain. Lazy: search/hint never need it."""
    global _llm_ready
    if _llm_ready:
        return
    import mem0_chain
    chain = mem0_chain.ChainLLM(mem0_chain.MEM0_LLM_CHAIN)
    chain._active()  # fail loudly here, not mid-extraction

    def call(system, user):
        return chain.generate_response(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"})
    c = core()
    c.llm = call
    c.custom_instructions = mem0_chain.CUSTOM_INSTRUCTIONS
    _llm_ready = True


# --- row helpers ----------------------------------------------------------

def _meta(row) -> dict:
    try:
        return json.loads(row["metadata"] or "{}")
    except (TypeError, ValueError):
        return {}


def _created(row, md=None) -> str:
    """Creation date; metadata 'date' wins when it is earlier."""
    md = _meta(row) if md is None else md
    c = row["created_at"]
    stamp = datetime.fromtimestamp(c, timezone.utc).date().isoformat() if c else ""  # mem0 stored UTC
    d = str(md.get("date") or "")[:10]
    if d and d <= (stamp or "9999"):
        stamp = d
    return stamp


def _salience_log(line: str) -> None:
    try:
        p = MEM0_DIR / "logs" / "salience.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  [v2] {line}\n")
    except Exception:
        pass


def _bump_use(ids) -> None:
    """Once per fact per day: use_count += 1, metadata.last_used = today."""
    ids = [str(i) for i in (ids or []) if i]
    if not ids:
        return
    today = date.today().isoformat()
    c = core()
    bumped = []
    try:
        with c.store.lock:
            for mid in ids:
                row = c.store.get(mid)
                if row is None:
                    continue
                md = _meta(row)
                if str(md.get("last_used") or "")[:10] == today:
                    continue
                md["last_used"] = today
                n = int(row["use_count"] or 0) + 1
                c.store.db.execute(
                    "UPDATE memories SET use_count=?, metadata=? WHERE id=?",
                    (n, json.dumps(md), mid))
                bumped.append(f"{mid}(uc={n})")
            c.store.db.commit()  # no version bump: vectors unchanged
        if bumped:
            _salience_log(f"consolidated {len(bumped)} fact(s): " + ", ".join(bumped))
    except Exception as exc:  # never break a search over bookkeeping
        _salience_log(f"bump skipped: {type(exc).__name__}: {exc}")


def _hybrid(query: str, limit: int) -> list[dict]:
    c = core()
    pool = max(limit * 3, 12)
    rows, index, scan, cols = c._snapshot()
    if not rows:
        return []
    q = normalize(c.embedder.embed(query))
    mask = cols["user_id"] == USER_ID
    sims = scan.sims(q, mask=mask, k=SEM_TOP_K, exact=c._exact_vectors(rows))
    sims_m = np.where(mask & (sims >= SEM_THRESHOLD), sims, -np.inf)
    top = [int(i) for i in np.argsort(-sims_m)[:SEM_TOP_K] if np.isfinite(sims_m[i])]
    semantic = top[:pool]
    keyword = []
    match = _fts_match_query(query)
    if match:
        for mid in c.store.keyword(match, pool * 3):
            i = index.get(mid)
            if i is not None and mask[i]:
                keyword.append(i)
            if len(keyword) >= pool:
                break
    fused: dict[int, dict] = {}
    for via, hits in (("sem", semantic), ("kw", keyword)):
        for rank, i in enumerate(hits):
            s = fused.get(i)
            if s is None:
                row = c.store.get(rows[i]["id"]) or rows[i]  # live use_count/last_used
                md = _meta(row)
                s = fused[i] = {
                    "id": row["id"], "memory": (row["text"] or "").strip(),
                    "created_at": _created(row, md),
                    "last_used": str(md.get("last_used") or "")[:10],
                    "use_count": row["use_count"] or 0,
                    "lesson": is_lesson(row["text"], md),
                    "cos": None, "via": [], "score": 0.0}
            s["score"] += 1.0 / (RRF_K + rank)
            s["via"].append(via)
            if via == "sem":
                s["cos"] = float(sims[i])
    ranked = sorted(fused.values(), key=lambda s: -s["score"])
    for s in ranked:
        s["via"] = "+".join(dict.fromkeys(s["via"]))
    return [s for s in ranked if s["memory"]][:limit]


# --- the tools ------------------------------------------------------------

def search(query: str, limit: int = 5) -> str:
    query = (query or "").strip()
    if not query:
        return "empty query"
    limit = max(1, limit)
    ranked = _hybrid(query, max(limit * 3, 12))
    for s in ranked:
        adj = _salience_adjust(
            s.get("last_used"), s["created_at"], s.get("use_count"),
            peak=HYBRID_RECENCY_BONUS, horizon=HYBRID_RECENCY_DAYS,
            floor=HYBRID_STALE_FLOOR, stale_days=HYBRID_STALE_DAYS,
            freq_cap=HYBRID_FREQ_CAP, freq_k=HYBRID_FREQ_K)
        if s["lesson"]:
            adj = max(adj, 0.0) + LESSON_SEARCH_BONUS      # never stale, small lift
        s["score"] += adj
    ranked.sort(key=lambda s: -s["score"])
    ranked = ranked[:limit]
    if not ranked:
        return "no memories matched"
    _bump_use([s["id"] for s in ranked])
    out = []
    for s in ranked:
        cos = s["cos"]
        prefix = f"[{cos:.3f}]" if isinstance(cos, float) else "[kw]"
        # The short id is what memory_source / memory_delete take.
        out.append(f"{prefix} {s['created_at']} ({s['id'][:8]}) {s['memory']}".strip())
    return "\n".join(out)


def hint(text: str, threshold: float = HINT_THRESHOLD,
         limit: int = HINT_LIMIT) -> list[tuple[float, str, str]]:
    text = (text or "").strip()
    if not text:
        return []
    ranked = _hybrid(text, max(limit * 2, 6))
    out = []
    kw_only = 0
    for s in ranked:
        mem, via, cos = s["memory"], s["via"], s["cos"]
        eff = None
        if isinstance(cos, float):
            adj = _salience_adjust(
                s.get("last_used"), s["created_at"], s.get("use_count"),
                peak=HINT_RECENCY_BONUS, horizon=HINT_RECENCY_DAYS,
                floor=HINT_STALE_FLOOR, stale_days=HINT_STALE_DAYS,
                freq_cap=HINT_FREQ_CAP, freq_k=HINT_FREQ_K)
            # A lesson is never faded for age or disuse, and clears a lower bar.
            eff = cos + (max(adj, 0.0) + LESSON_HINT_MARGIN if s["lesson"] else adj)
        if "sem" in via and eff is not None and eff >= threshold:
            out.append((eff, cos, s["created_at"], mem))
        elif via == "sem+kw" and eff is not None and eff >= threshold - 0.15:
            out.append((eff, cos, s["created_at"], mem))
        elif via == "kw" and kw_only < 1:
            out.append((0.0, 0.0, s["created_at"], mem))
            kw_only += 1
        if len(out) >= limit:
            break
    out.sort(key=lambda t: -t[0])
    return [(cos, stamp, mem) for _eff, cos, stamp, mem in out[:limit]]


# Calibrated 2026-11-10 against bge-small: pairs >=0.926 were restatements,
# <=0.916 were distinct facts on one topic. Re-derive after any embedder swap.
GUARD_CUTOFF = 0.865  # MiniLM (2026-11-11): selectivity-matched to bge 0.92 = 0.858, rounded up (under-delete)


def _guard(added: list[dict], cutoff: float) -> None:
    """Drop a just-extracted fact whose nearest
    other fact (top-5, this user) is >= cutoff cosine. Extraction-time
    delete -- allowed by the delete policy."""
    c = core()
    rows, index, scan, cols = c._snapshot()
    if not rows:
        return
    user = cols["user_id"] == USER_ID
    # The snapshot is taken once, so a fact deleted below is still in `matrix`.
    # Track what this pass removed: otherwise two new facts that restate each
    # other each see the other as "existing" and BOTH are deleted.
    dropped: list[int] = []
    for r in added:
        i = index.get(r["id"])
        if i is None:
            continue
        sims = scan.sims(scan.vector(i), mask=user, k=5).copy()
        sims[i] = -np.inf
        if dropped:
            sims[dropped] = -np.inf
        if float(np.max(sims)) >= cutoff:
            c.store.delete(r["id"])
            r["event"] = "GUARDED"
            dropped.append(i)


def add(text: str, extract: bool = True, metadata: dict | None = None,
        idempotency_key: str | None = None, guard: bool = False) -> str:
    text = (text or "").strip()
    if not text:
        return "nothing to store"
    key = idempotency_key or (chunk_digest(text) if not extract else None)
    with _store_lock:
        if key is not None and key in _seen_keys:
            return f"skipped (idempotency key {key} already ingested)"
        if extract:
            _ensure_llm()
        result = core().add(text, user_id=USER_ID, infer=bool(extract),
                            metadata=dict(metadata) if metadata else None)
        r = result.get("results", [])
        if guard and r:
            try:
                _guard(r, cutoff=GUARD_CUTOFF)
            except Exception as exc:  # safety net, never fatal
                print(f"[mem0_v2] guard pass skipped: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
        if key is not None:
            _seen_keys.add(key)
    if not extract:
        return f"stored verbatim: {text[:100]}"
    r = [x for x in r if x.get("event") != "GUARDED"]
    for x in r:                       # the extractor marks lessons by their prefix
        if is_lesson(x.get("memory", "")):
            try:
                core().update(x["id"], metadata={"lesson": True})
            except Exception:  # noqa: BLE001 -- tagging must never fail an ingest
                pass
    if not r:
        return "no new facts (already covered by existing memories)"
    return "\n".join(f"{(x.get('event') or 'ADD').upper()}: {x.get('memory', '')[:120]}"
                     for x in r)


def add_lesson(text: str, metadata: dict | None = None) -> str:
    """Store a lesson verbatim. Returns a status line (the tool-facing op)."""
    text = (text or "").strip()
    if not text:
        return "nothing to store"
    if not text.startswith(LESSON_PREFIX):
        text = f"{LESSON_PREFIX} {text}"
    md = {"kind": "lesson", "date": date.today().isoformat(), **(metadata or {}), "lesson": True}
    with _store_lock:
        if core().store.has_active_hash(hashlib.md5(text.encode("utf-8")).hexdigest(), USER_ID, None, None):
            return "already stored"
        rid = add_verbatim(text, metadata=md)
    _salience_log(f"lesson {rid}")
    return f"stored lesson ({rid[:8]}): {text[:120]}"


def add_verbatim(text: str, metadata: dict | None = None) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("add_verbatim needs non-empty text")
    with _store_lock:
        r = core().add_verbatim(text, user_id=USER_ID,
                                metadata=dict(metadata) if metadata else None)
    rid = (r.get("results") or [{}])[0].get("id")
    if not rid:
        raise RuntimeError(f"add() returned no id for verbatim text: {r}")
    return rid


def _all_rows(include_superseded=True):
    c = core()
    with c.store.lock:
        where = "user_id=?" + ("" if include_superseded else " AND superseded_by IS NULL")
        return [dict(x) for x in c.store.db.execute(
            f"SELECT id, text, metadata, created_at, superseded_by FROM memories "
            f"WHERE {where} ORDER BY created_at DESC", (USER_ID,))]


def list_memories(limit: int = 20) -> str:
    r = _all_rows()
    total = len(r)
    for x in r:
        x["stamp"] = _created(x)
    r.sort(key=lambda x: x["stamp"], reverse=True)
    r = r[: max(1, limit)]
    if not r:
        return "memory store is empty"
    out = [f"{total} memories total, showing {len(r)}:"]
    for x in r:
        tag = ""
        if x["superseded_by"] is not None:
            tag = f" [superseded -> {str(x['superseded_by'] or '?')[:8]}]"
        out.append(f"- {x['stamp']} ({x['id'][:8]}) {x['text']}{tag}")
    return "\n".join(out)


def list_facts(include_superseded: bool = False) -> list[dict]:
    out = []
    for x in _all_rows(include_superseded):
        md = _meta(x)
        out.append({
            "id": str(x["id"]),
            "date": _created(x, md) or "?",
            "kind": md.get("kind", "?"),
            "attr": md.get("attributed_to") or "?",
            "text": (x["text"] or "")[:500],
        })
    return out


def _resolve(prefix: str):
    prefix = (prefix or "").strip()
    if not prefix:
        return None, "no id given"
    c = core()
    with c.store.lock:
        m = c.store.db.execute(
            "SELECT id, text, superseded_by FROM memories WHERE user_id=? AND id LIKE ?",
            (USER_ID, prefix.replace("%", "") + "%")).fetchall()
    if not m:
        return None, f"no memory with id starting {prefix!r}"
    if len(m) > 1:
        return None, f"{prefix!r} matches {len(m)} memories; use a longer id"
    return dict(m[0]), None


def source(ref: str, limit: int = 2) -> str:
    """Where a memory came from: the transcript passage(s) it was extracted
    from. `ref` is an id (or unique prefix), or words to find the memory by."""
    ref = (ref or "").strip()
    if not ref:
        return "give a memory id (from memory_search / memory_list) or some of its words"
    row, err = _resolve(ref)
    if row is None:
        hits = _hybrid(ref, 1)                  # not an id: take the best match for the words
        if not hits:
            return err
        row = {"id": hits[0]["id"]}
    c = core()
    full = c.store.get(row["id"])
    md = _meta(full)
    day = _created(full, md)
    head = f"memory ({full['id'][:8]}, {day or '?'}): {full['text']}"
    if not full.get("embedding"):
        return head + "\nsource: this memory has no stored vector to match against"
    vec = np.frombuffer(full["embedding"], dtype=np.float32)
    import mem2_core
    found = mem2_core.passages_for(vec, day, limit) if day else []
    if found:
        out = [head, f"source: {Path(found[0]['source_file']).name} -- the passage(s) of that "
                     f"day's transcript closest to it:"]
        for n, p in enumerate(found, 1):
            out.append(f"[{n}] cos={p['cosine']:.3f} chat={p['chat_title']} "
                       f"ref={p['source_file']}#{p['chunk_index']}\n{p['text']}")
        out.append("(mem2_get_chunk with the ref's file and the next/previous number reads on)")
        return "\n\n".join(out)
    # Not in full memory yet (today's log is indexed once it settles): read the log itself.
    from paths import INGESTED_DIR, MEM0_LOG_DIR
    for base in (MEM0_LOG_DIR, INGESTED_DIR):
        log = base / f"{day}.log"
        if day and log.is_file():
            import fetch_result
            body = fetch_result.find(log.read_text(encoding="utf-8", errors="replace"),
                                     full["text"], top=limit, budget=3000)
            return f"{head}\n\nsource: {log.name} (not indexed yet; matched by its words)\n{body}"
    origin = md.get("kind") or md.get("source") or "stored directly"
    return (f"{head}\n\nsource: no transcript for {day or 'its date'} "
            f"(origin: {origin}). It was not extracted from a logged conversation, "
            f"or that log is gone.")


def delete(memory_id: str) -> str:
    """Chat-facing delete = retire (recoverable). See purge() for removal."""
    row, err = _resolve(memory_id)
    if err:
        return err
    if row["superseded_by"] is not None:
        return f"already retired/superseded: {row['text'][:80]}"
    core().mark_superseded(row["id"], "", "retired via delete op")
    _salience_log(f"retired {row['id']}")
    return f"deleted (retired, recoverable): {row['text'][:80]}"


def purge(memory_id: str) -> str:
    """Hard removal. Integration/maintenance scripts only; never dispatched."""
    row, err = _resolve(memory_id)
    if err:
        return err
    core().delete(row["id"])
    return f"purged: {row['text'][:80]}"


def mark_superseded(drop_id: str, keep_id: str, reason: str = "",
                    superseded_at: str | None = None) -> str:
    drop_id = (drop_id or "").strip()
    keep_id = (keep_id or "").strip()
    if not drop_id or not keep_id:
        raise ValueError("mark_superseded needs both drop_id and keep_id")
    if drop_id == keep_id:
        raise ValueError(f"drop_id == keep_id ({drop_id!r}); refusing to self-supersede")
    c = core()
    row = c.store.get(drop_id)
    if row is None:
        raise KeyError(f"no memory with id {drop_id!r}")
    with c.store.lock:
        md = _meta(row)
        md["superseded_at"] = superseded_at or date.today().isoformat()
        c.store.update_metadata(drop_id, md)
        c.store.mark_superseded(drop_id, keep_id, reason or None)
    _salience_log(f"superseded {drop_id} -> {keep_id}"
                  + (f" ({reason[:80]})" if reason else ""))
    return f"marked {drop_id} superseded by {keep_id}"
