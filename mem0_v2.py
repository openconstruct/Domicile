"""Drop-in replacement for mem0_core's public API, backed by domicile_mem0.

Same function names, arguments and return strings as mem0_core, so the
cutover is one import in mem0_ipc._dispatch_to_core and mem0_server._direct.
Scoring reproduces the legacy path exactly (2026-11-08, mem0 2.0.19 with
BM25 off and spaCy absent => semantic score is raw cosine):
  semantic = top-20 by cosine (>= 0.1) over the user's facts, active only
  keyword  = FTS5 over active facts (same query builder)
  fused by RRF (k=60), then mem0_core's _salience_adjust on last_used /
  created date / use_count.
So HINT_THRESHOLD carries over without recalibration if the vectors match.

Delete policy (owner decision, 2026-11-08): delete is for extraction/integration
only. The chat-facing `delete` op retires a fact (supersede with no
replacement, recoverable); a real removal is purge(), never dispatched.
"""
from __future__ import annotations

import hashlib
import json
import math
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
from paths import MEM0_DIR  # noqa: E402

from domicile_mem0.core import MemoryCore  # noqa: E402
from domicile_mem0.fts import _fts_match_query  # noqa: E402
from domicile_mem0.vectors import cosine_scores, normalize  # noqa: E402

DB_PATH = Path(os.environ.get("MEM0_V2_DB") or (MEM0_DIR / "flash.db"))
USER_ID = CL.load_identity()["mem0_user_id"]
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIMS = 384

# --- constants copied verbatim from mem0_core (2026-11-08) ---------------
HINT_THRESHOLD = 0.66
HINT_LIMIT = 3
HINT_RECENCY_BONUS = 0.06
HINT_RECENCY_DAYS = 64
RRF_K = 60
HYBRID_RECENCY_BONUS = 0.6 / RRF_K
HYBRID_RECENCY_DAYS = HINT_RECENCY_DAYS
HINT_STALE_FLOOR = -0.10
HINT_STALE_DAYS = 180
HINT_FREQ_CAP = 0.12
HINT_FREQ_K = 0.03
EVENT_WEIGHT = 0.25
_HYBRID_SCALE = HYBRID_RECENCY_BONUS / HINT_RECENCY_BONUS
HYBRID_STALE_FLOOR = HINT_STALE_FLOOR * _HYBRID_SCALE
HYBRID_STALE_DAYS = HINT_STALE_DAYS
HYBRID_FREQ_CAP = HINT_FREQ_CAP * _HYBRID_SCALE
HYBRID_FREQ_K = HINT_FREQ_K * _HYBRID_SCALE

class StoreBusy(RuntimeError):
    """Never raised by SQLite (WAL, many readers); exists so callers that `except core.StoreBusy` still work."""


SEM_TOP_K = 20        # mem0 Memory.search default top_k
SEM_THRESHOLD = 0.1   # mem0 Memory.search default threshold

_store_lock = threading.RLock()
_core: MemoryCore | None = None
_llm_ready = False
_seen_keys: set[str] = set()


# --- store ----------------------------------------------------------------

class _FastEmbed:
    def __init__(self):
        os.environ.setdefault("FASTEMBED_CACHE_PATH",
                              os.path.expanduser("~/domicile/data/fastembed_cache"))
        from fastembed import TextEmbedding
        self._m = TextEmbedding(EMBED_MODEL, cache_dir=os.environ["FASTEMBED_CACHE_PATH"])
        self.dim = EMBED_DIMS
        self.name = EMBED_MODEL

    def embed(self, text):
        return normalize(np.asarray(next(iter(self._m.embed([text]))), dtype=np.float32))


def core(embedder=None) -> MemoryCore:
    global _core
    if _core is None:
        with _store_lock:
            if _core is None:
                DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                _core = MemoryCore(str(DB_PATH), embedder or _FastEmbed())
    return _core


def _ensure_llm() -> None:
    """Extraction rides mem0_core's provider chain (ChainLLM + adapters).
    Lazy: importing mem0_core pulls in the mem0 library, which search/hint
    never need. Last remaining mem0 dependency; port the adapters to drop it."""
    global _llm_ready
    if _llm_ready:
        return
    import mem0_core as legacy
    chain = legacy.ChainLLM(legacy.MEM0_LLM_CHAIN)
    chain._active()  # fail loudly here, not mid-extraction

    def call(system, user):
        return chain.generate_response(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"})
    c = core()
    c.llm = call
    c.custom_instructions = legacy.CUSTOM_INSTRUCTIONS
    _llm_ready = True


# --- row helpers ----------------------------------------------------------

def _meta(row) -> dict:
    try:
        return json.loads(row["metadata"] or "{}")
    except (TypeError, ValueError):
        return {}


def _created(row, md=None) -> str:
    """mem0_core.rows(): metadata 'date' wins when it is earlier."""
    md = _meta(row) if md is None else md
    c = row["created_at"]
    stamp = datetime.fromtimestamp(c, timezone.utc).date().isoformat() if c else ""  # mem0 stored UTC
    d = str(md.get("date") or "")[:10]
    if d and d <= (stamp or "9999"):
        stamp = d
    return stamp


def _recency_bonus(created_at: str, peak: float = HINT_RECENCY_BONUS,
                   horizon: int = HINT_RECENCY_DAYS) -> float:
    try:
        age = (date.today() - date.fromisoformat((created_at or "")[:10])).days
    except ValueError:
        return 0.0
    if age <= 0:
        return peak
    if age >= horizon:
        return 0.0
    step = peak / max(1.0, math.log2(horizon))
    return max(0.0, peak - step * math.floor(math.log2(age)))


def _salience_adjust(last_used, created_at, use_count, *, peak, horizon, floor,
                     stale_days, freq_cap, freq_k, event_weight=EVENT_WEIGHT):
    used_stamp = last_used or created_at
    try:
        age = (date.today() - date.fromisoformat((used_stamp or "")[:10])).days
    except ValueError:
        age = 0
    if age <= horizon:
        used_decay = _recency_bonus(used_stamp, peak=peak, horizon=horizon)
        created_decay = _recency_bonus(created_at, peak=peak, horizon=horizon)
        decay = (1 - event_weight) * used_decay + event_weight * created_decay
    else:
        frac = min(1.0, (age - horizon) / max(1, stale_days - horizon))
        decay = floor * frac
    try:
        uc = max(0, int(use_count or 0))
    except (TypeError, ValueError):
        uc = 0
    freq = min(freq_cap, freq_k * math.log2(uc + 1)) if uc else 0.0
    return decay + freq


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
    rows, index, matrix, cols = c._snapshot()
    if not rows:
        return []
    q = normalize(c.embedder.embed(query))
    sims = cosine_scores(matrix, q)
    mask = cols["user_id"] == USER_ID
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
                    "cos": None, "via": [], "score": 0.0}
            s["score"] += 1.0 / (RRF_K + rank)
            s["via"].append(via)
            if via == "sem":
                s["cos"] = float(sims[i])
    ranked = sorted(fused.values(), key=lambda s: -s["score"])
    for s in ranked:
        s["via"] = "+".join(dict.fromkeys(s["via"]))
    return [s for s in ranked if s["memory"]][:limit]


# --- the tools (mem0_core signatures and return strings) -------------------

def search(query: str, limit: int = 5) -> str:
    query = (query or "").strip()
    if not query:
        return "empty query"
    limit = max(1, limit)
    ranked = _hybrid(query, max(limit * 3, 12))
    for s in ranked:
        s["score"] += _salience_adjust(
            s.get("last_used"), s["created_at"], s.get("use_count"),
            peak=HYBRID_RECENCY_BONUS, horizon=HYBRID_RECENCY_DAYS,
            floor=HYBRID_STALE_FLOOR, stale_days=HYBRID_STALE_DAYS,
            freq_cap=HYBRID_FREQ_CAP, freq_k=HYBRID_FREQ_K)
    ranked.sort(key=lambda s: -s["score"])
    ranked = ranked[:limit]
    if not ranked:
        return "no memories matched"
    _bump_use([s["id"] for s in ranked])
    out = []
    for s in ranked:
        cos = s["cos"]
        prefix = f"[{cos:.3f}]" if isinstance(cos, float) else "[kw]"
        out.append(f"{prefix} {s['created_at']} {s['memory']}".strip())
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
        eff = (cos + _salience_adjust(
            s.get("last_used"), s["created_at"], s.get("use_count"),
            peak=HINT_RECENCY_BONUS, horizon=HINT_RECENCY_DAYS,
            floor=HINT_STALE_FLOOR, stale_days=HINT_STALE_DAYS,
            freq_cap=HINT_FREQ_CAP, freq_k=HINT_FREQ_K)
        ) if isinstance(cos, float) else None
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


def chunk_digest(text: str) -> str:
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()[:16]


# Calibrated 2026-11-10 against bge-small: pairs >=0.926 were restatements,
# <=0.916 were distinct facts on one topic. Re-derive after any embedder swap.
GUARD_CUTOFF = 0.92


def _guard(added: list[dict], cutoff: float) -> None:
    """mem0_guard.guard_new_facts: drop a just-extracted fact whose nearest
    other fact (top-5, this user) is >= cutoff cosine. Extraction-time
    delete -- allowed by the delete policy."""
    c = core()
    rows, index, matrix, cols = c._snapshot()
    if matrix is None:
        return
    for r in added:
        i = index.get(r["id"])
        if i is None:
            continue
        sims = cosine_scores(matrix, matrix[i] / (np.linalg.norm(matrix[i]) or 1.0))
        sims = np.where(cols["user_id"] == USER_ID, sims, -np.inf)
        sims[i] = -np.inf
        if float(np.max(sims)) >= cutoff:
            c.store.delete(r["id"])
            r["event"] = "GUARDED"


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
    if not r:
        return "no new facts (already covered by existing memories)"
    return "\n".join(f"{(x.get('event') or 'ADD').upper()}: {x.get('memory', '')[:120]}"
                     for x in r)


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
