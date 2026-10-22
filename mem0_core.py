#!/usr/bin/env python3
"""mem0 store ownership, extraction, and retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("MEM0_TELEMETRY", "false")

os.environ.setdefault("OPENAI_API_KEY", "unused-mem0-runs-inference-free")

from mem0 import Memory  # noqa: E402

import mem0_claude_llm  # noqa: E402
import mem0_llm_common  # noqa: E402
import mem0_or_llm  # noqa: E402
import mem0_oai_llm  # noqa: E402

mem0_claude_llm.register()
mem0_or_llm.register()

from paths import MEM0_DIR, QDRANT_PATH
import config_loader as CL


def _llm_spec(prov: str, model) -> tuple | None:
    """One (backend, model, endpoint_cfg) for a provider name, or None."""
    blk = CL.provider_block(prov) or {}
    if not blk:
        return None
    if CL.is_cli_provider(prov):
        return ("claude_cli", model, None)
    if blk.get("base_url"):
        return ("oai", model, {
            "base_url": blk["base_url"],
            "key_file": blk.get("key_file"),
            "key_env": blk.get("key_env"),
        })
    return None


def resolve_llm_chain() -> list:
    """Every extraction backend worth trying, best first.

    MEM0_LLM is a *preference*, not a pin: it goes first (that is what the
    systemd drop-in is for -- keep extraction off the subscription while a
    cheap provider is alive), and the compaction chain for the active
    provider picks up when it caps out or its key dies.
    """
    specs, seen = [], set()

    def _push(spec):
        if not spec:
            return
        key = (spec[0], spec[1])
        if key in seen:
            return
        seen.add(key)
        specs.append(spec)

    ovr = os.environ.get("MEM0_LLM")
    if ovr:
        blk = CL.provider_block(ovr) or {}
        _push(_llm_spec(ovr, (blk.get("models", {}).get("compact")
                              or [None])[0]))

    for prov, model in CL.resolve_chain("compact"):
        _push(_llm_spec(prov, model))

    return specs or [("claude_cli", None, None)]


def _resolve_llm() -> tuple:
    """(backend, model, endpoint_cfg) -- the chain's first link."""
    return resolve_llm_chain()[0]


MEM0_LLM_CHAIN = resolve_llm_chain()
MEM0_LLM_BACKEND, MEM0_LLM_MODEL, MEM0_LLM_ENDPOINT = MEM0_LLM_CHAIN[0]

USER_ID = CL.load_identity()["mem0_user_id"]

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIMS = 384

LOCK_TIMEOUT_SECONDS = 15

ALL_ROWS = 100000

HINT_THRESHOLD = 0.71
HINT_LIMIT = 3

HINT_RECENCY_BONUS = 0.06
HINT_RECENCY_DAYS = 64

# hybrid retrieval
MEM0_FTS_PATH = MEM0_DIR / "mem0-fts.db"
RRF_K = 60  # reciprocal-rank-fusion damping; 60 is the standard default

HYBRID_RECENCY_BONUS = 0.6 / RRF_K
HYBRID_RECENCY_DAYS = HINT_RECENCY_DAYS

# salience
HINT_STALE_FLOOR = -0.10     # cosine-domain adj at full staleness
HINT_STALE_DAYS = 180        # linear 0 (@ HINT_RECENCY_DAYS) -> floor (here)
HINT_FREQ_CAP = 0.12         # max buy-back from frequent explicit use
HINT_FREQ_K = 0.03           # geometric: min(cap, k * log2(days_used + 1))

EVENT_WEIGHT = 0.25

_HYBRID_SCALE = HYBRID_RECENCY_BONUS / HINT_RECENCY_BONUS
HYBRID_STALE_FLOOR = HINT_STALE_FLOOR * _HYBRID_SCALE
HYBRID_STALE_DAYS = HINT_STALE_DAYS
HYBRID_FREQ_CAP = HINT_FREQ_CAP * _HYBRID_SCALE
HYBRID_FREQ_K = HINT_FREQ_K * _HYBRID_SCALE

FTS_SYNC_TTL_SECONDS = 30.0

_FTS_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "it", "its", "for", "on", "with", "that", "this", "as", "at", "be", "by",
    "from", "we", "i", "you", "he", "she", "they", "do", "did", "does", "not",
    "what", "when", "where", "why", "how", "which", "there", "here", "about",
}

_fts_db: sqlite3.Connection | None = None
_fts_ok: bool | None = None          # None = untried, True/False = FTS5 usable
_fts_sync_last = 0.0                  # time.monotonic() of last mirror sync

_IDENT = CL.load_identity()

CUSTOM_INSTRUCTIONS = (
    f"These memories form the long-term continuity store for an AI agent "
    f"({_IDENT['agent_name']}) and its operator ({_IDENT['operator_name']}). "
    "They are re-read months later, cold, without the "
    "conversation that produced them. Prefer fuller memories that carry the "
    "reasoning behind a decision, not just the decision itself. When a fact came "
    "from debugging or analysis, keep enough of the 'why' and the hard "
    "specifics -- file paths, function names, error strings, version numbers, "
    "config values, dates -- that someone could act on it without the original "
    "context. Record what changed and what it replaced. Attribute decisions to "
    "whoever made them. Err toward the upper end of the length guidance; "
    "terseness that drops the rationale is a failure here, not concision."
)

CONFIG = {
    "custom_instructions": CUSTOM_INSTRUCTIONS,
    "history_db_path": ":memory:",
    "embedder": {"provider": "fastembed", "config": {"model": EMBED_MODEL}},
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "ccchat",
            "path": str(QDRANT_PATH),
            "on_disk": True,
            "embedding_model_dims": EMBED_DIMS,
        },
    },
}

_memory = None

_store_lock = threading.RLock()

_seen_keys: set[str] = set()


def _seen_key(key: str) -> bool:
    return key in _seen_keys


def _mark_key(key: str) -> None:
    _seen_keys.add(key)


def _reset_keys() -> None:
    _seen_keys.clear()


class StoreBusy(RuntimeError):
    """The qdrant directory lock is held by another process."""


# extraction LLM: a chain, not a pin

DEMOTION_RETRY_SECONDS = 1800.0   # how long a capped link stays demoted


def _build_llm(spec):
    """One mem0 LLM adapter from a (backend, model, endpoint) spec."""
    backend, model, ep = spec
    if backend == "claude_cli":
        return mem0_claude_llm.ClaudeCliLLM(
            {"model": model or mem0_claude_llm.DEFAULT_MODEL})
    if backend == "oai":
        if not ep:
            raise RuntimeError(
                "no OAI endpoint resolved for mem0 extraction -- "
                "every provider is marked down or missing its key")
        return mem0_oai_llm.OpenAILLM({
            "model": model or mem0_oai_llm.DEFAULT_MODEL,
            "base_url": ep["base_url"],
            "key_file": ep.get("key_file"),
            "key_env": ep.get("key_env"),
        })
    raise RuntimeError(
        f"MEM0_LLM={backend!r} unrecognized; use 'claude_cli' or 'oai'")


class ChainLLM:
    """Walks the extraction chain when a link reports a usage limit.

    A capped link is demoted, not discarded: after DEMOTION_RETRY_SECONDS
    the walk restarts from the top, so a rolling-window quota recovers on
    its own instead of pinning us to the fallback for the process's life.
    """

    def __init__(self, specs: list):
        self.specs = list(specs)
        self._i = 0
        self._llm = None
        self._demoted_at = 0.0

    @property
    def spec(self):
        return self.specs[min(self._i, len(self.specs) - 1)]

    def _active(self):
        while self._llm is None and self._i < len(self.specs):
            try:
                self._llm = _build_llm(self.spec)
            except Exception as exc:  # noqa: BLE001  -- missing key, bad cfg
                self._demote(f"unusable: {type(exc).__name__}: {exc}")
        if self._llm is None:
            raise RuntimeError(
                "no usable mem0 extraction provider in "
                f"{[(s[0], s[1]) for s in self.specs]}")
        return self._llm

    def _demote(self, why: str) -> None:
        import sys
        sys.stderr.write(
            f"mem0: extraction link {self.spec[0]}/{self.spec[1]} capped "
            f"({why[:120]}); falling through\n")
        self._i += 1
        self._llm = None
        self._demoted_at = time.monotonic()

    def _maybe_reset(self) -> None:
        if self._i and (time.monotonic() - self._demoted_at
                        > DEMOTION_RETRY_SECONDS):
            self._i, self._llm = 0, None

    def generate_response(self, *args, **kwargs) -> str:
        self._maybe_reset()
        last = None
        while self._i < len(self.specs):
            try:
                return self._active().generate_response(*args, **kwargs)
            except mem0_llm_common.UsageLimitReached as exc:
                last, _ = exc, self._demote(str(exc))
            except Exception as exc:  # noqa: BLE001
                if not mem0_llm_common.is_usage_limit(str(exc)):
                    raise
                last, _ = exc, self._demote(str(exc))
        self._i = len(self.specs) - 1   # stay on the last link; reset later
        raise mem0_llm_common.UsageLimitReached(
            f"every extraction provider capped; last: {str(last)[:200]}")


def store() -> Memory:
    """Built on first use, not at import, and retried while the lock is held."""
    global _memory
    if _memory is not None:
        return _memory
    MEM0_DIR.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    while True:
        try:
            _memory = Memory.from_config(CONFIG)
            _memory.llm = ChainLLM(MEM0_LLM_CHAIN)
            _memory.llm._active()      # fail loudly here, not mid-extraction
            return _memory
        except RuntimeError as exc:
            if "already accessed" not in str(exc):
                raise
            if time.monotonic() >= deadline:
                raise StoreBusy(
                    "another process is holding the memory store; "
                    "it should free up shortly -- try again"
                ) from exc
            time.sleep(0.4)


def rows(result) -> list:
    if isinstance(result, dict):
        out = result.get("results", []) or []
    else:
        out = result or []
    for r in out:
        md = r.get("metadata") or {}
        d = (md.get("date") or "")
        if d and str(d)[:10] <= str(r.get("created_at") or "9999")[:10]:
            r["created_at"] = str(d)[:10]
        if md.get("last_used") and "last_used" not in r:
            r["last_used"] = str(md["last_used"])[:10]
        if md.get("use_count") is not None and "use_count" not in r:
            r["use_count"] = md["use_count"]
        if md.get("superseded") and "superseded" not in r:
            r["superseded"] = True
            r["superseded_by"] = md.get("superseded_by")
            r["superseded_at"] = md.get("superseded_at")
            r["superseded_reason"] = md.get("superseded_reason")
    return out


# FTS5 keyword mirror

def _get_fts() -> sqlite3.Connection | None:
    global _fts_db, _fts_ok
    if _fts_ok is False:
        return None
    if _fts_db is not None:
        return _fts_db
    try:
        MEM0_FTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(MEM0_FTS_PATH), check_same_thread=False)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memories USING fts5("
            "  text, mem_id UNINDEXED, created_at UNINDEXED,"
            "  tokenize='unicode61'"
            ")"
        )
        db.commit()
        _fts_db, _fts_ok = db, True
        return db
    except sqlite3.OperationalError as exc:
        import sys
        sys.stderr.write(f"mem0_core: FTS5 unavailable ({exc}); semantic-only\n")
        _fts_ok = False
        return None


def _fts_match_query(query: str) -> str | None:
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


def _sync_fts(all_rows) -> None:
    """Reconcile the FTS mirror against the current mem0 fact set."""
    db = _get_fts()
    if db is None:
        return
    current: dict[str, tuple[str, str]] = {}
    for r in all_rows:
        if r.get("superseded"):
            continue
        mid = str(r.get("id") or "")
        mem = (r.get("memory") or "").strip()
        if mid and mem:
            current[mid] = (mem, (r.get("created_at") or "")[:10])
    existing = {
        row[0]: (row[1], (row[2] or "")[:10])
        for row in db.execute("SELECT mem_id, text, created_at FROM memories")
    }
    to_del = existing.keys() - current.keys()
    to_up = [m for m, cur in current.items() if existing.get(m) != cur]
    if not to_del and not to_up:
        return
    with db:
        for m in to_del:
            db.execute("DELETE FROM memories WHERE mem_id = ?", (m,))
        for m in to_up:
            mem, created = current[m]
            db.execute("DELETE FROM memories WHERE mem_id = ?", (m,))
            db.execute(
                "INSERT INTO memories (text, mem_id, created_at) VALUES (?, ?, ?)",
                (mem, m, created),
            )


def _all_rows() -> list:
    return rows(store().get_all(filters={"user_id": USER_ID}, top_k=ALL_ROWS))


def _maybe_sync_fts(force: bool = False) -> None:
    global _fts_sync_last
    if _get_fts() is None:
        return
    now = time.monotonic()
    if not force and (now - _fts_sync_last) < FTS_SYNC_TTL_SECONDS:
        return
    try:
        _sync_fts(_all_rows())
        _fts_sync_last = now
    except Exception:
        pass


def _invalidate_fts() -> None:
    """Force the next read to re-sync -- called after a write changed the store."""
    global _fts_sync_last
    _fts_sync_last = 0.0


def _fts_search(query: str, limit: int) -> list[dict]:
    db = _get_fts()
    if db is None:
        return []
    match = _fts_match_query(query)
    if not match:
        return []
    out = []
    for row in db.execute(
        "SELECT mem_id, text, created_at, bm25(memories) AS rank FROM memories"
        " WHERE memories MATCH ? ORDER BY rank LIMIT ?",
        (match, int(limit)),
    ):
        out.append({"id": row[0], "memory": row[1],
                    "created_at": row[2], "score": None})
    return out


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


def _salience_adjust(last_used: str, created_at: str, use_count, *,
                     peak: float, horizon: int, floor: float,
                     stale_days: int, freq_cap: float, freq_k: float,
                     event_weight: float = EVENT_WEIGHT) -> float:
    """Score adjustment for "used stays, unused fades". Two summed terms:"""
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
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with p.open("a") as fh:
            fh.write(f"{stamp}  {line}\n")
    except Exception:
        pass


def _bump_use(ids) -> None:
    ids = [str(i) for i in (ids or []) if i]
    if not ids:
        return
    today = date.today().isoformat()
    try:
        vs = store().vector_store
        bumped = []
        with _store_lock:
            recs = vs.client.retrieve(collection_name=vs.collection_name,
                                      ids=ids, with_payload=True)
            payloads = {str(r.id): (r.payload or {}) for r in recs}
            for mid in ids:
                pl = payloads.get(mid, {})
                if str(pl.get("last_used") or "")[:10] == today:
                    continue
                new_count = int(pl.get("use_count") or 0) + 1
                vs.client.set_payload(
                    collection_name=vs.collection_name,
                    payload={"last_used": today, "use_count": new_count},
                    points=[mid],
                )
                bumped.append(f"{mid}(uc={new_count})")
        if bumped:
            _salience_log(f"consolidated {len(bumped)} fact(s): "
                          + ", ".join(bumped))
    except Exception as exc:  # never break a search over a bookkeeping write
        _salience_log(f"bump skipped: {type(exc).__name__}: {exc}")


def mark_superseded(drop_id: str, keep_id: str, reason: str = "",
                     superseded_at: str | None = None) -> str:
    """Tag `drop_id` as superseded by `keep_id` instead of deleting it."""
    drop_id = (drop_id or "").strip()
    keep_id = (keep_id or "").strip()
    if not drop_id or not keep_id:
        raise ValueError("mark_superseded needs both drop_id and keep_id")
    if drop_id == keep_id:
        raise ValueError(f"drop_id == keep_id ({drop_id!r}); refusing to self-supersede")
    stamp = superseded_at or date.today().isoformat()
    vs = store().vector_store
    with _store_lock:
        recs = vs.client.retrieve(collection_name=vs.collection_name,
                                  ids=[drop_id], with_payload=True)
        if not recs:
            raise KeyError(f"no memory with id {drop_id!r}")
        payload = {
            "superseded": True,
            "superseded_by": keep_id,
            "superseded_at": stamp,
        }
        if reason:
            payload["superseded_reason"] = reason
        vs.client.set_payload(
            collection_name=vs.collection_name,
            payload=payload,
            points=[drop_id],
        )
    _invalidate_fts()
    _salience_log(f"superseded {drop_id} -> {keep_id}"
                  + (f" ({reason[:80]})" if reason else ""))
    return f"marked {drop_id} superseded by {keep_id}"


def _hybrid(query: str, limit: int) -> list[dict]:
    pool = max(limit * 3, 12)
    semantic = [r for r in rows(store().search(query, filters={"user_id": USER_ID}))
                if not r.get("superseded")][:pool]
    keyword = _fts_search(query, pool)

    fused: dict[str, dict] = {}
    for retriever, hits in (("sem", semantic), ("kw", keyword)):
        for rank, hit in enumerate(hits):
            mid = str(hit.get("id") or "")
            if not mid:
                continue
            slot = fused.get(mid)
            if slot is None:
                slot = fused[mid] = {"id": mid, "memory": "", "created_at": "",
                                     "last_used": "", "use_count": 0,
                                     "cos": None, "via": [], "score": 0.0}
            slot["score"] += 1.0 / (RRF_K + rank)
            slot["via"].append(retriever)
            mem = (hit.get("memory") or "").strip()
            if mem:
                slot["memory"] = mem
            if hit.get("created_at"):
                slot["created_at"] = (hit.get("created_at") or "")[:10]
            if hit.get("last_used"):
                slot["last_used"] = str(hit.get("last_used"))[:10]
            if hit.get("use_count"):
                slot["use_count"] = hit.get("use_count")
            sc = hit.get("score")
            if retriever == "sem" and isinstance(sc, (int, float)):
                slot["cos"] = float(sc)

    ranked = sorted(fused.values(), key=lambda s: -s["score"])
    for s in ranked:
        s["via"] = "+".join(dict.fromkeys(s["via"]))
    return [s for s in ranked if s["memory"]][:limit]


# the tools

def search(query: str, limit: int = 5) -> str:
    query = (query or "").strip()
    if not query:
        return "empty query"
    _maybe_sync_fts(force=True)
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
        stamp = s["created_at"]
        prefix = f"[{cos:.3f}]" if isinstance(cos, float) else "[kw]"
        out.append(f"{prefix} {stamp} {s['memory']}".strip())
    return "\n".join(out)


def chunk_digest(text: str) -> str:
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()[:16]


def add(text: str, extract: bool = True, metadata: dict | None = None,
        idempotency_key: str | None = None, guard: bool = False) -> str:
    """Add text to the store."""
    text = (text or "").strip()
    if not text:
        return "nothing to store"

    key = idempotency_key or (chunk_digest(text) if not extract else None)
    if key is not None:
        with _store_lock:
            if _seen_key(key):
                return f"skipped (idempotency key {key} already ingested)"

    with _store_lock:
        result = store().add(
            text, user_id=USER_ID, infer=bool(extract),
            metadata=dict(metadata) if metadata else None,
        )
        _invalidate_fts()  # store changed -- next read re-syncs the mirror
        if guard:
            try:
                import mem0_guard
                added_rows = rows(result)
                mem0_guard.guard_new_facts(
                    store(), added_rows, USER_ID,
                    cutoff=getattr(mem0_guard, "DEFAULT_CUTOFF", 0.70))
            except Exception as exc:  # guard is a safety net, never fatal
                import sys
                print(f"[mem0_core] guard pass skipped: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
        if key is not None:
            _mark_key(key)
    if not extract:
        return f"stored verbatim: {text[:100]}"
    r = rows(result)
    if not r:
        return "no new facts (already covered by existing memories)"
    out = []
    for row in r:
        event = (row.get("event") or "ADD").upper()
        out.append(f"{event}: {row.get('memory', '')[:120]}")
    return "\n".join(out)


def add_verbatim(text: str, metadata: dict | None = None) -> str:
    """Store text with no extractor pass and return the new memory's id."""
    text = (text or "").strip()
    if not text:
        raise ValueError("add_verbatim needs non-empty text")
    with _store_lock:
        result = store().add(
            text, user_id=USER_ID, infer=False,
            metadata=dict(metadata) if metadata else None,
        )
        _invalidate_fts()
    r = rows(result)
    if not r or not r[0].get("id"):
        raise RuntimeError(f"add() returned no id for verbatim text: {result}")
    return r[0]["id"]


def list_memories(limit: int = 20) -> str:
    r = rows(store().get_all(filters={"user_id": USER_ID}, top_k=ALL_ROWS))
    total = len(r)
    r.sort(key=lambda x: (x.get("created_at") or "")[:19], reverse=True)
    r = r[: max(1, limit)]
    if not r:
        return "memory store is empty"
    out = [f"{total} memories total, showing {len(r)}:"]
    for row in r:
        stamp = (row.get("created_at") or "")[:10]
        tag = ""
        if row.get("superseded"):
            tag = f" [superseded -> {str(row.get('superseded_by') or '?')[:8]}]"
        out.append(f"- {stamp} ({row.get('id', '?')[:8]}) {row.get('memory', '')}{tag}")
    return "\n".join(out)


def list_facts(include_superseded: bool = False) -> list[dict]:
    r = rows(store().get_all(filters={"user_id": USER_ID}, top_k=ALL_ROWS))
    out = []
    for row in r:
        if not include_superseded and row.get("superseded"):
            continue
        md = row.get("metadata") or {}
        out.append({
            "id": str(row.get("id", "")),
            "date": str(row.get("created_at") or "?")[:10],
            "kind": md.get("kind", "?"),
            "attr": md.get("attributed_to") or "?",
            "text": (row.get("memory") or "")[:500],
        })
    return out


def delete(memory_id: str) -> str:
    memory_id = (memory_id or "").strip()
    if not memory_id:
        return "no id given"
    r = rows(store().get_all(filters={"user_id": USER_ID}, top_k=ALL_ROWS))
    matches = [row for row in r if str(row.get("id", "")).startswith(memory_id)]
    if not matches:
        return f"no memory with id starting {memory_id!r}"
    if len(matches) > 1:
        return f"{memory_id!r} matches {len(matches)} memories; use a longer id"
    store().delete(memory_id=matches[0]["id"])
    _invalidate_fts()
    return f"deleted: {matches[0].get('memory', '')[:80]}"


def hint(text: str, threshold: float = HINT_THRESHOLD,
         limit: int = HINT_LIMIT) -> list[tuple[float, str, str]]:
    text = (text or "").strip()
    if not text:
        return []
    _maybe_sync_fts()  # throttled: at most once per FTS_SYNC_TTL_SECONDS
    ranked = _hybrid(text, max(limit * 2, 6))
    out: list[tuple[float, str, str]] = []
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
