#!/usr/bin/env python3
"""Rebuild mem0 from archived transcripts into qdrant-rebuild.

Replay is chronological and checkpointed. Validate with --validate;
promotion to the live store is manual.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import config_loader as _CL
import paths as _P

LIVE_QDRANT = _P.QDRANT_PATH
REBUILD_QDRANT = _P.MEM0_DIR / "qdrant-rebuild"
REBUILD_FTS = _P.MEM0_DIR / "mem0-fts-rebuild.db"
CHECKPOINT = _P.MEM0_DIR / "rescan-checkpoint.json"
RESCAN_LOG = _P.MEM0_DIR / "rescan.log"

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
CHUNK_CHARS = 6000        # cchat-era compaction logs: match mem0_nightly.CHUNK_CHARS
CMEM_CHUNK_CHARS = 12000

CONN_BACKOFF = [5, 15, 45, 90]     # transient extractor-unreachable
BUSY_BACKOFF = [1, 3, 8]           # qdrant lock contention (shouldn't happen: services down)

RUN_ID = "rescan"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(RESCAN_LOG, "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _apply_rebuild_paths() -> None:
    """Point mem0_core at the parallel store. Idempotent."""
    import mem0_core as C

    C.QDRANT_PATH = REBUILD_QDRANT
    C.CONFIG["vector_store"]["config"]["path"] = str(REBUILD_QDRANT)
    C.MEM0_FTS_PATH = REBUILD_FTS
    REBUILD_QDRANT.mkdir(parents=True, exist_ok=True)


# job building

def _iter_log_files() -> list[Path]:
    """Every transcript log in the archive, chronological."""
    seen: dict[str, Path] = {}
    ingested = _P.INGESTED_DIR
    pending = _P.MEM0_LOG_DIR

    files: list[Path] = []
    if ingested.is_dir():
        files += sorted(ingested.glob("*.log"))
    if pending.is_dir():
        files += [p for p in sorted(pending.glob("*.log")) if p.name != "nightly.log"]

    out = []
    for p in files:
        if p.name in seen:
            continue
        seen[p.name] = p
        out.append(p)

    def sortkey(p: Path):
        m = DATE_RE.search(p.name)
        return (m.group(1) if m else "9999-99-99", p.name)

    return sorted(out, key=sortkey)


def _chunk_blocks(blocks: list[str], chunk_chars: int = CHUNK_CHARS) -> list[str]:
    """Group blocks into ~chunk_chars units. Copy of mem0_nightly.chunk()."""
    out, cur, size = [], [], 0
    for b in blocks:
        if not b.strip():
            continue
        if cur and size + len(b) > chunk_chars:
            out.append("\n\n".join(cur))
            cur, size = [], 0
        cur.append(b)
        size += len(b)
    if cur:
        out.append("\n\n".join(cur))
    return out


def _header(date: str, name: str) -> str:
    operator = _CL.load_identity()["operator_name"]
    return (
        f"Transcript archive from {date} (source file: {name}) -- a past "
        f"conversation captured verbatim, possibly spanning several sessions "
        f"that day. Extract durable facts about {operator}, their family, "
        f"their projects, their preferences, and decisions made and the "
        f"reasoning behind them. Ignore transient chatter and anything "
        f"already obvious.\n\n"
    )


def build_jobs() -> list[tuple[str, str, dict]]:
    """(chunk_text, label, metadata) for every extraction unit in the archive."""
    jobs: list[tuple[str, str, dict]] = []
    for path in _iter_log_files():
        try:
            body = path.read_text(errors="replace").strip()
        except Exception as exc:
            log(f"  !! unreadable {path.name}: {exc}")
            continue
        if not body:
            continue
        m = DATE_RE.search(path.name)
        date = m.group(1) if m else "unknown"
        kind = "cmem" if path.name.startswith("claude-") else "compaction-log"
        hdr = _header(date, path.name)
        cc = CMEM_CHUNK_CHARS if kind == "cmem" else CHUNK_CHARS
        chunks = _chunk_blocks(body.split("\n\n"), cc)
        for i, c in enumerate(chunks, 1):
            meta = {"date": date, "source": path.name, "kind": kind,
                    "extract_run": RUN_ID,
                    "chunk": f"{i}/{len(chunks)}"}
            jobs.append((hdr + c, f"{path.name}[{i}/{len(chunks)}]", meta))
    return jobs


# checkpoint

def _load_ckpt() -> dict:
    try:
        return json.loads(CHECKPOINT.read_text())
    except Exception:
        return {"done_keys": [], "added": 0, "deleted": 0, "nothing": 0,
                "failed": 0, "started": None, "last": None}


def _save_ckpt(ck: dict) -> None:
    CHECKPOINT.write_text(json.dumps(ck, indent=2))


# build

def cmd_build(args) -> int:
    if REBUILD_QDRANT.exists() and args.fresh:
        import shutil
        log(f"--fresh: wiping {REBUILD_QDRANT}")
        shutil.rmtree(REBUILD_QDRANT)
        REBUILD_FTS.unlink(missing_ok=True)
        CHECKPOINT.unlink(missing_ok=True)

    _apply_rebuild_paths()
    import mem0_core as C
    import mem0_guard  # noqa: F401  -- ensure importable before we depend on it

    if C.MEM0_LLM_BACKEND != "claude_cli":
        log(f"WARNING: extraction backend is {C.MEM0_LLM_BACKEND!r}, not "
            f"claude_cli/haiku. Set config or MEM0_LLM. Continuing anyway.")

    C._reset_keys()
    ck = _load_ckpt()
    done = set(ck["done_keys"])
    if ck["started"] is None:
        ck["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
        ck["run_id"] = RUN_ID
        _save_ckpt(ck)
    else:
        globals()["RUN_ID"] = ck.get("run_id") or RUN_ID

    log(f"store -> {C.CONFIG['vector_store']['config']['path']}")
    log(f"extractor -> {C.MEM0_LLM_BACKEND} / {C.MEM0_LLM_MODEL}")

    jobs = build_jobs()
    total = len(jobs)
    log(f"{total} extraction chunk(s) across the archive; "
        f"{len(done)} already done per checkpoint")

    try:
        C.store()
    except C.StoreBusy:
        log("FATAL: rebuild store is locked -- another process is using "
            "qdrant-rebuild/. Stop it and retry.")
        return 2

    t0 = time.time()
    processed = 0
    for n, (text, label, meta) in enumerate(jobs, 1):
        key = C.chunk_digest(text)
        if key in done:
            continue
        if args.max_chunks and processed >= args.max_chunks:
            log(f"stopping at --max-chunks {args.max_chunks}")
            break

        result = _add_with_retry(C, text, meta, key, label, n, total)
        processed += 1
        if result is None:
            ck["failed"] += 1
            _save_ckpt(ck)
            continue

        for line in result.splitlines():
            s = line.strip()
            if s.startswith("ADD"):
                ck["added"] += 1
            elif s.startswith("DELETE"):
                ck["deleted"] += 1
            elif s.startswith(("no new facts", "skipped")):
                ck["nothing"] += 1
        head = result.splitlines()[0][:120] if result else "(empty)"
        rate = (time.time() - t0) / max(1, processed)
        eta = rate * (total - n) / 60
        log(f"  [{n}/{total}] {label}: {head}  (~{eta:.0f}m left)")

        done.add(key)
        ck["done_keys"] = sorted(done)
        ck["last"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _save_ckpt(ck)

    dur = (time.time() - t0) / 60
    log(f"DONE in {dur:.1f}m: added={ck['added']} deleted(guard)={ck['deleted']} "
        f"nothing/skip={ck['nothing']} failed={ck['failed']}")
    try:
        cnt = len(C._all_rows())
        log(f"rebuild store now holds {cnt} facts")
    except Exception as exc:
        log(f"(couldn't count rebuild store: {exc})")
    log("Next: python3 mem0_rescan.py validate   then   ./rescan.sh promote")
    return 0


def _add_with_retry(C, text, meta, key, label, n, total):
    conn_tries = 0
    while True:
        try:
            result = C.add(text, extract=True, metadata=meta,
                           idempotency_key=key, guard=True)
        except C.StoreBusy:
            for d in BUSY_BACKOFF:
                log(f"  [{n}/{total}] {label}: store busy, retry in {d}s")
                time.sleep(d)
                try:
                    result = C.add(text, extract=True, metadata=meta,
                                   idempotency_key=key, guard=True)
                    break
                except C.StoreBusy:
                    continue
            else:
                log(f"  [{n}/{total}] {label}: still busy, skipping chunk")
                return None
        except Exception as exc:
            log(f"  [{n}/{total}] {label}: add raised {type(exc).__name__}: "
                f"{str(exc)[:200]}")
            return None

        low = result.lower()
        transient = ("connection error" in low or "timed out" in low
                     or "timeout" in low or "unreachable" in low
                     or result.startswith(("memory error", "memory unavailable")))
        if transient and conn_tries < len(CONN_BACKOFF):
            d = CONN_BACKOFF[conn_tries]
            conn_tries += 1
            log(f"  [{n}/{total}] {label}: transient ({result[:80]!r}); "
                f"retry {conn_tries}/{len(CONN_BACKOFF)} in {d}s")
            time.sleep(d)
            continue
        return result


# validate

def cmd_validate(args) -> int:
    _apply_rebuild_paths()
    import mem0_core as C
    import random

    rows = C._all_rows()
    if not rows:
        log("rebuild store is empty -- nothing to validate. Build first.")
        return 2
    log(f"rebuild store: {len(rows)} facts")

    by_source: dict[str, list[dict]] = {}
    for r in rows:
        src = (r.get("metadata") or {}).get("source", "(none)")
        by_source.setdefault(src, []).append(r)
    log(f"provenance: {len(by_source)} distinct source files; "
        f"{sum(1 for r in rows if not (r.get('metadata') or {}).get('date'))} "
        f"facts missing a date")

    sample = random.sample(rows, min(args.sample, len(rows)))
    flagged = []
    for r in sample:
        text = r.get("memory") or ""
        meta = r.get("metadata") or {}
        src = meta.get("source")
        if not src:
            flagged.append(("no-source", text, meta))
            continue
        p = _find_source(src)
        if not p:
            flagged.append(("source-missing", text, meta))
            continue
        body = p.read_text(errors="replace").lower()
        toks = [w for w in re.findall(r"[a-z0-9]{4,}", text.lower())
                if w not in _STOP]
        if not toks:
            continue
        hit = sum(1 for w in set(toks) if w in body)
        frac = hit / len(set(toks))
        if frac < args.min_overlap:
            flagged.append((f"low-overlap {frac:.2f}", text, meta))

    log(f"sampled {len(sample)}; {len(flagged)} flagged for review "
        f"(overlap < {args.min_overlap})")
    for tag, text, meta in flagged:
        log(f"  [{tag}] ({meta.get('source')}) {text[:160]}")
    if not flagged:
        log("No fabrication signals in the sample. Spot-check a few by hand "
            "anyway, then ./rescan.sh promote")
    return 0


_STOP = {"that", "this", "with", "from", "have", "which", "were",
         "into", "then", "than", "when", "what", "your", "their", "them",
         "would", "could", "should", "about", "over", "also", "been", "being"}
_STOP.add(_CL.load_identity()["operator_name"].lower())   # the operator is in nearly every fact


def _find_source(name: str) -> Path | None:
    for base in (_P.INGESTED_DIR, _P.MEM0_LOG_DIR):
        p = base / name
        if p.is_file():
            return p
    return None


# cli

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="replay the archive into qdrant-rebuild/")
    b.add_argument("--fresh", action="store_true",
                   help="wipe any existing rebuild store + checkpoint first")
    b.add_argument("--max-chunks", type=int, default=0,
                   help="stop after N new chunks this run (0 = no limit)")
    b.set_defaults(fn=cmd_build)

    v = sub.add_parser("validate", help="spot-check rebuild vs source logs")
    v.add_argument("--sample", type=int, default=120)
    v.add_argument("--min-overlap", type=float, default=0.30)
    v.set_defaults(fn=cmd_validate)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
