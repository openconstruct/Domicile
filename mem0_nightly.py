#!/usr/bin/env python3
"""Nightly transcript and legacy-note ingestion."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mem0_server as M  # noqa: E402
import mem0_claude_llm as L  # noqa: E402
import mem0_or_llm as OR  # noqa: E402

from paths import MEM0_LOG_DIR as LOG_DIR, INGESTED_DIR as LOG_INGESTED_DIR, \
    MEM0_STATE_PATH as STATE_PATH, INGEST_STATUS_PATH as STATUS_PATH
REMEMBER_DIR = Path.home() / ".remember"
LOG_PATH = LOG_DIR / "nightly.log"

import config_loader as _CL  # noqa: E402
_OPERATOR_NAME = _CL.load_identity()["operator_name"]


CHUNK_CHARS = 6000


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    line = f"{stamp}  {msg}"
    print(line)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def _is_usage_limit(text: str) -> bool:
    """Either backend's flavor of 'stop calling me, quota's gone'."""
    return L._is_usage_limit(text) or OR._is_usage_limit(text)


BUSY_BACKOFF_SECONDS = (2, 8, 30)
BUSY_ABORT_AFTER_CHUNKS = 3

CONN_BACKOFF_SECONDS = (5, 20, 60)
CONN_RETRIES = len(CONN_BACKOFF_SECONDS)


def _is_connection_error(text: str) -> bool:
    return "Connection error" in text or "ConnectionError" in text \
        or "ConnectError" in text




def _is_busy(text: str) -> bool:
    """A transient store failure, as opposed to a bad chunk."""
    return text.startswith("memory unavailable")


def write_status(ok: bool, reason: str, *, done: int = 0, total: int = 0,
                 added: int = 0, pending: int = 0) -> None:
    """Record how this run went, for mem1 to surface in context."""
    prev = {}
    try:
        prev = json.loads(STATUS_PATH.read_text())
    except Exception:
        pass
    stalls = int(prev.get("consecutive_stalls") or 0)
    payload = {
        "last_run": dt.datetime.now().isoformat(timespec="seconds"),
        "ok": bool(ok),
        "reason": reason,
        "chunks_done": done,
        "chunks_total": total,
        "added": added,
        "sources_pending": pending,
        "consecutive_stalls": 0 if ok else stalls + 1,
    }
    try:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(json.dumps(payload, indent=2))
    except Exception as exc:  # noqa: BLE001
        log(f"could not write status: {type(exc).__name__}: {exc}")


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {"chats": {}, "files": {}}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def chunk(blocks: list[str]) -> list[str]:
    """Group text blocks into CHUNK_CHARS-sized extraction units."""
    out, cur, size = [], [], 0
    for b in blocks:
        if not b.strip():
            continue
        if cur and size + len(b) > CHUNK_CHARS:
            out.append("\n\n".join(cur))
            cur, size = [], 0
        cur.append(b)
        size += len(b)
    if cur:
        out.append("\n\n".join(cur))
    return out


def collect_daily_logs() -> list[tuple[str, list[str], tuple]]:
    """(label, chunks, commit) for each completed day's compaction log."""
    jobs = []
    today = dt.date.today().isoformat()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(LOG_DIR.glob("*.log")):
        if path.stem >= today:
            continue  # still being written today -- leave it
        body = path.read_text().strip()
        if not body:
            continue
        header = (
            f"Compaction log from {path.stem} -- conversation dropped from "
            f"live context that day (possibly across several chats), "
            f"captured verbatim at the moment it was summarized away. "
            f"Extract durable facts about {_OPERATOR_NAME}, his projects, his "
            f"preferences, and decisions made and why.\n\n"
        )
        blocks = body.split("\n\n")
        chunks = [header + c for c in chunk(blocks)]
        if not chunks:
            continue
        jobs.append((f"log:{path.name}", chunks, ("logs", str(path))))
    return jobs


def collect_remember(state: dict) -> list[tuple[str, list[str], tuple]]:
    """(label, chunks, commit) for daily files changed since the last run."""
    jobs = []
    for path in sorted(REMEMBER_DIR.glob("today-*.md")):
        try:
            body = path.read_text().strip()
        except Exception:
            continue
        if not body:
            continue
        digest = hashlib.sha256(body.encode()).hexdigest()[:16]
        if state["files"].get(path.name) == digest:
            continue
        header = (
            f"Daily activity log from {path.stem}. Each entry is a compressed "
            f"summary of a work session. Extract what was actually built, "
            f"decided, or learned -- not the fact that a log exists.\n\n"
        )
        jobs.append((f"remember:{path.name}", [header + body],
                     ("files", path.name, digest)))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be sent; make no LLM calls")
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="stop after N extraction calls (0 = no limit)")
    args = ap.parse_args()

    _terminate = {"flag": False}

    def _on_term(signum, frame):
        _terminate["flag"] = True
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)


    state = load_state()
    jobs = collect_daily_logs() + collect_remember(state)

    skipped = []
    try:
        import mem0_hourly as _HOURLY  # noqa: E402
    except Exception as exc:  # noqa: BLE001
        log(f"hourly marker check unavailable ({type(exc).__name__}); "
            f"re-reading as usual")
    else:
        kept = []
        for job in jobs:
            if job[0].startswith("log:"):
                path = Path(job[2][1])
                do_skip, why = _HOURLY._nightly_skip(path)
                if do_skip:
                    if args.dry_run:
                        log(f"[DRY RUN] hourly done-marker for "
                            f"{path.name}: would move without re-extracting")
                    else:
                        shutil.move(str(path),
                                    str(LOG_INGESTED_DIR / path.name))
                        log(f"hourly done-marker for {path.name}: {why}; "
                            f"moved without re-extracting")
                    skipped.append(path.name)
                    continue
            kept.append(job)
        jobs = kept

    total = sum(len(c) for _, c, _ in jobs)
    if not total:
        log("nothing new to ingest")
        if not args.dry_run:
            write_status(True, "nothing new to ingest")
        return 0
    log(f"{len(jobs)} source(s), {total} extraction chunk(s)"
        + (" [DRY RUN]" if args.dry_run else ""))

    done = 0
    added = updated = deleted = nothing = failed = 0
    busy_streak = 0
    for label, chunks, commit in jobs:
        log(f"-- {label}: {len(chunks)} chunk(s)")
        source_ok = True
        for i, c in enumerate(chunks, 1):
            if _terminate["flag"]:
                log("   ! SIGTERM/SIGINT caught — flushing and stopping")
                save_state(state)
                write_status(added > 0,
                             f"interrupted at chunk {done}/{total}; "
                             f"cursors held, resumes next run",
                             done=done, total=total, added=added,
                             pending=len(jobs))
                return 0
            if args.max_chunks and done >= args.max_chunks:
                log(f"stopping at --max-chunks {args.max_chunks}")
                if not args.dry_run:
                    save_state(state)
                    write_status(True, f"stopped at --max-chunks "
                                 f"{args.max_chunks} (as asked)",
                                 done=done, total=total, added=added,
                                 pending=len(jobs))
                return 0
            done += 1
            if args.dry_run:
                head = " ".join(c.split())[:160]
                log(f"   [{done}/{total}] would send {len(c)}c: {head}…")
                continue
            for attempt in range(1, CONN_RETRIES + 1):
                result = M.memory_add(c)
                if not _is_connection_error(result):
                    break
                if attempt < CONN_RETRIES:
                    log(f"   [{done}/{total}] extract LLM connection error "
                        f"(attempt {attempt}/{CONN_RETRIES}); retrying in "
                        f"{CONN_BACKOFF_SECONDS[attempt-1]}s")
                    time.sleep(CONN_BACKOFF_SECONDS[attempt-1])
            if _is_connection_error(result):
                log(f"   [{done}/{total}] extraction LLM unreachable after "
                    f"{CONN_RETRIES} retries — aborting run")
                save_state(state)
                write_status(False,
                             f"extract LLM down at chunk {done}/{total}; "
                             f"resumes next run",
                             done=done, total=total, added=added,
                             pending=len(jobs))
                return 0
            for delay in BUSY_BACKOFF_SECONDS:
                if not _is_busy(result):
                    break
                log(f"   [{done}/{total}] store busy; retrying in {delay}s")
                time.sleep(delay)
                result = M.memory_add(c)
            errored = result.startswith(("memory error", "memory unavailable"))
            if errored and _is_usage_limit(result):
                log(f"   [{done}/{total}] USAGE LIMIT reached — stopping early")
                log(f"        {' '.join(result.split())[:200]}")
                log(f"partial run: {added} added before the limit; "
                    f"cursors held, will resume on the next run")
                save_state(state)
                write_status(added > 0,
                             f"usage limit at chunk {done}/{total}; "
                             f"cursors held, resumes next run",
                             done=done, total=total, added=added,
                             pending=len(jobs))
                return 0
            if errored:
                source_ok = False
                failed += 1
            if _is_busy(result):
                busy_streak += 1
                if busy_streak >= BUSY_ABORT_AFTER_CHUNKS:
                    log(f"   [{done}/{total}] store held through "
                        f"{busy_streak} chunks of retries — aborting run")
                    log(f"partial run: {added} added; cursors held, "
                        f"will resume on the next run")
                    save_state(state)
                    write_status(added > 0,
                                 f"store busy at chunk {done}/{total}; "
                                 f"cursors held, resumes next run",
                                 done=done, total=total, added=added,
                                 pending=len(jobs))
                    return 0
            else:
                busy_streak = 0
            for line in result.splitlines():
                if line.startswith("ADD"):
                    added += 1
                elif line.startswith("UPDATE"):
                    updated += 1
                elif line.startswith("DELETE"):
                    deleted += 1
                elif line.startswith("no new facts"):
                    nothing += 1
            log(f"   [{done}/{total}] {result.splitlines()[0][:150] if result else '(empty)'}")
            for line in result.splitlines()[1:]:
                log(f"        {line[:150]}")
        if not source_ok:
            log(f"   ! {label}: a chunk failed; left in place for a retry")
            continue
        if args.dry_run:
            continue
        bucket = commit[0]
        if bucket == "logs":
            _, path_str = commit
            src = Path(path_str)
            LOG_INGESTED_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(LOG_INGESTED_DIR / src.name))
        else:
            _, key, value = commit
            state[bucket][key] = value

    if not args.dry_run:
        save_state(state)
        log(f"done: {added} added, {updated} updated, {deleted} deleted, "
            f"{nothing} chunk(s) contributed nothing new, {failed} failed")
        broke = failed and not (added or updated or deleted)
        write_status(
            not broke,
            f"{added} added, {updated} updated, {deleted} deleted, "
            f"{failed} chunk(s) failed",
            done=done, total=total, added=added,
            pending=1 if failed else 0,
        )
    else:
        log("dry run complete; state not advanced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
