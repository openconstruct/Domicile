"""Incremental transcript ingestion through memory IPC."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import time
from pathlib import Path

import mem0_ipc
import config_loader as _CL

from paths import MEM0_LOG_DIR, MEM0_DIR

_OPERATOR_NAME = _CL.load_identity()["operator_name"]

CHUNK_CHARS = 6000          # same extraction-unit size as the nightly
MAX_CHUNKS_PER_RUN = 30     # drain budget per tick; the rest waits an hour
FAIL_TICKS_BEFORE_SKIP = 3  # poison-chunk budget before advancing past it

BUSY_BACKOFF_SECONDS = (2, 8, 30)
CONN_BACKOFF_SECONDS = (5, 20, 60)
CONN_RETRIES = len(CONN_BACKOFF_SECONDS)

STATE_PATH = MEM0_DIR / "hourly-state.json"
STATUS_PATH = MEM0_DIR / "hourly-ingest-status.json"
DONE_DIR = MEM0_DIR / "hourly-done"          # retired (front-runner era)
INGESTED_DIR = MEM0_LOG_DIR / "ingested"     # nightly's close-out bucket
REMEMBER_DIR = Path.home() / ".remember"     # nightly's 2nd source, adopted
LOG_PATH = MEM0_LOG_DIR / "hourly.log"

_CONN_MARKERS = ("Connection error", "ConnectionError", "ConnectError")
_LIMIT_MARKERS = ("usage limit", "rate limit", "quota", "429", "insufficient")


# plumbing

def _log(line: str) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%m-%d %H:%M:%S")
        with LOG_PATH.open("a") as f:
            f.write(f"{stamp} {line}\n")
    except Exception:
        pass


def _is_conn_error(text: str) -> bool:
    return any(m in text for m in _CONN_MARKERS)


def _is_limit(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _LIMIT_MARKERS)


def _is_busy(text: str) -> bool:
    """A transient store failure, as opposed to a bad chunk."""
    low = (text or "").lower()
    return ("memory unavailable" in low or "storebusy" in low
            or "portalocker" in low or "flock" in low)


def load_state(state_path: Path | None = None) -> dict:
    p = state_path or STATE_PATH
    try:
        st = json.loads(p.read_text())
        if isinstance(st, dict):
            files = st.get("files")
            if not isinstance(files, dict):
                files = {}
                if st.get("file"):
                    files[str(st["file"])] = {"offset": int(st.get("offset", 0))}
            st["files"] = files
            return st
    except Exception:
        pass
    return {"files": {}}


def save_state(state: dict, state_path: Path | None = None) -> None:
    p = state_path or STATE_PATH
    try:
        p.write_text(json.dumps(state, indent=2))
    except Exception as exc:  # noqa: BLE001
        _log(f"could not save state: {type(exc).__name__}: {exc}")


def write_status(ok: bool, reason: str, *, done: int = 0, total: int = 0,
                 added: int = 0) -> None:
    now = dt.datetime.now().isoformat(timespec="seconds")
    prior = {}
    try:
        prior = json.loads(STATUS_PATH.read_text())
    except Exception:
        pass
    # last_run is every attempt; last_success only counts a run that got
    # through its chunks. The drain backstop measures the latter -- a held
    # run must not buy itself another full deferral window.
    payload = {
        "last_run": now,
        "last_success": now if ok else prior.get("last_success"),
        "ok": bool(ok),
        "reason": reason,
        "chunks_done": done,
        "chunks_total": total,
        "added": added,
    }
    try:
        STATUS_PATH.write_text(json.dumps(payload, indent=2))
    except Exception as exc:  # noqa: BLE001
        _log(f"could not write status: {type(exc).__name__}: {exc}")


def _file_state(state: dict, name: str) -> dict:
    files = state.setdefault("files", {})
    st = files.setdefault(name, {})
    st.setdefault("offset", 0)
    st.setdefault("fail_streak", 0)
    return st


# one chunk

def _add_with_retry(chunk_text: str, metadata: dict) -> tuple[str, str]:
    """One chunk through IPC, with the nightly's backoff shapes."""
    def _call():
        return mem0_ipc.call("add", {"text": chunk_text,
                                     "extract": True,
                                     "metadata": metadata},
                             timeout=300.0)

    text = ""
    for attempt in range(1, CONN_RETRIES + 1):
        try:
            text = _call()
        except mem0_ipc.Mem0IpcDown as exc:
            return "hold", f"ipc down: {exc}"
        except Exception as exc:  # noqa: BLE001
            text = f"{type(exc).__name__}: {exc}"
        if _is_conn_error(text):
            if attempt < CONN_RETRIES:
                _log(f"   extract LLM connection error (attempt {attempt}/"
                     f"{CONN_RETRIES}); retrying in "
                     f"{CONN_BACKOFF_SECONDS[attempt-1]}s")
                time.sleep(CONN_BACKOFF_SECONDS[attempt - 1])
                continue
            return "hold", text
        break  # definitive answer of one kind or another

    for delay in BUSY_BACKOFF_SECONDS:
        if not _is_busy(text):
            break
        _log(f"   store busy; retrying in {delay}s")
        time.sleep(delay)
        try:
            text = _call()
        except mem0_ipc.Mem0IpcDown as exc:
            return "hold", f"ipc down: {exc}"
        except Exception as exc:  # noqa: BLE001
            text = f"{type(exc).__name__}: {exc}"
    if _is_busy(text):
        return "hold", text

    if text.startswith(("RuntimeError:", "ValueError:", "TypeError:",
                        "KeyError:", "AttributeError:", "JSONDecodeError")):
        return "fail", text
    return "ok", text


# per file

_HEADER_TMPL = (
    "Compaction log from {stem} -- conversation dropped from "
    "live context that day (possibly across several chats), "
    "captured verbatim at the moment it was summarized away. "
    "Extract durable facts about {_OP}, his projects, his "
    "preferences, and decisions made and why.\n\n"
)


def _plan(path: Path, body: str, offset: int, include_tail: bool
          ) -> list[tuple[str, int]]:
    """(chunk_text, end_offset) for everything new in body."""
    blocks = body.split("\n\n")[:-1]  # body ends with the boundary
    pairs = []
    pos = 0
    for b in blocks:
        end = pos + len(b) + 2
        pairs.append((b, pos, end))
        pos = end
    if not include_tail and pairs:
        pairs = pairs[:-1]
    todo = [(b, end) for (b, start, end) in pairs
            if start >= offset and b.strip()]
    return todo


def _chunk_up(header: str, todo: list[tuple[str, int]]
              ) -> list[tuple[str, int]]:
    """Group todo blocks into CHUNK_CHARS units, nightly-identical."""
    chunks: list[tuple[str, int]] = []
    cur: list[str] = []
    cur_end, size = 0, 0
    for b, end in todo:
        if cur and size + len(b) > CHUNK_CHARS:
            chunks.append((header + "\n\n".join(cur), cur_end))
            cur, size = [], 0
        cur.append(b)
        size += len(b)
        cur_end = end
    if cur:
        chunks.append((header + "\n\n".join(cur), cur_end))
    return chunks


def _drain_file(path: Path, header: str, metadata: dict, fst: dict,
                summary: dict, *, include_tail: bool, dry_run: bool,
                budget: int) -> tuple[int, bool]:
    """Drain one file within budget. Returns (remaining budget, fully_caught_up)."""
    try:
        text = path.read_text()
    except Exception as exc:  # noqa: BLE001
        _log(f"   ! {path.name}: unreadable: {type(exc).__name__}")
        return budget, False

    cut = text.rfind("\n\n")
    if cut == -1:
        return budget, False
    body = text[:cut + 2]

    todo = _plan(path, body, int(fst["offset"]), include_tail)
    if not todo:
        return budget, True

    chunks = _chunk_up(header, todo)
    total = len(chunks)
    _log(f"-- {path.name}: {total} new chunk(s) "
         f"({'DRY RUN' if dry_run else 'live'})")
    summary["chunks_total"] += total

    busy_streak = 0
    file_done = 0
    completed = True  # flips False on any break; budget truncation catches
    if len(chunks) > budget:
        completed = False
    for i, (c, end_off) in enumerate(chunks[:budget], 1):
        if dry_run:
            head = " ".join(c.split())[:160]
            _log(f"   [{i}/{total}] would send {len(c)}c: {head}…")
            summary["chunks_done"] += 1
            continue

        status, result = _add_with_retry(c, metadata)
        if status == "hold":
            if _is_busy(result):
                busy_streak += 1
            else:
                busy_streak = 0
            summary["held"] = True
            summary["reason"] = f"transient failure, {path.name} {i}/{total}"
            _log(f"   [{i}/{total}] held (retry next tick): {result[:150]}")
            completed = False
            break
        busy_streak = 0
        if _is_limit(result):
            summary["held"] = True
            summary["reason"] = f"usage limit, {path.name} {i}/{total}"
            _log(f"   [{i}/{total}] usage limit; holding offset")
            completed = False
            break
        if status == "fail":
            summary["failed"] += 1
            fst["fail_streak"] = int(fst.get("fail_streak", 0)) + 1
            if fst["fail_streak"] >= FAIL_TICKS_BEFORE_SKIP:
                summary["held"] = True
                summary["reason"] = (f"poison chunk, {path.name} {i}/{total} "
                                     f"({fst['fail_streak']} ticks) -- manual "
                                     f"nightly catchup can heal it")
            else:
                summary["held"] = True
                summary["reason"] = (f"bad chunk, {path.name} {i}/{total} "
                                     f"(attempt {fst['fail_streak']})")
            _log(f"   [{i}/{total}] {result[:150]}")
            completed = False
            break
        if busy_streak >= 3:
            summary["held"] = True
            summary["reason"] = f"store busy through {busy_streak} chunks"
            completed = False
            break

        first = result.splitlines()[0][:150] if result else "(empty)"
        _log(f"   [{i}/{total}] {first}")
        for line in result.splitlines()[1:]:
            _log(f"        {line[:150]}")
        for line in result.splitlines():
            if line.startswith("ADD"):
                summary["added"] += 1
            elif line.startswith("UPDATE"):
                summary["updated"] += 1
            elif line.startswith("DELETE"):
                summary["deleted"] += 1
            elif line.startswith("no new facts"):
                summary["nothing"] += 1
        summary["chunks_done"] += 1
        file_done += 1
        fst["offset"] = end_off

    return budget - file_done, completed


# entry

def run_once(*, dry_run: bool = False, log_dir: Path | None = None,
             state_path: Path | None = None, now: dt.datetime | None = None
             ) -> dict:
    """Drain today's log incrementally, plus any closed previous-day file."""
    state = load_state(state_path)
    now = now or dt.datetime.now()
    today = now.date().isoformat()
    log_dir = log_dir or MEM0_LOG_DIR

    summary = {"chunks_done": 0, "chunks_total": 0, "added": 0, "updated": 0,
               "deleted": 0, "nothing": 0, "failed": 0, "held": False,
               "reason": "", "day_closed": False}

    if state.get("enabled") is False:
        summary["reason"] = "disabled in hourly-state.json"
        return summary

    if not log_dir.exists():
        summary["reason"] = "no log dir"
        return summary

    budget = MAX_CHUNKS_PER_RUN
    for path in sorted(log_dir.glob("*.log")):
        if budget <= 0:
            summary["reason"] = "budget exhausted before rollover drain"
            summary["held"] = True
            break
        try:
            dt.date.fromisoformat(path.stem)
        except ValueError:
            continue
        if path.stem >= today:
            continue
        marker = DONE_DIR / f"{path.stem}.json"
        if marker.exists():
            _prune_marker_if_stale(path, marker, log_dir)
            if marker.exists():
                continue
        fst = _file_state(state, path.name)
        header = _HEADER_TMPL.format(stem=path.stem, _OP=_OPERATOR_NAME)
        metadata = {"kind": "compaction-log", "date": path.stem,
                    "source": "hourly"}
        budget, file_caught_up = _drain_file(path, header, metadata, fst,
                                             summary, include_tail=True,
                                             dry_run=dry_run, budget=budget)
        fully = file_caught_up and not summary["held"]
        if not dry_run and fully:
            try:
                INGESTED_DIR.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(INGESTED_DIR / path.name))
                summary["day_closed"] = True
                _log(f"-- {path.stem}: fully drained; moved to ingested/")
            except Exception as exc:  # noqa: BLE001
                _log(f"   ! close-out move failed: {type(exc).__name__}: {exc}"
                     f" (file left in place; cursor is safe)")
        save_state(state, state_path)

    todays = log_dir / f"{today}.log"
    if budget > 0 and todays.exists():
        fst = _file_state(state, todays.name)
        header = _HEADER_TMPL.format(stem=today, _OP=_OPERATOR_NAME)
        metadata = {"kind": "compaction-log", "date": today,
                    "source": "hourly"}
        budget, _ = _drain_file(todays, header, metadata, fst, summary,
                                include_tail=False, dry_run=dry_run,
                                budget=budget)
        save_state(state, state_path)

    if not summary["reason"]:
        summary["reason"] = ("drained" if summary["chunks_done"]
                             else "nothing new")

    if budget > 0 and not dry_run and REMEMBER_DIR.is_dir():
        for rpath in sorted(REMEMBER_DIR.glob("today-*.md")):
            if budget <= 0:
                summary["reason"] = "budget exhausted (remember)"
                break
            try:
                body = rpath.read_text().strip()
            except Exception:
                continue
            if not body:
                continue
            digest = hashlib.sha256(body.encode()).hexdigest()[:16]
            if state["files"].get(rpath.name) == digest:
                continue
            header = (f"Daily activity log from {rpath.stem}. Each entry is "
                      f"a compressed summary of a work session. Extract what "
                      f"was actually built, decided, or learned -- not the "
                      f"fact that a log exists.\n\n")
            meta = {"kind": "remember-log", "date": rpath.stem.replace(
                "today-", ""), "source": "hourly"}
            before = summary["chunks_done"]
            held_before = summary["held"]
            budget, _ = _drain_file(rpath, header, meta, _file_state(
                state, f"remember:{rpath.name}"), summary,
                include_tail=True, dry_run=False, budget=budget)
            if (summary["chunks_done"] > before
                    and not summary["held"] and not held_before):
                state["files"][rpath.name] = digest
        save_state(state, state_path)

    if not dry_run:
        ok = (summary["added"] + summary["updated"] + summary["deleted"]
              > 0) or summary["reason"] in ("drained", "nothing new") \
            or summary["reason"].startswith("budget")
        write_status(ok, summary["reason"], done=summary["chunks_done"],
                     total=summary["chunks_total"], added=summary["added"])
        _log(f"run: {summary['added']} added, {summary['updated']} updated, "
             f"{summary['failed']} failed; {summary['reason']}")
    return summary


def _prune_marker_if_stale(path: Path, marker: Path, log_dir: Path) -> None:
    """Drop the marker once its day file has left logs/ (nightly moved it)."""
    if not path.exists():
        try:
            marker.unlink()
        except Exception:
            pass


def _nightly_skip(path: Path) -> tuple[bool, str]:
    """For mem0_nightly: should this day file be moved WITHOUT extraction?"""
    try:
        marker = DONE_DIR / f"{path.stem}.json"
        if not marker.exists():
            return False, ""
        data = json.loads(marker.read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if data.get("sha256") == digest:
            return True, f"already drained hourly at {data.get('at')}"
        return False, "marker digest mismatch; file grew after rollover"
    except Exception as exc:  # noqa: BLE001
        return False, f"marker check failed: {type(exc).__name__}: {exc}"
