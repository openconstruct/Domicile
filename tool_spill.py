"""Full copies of tool outputs that get trimmed, so a cut is recoverable.

Any provider path that shortens a tool result writes the full text here first
(keyed by tool-call id + a hash of the text, so a reused id can never serve the
wrong output), and the trim marker carries a fetch_result.py command that reads
it back. Failure to spill is never fatal: the caller falls back to an
unrecoverable marker, exactly as before.

CCHAT_SPILL_DIR overrides the location (tests).
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

try:
    from paths import CCHAT_DATA
    _DEFAULT = Path(CCHAT_DATA) / "tool_spill"
except Exception:  # pragma: no cover
    _DEFAULT = Path(__file__).resolve().parent / "data" / "tool_spill"

SPILL_DIR = Path(os.environ.get("CCHAT_SPILL_DIR") or _DEFAULT)
MIN_SPILL_CHARS = 1000      # = the smallest cut threshold (context_diet's 500+500 floor pass)
KEEP_DAYS = 30
FETCH_SCRIPT = Path(__file__).resolve().parent / "fetch_result.py"
KEY_IN_MARKER = re.compile(r"fetch_result\.py (\S+) --start")
# Any of the trim markers this codebase writes (context_diet, the provider's
# source cut, local_tools.dispatch). A text carrying one is already cut.
ANY_TRIM_MARKER = re.compile(
    r"characters trimmed (?:from context|at source|from the middle)|chars trimmed\] \.\.\.")

_pruned = False


def _spill_dir() -> Path:
    return Path(os.environ.get("CCHAT_SPILL_DIR") or SPILL_DIR)


def key_for(call_id, text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(call_id or "noid"))[:80]
    digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:10]
    return f"{safe}-{digest}"


def _prune_once() -> None:
    global _pruned
    if _pruned:
        return
    _pruned = True
    try:
        cutoff = time.time() - KEEP_DAYS * 86400
        for p in _spill_dir().glob("*.txt"):
            if p.stat().st_mtime < cutoff:
                p.unlink()
    except Exception:
        pass


def spill(call_id, text: str) -> str | None:
    """Save the full text; return its key, or None if too short / on failure."""
    if not isinstance(text, str) or len(text) <= MIN_SPILL_CHARS:
        return None
    try:
        key = key_for(call_id, text)
        d = _spill_dir()
        path = d / f"{key}.txt"
        if not path.exists():
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f"{key}.tmp"
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
            _prune_once()
        return key
    except Exception:
        return None


def load(key: str) -> str | None:
    if not key or not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
        return None
    try:
        return (_spill_dir() / f"{key}.txt").read_text(encoding="utf-8")
    except Exception:
        return None


def cut_middle(call_id, text: str, head: int, tail: int, *,
               where: str = "from the middle", fallback: str = "") -> str:
    """The one head+tail cut for a fresh tool result: save the full text, then
    keep `head` + `tail` chars around a marker that says how to get the rest.

    Callers decide WHETHER to cut (their own size cap); this decides how, so
    the marker wording and the save-before-cut order cannot drift apart again.
    `fallback` is appended when no copy could be saved.
    """
    key = spill(call_id, text)
    cut = len(text) - head - tail
    if cut <= 0:
        return text
    note = recovery_hint(key, head, cut) or fallback
    return (text[:head] + f"\n\n… [{cut:,} characters trimmed {where}.{note}] …\n\n"
            + text[-tail:])


def recovery_hint(key: str | None, start: int, length: int) -> str:
    if not key:
        return ""
    return (f" Recover the cut part: python3 {FETCH_SCRIPT} "
            f"{key} --start {start} --len {length}"
            f" -- or search the whole output: replace --start/--len with"
            f" --grep 'REGEX' or --find 'words'.")
