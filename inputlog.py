"""Debug capture of the exact input sent to the model, for N armed turns.

Arm:    echo 5 > <install-dir>/data/inputlog/ARM      (or: python3 inputlog.py arm 5)
Each turn that starts while ARM > 0 decrements it and writes one directory:
    data/inputlog/<ts>-<provider>/
        system.txt        exact system prompt (Claude path; opencode: per-round)
        prompt.txt        exact user prompt for the turn (stamp/seed/hints/HUD)
        meta.json         provider, model, tools, resume id
        resumed.jsonl     Claude: the TRIMMED transcript the CLI was handed
        round-N-request.json   opencode: full messages+tools of each API call
        usage.jsonl       one row per API round (input/cache-read/cache-write)
Everything here is best-effort and never raises into the turn. Unarmed cost
is one stat() per turn. Files hold the full conversation -- delete the dir
when done.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "data" / "inputlog"
ARM = ROOT / "ARM"

CURRENT: Path | None = None   # single-session app: one active capture dir
_round = 0


def arm(n: int) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    ARM.write_text(str(int(n)))


def begin(provider: str) -> Path | None:
    """Start a capture if armed; consumes one count. None when unarmed."""
    global CURRENT, _round
    CURRENT, _round = None, 0
    try:
        if not ARM.exists():
            return None
        n = int((ARM.read_text().strip() or "0"))
        if n <= 0:
            return None
        ARM.write_text(str(n - 1))
        d = ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{provider}"
        d.mkdir(parents=True, exist_ok=True)
        CURRENT = d
        return d
    except Exception:  # noqa: BLE001
        return None


def end() -> None:
    global CURRENT
    CURRENT = None


def write(name: str, content) -> None:
    if CURRENT is None:
        return
    try:
        p = CURRENT / name
        if isinstance(content, (dict, list)):
            content = json.dumps(content, indent=1, default=str)
        p.write_text(content if isinstance(content, str) else str(content),
                     encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def write_jsonl(name: str, entries) -> None:
    if CURRENT is None:
        return
    try:
        with (CURRENT / name).open("w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass


def usage_row(row: dict) -> None:
    """Append one per-round usage record."""
    if CURRENT is None:
        return
    try:
        row = {"t": time.strftime("%H:%M:%S"), **row}
        with (CURRENT / "usage.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass


def next_round() -> int:
    global _round
    _round += 1
    return _round


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "arm":
        arm(int(sys.argv[2]))
        print(f"armed: {ARM.read_text()}")
    else:
        print(__doc__)
