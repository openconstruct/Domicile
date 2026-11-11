#!/usr/bin/env python3
"""Domicile heartbeat and service recovery."""

from __future__ import annotations

import datetime as dt
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import HEARTBEAT_FILE, WATCHDOG_LOG  # noqa: E402

UNIT = "cchat.service"
TMUX_SOCKET = "cchat"
TMUX_SESSION = "main"
STALE_SECS = 600          # 10 min; heartbeat writes every 60s
TMUX = shutil.which("tmux") or "/usr/bin/tmux"
SYSTEMCTL = shutil.which("systemctl") or "/usr/bin/systemctl"


def log(msg: str) -> None:
    WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    with WATCHDOG_LOG.open("a") as f:
        f.write(f"{stamp}  {msg}\n")


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=30)


def tmux_session_alive() -> bool:
    try:
        cp = _run(TMUX, "-L", TMUX_SOCKET, "has-session", "-t", TMUX_SESSION)
        return cp.returncode == 0
    except Exception as exc:
        log(f"tmux check failed ({type(exc).__name__}: {exc}); assuming alive")
        return True


def heartbeat_age() -> float | None:
    try:
        return time.time() - HEARTBEAT_FILE.stat().st_mtime
    except FileNotFoundError:
        return None
    except Exception as exc:
        log(f"heartbeat stat failed ({type(exc).__name__}: {exc})")
        return None


def unit_exit_status() -> str:
    try:
        cp = _run(SYSTEMCTL, "--user", "show", UNIT,
                  "-p", "ExecMainStatus", "--value")
        return cp.stdout.strip()
    except Exception:
        return ""


def restart(reason: str) -> None:
    if unit_exit_status() == "3":
        log(f"{reason}; but unit last exited 3 (sync guard) -- not restarting")
        return
    log(f"{reason}; restarting {UNIT}")
    _run(SYSTEMCTL, "--user", "reset-failed", UNIT)
    cp = _run(SYSTEMCTL, "--user", "restart", UNIT)
    if cp.returncode == 0:
        log(f"restart ok")
    else:
        log(f"restart FAILED rc={cp.returncode}: {cp.stderr.strip()}")


def main() -> int:
    if not tmux_session_alive():
        restart("tmux session gone")
        return 0

    age = heartbeat_age()
    if age is None:
        log("heartbeat file missing; tmux session present, holding off")
        return 0

    if age > STALE_SECS:
        restart(f"heartbeat stale ({age:.0f}s > {STALE_SECS}s)")
        return 0

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"watchdog crashed: {type(exc).__name__}: {exc}")
        sys.exit(1)
