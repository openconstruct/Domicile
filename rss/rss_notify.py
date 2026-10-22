#!/usr/bin/env python3
"""
Run the feed checker and hand new items to the agent.

If rss_check.py found anything new (exit 0), write a finished bgtask entry
into BGTASK_DIR (same meta.json / <id>.log shape as a real background job).
cchat's bgtask loop notices it and opens a turn with the digest, so the
agent decides what is worth passing on. Exit 3 (nothing new) writes nothing.

    python3 rss_notify.py       run once
    domicile-rss.timer          runs it hourly (installed by the wizard)
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402

CHECK_SCRIPT = HERE / "rss_check.py"
LATEST_FILE = paths.RSS_LATEST
BGTASK_DIR = paths.BGTASK_DIR


def count_new(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.startswith("- **"))


def main() -> int:
    started = time.time()
    started_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    proc = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        capture_output=True, text=True, timeout=120,
    )

    if proc.returncode == 3:
        print("rss_notify: nothing new, no bgtask written")
        return 0

    if proc.returncode != 0:
        print(f"rss_notify: rss_check.py failed (exit {proc.returncode}): "
              f"{proc.stderr[-500:]}", file=sys.stderr)
        return proc.returncode

    latest = LATEST_FILE.read_text() if LATEST_FILE.exists() else proc.stdout
    n = count_new(latest)

    finished = time.time()
    BGTASK_DIR.mkdir(parents=True, exist_ok=True)
    task_id = f"rss-{int(started)}-{os.urandom(3).hex()}"
    meta = {
        "id": task_id,
        "label": f"RSS check — {n} new item{'s' if n != 1 else ''}",
        "argv": [sys.executable, str(CHECK_SCRIPT)],
        "cwd": str(HERE),
        "pid": os.getpid(),
        "started": started,
        "started_iso": started_iso,
        "status": "done",
        "exit_code": 0,
        "finished": finished,
        "notified": False,
    }
    (BGTASK_DIR / f"{task_id}.json").write_text(json.dumps(meta, indent=2) + "\n")
    (BGTASK_DIR / f"{task_id}.log").write_text(latest)
    print(f"rss_notify: wrote bgtask {task_id} ({n} new items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
