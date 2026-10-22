#!/usr/bin/env python3
"""
Poll the agent's Gmail for unread mail and hand new messages to the agent
through cchat's bgtask mechanism (same pattern as rss/rss_notify.py). The
agent reads the digest and decides what is worth telling the user; this script
never messages anyone directly.

State: data/google/gmail_seen.json holds the message IDs already turned into a
bgtask (last 500), so a message that stays unread isn't announced every run.
"seen by this script" is separate from Gmail's own unread flag.

Exit 0 with a bgtask written = new mail found.
Exit 3 = nothing new (the common case on a tight timer).

Run manually to test:
    python3 gmail_notify.py
Run by systemd (domicile-gmail.timer -> domicile-gmail.service).
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402

SEEN_FILE = paths.GMAIL_SEEN
UNREAD_COUNT_FILE = paths.GMAIL_UNREAD
BGTASK_DIR = paths.BGTASK_DIR
SEEN_CAP = 500

sys.path.insert(0, str(HERE))
from gmail_client import list_unread  # noqa: E402


def load_seen() -> set:
    if not SEEN_FILE.exists():
        return set()
    try:
        return set(json.loads(SEEN_FILE.read_text()))
    except (json.JSONDecodeError, OSError):
        return set()


def save_seen(seen: set) -> None:
    trimmed = list(seen)[-SEEN_CAP:]
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEEN_FILE.write_text(json.dumps(trimmed, indent=2) + "\n")


def _account() -> str:
    try:
        return json.loads(paths.GOOGLE_TOKEN.read_text()).get("account") or "the agent's inbox"
    except (OSError, ValueError):
        return "the agent's inbox"


def format_digest(messages: list[dict]) -> str:
    lines = [f"# New mail — {_account()} ({len(messages)} unread, unseen by this script)\n"]
    for m in messages:
        lines.append(f"- **{m['subject'] or '(no subject)'}**")
        lines.append(f"  From: {m['from']}")
        lines.append(f"  Date: {m['date']}")
        lines.append(f"  {m['snippet']}")
        lines.append(f"  id: {m['id']}\n")
    return "\n".join(lines)


def main() -> int:
    started = time.time()
    started_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    try:
        unread = list_unread(max_results=30)
    except Exception as exc:  # noqa: BLE001 -- surface any auth/API failure, don't swallow it
        print(f"gmail_notify: list_unread failed: {exc}", file=sys.stderr)
        return 1

    # Piggyback a cheap cache file on this already-scheduled network call, so
    # cchat's per-turn HUD (domicile_hud in cchat.py) can show a real unread
    # count by reading this file -- same zero-network-call-per-turn pattern
    # as rss_notify.py/latest.md. Written every run (not just on "new mail")
    # so the count stays live even when nothing new has arrived. Capped at
    # 30 by list_unread's max_results, so a heavily backlogged inbox will
    # show "30+" rather than a true count -- fine for a glance HUD.
    UNREAD_COUNT_FILE.parent.mkdir(parents=True, exist_ok=True)
    UNREAD_COUNT_FILE.write_text(json.dumps({
        "count": len(unread),
        "capped": len(unread) >= 30,
        "checked_iso": started_iso,
    }, indent=2) + "\n")

    seen = load_seen()
    new = [m for m in unread if m["id"] not in seen]

    if not new:
        print("gmail_notify: nothing new, no bgtask written")
        return 3

    digest = format_digest(new)
    seen.update(m["id"] for m in new)
    save_seen(seen)

    finished = time.time()
    BGTASK_DIR.mkdir(parents=True, exist_ok=True)
    task_id = f"gmail-{int(started)}-{os.urandom(3).hex()}"
    meta = {
        "id": task_id,
        "label": f"Gmail — {len(new)} new message{'s' if len(new) != 1 else ''}",
        "argv": [sys.executable, str(HERE / "gmail_notify.py")],
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
    (BGTASK_DIR / f"{task_id}.log").write_text(digest)
    print(f"gmail_notify: wrote bgtask {task_id} ({len(new)} new messages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
