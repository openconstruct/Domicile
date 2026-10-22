#!/usr/bin/env python3
"""Backup synchronization checks before startup."""

import json
import os
import socket
import subprocess
import time
from pathlib import Path

HOME = Path.home()
from paths import SYNC_STATE, ROOT


def _load() -> dict:
    try:
        return json.loads(SYNC_STATE.read_text())
    except (OSError, ValueError):
        return {}


def record(commit: str, note: str = "") -> None:
    """Mark `commit` as the state this box is now current with."""
    if not commit:
        return
    SYNC_STATE.parent.mkdir(parents=True, exist_ok=True)
    SYNC_STATE.write_text(json.dumps({
        "synced_commit": commit,
        "note": note,
        "host": socket.gethostname(),
        "updated": time.time(),
        "updated_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))


def _remote_head(timeout: int = 30):
    """(sha, provider_name) for refs/heads/main on the first reachable remote, or None if none answered."""
    try:
        import backup
    except Exception:
        return None
    tokens = backup.read_tokens()
    if not tokens:
        return None
    for name, url in backup.collect_targets(tokens):
        try:
            r = subprocess.run(
                ["git", "ls-remote", url, "refs/heads/main"],
                capture_output=True, text=True, timeout=timeout)
        except (subprocess.SubprocessError, OSError):
            continue
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.split()[0], name
    return None


def check() -> tuple[bool, str]:
    """(ok, message). ok=False => caller must not launch."""
    if os.environ.get("CCHAT_SKIP_SYNC_GUARD"):
        return True, "sync guard: skipped (CCHAT_SKIP_SYNC_GUARD set)"

    try:
        head = _remote_head()
    except Exception as e:  # never let the net check itself block launch
        return True, f"sync guard: check errored ({type(e).__name__}), launching anyway"

    if head is None:
        return True, "sync guard: no remote reachable, launching (offline-safe)"

    remote_sha, provider = head
    synced = _load().get("synced_commit")

    if not synced:
        record(remote_sha, "first-run auto-accept")
        return True, (f"sync guard: first run on {socket.gethostname()}, "
                      f"accepting {provider}@{remote_sha[:10]}")

    if remote_sha == synced:
        return True, f"sync guard: in sync with {provider}@{remote_sha[:10]}"

    return False, (
        "\n  REFUSING TO LAUNCH -- another box has written since this one last synced.\n"
        f"    remote ({provider}): {remote_sha[:12]}\n"
        f"    this box last synced: {synced[:12]}\n\n"
        "  Catch up before launching:\n"
        f"    python3 {ROOT}/backup.py --pull\n"
        f"    python3 {ROOT}/backup.py --apply --yes\n"
        "  then relaunch Domicile.\n\n"
        "  Override only if you are certain this box is the canonical writer:\n"
        f"    CCHAT_SKIP_SYNC_GUARD=1 python3 {ROOT}/cchat.py\n")


if __name__ == "__main__":
    ok, msg = check()
    print(msg)
    raise SystemExit(0 if ok else 3)
