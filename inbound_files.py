"""Save files that arrive over a bridge (Discord, Google Chat) into UPLOADS_DIR,
the same place the web UI puts uploads, and describe them for the turn."""
from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

from paths import UPLOADS_DIR

MAX_FILE_BYTES = 25 * 1024 * 1024  # matches api_server.MAX_UPLOAD


class SavedFile(NamedTuple):
    path: Path
    size: int
    media_type: str


def _safe_name(name: str) -> str:
    raw = Path(name or "").name.strip().replace("\\", "_")
    raw = raw.lstrip(".") or "file"
    return raw[:120]


def save_inbound(name: str, data: bytes, media_type: str, *, prefix: str) -> SavedFile:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    raw = _safe_name(name)
    dest = UPLOADS_DIR / f"{stamp}-{prefix}-{raw}"
    n = 1
    while dest.exists():
        dest = UPLOADS_DIR / f"{stamp}-{prefix}-{n}-{raw}"
        n += 1
    dest.write_bytes(data)
    dest.chmod(0o600)
    return SavedFile(dest, len(data), media_type or "application/octet-stream")


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024 or unit == "MB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def describe(text: str, files: list[SavedFile], *, via: str, skipped=()) -> str:
    """Append an attachment notice to the message text (like upload_turn does)."""
    if not files and not skipped:
        return text
    lines = [f"- `{f.path}` ({_human(f.size)}, {f.media_type})" for f in files]
    notice = f"[Attached via {via}, saved to disk:]\n" + "\n".join(lines) if files else ""
    if skipped:
        notice += ("\n" if notice else "") + "[Attachment problems:]\n" + \
                  "\n".join(f"- {s}" for s in skipped)
    return (text.strip() + "\n\n" + notice) if text.strip() else notice
