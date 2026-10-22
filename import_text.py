"""Import plain text / markdown files or folders as memory logs.

Each file becomes one titled chat (title = filename), dated by the file's
modification time. Handles .txt .md .markdown; skips binaries and empties.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import import_common as IC

EXTS = {".txt", ".md", ".markdown"}


def collect(src) -> list:
    p = Path(src).expanduser()
    if not p.exists():
        raise IC.ImportError_(f"Not found: {p}")
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*")
                                          if f.is_file() and f.suffix.lower() in EXTS)
    if not files:
        raise IC.ImportError_("No .txt or .md files found there.")
    return files


def to_messages(files: list, who: str = "Notes") -> list:
    out = []
    for f in files:
        try:
            body = f.read_text(encoding="utf-8").strip()
        except (UnicodeDecodeError, OSError):
            continue
        if not body:
            continue
        ts = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        out.append({"ts": ts, "who": who, "title": f.stem, "body": body})
    return out


def run(src, dest=None) -> dict:
    msgs = to_messages(collect(src))
    if not msgs:
        raise IC.ImportError_("Those files were empty or unreadable as text.")
    return IC.write_by_day("text", msgs, dest)
