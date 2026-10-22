#!/usr/bin/env python3
"""Claude archive conversion to Domicile transcript logs."""

import re
import sys
from pathlib import Path

HOME = Path.home()
SRC = HOME / "Downloads" / "claude_by_day"
from paths import MEM0_LOG_DIR as DEST

TARGET_CHARS = 3200

MSG_RE = re.compile(
    r"^### \[(?P<ts>\d{4}-\d{2}-\d{2}[^\]]*)\] (?P<who>[^_\n]+?)"
    r"(?:\s+_\(in: (?P<title>.*?)\)_)?\s*$",
    re.M,
)


def parse_messages(text: str) -> list[dict]:
    """Split a day file into messages, each with its chat title."""
    out = []
    matches = list(MSG_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        body = body.strip("-").strip()  # trailing --- rules between messages
        if not body:
            continue
        out.append({
            "ts": m.group("ts").strip(),
            "who": m.group("who").strip(),
            "title": (m.group("title") or "untitled").strip(),
            "body": body,
        })
    return out


def group(messages: list[dict]) -> list[dict]:
    """Group consecutive same-chat messages into <= TARGET_CHARS blocks."""
    blocks, buf, title, first_ts = [], [], None, None
    for msg in messages:
        line = f"[{msg['ts']}] {msg['who']}: {msg['body']}"
        too_big = buf and sum(len(x) for x in buf) + len(line) > TARGET_CHARS
        if buf and (msg["title"] != title or too_big):
            blocks.append({"title": title, "ts": first_ts, "text": "\n".join(buf)})
            buf, first_ts = [], None
        title = msg["title"]
        first_ts = first_ts or msg["ts"]
        buf.append(line)
    if buf:
        blocks.append({"title": title, "ts": first_ts, "text": "\n".join(buf)})
    return blocks


def slug(title: str) -> str:
    """Stable pseudo chat_id from the title -- the export has no real ids."""
    import hashlib
    return "cl" + hashlib.md5(title.encode()).hexdigest()[:10]


def convert(path: Path) -> tuple[Path, int]:
    date = path.stem
    messages = parse_messages(path.read_text(encoding="utf-8", errors="replace"))
    blocks = group(messages)
    lines = []
    for b in blocks:
        safe_title = b["title"].replace("'", "").replace("\n", " ")[:80]
        lines.append(
            f"=== compacted from chat '{slug(b['title'])}' "
            f"('{safe_title}') at {b['ts']} ==="
        )
        lines.append(b["text"])
        lines.append("")
    dest = DEST / f"claude-{date}.log"
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest, len(blocks)


def main() -> int:
    if not SRC.exists():
        print(f"source missing: {SRC}", file=sys.stderr)
        return 1
    DEST.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.doc"))
    total_blocks = 0
    empty = []
    for path in files:
        dest, n = convert(path)
        total_blocks += n
        if n == 0:
            empty.append(path.name)
    print(f"converted {len(files)} day files -> {total_blocks} chunks in {DEST}")
    if empty:
        print(f"WARNING: {len(empty)} files produced zero chunks: {empty[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
