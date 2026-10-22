"""Shared pieces for the memory importers (wizard step 3.4).

Every importer turns its source into transcript logs in MEM0_LOG_DIR, in the
same `=== compacted from chat ... ===` block format mem0_nightly already
ingests. Importers stay separate modules (decided 09-21); only the block
writer and chunker live here. Pure logic, no UI, no prompts.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from paths import MEM0_LOG_DIR

TARGET_CHARS = 3200


class ImportError_(ValueError):
    """User-correctable problem; message is safe to show verbatim."""


def chat_id(prefix: str, title: str) -> str:
    return prefix + hashlib.md5(title.encode()).hexdigest()[:10]


def group(messages: list[dict]) -> list[dict]:
    """messages: {ts, who, title, body}. Consecutive same-title messages are
    packed into blocks of <= TARGET_CHARS. One oversized message is split."""
    blocks, buf, size, title, first = [], [], 0, None, None

    def flush():
        nonlocal buf, size, first
        if buf:
            blocks.append({"title": title, "ts": first, "text": "\n".join(buf)})
        buf, size, first = [], 0, None

    for m in messages:
        body = m["body"]
        pieces = [body[i:i + TARGET_CHARS] for i in range(0, len(body), TARGET_CHARS)] or [""]
        for n, piece in enumerate(pieces):
            line = f"[{m['ts']}] {m['who']}: {piece}" if n == 0 else piece
            if buf and (m["title"] != title or size + len(line) > TARGET_CHARS):
                flush()
            title = m["title"]
            first = first or m["ts"]
            buf.append(line)
            size += len(line)
    flush()
    return blocks


def write_blocks(prefix: str, day: str, blocks: list[dict],
                 dest: Path | None = None) -> Path | None:
    """Write one `<prefix>-<day>.log`. Refuses to overwrite: a re-run of the
    same import must not silently replace logs mem0 may already have ingested."""
    if not blocks:
        return None
    dest = Path(dest or MEM0_LOG_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{day}-{prefix}.log"  # date FIRST: mem0_nightly skips stems >= today
    if out.exists():
        return None
    lines = []
    for b in blocks:
        t = (b["title"] or "untitled").replace("'", "").replace("\n", " ")[:80]
        lines += [f"=== compacted from chat '{chat_id(prefix[:2], b['title'] or 'untitled')}' "
                  f"('{t}') at {b['ts']} ===", b["text"], ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_by_day(prefix: str, messages: list[dict], dest: Path | None = None) -> dict:
    """Bucket by the first 10 chars of ts (YYYY-MM-DD). Returns a summary."""
    days: dict[str, list] = {}
    for m in messages:
        days.setdefault(str(m["ts"])[:10], []).append(m)
    written = skipped = chunks = 0
    for day, msgs in sorted(days.items()):
        blocks = group(msgs)
        if write_blocks(prefix, day, blocks, dest):
            written += 1
            chunks += len(blocks)
        else:
            skipped += 1
    return {"messages": len(messages), "files": written,
            "skipped_existing": skipped, "chunks": chunks}
