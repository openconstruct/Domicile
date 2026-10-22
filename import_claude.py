"""Import a Claude.ai data export (conversations.json) as transcript logs.

Accepts the .zip Anthropic emails you, the extracted folder, or the
conversations.json itself. Only reads; writes `claude-<day>.log` files.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import import_common as IC


def load_conversations(src) -> list:
    p = Path(src).expanduser()
    if not p.exists():
        raise IC.ImportError_(f"Not found: {p}")
    if p.is_dir():
        p = p / "conversations.json"
        if not p.exists():
            raise IC.ImportError_("That folder has no conversations.json.")
    try:
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                name = next((n for n in z.namelist()
                             if n.endswith("conversations.json")), None)
                if not name:
                    raise IC.ImportError_("No conversations.json inside that zip.")
                data = json.loads(z.read(name))
        else:
            data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, zipfile.BadZipFile) as e:
        raise IC.ImportError_(f"Couldn't read that file as an export: {e}") from None
    if not isinstance(data, list):
        raise IC.ImportError_("Unexpected export shape (expected a list of conversations).")
    return data


def _text(msg: dict) -> str:
    t = (msg.get("text") or "").strip()
    if t:
        return t
    parts = [c.get("text", "") for c in (msg.get("content") or [])
             if isinstance(c, dict) and c.get("type") == "text"]
    return "\n".join(x for x in parts if x).strip()


def to_messages(convs: list, user_name: str = "User", agent_name: str = "Claude") -> list:
    out = []
    for c in convs:
        title = (c.get("name") or "").strip() or "untitled"
        for m in c.get("chat_messages") or []:
            body = _text(m)
            ts = m.get("created_at") or c.get("created_at") or ""
            if not body or not ts:
                continue
            who = user_name if m.get("sender") == "human" else agent_name
            out.append({"ts": ts[:19].replace("T", " "), "who": who,
                        "title": title, "body": body})
    out.sort(key=lambda m: m["ts"])
    return out


def run(src, user_name="User", agent_name="Claude", dest=None) -> dict:
    msgs = to_messages(load_conversations(src), user_name, agent_name)
    if not msgs:
        raise IC.ImportError_("No messages found in that export.")
    return IC.write_by_day("claude", msgs, dest)
