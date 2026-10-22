"""Import a ChatGPT data export as transcript logs.

Accepts the .zip OpenAI emails you (it nests a Conversations zip inside),
the inner Conversations zip, an extracted folder, or one conversations-*.json.
Only reads; writes `<day>-chatgpt.log` files.

ChatGPT stores each chat as a tree (regenerated replies are branches). We
follow `current_node` back to the root so only the branch you actually kept
is imported. Hidden reasoning ("thoughts", "reasoning_recap") is skipped.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import import_common as IC

_CONV = re.compile(r"(^|/)conversations(-\d+)?\.json$")


def _from_zip(z: zipfile.ZipFile, depth: int = 0) -> list:
    out = []
    for n in z.namelist():
        if _CONV.search(n):
            out.extend(_as_list(json.loads(z.read(n))))
        elif n.lower().endswith(".zip") and "onversations" in n and depth < 2:
            with zipfile.ZipFile(io.BytesIO(z.read(n))) as inner:
                out.extend(_from_zip(inner, depth + 1))
    return out


def _as_list(data) -> list:
    if not isinstance(data, list):
        raise IC.ImportError_("Unexpected export shape (expected a list of conversations).")
    return data


def load_conversations(src) -> list:
    p = Path(src).expanduser()
    if not p.exists():
        raise IC.ImportError_(f"Not found: {p}")
    try:
        if p.is_dir():
            files = sorted(f for f in p.rglob("*.json") if _CONV.search(f.as_posix()))
            data = [c for f in files for c in _as_list(json.loads(f.read_text(encoding="utf-8")))]
        elif p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                data = _from_zip(z)
        else:
            data = _as_list(json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, zipfile.BadZipFile) as e:
        raise IC.ImportError_(f"Couldn't read that file as an export: {e}") from None
    if not data:
        raise IC.ImportError_("No conversations found in that export.")
    return data


def _branch(conv: dict) -> list:
    """Messages on the kept branch, root -> current_node."""
    mapping = conv.get("mapping") or {}
    node, chain, seen = conv.get("current_node"), [], set()
    while node and node in mapping and node not in seen:
        seen.add(node)
        chain.append(mapping[node])
        node = mapping[node].get("parent")
    return [n["message"] for n in reversed(chain) if n.get("message")]


def _text(msg: dict) -> str:
    c = msg.get("content") or {}
    if c.get("content_type") not in ("text", "multimodal_text"):
        return ""
    return "\n".join(p.strip() for p in (c.get("parts") or [])
                     if isinstance(p, str) and p.strip())


def _ts(t) -> str:
    if not t:
        return ""
    return datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def to_messages(convs: list, user_name: str = "User", agent_name: str = "ChatGPT") -> list:
    out = []
    for c in convs:
        title = (c.get("title") or "").strip() or "untitled"
        for m in _branch(c):
            role = (m.get("author") or {}).get("role")
            if role not in ("user", "assistant"):
                continue
            if (m.get("metadata") or {}).get("is_visually_hidden_from_conversation"):
                continue
            body = _text(m)
            ts = _ts(m.get("create_time") or c.get("create_time"))
            if not body or not ts:
                continue
            out.append({"ts": ts, "who": user_name if role == "user" else agent_name,
                        "title": title, "body": body})
    out.sort(key=lambda m: m["ts"])
    return out


def run(src, user_name="User", agent_name="ChatGPT", dest=None) -> dict:
    msgs = to_messages(load_conversations(src), user_name, agent_name)
    if not msgs:
        raise IC.ImportError_("No messages found in that export.")
    return IC.write_by_day("chatgpt", msgs, dest)
