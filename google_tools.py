"""Model-callable Google tools: thin wrappers over google_auth/*_client.py.

Registered into local_tools (schemas, impl, mutating set) at import time of
local_tools. Opt-in only: the names are NOT in PHASE_READONLY/WRITE/FULL, so a
role gets them only if its tool spec lists them (a privacy decision per role).

Write actions (GmailSend, CalendarCreate, BloggerPost, ChatSend) call the
approver hook before touching the network. With no approver configured they are
refused (DENIED), which is the safe default. The approve popup plugs in via
set_approver(fn): fn(tool, summary, args) -> bool, may block.

Result text follows the file tools' style; see bedroom/TOOL-CONTRACT.md section 5.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

_GA = str(Path(__file__).resolve().parent / "google_auth")

READ_TOOLS = ["GmailSearch", "GmailRead", "CalendarList", "ChatRead"]
WRITE_TOOLS = ["GmailSend", "CalendarCreate", "BloggerPost", "ChatSend"]
# GmailDraft writes but delivers nothing, so it needs no approval: the whole
# point of a draft is that a human reviews it before anything is sent.
UNGATED_TOOLS = ["GmailDraft"]
ALL_TOOLS = READ_TOOLS + WRITE_TOOLS + UNGATED_TOOLS

_APPROVER: Callable[[str, str, dict], bool] | None = None


def set_approver(fn: Callable[[str, str, dict], bool] | None) -> None:
    """Install (or clear) the approval callback for write actions."""
    global _APPROVER
    _APPROVER = fn


def _denied(msg: str):
    import local_tools as LT
    return LT.ToolDenied(msg)


def _approve(tool: str, summary: str, args: dict) -> None:
    if _APPROVER is None:
        raise _denied(f"{tool} needs your approval and no approver is configured, so it was not run.")
    if not _APPROVER(tool, summary, args):
        raise _denied(f"{tool} was not approved.")


def _client(name: str):
    if _GA not in sys.path:
        sys.path.insert(0, _GA)
    return __import__(name)


def _s(args: dict, key: str, required: bool = True) -> str:
    v = args.get(key)
    if v is None or str(v).strip() == "":
        if required:
            raise ValueError(f"{key} is required")
        return ""
    return str(v)


def _n(args: dict, key: str, default: int, lo: int = 1, hi: int = 100) -> int:
    v = args.get(key)
    return max(lo, min(int(v if v is not None else default), hi))


# read

def gmail_search(args: dict) -> dict:
    gc = _client("gmail_client")
    rows = gc.list_messages(str(args.get("query") or ""), _n(args, "max_results", 20))
    if not rows:
        return {"ok": True, "tool": "GmailSearch", "content": "(no matches)"}
    blocks = [f"{m['id']}  {m['date']}  {m['from']}  {m['subject']}\n    {m['snippet']}" for m in rows]
    return {"ok": True, "tool": "GmailSearch", "content": "\n".join(blocks)}


def gmail_read(args: dict) -> dict:
    m = _client("gmail_client").get_message(_s(args, "id"))
    text = (f"From: {m['from']}\nTo: {m['to']}\nSubject: {m['subject']}\nDate: {m['date']}\n\n"
            f"{m['body'] or '(no plain-text body)'}")
    return {"ok": True, "tool": "GmailRead", "content": text}


def calendar_list(args: dict) -> dict:
    rows = _client("calendar_client").list_upcoming(
        _n(args, "max_results", 10), str(args.get("calendar_id") or "primary"))
    if not rows:
        return {"ok": True, "tool": "CalendarList", "content": "(no upcoming events)"}
    lines = [f"{e['start']}  {e['summary']}" + (f"  [{e['location']}]" if e.get("location") else "")
             for e in rows]
    return {"ok": True, "tool": "CalendarList", "content": "\n".join(lines)}


def chat_read(args: dict) -> dict:
    rows = _client("chat_client").list_messages(_s(args, "space"), _n(args, "page_size", 20))
    if not rows:
        return {"ok": True, "tool": "ChatRead", "content": "(no messages)"}
    lines = [f"{m['createTime']}  {m['sender_display'] or m['sender_id']} ({m['sender_type']}): {m['text']}"
             for m in rows]
    return {"ok": True, "tool": "ChatRead", "content": "\n".join(lines)}


# write (approval-gated)

def gmail_send(args: dict) -> dict:
    to, subject, body = _s(args, "to"), _s(args, "subject"), _s(args, "body")
    _approve("GmailSend", f"Send email to {to}: {subject!r}", args)
    r = _client("gmail_client").send_message(to, subject, body, args.get("thread_id") or None)
    return {"ok": True, "tool": "GmailSend", "to": to, "id": r.get("id", "")}


def gmail_draft(args: dict) -> dict:
    """Ungated: a draft is not delivered, so nothing leaves the account."""
    to, subject, body = _s(args, "to"), _s(args, "subject"), _s(args, "body")
    r = _client("gmail_client").create_draft(to, subject, body, args.get("thread_id") or None)
    return {"ok": True, "tool": "GmailDraft", "to": to, "id": r.get("id", "")}


def calendar_create(args: dict) -> dict:
    summary, start, end = _s(args, "summary"), _s(args, "start"), _s(args, "end")
    _approve("CalendarCreate", f"Create event {summary!r} at {start}", args)
    r = _client("calendar_client").create_event(
        summary, start, end, str(args.get("description") or ""),
        str(args.get("calendar_id") or "primary"))
    return {"ok": True, "tool": "CalendarCreate", "id": r.get("id", ""), "start": start}


def blogger_post(args: dict) -> dict:
    title, html = _s(args, "title"), _s(args, "content_html")
    draft = True if args.get("draft") is None else bool(args.get("draft"))
    _approve("BloggerPost", f"{'Draft' if draft else 'PUBLISH LIVE'} blog post {title!r}", args)
    bc = _client("blogger_client")
    r = bc.create_post(bc.get_blog_id(), title, html, is_draft=draft)
    return {"ok": True, "tool": "BloggerPost", "id": r.get("id", ""),
            "status": "draft" if draft else "live"}


def chat_send(args: dict) -> dict:
    space, text = _s(args, "space"), _s(args, "text")
    _approve("ChatSend", f"Post to Google Chat {space}: {text[:60]!r}", args)
    r = _client("chat_client").send_message(space, text, args.get("thread") or None)
    return {"ok": True, "tool": "ChatSend", "space": space, "name": r.get("name", "")}


IMPL = {
    "GmailSearch": gmail_search, "GmailRead": gmail_read,
    "CalendarList": calendar_list, "ChatRead": chat_read,
    "GmailSend": gmail_send, "GmailDraft": gmail_draft,
    "CalendarCreate": calendar_create,
    "BloggerPost": blogger_post, "ChatSend": chat_send,
}

_STR = {"type": "string"}
_INT = {"type": "integer"}

SCHEMAS = {
    "GmailSearch": {
        "description": "Search Gmail. Returns one block per message: id, date, sender, subject, then the snippet.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Gmail search syntax, e.g. is:unread or from:x newer_than:2d"},
            "max_results": {"type": "integer", "description": "Default 20, max 100"},
        }}},
    "GmailRead": {
        "description": "Read one email by id (from GmailSearch). Returns headers and the plain-text body.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string", "description": "Message id"}},
                       "required": ["id"]}},
    "GmailSend": {
        "description": "Send a plain-text email. Needs the user's approval.",
        "parameters": {"type": "object", "properties": {
            "to": _STR, "subject": _STR, "body": _STR,
            "thread_id": {"type": "string", "description": "Reply within this thread"},
        }, "required": ["to", "subject", "body"]}},
    "GmailDraft": {
        "description": "Save a plain-text email as a draft instead of sending it. Nothing is "
                       "delivered; a human reviews the draft and sends it. Use this when asked "
                       "to draft something for review.",
        "parameters": {"type": "object", "properties": {
            "to": _STR, "subject": _STR, "body": _STR,
            "thread_id": {"type": "string", "description": "Reply within this thread"},
        }, "required": ["to", "subject", "body"]}},
    "CalendarList": {
        "description": "List upcoming calendar events, soonest first: start, title, [location].",
        "parameters": {"type": "object", "properties": {
            "max_results": {"type": "integer", "description": "Default 10"},
            "calendar_id": {"type": "string", "description": "Default primary"},
        }}},
    "CalendarCreate": {
        "description": "Create a calendar event. Needs the user's approval.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "Event title"},
            "start": {"type": "string", "description": "RFC3339, e.g. 2026-09-25T14:00:00-04:00"},
            "end": {"type": "string", "description": "RFC3339"},
            "description": _STR, "calendar_id": _STR,
        }, "required": ["summary", "start", "end"]}},
    "BloggerPost": {
        "description": "Create a Blogger post (a draft unless draft is false). Needs the user's approval.",
        "parameters": {"type": "object", "properties": {
            "title": _STR, "content_html": {"type": "string", "description": "Post body as HTML"},
            "draft": {"type": "boolean", "description": "Default true"},
        }, "required": ["title", "content_html"]}},
    "ChatRead": {
        "description": "Read recent Google Chat messages in a space, newest first.",
        "parameters": {"type": "object", "properties": {
            "space": {"type": "string", "description": "e.g. spaces/AAAA"},
            "page_size": {"type": "integer", "description": "Default 20"},
        }, "required": ["space"]}},
    "ChatSend": {
        "description": "Post a message to a Google Chat space. Needs the user's approval.",
        "parameters": {"type": "object", "properties": {
            "space": _STR, "text": _STR,
            "thread": {"type": "string", "description": "Thread name to reply in"},
        }, "required": ["space", "text"]}},
}
