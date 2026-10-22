"""
Thin Google Chat API wrapper, matching gmail_client.py/blogger_client.py's
shape (import the service builder, call a small function, don't touch raw
HTTP anywhere else).

Usage:
    from chat_client import list_messages, send_message

    for m in list_messages("spaces/nqc_EqAAAAE"):
        print(m["sender_name"], m["text"])

    send_message("spaces/nqc_EqAAAAE", "hello")
"""

from google_client import get_chat_service


def list_messages(space: str, page_size: int = 20) -> list[dict]:
    """Most recent messages in a space, newest first. Cheap metadata-shaped
    dicts -- sender id/name/type, text, createTime, message resource name."""
    service = get_chat_service()
    resp = service.spaces().messages().list(
        parent=space, pageSize=page_size, orderBy="createTime desc"
    ).execute()
    out = []
    for m in resp.get("messages", []):
        sender = m.get("sender", {})
        out.append({
            "name": m.get("name", ""),
            "createTime": m.get("createTime", ""),
            "text": m.get("text", ""),
            "sender_id": sender.get("name", ""),        # e.g. "users/1234..."
            "sender_display": sender.get("displayName", ""),
            "sender_type": sender.get("type", ""),        # "HUMAN" or "BOT"
            "thread": (m.get("thread") or {}).get("name", ""),
        })
    return out


def send_message(space: str, text: str, thread: str | None = None) -> dict:
    """Post a message into a space. Pass thread (a thread resource name from
    list_messages) to reply within an existing thread."""
    service = get_chat_service()
    body: dict = {"text": text}
    if thread:
        body["thread"] = {"name": thread}
    return service.spaces().messages().create(
        parent=space, body=body,
        **({"messageReplyOption": "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"} if thread else {}),
    ).execute()
