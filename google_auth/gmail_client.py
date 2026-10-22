"""
Thin Gmail API wrapper, matching blogger_client.py's shape (import the
service builder, call a small function, don't touch raw HTTP anywhere else).

Usage:
    from gmail_client import list_unread, get_message, send_message, mark_read

    for m in list_unread():
        print(m["from"], m["subject"], m["snippet"])
"""

import base64
from email.mime.text import MIMEText

from google_client import get_gmail_service


def _headers_to_dict(payload: dict) -> dict:
    return {h["name"]: h["value"] for h in payload.get("headers", [])}


def list_messages(query: str = "", max_results: int = 20) -> list[dict]:
    """List messages matching a Gmail search query (e.g. 'is:unread',
    'newer_than:1d'). Returns metadata (id, from, subject, date, snippet),
    not full bodies -- cheap to call often."""
    service = get_gmail_service()
    resp = service.users().messages().list(
        userId="me", q=query, maxResults=max_results
    ).execute()
    out = []
    for m in resp.get("messages", []):
        full = service.users().messages().get(
            userId="me", id=m["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()
        h = _headers_to_dict(full["payload"])
        out.append({
            "id": full["id"],
            "threadId": full.get("threadId"),
            "from": h.get("From", ""),
            "subject": h.get("Subject", ""),
            "date": h.get("Date", ""),
            "snippet": full.get("snippet", ""),
            "labelIds": full.get("labelIds", []),
        })
    return out


def list_unread(max_results: int = 20) -> list[dict]:
    return list_messages(query="is:unread", max_results=max_results)


def get_message(msg_id: str) -> dict:
    """Full message including plain-text body (best-effort extraction from
    the MIME tree -- takes the first text/plain part it finds)."""
    service = get_gmail_service()
    full = service.users().messages().get(
        userId="me", id=msg_id, format="full"
    ).execute()
    h = _headers_to_dict(full["payload"])
    body = _extract_plain_text(full["payload"])
    return {
        "id": full["id"],
        "threadId": full.get("threadId"),
        "from": h.get("From", ""),
        "to": h.get("To", ""),
        "subject": h.get("Subject", ""),
        "date": h.get("Date", ""),
        "snippet": full.get("snippet", ""),
        "body": body,
        "labelIds": full.get("labelIds", []),
    }


def _extract_plain_text(payload: dict) -> str:
    if payload.get("mimeType") == "text/plain" and "data" in payload.get("body", {}):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", "replace")
    for part in payload.get("parts", []) or []:
        text = _extract_plain_text(part)
        if text:
            return text
    return ""


def mark_read(msg_id: str) -> None:
    service = get_gmail_service()
    service.users().messages().modify(
        userId="me", id=msg_id, body={"removeLabelIds": ["UNREAD"]}
    ).execute()


def send_message(to: str, subject: str, body: str, thread_id: str | None = None) -> dict:
    """Send a plain-text email. Pass thread_id to reply within an existing
    thread (subject should then carry the Re: prefix if you want that)."""
    service = get_gmail_service()
    msg = MIMEText(body)
    msg["to"] = to
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    request_body = {"raw": raw}
    if thread_id:
        request_body["threadId"] = thread_id
    return service.users().messages().send(userId="me", body=request_body).execute()
