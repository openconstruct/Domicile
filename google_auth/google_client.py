"""
Shared Google API client helper.

Loads data/google/token.json (written by the setup wizard's Google step),
auto-refreshes it if expired, and hands back authenticated Gmail/Calendar/
Blogger/Chat service objects. Deliberately does NOT fall back to a browser
flow if the token is missing or the refresh token is dead: that failure needs a
human (python3 wizard_google.py signin), not a silent hang.

Usage:
    from google_client import (
        get_gmail_service, get_calendar_service,
        get_blogger_service, get_chat_service,
    )

    gmail = get_gmail_service()
    cal = get_calendar_service()
    blogger = get_blogger_service()
    chat = get_chat_service()
"""

import os
import sys
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

_DOMICILE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DOMICILE not in sys.path:
    sys.path.insert(0, _DOMICILE)

import paths  # noqa: E402

TOKEN_FILE = str(paths.GOOGLE_TOKEN)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/blogger",
    "https://www.googleapis.com/auth/chat.messages",
    "https://www.googleapis.com/auth/chat.spaces",
]


def _load_creds():
    if not os.path.exists(TOKEN_FILE):
        raise RuntimeError(
            f"{TOKEN_FILE} not found. Run the Google step of the setup wizard "
            f"(python3 wizard_google.py signin)."
        )

    # Use the scopes recorded in token.json (what Google actually granted), not the
    # SCOPES list above: refreshing with a scope that was never granted is rejected,
    # which breaks any install that skipped a service (e.g. no Chat on a plain Gmail account).
    creds = Credentials.from_authorized_user_file(TOKEN_FILE)

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Keep the token private and never leave a half-written file behind.
        tmp = TOKEN_FILE + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(creds.to_json())
        os.replace(tmp, TOKEN_FILE)

    return creds


def get_gmail_service():
    return build("gmail", "v1", credentials=_load_creds())


def get_calendar_service():
    return build("calendar", "v3", credentials=_load_creds())


def get_blogger_service():
    return build("blogger", "v3", credentials=_load_creds())


def get_chat_service():
    return build("chat", "v1", credentials=_load_creds())
