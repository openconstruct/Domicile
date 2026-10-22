"""Domicile data paths; existing on-disk names remain compatible."""

from __future__ import annotations

import os
from pathlib import Path

HOME = Path.home()

# Where Domicile is installed: wherever this file is, whatever the folder is called.
# Data lives in <root>/data unless DOMICILE_DATA says otherwise (the test runner uses
# that to point at a disposable directory; the wizard copies it into the service units).
ROOT = Path(__file__).resolve().parent
CCHAT_DATA = (Path(os.environ["DOMICILE_DATA"]).expanduser().resolve()
              if os.environ.get("DOMICILE_DATA") else ROOT / "data")
MEM0_DIR = CCHAT_DATA / "mem0"

# fastembed defaults to /tmp, which is tmpfs on many distros and wiped each
# boot (model re-downloads; offline boot breaks memory search). Keep it in data/.
FASTEMBED_CACHE = CCHAT_DATA / "fastembed_cache"
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(FASTEMBED_CACHE))

# cchat data (was ~/.ccchat)
CONFIG_PATH = CCHAT_DATA / "config.json"
ROLES_PATH = CCHAT_DATA / "roles.json"
IDENTITY_PATH = CCHAT_DATA / "identity.json"
RUNTIME_STATE = CCHAT_DATA / "runtime-state.json"
CHAT_DIR = CCHAT_DATA / "chats"
TRASH_DIR = CCHAT_DATA / "trash"
MEM1_DIR = CCHAT_DATA / "memory"
VERSIONS_DIR = MEM1_DIR / ".versions"
SESSIONS_DIR = CCHAT_DATA / "sessions"
MEDIA_DIR = CCHAT_DATA / "media"
UPLOADS_DIR = CCHAT_DATA / "uploads"
NOTES_DIR = CCHAT_DATA / "notes"
ARCHIVE_DIR = CCHAT_DATA / "archive"
PRE_APPLY_DIR = CCHAT_DATA / "pre-apply"

CRASH_LOG = CCHAT_DATA / "crash.log"
WAKEUP_LOG = CCHAT_DATA / "wakeup.log"
HEARTBEAT_FILE = CCHAT_DATA / "heartbeat"
WATCHDOG_LOG = CCHAT_DATA / "watchdog.log"
SELFTEST_LOG = CCHAT_DATA / "selftest.log"
SELFTEST_STATUS = CCHAT_DATA / "selftest-status.json"
WAKEUP_CONFIG = CCHAT_DATA / "wakeup.json"
COMPACT_REQUEST = CCHAT_DATA / "compact-request"   # marker: compact after the current turn
CONTEXT_DIET_LOG = CCHAT_DATA / "context_diet.log"
BGTASK_DIR = CCHAT_DATA / "bgtasks"
EXPRESSION_FILE = CCHAT_DATA / "expression.md"
TOKEN_LOG = CCHAT_DATA / "token_usage.csv"
CHAT_CONFIG = CCHAT_DATA / "chat.json"
CHAT_STATE = CCHAT_DATA / "chat-state.json"
# Google (Gmail / Calendar / Blogger / Chat): code in domicile/google_auth/, state here.
GOOGLE_DIR = CCHAT_DATA / "google"
GOOGLE_TOKEN = GOOGLE_DIR / "token.json"
GOOGLE_CLIENT_SECRET = GOOGLE_DIR / "client_secret.json"
GMAIL_SEEN = GOOGLE_DIR / "gmail_seen.json"
GMAIL_UNREAD = GOOGLE_DIR / "gmail_unread_count.json"
# RSS: code in rss/, state here, starter feed list in templates/rss/feeds.txt.
RSS_DIR = CCHAT_DATA / "rss"
RSS_FEEDS = RSS_DIR / "feeds.txt"
RSS_SEEN = RSS_DIR / "seen.json"
RSS_LATEST = RSS_DIR / "latest.md"
RSS_FULL = RSS_DIR / "latest-full.md"
SYNC_STATE = CCHAT_DATA / "sync-state.json"
ROLLOVER_LOG = CCHAT_DATA / "rollover.log"
OPENROUTER_LOG = CCHAT_DATA / "openrouter.log"


BACKUP_STAGE = CCHAT_DATA / "backup"
BACKUP_LOG = CCHAT_DATA / "backup.log"
BACKUP_STATE = CCHAT_DATA / "backup-state.json"

# mem0 store (was ~/.mem0)
MEM0_LOG_DIR = MEM0_DIR / "logs"
INGESTED_DIR = MEM0_LOG_DIR / "ingested"
SOCKET_PATH = MEM0_DIR / "cchat-memory.sock"
QDRANT_PATH = MEM0_DIR / "qdrant"
QDRANT_MEM2_PATH = MEM0_DIR / "qdrant-mem2"
MEM2_FTS_PATH = MEM0_DIR / "mem2-fts.db"
MEM2_INDEX_STATE = MEM0_DIR / "mem2-index-state.json"
MEM2_FTS_STATE = MEM0_DIR / "mem2-fts-state.json"
MEM0_STATE_PATH = MEM0_DIR / "nightly-state.json"
INGEST_STATUS_PATH = MEM0_DIR / "ingest-status.json"
HOURLY_INGEST_STATUS_PATH = MEM0_DIR / "hourly-ingest-status.json"  # in-process hourly drain -- the live ingest