#!/usr/bin/env python3
"""
Google Chat bridge for Domicile -- Discord's poll-based sibling.

Google Chat has no equivalent of Discord's gateway websocket reachable from
a plain OAuth user token (that needs a registered Chat app + Pub/Sub, a much
bigger lift). So instead of push, this polls spaces.messages.list on an
interval, diffs against a persisted high-water mark, and feeds anything new
through the same on_message handler shape DiscordBridge uses -- cchat's
_on_chat calls run_turn with source="C", same as _on_discord does with "D".

Owns polling + dedup + the reply post. Knows nothing about Claude.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta
import sys
from pathlib import Path
from typing import Awaitable, Callable, NamedTuple

from paths import CHAT_CONFIG as CONFIG_PATH, CHAT_STATE as STATE_PATH
from inbound_files import MAX_FILE_BYTES, describe, save_inbound

HERE = Path(__file__).resolve().parent
GOOGLE_AUTH_DIR = HERE / "google_auth"
if str(GOOGLE_AUTH_DIR) not in sys.path:
    sys.path.insert(0, str(GOOGLE_AUTH_DIR))

import chat_client  # noqa: E402  -- needs the sys.path insert above

class InboundImage(NamedTuple):
    filename: str
    media_type: str
    data: bytes


Handler = Callable[[str, "list[InboundImage]"], Awaitable[str]]

MESSAGE_LIMIT = 4096  # Chat's text message cap is generous; this is a safety net
INLINE_IMAGE_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp")
MAX_INLINE_IMAGE = 3_750_000


# config

class BridgeConfig(NamedTuple):
    space: str
    authorized_sender: str | None   # "users/123..." ; None = accept anyone in the space
    poll_interval: float = 35.0


def load_config() -> BridgeConfig:
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"No {CONFIG_PATH}. Write it as "
            '{"space": "spaces/...", "authorized_sender": "users/...", '
            '"poll_interval": 35}'
        )
    try:
        data = json.loads(CONFIG_PATH.read_text())
    except Exception as exc:
        raise SystemExit(f"{CONFIG_PATH} is not valid JSON: {exc}")
    space = str(data.get("space", "")).strip()
    if not space:
        raise SystemExit(f"{CONFIG_PATH} needs a 'space' (e.g. 'spaces/abc123').")
    sender = str(data.get("authorized_sender", "") or "").strip() or None
    interval = float(data.get("poll_interval", 35.0))
    return BridgeConfig(space, sender, interval)


# state (high-water mark so a restart doesn't replay old messages)

def _load_state() -> dict:
    try:
        state = json.loads(STATE_PATH.read_text())
    except FileNotFoundError:
        return {"last_create_time": "", "last_seen_names": []}
    if not isinstance(state, dict):
        raise ValueError("Chat delivery state must be a JSON object")
    return state


def _save_state(state: dict) -> None:
    from config_loader import _write_json_atomic
    _write_json_atomic(STATE_PATH, state)


def _created(message):
    # RFC3339 representations can differ in fractional precision or UTC offset.
    return datetime.fromisoformat(message["createTime"].replace("Z", "+00:00"))


def _list_messages(space: str, since: str) -> list[dict]:
    """Fetch the complete backlog; retain the initial 20-message startup window."""
    messages = chat_client.get_chat_service().spaces().messages()
    params = {"parent": space, "pageSize": 100 if since else 20,
              "orderBy": "createTime asc" if since else "createTime desc"}
    if since:
        # The API supports >, not >=. Overlap a second to retain timestamp ties.
        cutoff = _created({"createTime": since}) - timedelta(seconds=1)
        params["filter"] = "createTime > " + json.dumps(cutoff.isoformat())
    out = []
    while True:
        page = messages.list(**params).execute()
        for message in page.get("messages", []):
            sender = message.get("sender", {})
            out.append({
                "name": message["name"], "createTime": message["createTime"],
                "text": message.get("text", ""),
                "sender_id": sender.get("name", ""),
                "sender_display": sender.get("displayName", ""),
                "sender_type": sender.get("type", ""),
                "thread": (message.get("thread") or {}).get("name", ""),
                "attachments": [{
                    "name": a.get("contentName", "attachment"),
                    "type": a.get("contentType", ""),
                    "ref": (a.get("attachmentDataRef") or {}).get("resourceName", ""),
                    "drive": bool(a.get("driveDataRef")),
                } for a in message.get("attachment") or []],
            })
        token = page.get("nextPageToken")
        if not token or (not since and len(out) >= 20):
            break
        params["pageToken"] = token
    if not since:
        out = out[:20]
    return sorted(out, key=lambda message: (_created(message), message["name"]))


def _download(resource_name: str) -> bytes:
    import io
    from googleapiclient.http import MediaIoBaseDownload
    req = chat_client.get_chat_service().media().download_media(resourceName=resource_name)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
        if buf.tell() > MAX_FILE_BYTES:
            raise ValueError("over the file size limit")
    return buf.getvalue()


def _fetch_attachments(atts: list[dict]):
    """(images, files, problems) for one message's attachments."""
    images, files, problems = [], [], []
    for a in atts:
        if a.get("drive") or not a.get("ref"):
            problems.append(f"{a['name']}: Drive attachment, not downloadable here")
            continue
        try:
            data = _download(a["ref"])
            saved = save_inbound(a["name"], data, a["type"], prefix="gchat")
        except Exception as exc:
            problems.append(f"{a['name']}: download failed ({type(exc).__name__}: {exc})")
            continue
        files.append(saved)
        if a["type"] in INLINE_IMAGE_TYPES and len(data) <= MAX_INLINE_IMAGE and len(images) < 6:
            images.append(InboundImage(a["name"], a["type"], data))
    return images, files, problems


def _split(body: str, limit: int) -> list[str]:
    chunks = []
    while len(body) > limit:
        cut = body.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = body.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(body[:cut].rstrip())
        body = body[cut:].lstrip("\n")
    return chunks + ([body] if body else [])


def _send_reply(space, body, thread, request_id):
    params = {"parent": space, "body": {"text": body}, "requestId": request_id}
    if thread:
        params["body"]["thread"] = {"name": thread}
        params["messageReplyOption"] = "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"
    messages = chat_client.get_chat_service().spaces().messages()
    response = messages.create(**params).execute()
    # An idempotent retry may return only names; obtain sender metadata for self-filtering.
    if response.get("name") and not (response.get("sender") or {}).get("name"):
        return messages.get(name=response["name"]).execute()
    return response


# bridge

class ChatBridge:
    """Owns the poll loop and the Chat API calls. Knows nothing about Claude."""

    def __init__(
        self,
        space: str,
        authorized_sender: str | None,
        on_message: Handler | None = None,
        *,
        poll_interval: float = 35.0,
        verbose: bool = True,
    ):
        self.space = space
        self.authorized_sender = authorized_sender
        self.on_message = on_message
        self.poll_interval = poll_interval
        self.verbose = verbose

        state = _load_state()
        if state.get("space", self.space) != self.space:
            state = {}
        self._last_create_time: str = state.get("last_create_time", "")
        # names already handled at the same createTime, to dedup ties
        self._last_seen_names: set[str] = set(state.get("last_seen_names", []))

        self._pending = state.get("pending")
        self._self_id: str | None = state.get("self_id")
        from config_loader import load_identity
        self._agent_name = load_identity()["agent_name"]
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[chat] {msg}", file=sys.stderr, flush=True)

    def _persist(self) -> None:
        _save_state({
            "space": self.space, "pending": self._pending, "self_id": self._self_id,
            "last_create_time": self._last_create_time,
            "last_seen_names": sorted(self._last_seen_names),
        })

    def _is_inbound(self, m: dict) -> bool:
        """A message worth reacting to: not from the agent, and (if configured)
        from the one authorized human."""
        sender = m["sender_id"]
        if self._self_id and sender == self._self_id:
            return False
        if sender.startswith("users/") and m["sender_display"] == self._agent_name:
            # belt-and-suspenders self-skip even before _self_id is known
            # (e.g. cold start, first poll after a restart)
            return False
        if self.authorized_sender and sender != self.authorized_sender:
            return False
        return True

    def _seen(self, message):
        if not self._last_create_time:
            return False
        created = _created(message)
        last = _created({"createTime": self._last_create_time})
        return created < last or (created == last and message["name"] in self._last_seen_names)

    def _acknowledge(self, message):
        created = message["createTime"]
        names = (self._last_seen_names | {message["name"]}
                 if self._last_create_time and _created(message) ==
                 _created({"createTime": self._last_create_time}) else {message["name"]})
        # Persist before advancing in memory, including after a successful send.
        _save_state({"space": self.space, "last_create_time": created,
                     "last_seen_names": sorted(names), "pending": None,
                     "self_id": self._self_id})
        self._last_create_time, self._last_seen_names = created, names
        self._pending = None

    async def _finish_pending(self):
        pending = self._pending
        message = pending["message"]
        # Retry a failed checkpoint before invoking the handler or sending anything.
        self._persist()
        if pending["reply"] is None:
            text = (message.get("text") or "").strip()
            images = []
            if message.get("attachments"):
                images, files, problems = await asyncio.to_thread(
                    _fetch_attachments, message["attachments"])
                text = describe(text, files, via="Google Chat", skipped=problems)
            pending["reply"] = (await self.on_message(text, images) if self.on_message
                                else f"echo: {text}")
            self._persist()
        await self._deliver(pending["reply"], thread=message.get("thread") or None,
                            request_id=pending["request_id"])
        self._acknowledge(message)

    async def _poll_once(self) -> None:
        try:
            # A saved reply is delivered even if polling is currently unavailable.
            if self._pending:
                await self._finish_pending()
            messages = await asyncio.to_thread(_list_messages, self.space, self._last_create_time)
            for message in messages:
                if self._seen(message):
                    continue
                if not self._is_inbound(message) or not (
                        (message.get("text") or "").strip() or message.get("attachments")):
                    self._acknowledge(message)
                    continue
                self._pending = {"message": message, "reply": None,
                                 "request_id": str(uuid.uuid4())}
                await self._finish_pending()
        except Exception as exc:
            # Leave the current message/reply pending and stop before the next one.
            self._log(f"poll/delivery failed; will retry: {type(exc).__name__}: {exc}")

    async def _deliver(self, text: str, *, thread: str | None = None,
                       request_id: str | None = None) -> None:
        body = (text or "").strip() or "*(empty reply)*"
        for i, chunk in enumerate(_split(body, MESSAGE_LIMIT)):
            if request_id:
                # per-chunk id keeps retries idempotent chunk-by-chunk
                rid = request_id if i == 0 else f"{request_id}-{i}"
                resp = await asyncio.to_thread(_send_reply, self.space, chunk, thread, rid)
            else:
                resp = await asyncio.to_thread(chat_client.send_message, self.space, chunk, thread)
            sender = (resp.get("sender") or {}).get("name")
            if sender:
                self._self_id = sender

    async def send(self, text: str) -> None:
        """Push an unprompted message. This is the wakeup's only voice on Chat."""
        await self._deliver(text)

    async def start(self) -> None:
        """Run until stop() is called or the task is cancelled."""
        self._ready.set()
        try:
            while not self._stop.is_set():
                await self._poll_once()
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        self._stop.set()

    async def wait_ready(self) -> None:
        await self._ready.wait()


# main (standalone testing)

def _bridge(**kw) -> ChatBridge:
    cfg = load_config()
    return ChatBridge(cfg.space, cfg.authorized_sender, poll_interval=cfg.poll_interval, **kw)


async def _echo_mode() -> None:
    bridge = _bridge()
    print(f"Polling {bridge.space} every {bridge.poll_interval}s. ctrl-c to stop.",
          file=sys.stderr)
    try:
        await bridge.start()
    except KeyboardInterrupt:
        pass


async def _send_mode(text: str) -> None:
    bridge = _bridge()
    await bridge.send(text)
    print(f"sent to {bridge.space}", file=sys.stderr)


if __name__ == "__main__":
    args = sys.argv[1:]
    try:
        if args and args[0] == "--send":
            asyncio.run(_send_mode(" ".join(args[1:]) or "ping from Domicile"))
        else:
            asyncio.run(_echo_mode())
    except KeyboardInterrupt:
        pass
