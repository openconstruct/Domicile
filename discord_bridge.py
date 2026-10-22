#!/usr/bin/env python3
"""Discord messages and attachments for Domicile."""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Awaitable, Callable, NamedTuple

import discord

import credentials
from inbound_files import MAX_FILE_BYTES, SavedFile, describe, save_inbound
MESSAGE_LIMIT = 2000

Handler = Callable[[str, "list[InboundImage]"], Awaitable[str]]


class InboundImage(NamedTuple):
    """One image the human attached, already downloaded."""

    filename: str
    media_type: str          # e.g. "image/png" -- what the API wants
    data: bytes


MAX_IMAGE_BYTES = 3_750_000

MAX_IMAGES = 6

SUPPORTED_IMAGE_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp")


# config

class BridgeConfig(NamedTuple):
    token: str
    user_id: int
    channel_id: int | None = None      # None = DM-only, the old behaviour
    require_mention: bool = False      # server channel only; DMs never need it


def load_config() -> BridgeConfig:
    """Returns a BridgeConfig. Raises SystemExit with a useful message."""
    token = credentials.get("CCCHAT_DISCORD_TOKEN")
    uid_raw = credentials.get("CCCHAT_DISCORD_USER_ID")
    chan_raw = credentials.get("CCCHAT_DISCORD_CHANNEL_ID")
    require_mention = credentials.get("CCCHAT_DISCORD_REQUIRE_MENTION", "false").lower() in (
        "1", "true", "yes", "on")
    if not token:
        raise SystemExit(f"Set CCCHAT_DISCORD_TOKEN in {credentials.ENV_PATH}.")
    if not uid_raw:
        raise SystemExit("No authorized user id. Set CCCHAT_DISCORD_USER_ID.")
    try:
        user_id = int(uid_raw)
    except ValueError:
        raise SystemExit(f"user_id must be a number, got {uid_raw!r}")

    channel_id: int | None = None
    if chan_raw:
        try:
            channel_id = int(chan_raw)
        except ValueError:
            raise SystemExit(f"channel_id must be a number, got {chan_raw!r}")

    return BridgeConfig(token, user_id, channel_id, require_mention)


def save_config(
    token: str,
    user_id: int,
    channel_id: int | None = None,
    require_mention: bool = False,
) -> None:
    credentials.save({
        "CCCHAT_DISCORD_TOKEN": token,
        "CCCHAT_DISCORD_USER_ID": str(user_id),
        "CCCHAT_DISCORD_CHANNEL_ID": str(channel_id or ""),
        "CCCHAT_DISCORD_REQUIRE_MENTION": str(require_mention).lower(),
    })


# chunking

def split_message(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    """Split for Discord's 2000-char cap, keeping code fences balanced."""
    text = (text or "").strip()
    if not text:
        return ["*(empty reply)*"]
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    fence: str | None = None  # the opening fence line, if we're inside one

    def flush() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        body = "\n".join(cur)
        if fence is not None:
            body += "\n```"
        chunks.append(body)
        cur = [fence] if fence is not None else []
        cur_len = len(fence) + 1 if fence is not None else 0

    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            if fence is None:
                if cur_len + len(line) + 5 > limit:
                    flush()
                fence = line
            else:
                fence = None

        while len(line) > limit - 8:
            head, line = line[: limit - 8], line[limit - 8 :]
            if cur:
                flush()
            chunks.append(head)

        if cur_len + len(line) + 1 > limit - (4 if fence else 0):
            flush()
        cur.append(line)
        cur_len += len(line) + 1

    flush()
    return [c for c in chunks if c.strip() and c.strip() != "```"]


# attachments

SEND_FILE_RE = re.compile(
    r"\[\[\s*send_file\s*:\s*(?P<path>[^\]]+?)"
    r"(?:\s+as\s+(?P<name>[^\]/]+?))?\s*\]\]",
    re.IGNORECASE,
)

MAX_OUTBOUND_FILES = 4
MAX_OUTBOUND_BYTES = 25_000_000


def extract_send_files(text: str):
    """(clean_text, [(Path, rename_or_None)], [problem strings])."""
    attachments: list[tuple[Path, str | None]] = []
    problems: list[str] = []

    for m in SEND_FILE_RE.finditer(text or ""):
        raw = (m.group("path") or "").strip().strip("'\"")
        rename = (m.group("name") or "").strip() or None
        p = Path(raw).expanduser()
        if len(attachments) >= MAX_OUTBOUND_FILES:
            problems.append(f"{p.name}: over the {MAX_OUTBOUND_FILES}-file limit")
            continue
        if not p.is_file():
            problems.append(f"{raw}: no such file")
            continue
        try:
            size = p.stat().st_size
        except OSError as exc:
            problems.append(f"{p.name}: unreadable ({type(exc).__name__})")
            continue
        if size > MAX_OUTBOUND_BYTES:
            problems.append(
                f"{p.name}: {size / 1e6:.1f}MB is over Discord's "
                f"{MAX_OUTBOUND_BYTES / 1e6:.0f}MB limit"
            )
            continue
        attachments.append((p, rename))

    clean = SEND_FILE_RE.sub("", text or "")
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, attachments, problems


# bridge

class DiscordBridge:
    """Owns the discord client. Knows nothing about Claude."""

    def __init__(
        self,
        token: str,
        user_id: int,
        on_message: Handler | None = None,
        *,
        channel_id: int | None = None,
        require_mention: bool = False,
        verbose: bool = True,
    ):
        self.token = token
        self.user_id = user_id
        self.on_message = on_message
        self.channel_id = channel_id
        self.require_mention = require_mention
        self.verbose = verbose

        self._last_channel: discord.abc.Messageable | None = None
        self._ready = asyncio.Event()

        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True
        intents.guild_messages = True
        self.client = discord.Client(intents=intents)

        @self.client.event
        async def on_ready() -> None:
            where = f"channel {self.channel_id}" if self.channel_id else "DM"
            self._log(
                f"connected as {self.client.user} "
                f"(authorized user {self.user_id}, outbound -> {where})"
            )
            self._ready.set()

        @self.client.event
        async def on_message(message: discord.Message) -> None:
            await self._handle(message)


    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[discord] {msg}", file=sys.stderr, flush=True)

    def _strip_mention(self, text: str) -> str:
        """Drop leading @bot so the model never sees Discord's <@123> noise."""
        me = getattr(self.client.user, "id", None)
        if me is None:
            return text
        return re.sub(rf"<@!?{me}>", "", text).strip()

    def _addressed(self, message: discord.Message) -> bool:
        """Is this message meant for us?"""
        if isinstance(message.channel, discord.DMChannel):
            return True
        if self.channel_id is None or message.channel.id != self.channel_id:
            return False
        if self.require_mention:
            me = getattr(self.client.user, "id", None)
            return any(u.id == me for u in message.mentions)
        return True

    async def _handle(self, message: discord.Message) -> None:
        if message.author.id == getattr(self.client.user, "id", None):
            return  # our own message
        if message.author.id != self.user_id:
            return  # not the authorized human; silence, not an error reply
        if not self._addressed(message):
            return

        text = self._strip_mention((message.content or "").strip())
        images, skipped, files = await self._collect_images(message)
        if not text and not images and not skipped and not files:
            return
        text = describe(text, files, via="Discord", skipped=skipped)

        self._last_channel = message.channel  # fallback for unprompted pushes
        self._log(f"<- {text[:80]}" + (f" [+{len(images)} image(s)]" if images else ""))

        if skipped:
            await message.channel.send("\n".join(f"⚠️ {s}" for s in skipped))

        if self.on_message is None:
            reply = f"echo: {text}" + (f" [{len(images)} image(s)]" if images else "")
        else:
            try:
                async with message.channel.typing():
                    reply = await self.on_message(text, images)
            except Exception as exc:
                reply = f"**Bridge error:** `{type(exc).__name__}: {exc}`"
                self._log(reply)

        await self._deliver(message.channel, reply)
        self._log(f"-> {reply[:80]}")

    async def _collect_images(self, message: discord.Message):
        """(images, skipped, files) for a message's attachments.

        Every attachment is saved to UPLOADS_DIR (files); supported images
        are additionally passed inline (images)."""
        images: list[InboundImage] = []
        skipped: list[str] = []
        files: list[SavedFile] = []
        for att in message.attachments:
            ctype = (att.content_type or "").split(";")[0].strip().lower()
            if att.size and att.size > MAX_FILE_BYTES:
                skipped.append(f"{att.filename}: {att.size / 1e6:.1f}MB is over the "
                               f"{MAX_FILE_BYTES / 1e6:.0f}MB file limit")
                continue
            try:
                data = await att.read()
                files.append(save_inbound(att.filename, data, ctype, prefix="discord"))
            except Exception as exc:
                skipped.append(f"{att.filename}: save failed ({type(exc).__name__})")
                continue
            if not ctype.startswith("image/"):
                continue  # saved to disk; path goes in the text
            if len(images) >= MAX_IMAGES:
                skipped.append(f"{att.filename}: over the {MAX_IMAGES}-image limit")
                continue
            if ctype not in SUPPORTED_IMAGE_TYPES:
                skipped.append(f"{att.filename}: {ctype} isn't a supported image type")
                continue
            if att.size and att.size > MAX_IMAGE_BYTES:
                skipped.append(
                    f"{att.filename}: {att.size / 1e6:.1f}MB is over the "
                    f"{MAX_IMAGE_BYTES / 1e6:.1f}MB limit"
                )
                continue
            if len(data) > MAX_IMAGE_BYTES:
                skipped.append(f"{att.filename}: {len(data) / 1e6:.1f}MB is too large")
                continue
            images.append(InboundImage(att.filename, ctype, data))
        return images, skipped, files

    async def _resolve_outbound(self) -> discord.abc.Messageable:
        """Where unprompted messages go."""
        if self.channel_id is not None:
            channel = self.client.get_channel(self.channel_id)
            if channel is None:
                channel = await self.client.fetch_channel(self.channel_id)
            return channel
        if self._last_channel is not None:
            return self._last_channel
        user = await self.client.fetch_user(self.user_id)
        return user.dm_channel or await user.create_dm()


    async def describe_target(self) -> str:
        """Human-readable outbound destination, for logs and --where."""
        try:
            channel = await self._resolve_outbound()
        except Exception as exc:
            return f"unresolvable: {type(exc).__name__}: {exc}"
        guild = getattr(channel, "guild", None)
        if guild is not None:
            return f"#{channel.name} in {guild.name}"
        return f"DM with {self.user_id}"

    async def _deliver(self, channel, text: str) -> None:
        """Send one reply: text first, then anything it asked to attach."""
        body, attachments, problems = extract_send_files(text)
        if body or not (attachments or problems):
            for chunk in split_message(body):
                await channel.send(chunk)
        for path, rename in attachments:
            try:
                await channel.send(file=discord.File(str(path), filename=rename or path.name))
            except Exception as exc:
                problems.append(f"{path.name}: upload failed ({type(exc).__name__})")
        if problems:
            await channel.send("\n".join(f"⚠️ {p}" for p in problems))

    async def send(self, text: str) -> None:
        """Push an unprompted message. This is the wakeup's only voice."""
        await self._ready.wait()
        channel = await self._resolve_outbound()
        await self._deliver(channel, text)

    async def send_file(self, path: str | Path, *, caption: str | None = None) -> None:
        await self._ready.wait()
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(p)
        channel = await self._resolve_outbound()
        await channel.send(content=caption, file=discord.File(str(p), filename=p.name))

    async def start(self) -> None:
        """Run until cancelled. Co-runnable with a Textual app."""
        await self.client.start(self.token)

    async def close(self) -> None:
        if not self.client.is_closed():
            await self.client.close()

    async def wait_ready(self) -> None:
        await self._ready.wait()


# main

def _bridge(**kw) -> DiscordBridge:
    cfg = load_config()
    return DiscordBridge(
        cfg.token,
        cfg.user_id,
        channel_id=cfg.channel_id,
        require_mention=cfg.require_mention,
        **kw,
    )


async def _echo_mode() -> None:
    bridge = _bridge()
    print("Echo mode. DM or @mention the bot; ctrl-c to stop.", file=sys.stderr)
    try:
        await bridge.start()
    except asyncio.CancelledError:
        pass
    finally:
        await bridge.close()


async def _with_ready(fn) -> None:
    """Connect, run one coroutine against a ready bridge, disconnect."""
    bridge = _bridge()
    task = asyncio.create_task(bridge.start())
    try:
        await asyncio.wait_for(bridge.wait_ready(), timeout=30)
        await fn(bridge)
    finally:
        await bridge.close()
        task.cancel()


async def _send_mode(text: str) -> None:
    async def go(bridge: DiscordBridge) -> None:
        await bridge.send(text)
        print(f"sent to {await bridge.describe_target()}", file=sys.stderr)

    await _with_ready(go)


async def _where_mode() -> None:
    async def go(bridge: DiscordBridge) -> None:
        print(await bridge.describe_target())

    await _with_ready(go)


async def _send_file_mode(path: str, caption: str | None) -> None:
    async def go(bridge: DiscordBridge) -> None:
        await bridge.send_file(path, caption=caption)
        print(f"sent {path} to {await bridge.describe_target()}", file=sys.stderr)

    await _with_ready(go)


if __name__ == "__main__":
    args = sys.argv[1:]
    try:
        if args and args[0] == "--send":
            asyncio.run(_send_mode(" ".join(args[1:]) or "ping from Domicile"))
        elif args and args[0] == "--where":
            asyncio.run(_where_mode())
        elif args and args[0] == "--send-file":
            if len(args) < 2:
                raise SystemExit("usage: discord_bridge.py --send-file PATH [caption...]")
            asyncio.run(_send_file_mode(args[1], " ".join(args[2:]) or None))
        else:
            asyncio.run(_echo_mode())
    except KeyboardInterrupt:
        pass
