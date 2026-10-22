"""Discord connector setup: no UI here, so the TUI and a future web step share it.

Two live checks against Discord, each a short-lived gateway session:
  check_bot(token)    token works, Message Content Intent is on; returns the bot's
                      name, the servers it is in, and an invite link.
  wait_for_dm(token)  the human DMs the bot; their user id becomes the allowlist,
                      so nobody has to turn on developer mode and copy an id.
save() then writes the credentials through discord_bridge.save_config.

discord.py is imported lazily: a user who skips this connector never needs it.
"""
from __future__ import annotations

import asyncio
import contextlib
from typing import Callable, NamedTuple

DEVELOPER_PORTAL = "https://discord.com/developers/applications"
# View Channels + Send Messages + Attach Files + Read Message History
INVITE_PERMISSIONS = 1024 + 2048 + 32768 + 65536


class DiscordSetupError(Exception):
    """A setup problem, worded for the person running the wizard."""


class BotInfo(NamedTuple):
    name: str
    bot_id: int
    guilds: list
    invite_url: str


class DMResult(NamedTuple):
    user_id: int
    username: str
    reply_sent: bool


def invite_url(client_id: int) -> str:
    return (f"https://discord.com/oauth2/authorize?client_id={client_id}"
            f"&scope=bot&permissions={INVITE_PERMISSIONS}")


def _discord():
    try:
        import discord
        return discord
    except ImportError:
        raise DiscordSetupError(
            "discord.py isn't installed. Run: pip install discord.py   then try again.")


def _intents(discord):
    # Same set the running bridge asks for, so a pass here means a pass there.
    intents = discord.Intents.default()
    intents.message_content = True
    intents.dm_messages = True
    intents.guild_messages = True
    return intents


def _translate(exc: BaseException) -> DiscordSetupError:
    discord = _discord()
    if isinstance(exc, discord.LoginFailure):
        return DiscordSetupError(
            "Discord rejected that token. In the Developer Portal open your app, "
            "Bot, Reset Token, and copy the new one (it is only shown once).")
    if isinstance(exc, discord.PrivilegedIntentsRequired):
        return DiscordSetupError(
            "Message Content Intent is off. In the Developer Portal open your app, "
            "Bot, scroll to Privileged Gateway Intents, switch on Message Content "
            "Intent, press Save Changes, then try again.")
    return DiscordSetupError(f"Couldn't connect to Discord: {type(exc).__name__}: {exc}")


def clean_token(raw: str) -> str:
    token = (raw or "").strip().strip('"').strip("'")
    if token.lower().startswith("bot "):
        token = token[4:].strip()
    if not token:
        raise DiscordSetupError("Paste the bot token first.")
    if token.count(".") != 2 or " " in token:
        raise DiscordSetupError(
            "That doesn't look like a bot token (three parts joined by dots). "
            "Use Bot, Reset Token, not the Application ID or the public key.")
    return token


@contextlib.asynccontextmanager
async def _session(token: str, on_message=None, timeout: float = 30.0, _factory=None):
    """A connected client, always closed on the way out."""
    discord = _discord()
    client = _factory() if _factory else discord.Client(intents=_intents(discord))
    ready = asyncio.Event()

    @client.event
    async def on_ready():
        ready.set()

    if on_message is not None:
        client.event(on_message)

    runner = asyncio.ensure_future(client.start(token))
    waiter = asyncio.ensure_future(ready.wait())
    try:
        await asyncio.wait({runner, waiter}, timeout=timeout,
                           return_when=asyncio.FIRST_COMPLETED)
        if runner.done() and not ready.is_set():
            exc = runner.exception()
            raise _translate(exc) if exc else DiscordSetupError(
                "Discord closed the connection before the bot was ready.")
        if not ready.is_set():
            raise DiscordSetupError(
                f"No answer from Discord after {timeout:.0f}s. Check the network and try again.")
        yield client
    finally:
        waiter.cancel()
        with contextlib.suppress(Exception):
            await client.close()
        runner.cancel()
        with contextlib.suppress(BaseException):
            await runner


async def check_bot(token: str, *, timeout: float = 30.0, _factory=None) -> BotInfo:
    token = clean_token(token)
    async with _session(token, timeout=timeout, _factory=_factory) as client:
        user = client.user
        return BotInfo(name=str(user), bot_id=user.id,
                       guilds=[g.name for g in client.guilds],
                       invite_url=invite_url(user.id))


async def wait_for_dm(token: str, *, timeout: float = 180.0, connect_timeout: float = 30.0,
                      on_waiting: Callable[[str], None] | None = None,
                      _factory=None) -> DMResult:
    """Connect, then wait for the first DM from a human. Their id is the answer."""
    token = clean_token(token)
    discord = _discord()
    loop = asyncio.get_running_loop()
    got: asyncio.Future = loop.create_future()

    async def on_message(message):
        if getattr(message.author, "bot", False) or got.done():
            return
        if getattr(message.channel, "type", None) != discord.ChannelType.private:
            return          # a server message is not the handshake; ask for a DM
        got.set_result(message)

    async with _session(token, on_message=on_message, timeout=connect_timeout,
                        _factory=_factory) as client:
        if on_waiting:
            on_waiting(f"Connected as {client.user}. Waiting for your DM...")
        try:
            message = await asyncio.wait_for(got, timeout)
        except asyncio.TimeoutError:
            raise DiscordSetupError(
                f"No DM arrived in {timeout:.0f}s. The bot has to share a server with "
                "you before it can be messaged: use the invite link from Check bot, "
                "then open the bot's profile in that server and send it a direct message.")
        sent = True
        try:
            await message.channel.send(
                "Connected. Only you can talk to this agent from Discord now. "
                "Finish setup in the wizard, then start Domicile and say hello here.")
        except Exception:
            sent = False    # the id is already captured; a failed greeting is not a failed setup
        return DMResult(user_id=message.author.id,
                        username=str(message.author), reply_sent=sent)


def save(token: str, user_id: int) -> None:
    import discord_bridge
    discord_bridge.save_config(clean_token(token), int(user_id))
