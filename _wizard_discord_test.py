"""Discord connector logic against a scripted fake client (no network)."""
import asyncio, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import discord
import credentials
import wizard_discord as wd

TOKEN = "MTIzNDU2.GabcDE.fakefakefakefakefake"


class FakeClient:
    """Stands in for discord.Client. `mode` scripts what start() does."""
    def __init__(self, mode="ok", dms=(), guilds=("Home",), greet_fails=False):
        self.mode, self.dms, self.greet_fails = mode, list(dms), greet_fails
        self.handlers, self.closed = {}, False
        self.user = type("U", (), {"id": 424242, "__str__": lambda s: "Agent#0001"})()
        self.guilds = [NS(name=n) for n in guilds]
        self._stop = asyncio.Event()

    def event(self, fn):
        self.handlers[fn.__name__] = fn
        return fn

    async def start(self, token):
        if self.mode == "login":
            raise discord.LoginFailure("Improper token")
        if self.mode == "intents":
            raise discord.PrivilegedIntentsRequired(None)
        if self.mode == "never":
            await self._stop.wait(); return
        await self.handlers["on_ready"]()
        for m in self.dms:
            await asyncio.sleep(0.01)
            await self.handlers["on_message"](m)
        await self._stop.wait()

    async def close(self):
        self.closed = True
        self._stop.set()


def msg(uid=777, name="alex#0", bot=False, kind=discord.ChannelType.private, fail=False):
    sent = []
    async def send(text):
        if fail: raise RuntimeError("cannot send")
        sent.append(text)
    m = NS(author=type("A", (), {"id": uid, "bot": bot, "__str__": lambda s: name})(),
           channel=NS(type=kind, send=send), sent=sent)
    return m


def run(coro): return asyncio.run(coro)


class CleanToken(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(wd.clean_token(f'  "{TOKEN}" \n'), TOKEN)
        self.assertEqual(wd.clean_token(f"Bot {TOKEN}"), TOKEN)
    def test_rejects(self):
        for bad in ("", "   ", "123456789012345678", "abc def.ghi.jkl", "only.two"):
            with self.assertRaises(wd.DiscordSetupError, msg=bad):
                wd.clean_token(bad)


class Invite(unittest.TestCase):
    def test_url(self):
        u = wd.invite_url(424242)
        self.assertIn("client_id=424242", u)
        self.assertIn("scope=bot", u)
        self.assertIn(f"permissions={1024 + 2048 + 32768 + 65536}", u)


class CheckBot(unittest.TestCase):
    def test_ok(self):
        c = FakeClient(guilds=("Home", "Lab"))
        info = run(wd.check_bot(TOKEN, _factory=lambda: c))
        self.assertEqual((info.name, info.bot_id, info.guilds), ("Agent#0001", 424242, ["Home", "Lab"]))
        self.assertIn("client_id=424242", info.invite_url)
        self.assertTrue(c.closed)

    def test_bad_token(self):
        c = FakeClient("login")
        with self.assertRaisesRegex(wd.DiscordSetupError, "rejected that token"):
            run(wd.check_bot(TOKEN, _factory=lambda: c))
        self.assertTrue(c.closed)

    def test_intents_off(self):
        c = FakeClient("intents")
        with self.assertRaisesRegex(wd.DiscordSetupError, "Message Content Intent"):
            run(wd.check_bot(TOKEN, _factory=lambda: c))
        self.assertTrue(c.closed)

    def test_connect_timeout(self):
        c = FakeClient("never")
        with self.assertRaisesRegex(wd.DiscordSetupError, "No answer from Discord"):
            run(wd.check_bot(TOKEN, timeout=0.2, _factory=lambda: c))
        self.assertTrue(c.closed)

    def test_malformed_token_never_connects(self):
        made = []
        with self.assertRaises(wd.DiscordSetupError):
            run(wd.check_bot("nope", _factory=lambda: made.append(1)))
        self.assertEqual(made, [])


class WaitForDM(unittest.TestCase):
    def test_takes_first_human_dm_and_replies(self):
        human = msg(uid=777, name="alex#0")
        c = FakeClient(dms=[msg(uid=1, bot=True),                          # another bot
                            msg(uid=9, kind=discord.ChannelType.text),     # server message
                            human])
        seen = []
        r = run(wd.wait_for_dm(TOKEN, timeout=2, _factory=lambda: c, on_waiting=seen.append))
        self.assertEqual((r.user_id, r.username, r.reply_sent), (777, "alex#0", True))
        self.assertEqual(len(human.sent), 1)
        self.assertTrue(c.closed)
        self.assertIn("Agent#0001", seen[0])

    def test_server_message_is_not_the_handshake(self):
        c = FakeClient(dms=[msg(uid=9, kind=discord.ChannelType.text)])
        with self.assertRaisesRegex(wd.DiscordSetupError, "share a server"):
            run(wd.wait_for_dm(TOKEN, timeout=0.3, _factory=lambda: c))
        self.assertTrue(c.closed)

    def test_bot_dm_is_ignored(self):
        c = FakeClient(dms=[msg(uid=1, bot=True)])
        with self.assertRaises(wd.DiscordSetupError):
            run(wd.wait_for_dm(TOKEN, timeout=0.3, _factory=lambda: c))

    def test_failed_greeting_keeps_the_id(self):
        c = FakeClient(dms=[msg(uid=777, fail=True)])
        r = run(wd.wait_for_dm(TOKEN, timeout=2, _factory=lambda: c))
        self.assertEqual((r.user_id, r.reply_sent), (777, False))

    def test_no_dm_times_out_with_advice(self):
        c = FakeClient()
        with self.assertRaisesRegex(wd.DiscordSetupError, "invite link"):
            run(wd.wait_for_dm(TOKEN, timeout=0.3, _factory=lambda: c))
        self.assertTrue(c.closed)

    def test_intents_error_surfaces_here_too(self):
        c = FakeClient("intents")
        with self.assertRaisesRegex(wd.DiscordSetupError, "Message Content Intent"):
            run(wd.wait_for_dm(TOKEN, _factory=lambda: c))


class Save(unittest.TestCase):
    def test_round_trip_through_the_bridge_loader(self):
        import discord_bridge
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(credentials, "ENV_PATH", Path(d) / ".env"):
            wd.save(f"Bot {TOKEN}", 777)
            cfg = discord_bridge.load_config()
            self.assertEqual((cfg.token, cfg.user_id, cfg.channel_id), (TOKEN, 777, None))
            self.assertEqual((Path(d) / ".env").stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
