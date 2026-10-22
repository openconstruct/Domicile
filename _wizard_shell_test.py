"""Wizard shell: drive the real screens with Textual's pilot, temp paths only.
Run: python3 _test_runner.py _wizard_shell_test.py"""
import asyncio, json, tempfile, unittest
from pathlib import Path
from unittest import mock

import config_loader as CL, credentials, import_common as IC, mem1
import wizard
import wizard_discord as dd
from textual.widgets import Input, RadioSet, Static


async def tap(pilot, sel):
    """Click then settle. Back-to-back clicks on one spot get swallowed."""
    await pilot.pause(0.3)
    await pilot.click(sel)
    await pilot.pause(0.3)


class T(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); self.d = Path(self.t.name)
        self.logs = self.d / "logs"
        self.ps = [mock.patch.object(CL, "CONFIG_PATH", self.d / "config.json"),
                   mock.patch.object(credentials, "ENV_PATH", self.d / ".env"),
                   mock.patch.object(IC, "MEM0_LOG_DIR", self.logs),
                   mock.patch.object(CL, "IDENTITY_PATH", self.d / "identity.json"),
                   mock.patch.object(mem1, "MEM1_DIR", self.d / "memory")]
        for p in self.ps: p.start()
    def tearDown(self):
        for p in self.ps: p.stop()
        self.t.cleanup()

    def run_async(self, coro): return asyncio.run(coro)

    def test_full_happy_path(self):
        (self.d / "n").mkdir(); (self.d / "n/a.md").write_text("a note")
        async def go():
            app = wizard.WizardApp()
            async with app.run_test() as pilot:
                await pilot.click("#next")                       # welcome
                await pilot.pause()
                await pilot.click("#next")                       # provider: anthropic default
                await pilot.pause()
                s = app.screen
                s.query_one("#window", Input).value = "200000"
                await tap(pilot, "#next")  # context
                s = app.screen
                self.assertEqual(type(s).__name__, "ImportStep")
                s.query_one("#src", RadioSet).query("RadioButton")[3].value = True
                s.query_one("#path", Input).value = str(self.d / "n")
                await tap(pilot, "#run")
                await tap(pilot, "#next")  # -> Persona
                self.assertEqual(type(app.screen).__name__, "PersonaStep")
                app.screen.query_one("#user-name", Input).value = "Test User"
                await tap(pilot, "#next")  # -> EditStep
                self.assertEqual(type(app.screen).__name__, "EditStep")
                await tap(pilot, "#next")  # -> Discord (optional)
                self.assertEqual(type(app.screen).__name__, "DiscordStep")
                await tap(pilot, "#next")  # empty token = skip
                self.assertEqual(type(app.screen).__name__, "GoogleStep")
                self.assertEqual(app.answers["discord"], "skipped")
                await tap(pilot, "#next")  # empty path = skip
                self.assertEqual(type(app.screen).__name__, "ServiceStep")
                self.assertEqual(app.answers["google"], "skipped")
                app.screen.query_one("#svc-kind", RadioSet).query("RadioButton")[2].value = True  # skip: never touch real systemd
                await tap(pilot, "#next")  # -> Backup
                self.assertEqual(type(app.screen).__name__, "BackupStep")
                await tap(pilot, "#next")  # skip backup -> Smoke
                self.assertEqual(type(app.screen).__name__, "SmokeStep")
                self.assertEqual(app.answers["backup"], "skipped")
                await tap(pilot, "#next")  # -> Done
                self.assertEqual(type(app.screen).__name__, "Done")
                await tap(pilot, "#next")
            return app
        app = self.run_async(go())
        cfg = json.loads((self.d / "config.json").read_text())
        self.assertEqual(cfg["active_provider"], "anthropic")
        self.assertEqual(cfg["context"]["compact_ceiling_tokens"], 100000)
        self.assertEqual(len(list(self.logs.glob("*-text.log"))), 1)
        self.assertEqual(app.answers["provider"], "anthropic")
        self.assertEqual(app.answers["persona"]["mode"], "grow")
        self.assertIn("first conversation", (self.d / "memory/identity.md").read_text())
        self.assertEqual(json.loads((self.d / "identity.json").read_text())["operator_name"], "Test User")

    def test_persona_can_adjust_mem1_budget(self):
        async def go():
            app = wizard.WizardApp()
            async with app.run_test() as pilot:
                await tap(pilot, "#next")
                await tap(pilot, "#next")
                app.screen.query_one("#window", Input).value = "200000"
                await tap(pilot, "#next")
                await tap(pilot, "#next")  # skip imports
                self.assertEqual(type(app.screen).__name__, "PersonaStep")
                await tap(pilot, "#adjust-memory")
                self.assertEqual(type(app.screen).__name__, "ContextStep")
                self.assertEqual(app.screen.query_one("#mem1", Input).value, "19000")
                app.screen.query_one("#mem1", Input).value = "25000"
                await tap(pilot, "#next")
                self.assertEqual(type(app.screen).__name__, "PersonaStep")
                self.assertEqual(json.loads((self.d / "config.json").read_text())
                                 ["context"]["mem1_total"], 25000)
                self.assertFalse((self.d / "identity.json").exists())
        self.run_async(go())

    def test_validation_blocks_advance(self):
        async def go():
            app = wizard.WizardApp()
            async with app.run_test() as pilot:
                await tap(pilot, "#next")
                s = app.screen
                s.query_one("#kind", RadioSet).query("RadioButton")[2].value = True
                await tap(pilot, "#next")   # api with nothing filled in
                self.assertEqual(type(app.screen).__name__, "ProviderStep")
                msg = str(app.screen.query_one("#msg", Static).render())
                self.assertIn("base_url", msg)
        self.run_async(go())
        self.assertFalse((self.d / "config.json").exists())

    def test_small_window_warns_then_accepts(self):
        async def go():
            app = wizard.WizardApp()
            async with app.run_test() as pilot:
                await tap(pilot, "#next")
                await tap(pilot, "#next")
                s = app.screen
                s.query_one("#window", Input).value = "40000"
                await tap(pilot, "#next")
                self.assertEqual(type(app.screen).__name__, "ContextStep")
                await tap(pilot, "#next")
                self.assertEqual(type(app.screen).__name__, "ImportStep")
        self.run_async(go())

    # --- Discord step -------------------------------------------------------
    TOKEN = "MTIzNDU2.GabcDE.fakefakefakefakefake"

    def _discord_app(self):
        app = wizard.WizardApp()
        return app

    def _to_discord(self, app, pilot):
        async def go():
            await pilot.pause(0.2)
            app.push_screen(wizard.DiscordStep())
            await pilot.pause(0.3)
        return go()

    def test_discord_check_shows_invite_and_errors_plainly(self):
        info = dd.BotInfo("Agent#0001", 424242, ["Home"], dd.invite_url(424242))
        async def go():
            app = self._discord_app()
            async with app.run_test(size=(80, 24)) as pilot:
                await self._to_discord(app, pilot)
                # both buttons and Continue sit inside the 80x24 screen
                for sel in ("#dc-check", "#dc-wait", "#next"):
                    r = app.screen.query_one(sel).region
                    self.assertTrue(r.bottom <= 24 and r.right <= 80, (sel, r))
                app.screen.query_one("#dc-token", Input).value = self.TOKEN
                with mock.patch.object(dd, "check_bot", mock.AsyncMock(return_value=info)):
                    await tap(pilot, "#dc-check"); await pilot.pause(0.3)
                status = str(app.screen.query_one("#dc-status", Static).render())
                self.assertIn("client_id=424242", status)
                self.assertIn("Home", status)
                with mock.patch.object(dd, "check_bot", mock.AsyncMock(
                        side_effect=dd.DiscordSetupError("Message Content Intent is off."))):
                    await tap(pilot, "#dc-check"); await pilot.pause(0.3)
                self.assertIn("Message Content Intent", str(app.screen.query_one("#msg", Static).render()))
        self.run_async(go())
        self.assertFalse((self.d / ".env").exists())     # checking saves nothing

    def test_discord_handshake_saves_and_continues(self):
        got = dd.DMResult(777, "alex#0", True)
        async def go():
            app = self._discord_app()
            async with app.run_test(size=(80, 24)) as pilot:
                await self._to_discord(app, pilot)
                app.screen.query_one("#dc-token", Input).value = self.TOKEN
                # token entered but no handshake yet: Continue refuses
                await tap(pilot, "#next")
                self.assertEqual(type(app.screen).__name__, "DiscordStep")
                self.assertIn("not connected", str(app.screen.query_one("#msg", Static).render()))
                # a failed wait saves nothing
                with mock.patch.object(dd, "wait_for_dm", mock.AsyncMock(
                        side_effect=dd.DiscordSetupError("No DM arrived."))):
                    await tap(pilot, "#dc-wait"); await pilot.pause(0.3)
                self.assertFalse((self.d / ".env").exists())
                with mock.patch.object(dd, "wait_for_dm", mock.AsyncMock(return_value=got)):
                    await tap(pilot, "#dc-wait"); await pilot.pause(0.3)
                self.assertIn("Connected", str(app.screen.query_one("#msg", Static).render()))
                await tap(pilot, "#next")
                self.assertEqual(type(app.screen).__name__, "GoogleStep")
            return app
        app = self.run_async(go())
        self.assertEqual(app.answers["discord"], {"user": "alex#0", "user_id": 777})
        import discord_bridge
        cfg = discord_bridge.load_config()
        self.assertEqual((cfg.token, cfg.user_id), (self.TOKEN, 777))

    def test_discord_editing_the_token_voids_the_handshake(self):
        got = dd.DMResult(777, "alex#0", True)
        async def go():
            app = self._discord_app()
            async with app.run_test(size=(80, 24)) as pilot:
                await self._to_discord(app, pilot)
                app.screen.query_one("#dc-token", Input).value = self.TOKEN
                with mock.patch.object(dd, "wait_for_dm", mock.AsyncMock(return_value=got)):
                    await tap(pilot, "#dc-wait"); await pilot.pause(0.3)
                app.screen.query_one("#dc-token", Input).value = self.TOKEN + "x"
                await pilot.pause(0.2)
                await tap(pilot, "#next")
                self.assertEqual(type(app.screen).__name__, "DiscordStep")
        self.run_async(go())

    def test_backup_screen_checks_before_saving(self):
        import wizard_backup as wb
        info = wb.GitHubInfo("alice", "domicile-backup", True, True,
                             "Private repository found.")
        async def go():
            app = wizard.WizardApp()
            async with app.run_test(size=(80, 24)) as pilot:
                app.push_screen(wizard.BackupStep())
                await pilot.pause(0.3)
                app.screen.query_one("#backup-repo", Input).value = "domicile-backup"
                app.screen.query_one("#backup-token", Input).value = "secret-token"
                await tap(pilot, "#next")
                self.assertEqual(type(app.screen).__name__, "BackupStep")
                self.assertFalse((self.d / ".env").exists())
                with mock.patch.object(wb, "check_github", return_value=info) as check:
                    await tap(pilot, "#backup-check")
                    await pilot.pause(0.3)
                check.assert_called_once_with("secret-token", "domicile-backup")
                self.assertFalse((self.d / ".env").exists())
                app.screen.query_one("#backup-token", Input).value = "changed"
                await pilot.pause(0.2)
                await tap(pilot, "#next")
                self.assertEqual(type(app.screen).__name__, "BackupStep")
                app.screen.query_one("#backup-token", Input).value = "secret-token"
                with mock.patch.object(wb, "check_github", return_value=info):
                    await tap(pilot, "#backup-check")
                    await pilot.pause(0.3)
                await tap(pilot, "#next")
                self.assertEqual(type(app.screen).__name__, "SmokeStep")
                self.assertIn("alice/domicile-backup", app.answers["backup"])
        self.run_async(go())
        self.assertEqual(credentials.get("GITHUB_TOKEN"), "secret-token")
        self.assertEqual(json.loads((self.d / "config.json").read_text())
                         ["backup"]["repo"], "domicile-backup")

    def test_backup_screen_rejects_public_repo(self):
        import wizard_backup as wb
        async def go():
            app = wizard.WizardApp()
            async with app.run_test() as pilot:
                app.push_screen(wizard.BackupStep())
                await pilot.pause(0.3)
                app.screen.query_one("#backup-repo", Input).value = "public"
                app.screen.query_one("#backup-token", Input).value = "secret-token"
                with mock.patch.object(wb, "check_github",
                                       side_effect=wb.BackupSetupError("Repository is public.")):
                    await tap(pilot, "#backup-check")
                    await pilot.pause(0.3)
                self.assertIn("public", str(app.screen.query_one("#msg", Static).render()))
                await tap(pilot, "#next")
                self.assertEqual(type(app.screen).__name__, "BackupStep")
        self.run_async(go())
        self.assertFalse((self.d / ".env").exists())

    # --- Google step --------------------------------------------------------
    def _google_app_ctx(self):
        import wizard_google as wg
        ga = self.d / "ga"
        client = self.d / "client.json"
        client.write_text(json.dumps({"installed": {"client_id": "abc", "client_secret": "s"}}))
        return wg, ga, client

    class FakeSignIn:
        def __init__(self, client, services):
            self.services = services
        def start(self):
            self.state = "S"
            return "https://accounts.google.com/o/oauth2/auth?client_id=abc&state=S"
        def finish(self, pasted):
            from _wizard_google_test import creds
            import wizard_google as wg
            if "bad" in pasted:
                raise wg.GoogleSetupError("Google rejected the code.")
            return creds(wg.scopes_for(self.services))

    def _checks(self, wg, services, ok=True):
        return [wg.Check(k, wg.SERVICES[k].label, ok,
                         "agent@example.com" if k == "gmail" and ok else ("ok" if ok else "API off"))
                for k in services]

    def test_google_layout_fits_80x24(self):
        async def go():
            app = wizard.WizardApp()
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause(0.2)
                app.push_screen(wizard.GoogleStep())
                await pilot.pause(0.3)
                for sel in ("#g-start", "#g-finish", "#g-recheck", "#g-chat-go", "#next"):
                    r = app.screen.query_one(sel).region
                    self.assertTrue(r.bottom <= 24 and r.right <= 80, (sel, r))
        self.run_async(go())

    def test_google_signin_verifies_saves_and_continues(self):
        wg, ga, client = self._google_app_ctx()
        async def go():
            app = wizard.WizardApp()
            with mock.patch.object(wg, "GOOGLE_DIR", ga), \
                 mock.patch.object(wg, "SignIn", self.FakeSignIn), \
                 mock.patch.object(wg, "verify", lambda c, sv, **k: self._checks(wg, sv)):
                async with app.run_test(size=(80, 24)) as pilot:
                    await pilot.pause(0.2)
                    app.push_screen(wizard.GoogleStep())
                    await pilot.pause(0.3)
                    s = app.screen
                    s.query_one("#g-client", Input).value = str(client)
                    await tap(pilot, "#next")                       # entered but not signed in
                    self.assertEqual(type(app.screen).__name__, "GoogleStep")
                    self.assertIn("haven't signed in", str(s.query_one("#msg", Static).render()))
                    await tap(pilot, "#g-start")
                    self.assertIn("accounts.google.com", str(s.query_one("#g-link", Static).render()))
                    s.query_one("#g-paste", Input).value = "bad"
                    await tap(pilot, "#g-finish"); await pilot.pause(0.3)
                    self.assertIn("rejected", str(s.query_one("#msg", Static).render()))
                    self.assertFalse((ga / "token.json").exists())  # a failed sign-in saves nothing
                    s.query_one("#g-paste", Input).value = "http://localhost:8765/?state=S&code=4%2Fx"
                    await tap(pilot, "#g-finish"); await pilot.pause(0.5)
                    self.assertIn("agent@example.com", str(s.query_one("#g-status", Static).render()))
                    await tap(pilot, "#next")
                    self.assertEqual(type(app.screen).__name__, "ServiceStep")
                return app
        app = self.run_async(go())
        self.assertEqual(app.answers["google"], {"account": "agent@example.com",
                                                 "services": ["calendar", "gmail"]})
        tok = json.loads((ga / "token.json").read_text())
        self.assertEqual(tok["account"], "agent@example.com")
        self.assertTrue((ga / "client_secret.json").exists())

    def test_google_failed_check_blocks_until_recheck_passes(self):
        wg, ga, client = self._google_app_ctx()
        state = {"ok": False}
        async def go():
            app = wizard.WizardApp()
            with mock.patch.object(wg, "GOOGLE_DIR", ga), \
                 mock.patch.object(wg, "SignIn", self.FakeSignIn), \
                 mock.patch.object(wg, "verify", lambda c, sv, **k: self._checks(wg, sv, state["ok"])):
                async with app.run_test(size=(80, 24)) as pilot:
                    await pilot.pause(0.2)
                    app.push_screen(wizard.GoogleStep())
                    await pilot.pause(0.3)
                    s = app.screen
                    s.query_one("#g-client", Input).value = str(client)
                    await tap(pilot, "#g-start")
                    s.query_one("#g-paste", Input).value = "code"
                    await tap(pilot, "#g-finish"); await pilot.pause(0.5)
                    self.assertIn("FAIL", str(s.query_one("#g-status", Static).render()))
                    self.assertTrue((ga / "token.json").exists())   # signed in, so no second sign-in needed
                    await tap(pilot, "#next")
                    self.assertEqual(type(app.screen).__name__, "GoogleStep")
                    state["ok"] = True
                    await tap(pilot, "#g-recheck"); await pilot.pause(0.5)
                    await tap(pilot, "#next")
                    self.assertEqual(type(app.screen).__name__, "ServiceStep")
        self.run_async(go())

    def test_google_chat_needs_its_handshake(self):
        from textual.widgets import Checkbox
        wg, ga, client = self._google_app_ctx()
        link = wg.ChatLink("spaces/DM1", "users/777", "Alex", "spaces/DM1/messages/m1", "2026-10-17T22:00:00Z")
        async def go():
            app = wizard.WizardApp()
            with mock.patch.object(wg, "GOOGLE_DIR", ga), \
                 mock.patch.object(wg, "SignIn", self.FakeSignIn), \
                 mock.patch.object(wg, "verify", lambda c, sv, **k: self._checks(wg, sv)), \
                 mock.patch.object(wg, "wait_for_chat_message", lambda *a, **k: link), \
                 mock.patch.object(wg, "complete_chat", lambda *a, **k: True):
                async with app.run_test(size=(80, 24)) as pilot:
                    await pilot.pause(0.2)
                    app.push_screen(wizard.GoogleStep())
                    await pilot.pause(0.3)
                    s = app.screen
                    s.query_one("#g-client", Input).value = str(client)
                    s.query_one("#g-chat", Checkbox).value = True
                    await tap(pilot, "#g-start")
                    s.query_one("#g-paste", Input).value = "code"
                    await tap(pilot, "#g-finish"); await pilot.pause(0.5)
                    await tap(pilot, "#next")
                    self.assertEqual(type(app.screen).__name__, "GoogleStep")
                    self.assertIn("Chat handshake", str(s.query_one("#msg", Static).render()))
                    await tap(pilot, "#g-chat-go"); await pilot.pause(0.5)
                    self.assertIn("Alex", str(s.query_one("#msg", Static).render()))
                    await tap(pilot, "#next")
                    self.assertEqual(type(app.screen).__name__, "ServiceStep")
                return app
        app = self.run_async(go())
        self.assertIn("chat", app.answers["google"]["services"])

if __name__ == "__main__": unittest.main()
