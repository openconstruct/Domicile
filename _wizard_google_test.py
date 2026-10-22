"""Offline tests for the Google wizard logic: client file, sign-in, verify, Chat handshake."""
import json
import os
import stat
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import httplib2
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError

import paths
import wizard_google as wg

CLIENT = {"installed": {"client_id": "abc.apps.googleusercontent.com",
                        "client_secret": "s3cret",
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token"}}


def write(d, name, obj):
    p = Path(d) / name
    p.write_text(json.dumps(obj))
    return str(p)


def http_error(status, message, reason=""):
    body = {"error": {"code": status, "message": message, "errors": [{"reason": reason}]}}
    return HttpError(httplib2.Response({"status": status}), json.dumps(body).encode())


def creds(scopes, granted=None, refresh="r"):
    return Credentials(token="t", refresh_token=refresh, client_id="c", client_secret="s",
                       token_uri="https://oauth2.googleapis.com/token",
                       scopes=scopes, granted_scopes=granted if granted is not None else scopes)


class Req:
    def __init__(self, value):
        self.value = value

    def execute(self):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class FakeMessages:
    def __init__(self, by_space, post_sender="users/999", post_fails=False):
        self.by_space, self.post_sender, self.post_fails = by_space, post_sender, post_fails
        self.posted = []

    def list(self, parent, **kw):
        return Req({"messages": self.by_space.get(parent, [])})

    def create(self, parent, body, **kw):
        if self.post_fails:
            return Req(RuntimeError("blocked"))
        self.posted.append((parent, body))
        return Req({"name": parent + "/messages/g1", "sender": {"name": self.post_sender}})


class FakeChat:
    def __init__(self, spaces, by_space, **kw):
        self._spaces, self._messages = spaces, FakeMessages(by_space, **kw)
        self.list_filters = []

    def spaces(self):
        return self

    def list(self, **kw):
        self.list_filters.append(kw.get("filter"))
        if isinstance(self._spaces, Exception):
            return Req(self._spaces)
        return Req({"spaces": [{"name": n} for n in self._spaces]})

    def messages(self):
        return self._messages


def msg(text, sender="users/123", kind="HUMAN", name="m1", at="2026-10-17T22:00:00.123456Z"):
    return {"name": f"spaces/DM1/messages/{name}", "text": text, "createTime": at,
            "sender": {"name": sender, "displayName": "Alex", "type": kind}}


class ClientFileTests(unittest.TestCase):
    def test_accepts_desktop_client(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(wg.load_client_secret(write(d, "c.json", CLIENT)), CLIENT)

    def test_rejects_web_client_with_instructions(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(wg.GoogleSetupError, "Desktop app"):
                wg.load_client_secret(write(d, "c.json", {"web": CLIENT["installed"]}))

    def test_rejects_missing_empty_and_broken(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in ("", str(Path(d) / "nope.json")):
                with self.assertRaises(wg.GoogleSetupError):
                    wg.load_client_secret(bad)
            (Path(d) / "x.json").write_text("{not json")
            with self.assertRaises(wg.GoogleSetupError):
                wg.load_client_secret(str(Path(d) / "x.json"))
            with self.assertRaisesRegex(wg.GoogleSetupError, "client_secret"):
                wg.load_client_secret(write(d, "y.json", {"installed": {"client_id": "a"}}))


class ParseTests(unittest.TestCase):
    def test_forms_of_pasted_result(self):
        url = "http://localhost:8765/?state=ST&code=4%2F0AbC-def&scope=x"
        self.assertEqual(wg.parse_redirect(url, "ST"), "4/0AbC-def")
        self.assertEqual(wg.parse_redirect("state=ST&code=4%2Fxyz123", "ST"), "4/xyz123")
        self.assertEqual(wg.parse_redirect(" <" + url + "> ", "ST"), "4/0AbC-def")
        self.assertEqual(wg.parse_redirect("4/0AbCdefGHIjkl", "ST"), "4/0AbCdefGHIjkl")

    def test_refusals(self):
        for pasted, text in [("", "Paste"),
                             ("http://localhost:8765/?error=access_denied&state=ST", "declined"),
                             ("http://localhost:8765/?state=ST", "no code"),
                             ("http://localhost:8765/?state=OLD&code=4%2Fxx", "older"),
                             ("hello there", "doesn't look")]:
            with self.assertRaisesRegex(wg.GoogleSetupError, text):
                wg.parse_redirect(pasted, "ST")


class FakeFlow:
    def __init__(self, fail=None, refresh="r"):
        self.fail, self.refresh, self.fetched = fail, refresh, None

    def authorization_url(self, **kw):
        self.kw = kw
        return "https://accounts.google.com/auth?x=1", "STATE1"

    def fetch_token(self, code):
        if self.fail:
            raise Exception(self.fail)
        self.fetched = code

    @property
    def credentials(self):
        return creds(["a"], refresh=self.refresh)


class SignInTests(unittest.TestCase):
    def test_real_flow_builds_a_headless_safe_link(self):
        s = wg.SignIn(CLIENT, ["gmail", "calendar"])
        url = s.start()
        for want in ("client_id=abc.apps", "access_type=offline", "prompt=consent",
                     "code_challenge=", "redirect_uri=http%3A%2F%2Flocalhost%3A8765%2F",
                     "gmail.modify", "auth%2Fcalendar"):
            self.assertIn(want, url)
        self.assertNotIn("chat", url)
        self.assertNotEqual(s.start(), url)       # every press is a fresh attempt

    def test_scopes_for(self):
        self.assertEqual(len(wg.scopes_for(["gmail", "chat"])), 3)
        with self.assertRaises(wg.GoogleSetupError):
            wg.scopes_for([])
        with self.assertRaises(wg.GoogleSetupError):
            wg.scopes_for(["nope"])

    def test_finish_trades_code_for_credentials(self):
        flow = FakeFlow()
        s = wg.SignIn(CLIENT, ["gmail"], flow_factory=lambda *a: flow)
        with self.assertRaises(wg.GoogleSetupError):
            s.finish("x")                                   # before start
        s.start()
        got = s.finish("http://localhost:8765/?state=STATE1&code=4%2Fthecode")
        self.assertEqual(flow.fetched, "4/thecode")
        self.assertEqual(got.refresh_token, "r")
        self.assertNotIn("OAUTHLIB_RELAX_TOKEN_SCOPE", os.environ)   # restored

    def test_google_errors_are_translated(self):
        for raised, text in [("invalid_grant: Bad Request", "press Get sign-in link"),
                             ("invalid_client: Unauthorized", "doesn't recognise")]:
            s = wg.SignIn(CLIENT, ["gmail"], flow_factory=lambda *a, r=raised: FakeFlow(fail=r))
            s.start()
            with self.assertRaisesRegex(wg.GoogleSetupError, text):
                s.finish("state=STATE1&code=4%2Fabc")

    def test_no_refresh_token_is_refused(self):
        s = wg.SignIn(CLIENT, ["gmail"], flow_factory=lambda *a: FakeFlow(refresh=None))
        s.start()
        with self.assertRaisesRegex(wg.GoogleSetupError, "refresh token"):
            s.finish("state=STATE1&code=4%2Fabc")


class SaveTests(unittest.TestCase):
    def test_token_records_granted_not_requested_scopes(self):
        allscopes = wg.scopes_for(["gmail", "calendar"])
        c = creds(allscopes, granted=allscopes[:1])
        data = json.loads(wg.token_json(c, "agent@example.com"))
        self.assertEqual(data["scopes"], allscopes[:1])
        self.assertEqual(data["account"], "agent@example.com")
        self.assertEqual(data["refresh_token"], "r")

    def test_files_are_private_and_loadable_by_google_client(self):
        with tempfile.TemporaryDirectory() as d, patch.object(wg, "GOOGLE_DIR", Path(d) / "ga"):
            c = creds(wg.scopes_for(["gmail"]))
            path = wg.save_token(c, CLIENT, "a@x.com")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE((path.parent / "client_secret.json").stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
            back = wg.load_token()
            self.assertEqual(back.refresh_token, "r")
            self.assertEqual(list(back.scopes), wg.scopes_for(["gmail"]))
            self.assertEqual([p.name for p in path.parent.iterdir() if p.name.startswith(".tmp")], [])

    def test_load_token_without_signin(self):
        with tempfile.TemporaryDirectory() as d, patch.object(wg, "GOOGLE_DIR", Path(d)):
            with self.assertRaises(wg.GoogleSetupError):
                wg.load_token()


class VerifyTests(unittest.TestCase):
    def fake_build(self, outcomes):
        class Svc:
            def __init__(self, key):
                self.key = key

            def _go(self):
                out = outcomes[self.key]
                return Req(out)

            def users(self): return self
            def getProfile(self, **kw): return self._go()
            def calendarList(self): return self
            def blogs(self): return self
            def listByUser(self, **kw): return self._go()
            def spaces(self): return self
            def list(self, **kw): return self._go()
        return lambda name, version, c: Svc(name)

    def test_each_service_reports_separately(self):
        c = creds(wg.scopes_for(["gmail", "calendar", "blogger"]))
        build = self.fake_build({
            "gmail": {"emailAddress": "agent@x.com"},
            "calendar": {"items": []},
            "blogger": http_error(403, "Blogger API has not been used in project 1 before "
                                       "or it is disabled.", "accessNotConfigured")})
        out = wg.verify(c, ["gmail", "calendar", "blogger"], build=build)
        self.assertEqual([x.ok for x in out], [True, True, False])
        self.assertEqual(wg.account_of(out), "agent@x.com")
        self.assertIn("apis/library/blogger.googleapis.com", out[2].detail)
        self.assertIn("Enable", out[2].detail)

    def test_unticked_consent_box_is_reported_not_called(self):
        gm = wg.scopes_for(["gmail"])
        c = creds(gm + wg.scopes_for(["calendar"]), granted=gm)
        calls = []
        out = wg.verify(c, ["gmail", "calendar"],
                        build=lambda n, v, cr: calls.append(n) or self.fake_build(
                            {"gmail": {"emailAddress": "a@x.com"}})(n, v, cr))
        self.assertEqual([x.ok for x in out], [True, False])
        self.assertIn("wasn't granted", out[1].detail)
        self.assertEqual(calls, ["gmail"])

    def test_chat_failure_mentions_workspace(self):
        c = creds(wg.scopes_for(["chat"]))
        out = wg.verify(c, ["chat"], build=self.fake_build(
            {"chat": http_error(403, "The caller does not have permission", "forbidden")}))
        self.assertFalse(out[0].ok)
        self.assertIn("Workspace", out[0].detail)

    def test_non_http_failure_is_still_a_check(self):
        c = creds(wg.scopes_for(["gmail"]))
        out = wg.verify(c, ["gmail"], build=self.fake_build({"gmail": OSError("network down")}))
        self.assertFalse(out[0].ok)
        self.assertIn("network down", out[0].detail)


class ChatTests(unittest.TestCase):
    def wait(self, chat, **kw):
        t = [0.0]
        return wg.wait_for_chat_message(
            None, "domicile-ab12cd", build=lambda *a: chat, interval=5, timeout=kw.pop("timeout", 20),
            clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s), **kw)

    def test_finds_code_case_insensitively_and_uses_dm_filter(self):
        chat = FakeChat(["spaces/DM1"], {"spaces/DM1": [
            msg("hello"), msg("Domicile-AB12CD", sender="users/777", name="m2")]})
        link = self.wait(chat)
        self.assertEqual((link.space, link.sender), ("spaces/DM1", "users/777"))
        self.assertEqual(chat.list_filters[0], 'spaceType = "DIRECT_MESSAGE"')

    def test_ignores_other_people_and_bots(self):
        chat = FakeChat(["spaces/DM1"], {"spaces/DM1": [
            msg("hi there"), msg("domicile-ab12cd", kind="BOT", sender="users/1")]})
        with self.assertRaisesRegex(wg.GoogleSetupError, "No message containing"):
            self.wait(chat)

    def test_keeps_polling_until_it_arrives(self):
        chat = FakeChat(["spaces/DM1"], {})
        orig = chat._messages.list
        n = [0]

        def later(parent, **kw):
            n[0] += 1
            if n[0] == 3:
                chat._messages.by_space[parent] = [msg("domicile-ab12cd")]
            return orig(parent, **kw)
        chat._messages.list = later
        self.assertEqual(self.wait(chat).sender, "users/123")
        self.assertGreaterEqual(n[0], 3)

    def test_cancel_and_api_errors(self):
        import threading
        ev = threading.Event()
        ev.set()
        with self.assertRaisesRegex(wg.GoogleSetupError, "Cancelled"):
            self.wait(FakeChat([], {}), cancel=ev)
        bad = FakeChat(http_error(403, "Google Chat API has not been used in project 1 or it is disabled."), {})
        with self.assertRaisesRegex(wg.GoogleSetupError, "apis/library/chat"):
            self.wait(bad)

    def test_complete_chat_writes_config_state_and_greets(self):
        link = wg.ChatLink("spaces/DM1", "users/777", "Alex", "spaces/DM1/messages/m2",
                           "2026-10-17T22:01:02.5Z")
        with tempfile.TemporaryDirectory() as d, \
                patch.object(paths, "CHAT_CONFIG", Path(d) / "chat.json"), \
                patch.object(paths, "CHAT_STATE", Path(d) / "chat-state.json"):
            chat = FakeChat([], {})
            self.assertTrue(wg.complete_chat(None, link, build=lambda *a: chat))
            self.assertEqual(chat._messages.posted[0][0], "spaces/DM1")
            cfg = json.loads((Path(d) / "chat.json").read_text())
            self.assertEqual(cfg, {"space": "spaces/DM1", "authorized_sender": "users/777",
                                   "poll_interval": 35})
            state = json.loads((Path(d) / "chat-state.json").read_text())
            self.assertEqual(state["self_id"], "users/999")
            self.assertEqual(state["last_seen_names"], ["spaces/DM1/messages/m2"])

    def test_failed_greeting_still_saves(self):
        link = wg.ChatLink("spaces/DM1", "users/777", "Alex", "spaces/DM1/messages/m2", "2026-10-17T22:01:02Z")
        with tempfile.TemporaryDirectory() as d, \
                patch.object(paths, "CHAT_CONFIG", Path(d) / "chat.json"), \
                patch.object(paths, "CHAT_STATE", Path(d) / "chat-state.json"):
            chat = FakeChat([], {}, post_fails=True)
            self.assertFalse(wg.complete_chat(None, link, build=lambda *a: chat))
            self.assertTrue((Path(d) / "chat.json").exists())
            self.assertIsNone(json.loads((Path(d) / "chat-state.json").read_text())["self_id"])

    def test_saved_files_match_what_the_real_bridge_reads(self):
        import chat_bridge
        link = wg.ChatLink("spaces/DM1", "users/777", "Alex", "spaces/DM1/messages/m2",
                           "2026-10-17T22:01:02.5Z")
        with tempfile.TemporaryDirectory() as d, \
                patch.object(paths, "CHAT_CONFIG", Path(d) / "chat.json"), \
                patch.object(paths, "CHAT_STATE", Path(d) / "chat-state.json"), \
                patch.object(chat_bridge, "CONFIG_PATH", Path(d) / "chat.json"), \
                patch.object(chat_bridge, "STATE_PATH", Path(d) / "chat-state.json"):
            wg.save_chat(link, "users/999")
            cfg = chat_bridge.load_config()
            self.assertEqual((cfg.space, cfg.authorized_sender, cfg.poll_interval),
                             ("spaces/DM1", "users/777", 35.0))
            bridge = chat_bridge.ChatBridge(cfg.space, cfg.authorized_sender, poll_interval=1)
            handshake = {"name": link.message_name, "createTime": link.create_time}
            self.assertTrue(bridge._seen(handshake))        # not replayed to the model
            later = {"name": "spaces/DM1/messages/m3", "createTime": "2026-10-17T22:05:00Z"}
            self.assertFalse(bridge._seen(later))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.d = Path(self.t.name)
        self.p = patch.object(wg, "GOOGLE_DIR", self.d / "g")
        self.p.start()
        self.out = []

    def tearDown(self):
        self.p.stop()
        self.t.cleanup()

    def say(self, text=""):
        self.out.append(str(text))

    def text(self):
        return "\n".join(self.out)

    def saved(self, services, account="old@x.com"):
        c = creds(wg.scopes_for(services))
        wg.save_token(c, CLIENT, account)

    def good_build(self):
        return VerifyTests().fake_build({
            "gmail": {"emailAddress": "new@x.com"}, "calendar": {"items": []},
            "blogger": {}, "chat": {}})

    def test_services_come_from_granted_scopes_not_requested(self):
        self.saved(["gmail", "chat"])
        self.assertEqual(wg.services_in_token(), ["gmail", "chat"])
        both = wg.scopes_for(["gmail", "calendar"])
        wg.save_token(creds(both, granted=both[:1]), None)
        self.assertEqual(wg.services_in_token(), ["gmail"])

    def test_check_passes_fails_and_reports_missing(self):
        self.assertEqual(wg.cmd_check(self.say), 1)
        self.assertIn("No saved Google sign-in", self.text())
        self.saved(["gmail", "calendar"])
        with patch.object(wg, "load_token", lambda: creds(wg.scopes_for(["gmail", "calendar"]))):
            self.assertEqual(wg.cmd_check(self.say, build=self.good_build()), 0)
            self.out.clear()
            bad = VerifyTests().fake_build({"gmail": {"emailAddress": "a@x.com"},
                                            "calendar": http_error(403, "Calendar API has not been used")})
            self.assertEqual(wg.cmd_check(self.say, build=bad), 1)
        self.assertIn("FAIL Calendar", self.text())

    def test_dead_token_gets_plain_words_and_the_fix(self):
        from google.auth.exceptions import RefreshError
        self.saved(["gmail"])
        dead = VerifyTests().fake_build({"gmail": RefreshError("invalid_grant")})
        with patch.object(wg, "load_token", lambda: creds(wg.scopes_for(["gmail"]))):
            self.assertEqual(wg.cmd_check(self.say, build=dead), 1)
        self.assertIn("no longer accepts", self.text())
        self.assertIn("wizard_google.py signin", self.text())

    def run_signin(self, answers, flow=None, **kw):
        it = iter(answers)
        return wg.cmd_signin(ask=lambda prompt="": next(it), say=self.say,
                             flow_factory=(lambda *a: flow) if flow else None, **kw)

    def test_signin_reuses_services_client_and_keeps_old_token_on_failure(self):
        self.saved(["gmail", "calendar"])
        old = (wg.GOOGLE_DIR / "token.json").read_text()
        self.assertEqual(self.run_signin(["http://localhost:8765/?state=STATE1&code=4%2Fx"],
                                         flow=FakeFlow(fail="invalid_grant: Bad")), 1)
        self.assertIn("Google rejected the code", self.text())
        self.assertEqual((wg.GOOGLE_DIR / "token.json").read_text(), old)   # untouched

    def test_signin_success_saves_updates_account_and_is_private(self):
        self.saved(["gmail", "calendar"], account="")
        flow = FakeFlow()
        scopes = wg.scopes_for(["gmail", "calendar"])
        with patch.object(FakeFlow, "credentials", property(lambda s: creds(scopes))):
            rc = self.run_signin(["http://localhost:8765/?state=STATE1&code=4%2Fx"], flow=flow,
                                 build=self.good_build())
        self.assertEqual(rc, 0, self.text())
        self.assertEqual(flow.fetched, "4/x")
        tok = wg.GOOGLE_DIR / "token.json"
        self.assertEqual(json.loads(tok.read_text())["account"], "new@x.com")
        self.assertEqual(stat.S_IMODE(tok.stat().st_mode), 0o600)
        self.assertIn("No restart needed", self.text())

    def test_signin_needs_services_and_client_and_survives_ctrl_c(self):
        self.assertEqual(self.run_signin([], client_path=""), 1)          # no token, no services
        self.assertIn("--services", self.text())
        self.out.clear()
        self.assertEqual(self.run_signin([str(self.d / "missing.json")], services=["gmail"]), 1)
        self.assertIn("No file at", self.text())
        self.out.clear()
        c = write(self.d, "c.json", CLIENT)

        def interrupt(prompt=""):
            raise KeyboardInterrupt
        self.saved(["gmail"])
        old = (wg.GOOGLE_DIR / "token.json").read_text()
        self.assertEqual(wg.cmd_signin(client_path=c, ask=interrupt, say=self.say,
                                       flow_factory=lambda *a: FakeFlow()), 1)
        self.assertIn("Nothing was changed", self.text())
        self.assertEqual((wg.GOOGLE_DIR / "token.json").read_text(), old)

    def test_signin_needs_only_token_json(self):
        self.saved(["gmail"])
        (wg.GOOGLE_DIR / "client_secret.json").unlink()
        tok = json.loads((wg.GOOGLE_DIR / "token.json").read_text())
        tok.update(client_id="tok-id", client_secret="tok-secret")
        (wg.GOOGLE_DIR / "token.json").write_text(json.dumps(tok))
        self.assertEqual(wg.client_from_token()["installed"]["client_id"], "tok-id")
        rc = self.run_signin([""])      # never asked for a path: the first prompt is the paste
        self.assertEqual(rc, 1)
        self.assertIn("accounts.google.com", self.text())
        self.assertIn("client_id=tok-id", self.text())
        self.assertNotIn("Path to the client_secret", self.text())

    def test_half_empty_client_file_is_an_error_not_a_traceback(self):
        c = write(self.d, "half.json", {"installed": {"client_id": "a", "client_secret": "b"}})
        self.assertEqual(self.run_signin([""], services=["gmail"], client_path=c), 1)
        self.assertIn("isn't in the format Google downloads", self.text())

    def test_real_flow_prints_a_usable_link(self):
        self.saved(["gmail"])
        (wg.GOOGLE_DIR / "client_secret.json").write_text(json.dumps(CLIENT))
        rc = self.run_signin([""])           # empty paste -> refused, but the link was shown
        self.assertEqual(rc, 1)
        self.assertIn("accounts.google.com", self.text())
        self.assertIn("Paste the address from the browser", self.text())


class MovedModulesTests(unittest.TestCase):
    """The google_auth/ code now lives inside domicile/ and keeps its state in data/google/."""

    def setUp(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent / "google_auth"))

    def test_paths_point_into_data_google(self):
        import google_client
        self.assertEqual(Path(google_client.TOKEN_FILE), paths.GOOGLE_TOKEN)
        self.assertEqual(paths.GOOGLE_TOKEN.parent, paths.CCHAT_DATA / "google")
        self.assertEqual(wg.GOOGLE_DIR, paths.GOOGLE_DIR)

    def test_wizard_written_token_is_what_google_client_loads_and_refreshes(self):
        import google_client
        with tempfile.TemporaryDirectory() as d, patch.object(wg, "GOOGLE_DIR", Path(d)), \
                patch.object(google_client, "TOKEN_FILE", str(Path(d) / "token.json")):
            fresh = creds(wg.scopes_for(["gmail"]))
            fresh.expiry = datetime.utcnow() + timedelta(hours=1)
            wg.save_token(fresh, None, "a@x.com")
            c = google_client._load_creds()
            self.assertEqual(list(c.scopes), wg.scopes_for(["gmail"]))    # not all five
            with patch.object(google_client.Credentials, "from_authorized_user_file",
                              return_value=type("C", (), {"expired": True, "refresh_token": "r",
                                  "refresh": lambda s, req: None, "to_json": lambda s: '{"a":1}'})()):
                google_client._load_creds()
            tok = Path(d) / "token.json"
            self.assertEqual(stat.S_IMODE(tok.stat().st_mode), 0o600)
            self.assertFalse((Path(d) / "token.json.tmp").exists())

    def test_gmail_notify_uses_data_google_and_account_from_token(self):
        import gmail_notify
        self.assertEqual(gmail_notify.SEEN_FILE, paths.GMAIL_SEEN)
        self.assertEqual(gmail_notify.UNREAD_COUNT_FILE, paths.GMAIL_UNREAD)
        self.assertEqual(gmail_notify.BGTASK_DIR, paths.BGTASK_DIR)
        with tempfile.TemporaryDirectory() as d:
            tok = Path(d) / "token.json"
            tok.write_text(json.dumps({"account": "agent@example.com"}))
            with patch.object(paths, "GOOGLE_TOKEN", tok):
                digest = gmail_notify.format_digest([{"subject": "s", "from": "f", "date": "d",
                                                      "snippet": "x", "id": "1"}])
            self.assertIn("agent@example.com", digest)
            self.assertNotIn("sage@", digest)

    def test_no_personal_strings_in_moved_code(self):
        root = Path(__file__).resolve().parent / "google_auth"
        for f in root.glob("*.py"):
            text = f.read_text()
            for bad in ("example-domain", "somehandle", "marc", "HomeServer"):
                self.assertNotIn(bad, text, f"{f.name} still mentions {bad}")


if __name__ == "__main__":
    unittest.main()
