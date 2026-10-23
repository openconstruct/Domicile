"""Google tool wrappers: result text, approval gate, opt-in registration. Clients are mocked."""
import asyncio
import sys
import types
import unittest
from unittest.mock import patch

import google_tools as GT
import local_tools as LT

ALL = list(GT.ALL_TOOLS)


def run(name, args):
    return asyncio.run(LT.dispatch(name, args, ALL))


def fake_module(name, **fns):
    m = types.ModuleType(name)
    for k, v in fns.items():
        setattr(m, k, v)
    return m


class Registration(unittest.TestCase):
    def test_not_in_default_tiers(self):
        for tier in ("readonly", "write", "sandbox", "wakeup", "full"):
            self.assertFalse(set(GT.ALL_TOOLS) & set(LT.resolve_phase(tier)), tier)

    def test_opt_in_via_list(self):
        self.assertEqual(LT.resolve_phase(["Read", "GmailSearch", "bogus"]), ["Read", "GmailSearch"])

    def test_schemas_have_required_and_names(self):
        names = [t["function"]["name"] for t in LT.schemas(ALL)]
        self.assertEqual(sorted(names), sorted(ALL))

    def test_write_tools_are_mutating(self):
        self.assertTrue(set(GT.WRITE_TOOLS) <= LT._MUTATING_TOOLS)
        self.assertFalse(set(GT.READ_TOOLS) & LT._MUTATING_TOOLS)

    def test_not_enabled_is_refused(self):
        out = asyncio.run(LT.dispatch("GmailSearch", {}, ["Read"]))
        self.assertIn("not enabled", out)


class ReadTools(unittest.TestCase):
    def setUp(self):
        GT.set_approver(None)

    def test_gmail_search_text(self):
        gm = fake_module("gmail_client", list_messages=lambda q, n: [
            {"id": "m1", "date": "Wed", "from": "a@x", "subject": "Hi", "snippet": "yo"}])
        with patch.dict(sys.modules, {"gmail_client": gm}):
            self.assertEqual(run("GmailSearch", {"query": "is:unread"}), "m1  Wed  a@x  Hi\n    yo")

    def test_gmail_search_empty(self):
        gm = fake_module("gmail_client", list_messages=lambda q, n: [])
        with patch.dict(sys.modules, {"gmail_client": gm}):
            self.assertEqual(run("GmailSearch", {}), "(no matches)")

    def test_gmail_read(self):
        gm = fake_module("gmail_client", get_message=lambda i: {
            "from": "a", "to": "b", "subject": "S", "date": "D", "body": "text"})
        with patch.dict(sys.modules, {"gmail_client": gm}):
            self.assertEqual(run("GmailRead", {"id": "m1"}), "From: a\nTo: b\nSubject: S\nDate: D\n\ntext")

    def test_gmail_read_requires_id(self):
        self.assertEqual(run("GmailRead", {}), "ERROR: ValueError: id is required")

    def test_calendar_list(self):
        cm = fake_module("calendar_client", list_upcoming=lambda n, c: [
            {"start": "2026-10-23T09:00", "summary": "Dentist", "location": "Main St"},
            {"start": "2026-10-24", "summary": "Trash", "location": ""}])
        with patch.dict(sys.modules, {"calendar_client": cm}):
            self.assertEqual(run("CalendarList", {}),
                             "2026-10-23T09:00  Dentist  [Main St]\n2026-10-24  Trash")

    def test_client_error_is_error_text(self):
        def boom(q, n):
            raise RuntimeError("token expired")
        gm = fake_module("gmail_client", list_messages=boom)
        with patch.dict(sys.modules, {"gmail_client": gm}):
            self.assertEqual(run("GmailSearch", {}), "ERROR: RuntimeError: token expired")


class WriteTools(unittest.TestCase):
    def tearDown(self):
        GT.set_approver(None)

    def test_refused_without_approver_and_client_not_called(self):
        called = []
        gm = fake_module("gmail_client", send_message=lambda *a: called.append(a) or {"id": "x"})
        with patch.dict(sys.modules, {"gmail_client": gm}):
            out = run("GmailSend", {"to": "a@x", "subject": "s", "body": "b"})
        self.assertTrue(out.startswith("DENIED: GmailSend needs your approval"), out)
        self.assertEqual(called, [])

    def test_declined(self):
        GT.set_approver(lambda t, s, a: False)
        gm = fake_module("gmail_client", send_message=lambda *a: {"id": "x"})
        with patch.dict(sys.modules, {"gmail_client": gm}):
            self.assertEqual(run("GmailSend", {"to": "a", "subject": "s", "body": "b"}),
                             "DENIED: GmailSend was not approved.")

    def test_approved_send(self):
        seen = []
        GT.set_approver(lambda t, s, a: seen.append((t, s)) or True)
        gm = fake_module("gmail_client", send_message=lambda to, sub, body, th: {"id": "abc"})
        with patch.dict(sys.modules, {"gmail_client": gm}):
            out = run("GmailSend", {"to": "a@x", "subject": "Hello", "body": "b"})
        self.assertEqual(out, "tool=GmailSend; to=a@x; id=abc")
        self.assertEqual(seen, [("GmailSend", "Send email to a@x: 'Hello'")])

    def test_blogger_defaults_to_draft_and_live_is_flagged(self):
        seen = []
        GT.set_approver(lambda t, s, a: seen.append(s) or True)
        calls = []
        bm = fake_module("blogger_client", get_blog_id=lambda: "B1",
                         create_post=lambda b, t, h, is_draft: calls.append(is_draft) or {"id": "p1"})
        with patch.dict(sys.modules, {"blogger_client": bm}):
            self.assertEqual(run("BloggerPost", {"title": "T", "content_html": "<p>x</p>"}),
                             "tool=BloggerPost; id=p1; status=draft")
            self.assertEqual(run("BloggerPost", {"title": "T", "content_html": "x", "draft": False}),
                             "tool=BloggerPost; id=p1; status=live")
        self.assertEqual(calls, [True, False])
        self.assertIn("PUBLISH LIVE", seen[1])

    def test_calendar_create(self):
        GT.set_approver(lambda t, s, a: True)
        cm = fake_module("calendar_client", create_event=lambda s, st, en, d, c: {"id": "e1"})
        with patch.dict(sys.modules, {"calendar_client": cm}):
            self.assertEqual(run("CalendarCreate", {"summary": "X", "start": "2026-10-25T10:00:00-04:00",
                                                    "end": "2026-10-25T11:00:00-04:00"}),
                             "tool=CalendarCreate; id=e1; start=2026-10-25T10:00:00-04:00")

    def test_chat_send_and_read(self):
        GT.set_approver(lambda t, s, a: True)
        cm = fake_module("chat_client", send_message=lambda sp, tx, th: {"name": "spaces/A/messages/1"},
                         list_messages=lambda sp, n: [{"createTime": "T", "sender_display": "Jerry",
                                                       "sender_id": "u", "sender_type": "HUMAN", "text": "hi"}])
        with patch.dict(sys.modules, {"chat_client": cm}):
            self.assertEqual(run("ChatSend", {"space": "spaces/A", "text": "yo"}),
                             "tool=ChatSend; space=spaces/A; name=spaces/A/messages/1")
            self.assertEqual(run("ChatRead", {"space": "spaces/A"}), "T  Jerry (HUMAN): hi")


class DraftTool(unittest.TestCase):
    """GmailDraft writes but delivers nothing, so it is deliberately ungated."""

    def tearDown(self):
        GT.set_approver(None)

    def test_no_approver_needed_and_nothing_sent(self):
        sent, drafted = [], []
        gm = fake_module("gmail_client",
                         send_message=lambda *a: sent.append(a) or {"id": "s"},
                         create_draft=lambda to, sub, body, th: drafted.append((to, sub, body, th))
                         or {"id": "d1"})
        with patch.dict(sys.modules, {"gmail_client": gm}):
            out = run("GmailDraft", {"to": "a@x", "subject": "Re: s", "body": "b"})
        self.assertEqual(out, "tool=GmailDraft; to=a@x; id=d1")
        self.assertEqual(drafted, [("a@x", "Re: s", "b", None)])
        self.assertEqual(sent, [])

    def test_missing_recipient_is_error_text(self):
        with patch.dict(sys.modules, {"gmail_client": fake_module("gmail_client")}):
            self.assertEqual(run("GmailDraft", {"subject": "s", "body": "b"}),
                             "ERROR: ValueError: to is required")


if __name__ == "__main__":
    unittest.main()
