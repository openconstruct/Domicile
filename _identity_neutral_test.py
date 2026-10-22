"""A fresh install names nobody: identity defaults, bridge self-skip, no personal data in code.
Run via _test_runner.py."""
import getpass
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config_loader as CL

ROOT = Path(__file__).resolve().parent


class IdentityDefaults(unittest.TestCase):
    def load(self, payload):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "identity.json"
            if payload is not None:
                p.write_text(json.dumps(payload))
            with patch.object(CL, "IDENTITY_PATH", p):
                return CL.load_identity()

    def test_no_identity_file_names_nobody(self):
        with patch.object(getpass, "getuser", return_value="alex"):
            ident = self.load(None)
        blob = json.dumps(ident)
        for word in ("Marcus", "marcus", "marc", "Felix", "felix"):
            self.assertNotIn(word, blob)
        self.assertEqual(ident["operator_user"], "alex")         # from the shell user
        self.assertEqual(ident["mem0_user_id"], "alex")
        self.assertEqual(ident["operator_name"], "the user")
        self.assertEqual(ident["agent_name"], "the agent")

    def test_explicit_values_always_win_including_the_memory_scope_key(self):
        ident = self.load({"operator_user": "alex", "operator_name": "Alex",
                           "agent_name": "Sage", "mem0_user_id": "legacy-scope"})
        self.assertEqual(ident["mem0_user_id"], "legacy-scope")   # never re-derived
        self.assertEqual((ident["operator_name"], ident["agent_name"]), ("Alex", "Sage"))

    def test_partial_file_fills_only_the_gaps(self):
        ident = self.load({"operator_user": "alex", "operator_name": "Alex"})
        self.assertEqual(ident["mem0_user_id"], "alex")
        self.assertEqual(ident["agent_name"], "the agent")


class ChatBridgeSelfSkip(unittest.TestCase):
    def bridge(self, agent_name):
        import chat_bridge
        with patch.object(CL, "load_identity", return_value={"agent_name": agent_name}):
            return chat_bridge.ChatBridge("spaces/X", None, poll_interval=1, verbose=False)

    @staticmethod
    def msg(display, sender="users/1"):
        return {"sender_id": sender, "sender_display": display}

    def test_skips_the_configured_agent_name_and_not_a_hardcoded_one(self):
        b = self.bridge("Sage")
        self.assertFalse(b._is_inbound(self.msg("Sage")))
        self.assertTrue(b._is_inbound(self.msg("Felix")))     # just somebody named Felix now

    def test_known_self_id_is_always_skipped(self):
        b = self.bridge("Sage")
        b._self_id = "users/9"
        self.assertFalse(b._is_inbound(self.msg("Anything", "users/9")))


class ApiModelNames(unittest.TestCase):
    def test_slug(self):
        import api_server as API
        for name, want in (("Sage", "sage"), ("Dr. Who?", "dr-who"), ("the agent", "the-agent"),
                           ("", "agent"), ("???", "agent"), ("  Ada Lovelace ", "ada-lovelace")):
            self.assertEqual(API._model_slug(name), want, name)

    def test_models_follow_the_configured_agent_not_a_fixed_name(self):
        import api_server as API
        self.assertEqual(API.MODELS, (API.MODEL_LIVE, API.MODEL_FRESH))
        self.assertEqual(API.MODEL_FRESH, API.MODEL_LIVE + "-fresh")
        self.assertEqual(API.MODEL_LIVE, API._model_slug(CL.load_identity()["agent_name"]))

    def test_webui_has_no_fixed_model_or_agent_name(self):
        html = (ROOT / "webui" / "index.html").read_text()
        self.assertNotRegex(html, r'(?i)\brook\b')
        self.assertIn("d.model", html)             # model id comes from /api/identity


class NoPersonalDataInCode(unittest.TestCase):
    def test_every_banned_term_is_actually_caught(self):
        """The pattern itself: each alternative fires, and ordinary words do not."""
        for hit in ("Margaret", "Bubba", "insomnia", "the nephews", "Marcus said", "/home/marc",
                    "Felix", "FelixNest", "Felixback", "felix_tools", "Orion bridge", "x@example-domain.net",
                    "seekandfind", "someorg/oldname", "HostVendor", "Pico", "Shade", "Zentara", "Laptopy"):
            self.assertRegex(hit, self.BANNED, hit)
        for ok in ("a felixed bishop", "superorion", "marcuscan", "orions", "piccolo", "someorg/Domicile"):
            self.assertNotRegex(ok, self.BANNED, ok)

    """Shipped files (everything except tests and data/) name no real person or machine.

    The terms below are synthetic stand-ins with the same shapes as the real
    list. The production terms (the maintainer's actual names, machines,
    hosts and projects) live outside the shipped tree in
    banned_terms_local.txt -- one extra regex alternative per line,
    gitignored, never committed -- and are folded in below when present.
    """
    BANNED = re.compile(
        r"margaret|bubba|insomnia|rheumatoid|pension|\bnephews?\b"
        r"|\bmarcus\b|\bmarc\b|\bfelix(nest|back|_tools)?\b|\borion\b|example-domain|seekandfind"
        r"|someorg/oldname|hostvendor|\bpico\b|\bshade\b|zentara|\blaptopy\b", re.I)

    _local_terms_file = ROOT / "banned_terms_local.txt"
    if _local_terms_file.exists():
        _extra_terms = [ln.strip() for ln in _local_terms_file.read_text().splitlines()
                        if ln.strip() and not ln.startswith("#")]
        if _extra_terms:
            BANNED = re.compile(BANNED.pattern + "|" + "|".join(_extra_terms), re.I)

    def test_no_personal_names_or_infrastructure_in_shipped_code(self):
        hits = []
        for path in ROOT.rglob("*"):
            if (path.suffix not in {".py", ".sh", ".md", ".json", ".txt", ".html", ".service", ".timer"}
                    or "data" in path.relative_to(ROOT).parts[:1]
                    or path.name.startswith("_") and path.name.endswith("_test.py")):
                continue
            for n, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
                if self.BANNED.search(line):
                    hits.append(f"{path.relative_to(ROOT)}:{n}")
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
