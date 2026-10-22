"""Codex native-tool lockdown: catalog nulling, fail-closed behavior, launch flags.

No model calls, no network (`codex debug models` is faked).
Run: python3 _test_runner.py _codex_lockdown_test.py
"""
import asyncio
import copy
import json
import os
import subprocess
import unittest
from unittest.mock import patch

os.environ.setdefault("CCHAT_NO_CRASH_HOOKS", "1")

import _test_runner
_test_runner.require_sandbox()

import providers.codex_cli as CC

RAW = {"models": [
    {"slug": "gpt-6-sol", "apply_patch_tool_type": "freeform",
     "multi_agent_version": "v2", "shell_type": "unified_exec", "tool_mode": "code_mode_only"},
    {"slug": "gpt-6-luna", "apply_patch_tool_type": "freeform",
     "multi_agent_version": "v1", "shell_type": "unified_exec"},
    {"slug": "gpt-5.5", "apply_patch_tool_type": "freeform", "multi_agent_version": None},
], "extra_top_level": 1}


def fake_run(raw=RAW, fail=None):
    def run(argv, **kwargs):
        if fail:
            raise fail
        assert argv[1:3] == ["debug", "models"], argv
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(raw), stderr="")
    return run


class LockdownTests(unittest.TestCase):
    def setUp(self):
        CC._catalog_memo.update(at=0.0, slugs=frozenset(), path=None)
        path = CC._catalog_path()
        if path.exists():
            path.unlink()
        self.addCleanup(lambda: path.exists() and path.unlink())

    def test_builder_nulls_both_fields_on_every_model_without_mutating_input(self):
        before = copy.deepcopy(RAW)
        out = CC.build_lockdown_catalog(RAW)
        self.assertEqual(RAW, before)
        for model in out["models"]:
            self.assertIsNone(model["apply_patch_tool_type"])
            self.assertIsNone(model["multi_agent_version"])
        self.assertEqual(out["extra_top_level"], 1)
        self.assertEqual(out["models"][0]["tool_mode"], "code_mode_only")  # rest preserved
        self.assertEqual([m["slug"] for m in out["models"]], ["gpt-6-sol", "gpt-6-luna", "gpt-5.5"])

    def test_builder_rejects_empty_or_malformed(self):
        for bad in ({}, {"models": []}, {"models": ["x"]}, {"models": [{"no": "slug"}]}, [], None):
            with self.assertRaises(CC.CodexLockdownError):
                CC.build_lockdown_catalog(bad)

    def test_options_carry_catalog_and_every_disable(self):
        with patch.object(CC.subprocess, "run", fake_run()):
            opts = CC.lockdown_options("gpt-6-sol")
        path = CC._catalog_path()
        self.assertEqual(opts[:2], ["-c", f"model_catalog_json={json.dumps(str(path))}"])
        disabled = [opts[i + 1] for i, o in enumerate(opts) if o == "--disable"]
        self.assertEqual(disabled, list(CC._NATIVE_DISABLES))
        for must in ("shell_tool", "browser_use", "computer_use", "multi_agent",
                     "multi_agent_v2", "apps", "goals", "image_generation",
                     "in_app_browser", "tool_suggest", "view_image", "sleep_tool"):
            self.assertIn(must, disabled)
        written = json.loads(path.read_text())
        self.assertTrue(CC._catalog_is_locked(written))

    def test_unknown_model_fails_closed(self):
        with patch.object(CC.subprocess, "run", fake_run()):
            with self.assertRaises(CC.CodexLockdownError) as ctx:
                CC.lockdown_options("gpt-future")
        self.assertIn("gpt-future", str(ctx.exception))

    def test_codex_failure_with_no_last_good_copy_fails_closed(self):
        with patch.object(CC.subprocess, "run", fake_run(fail=OSError("no codex"))):
            with self.assertRaises(CC.CodexLockdownError):
                CC.lockdown_catalog(force=True)

    def test_codex_failure_falls_back_to_last_good_locked_copy(self):
        with patch.object(CC.subprocess, "run", fake_run()):
            CC.lockdown_catalog(force=True)
        with patch.object(CC.subprocess, "run", fake_run(fail=OSError("offline"))):
            path, slugs = CC.lockdown_catalog(force=True)
        self.assertIn("gpt-6-luna", slugs)
        self.assertEqual(path, str(CC._catalog_path()))

    def test_tampered_last_good_copy_is_not_trusted(self):
        with patch.object(CC.subprocess, "run", fake_run()):
            CC.lockdown_catalog(force=True)
        path = CC._catalog_path()
        data = json.loads(path.read_text())
        data["models"][1]["apply_patch_tool_type"] = "freeform"  # someone re-enabled it
        path.write_text(json.dumps(data))
        with patch.object(CC.subprocess, "run", fake_run(fail=OSError("offline"))):
            with self.assertRaises(CC.CodexLockdownError):
                CC.lockdown_catalog(force=True)

    def test_memo_avoids_a_second_codex_call_within_ttl(self):
        calls = []
        base = fake_run()
        def counting(argv, **kw):
            calls.append(argv)
            return base(argv, **kw)
        with patch.object(CC.subprocess, "run", counting):
            CC.lockdown_catalog()
            CC.lockdown_catalog()
        self.assertEqual(len(calls), 1)

    def test_base_options_use_lockdown_and_keep_approve_for_me(self):
        provider = CC.CodexCLIProvider("codex", {})
        provider._phase = lambda chat: "chat"
        provider._system_prompt = lambda chat: "sys"
        with patch.object(CC.subprocess, "run", fake_run()):
            opts = provider._base_options({}, "gpt-6-luna")
        self.assertIn("--approve-for-me", opts)
        self.assertTrue(any(o.startswith("model_catalog_json=") for o in opts))
        self.assertEqual(opts.count("--disable"), len(CC._NATIVE_DISABLES))
        self.assertIn("web_search=\"disabled\"", opts)

    def test_image_options_never_use_the_variadic_short_flag(self):
        # -i is variadic and would swallow the prompt that follows it.
        self.assertEqual(CC._image_options([]), [])
        self.assertEqual(CC._image_options(["/a b/1.png", "/c.png"]),
                         ["--image=/a b/1.png", "--image=/c.png"])
        self.assertNotIn("-i", CC._image_options(["/x.png"]))

    def test_turn_refuses_to_start_when_lockdown_is_unavailable(self):
        provider = CC.CodexCLIProvider("codex", {})
        ran = []
        async def must_not_run(*a, **k):
            ran.append(1)
            return 0
        provider._run = must_not_run
        with patch.object(CC.subprocess, "run", fake_run(fail=OSError("no codex"))):
            text, status = asyncio.run(provider.astream_turn({"model": "gpt-6-luna"}, "hi"))
        self.assertEqual(status, "error")
        self.assertIn("lockdown unavailable", text)
        self.assertEqual(ran, [])

    def test_turn_refuses_a_model_missing_from_the_catalog(self):
        provider = CC.CodexCLIProvider("codex", {})
        ran = []
        async def must_not_run(*a, **k):
            ran.append(1)
            return 0
        provider._run = must_not_run
        with patch.object(CC.subprocess, "run", fake_run()):
            text, status = asyncio.run(provider.astream_turn({"model": "gpt-future"}, "hi"))
        self.assertEqual(status, "error")
        self.assertEqual(ran, [])


if __name__ == "__main__":
    unittest.main()
