"""Thinking-level wiring: capability declaration, wire translation, and the
guards that stop the UIs from offering a level nobody will actually send.

No model calls. The live knob behaviour these mappings encode was measured
separately (2026-09-23): DashScope honours extra_body.thinking_budget exactly
(reasoning_tokens == budget at 128; a natural stop at 1273 under a 4096 cap)
and *silently ignores* reasoning_effort. opencode zen/go rejects unknown
params with a 400 naming the param, so acceptance there is meaningful, and
reasoning_effort visibly moves output length.

Run under _test_runner.py, which points Path.home() at a temp dir -- the
fixture config written below is disposable by design and is not restored.
"""
import json
import os
import unittest
import unittest.mock
from pathlib import Path

import config_loader as CL
import cchat
import providers.openai_compatible as ORP
import providers.codex_cli as CC

MODELS = {"chat": ["test"], "code": ["test"], "compact": ["test"]}
OAI = {"adapter": "openai_compatible", "key_env": "FIXTURE_API_KEY",
       "base_url": "http://127.0.0.1:1/v1", "models": MODELS}

FIXTURE = {
    "active_provider": "ali",
    "provider_order": ["anthropic", "opencode", "ali", "bare", "liar", "partial", "codex"],
    "providers": {
        # claude_cli path: the level goes straight to ClaudeAgentOptions.effort
        "anthropic": {"adapter": "claude_cli", "thinking": {"kind": "effort"},
                      "models": MODELS},
        # zen/go: named effort levels on the OpenAI wire
        "opencode": dict(OAI, thinking={"kind": "reasoning_effort"}),
        # DashScope/Qwen: a numeric token cap in extra_body
        "ali": dict(OAI, thinking={"kind": "thinking_budget",
                                   "levels": {"low": 1024, "medium": 8192,
                                              "high": 16384}}),
        # no capability at all -- the UIs must render nothing, not three stops
        "bare": dict(OAI),
        # declares a knob its adapter has no sender for
        "liar": dict(OAI, thinking={"kind": "effort"}),
        # maps only some of the levels
        "partial": dict(OAI, thinking={"kind": "thinking_budget",
                                       "levels": {"low": 1024}}),
        # codex exec: -c model_reasoning_effort=<level>, sent by the adapter
        "codex": {"adapter": "codex_cli", "models": MODELS,
                  "thinking": {"kind": "codex_effort",
                               "levels": ["low", "medium", "high",
                                          "xhigh", "max"]}},
    },
}

# This write happens at import time, so "who ran me" decides what it lands on.
# On 2026-10-03 it was run directly (`python3 _thinking_levels_test.py`) and the
# FIXTURE below went into the live ~/domicile/data/config.json -- after which every
# Anthropic turn died with "there's an issue with the selected model (test)" and
# the cause was invisible, because a config that says `test` looks like a typo.
# Path.home() was the wrong anchor: unpatched, it *is* the live home. Use the
# sandbox root _test_runner exports, and refuse to load at all without it.
_SANDBOX = os.environ.get("DOMICILE_TEST_SANDBOX")
if not _SANDBOX:
    raise SystemExit(
        "refusing to write a fixture config outside the test sandbox -- that "
        "would overwrite the live config.\n"
        "  run it as: python3 _test_runner.py " + Path(__file__).name)

_CFG_PATH = Path(_SANDBOX) / "domicile" / "data" / "config.json"
_CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
_CFG_PATH.write_text(json.dumps(FIXTURE))


class CapabilityTests(unittest.TestCase):
    def test_levels_follow_the_declaration(self):
        for name in ("ali", "opencode", "anthropic"):
            self.assertEqual(CL.thinking_levels(name), ["low", "medium", "high"],
                             name)
            self.assertTrue(CL.supports_thinking(name), name)

    def test_no_declaration_means_no_capability(self):
        self.assertEqual(CL.thinking_levels("bare"), [])
        self.assertFalse(CL.supports_thinking("bare"))
        self.assertEqual(CL.thinking_param("bare", "high"), {})

    def test_kind_must_match_the_adapter(self):
        # "effort" is only sendable from the claude_cli path. Declaring it on an
        # openai_compatible provider must degrade to no-capability rather than
        # render a toggle that quietly does nothing.
        self.assertEqual(CL.thinking_levels("liar"), [])
        self.assertEqual(CL.thinking_block("liar"), {})
        self.assertEqual(CL.thinking_param("liar", "high"), {})

    def test_unknown_kind_means_no_capability(self):
        cfg = json.loads(json.dumps(FIXTURE))
        cfg["providers"]["bare"]["thinking"] = {"kind": "telepathy"}
        self.assertEqual(CL.thinking_levels("bare", cfg), [])

    def test_non_dict_declaration_means_no_capability(self):
        cfg = json.loads(json.dumps(FIXTURE))
        cfg["providers"]["bare"]["thinking"] = "high"
        self.assertEqual(CL.thinking_levels("bare", cfg), [])

    def test_partial_levels_offer_only_what_maps(self):
        self.assertEqual(CL.thinking_levels("partial"), ["low"])
        self.assertEqual(CL.thinking_param("partial", "low"),
                         {"extra_body": {"thinking_budget": 1024}})
        self.assertEqual(CL.thinking_param("partial", "high"), {})

    def test_thinking_budget_translation(self):
        self.assertEqual(CL.thinking_param("ali", "low"),
                         {"extra_body": {"thinking_budget": 1024}})
        self.assertEqual(CL.thinking_param("ali", "medium"),
                         {"extra_body": {"thinking_budget": 8192}})
        self.assertEqual(CL.thinking_param("ali", "high"),
                         {"extra_body": {"thinking_budget": 16384}})

    def test_enable_thinking_added_only_when_declared(self):
        cfg = json.loads(json.dumps(FIXTURE))
        cfg["providers"]["ali"]["thinking"]["enable_thinking"] = True
        self.assertEqual(CL.thinking_param("ali", "low", cfg),
                         {"extra_body": {"thinking_budget": 1024,
                                         "enable_thinking": True}})

    def test_reasoning_effort_translation(self):
        self.assertEqual(CL.thinking_param("opencode", "high"),
                         {"reasoning_effort": "high"})

    def test_effort_kind_is_not_sent_from_the_openai_path(self):
        # stream_claude handles that one; returning {} here is deliberate.
        self.assertEqual(CL.thinking_param("anthropic", "high"), {})

    def test_auto_and_garbage_send_nothing(self):
        for level in ("auto", "", None, "MAXIMUM", "xhigh", 3):
            self.assertEqual(CL.thinking_param("ali", level), {}, repr(level))
            self.assertEqual(CL.thinking_param("opencode", level), {}, repr(level))


class CodexEffortTests(unittest.TestCase):
    """codex_effort: wider level set than the default, sent by the adapter."""

    def test_codex_offers_its_declared_levels_in_order(self):
        self.assertEqual(CL.thinking_levels("codex"),
                         ["low", "medium", "high", "xhigh", "max"])

    def test_ultra_is_never_offered(self):
        # codex "ultra" = automatic task delegation; the lockdown removes that.
        self.assertNotIn("ultra", CL.ALL_THINKING_LEVELS)
        self.assertNotIn("ultra", CL.thinking_levels("codex"))

    def test_other_providers_keep_three_levels(self):
        # widening codex must not leak xhigh/max onto the other adapters
        for prov in ("anthropic", "opencode"):
            self.assertEqual(CL.thinking_levels(prov),
                             ["low", "medium", "high"], prov)

    def test_kind_is_dead_on_the_wrong_adapter(self):
        cfg = json.loads(json.dumps(FIXTURE))
        cfg["providers"]["opencode"]["thinking"] = {"kind": "codex_effort"}
        self.assertEqual(CL.thinking_levels("opencode", cfg), [])

    def test_list_form_filters_to_known_levels_in_order(self):
        cfg = json.loads(json.dumps(FIXTURE))
        cfg["providers"]["codex"]["thinking"]["levels"] = ["max", "ultra", "low"]
        self.assertEqual(CL.thinking_levels("codex", cfg), ["low", "max"])

    def test_param_is_empty_the_adapter_sends_it(self):
        self.assertEqual(CL.thinking_param("codex", "xhigh"), {})

    def test_cycle_for_codex(self):
        self.assertEqual(cchat.thinking_cycle_for("codex"),
                         ["auto", "low", "medium", "high", "xhigh", "max"])

    def test_effort_option_is_sent_for_an_offered_level(self):
        for lv in ("low", "xhigh", "max"):
            self.assertEqual(CC._effort_options({"thinking": lv}, "codex"),
                             ["-c", f'model_reasoning_effort="{lv}"'])

    def test_effort_option_absent_for_auto_stale_or_garbage(self):
        for chat in ({}, {"thinking": ""}, {"thinking": "ultra"},
                     {"thinking": "bogus"}, {"thinking": None}, None):
            self.assertEqual(CC._effort_options(chat, "codex"), [], repr(chat))

    def test_effort_option_absent_for_a_provider_without_the_capability(self):
        self.assertEqual(CC._effort_options({"thinking": "high"}, "bare"), [])


class CycleTests(unittest.TestCase):
    def test_cycle_puts_auto_first(self):
        self.assertEqual(cchat.thinking_cycle_for("ali"),
                         ["auto", "low", "medium", "high"])

    def test_cycle_is_empty_without_a_capability(self):
        for name in ("bare", "liar", "partial-nope"):
            self.assertEqual(cchat.thinking_cycle_for(name), [], name)

    def test_partial_cycle_offers_auto_plus_what_maps(self):
        self.assertEqual(cchat.thinking_cycle_for("partial"), ["auto", "low"])


class HudTests(unittest.TestCase):
    def test_hud_records_the_level_when_it_will_be_sent(self):
        hud = cchat.domicile_hud({"mode": "chat", "thinking": "high"}, "test-model")
        self.assertIn("think: high", hud)

    def test_hud_omits_auto(self):
        hud = cchat.domicile_hud({"mode": "chat"}, "test-model")
        self.assertNotIn("think:", hud)

    def test_hud_omits_a_level_the_active_provider_cannot_send(self):
        # The transcript must not claim a level that was silently dropped.
        with unittest.mock.patch.object(cchat, "provider_name",
                                        return_value="bare"):
            hud = cchat.domicile_hud({"mode": "chat", "thinking": "high"},
                                     "test-model")
        self.assertNotIn("think:", hud)


class WireTests(unittest.IsolatedAsyncioTestCase):
    """_one_call must put the translated knob on the actual request."""

    def _provider(self, name, thinking):
        p = object.__new__(ORP.OpenAICompatibleProvider)
        p.name = name
        p.model = "test"
        p.max_tokens = 64
        p._turn_thinking = thinking
        p._tool_schemas = lambda: []
        captured = {}

        class _Completions:
            async def create(self, **kw):
                captured.update(kw)
                raise RuntimeError("probe-stop")

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        p._client = _Client()
        return p, captured

    async def _kwargs_for(self, name, thinking):
        p, captured = self._provider(name, thinking)
        with self.assertRaises(RuntimeError):
            await p._one_call([], None, [])
        return captured

    async def test_thinking_budget_reaches_the_wire(self):
        kw = await self._kwargs_for("ali", "medium")
        self.assertEqual(kw.get("extra_body"), {"thinking_budget": 8192})

    async def test_reasoning_effort_reaches_the_wire(self):
        kw = await self._kwargs_for("opencode", "high")
        self.assertEqual(kw.get("reasoning_effort"), "high")

    async def test_auto_sends_no_knob(self):
        for name in ("ali", "opencode", "bare"):
            kw = await self._kwargs_for(name, "")
            self.assertNotIn("extra_body", kw, name)
            self.assertNotIn("reasoning_effort", kw, name)

    async def test_unmapped_level_sends_no_knob(self):
        kw = await self._kwargs_for("partial", "high")
        self.assertNotIn("extra_body", kw)

    async def test_base_request_shape_is_untouched(self):
        kw = await self._kwargs_for("ali", "low")
        self.assertEqual(kw["model"], "test")
        self.assertEqual(kw["max_tokens"], 64)
        self.assertTrue(kw["stream"])
        self.assertEqual(kw["stream_options"], {"include_usage": True})

    async def test_missing_turn_attribute_is_not_fatal(self):
        p, captured = self._provider("ali", None)
        del p._turn_thinking
        with self.assertRaises(RuntimeError):
            await p._one_call([], None, [])
        self.assertNotIn("extra_body", captured)


class DegradeTests(unittest.IsolatedAsyncioTestCase):
    """A gateway that rejects the knob must cost the level, not the turn.

    Capability is declared per provider; zen/go validates per upstream model.
    So a model swap can invalidate a mapping that was fine yesterday, and the
    sender has to notice and back off instead of 400ing every turn.
    """

    ZEN_REJECT = ("Error code: 400 - {'error': {'param': 'reasoning_effort', "
                  "'type': 'invalid_request_error', 'message': 'Upstream "
                  "request failed: [invalid_request_error] invalid'}}")
    CONTEXT_REJECT = ("Error code: 400 - {'error': {'message': 'maximum context "
                      "length exceeded'}}")

    def _provider(self, name, thinking, fail_with):
        p = object.__new__(ORP.OpenAICompatibleProvider)
        p.name = name
        p.model = "test"
        p.max_tokens = 64
        p._turn_thinking = thinking
        p._tool_schemas = lambda: []
        calls = []

        class _Completions:
            async def create(self, **kw):
                calls.append(kw)
                msg = fail_with(kw, len(calls))
                if msg:
                    raise Exception(msg)
                raise RuntimeError("probe-stop")

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        p._client = _Client()
        return p, calls

    def test_rejection_detector_is_narrow(self):
        self.assertTrue(ORP._is_thinking_param_rejection(Exception(self.ZEN_REJECT)))
        self.assertTrue(ORP._is_thinking_param_rejection(
            Exception("400 {'param': 'thinking_budget'}")))
        self.assertTrue(ORP._is_thinking_param_rejection(
            Exception("400 {'param': 'enable_thinking'}")))
        self.assertFalse(ORP._is_thinking_param_rejection(
            Exception(self.CONTEXT_REJECT)))
        self.assertFalse(ORP._is_thinking_param_rejection(
            Exception("401 invalid api key")))
        self.assertFalse(ORP._is_thinking_param_rejection(
            Exception("500 upstream exploded")))

    async def test_rejected_knob_retries_once_without_it(self):
        p, calls = self._provider(
            "opencode", "high",
            lambda kw, n: self.ZEN_REJECT if "reasoning_effort" in kw else None)
        with self.assertRaises(RuntimeError):
            await p._one_call([], None, [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["reasoning_effort"], "high")
        self.assertNotIn("reasoning_effort", calls[1])
        self.assertEqual(calls[1]["model"], "test")   # same request otherwise

    async def test_unrelated_400_is_not_retried(self):
        p, calls = self._provider("opencode", "high",
                                  lambda kw, n: self.CONTEXT_REJECT)
        with self.assertRaises(Exception) as ctx:
            await p._one_call([], None, [])
        self.assertIn("context length", str(ctx.exception))
        self.assertEqual(len(calls), 1)   # no pointless second failure

    async def test_no_knob_means_no_retry_logic(self):
        p, calls = self._provider("bare", "",
                                  lambda kw, n: self.ZEN_REJECT)
        with self.assertRaises(Exception):
            await p._one_call([], None, [])
        self.assertEqual(len(calls), 1)

    async def test_rejection_is_logged(self):
        p, _calls = self._provider("opencode", "high",
                                   lambda kw, n: self.ZEN_REJECT
                                   if "reasoning_effort" in kw else None)
        with unittest.mock.patch.object(ORP, "_log") as log:
            with self.assertRaises(RuntimeError):
                await p._one_call([], None, [])
        self.assertTrue(any("thinking param rejected" in str(c.args[0])
                            for c in log.call_args_list), log.call_args_list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
