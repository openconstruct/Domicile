"""Wizard provider step: validation + writes, against temp paths only.

Run: python3 _test_runner.py _wizard_provider_test.py
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config_loader as CL
import context_diet
import credentials
import wizard_provider as wp


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.cfg, self.env = d / "config.json", d / ".env"
        self.patches = [mock.patch.object(CL, "CONFIG_PATH", self.cfg),
                        mock.patch.object(credentials, "ENV_PATH", self.env)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()


class BuildTest(unittest.TestCase):
    def test_anthropic_needs_no_key(self):
        r = wp.build_provider("anthropic")
        self.assertEqual(r["env"], {})
        self.assertEqual(r["block"]["adapter"], CL.CLI_ADAPTER)

    def test_api_requires_url_key_models(self):
        with self.assertRaises(wp.WizardError):
            wp.build_provider("api", "x", base_url="nope", models="m", api_key="k")
        with self.assertRaises(wp.WizardError):
            wp.build_provider("api", "x", base_url="https://h/v1", models="", api_key="k")
        with self.assertRaises(wp.WizardError):
            wp.build_provider("api", "x", base_url="https://h/v1", models="m", api_key=" ")

    def test_api_block_shape(self):
        r = wp.build_provider("api", "My-Host", base_url="https://h/v1/", models="m1",
                              api_key="sk-1")
        self.assertEqual(r["name"], "my-host")
        self.assertEqual(r["block"]["base_url"], "https://h/v1")
        self.assertEqual(r["block"]["key_env"], "MY_HOST_API_KEY")
        self.assertEqual(r["env"], {"MY_HOST_API_KEY": "sk-1"})
        self.assertEqual(r["block"]["models"]["code"], ["m1"])

    def test_openai_is_codex_login(self):
        r = wp.build_provider("openai")
        self.assertEqual(r["name"], "codex")
        self.assertEqual(r["block"]["adapter"], "codex_cli")
        self.assertEqual(r["env"], {})
        self.assertNotIn("models", r["block"])
        r = wp.build_provider("openai", models="gpt-x")
        self.assertEqual(r["block"]["models"]["code"], ["gpt-x"])

    def test_custom_context(self):
        import context_diet
        out = wp.custom_context(90_000, 12_000)
        self.assertEqual(out["context"], {"compact_ceiling_tokens": 90_000,
                                          "mem1_total": 12_000})
        self.assertNotIn("mem1_total", wp.custom_context(90_000)["context"])
        with self.assertRaises(wp.WizardError):
            wp.custom_context(context_diet.MIN_COMPACT_CEILING - 1)
        with self.assertRaises(wp.WizardError):
            wp.custom_context("abc")

    def test_custom_context_floor(self):
        import context_diet as cd
        out = wp.custom_context(90_000, None, 40_000)
        self.assertEqual(out["context"]["compact_floor_tokens"], 40_000)
        with self.assertRaises(wp.WizardError):   # floor under summary+min tail
            wp.custom_context(90_000, None, cd.MIN_COMPACT_FLOOR - 1)
        with self.assertRaises(wp.WizardError):   # ceiling must clear floor+headroom
            wp.custom_context(40_000 + cd.COMPACT_HEADROOM - 1, None, 40_000)
        wp.custom_context(40_000 + cd.COMPACT_HEADROOM, None, 40_000)  # exactly enough
        with self.assertRaises(wp.WizardError):
            wp.custom_context(90_000, None, "abc")

    def test_derived_floor_valid(self):
        import context_diet as cd
        for window in (32_000, 64_000, 200_000, 1_000_000):
            c = wp.derive_context(window)["context"]
            self.assertGreaterEqual(c["compact_floor_tokens"], cd.MIN_COMPACT_FLOOR)
            self.assertGreaterEqual(c["compact_ceiling_tokens"],
                                    cd.min_ceiling(c["compact_floor_tokens"]))
        self.assertEqual(wp.derive_context(200_000)["context"]["compact_floor_tokens"],
                         55_000)

    def test_bad_names(self):
        for bad in ("", "1abc", "has space", "a" * 40):
            with self.assertRaises(wp.WizardError):
                wp.clean_name(bad)


class ApplyTest(Base):
    def test_apply_writes_key_and_config(self):
        r = wp.build_provider("api", "host", base_url="https://h/v1", models="m",
                              api_key="sk-z")
        wp.apply_provider(r)
        self.assertEqual(credentials.get("HOST_API_KEY"), "sk-z")
        raw = json.loads(self.cfg.read_text())
        self.assertIn("host", raw["providers"])
        self.assertEqual(raw["provider_order"], ["host"])
        self.assertEqual(raw["active_provider"], "host")
        self.assertEqual(oct(self.env.stat().st_mode & 0o777), "0o600")

    def test_second_provider_appends_and_keeps_active(self):
        wp.apply_provider(wp.build_provider("anthropic"))
        wp.apply_provider(wp.build_provider("api", "b", base_url="https://h/v1",
                                            models="m", api_key="k"))
        raw = json.loads(self.cfg.read_text())
        self.assertEqual(raw["provider_order"], ["anthropic", "b"])
        self.assertEqual(raw["active_provider"], "anthropic")
        self.assertEqual(CL.provider_block("b")["key_env"], "B_API_KEY")


class ContextTest(Base):
    def test_big_window_capped(self):
        c = wp.derive_context(1_000_000)
        self.assertEqual(c["context"]["compact_ceiling_tokens"], wp.CEILING_CAP)
        self.assertEqual(c["context"]["mem1_total"], 19_000)
        self.assertTrue(c["fits"])

    def test_small_window_warns_and_respects_floor(self):
        c = wp.derive_context(32_000)
        self.assertGreaterEqual(c["context"]["compact_ceiling_tokens"],
                                context_diet.MIN_COMPACT_CEILING)
        self.assertFalse(c["fits"])
        self.assertTrue(c["warning"])

    def test_garbage_rejected(self):
        for bad in ("lots", 100, None):
            with self.assertRaises(wp.WizardError):
                wp.derive_context(bad)

    def test_apply_context_merges(self):
        CL.save_config({"hourly_ingest": False})
        wp.apply_context(200_000)
        raw = json.loads(self.cfg.read_text())
        self.assertEqual(raw["hourly_ingest"], False)
        self.assertEqual(raw["context"]["compact_ceiling_tokens"], 100_000)


if __name__ == "__main__":
    unittest.main()
