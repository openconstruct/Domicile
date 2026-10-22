"""Persona wizard logic tests. Run via _test_runner.py."""
import json
import tempfile
import unittest
from pathlib import Path

import wizard_persona as WP


class T(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.d = Path(self.t.name)
        self.mem = self.d / "memory"
        self.ident = self.d / "identity.json"
        self.cfg = {"context": {"mem1_total": 19000, "mem1_caps": {}}}

    def tearDown(self):
        self.t.cleanup()

    def apply(self, mode="grow", user="Ada", agent="", sources=None):
        return WP.apply(mode, user, agent, sources, mem_dir=self.mem,
                        identity_path=self.ident, cfg=self.cfg)

    def test_grow_creates_neutral_identity_and_all_skeletons(self):
        result = self.apply()
        self.assertEqual(result["agent_name"], "Agent")
        identity = json.loads(self.ident.read_text())
        self.assertEqual(identity["operator_name"], "Ada")
        self.assertEqual(identity["agent_name"], "Agent")
        self.assertIn("choose one", (self.mem / "identity.md").read_text())
        self.assertEqual(sorted(p.name for p in self.mem.glob("*.md")), sorted(
            ["identity.md", "user.md", "practices.md", "goals.md", "open.md", "reflections.md",
             "chat.md", "code-building.md", "compact.md"]))

    def test_define_imports_document_and_preserves_source(self):
        src = self.d / "soul.md"
        src.write_text("# Soul\n\nCurious and stubborn.\n")
        self.apply("define", "Ada", "Sage", {"identity.md": str(src)})
        self.assertEqual(src.read_text(), "# Soul\n\nCurious and stubborn.\n")
        text = (self.mem / "identity.md").read_text()
        self.assertIn("Sage", text)
        self.assertIn("Curious and stubborn", text)

    def test_oversized_source_refused_before_any_write(self):
        src = self.d / "soul.md"
        src.write_text("too much\n" * 5000)
        with self.assertRaisesRegex(WP.PersonaError, "cap"):
            self.apply("define", "Ada", "Sage", {"identity.md": str(src)})
        self.assertFalse(self.ident.exists())
        self.assertFalse(self.mem.exists())

    def test_existing_persona_never_overwritten(self):
        self.mem.mkdir()
        path = self.mem / "identity.md"
        path.write_text("mine")
        with self.assertRaisesRegex(WP.PersonaError, "Refusing"):
            self.apply()
        self.assertEqual(path.read_text(), "mine")
        self.assertFalse(self.ident.exists())

    def test_existing_other_memory_preserved(self):
        self.mem.mkdir()
        path = self.mem / "goals.md"
        path.write_text("my goals")
        self.apply()
        self.assertEqual(path.read_text(), "my goals")

    def test_existing_identity_json_never_overwritten(self):
        self.ident.write_text('{"agent_name":"Mine"}')
        with self.assertRaisesRegex(WP.PersonaError, "Refusing"):
            self.apply()
        self.assertEqual(self.ident.read_text(), '{"agent_name":"Mine"}')

    def test_caps_use_new_config_not_mem1_import_state(self):
        self.cfg["context"] = {"mem1_total": 38000,
                               "mem1_caps": {"identity.md": 9000}}
        caps = WP.file_caps(self.cfg)
        self.assertEqual(caps["identity.md"], 9000)
        self.assertEqual(caps["user.md"], 7000)

    def test_role_files_are_created_personalised_and_within_caps(self):
        import mem1
        self.apply(user="Ada")
        for spec in mem1.ROLE_SPECS.values():
            text = (self.mem / spec.name).read_text()
            self.assertNotIn("{operator}", text, spec.name)
            self.assertIn("Ada", text, spec.name)
            for word in ("Walter", "Beatrix", "Casper"):
                self.assertNotIn(word, text, f"{spec.name} leaks {word}")
            self.assertNotRegex(text, r"\b(he|him|his)\b", f"{spec.name} assumes a gender")
            self.assertLessEqual(mem1.text_tokens(text), spec.hard, spec.name)

    def test_existing_role_file_is_preserved(self):
        self.mem.mkdir(parents=True)
        (self.mem / "chat.md").write_text("my own chat instructions\n")
        self.apply()
        self.assertEqual((self.mem / "chat.md").read_text(), "my own chat instructions\n")
        self.assertTrue((self.mem / "compact.md").exists())

    def test_missing_template_fails_before_writing_anything(self):
        from unittest.mock import patch
        with patch.object(WP, "TEMPLATE_DIR", self.d / "nowhere"):
            with self.assertRaisesRegex(WP.PersonaError, "Missing install template"):
                self.apply()
        self.assertFalse(self.ident.exists())
        self.assertEqual(list(self.mem.glob("*.md")) if self.mem.exists() else [], [])

    def test_fresh_install_hands_the_model_its_job_description(self):
        import mem1
        from unittest.mock import patch
        self.apply(user="Ada")
        with patch.object(mem1, "MEM1_DIR", self.mem):
            for role, marker in (("chat", "conversation, not a task queue"),
                                 ("build", "software engineer"),
                                 ("compact", "handover note")):
                block, _ = mem1.load(role=role)
                self.assertIn(marker, block, role)


if __name__ == "__main__":
    unittest.main()
