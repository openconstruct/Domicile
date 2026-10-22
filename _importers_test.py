"""Importers: synthetic inputs, temp dirs only.
Run: python3 _test_runner.py _importers_test.py"""
import json, os, tempfile, unittest, zipfile
from pathlib import Path
import import_common as IC, import_claude as ICL, import_text as ITX, import_chatgpt as IGP

EXPORT = [{"name": "Chat A", "created_at": "2026-01-02T10:00:00Z", "chat_messages": [
    {"sender": "human", "text": "hi there", "created_at": "2026-01-02T10:00:01Z"},
    {"sender": "assistant", "text": "", "content": [{"type": "text", "text": "hello"}],
     "created_at": "2026-01-02T10:00:09Z"},
    {"sender": "human", "text": "", "created_at": "2026-01-03T09:00:00Z"}]},
    {"name": "", "chat_messages": [{"sender": "human", "text": "x" * 7000,
     "created_at": "2026-01-03T11:00:00Z"}]}]


class T(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); self.d = Path(self.t.name)
    def tearDown(self): self.t.cleanup()

    def test_claude_json_and_zip(self):
        j = self.d / "conversations.json"; j.write_text(json.dumps(EXPORT))
        out = self.d / "o1"; r = ICL.run(j, "Alex", "Sage", out)
        self.assertEqual((r["files"], r["messages"]), (2, 3))
        txt = (out / "2026-01-02-claude.log").read_text()
        self.assertIn("Alex: hi there", txt); self.assertIn("Sage: hello", txt)
        z = self.d / "e.zip"
        with zipfile.ZipFile(z, "w") as zf: zf.writestr("x/conversations.json", json.dumps(EXPORT))
        self.assertEqual(ICL.run(z, dest=self.d / "o2")["files"], 2)

    def test_oversize_split_and_rerun_skips(self):
        j = self.d / "c.json"; j.write_text(json.dumps(EXPORT)); out = self.d / "o"
        ICL.run(j, dest=out)
        self.assertGreaterEqual((out / "2026-01-03-claude.log").read_text().count("==="), 6)
        r = ICL.run(j, dest=out); self.assertEqual((r["files"], r["skipped_existing"]), (0, 2))

    def test_claude_errors(self):
        bad = self.d / "b.json"; bad.write_text("{nope")
        for p in (bad, self.d / "missing", self.d):
            with self.assertRaises(IC.ImportError_): ICL.load_conversations(p)
        bad.write_text("{}")
        with self.assertRaises(IC.ImportError_): ICL.load_conversations(bad)

    def test_nightly_actually_collects_imported_logs(self):
        """Regression: nightly skips stems >= today; a 'claude-...' name was silently ignored."""
        import mem0_nightly as N
        from unittest import mock
        j = self.d / "c.json"; j.write_text(json.dumps(EXPORT)); out = self.d / "logs"
        ICL.run(j, dest=out)
        (self.d / "t").mkdir(); (self.d / "t/a.md").write_text("note")
        ITX.run(self.d / "t", out)       # text file mtime = today -> held back one day, by design
        with mock.patch.object(N, "LOG_DIR", out):
            jobs = N.collect_daily_logs()
        labels = [j[0] for j in jobs]
        self.assertIn("log:2026-01-02-claude.log", labels)
        self.assertIn("log:2026-01-03-claude.log", labels)

    def test_text(self):
        (self.d / "n").mkdir(); (self.d / "n/a.md").write_text("alpha note")
        (self.d / "n/b.txt").write_text("  "); (self.d / "n/c.png").write_bytes(b"\x89")
        r = ITX.run(self.d / "n", self.d / "o")
        self.assertEqual(r["messages"], 1)
        self.assertIn("('a')", next((self.d / "o").glob("*-text.log")).read_text())
        with self.assertRaises(IC.ImportError_): ITX.run(self.d / "o" / "none")



import sqlite3
import import_sql as ISQ, import_mem0 as IM


class T2(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); self.d = Path(self.t.name)
        self.db = self.d / "c.db"
        con = sqlite3.connect(self.db)
        con.execute("create table msgs(body text, at, who text, room text)")
        con.executemany("insert into msgs values(?,?,?,?)", [
            ("hello", 1767350400, "ann", "general"),            # unix s
            ("again", 1767350460000, "bob", "general"),         # ms
            ("iso", "2026-01-03T09:00:00Z", None, None),
            ("", 1767350400, "x", "y"), ("nots", None, "x", "y")])
        con.commit(); con.close()
    def tearDown(self): self.t.cleanup()

    def test_sql_flow(self):
        con = ISQ.connect(self.db)
        self.assertEqual(ISQ.tables(con), ["msgs"])
        self.assertEqual(ISQ.columns(con, "msgs"), ["body", "at", "who", "room"])
        self.assertEqual(len(ISQ.preview(con, "msgs", "body", "at", "who", "room")), 3)
        con.close()
        r = ISQ.run(self.db, "msgs", "body", "at", "who", "room", self.d / "o")
        self.assertEqual(r["messages"], 3)
        log = next((self.d / "o").glob("2026-01-02-sql.log")).read_text()
        self.assertIn("ann: hello", log); self.assertIn("bob: again", log)

    def test_sql_readonly_and_errors(self):
        con = ISQ.connect(self.db)
        with self.assertRaises(sqlite3.OperationalError):
            con.execute("insert into msgs values('a',1,'b','c')")
        con.close()
        for args in (("msgs", "nope", "at"), ("zzz", "body", "at")):
            with self.assertRaises(IC.ImportError_):
                ISQ.run(self.db, *args, dest=self.d / "o")
        junk = self.d / "j.db"; junk.write_text("not a db at all")
        with self.assertRaises(IC.ImportError_): ISQ.connect(junk)
        with self.assertRaises(IC.ImportError_): ISQ.connect(self.d / "none.db")

    def test_mem0_formats_and_idempotence(self):
        seen, calls = set(), []
        def add(text, meta, key):
            calls.append((text, meta))
            if key in seen: return "skipped (dup)"
            seen.add(key); return "stored"
        f = self.d / "m.json"
        f.write_text(json.dumps({"results": [
            {"memory": "likes tea", "created_at": "2026-01-01", "metadata": {"a": 1}},
            "plain string", {"text": "likes tea"}, 5, {"memory": "  "}]}))
        r = IM.run(f, add)
        self.assertEqual(r, {"memories": 3, "stored": 2, "skipped_existing": 1})
        self.assertEqual(calls[0][1]["original_created_at"], "2026-01-01")
        self.assertEqual(calls[0][1]["source"], "import")
        self.assertEqual(IM.run(f, add)["stored"], 0)

    def test_mem0_errors(self):
        f = self.d / "b.json"
        for body in ("{bad", "[]", '{"x":1}', '"str"'):
            f.write_text(body)
            with self.assertRaises(IC.ImportError_): IM.load(f)
        with self.assertRaises(IC.ImportError_): IM.load(self.d / "none")


    def test_chatgpt_nested_zip_branch_and_hidden(self):
        def n(i, par, role, text, t, ctype="text", kids=()):
            msg = {"author": {"role": role}, "create_time": t, "metadata": {},
                   "content": {"content_type": ctype, "parts": [text]}}
            return {"id": i, "parent": par, "children": list(kids), "message": msg}
        mp = {"r": {"id": "r", "parent": None, "children": ["u"], "message": None},
              "u": n("u", "r", "user", "hello", 1767355200, kids=["a1", "a2"]),
              "a1": n("a1", "u", "assistant", "DISCARDED", 1767355210),
              "a2": n("a2", "u", "assistant", "kept", 1767355220, kids=["th"]),
              "th": n("th", "a2", "assistant", "secret reasoning", 1767355230, "thoughts")}
        conv = [{"title": "T", "create_time": 1767355200, "current_node": "th", "mapping": mp}]
        inner = self.d / "inner.zip"
        with zipfile.ZipFile(inner, "w") as zf: zf.writestr("conversations-000.json", json.dumps(conv))
        outer = self.d / "outer.zip"
        with zipfile.ZipFile(outer, "w") as zf: zf.write(inner, "User Online Activity/Conversations__x.zip")
        out = self.d / "o"; r = IGP.run(outer, "Alex", dest=out)
        self.assertEqual(r["messages"], 2)
        txt = next(out.glob("*-chatgpt.log")).read_text()
        self.assertIn("Alex: hello", txt); self.assertIn("ChatGPT: kept", txt)
        self.assertNotIn("DISCARDED", txt); self.assertNotIn("secret", txt)
        with self.assertRaises(IC.ImportError_): IGP.load_conversations(self.d / "nope")


if __name__ == "__main__":
    unittest.main()
