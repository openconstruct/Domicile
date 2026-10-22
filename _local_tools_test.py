"""Local tool regressions using disposable files and child processes."""
import asyncio
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import local_tools as LT


class ProcessTests(unittest.TestCase):
    def run_python(self, code, **kwargs):
        return LT.exec_argv([sys.executable, "-c", code], **kwargs)

    def test_small_output_does_not_block_timeout(self):
        start = time.monotonic()
        result = self.run_python("import time; print('ready', flush=True); time.sleep(10)", timeout_s=0.2)
        self.assertLess(time.monotonic() - start, 2)
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["stdout"], "ready\n")

    def test_stdin_and_output_are_drained_together(self):
        result = self.run_python(
            "import sys; sys.stdout.write('x'*100000); sys.stdout.flush(); "
            "data=sys.stdin.read(); sys.stderr.write(str(len(data)))",
            stdin="y" * 100000, max_output_bytes=110000, timeout_s=3)
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["stderr"], "100000")
        self.assertEqual(len(result["stdout"]), 100000)

    def test_output_cap(self):
        result = self.run_python("print('x'*100000)", max_output_bytes=100)
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(result["stdout"]) + len(result["stderr"]), 100)

    def test_timeout_stops_descendants(self):
        result = self.run_python(
            "import os,time; child=os.fork(); "
            "print(os.getpid(), flush=True); time.sleep(10)", timeout_s=0.2)
        self.assertTrue(result["timed_out"])
        for pid in result["stdout"].splitlines():
            stat = Path(f"/proc/{int(pid)}/stat")
            try:
                self.assertEqual(stat.read_text().split()[2], "Z")
            except (FileNotFoundError, ProcessLookupError):
                pass

    def test_invalid_deadlines(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.run_python("pass", timeout_s=value)


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_independent_commands_remain_concurrent(self):
        both_started = threading.Barrier(2)

        def command(*args, **kwargs):
            both_started.wait(timeout=1)
            return {"content": "done"}

        with patch.object(LT, "exec_argv", command):
            results = await asyncio.gather(*(
                LT.dispatch("Bash", {"argv": ["true"]}, ["Bash"]) for _ in range(2)))
        self.assertEqual(results, ["done", "done"])

    async def test_cancel_signals_running_bash(self):
        started, stopped = threading.Event(), threading.Event()

        def command(*args, _cancel, **kwargs):
            started.set()
            _cancel.wait(2)
            if _cancel.is_set():
                stopped.set()
            return {"content": "stopped"}

        with patch.object(LT, "exec_argv", command):
            task = asyncio.create_task(LT.dispatch("Bash", {"argv": ["true"]}, ["Bash"]))
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(await asyncio.to_thread(stopped.wait, 1))

    async def test_mutations_are_exclusive_and_cancelled_queue_does_not_run(self):
        started, release = threading.Event(), threading.Event()
        called = []

        def write(args):
            called.append(args["id"])
            started.set()
            release.wait(2)
            return "done"

        with patch.dict(LT._IMPL, {"Write": write}):
            first = asyncio.create_task(LT.execute("Write", {"id": 1}, ["Write"]))
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            second = asyncio.create_task(LT.execute("Write", {"id": 2}, ["Write"]))
            await asyncio.sleep(0.02)
            second.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await second
            release.set()
            await first
            await asyncio.sleep(0.02)
        self.assertEqual(called, [1])


class FileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "example.txt"
        self.reads = set(LT._READ_FILES)
        self.addCleanup(lambda: (LT._READ_FILES.clear(), LT._READ_FILES.update(self.reads)))

    def test_edit_preserves_crlf_and_append(self):
        self.path.write_bytes(b"alpha\r\nbeta\r\n")
        LT.fs_read(str(self.path))
        LT.fs_edit(str(self.path), "beta", "beta plus")
        self.assertEqual(self.path.read_bytes(), b"alpha\r\nbeta plus\r\n")

    def test_empty_match_and_invalid_utf8_leave_file_intact(self):
        original = b"alpha\xffbeta\r\n"
        self.path.write_bytes(original)
        LT.fs_read(str(self.path))
        with self.assertRaises(ValueError):
            LT.fs_edit(str(self.path), "", "x", replace_all=True)
        with self.assertRaises(UnicodeDecodeError):
            LT.fs_edit(str(self.path), "alpha", "z")
        self.assertEqual(self.path.read_bytes(), original)

    def test_read_limit(self):
        self.path.write_bytes(b"12345")
        with patch.object(LT, "MAX_TEXT_FILE_BYTES", 4), self.assertRaises(ValueError):
            LT.fs_read(str(self.path))
        self.assertNotIn(str(self.path), LT._READ_FILES)

    def test_search_error_is_not_no_matches(self):
        self.path.write_text("hello\n")
        self.assertEqual(LT.fs_grep("absent", str(self.path))["count"], 0)
        with self.assertRaises(RuntimeError):
            LT.fs_grep("[", str(self.path))


class ParityTests(unittest.TestCase):
    """Behaviors added when the Claude path moved off the SDK built-ins."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/tmp")
        self.d = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_read_pages_and_footer(self):
        f = self.d / "big.txt"
        f.write_text("\n".join(f"line{i}" for i in range(1, 5001)) + "\n")
        r = LT.fs_read(str(f))
        self.assertEqual(r["lines"], "1-2000 of 5000")
        self.assertIn("offset=2001", r["content"])
        r = LT.fs_read(str(f), offset=4990)
        self.assertEqual(r["lines"], "4990-5000 of 5000")
        self.assertNotIn("call Read again", r["content"])
        self.assertIn("4990\tline4990", r["content"])
        self.assertIn("only 5000 lines", LT.fs_read(str(f), offset=9999)["content"])

    def test_read_truncates_long_lines_and_total_chars(self):
        f = self.d / "wide.txt"
        f.write_text(("x" * 5000 + "\n") * 100)
        r = LT.fs_read(str(f))
        self.assertIn("[line truncated]", r["content"])
        self.assertLessEqual(len(r["content"]), LT.READ_MAX_CHARS + 500)
        self.assertIn("call Read again", r["content"])

    def test_read_empty_binary(self):
        e = self.d / "e"; e.write_bytes(b"")
        self.assertIn("empty", LT.fs_read(str(e))["content"])
        b = self.d / "b"; b.write_bytes(b"ab\0cd")
        with self.assertRaises(ValueError):
            LT.fs_read(str(b))

    def test_stale_file_refused(self):
        f = self.d / "s.txt"; f.write_text("one\n")
        LT.fs_read(str(f))
        time.sleep(0.01)
        f.write_text("two\n")           # changed behind our back
        with self.assertRaises(LT.ToolDenied):
            LT.fs_edit(str(f), "two", "three")
        with self.assertRaises(LT.ToolDenied):
            LT.fs_write(str(f), "x")
        LT.fs_read(str(f))
        LT.fs_edit(str(f), "two", "three")
        self.assertEqual(f.read_text(), "three\n")
        LT.fs_edit(str(f), "three", "four")      # our own write must not look stale
        self.assertEqual(f.read_text(), "four\n")

    def test_write_atomic_keeps_mode_no_tmp_left(self):
        f = self.d / "m.sh"; f.write_text("a"); f.chmod(0o755)
        LT.fs_read(str(f)); LT.fs_write(str(f), "b")
        self.assertEqual(f.stat().st_mode & 0o777, 0o755)
        self.assertEqual([x.name for x in self.d.iterdir()], ["m.sh"])
        self.assertTrue(LT.fs_write(str(self.d / "new.txt"), "n")["created"])

    def test_glob_semantics_and_order(self):
        (self.d / "sub").mkdir()
        a = self.d / "a.py"; b = self.d / "sub" / "b.py"; c = self.d / "c.txt"
        for x in (a, b, c): x.write_text("x")
        import os
        os.utime(a, (1, 1)); os.utime(b, (3, 3))
        top = LT.fs_glob("*.py", str(self.d))["files"]
        self.assertEqual(top, [str(a)])                      # not recursive
        rec = LT.fs_glob("**/*.py", str(self.d))["files"]
        self.assertEqual(rec, [str(b), str(a)])              # newest first
        self.assertEqual(LT.fs_glob("*.nope", str(self.d))["files"], [])
        both = LT.fs_glob("*.{py,txt}", str(self.d))["files"]
        self.assertEqual(sorted(both), sorted([str(a), str(c)]))   # braces expand
        self.assertEqual(LT._expand_braces("a.{b,c{d,e}}"), ["a.b", "a.cd", "a.ce"])

    def test_grep_modes(self):
        f = self.d / "g.txt"; f.write_text("Alpha\nbeta\nALPHA two\n")
        g = lambda **k: LT.fs_grep("alpha", str(f), **k)["matches"]
        self.assertEqual(g(), "(no matches)")
        self.assertEqual(g(ignore_case=True).count("\n"), 1)
        self.assertIn("g.txt", g(ignore_case=True, output_mode="files_with_matches"))
        self.assertEqual(asyncio.run(LT.dispatch("Grep", {"pattern": "beta", "path": str(f), "glob": None,
            "ignore_case": None, "line_numbers": None, "after": None, "head_limit": None}, ["Grep"])),
            f"{f}:2:beta")                                    # typed-wrapper Nones behave like defaults
        self.assertIn(":2", g(ignore_case=True, output_mode="count"))
        self.assertIn("beta", g(ignore_case=True, after=1))
        self.assertEqual(len(g(ignore_case=True, head_limit=1).splitlines()), 2)  # 1 + notice

    def test_shell_requires_trust(self):
        async def run(trusted):
            return await LT.execute("Bash", {"command": "echo hi | tr a-z A-Z"},
                                    ["Bash"], trusted=trusted)
        with self.assertRaises(LT.ToolDenied):
            asyncio.run(run(False))
        r = asyncio.run(run(True))
        self.assertEqual(r["stdout"], "HI\n")
        self.assertFalse(LT._TRUSTED.get())                  # no leak after the call

    def test_shell_overflow_drains_not_kills(self):
        async def run():
            return await LT.execute(
                "Bash", {"command": "yes | head -c 300000; echo done >&2"}, ["Bash"],
                trusted=True)
        r = asyncio.run(run())
        self.assertTrue(r["truncated"] and r["drained"])
        self.assertEqual(r["exit_code"], 0)
        self.assertIn("done", r["stderr"])

    def test_untrusted_reads_stay_confined(self):
        with self.assertRaises(LT.ToolDenied):
            LT.fs_read("/etc/hostname")
        async def run():
            return await LT.execute("Read", {"path": "/etc/hostname"}, ["Read"], trusted=True)
        self.assertIn("content", asyncio.run(run()))
        with self.assertRaises(LT.ToolDenied):               # writes never widen
            LT.fs_write("/etc/zz-test", "x")

    def test_builtin_style_arg_names_accepted(self):
        f = self.d / "al.txt"; f.write_text("one two\n")
        run = lambda n, a: asyncio.run(LT.execute(n, a, [n]))
        self.assertIn("one two", run("Read", {"file_path": str(f)})["content"])
        run("Edit", {"file_path": str(f), "old_string": "two", "new_string": "three"})
        self.assertEqual(f.read_text(), "one three\n")

    def test_schema_params_are_identifiers(self):
        # codex_tools_server.py turns schema keys into Python parameters
        for table in (LT._SCHEMAS, LT._TRUSTED_SCHEMAS):
            for name, spec in table.items():
                for key in spec["parameters"]["properties"]:
                    self.assertTrue(key.isidentifier(), f"{name}.{key}")

    def test_untrusted_schema_unchanged(self):
        names = lambda tr: LT.schemas(["Bash"], trusted=tr)[0]["function"]["parameters"]["required"]
        self.assertEqual(names(False), ["argv"])
        self.assertEqual(names(True), ["command"])


if __name__ == "__main__":
    unittest.main()
