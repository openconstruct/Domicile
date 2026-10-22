#!/usr/bin/env python3
"""Codex ToolWake permission handoff and cross-process completion receipts."""
import asyncio
import json
import os
import runpy
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from _test_runner import require_sandbox
require_sandbox()
os.environ["CCHAT_NO_CRASH_HOOKS"] = "1"

import cchat
import config_loader as CL
import local_tools as LT
from providers import codex_cli


class CodexToolWakeTest(unittest.TestCase):
    def test_profile_reaches_codex_options_and_server(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = CL.Roots([root])
            roots.toolwake = True
            provider = codex_cli.CodexCLIProvider("codex", {})
            with patch.object(CL, "write_roots_for_role", return_value=roots), \
                    patch.object(CL, "tools_for_role", return_value="wakeup"), \
                    patch.object(codex_cli, "lockdown_options", return_value=[]), \
                    patch.object(provider, "_system_prompt", return_value="test"):
                opts = provider._base_options({"mode": "chat"}, "gpt-6-sol")
            self.assertIn('mcp_servers.domicile_tools.env.DOMICILE_TOOLWAKE="1"', opts)
            self.assertIn('mcp_servers.domicile_tools.env.DOMICILE_TOOL_PHASE="wakeup"', opts)
            self.assertIn("ToolWake", LT.resolve_phase("wakeup"))

            server_file = str(Path(__file__).with_name("codex_tools_server.py"))
            with patch.dict(os.environ, {"DOMICILE_WRITE_ROOTS": json.dumps([str(root)]),
                                      "DOMICILE_TOOLWAKE": "1", "DOMICILE_TOOL_PHASE": "wakeup"}):
                server = runpy.run_path(server_file)
            self.assertTrue(server["WRITE_ROOTS"].toolwake)
            self.assertIn("ToolWake", server["ENABLED"])

            with patch.dict(os.environ, {"DOMICILE_WRITE_ROOTS": json.dumps([str(root)]),
                                      "DOMICILE_TOOLWAKE": "0", "DOMICILE_TOOL_PHASE": "wakeup"}):
                safe = runpy.run_path(server_file)
            self.assertFalse(safe["WRITE_ROOTS"].toolwake)
            denial = asyncio.run(LT.dispatch("ToolWake", {"argv": ["true"]},
                                             safe["ENABLED"], write_roots=safe["WRITE_ROOTS"]))
            self.assertIn("DENIED:", denial)

            with patch.dict(os.environ, {"DOMICILE_WRITE_ROOTS": "broken",
                                      "DOMICILE_TOOLWAKE": "1", "DOMICILE_TOOL_PHASE": "wakeup"}):
                broken = runpy.run_path(server_file)
            self.assertEqual(broken["WRITE_ROOTS"], [])
            denial = asyncio.run(LT.dispatch("ToolWake", {"argv": ["true"]},
                                             broken["ENABLED"], write_roots=broken["WRITE_ROOTS"]))
            self.assertIn("DENIED:", denial)

    def test_codex_server_launches_confined_wakeup_job(self):
        import sandbox
        ok, why = sandbox.bwrap_probe()
        if not ok:
            self.skipTest(f"bubblewrap unavailable: {why}")
        with tempfile.TemporaryDirectory() as directory, \
                tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as outside_directory:
            base = Path(directory)
            root = base / "allowed"
            root.mkdir()
            task_dir = base / "tasks"
            env = {"DOMICILE_WRITE_ROOTS": json.dumps([str(root)]),
                   "DOMICILE_TOOLWAKE": "1", "DOMICILE_TOOL_PHASE": "wakeup",
                   "DOMICILE_MIDTURN_SOCKET": ""}
            with patch.dict(os.environ, env), patch.object(LT, "BGTASK_DIR", task_dir):
                server = runpy.run_path(str(Path(__file__).with_name("codex_tools_server.py")))
                program = ("from pathlib import Path; "
                           f"Path({str(root)!r}, 'inside').write_text('ok'); "
                           f"Path({outside_directory!r}, 'outside').write_text('bad')")
                result = asyncio.run(server["ToolWake"](argv=["python3", "-c", program],
                                                         cwd=str(root), label="sandbox probe"))
                self.assertIn("Started [", result)
                metas = list(task_dir.glob("*.json"))
                self.assertEqual(len(metas), 1)
                task_id = json.loads(metas[0].read_text())["id"]
                receipt = task_dir / f"{task_id}.exit"
                for _ in range(100):
                    if receipt.exists():
                        break
                    time.sleep(0.05)
                self.assertTrue(receipt.exists())
                self.assertEqual(int(receipt.read_text()), 1)  # outside write was denied
                self.assertEqual((root / "inside").read_text(), "ok")
                self.assertFalse((Path(outside_directory) / "outside").exists())
                LT._BG_PROCS.pop(task_id).wait(timeout=2)

    def test_detached_exit_receipts_survive_missing_handle_and_live_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            with patch.object(LT, "BGTASK_DIR", task_dir), \
                    patch.object(cchat, "BGTASK_DIR", task_dir):
                for code in (0, 7):
                    result = LT.spawn_bg(["python3", "-c", f"raise SystemExit({code})"])
                    task_id = result["id"]
                    proc = LT._BG_PROCS.pop(task_id)  # Codex tool server is another process
                    meta_path = task_dir / f"{task_id}.json"
                    meta = json.loads(meta_path.read_text())
                    meta["notified"] = True           # don't inject a real test turn
                    meta["started"] = cchat.PROCESS_START - 100
                    meta_path.write_text(json.dumps(meta))
                    receipt = task_dir / f"{task_id}.exit"
                    for _ in range(100):
                        if receipt.exists():
                            break
                        time.sleep(0.05)
                    self.assertTrue(receipt.exists())
                    fake = SimpleNamespace(_pid_alive=lambda pid: True)  # a zombie-looking pid
                    asyncio.run(cchat.CCChat._check_bgtasks(fake))
                    final = json.loads(meta_path.read_text())
                    self.assertEqual(final["exit_code"], code)
                    self.assertEqual(final["status"], "done" if code == 0 else "failed")
                    self.assertFalse(final["stale"])
                    proc.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
