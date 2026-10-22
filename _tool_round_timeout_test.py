"""Production tool-round regressions; no model or MCP server required."""
import asyncio
import unittest
from unittest.mock import patch
import local_tools as LT
import providers.openai_compatible as ORP


def call(name, args="{}", ident="t1"):
    return {"id": ident, "function": {"name": name, "arguments": args}}


class ToolRoundTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.provider = object.__new__(ORP.OpenAICompatibleProvider)
        self.provider._mcp = None
        self.provider.model = "test"
        self.provider._enabled_tools = lambda: ["Read", "Bash", "ReadImage"]
        self.provider._supports_image = lambda: True
        self.provider._TOOL_ROUND_TIMEOUT_S = 0.05
        self.log = patch.object(ORP, "_log")
        self.log.start()
        self.addCleanup(self.log.stop)

    async def test_timeout_keeps_completed_results(self):
        cancelled = asyncio.Event()

        async def dispatch(name, args, enabled, **kw):
            if name == "Read":
                return "completed read"
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.set()

        with patch.object(LT, "dispatch", dispatch):
            results = await self.provider._run_tools([
                call("Read"), call("Bash", ident="t2")])
        self.assertEqual(results[0]["content"], "completed read")
        self.assertIn("exceeded", results[1]["content"])
        self.assertIn("Check its outcome", results[1]["content"])
        self.assertTrue(cancelled.is_set())
        self.assertEqual([m["tool_call_id"] for m in results], ["t1", "t2"])

    async def test_invalid_arguments_do_not_execute(self):
        with patch.object(LT, "dispatch") as dispatch:
            for raw in ("[", "[]", "null", "42", '"text"', {"path": "x"}):
                result = await self.provider._run_tools([call("Read", raw)])
                self.assertTrue(result[0]["content"].startswith("ERROR:"), raw)
            dispatch.assert_not_called()

    async def test_mcp_failure_is_a_tool_result(self):
        class BrokenMCP:
            async def call(self, name, args):
                raise RuntimeError("offline")

        self.provider._mcp = BrokenMCP()
        result = await self.provider._run_tools([call("mcp__test__read")])
        self.assertIn("offline", result[0]["content"])

    async def test_images_follow_all_tool_results(self):
        image = {"tool": "ReadImage", "path": "/tmp/test.png", "media_type": "image/png",
                 "bytes": 3, "data_url": "data:image/png;base64,YWJj"}
        with patch.object(LT, "execute", return_value=image), patch.object(LT, "dispatch", return_value="text"):
            result = await self.provider._run_tools([
                call("ReadImage", '{"path":"/tmp/test.png"}'), call("Read", ident="t2")])
        self.assertEqual([m["role"] for m in result], ["tool", "tool", "user"])
        self.assertEqual(result[2]["content"][0]["image_url"]["url"], image["data_url"])

    async def test_image_gates_precede_execution(self):
        with patch.object(LT, "execute") as execute:
            self.provider._supports_image = lambda: False
            result = await self.provider._run_tools([call("ReadImage")])
            self.assertIn("does not support", result[0]["content"])
            self.provider._enabled_tools = lambda: []
            result = await self.provider._run_tools([call("ReadImage")])
            self.assertIn("not enabled", result[0]["content"])
            execute.assert_not_called()

    async def test_cancelling_round_cleans_up_calls(self):
        started, stopped = asyncio.Event(), asyncio.Event()

        async def dispatch(*args, **kw):
            started.set()
            try:
                await asyncio.sleep(10)
            finally:
                stopped.set()

        with patch.object(LT, "dispatch", dispatch):
            task = asyncio.create_task(self.provider._run_tools([call("Read")]))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(stopped.is_set())

    async def test_empty_round(self):
        self.assertEqual(await self.provider._run_tools([]), [])


if __name__ == "__main__":
    unittest.main()
