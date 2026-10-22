"""MCP owner lifecycle without starting external servers."""
import asyncio
import unittest

from providers.mcp_client import _Server, qualify, unqualify


class FakeServer(_Server):
    def __init__(self):
        super().__init__("test", {})
        self.hang = True
        self.exits = 0

    async def _own(self):
        try:
            if self.hang:
                await asyncio.sleep(10)
            self.session = object()
            self._ready.set()
            await self._stop.wait()
        finally:
            self.exits += 1
            self.session = None
            self._ready.set()


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_closes_owner_and_allows_restart(self):
        server = FakeServer()
        await server.start(timeout=0.01)
        self.assertTrue(server._task.done())
        self.assertEqual(server.exits, 1)
        self.assertIn("timed out", server.error)
        await server.stop()
        server.hang = False
        await server.start(timeout=1)
        self.assertIsNotNone(server.session)
        self.assertIsNone(server.error)
        await server.stop()
        self.assertEqual(server.exits, 2)
        await server.stop()

    async def test_cancel_start_closes_owner(self):
        server = FakeServer()
        task = asyncio.create_task(server.start())
        await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(server._task.done())
        self.assertEqual(server.exits, 1)

    def test_qualified_names_keep_tool_separators(self):
        self.assertEqual(unqualify(qualify("memory", "read__detail")), ("memory", "read__detail"))
        self.assertIsNone(unqualify("mcp__memory__"))


if __name__ == "__main__":
    unittest.main()
