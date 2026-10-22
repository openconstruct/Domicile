"""MidTurnQueue contract: deliver / fall back / cancel / single owner."""
import asyncio
import json
import os
import types
import unittest
from unittest.mock import AsyncMock, patch

from midturn import MidTurnQueue, MidTurnSocketBridge, take_from_socket


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_closed_queue_refuses(self):
        q = MidTurnQueue()
        self.assertIsNone(q.submit("D", "hi"))          # nothing open -> caller waits for lock

    async def test_deliver_then_close(self):
        q = MidTurnQueue()
        self.assertTrue(q.open())
        e = q.submit("W", "hello")
        self.assertIsNotNone(e)
        got = q.take()
        self.assertEqual([x.text for x in got], ["hello"])
        q.deliver(got[0])
        self.assertTrue(await e.fut)
        self.assertEqual(q.close(), 0)                  # nothing left behind
        self.assertIsNone(q.submit("W", "late"))        # closed again

    async def test_undelivered_resolves_false_on_close(self):
        q = MidTurnQueue()
        q.open()
        a, b = q.submit("D", "one"), q.submit("C", "two")
        self.assertEqual(q.close(), 2)
        self.assertFalse(await a.fut)
        self.assertFalse(await b.fut)

    async def test_cancel_removes_pending(self):
        q = MidTurnQueue()
        q.open()
        e = q.submit("D", "x")
        q.cancel(e)
        self.assertEqual(q.take(), [])
        q.cancel(e)                                     # idempotent

    async def test_single_owner_and_blank_ignored(self):
        q = MidTurnQueue()
        self.assertTrue(q.open())
        self.assertFalse(q.open())                      # second concurrent turn can't claim it
        self.assertIsNone(q.submit("D", "   "))
        q.close()
        self.assertTrue(q.open())                       # reusable after close


class AttachedTurnTests(unittest.IsolatedAsyncioTestCase):
    """The Discord/web entry point (cchat.Domicile._attached_turn) hands a
    mid-turn message to the running turn instead of waiting on the lock."""

    def setUp(self):
        import cchat
        from midturn import MIDTURN
        self.cchat, self.mt = cchat, MIDTURN
        self.cls = next(v for v in vars(cchat).values()
                        if isinstance(v, type) and "_attached_turn" in vars(v))
        self.app = type("StubApp", (), {})()
        self.app.turn_lock = asyncio.Lock()
        self.mt.active = False
        self.mt._pending = []
        self.addCleanup(self.mt.close)

    async def test_delivered_returns_ack(self):
        await self.app.turn_lock.acquire()               # a turn is "running"
        self.assertTrue(self.mt.open())
        task = asyncio.create_task(
            self.cls._attached_turn(self.app, "also check X", source="D"))
        await asyncio.sleep(0.05)
        (entry,) = self.mt.take()
        self.assertEqual((entry.source, entry.text), ("D", "also check X"))
        self.mt.deliver(entry)
        self.assertEqual(await asyncio.wait_for(task, 2), self.cchat.MIDTURN_ACK)

    async def test_cancelled_caller_leaves_nothing_queued(self):
        await self.app.turn_lock.acquire()
        self.mt.open()
        task = asyncio.create_task(
            self.cls._attached_turn(self.app, "never mind", source="W"))
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.mt.take(), [])

    async def test_fresh_api_chat_never_opens_the_queue(self):
        # a scratch chat must not receive messages meant for the main one
        import inspect
        src = inspect.getsource(self.cchat.stream_claude)
        self.assertIn('chat.get("id") != "api-fresh"', src)

    async def test_not_offered_when_unsuitable(self):
        # no open queue -> straight to the lock (blocks, as before)
        await self.app.turn_lock.acquire()
        task = asyncio.create_task(
            self.cls._attached_turn(self.app, "x", source="D"))
        await asyncio.sleep(0.05)
        self.assertFalse(task.done())
        task.cancel()
        # images, or the TUI/automation sources, never enter the queue
        self.mt.open()
        for kw in ({"source": "D", "images": [object()]}, {"source": "T"}, {"source": ""}):
            t = asyncio.create_task(self.cls._attached_turn(self.app, "x", **kw))
            await asyncio.sleep(0.05)
            self.assertEqual(self.mt.take(), [], kw)
            t.cancel()


class ToolBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_codex_socket_delivers_and_records(self):
        q = MidTurnQueue()
        q.open()
        entry = q.submit("W", "new instruction")
        recorded = []
        bridge = await MidTurnSocketBridge(
            q, lambda e: f"user:{e.text}", recorded.append).start()
        try:
            self.assertEqual(await take_from_socket(bridge.path),
                             ["user:new instruction"])
            self.assertTrue(await entry.fut)
            self.assertEqual(recorded, [entry])
            self.assertEqual(await take_from_socket(bridge.path), [])
        finally:
            await bridge.close()
            q.close()

    async def test_codex_mcp_result_carries_queued_message(self):
        import codex_tools_server as server
        q = MidTurnQueue()
        q.open()
        entry = q.submit("D", "please stop")
        bridge = await MidTurnSocketBridge(
            q, lambda e: f"[DOMICILE MID-TURN USER MESSAGE] {e.text}",
            lambda e: None).start()
        try:
            with patch.dict(os.environ, {"DOMICILE_MIDTURN_SOCKET": bridge.path}), \
                    patch.object(server.LT, "dispatch", new_callable=AsyncMock,
                                 return_value="file contents"):
                out = await server.Read(path="/tmp/test.txt")
            self.assertIn("file contents", out)
            self.assertIn("please stop", out)
            self.assertTrue(await entry.fut)
        finally:
            await bridge.close()
            q.close()

    async def test_codex_socket_no_tool_falls_back(self):
        q = MidTurnQueue()
        q.open()
        entry = q.submit("D", "late")
        bridge = await MidTurnSocketBridge(q, lambda e: e.text,
                                            lambda e: None).start()
        await bridge.close()
        self.assertEqual(q.close(), 1)
        self.assertFalse(await entry.fut)

    async def test_codex_provider_wires_socket_and_closes_queue(self):
        import cchat
        from providers.codex_cli import CodexCLIProvider
        cchat.MIDTURN.close()
        provider = CodexCLIProvider("codex", {})
        provider._model = lambda chat: "fake"
        provider._base_options = lambda chat, model: ["-c", 'instructions="test"']
        # the fail-closed native-tool lockdown would shell out to the real codex
        import providers.codex_cli as CC
        self.enterContext(patch.object(CC, "lockdown_catalog",
                                       return_value=("/x", frozenset({"fake"}))))
        self.enterContext(patch.object(CC, "lockdown_options", return_value=[]))
        async def no_hints(text):
            return ""
        provider._memory_hints = no_hints
        chat = {"id": "test", "mode": "chat", "messages": []}
        seen = []
        async def fake_run(argv, on_event):
            opt = next(x for x in argv if x.startswith(
                "mcp_servers.domicile_tools.env.DOMICILE_MIDTURN_SOCKET="))
            path = json.loads(opt.split("=", 1)[1])
            instruction = next(x for x in argv if x.startswith("instructions="))
            self.assertIn(cchat.MIDTURN.token, json.loads(instruction.split("=", 1)[1]))
            entry = cchat.MIDTURN.submit("W", "queued")
            self.assertIsNotNone(entry)
            notices = await take_from_socket(path)
            self.assertEqual(len(notices), 1)
            self.assertIn("queued", notices[0])
            self.assertTrue(await entry.fut)
            seen.append(path)
            on_event({"type": "item.completed", "item": {
                "type": "agent_message", "text": notices[0]}})
            return 0
        provider._run = fake_run
        try:
            with patch.object(cchat, "append_message"):
                reply, status = await provider.astream_turn(chat, "start")
            self.assertEqual(status, "ok")
            self.assertIn("queued", reply)
            self.assertFalse(cchat.MIDTURN.active)
            self.assertFalse(os.path.exists(seen[0]))
        finally:
            cchat.MIDTURN.close()

    async def test_direct_api_provider_delivers_before_next_model_call(self):
        import cchat
        from providers.openai_compatible import OpenAICompatibleProvider
        cchat.MIDTURN.close()
        provider = object.__new__(OpenAICompatibleProvider)
        provider.name = "test"
        provider.model = "fake"
        provider._client = types.SimpleNamespace(default_headers={})
        async def no_mcp():
            pass
        provider._ensure_mcp = no_mcp
        provider._max_rounds = lambda: 2
        provider._sanitize_tool_calls = lambda calls: None
        def build(chat, prompt, wire_prompt=None):
            chat["wire"] = [{"role": "user", "content": prompt}]
            return [{"role": "system", "content": "test"}] + chat["wire"][:]
        provider._build_messages = build
        seen = []
        async def one_call(messages, on_delta, parts):
            self.assertIn(cchat.MIDTURN.token, messages[0]["content"])
            if not seen:
                seen.append("first")
                return "", [{"id": "call-1", "function": {
                    "name": "Read", "arguments": "{}"}}], None, None
            seen.append(messages[-1]["content"])
            return "done", [], None, None
        provider._one_call = one_call
        async def run_tools(calls):
            entry = cchat.MIDTURN.submit("D", "new request")
            seen.append(entry)
            return [{"role": "tool", "tool_call_id": "call-1",
                     "content": "tool output"}]
        provider._run_tools = run_tools
        chat = {"id": "test", "wire": [], "messages": []}
        try:
            with patch.object(cchat, "append_message"):
                reply, status = await provider._astream_turn(chat, "start")
            self.assertEqual((reply, status), ("done", "ok"))
            self.assertIn("new request", seen[-1])
            self.assertTrue(await seen[1].fut)
            self.assertFalse(cchat.MIDTURN.active)
        finally:
            cchat.MIDTURN.close()

    async def test_direct_api_attaches_to_tool_message(self):
        import cchat
        q = cchat.MIDTURN
        q.close()
        q.open()
        entry = q.submit("W", "also check this")
        chat = {"messages": []}
        results = [{"role": "tool", "content": "tool output"}]
        recorded = []
        try:
            with patch.object(cchat, "append_message",
                              side_effect=lambda *a: recorded.append(a)):
                self.assertEqual(cchat._attach_midturn_to_tool_results(chat, results), 1)
            self.assertIn("also check this", results[0]["content"])
            self.assertIn("DOMICILE MID-TURN USER MESSAGE", results[0]["content"])
            self.assertEqual(recorded[0][1], "user")
            self.assertTrue(await entry.fut)
        finally:
            q.close()


if __name__ == "__main__":
    unittest.main()
