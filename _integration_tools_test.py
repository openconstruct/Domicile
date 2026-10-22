"""Attachment, adapter, and attached-turn regressions with fake services."""
import asyncio
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import AsyncMock, patch

import api_server as API
import cchat
import local_tools as LT
import providers.openai_compatible as ORP
import providers.openai_responses as RESP
import providers.mcp_client as MCP


class AttachmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.source = root / "source.txt"
        self.source.write_text("file content")
        outbox = patch.object(API, "OUTBOX_DIR", root / "outbox")
        outbox.start()
        self.addCleanup(outbox.stop)
        self.server = API.ApiServer(None)

    def test_marker_across_every_chunk_boundary(self):
        text = f"before [[send_file:{self.source} as renamed.txt]] after"
        for split in range(len(text) + 1):
            st = API._outbox_state(f"{split:012x}")
            output = self.server._feed_delta(st, text[:split], False)
            output += self.server._feed_delta(st, text[split:], False)
            output += self.server._feed_delta(st, "", True)
            self.assertEqual(output, "before  after", split)
            self.assertEqual(st["files"], ["renamed.txt"], split)
            self.assertEqual((API.OUTBOX_DIR / st["turn"] / "renamed.txt").read_text(), "file content")

    def test_streaming_and_plain_staging_match(self):
        marker = f"[[send_file:{self.source} as renamed.txt]]"
        text = f"first {marker} second {marker}"
        clean, files, problems = API._stage_attachments(text, "plain")
        st = API._outbox_state("stream")
        self.assertEqual(self.server._feed_delta(st, text, True), clean)
        self.assertEqual(st["files"], files)
        self.assertEqual(st["probs"], problems)
        self.assertEqual(files, ["renamed.txt", "renamed-1.txt"])

    def test_copy_failure_is_reported(self):
        with patch.object(API.shutil, "copy2", side_effect=OSError("disk full")):
            clean, files, problems = API._stage_attachments(f"[[send_file:{self.source}]]", "copy")
        self.assertEqual((clean, files), ("", []))
        self.assertIn("disk full", problems[0])

    def test_caps_apply_to_both_paths(self):
        text = f"[[send_file:{self.source}]]" * 5
        clean, files, problems = API._stage_attachments(text, "count")
        self.assertEqual(len(files), 4)
        self.assertEqual(len(problems), 1)
        with patch.object(API, "MAX_OUTBOUND_BYTES", 1):
            _, files, problems = API._stage_attachments(text, "size")
        self.assertEqual(files, [])
        self.assertEqual(len(problems), 5)

    def test_history_accepts_marker_case_and_spacing(self):
        chat = {"messages": [{"role": "assistant", "text": f"[[ SEND_FILE : {self.source}]]"}]}
        API._resolve_history_files(chat)
        self.assertIn("/files/", chat["messages"][0]["text"])


class AsyncIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_turns_start_one_mcp_manager(self):
        provider = object.__new__(ORP.OpenAICompatibleProvider)
        provider._mcp = None
        provider._mcp_lock = asyncio.Lock()
        provider.use_mcp = True
        provider.mcp_allow = None

        async def start():
            await asyncio.sleep(0.01)

        manager = types.SimpleNamespace(available=True, start=AsyncMock(side_effect=start), status=lambda: "up")
        with patch.object(MCP, "MCPManager", return_value=manager) as factory, \
                patch.object(cchat, "mcp_servers", return_value={}), patch.object(ORP, "_log"):
            await asyncio.gather(provider._ensure_mcp(), provider._ensure_mcp())
        factory.assert_called_once()
        manager.start.assert_awaited_once()
        self.assertIs(provider._mcp, manager)

    async def test_responses_retries_actually_sleep(self):
        class Rate(ORP._RateLimitError):
            def __init__(self):
                Exception.__init__(self, "429")

        provider = object.__new__(RESP.OpenAIResponsesProvider)
        create = AsyncMock(side_effect=[Rate(), "done"])
        provider._client = types.SimpleNamespace(responses=types.SimpleNamespace(create=create))
        with patch.object(RESP, "_RATE_LIMIT_BACKOFF", (1,)), \
                patch.object(ORP.asyncio, "sleep", new_callable=AsyncMock) as sleep, \
                patch.object(ORP, "_log"):
            self.assertEqual(await provider._aresp_retrying(model="test"), "done")
            sleep.assert_awaited_once()
            self.assertGreaterEqual(sleep.await_args.args[0], 1)
        self.assertEqual(create.await_count, 2)

    async def test_sdk_web_does_not_block_event_loop(self):
        started, release = threading.Event(), threading.Event()

        def fetch(args):
            started.set()
            release.wait(2)
            return {"content": "page"}

        with patch.dict(LT._IMPL, {"WebFetch": fetch}):
            task = asyncio.create_task(cchat._sdk_local_tool("WebFetch", {"url": "http://test"}))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 1))
                self.assertFalse(task.done())
            finally:
                release.set()
            self.assertEqual((await asyncio.wait_for(task, timeout=5))["content"][0]["text"], "page")

    async def test_attached_turns_stay_serial_and_release_lock(self):
        chat = {"messages": []}
        app = types.SimpleNamespace(turn_lock=asyncio.Lock(), _bridge_target=lambda: (chat, None))
        app._attached_turn = types.MethodType(cchat.CCChat._attached_turn, app)
        events = []

        def append(chat, role, text):
            self.assertTrue(app.turn_lock.locked())
            events.append((role, text))

        async def run(app, chat, text, *args, **kwargs):
            await asyncio.sleep(0)
            return text + " reply", "ok"

        async def compact(*args):
            self.assertTrue(app.turn_lock.locked())

        with patch.object(cchat, "append_message", append), patch.object(cchat, "run_turn", run), \
                patch.object(cchat, "maybe_compact", compact):
            replies = await asyncio.gather(cchat.CCChat.api_turn(app, "one"), cchat.CCChat.api_turn(app, "two"))
        self.assertEqual(replies, ["one reply", "two reply"])
        self.assertEqual(events, [("user", "one"), ("assistant", "one reply"),
                                  ("user", "two"), ("assistant", "two reply")])
        with patch.object(cchat, "append_message", append), \
                patch.object(cchat, "run_turn", side_effect=RuntimeError("failed")):
            with self.assertRaises(RuntimeError):
                await app._attached_turn("error")
        self.assertFalse(app.turn_lock.locked())

    async def test_fresh_turn_skips_attached_state(self):
        app = types.SimpleNamespace(_attached_turn=AsyncMock(), turn_lock=asyncio.Lock())
        with patch.object(cchat, "stream_claude", return_value=("fresh", "ok")) as stream:
            self.assertEqual(await cchat.CCChat.api_turn(app, "hello", fresh=True), "fresh")
        app._attached_turn.assert_not_called()
        self.assertEqual(stream.call_args.args[0]["messages"], [])

    async def test_invalid_api_shapes_return_400(self):
        app = types.SimpleNamespace(api_turn=AsyncMock())
        server = API.ApiServer(app)
        for body in ([], None, 1, {"messages": ["text"]}, {"messages": []}):
            request = types.SimpleNamespace(json=AsyncMock(return_value=body))
            self.assertEqual((await server._completions(request)).status, 400)
        app.api_turn.assert_not_called()

    async def test_stream_fallback_stages_attachments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("attachment")
            response = types.SimpleNamespace(prepare=AsyncMock(), write=AsyncMock())
            app = types.SimpleNamespace(api_turn=AsyncMock(return_value=f"done [[send_file:{source}]]"))
            with patch.object(API, "OUTBOX_DIR", root / "outbox"), \
                    patch.object(API.web, "StreamResponse", return_value=response):
                await asyncio.wait_for(
                    API.ApiServer(app)._stream(object(), "hello", False, "sage"), timeout=5)
            wire = b"".join(c.args[0] for c in response.write.await_args_list).decode()
            self.assertNotIn("[[send_file:", wire)
            self.assertIn("/files/", wire)
            self.assertIn("data: [DONE]", wire)

    async def test_cancelled_stream_cleans_up_turn(self):
        started, stopped = asyncio.Event(), asyncio.Event()

        async def turn(*args, **kwargs):
            started.set()
            try:
                await asyncio.sleep(10)
            finally:
                stopped.set()

        response = types.SimpleNamespace(prepare=AsyncMock(), write=AsyncMock())
        app = types.SimpleNamespace(api_turn=turn)
        before = asyncio.all_tasks()
        with patch.object(API.web, "StreamResponse", return_value=response):
            task = asyncio.create_task(API.ApiServer(app)._stream(object(), "hi", False, "sage"))
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(stopped.is_set())
        self.assertEqual(asyncio.all_tasks() - before, set())


if __name__ == "__main__":
    unittest.main()
