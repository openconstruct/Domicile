"""Mid-turn message queue shared by Domicile's three provider paths.

A message arriving during a turn is offered to that turn after its next tool
call. Claude receives it as a user message through the SDK input stream;
OpenAI-compatible providers append it to a tool result before the next request;
Codex's stdio MCP subprocess fetches it through a per-turn Unix socket and
appends it to its tool result. The latter two use a per-turn token so ordinary
tool data cannot trivially impersonate a Domicile user update.

Contract for submit():
  - returns None when no injectable turn is open  -> caller waits for the lock
    exactly as before;
  - otherwise returns an Entry whose `fut` resolves True once the message was
    pushed into the running turn, or False if the turn ended without another
    tool call (caller then falls back to the normal path -- nothing is lost).
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import secrets
from pathlib import Path
import tempfile


class Entry:
    __slots__ = ("source", "text", "ts", "fut")

    def __init__(self, source: str, text: str):
        self.source = source
        self.text = text
        self.ts = dt.datetime.now().astimezone()
        self.fut: asyncio.Future = asyncio.get_running_loop().create_future()


class MidTurnQueue:
    def __init__(self) -> None:
        self._pending: list[Entry] = []
        self.active = False
        self.token: str | None = None

    def open(self) -> bool:
        """Claim the queue for one turn. False if another turn already has it."""
        if self.active:
            return False
        self.active = True
        self.token = secrets.token_hex(16)
        self._pending = []
        return True

    def submit(self, source: str, text: str) -> Entry | None:
        if not self.active or not (text or "").strip():
            return None
        entry = Entry(source, text)
        self._pending.append(entry)
        return entry

    def take(self) -> list[Entry]:
        out, self._pending = self._pending, []
        return out

    def cancel(self, entry: Entry) -> None:
        """Caller went away before delivery."""
        try:
            self._pending.remove(entry)
        except ValueError:
            pass

    def close(self) -> int:
        """Turn is over: anything undelivered goes back to the normal path."""
        self.active = False
        self.token = None
        left = self.take()
        for e in left:
            if not e.fut.done():
                e.fut.set_result(False)
        return len(left)

    @staticmethod
    def deliver(entry: Entry) -> None:
        if not entry.fut.done():
            entry.fut.set_result(True)



class MidTurnSocketBridge:
    """Let a per-turn stdio MCP subprocess drain the owning process's queue.

    Only the cchat process owns the queue and chat history. The socket lives in
    a private temporary directory and is removed when the turn ends.
    """

    def __init__(self, queue: MidTurnQueue, render, record):
        self.queue = queue
        self.render = render
        self.record = record
        self._temp = None
        self._server = None
        self.path = None

    async def start(self):
        self._temp = tempfile.TemporaryDirectory(prefix="cchat-midturn-")
        self.path = str(Path(self._temp.name) / "queue.sock")
        try:
            self._server = await asyncio.start_unix_server(self._handle, self.path)
            os.chmod(self.path, 0o600)
        except Exception:
            self._temp.cleanup()
            self._temp = None
            raise
        return self

    async def _handle(self, reader, writer):
        entries = []
        try:
            if await asyncio.wait_for(reader.readline(), 3) != b"take\n":
                return
            entries = self.queue.take()
            payload = json.dumps([self.render(e) for e in entries], ensure_ascii=False)
            writer.write(payload.encode("utf-8") + b"\n")
            await writer.drain()
            if await asyncio.wait_for(reader.readline(), 3) != b"ack\n":
                raise ConnectionError("mid-turn socket client did not acknowledge")
            for e in entries:
                try:
                    self.record(e)
                except Exception:
                    pass  # the tool result still carries the message
                self.queue.deliver(e)
            entries = []
            writer.write(b"ok\n")
            await writer.drain()
        except Exception:
            # No acknowledged delivery: leave the message for another tool or
            # for the queue's normal end-of-turn fallback.
            if entries:
                if self.queue.active:
                    self.queue._pending[:0] = entries
                else:
                    for e in entries:
                        if not e.fut.done():
                            e.fut.set_result(False)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def close(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None


async def take_from_socket(path: str) -> list[str]:
    """Called from the stdio MCP server after a tool finishes."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(path, limit=2_000_000), 3)
        try:
            writer.write(b"take\n")
            await writer.drain()
            raw = await asyncio.wait_for(reader.readline(), 3)
            messages = json.loads(raw)
            if not isinstance(messages, list) or not all(isinstance(x, str) for x in messages):
                return []
            writer.write(b"ack\n")
            await writer.drain()
            if await asyncio.wait_for(reader.readline(), 3) != b"ok\n":
                return []
            return messages
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
    except Exception:
        return []


MIDTURN = MidTurnQueue()
