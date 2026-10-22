"""Long-lived MCP sessions owned by dedicated tasks."""

from __future__ import annotations

import asyncio
import json
import shutil
from contextlib import AsyncExitStack

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:  # pragma: no cover
    ClientSession = None

PREFIX = "mcp"
SEP = "__"


def qualify(server: str, tool: str) -> str:
    return f"{PREFIX}{SEP}{server}{SEP}{tool}"


def unqualify(name: str) -> tuple[str, str] | None:
    """mcp__memory__memory_search -> ("memory", "memory_search")."""
    if not name.startswith(PREFIX + SEP):
        return None
    rest = name[len(PREFIX) + len(SEP):]
    server, sep, tool = rest.partition(SEP)
    if not sep or not server or not tool:
        return None
    return server, tool


class _Server:
    """One stdio MCP server: its subprocess, its session, its tool list."""

    def __init__(self, name: str, spec: dict):
        self.name = name
        self.spec = spec
        self.session = None
        self.tools: list = []
        self.error: str | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self, timeout: float = 60.0) -> None:
        if self._task is None or self._task.done():
            self._ready.clear()
            self._stop.clear()
            self.error = None
            self._task = asyncio.create_task(self._own(), name=f"mcp:{self.name}")
        try:
            await asyncio.wait_for(self._ready.wait(), timeout)
        except asyncio.TimeoutError:
            await self._cancel_owner()
            self.error = f"timed out after {timeout:.0f}s starting"
        except asyncio.CancelledError:
            await self._cancel_owner()
            raise

    async def _cancel_owner(self) -> None:
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)

    async def _own(self) -> None:
        try:
            command = self.spec.get("command")
            resolved = shutil.which(command) if command else None
            if resolved is None:
                raise RuntimeError(f"command not found: {command!r}")
            params = StdioServerParameters(
                command=resolved,
                args=list(self.spec.get("args") or []),
                env=dict(self.spec.get("env") or {}) or None,
            )
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listed = await session.list_tools()
                self.tools = list(listed.tools)
                self.session = session
                self._ready.set()
                await self._stop.wait()
        except Exception as exc:  # noqa: BLE001 - a broken server is data
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._ready.set()
            self.session = None

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            if self._task.done():
                await asyncio.gather(self._task, return_exceptions=True)
                return
            try:
                await asyncio.wait_for(asyncio.shield(self._task), 10)
            except asyncio.TimeoutError:
                await self._cancel_owner()
            except asyncio.CancelledError:
                await self._cancel_owner()
                raise


class MCPManager:
    """Every configured server, lazily started, presented as one tool table."""

    def __init__(self, servers: dict, allow: list[str] | None = None):
        self.specs = {
            name: spec for name, spec in (servers or {}).items()
            if allow is None or name in allow
        }
        self._servers: dict[str, _Server] = {}
        self._started = False

    @property
    def available(self) -> bool:
        return bool(self.specs) and ClientSession is not None

    async def start(self) -> None:
        if self._started or not self.available:
            return
        self._started = True
        self._servers = {n: _Server(n, s) for n, s in self.specs.items()}
        try:
            await asyncio.gather(*(s.start() for s in self._servers.values()))
        except BaseException:
            await self.aclose()
            raise

    def schemas(self) -> list[dict]:
        """OpenAI-format function schemas for every live tool."""
        out: list[dict] = []
        for server in self._servers.values():
            if server.session is None:
                continue
            for tool in server.tools:
                schema = (
                    getattr(tool, "inputSchema", None)
                    or getattr(tool, "input_schema", None)
                    or {"type": "object", "properties": {}}
                )
                out.append({
                    "type": "function",
                    "function": {
                        "name": qualify(server.name, tool.name),
                        "description": (tool.description or "")[:1024],
                        "parameters": schema,
                    },
                })
        return out

    def handles(self, name: str) -> bool:
        parsed = unqualify(name)
        return parsed is not None and parsed[0] in self._servers

    async def call(self, name: str, args: dict) -> str:
        parsed = unqualify(name)
        if parsed is None:
            return f"ERROR: {name} is not an MCP tool name."
        server_name, tool = parsed
        server = self._servers.get(server_name)
        if server is None:
            return f"ERROR: no MCP server named {server_name!r}."
        if server.session is None:
            return (f"ERROR: MCP server {server_name!r} is not running "
                    f"({server.error or 'unknown reason'}).")
        try:
            result = await server.session.call_tool(tool, args or {})
        except Exception as exc:  # noqa: BLE001
            return f"ERROR calling {name}: {type(exc).__name__}: {exc}"
        return _flatten(result)

    async def aclose(self) -> None:
        await asyncio.gather(*(s.stop() for s in self._servers.values()),
                             return_exceptions=True)
        self._servers, self._started = {}, False

    def status(self) -> str:
        if not self.available:
            return "MCP: no servers configured"
        bits = []
        for name, server in self._servers.items():
            if server.session is not None:
                bits.append(f"{name}({len(server.tools)} tools)")
            else:
                bits.append(f"{name}(DOWN: {server.error or 'not started'})")
        return "MCP: " + ", ".join(bits) if bits else "MCP: not started"


def _flatten(result) -> str:
    """CallToolResult -> a string, because that is all a tool message holds."""
    if getattr(result, "isError", False):
        prefix = "ERROR: "
    else:
        prefix = ""
    parts: list[str] = []
    for block in (getattr(result, "content", None) or []):
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
            continue
        kind = getattr(block, "type", None) or type(block).__name__
        parts.append(f"[{kind} content, not renderable as text]")
    if not parts:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            try:
                return prefix + json.dumps(structured)[:20000]
            except (TypeError, ValueError):
                return prefix + str(structured)[:20000]
        return prefix + "(no content)"
    return (prefix + "\n".join(parts))[:20000]
