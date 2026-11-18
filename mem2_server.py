#!/usr/bin/env python3
"""mem2 MCP tools: a thin front for the full-memory engine (mem2_core).

Like mem0_server, this process holds no store and no model: every call goes
over the memory socket to cchat, which owns mem2.db and the one shared
embedder. Only when nothing is listening (cchat down, CLI use) does it open
the store itself.
"""

from __future__ import annotations

import functools
import sys

from mcp.server import MCPServer

import mem0_ipc

mcp = MCPServer("mem2")

# The first search after a big new log has to embed it before answering.
CALL_TIMEOUT = 600.0


def _call(op: str, args: dict) -> str:
    try:
        return str(mem0_ipc.call(op, args, timeout=CALL_TIMEOUT))
    except mem0_ipc.Mem0IpcNoListener:
        import mem2_core
        return mem2_core.dispatch(op, args)


def _safe(fn):
    """Return tool failures as text instead of raising."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            return f"mem2 error: {type(exc).__name__}: {exc}"
    return wrapper


@mcp.tool()
@_safe
def mem2_search(query: str, limit: int = 5, date_from: str = None,
                date_to: str = None) -> str:
    """Search past conversations. Hybrid retrieval: a semantic pass (matches
    meaning / paraphrase) and a keyword pass (matches exact strings — error
    text, PIDs, paths, config values) are merged. Use when mem0 hints aren't
    enough and you need the full context of what was said.

    Args:
        query: Search query. Works for natural-language questions and for
            literal strings like an error message or a file path.
        limit: Max results to return (default 5).
        date_from: Only search logs from this date onward (YYYY-MM-DD).
        date_to: Only search logs up to this date (YYYY-MM-DD).
    """
    return _call("mem2_search", {"query": query.strip(), "limit": int(limit),
                                 "date_from": date_from, "date_to": date_to})


@mcp.tool()
@_safe
def mem2_list_dates() -> str:
    """List dates with available conversation logs."""
    return _call("mem2_list_dates", {})


@mcp.tool()
@_safe
def mem2_get_chunk(source_file: str, chunk_index: int) -> str:
    """Get the full text of a specific conversation chunk. Use for follow-up
    detail after mem2_search returns a result you want to read in full.

    Args:
        source_file: Path before the '#' in a result's ref= field.
        chunk_index: Integer after the '#' in a result's ref= field.
    """
    return _call("mem2_get_chunk", {"source_file": source_file,
                                    "chunk_index": int(chunk_index)})


if __name__ == "__main__":
    if "--selftest" in sys.argv or "--reembed" in sys.argv:
        import mem2_core   # maintenance runs on the store directly: stop cchat first
        raise SystemExit(mem2_core._selftest() if "--selftest" in sys.argv
                         else mem2_core._reembed())
    mcp.run()
