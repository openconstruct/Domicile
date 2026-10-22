#!/usr/bin/env python3
"""mem0 MCP tools and IPC routing."""

from __future__ import annotations

import functools
import os
import sys


def _nested_extraction_child() -> bool:
    """True when we were spawned by our own extraction pipeline."""
    try:
        from mem0_claude_llm import NESTED_ENV
    except Exception:
        return False
    return os.environ.get(NESTED_ENV) == "1"

from mcp.server import MCPServer  # noqa: E402  (mcp 2.x renamed FastMCP)

import mem0_ipc  # noqa: E402  -- cheap: no mem0/qdrant import here

mcp = MCPServer("memory")


def _ipc(op: str, args: dict) -> str:
    return str(mem0_ipc.call(op, args))


def _direct(op: str, args: dict) -> str:
    import mem0_core as core
    try:
        if op == "search":
            return core.search(args.get("query", ""), int(args.get("limit", 5)))
        if op == "add":
            return core.add(args.get("text", ""),
                            extract=bool(args.get("extract", True)),
                            metadata=args.get("metadata"),
                            idempotency_key=args.get("idempotency_key"),
                            guard=bool(args.get("guard", False)))
        if op == "list":
            return core.list_memories(int(args.get("limit", 20)))
        if op == "delete":
            return core.delete(args.get("memory_id", ""))
        raise ValueError(f"unknown op {op!r}")
    except core.StoreBusy as exc:
        raise RuntimeError(f"StoreBusy: {exc}") from exc


def _call(op: str, args: dict) -> str:
    """IPC first; direct ONLY when nothing is bound to the socket."""
    try:
        return _ipc(op, args)
    except mem0_ipc.Mem0IpcNoListener:
        return _direct(op, args)


def _safe(fn):
    """Return tool failures as text instead of raising."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            msg = str(exc)
            name = type(exc).__name__
            if ("StoreBusy" in name or "StoreBusy" in msg
                    or isinstance(exc, mem0_ipc.Mem0IpcDown)):
                return f"memory unavailable: {msg}"
            return f"memory error: {type(exc).__name__}: {msg}"
    return wrapper


@mcp.tool()
@_safe
def memory_search(query: str, limit: int = 5) -> str:
    """Search long-term memory semantically. Use for anything that might have
    been said in an earlier conversation — past decisions, the operator's
    preferences, why something was built the way it was, what was already
    researched. Prefer this over guessing or saying you don't recall.

    Args:
        query: A natural-language question. Meaning is matched, not keywords.
        limit: Max memories to return (default 5).
    """
    return _call("search", {"query": query.strip(), "limit": int(limit)})


@mcp.tool()
@_safe
def memory_add(text: str, extract: bool = True, metadata: dict | None = None,
               idempotency_key: str | None = None, guard: bool = False) -> str:
    """Store something in long-term memory. Write one self-contained fact per
    call, phrased so it still makes sense read cold in six months — "the
    operator prefers prose over code unless asked", not "they prefer that".

    Use for durable things: preferences, decisions and their reasons, facts
    about the operator or their projects, conclusions worth not re-deriving. Don't
    store transient state, or anything already in ~/.remember/identity.md or
    core-memories.md — those are always in context and duplicating them here
    just pays for the same fact twice.

    Args:
        text: The memory, or a chunk of conversation to pull facts out of.
        extract: True (default) sends the text to the extractor, which splits
            it into discrete facts and reconciles them against what is
            already stored -- updating a memory that changed, dropping one
            that is now contradicted, skipping an exact duplicate. False
            stores the text verbatim with no LLM call; use it when the
            wording matters or when storing something already distilled.
        metadata: Optional dict (e.g. {"author": "...", "date": "...",
            "source": "notes.log"}) threaded verbatim onto each stored
            memory's payload. Carries provenance through the rebuild.
        idempotency_key: A stable key (use mem0_core.chunk_digest(text)) that
            makes a repeat feed of the same chunk a no-op instead of a
            re-ingest. Guards rebuild re-runs against duplicating themselves.
        guard: If True, run the semantic dedup guard after adding — deletes
            any new fact whose nearest older memory exceeds the calibrated
            cutoff (0.70). Second pass behind mem0's native LLM dedup.
    """
    return _call("add", {"text": (text or "").strip(),
                         "extract": bool(extract),
                         "metadata": metadata,
                         "idempotency_key": idempotency_key,
                         "guard": bool(guard)})


@mcp.tool()
@_safe
def memory_list(limit: int = 20) -> str:
    """List stored memories, newest first. Use to audit what's in the store or
    find something to correct — for search, use memory_search instead.

    Args:
        limit: Max memories to return (default 20).
    """
    return _call("list", {"limit": int(limit)})


@mcp.tool()
@_safe
def memory_delete(memory_id: str) -> str:
    """Delete one memory by id. Ids come from memory_list. Use when something
    stored is wrong or has gone stale — a corrected memory beats a
    contradicting second copy.

    Args:
        memory_id: The id (or unique id prefix) shown by memory_list.
    """
    return _call("delete", {"memory_id": (memory_id or "").strip()})


def _selftest() -> int:
    print("store  :", _direct("add", {"text":
          "Selftest: Domicile wires mem0 in over MCP.", "extract": True}))
    listing = _direct("list", {"limit": 3}).splitlines()[0]
    print("list   :", listing)

    import time as _t
    import mem0_core as core
    core._maybe_sync_fts(force=True)
    db = core._get_fts()
    if db is None:
        print("fts    : UNAVAILABLE (semantic-only fallback)")
    else:
        n = db.execute("SELECT count(*) FROM memories").fetchone()[0]
        print(f"fts    : {n} rows mirrored")

    print("search :")
    print(_direct("search",
                  {"query": "how is mem0 connected to Domicile?", "limit": 3}))

    print("hint   : (hybrid, per-turn path)")
    t0 = _t.monotonic()
    hits = core.hint("how does Domicile connect mem0 over MCP")
    print(f"         {len(hits)} hit(s) in {(_t.monotonic() - t0) * 1000:.0f}ms")
    for score, made, mem in hits:
        tag = "[kw]" if score <= 0 else f"[{score:.2f}]"
        print(f"         {tag} {made} {mem[:90]}")
    import mem0_core as core
    stale = [r for r in core.rows(core.store().get_all(
        filters={"user_id": core.USER_ID}, top_k=core.ALL_ROWS))
        if r.get("memory", "").startswith("Selftest:")]
    for r in stale:
        core.store().delete(memory_id=r["id"])
    print(f"cleaned: {len(stale)} selftest row(s)")
    return 0


if __name__ == "__main__":
    if _nested_extraction_child():
        sys.exit("mem0_server: refusing to run nested inside its own summarizer")
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    mcp.run()
