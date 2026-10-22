#!/usr/bin/env python3
"""Forced compaction: the typed /compact command and the CompactNow tool both
leave a marker; maybe_compact() consumes it and compacts below the ceiling.
No network, no model. Run: python3 _compact_command_test.py
"""
import asyncio
import cchat
import local_tools as LT
from paths import COMPACT_REQUEST
import context_diet as cd

assert cchat.COMPACT is not None and cd.compact_enabled()
cd.compact_tail_tokens = lambda: 2_000

def mk(role, i):
    return {"role": role, "text": f"MSG{i:04d} " + f"{role} line {i}. " * 18,
            "ts": f"2026-10-15T10:{i % 60:02d}:00"}

def chat_with(n):
    return {"id": "t-compact", "title": "t", "mode": "chat",
            "messages": [mk("user" if k % 2 == 0 else "assistant", k) for k in range(n)]}

calls = []
async def fake_summary(prompt, max_tokens):
    calls.append(1)
    return "HANDOVER"
cchat.summarize_for_compaction = fake_summary
cchat.save_chat = lambda c: None
cd.append_daily_log = lambda *a, **k: None
cchat.COMPACT.append_daily_log = lambda *a, **k: None

def clear():
    COMPACT_REQUEST.unlink(missing_ok=True)

async def main():
    clear()
    ceiling = cd.COMPACT_CEILING_TOKENS

    # 1. typed command: no model call, marker written, reply ok
    async def boom(*a, **k): raise AssertionError("model must not be called")
    orig = cchat.stream_claude; cchat.stream_claude = boom
    try:
        for t in ("/compact", "  /Compact now ", "/compact\n"):
            clear()
            reply, status = await cchat.run_turn(None, chat_with(2), t)
            assert status == "ok" and "ompact" in reply, (t, reply, status)
            assert COMPACT_REQUEST.exists(), t
    finally:
        cchat.stream_claude = orig
    # a message that merely mentions it is NOT a command
    clear()
    async def fake_stream(*a, **k): return "normal", "ok"
    cchat.stream_claude = fake_stream
    r, s = await cchat.run_turn(None, chat_with(2), "can you /compact later?")
    assert r == "normal" and not COMPACT_REQUEST.exists()
    cchat.stream_claude = orig

    # 2. marker + big-enough history below the ceiling -> compacts, marker consumed
    clear(); c = chat_with(40)
    total = cd.history_tokens(c["messages"])
    assert cchat.FORCED_COMPACT_MIN_TOKENS < total < ceiling, (total, ceiling)
    await cchat.maybe_compact(c)           # no marker: below ceiling, nothing happens
    assert not c.get("pending_seed") and not calls
    COMPACT_REQUEST.write_text("x")
    await cchat.maybe_compact(c)
    assert c.get("pending_seed"), "forced compaction should stage a seed"
    assert calls, "summariser should have run"
    assert not COMPACT_REQUEST.exists(), "marker must be consumed"

    # 3. tiny history: refused, marker still consumed (no re-fire next turn)
    clear(); calls.clear(); c = chat_with(2)
    COMPACT_REQUEST.write_text("x")
    await cchat.maybe_compact(c)
    assert not c.get("pending_seed") and not calls and not COMPACT_REQUEST.exists()

    # 4. already staged: no second compaction
    clear(); calls.clear(); c = chat_with(40); c["pending_seed"] = "staged"
    COMPACT_REQUEST.write_text("x")
    await cchat.maybe_compact(c)
    assert c["pending_seed"] == "staged" and not calls and not COMPACT_REQUEST.exists()

    # 5. a previous failure does not block an explicit request
    clear(); calls.clear(); c = chat_with(40); c["compact_failed_at"] = total
    COMPACT_REQUEST.write_text("x")
    await cchat.maybe_compact(c)
    assert c.get("pending_seed") and calls

    # 6. the tool: in every tier, writes the marker, returns text
    for tier in ("readonly", "write", "sandbox", "full"):
        assert "CompactNow" in LT.tools_for(tier) if hasattr(LT, "tools_for") else True
    assert "CompactNow" in LT.PHASE_READONLY and "CompactNow" in LT.PHASE_SANDBOX
    assert "CompactNow" in LT._SCHEMAS
    clear()
    out = await LT.dispatch("CompactNow", {}, list(LT.PHASE_FULL))
    assert COMPACT_REQUEST.exists() and "requested" in out.lower(), out
    # ...and the sandbox does not block it (it is not a filesystem write by the model)
    clear()
    out = await LT.dispatch("CompactNow", {}, list(LT.PHASE_SANDBOX), write_roots=[])
    assert COMPACT_REQUEST.exists(), out
    clear()

asyncio.run(main())
print("OK")
