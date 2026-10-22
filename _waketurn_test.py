#!/usr/bin/env python3
"""Wakeup turns run under the `wakeup` role's permissions, and only they do."""
import asyncio, json, os, sys, tempfile, time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_loader as CL
import cchat

fails = []
def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r} want {want!r}")

tmp = Path(tempfile.mkdtemp(prefix=".waketurn-", dir=Path.home())).resolve()
RP = tmp / "roles.json"
CL.ROLES_PATH = RP
def put(d):
    RP.write_text(json.dumps(d))
    t = time.time() + len(fails) + 1
    os.utime(RP, (t, t))

try:
    # chat is full (like the live box); wakeup is not given, so defaults apply
    put({"chat": {"profile": "full"}})
    check("chat is full", CL.write_roots_for_role("chat"), None)
    with CL.permission_role("wakeup"):
        r = CL.write_roots_for_role("chat")
        check("override confines", r is not None, True)
        check("override toolwake", getattr(r, "toolwake", None), True)
        check("override tier", CL.tools_for_role("chat"), "wakeup")
        check("override profile", CL.profile_for_role("chat"), "wakeup")
        check("override rounds", CL.max_tool_rounds("chat"), 60)
        check("claude allow list has ToolWake",
              any("ToolWake" in t for t in cchat.allowed_tools_for_mode("chat")), True)
        check("builtins off under override", cchat.builtin_tools_enabled("chat"), False)
        # model/memory lookups are untouched by the override
        check("model role unaffected", CL.model_role_for("chat"), "chat")
    check("restored after block", CL.write_roots_for_role("chat"), None)
    check("restored tier", CL.tools_for_role("chat"), "full")

    # restored even if the block raises
    try:
        with CL.permission_role("wakeup"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    check("restored after exception", CL.tools_for_role("chat"), "full")

    # a user can narrow or widen the wakeup role in roles.json
    ws = tmp / "ws"; ws.mkdir()
    put({"chat": {"profile": "full"}, "wakeup": {"profile": "safe"}})
    with CL.permission_role("wakeup"):
        check("wakeup role set to safe", getattr(CL.write_roots_for_role("chat"), "toolwake", None), False)
    put({"chat": {"profile": "full"}, "wakeup": {"profile": "wakeup", "write_roots": [str(ws)]}})
    with CL.permission_role("wakeup"):
        check("custom wakeup roots", [str(x) for x in CL.write_roots_for_role("chat")], [str(ws)])
    # a stray null / full-via-tools cannot unlock the wakeup role either
    put({"chat": {"profile": "full"}, "wakeup": {"write_roots": None, "tools": "full"}})
    with CL.permission_role("wakeup"):
        check("wakeup cannot be unlocked by null", CL.write_roots_for_role("chat") is not None, True)

    # concurrent tasks: the override does not leak into a sibling turn
    put({"chat": {"profile": "full"}})
    async def sibling():
        await asyncio.sleep(0.05)
        return CL.tools_for_role("chat")
    async def waking():
        with CL.permission_role("wakeup"):
            await asyncio.sleep(0.1)
            return CL.tools_for_role("chat")
    async def both():
        return await asyncio.gather(waking(), sibling())
    got = asyncio.run(both())
    check("waking task sees wakeup", got[0], "wakeup")
    check("sibling task unaffected", got[1], "full")

    # the real call site: wakeup_tick runs its turn under the override
    seen = {}
    async def fake_run_turn(app, chat, prompt, *a, **k):
        seen["tier"] = CL.tools_for_role("chat")
        seen["roots"] = CL.write_roots_for_role("chat")
        return "ok reply", "ok"
    async def tick():
        ready = asyncio.Event(); ready.set()
        bridge = NS(_ready=ready, send=AsyncMock(), describe_target=AsyncMock(return_value="x"))
        app = NS(bridge=bridge, turn_lock=asyncio.Lock(), notify=lambda *a, **k: None,
                 _wakeup_target=lambda: ({"messages": [], "mode": "chat"}, None))
        with patch.object(cchat, "run_turn", fake_run_turn), \
                patch.object(cchat, "append_message"), \
                patch.object(cchat, "maybe_compact", new_callable=AsyncMock):
            return await cchat.CCChat.wakeup_tick(app, forced=True)
    note = asyncio.run(tick())
    check("wakeup_tick completed", note.startswith("sent to"), True)
    check("wakeup turn ran under wakeup tier", seen.get("tier"), "wakeup")
    check("wakeup turn confined", seen.get("roots") is not None, True)
    check("tier restored after tick", CL.tools_for_role("chat"), "full")
finally:
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

if fails:
    print("FAIL"); [print(" ", f) for f in fails]; sys.exit(1)
print("ALL GOOD")
