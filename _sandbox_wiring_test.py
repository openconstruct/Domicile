#!/usr/bin/env python3
"""Each tool-running path must enforce the role's write sandbox. Temp dirs only."""
import asyncio, json, os, subprocess, sys, tempfile
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config_loader as CL
import local_tools as lt

tmp = Path(tempfile.mkdtemp(prefix="sbxw-")).resolve()
ws, out = tmp / "ws", tmp / "out"; ws.mkdir(); out.mkdir()
fails = []
def check(label, got, want):
    if got != want: fails.append(f"{label}: got {got!r} want {want!r}")

roles = {"chat": {"tools": "write", "write_roots": [str(ws)]},
         "build": {"tools": "full"},
         "broken": {"tools": "write", "write_roots": ["/"]}}
check("chat roots", CL.write_roots_for_role("chat", roles), [ws])
check("build unrestricted", CL.write_roots_for_role("build", roles), None)
try:
    CL.write_roots_for_role("broken", roles); fails.append("broken sandbox config accepted")
except Exception:
    pass

# --- Claude bridge: cchat._sdk_local_tool
try:
    import cchat
    async def go(root, target):
        return await cchat._sdk_local_tool("Write", {"path": str(target), "content": "x"}, write_roots=root)
    r = asyncio.run(go([ws], out / "a.txt"))
    check("claude: outside denied", bool(r.get("is_error")) and "DENIED" in r["content"][0]["text"], True)
    check("claude: outside not written", (out / "a.txt").exists(), False)
    r = asyncio.run(go([ws], ws / "a.txt"))
    check("claude: inside ok", r.get("is_error", False), False)
    r = asyncio.run(go(None, out / "b.txt"))
    check("claude: None unrestricted", (out / "b.txt").exists(), True)
except ImportError as e:
    fails.append(f"could not import cchat for the Claude-path test: {e}")

# --- OpenAI-compatible provider: dispatch + the fail-closed helper
from providers.openai_compatible import OpenAICompatibleProvider as OC
class Stub: _turn_mode = "chat"
orig = CL.write_roots_for_role
CL.write_roots_for_role = lambda role, roles=None: orig(role, roles or roles_)
roles_ = roles
check("openai: chat roots", OC._write_roots(Stub()), [ws])
Stub._turn_mode = "broken"
check("openai: broken config => nothing writable", OC._write_roots(Stub()), [])
CL.write_roots_for_role = orig
res = asyncio.run(lt.dispatch("Write", {"path": str(out / "c.txt"), "content": "x"}, ["Write"], write_roots=[ws]))
check("dispatch: outside denied", res.startswith("DENIED"), True)
check("dispatch: outside not written", (out / "c.txt").exists(), False)

# --- Codex: env encoding and the tools-server decoding
from providers import codex_cli
CL.write_roots_for_role = lambda role, roles=None: orig(role, roles_)
check("codex env chat", json.loads(codex_cli._write_roots_env({"mode": "chat"})), [str(ws)])
check("codex env build", codex_cli._write_roots_env({"mode": "build"}), "")
check("codex env broken => []", codex_cli._write_roots_env({"mode": "broken"}), "[]")
CL.write_roots_for_role = orig

def server_roots(val):
    env = dict(os.environ); env.pop("DOMICILE_WRITE_ROOTS", None)
    if val is not None: env["DOMICILE_WRITE_ROOTS"] = val
    code = "import codex_tools_server as s; print(repr(s.WRITE_ROOTS))"
    p = subprocess.run([sys.executable, "-c", code], cwd=HERE, env=env,
                       capture_output=True, text=True, timeout=60)
    return p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "ERR " + p.stderr[-300:]
check("server: unset => None", server_roots(None), "None")
check("server: empty => None", server_roots(""), "None")
check("server: list", server_roots(json.dumps([str(ws)])), repr([ws]))
check("server: garbage => [] (closed)", server_roots("{not json"), "[]")

import shutil; shutil.rmtree(tmp)
if fails:
    print("FAIL"); [print(" -", f) for f in fails]; sys.exit(1)
print("sandbox wiring tests: all passed")
