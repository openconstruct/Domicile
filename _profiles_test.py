#!/usr/bin/env python3
"""Permission profiles: resolution, fail-closed behaviour, ToolWake, allow.py."""
import asyncio, json, os, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_loader as CL, local_tools as lt, sandbox, allow

fails = []
def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r} want {want!r}")

tmp = Path(tempfile.mkdtemp(prefix=".prof-", dir=Path.home())).resolve()
RP = tmp / "roles.json"
lt.BGTASK_DIR = tmp / "bgtasks"      # never spawn into the live task dir: cchat's watcher would announce it
CL.ROLES_PATH = RP
def put(d): RP.write_text(json.dumps(d)); os.utime(RP, (time.time()+len(fails)+1, time.time()+len(fails)+1))
def roots(role): return CL.write_roots_for_role(role)

# --- defaults: no roles.json at all -> chat is safe, build is full
put({})
check("default chat profile", CL.profile_for_role("chat"), "safe")
check("default chat confined", roots("chat") is not None, True)
check("default chat no toolwake", getattr(roots("chat"), "toolwake", None), False)
check("default build unrestricted", roots("build"), None)

# --- the stray null that used to unlock everything now does nothing
put({"chat": {"write_roots": None, "tools": "full"}})
check("null + tools:full cannot unlock", roots("chat") is not None, True)
check("tools forced to sandbox", CL.tools_for_role("chat"), "sandbox")
put({"chat": {"profile": "safe", "write_roots": []}})
check("empty roots != unconfined", len(roots("chat")), len(CL.DEFAULT_WRITE_ROOTS))

# --- unknown / mangled profile fails closed to safe
put({"chat": {"profile": "ful"}})
check("typo fails closed", roots("chat") is not None, True)
check("typo flagged", "profile_error" in CL.role_block("chat"), True)
put({"chat": {"profile": ["full"]}})
try: r = roots("chat"); check("non-string profile fails closed", r is not None, True)
except Exception: pass                     # raising is also closed

# --- full / wakeup
put({"chat": {"profile": "full"}})
check("full unrestricted", roots("chat"), None)
put({"chat": {"profile": "wakeup", "write_roots": [str(tmp / "ws")]}})
check("wakeup confined", [str(x) for x in roots("chat")], [str(tmp / "ws")])
check("wakeup toolwake flag", roots("chat").toolwake, True)
check("wakeup tool tier", CL.tools_for_role("chat"), "wakeup")
check("wakeup exposes ToolWake", "ToolWake" in lt.resolve_phase(CL.tools_for_role("chat")), True)
check("safe hides ToolWake", "ToolWake" in lt.resolve_phase("sandbox"), False)

# --- ToolWake enforcement
ws = tmp / "ws"; ws.mkdir(exist_ok=True)
outside = tmp / "outside"; outside.mkdir(exist_ok=True)
def wake(argv, rts):
    try:
        r = asyncio.run(lt.execute("ToolWake", {"argv": argv}, ["ToolWake"], trusted=True, write_roots=rts))
        return r
    except lt.ToolDenied as e:
        return {"denied": str(e)}
safe_roots = CL.Roots([ws])                         # toolwake False
check("safe denies ToolWake", "denied" in wake(["true"], safe_roots), True)
check("plain list denies ToolWake (Codex path fails closed)", "denied" in wake(["true"], [ws]), True)
w = CL.Roots([ws]); w.toolwake = True
ok, why = sandbox.bwrap_probe()
if ok:
    prog = (f"import pathlib\n"
            f"pathlib.Path({str(ws)!r}, 'a').write_text('in')\n"
            f"try: pathlib.Path({str(outside)!r}, 'b').write_text('out')\n"
            f"except OSError: pass\n"
            f"n = len(open('/proc/net/dev').read().splitlines()) - 2\n"
            f"pathlib.Path({str(ws)!r}, 'net').write_text(str(n))\n")
    res = wake(["python3", "-c", prog], w)
    check("wakeup ToolWake starts", res.get("ok"), True)
    if not res.get("ok"): print("DEBUG ToolWake result:", res)
    log = Path(res.get("log", "/nonexistent"))
    for _ in range(50):
        if (ws / "net").exists() and (ws / "net").read_text().strip(): break
        time.sleep(0.1)
    check("wakeup job wrote inside root", (ws / "a").exists(), True)
    check("wakeup job blocked outside root", (outside / "b").exists(), False)
    host_ifaces = len(open("/proc/net/dev").read().splitlines()) - 2
    if host_ifaces > 1:     # only meaningful if the host has more than loopback
        check("wakeup job has network", int((ws / "net").read_text()) > 1, True)
    argv = sandbox.bwrap_argv(["true"], [ws], ws, network=True, die_with_parent=False)
    check("detached job not tied to parent", "--die-with-parent" in argv, False)
    check("network flag respected", "--unshare-net" in argv, False)
    check("default still dies with parent", "--die-with-parent" in sandbox.bwrap_argv(["true"], [ws], ws), True)
    # the job process is wrapped, i.e. meta argv[0] is bwrap
    meta = json.loads(lt._bgtask_meta_path(res["id"]).read_text())
    check("job runs under bwrap", Path(meta["argv"][0]).name, "bwrap")
else:
    print("NOTE: bwrap unusable, ToolWake confinement checks skipped:", why)

# --- allow.py
put({})
base = tmp
check("set wakeup", "wakeup profile" in allow.set_profile("chat", "wakeup", RP), True)
check("wakeup persisted", json.loads(RP.read_text())["chat"]["profile"], "wakeup")
for bad in ("ful", "none", "", "root"):
    try: allow.set_profile("chat", bad, RP); check(f"reject {bad!r}", "accepted", "rejected")
    except ValueError: pass
try: allow.set_profile("nosuch", "safe", RP); check("reject unknown role", "accepted", "rejected")
except ValueError: pass
try: allow.set_profile("chat", "full", RP); check("full without confirm refused", "accepted", "refused")
except ValueError: pass
try: allow.set_profile("chat", "full", RP, confirm=lambda q: "y"); check("'y' is not yes", "accepted", "refused")
except ValueError: pass
check("still wakeup after refusals", json.loads(RP.read_text())["chat"]["profile"], "wakeup")
allow.set_profile("chat", "full", RP, confirm=lambda q: "yes")
check("full with typed yes", json.loads(RP.read_text())["chat"]["profile"], "full")
# adding a root while on full keeps the defaults
allow.add("chat", str(ws), RP, base)
saved = json.loads(RP.read_text())["chat"]["write_roots"]
check("add on full keeps defaults", all(d in saved for d in CL.DEFAULT_WRITE_ROOTS), True)
check("add on full adds the root", str(ws) in saved, True)
# a corrupt file is not overwritten
RP.write_text("{not json")
try: allow.set_profile("chat", "safe", RP); check("corrupt json refused", "wrote", "refused")
except ValueError: pass
check("corrupt file untouched", RP.read_text(), "{not json")

# --- CompactNow must be exposed on the Claude path (it was missing from LOCAL_FS_TOOLS)
import cchat
for spec in ("sandbox", "full", "readonly"):
    check(f"CompactNow exposed on Claude path ({spec})",
          "CompactNow" in [n for n in lt.resolve_phase(spec) if n in cchat.LOCAL_FS_TOOLS], True)
check("no tools => no CompactNow", [n for n in lt.resolve_phase("none") if n in cchat.LOCAL_FS_TOOLS], [])

import shutil; shutil.rmtree(tmp, ignore_errors=True)
if fails:
    print("FAIL"); [print(" ", f) for f in fails]; sys.exit(1)
print("OK")
