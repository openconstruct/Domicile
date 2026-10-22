#!/usr/bin/env python3
"""Bash confinement via bubblewrap. Real bwrap, temp dirs under $HOME."""
import asyncio, os, shutil, socket, sys, tempfile, threading, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_tools as lt, sandbox, config_loader as CL

ok, why = sandbox.bwrap_probe()
if not ok:
    print(f"SKIPPED (NOT PASSED): bwrap unusable here: {why}"); sys.exit(0)

tmp = Path(tempfile.mkdtemp(prefix=".bwt-", dir=Path.home())).resolve()
ws, out = tmp / "ws", tmp / "out"; ws.mkdir(); out.mkdir()
(out / "f").write_text("orig")
fails = []
def check(l, g, w):
    if g != w: fails.append(f"{l}: got {g!r} want {w!r}")

def bash(cmd, roots=[ws], trusted=True, **kw):
    args = {"command": cmd, **kw} if trusted else {"argv": cmd, **kw}
    try:
        r = asyncio.run(lt.execute("Bash", args, ["Bash"], trusted=trusted, write_roots=roots))
        return r if isinstance(r, dict) else {"raw": r}
    except lt.ToolDenied as e:
        return {"denied": str(e)}
    except Exception as e:                      # a crash is a result, not a test abort
        return {"error": f"{type(e).__name__}: {e}"}
def so(r): return (r.get("stdout", "") + r.get("stderr", ""))

r = bash(f"echo hi > {ws}/a && cat {ws}/a"); check("write inside", "hi" in so(r), True)
check("inside persisted on host", (ws / "a").read_text().strip(), "hi")
bash(f"echo x > {out}/f"); check("write outside blocked", (out / "f").read_text(), "orig")
_hp = Path.home() / f".bwt-probe-{os.getpid()}"
bash(f"touch {_hp}"); check("write HOME blocked", _hp.exists(), False); _hp.unlink(missing_ok=True)
(ws / "lnk").symlink_to(out)
bash(f"echo y > {ws}/lnk/f"); check("symlink escape blocked", (out / "f").read_text(), "orig")
bash(f"rm -rf {out}"); check("rm outside blocked", (out / "f").exists(), True)
out.mkdir(exist_ok=True); (out / "f").exists() or (out / "f").write_text("orig")

# argv form is confined too
r = bash(["bash", "-c", f"echo z > {out}/f"], trusted=False); check("argv form blocked", (out / "f").read_text(), "orig")

# network
srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1); port = srv.getsockname()[1]
r = bash(f"python3 -c \"import socket;socket.create_connection(('127.0.0.1',{port}),3);print('REACHED'+'_SERVER')\"")
# stdout only: a traceback on stderr echoes the source line, which would false-match
check("loopback network blocked", "REACHED_SERVER" in r.get("stdout", ""), False)
check("...and it failed rather than silently skipping", r.get("returncode", r.get("exit_code")) not in (0, None), True)
# control: the same probe reaches the server when the sandbox allows network
ctl = sandbox.bwrap_argv(["python3", "-c", f"import socket;socket.create_connection(('127.0.0.1',{port}),3);print('REACHED'+'_SERVER')"], [], "/tmp", network=True)
import subprocess; check("control reaches server", "REACHED_SERVER" in subprocess.run(ctl, capture_output=True, text=True, timeout=20).stdout, True)
srv.close()

# private /tmp
(Path("/tmp") / "bwt_host_probe").write_text("s")
r = bash("ls /tmp/bwt_host_probe"); check("host /tmp hidden", "bwt_host_probe" in r.get("stdout", ""), False)
(Path("/tmp") / "bwt_host_probe").unlink()

# systemd/dbus escape route
# systemd-run's output goes to the journal, so detect an escape by its side effect:
# a file appearing on the HOST outside every writable root.
esc = out / "escaped-via-systemd"
bash(f"systemd-run --user --wait --collect touch {esc} 2>&1")
check("no systemd-run escape (side effect on host)", esc.exists(), False)
esc.unlink(missing_ok=True)

# .env masked (install .env lives at CCHAT_DATA.parent/.env)
env = lt.CL_ENV_FILE(); made = False
if not env.exists(): env.write_text("KEY=secret-value\n"); made = True
try:
    r = bash(f"cat {env}"); check(".env unreadable inside", "secret-value" in so(r), False)
finally:
    if made: env.unlink()

# ToolWake refused
try:
    asyncio.run(lt.execute("ToolWake", {"argv": ["true"]}, ["ToolWake"], write_roots=[ws])); fails.append("ToolWake allowed")
except lt.ToolDenied: pass

# empty roots: runs, nothing writable
r = bash(f"echo q > {ws}/n 2>&1", roots=[]); check("empty roots: nothing writable", (ws / "n").exists(), False)

# exit code + stdout passthrough, default cwd = first root
r = bash("pwd; exit 7"); check("rc passthrough", r.get("returncode", r.get("exit_code")), 7)
check("cwd defaults to first root", str(ws) in r.get("stdout", ""), True)

# timeout kills the whole tree
mark = f"bwtmark{os.getpid()}"
t0 = time.time(); r = bash(f"exec -a {mark} sleep 40", timeout_s=2); dt = time.time() - t0
check("timeout enforced", dt < 10, True); time.sleep(0.7)
left = os.popen(f"pgrep -fc '^{mark}'").read().strip(); check("no orphan after timeout", left, "0")

# FAIL CLOSED: no bwrap on PATH -> denied, and the command must NOT run
canary = tmp / "canary"
saved = os.environ["PATH"]; os.environ["PATH"] = "/nonexistent"
try:
    shutil.which.__module__  # noqa
    r = bash(f"touch {canary}")
finally:
    os.environ["PATH"] = saved
check("missing bwrap => denied", "denied" in r, True)
check("missing bwrap => command NOT run", canary.exists(), False)

# config: sandbox tier without roots fails closed (raises), never None
try:
    CL.write_roots_for_role("chat", {"chat": {"tools": "sandbox"}}); fails.append("tier without roots accepted")
except sandbox.SandboxConfigError: pass
check("tier with roots ok", CL.write_roots_for_role("chat", {"chat": {"tools": "sandbox", "write_roots": [str(ws)]}}), [ws])
check("sandbox tier has Bash, not ToolWake",
      ("Bash" in lt.resolve_phase("sandbox"), "ToolWake" in lt.resolve_phase("sandbox")), (True, False))
import copy
_defaults = copy.deepcopy(CL.ROLE_DEFAULTS)
[CL._expand_profile(b) for b in _defaults.values()]
check("default chat role resolves to the sandbox tier", _defaults["chat"]["tools"], "sandbox")
check("default chat profile is safe", CL.ROLE_DEFAULTS["chat"]["profile"], "safe")

shutil.rmtree(tmp)
if fails:
    print("FAIL"); [print(" -", f) for f in fails]; sys.exit(1)
print("bash sandbox tests: all passed")
