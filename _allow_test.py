#!/usr/bin/env python3
"""Sandbox defaults, protected files, and allow.py. Temp dirs only."""
import json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_loader as CL, sandbox, allow, cchat

tmp = Path(tempfile.mkdtemp(prefix="allow-")).resolve()
inst = tmp / "inst"; (inst / "data").mkdir(parents=True)
rp = inst / "data" / "roles.json"
cp = inst / "data" / "config.json"; ep = inst / ".env"
prot = [rp, cp, ep]
fails = []
def check(l, g, w):
    if g != w: fails.append(f"{l}: got {g!r} want {w!r}")

# shipped default
d = CL.ROLE_DEFAULTS
check("chat default profile", d["chat"]["profile"], "safe")
check("build unrestricted", "write_roots" in d["build"], False)
roots = sandbox.resolve_roots(d["chat"]["write_roots"], base=inst, protected=prot)
check("default roots", [r.relative_to(inst).as_posix() for r in roots],
      ["data/memory", "data/notes", "data/uploads", "workspace"])

# protected: roots containing the files that define the sandbox are refused
for bad in ("data", ".", str(inst), "data/roles.json"):
    try:
        sandbox.resolve_roots([bad], base=inst, protected=prot); fails.append(f"accepted {bad!r}")
    except sandbox.SandboxConfigError: pass
sandbox.resolve_roots(["data/memory"], base=inst, protected=prot)   # must not raise
# symlink pointing at the data dir is resolved before the check
(tmp / "sneaky").symlink_to(inst / "data")
try:
    sandbox.resolve_roots([str(tmp / "sneaky")], base=inst, protected=prot); fails.append("symlink to data/ accepted")
except sandbox.SandboxConfigError: pass

# allow.py add / remove / list against a temp roles.json
CL.CCHAT_DATA = inst / "data"; CL.ROLES_PATH = rp; CL.CONFIG_PATH = cp
proj = tmp / "proj"; proj.mkdir()
msg = allow.add("chat", str(proj), rp, inst)
check("add msg", str(proj) in msg, True)
check("persisted", str(proj) in json.loads(rp.read_text())["chat"]["write_roots"], True)
check("defaults kept", "data/memory" in json.loads(rp.read_text())["chat"]["write_roots"], True)
check("effective", proj in CL.write_roots_for_role("chat"), True)
check("add twice", "already" in allow.add("chat", str(proj), rp, inst), True)
for bad in ("/", str(Path.home()), str(inst / "data"), str(inst)):
    try:
        allow.add("chat", bad, rp, inst); fails.append(f"allow accepted {bad}")
    except sandbox.SandboxConfigError: pass
check("rejected add didn't persist", str(inst / "data") in json.loads(rp.read_text())["chat"]["write_roots"], False)
allow.remove("chat", str(proj), rp, inst)
check("removed", proj in CL.write_roots_for_role("chat"), False)
check("remove missing", "not in" in allow.remove("chat", str(proj), rp, inst), True)
rp.write_text("{not json")
try:
    allow.add("chat", str(proj), rp, inst); fails.append("clobbered corrupt roles.json")
except ValueError: pass
check("corrupt left intact", rp.read_text(), "{not json")

# the sandbox can't be edited by the agent it confines: Write to roles.json refused
import asyncio, local_tools as lt
rp.write_text("{}")
r = asyncio.run(lt.dispatch("Write", {"path": str(rp), "content": "{}"}, ["Write"],
                            write_roots=CL.write_roots_for_role("chat")))
check("agent can't write roles.json", r.startswith("DENIED"), True)

# builtin-tools rollback flag ignored for sandboxed modes
cchat.builtin_tools_enabled.__globals__["CONFIG_PATH"] = cp
cp.write_text(json.dumps({"claude_builtin_tools": True}))
check("flag ignored for chat", cchat.builtin_tools_enabled("chat"), False)
check("flag honored for build", cchat.builtin_tools_enabled("build"), True)

import shutil; shutil.rmtree(tmp)
if fails:
    print("FAIL"); [print(" -", f) for f in fails]; sys.exit(1)
print("allow/defaults tests: all passed")
