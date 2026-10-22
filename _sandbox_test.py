#!/usr/bin/env python3
"""Write-sandbox tests. Temp dirs only."""
import asyncio, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_tools as lt
import sandbox

tmp = Path(tempfile.mkdtemp(prefix="sbx-")).resolve()
ws, out, sib = tmp / "ws", tmp / "outside", tmp / "ws2"
for d in (ws, out, sib):
    d.mkdir()
roots = sandbox.resolve_roots(["ws"], base=tmp)
fails = []

def run(name, args, wr):
    try:
        r = asyncio.run(lt.execute(name, args, [name], write_roots=wr))
        return "ok", r
    except lt.ToolDenied as e:
        return "denied", str(e)

def check(label, got, want):
    if got != want:
        fails.append(f"{label}: got {got!r} want {want!r}")

W = lambda p, c="x", wr=roots: run("Write", {"path": str(p), "content": c}, wr)[0]

check("inside ok", W(ws / "a.txt"), "ok")
check("new subdir inside ok", W(ws / "sub/deep/b.txt"), "ok")
check("outside denied", W(out / "c.txt"), "denied")
check("dotdot escape denied", W(ws / ".." / "outside" / "d.txt"), "denied")
check("sibling-prefix dir denied", W(sib / "e.txt"), "denied")
check("None = unrestricted", W(out / "f.txt", wr=None), "ok")

(ws / "link").symlink_to(out)                       # dir symlink out
check("symlinked dir denied", W(ws / "link" / "g.txt"), "denied")
(out / "target.txt").write_text("orig")
(ws / "flink").symlink_to(out / "target.txt")       # file symlink out
lt._READ_FILES.add(str((out / "target.txt").resolve()))
check("symlinked file write denied", W(ws / "flink"), "denied")
check("outside content untouched", (out / "target.txt").read_text(), "orig")

# hardlink: replace-on-write must not touch the outside inode
os.link(out / "target.txt", ws / "hard.txt")
lt._READ_FILES.add(str((ws / "hard.txt").resolve()))
lt._READ_STAMP[str((ws / "hard.txt").resolve())] = (ws / "hard.txt").stat().st_mtime_ns
W(ws / "hard.txt", "changed")
check("hardlink: outside inode untouched", (out / "target.txt").read_text(), "orig")

# Edit
(out / "h.txt").write_text("hello")
p = str((out / "h.txt").resolve()); lt._READ_FILES.add(p); lt._READ_STAMP[p] = (out / "h.txt").stat().st_mtime_ns
check("edit outside denied", run("Edit", {"path": p, "old": "hello", "new": "bye"}, roots)[0], "denied")
check("edit outside left alone", (out / "h.txt").read_text(), "hello")
(ws / "i.txt").write_text("hello")
q = str((ws / "i.txt").resolve()); lt._READ_FILES.add(q); lt._READ_STAMP[q] = (ws / "i.txt").stat().st_mtime_ns
check("edit inside ok", run("Edit", {"path": q, "old": "hello", "new": "bye"}, roots)[0], "ok")

# config validation
for bad in ("/", str(Path.home())):
    try:
        sandbox.resolve_roots([bad], base=tmp); fails.append(f"accepted bad root {bad}")
    except sandbox.SandboxConfigError:
        pass
check("empty entries skipped", sandbox.resolve_roots(["", "  "], base=tmp), [])
check("dedupe", len(sandbox.resolve_roots(["ws", str(ws)], base=tmp)), 1)
msg = run("Write", {"path": str(out / "z"), "content": "x"}, roots)[1]
check("refusal names allow.py", "allow.py" in msg and str(ws) in msg, True)

import shutil; shutil.rmtree(tmp)
if fails:
    print("FAIL"); [print(" -", f) for f in fails]; sys.exit(1)
print("sandbox tests: all passed")
