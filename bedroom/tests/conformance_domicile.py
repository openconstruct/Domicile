#!/usr/bin/env python3
"""Conformance: Bedroom's dollhouse tools vs Domicile's real local_tools.

Runs the same calls through both and compares what the MODEL SEES:
  real     = local_tools.dispatch()  (the rendered string, incl. DENIED:/ERROR: prefixes)
  dollhouse= what bedroom.multi puts in obs["history"][i]["result"]
Needs a full Domicile checkout (local_tools importable); skips otherwise.

  python3 tests/conformance_domicile.py [--domicile ~/domicile] [-v]
Exit 0 = every case matches, 1 = deltas (printed). Deltas are the point: each
one is a place where a model trained in Bedroom learns something false.
"""
import argparse, asyncio, os, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def load_real(domicile):
    sys.path.insert(0, str(domicile))
    try:
        import local_tools
        return local_tools
    except Exception as e:  # noqa
        print(f"SKIP: cannot import local_tools from {domicile}: {e}")
        sys.exit(0)


FILES = {
    "/d/a.txt": "alpha\nbeta\ngamma\n",
    "/d/b.txt": "one\ntwo\nalpha again\n",
    "/d/sub/c.md": "# title\nbody\n",
    "/d/nonl.txt": "no trailing newline",
}

# (label, tool, args, pre) ; pre = list of (tool,args) run first on BOTH sides (e.g. a Read)
# Inherent simulation limits (the dollhouse implements a fixed command subset):
# these are expected to differ and are listed, not hidden.
SIM_LIMITS = {"bash rm", "bash missing exe"}

CASES = [
    ("read ok", "Read", {"path": "/d/a.txt"}, []),
    ("read missing", "Read", {"path": "/d/nope.txt"}, []),
    ("read offset/limit", "Read", {"path": "/d/a.txt", "offset": 2, "limit": 1}, []),
    ("glob *.txt", "Glob", {"pattern": "*.txt", "path": "/d"}, []),
    ("glob **/*.md", "Glob", {"pattern": "**/*.md", "path": "/d"}, []),
    ("grep literal", "Grep", {"pattern": "alpha", "path": "/d"}, []),
    ("grep regex", "Grep", {"pattern": "^al.ha$", "path": "/d"}, []),
    ("grep files_with_matches", "Grep", {"pattern": "alpha", "path": "/d", "output_mode": "files_with_matches"}, []),
    ("grep count", "Grep", {"pattern": "alpha", "path": "/d", "output_mode": "count"}, []),
    ("grep no match", "Grep", {"pattern": "zzzz", "path": "/d"}, []),
    ("write new", "Write", {"path": "/d/new.txt", "content": "hi\n"}, []),
    ("write overwrite unread", "Write", {"path": "/d/a.txt", "content": "X\n"}, []),
    ("write overwrite after read", "Write", {"path": "/d/a.txt", "content": "X\n"}, [("Read", {"path": "/d/a.txt"})]),
    ("edit unread", "Edit", {"path": "/d/a.txt", "old": "beta", "new": "BETA"}, []),
    ("edit ok", "Edit", {"path": "/d/a.txt", "old": "beta", "new": "BETA"}, [("Read", {"path": "/d/a.txt"})]),
    ("edit not found", "Edit", {"path": "/d/a.txt", "old": "zzz", "new": "y"}, [("Read", {"path": "/d/a.txt"})]),
    ("edit not unique", "Edit", {"path": "/d/b.txt", "old": "a", "new": "A"}, [("Read", {"path": "/d/b.txt"})]),
    ("edit alias file_path", "Edit", {"file_path": "/d/a.txt", "old_string": "beta", "new_string": "B"}, [("Read", {"path": "/d/a.txt"})]),
    ("bash ls", "Bash", {"argv": ["ls", "/d"]}, []),
    ("bash cat", "Bash", {"argv": ["cat", "/d/a.txt"]}, []),
    ("bash wc -l", "Bash", {"argv": ["wc", "-l", "/d/a.txt"]}, []),
    ("bash grep -c", "Bash", {"argv": ["grep", "-c", "alpha", "/d/b.txt"]}, []),
    ("bash grep no match", "Bash", {"argv": ["grep", "zzz", "/d/b.txt"]}, []),
    ("bash glob arg", "Bash", {"argv": ["ls", "/d/*.txt"]}, []),
    ("bash redirect >>", "Bash", {"argv": ["echo", "x", ">>", "/d/a.txt"]}, []),
    ("bash sed -i $a", "Bash", {"argv": ["sed", "-i", "$aNEW", "/d/a.txt"]}, []),
    ("bash bash -c", "Bash", {"argv": ["bash", "-c", "ls /d"]}, []),
    ("bash python -c destructive", "Bash", {"argv": ["python3", "-c", "import os; os.remove('/d/a.txt')"]}, []),
    ("bash rm", "Bash", {"argv": ["rm", "/d/a.txt"]}, []),
    ("bash missing exe", "Bash", {"argv": ["notacommand"]}, []),
    ("bash empty argv", "Bash", {"argv": []}, []),
    ("bash string argv", "Bash", {"argv": "ls /d"}, []),
]


def build_real(lt, root):
    for i, (rel, text) in enumerate(FILES.items()):
        p = root / rel.lstrip("/")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        os.utime(p, (1_700_000_000 + i, 1_700_000_000 + i))
    lt.ALLOWED_ROOTS = [root]
    lt._READ_FILES.clear(); lt._READ_STAMP.clear()


def real_call(lt, root, tool, args):
    def fix(v):
        return str(root) + v if isinstance(v, str) and v.startswith("/d") else v
    a = {}
    for k, v in args.items():
        if k == "argv" and isinstance(v, str):
            a[k] = v.replace("/d", str(root) + "/d")
        else:
            a[k] = [fix(x) for x in v] if isinstance(v, list) else fix(v)
    if tool == "Bash":
        a.setdefault("cwd", str(root))
    enabled = list(lt._SCHEMAS)
    out = asyncio.run(lt.dispatch(tool, a, enabled))
    return out.replace(str(root), "")


def doll_call(sb, ctx, tool, args):
    from bedroom.multi import _dispatch
    try:
        r = _dispatch(*ctx, tool, args)
        return str(r)[:500]
    except Exception as e:  # noqa  (mirrors multi.run_episode_multi)
        return f"error: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domicile", default=str(Path.home() / "domicile"))
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    lt = load_real(Path(a.domicile))
    from bedroom.core import Sandbox
    from bedroom.memory import FactStore, TranscriptStore
    from bedroom.web import Dollhouse
    from bedroom.social import SocialState

    bad = 0
    # vendored guard constants must equal the live ones
    from bedroom import render as R
    for name, live in (("BLOCKED_EXE", lt._BLOCKED_EXE), ("INTERPRETERS", lt._INTERPRETERS),
                       ("EXEC_FLAGS", lt._EXEC_FLAGS), ("ARG_ALIASES", lt._ARG_ALIASES),
                       ("ALIASED_TOOLS", lt._ALIASED_TOOLS)):
        ok = getattr(R, name) == live
        bad += (not ok)
        print(f"{'ok  ' if ok else 'DIFF'} constant {name}")
    for name, live in (("DESTRUCTIVE", lt._DESTRUCTIVE), ("GIT_DESTRUCTIVE", lt._GIT_DESTRUCTIVE)):
        ok = getattr(R, name).pattern == live.pattern and getattr(R, name).flags == live.flags
        bad += (not ok)
        print(f"{'ok  ' if ok else 'DIFF'} constant {name}")
    if R.TOOL_RESULT_CHARS != lt.TOOL_RESULT_CHARS:
        bad += 1; print("DIFF constant TOOL_RESULT_CHARS")
    # vendored model-facing schemas must equal what Domicile shows a native local model
    import json
    from bedroom import sync_domicile_schemas as S
    vend = json.loads(S.OUT.read_text())
    ok = vend == json.loads(json.dumps(S.live(Path(a.domicile))))
    bad += (not ok)
    print(f"{'ok  ' if ok else 'DIFF'} vendored schemas (bedroom/domicile_tools.json)"
          + ("" if ok else "  -> run: python3 -m bedroom.sync_domicile_schemas"))
    skipped = 0
    for label, tool, args, pre in CASES:
        if label in SIM_LIMITS:
            skipped += 1
            continue
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            build_real(lt, root)
            for pt, pa in pre:
                real_call(lt, root, pt, pa)
            real = real_call(lt, root, tool, args)
        sb = Sandbox(FILES)
        ctx = (FactStore(None), TranscriptStore(None), sb, Dollhouse(None), SocialState(None))
        try:
            for pt, pa in pre:
                doll_call(sb, ctx, pt, pa)
            doll = doll_call(sb, ctx, tool, args)
        finally:
            sb.cleanup()
        ok = real.strip() == doll.strip()
        bad += (not ok)
        if not ok or a.verbose:
            print(f"{'ok  ' if ok else 'DIFF'} {label}")
            if not ok:
                print(f"   real    : {real[:160]!r}")
                print(f"   dollhouse: {doll[:160]!r}")
    print(f"\n{bad} deltas; {len(CASES)-skipped} cases compared, {skipped} listed sim-limits skipped")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
