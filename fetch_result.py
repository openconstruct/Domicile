#!/usr/bin/env python3
"""Fetch a tool result (or a slice of it) back from the full transcripts.

context_diet trims old tool results when a session is resumed; the marker it
leaves carries a command that calls this script. The full text is still in the
verbatim mirror (data/sessions/*/*.jsonl) and in the Claude CLI's own files
(~/.claude/projects/*/*.jsonl).

    fetch_result.py toolu_01ABC...                  # first 8000 chars
    fetch_result.py toolu_01ABC... --start 4500 --len 6000
    fetch_result.py toolu_01ABC... --info           # length and where found

Prints at most --max chars (default 8000) so a recovery cannot flood context
again; says where to continue from when it stops short.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def default_roots():
    roots = []
    try:
        from paths import SESSIONS_DIR
        roots.append(Path(SESSIONS_DIR))
    except Exception:
        roots.append(HERE / "data" / "sessions")
    roots.append(Path.home() / ".claude" / "projects")
    return [r for r in roots if r.is_dir()]


def result_text(block):
    c = block.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c
                       if isinstance(p, dict) and p.get("type") == "text")
    return ""


def find(tool_use_id, roots):
    """-> list of (path, text) for every tool_result carrying this id."""
    hits = []
    for root in roots:
        # grep first: session files run to several MB and there are many.
        proc = subprocess.run(
            ["grep", "-rlF", "--include=*.jsonl", tool_use_id, str(root)],
            capture_output=True, text=True)
        for path in proc.stdout.split():
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if tool_use_id not in line:
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = (e.get("message") or {}).get("content")
                    if not isinstance(content, list):
                        continue
                    for b in content:
                        if (isinstance(b, dict) and b.get("type") == "tool_result"
                                and b.get("tool_use_id") == tool_use_id):
                            hits.append((path, result_text(b)))
    return hits


def best(hits):
    """Prefer a copy that is not itself already trimmed, then the longest."""
    def key(h):
        t = h[1]
        return ("characters trimmed from context" not in t, len(t))
    return max(hits, key=key)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tool_use_id")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--len", type=int, default=None, dest="length")
    ap.add_argument("--max", type=int, default=8000)
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--root", action="append", default=None,
                    help="directory of .jsonl transcripts (repeatable; for tests)")
    a = ap.parse_args(argv)

    # A tool_spill key (provider paths other than the Claude CLI) is a file.
    try:
        import tool_spill
        saved = tool_spill.load(a.tool_use_id)
    except Exception:
        saved = None
    if saved is not None:
        hits = [(str(tool_spill._spill_dir() / f"{a.tool_use_id}.txt"), saved)]
        roots = []
    else:
        roots = [Path(r) for r in a.root] if a.root else default_roots()
        hits = find(a.tool_use_id, roots)
    if not hits:
        print(f"no tool_result with id {a.tool_use_id} under: "
              + ", ".join(map(str, roots)), file=sys.stderr)
        return 1
    path, text = best(hits)
    if "characters trimmed from context" in text:
        print("warning: only a trimmed copy was found; the full text is not "
              "recoverable from these transcripts.", file=sys.stderr)
    if a.info:
        print(f"{len(text):,} chars in {path} ({len(hits)} copy/copies)")
        return 0
    want = a.length if a.length is not None else a.max
    n = min(want, a.max)
    chunk = text[a.start:a.start + n]
    sys.stdout.write(chunk)
    end = a.start + len(chunk)
    if end < min(len(text), a.start + want):
        sys.stdout.write(f"\n[stopped at --max {a.max}; continue with --start {end}]\n")
    elif a.length is None and end < len(text):
        sys.stdout.write(f"\n[{len(text) - end:,} more chars; --start {end} to continue]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
