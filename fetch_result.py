#!/usr/bin/env python3
"""Fetch a tool result (or a slice of it) back from the full transcripts.

context_diet trims old tool results when a session is resumed; the marker it
leaves carries a command that calls this script. The full text is still in the
verbatim mirror (data/sessions/*/*.jsonl) and in the Claude CLI's own files
(~/.claude/projects/*/*.jsonl).

    fetch_result.py toolu_01ABC...                  # first 5000 chars
    fetch_result.py toolu_01ABC... --start 4500 --len 6000
    fetch_result.py toolu_01ABC... --info           # length and where found
    fetch_result.py toolu_01ABC... --grep 'Traceback|ERROR'   # matching lines, with context
    fetch_result.py toolu_01ABC... --find 'retry backoff config'  # best passages for these words

--grep and --find search the WHOLE saved text, so you can go straight to the
part you need instead of paging through it. Each hit is labelled with its
character offset; pass that to --start to read on from there.

Prints at most --max chars (default 5000) so a recovery cannot flood context
again; says where to continue from when it stops short. The default has to
stay under the smallest at-source tool-output cap (6000 on the wire path):
a bigger slice would get its own middle cut out on the way back.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


_TRIMMED = re.compile(
    r"characters trimmed (?:from context|at source|from the middle)")


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
        for path in proc.stdout.splitlines():   # not split(): paths may contain spaces
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


_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were", "it",
         "for", "on", "with", "that", "this", "as", "at", "be", "by", "from", "not", "what",
         "when", "where", "why", "how", "which", "there", "here", "about"}


def _line_starts(text):
    starts, pos = [0], 0
    for line in text.splitlines(keepends=True):
        pos += len(line)
        starts.append(pos)
    return starts


def grep(text, pattern, context=2, budget=5000):
    """Lines matching `pattern` (case-insensitive regex; a bad regex is taken
    literally) with `context` lines either side, merged where they overlap."""
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        rx = re.compile(re.escape(pattern), re.IGNORECASE)
    lines = text.splitlines()
    starts = _line_starts(text)
    hit = [i for i, line in enumerate(lines) if rx.search(line)]
    if not hit:
        return f"no match for {pattern!r} in {len(text):,} chars\n"
    blocks = []
    for i in hit:
        lo, hi = max(0, i - context), min(len(lines), i + context + 1)
        if blocks and lo <= blocks[-1][1]:
            blocks[-1][1] = hi
        else:
            blocks.append([lo, hi])
    out, used, shown = [f"{len(hit)} matching line(s) for {pattern!r} in {len(text):,} chars"], 0, 0
    for lo, hi in blocks:
        body = "\n".join(lines[lo:hi])
        if used and used + len(body) > budget:
            break
        body = body[:budget]
        out.append(f"--- @{starts[lo]:,} (line {lo + 1}) ---\n{body}")
        used += len(body)
        shown += 1
    if shown < len(blocks):
        out.append(f"[{len(blocks) - shown} more region(s) not shown; narrow the pattern, "
                   f"or read one with --start <offset>]")
    return "\n".join(out) + "\n"


def find(text, query, top=3, window=800, budget=5000):
    """The passages that best cover the words in `query`. Lexical on purpose:
    no model to load, and tool output is mostly identifiers and literals."""
    terms = [t for t in dict.fromkeys(re.findall(r"[A-Za-z0-9_./:-]{2,}", query.lower()))
             if t not in _STOP]
    if not terms:
        return "nothing to search for (only stop-words given)\n"
    lines = text.splitlines(keepends=True)
    starts = _line_starts(text)
    windows, lo = [], 0                          # windows of whole lines, half-overlapping
    while lo < len(lines):
        hi, size = lo, 0
        while hi < len(lines) and (size < window or hi == lo):
            size += len(lines[hi])
            hi += 1
        windows.append((lo, hi))
        if hi >= len(lines):
            break
        lo = max(lo + 1, lo + (hi - lo) // 2)
    scored = []
    for lo, hi in windows:
        body = "".join(lines[lo:hi]).lower()
        counts = [body.count(t) for t in terms]
        distinct = sum(1 for c in counts if c)
        if distinct:
            scored.append((distinct, sum(min(c, 5) for c in counts), -lo, lo, hi))
    if not scored:
        return f"none of {terms} found in {len(text):,} chars\n"
    scored.sort(reverse=True)
    picked = []
    for _d, _f, _n, lo, hi in scored:
        if all(hi <= a or lo >= b for a, b in picked):   # skip overlaps of a better window
            picked.append((lo, hi))
        if len(picked) == top:
            break
    out, used = [f"best {len(picked)} passage(s) for {terms} in {len(text):,} chars"], 0
    for lo, hi in sorted(picked):
        body = "".join(lines[lo:hi]).rstrip("\n")[:max(200, budget - used)]
        have = [t for t in terms if t in body.lower()]
        out.append(f"--- @{starts[lo]:,} (line {lo + 1}; has {', '.join(have)}) ---\n{body}")
        used += len(body)
        if used >= budget:
            break
    return "\n".join(out) + "\n"


def best(hits):
    """Prefer a copy that is not itself already trimmed, then the longest."""
    def key(h):
        t = h[1]
        return (not _TRIMMED.search(t), len(t))
    return max(hits, key=key)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tool_use_id")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--len", type=int, default=None, dest="length")
    ap.add_argument("--max", type=int, default=5000)
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--grep", metavar="REGEX", help="show matching lines with context")
    ap.add_argument("--find", metavar="WORDS", help="show the passages that best match these words")
    ap.add_argument("-C", "--context", type=int, default=2, help="lines of context for --grep")
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
    if _TRIMMED.search(text):
        print("warning: only a trimmed copy was found; the full text is not "
              "recoverable from these transcripts.", file=sys.stderr)
    if a.info:
        print(f"{len(text):,} chars in {path} ({len(hits)} copy/copies)")
        return 0
    if a.grep or a.find:
        budget = max(500, a.max - 300)        # leave room for the headers under --max
        sys.stdout.write(grep(text, a.grep, a.context, budget) if a.grep
                         else find(text, a.find, budget=budget))
        return 0
    a.start = max(0, a.start)             # a negative start would slice from the end
    want = a.length if a.length is not None else a.max
    n = max(0, min(want, a.max))
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
