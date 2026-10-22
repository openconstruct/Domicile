"""Bedroom: model-facing tool results in Domicile's real shape.

What a policy learns from is the STRING it sees after a tool call. In Domicile
that string is local_tools.dispatch(): rendered text for Read/Glob/Grep/Bash,
`tool=Write; path=...; bytes=N; created=True` for writes, `DENIED: ...` when a
guard refuses and `ERROR: <ExcType>: <msg>` when a tool raises. This module
produces those strings over the per-episode Sandbox, so a model trained here
sees what it will see in the house.

The rules below are vendored from domicile/local_tools.py (constants and message
text). tests/conformance_domicile.py runs the same calls through the real module
and fails on any drift -- run it whenever local_tools changes.

Known simulation limits (not bugs): Bash executes only the commands in
bash.ALLOWED_COMMANDS; Grep uses Python `re` (rg is Rust regex; they agree on
ordinary patterns) and ignores the `type` filter and `multiline`; no PDF/image
handling.
"""
import fnmatch, os, re, shlex
from pathlib import Path

from .bash import ALLOWED_COMMANDS, run_bash_tool

TOOL_RESULT_CHARS = 12_000
READ_DEFAULT_LIMIT = 2000
READ_LINE_CHARS = 2000
READ_MAX_CHARS = 40_000

BLOCKED_EXE = {
    "bash", "sh", "zsh", "fish", "dash", "ksh", "csh", "tcsh",
    "cmd", "powershell", "pwsh",
    "sudo", "doas", "su", "pkexec",
}
INTERPRETERS = {
    "python", "python2", "python3", "perl", "ruby", "node", "nodejs",
    "deno", "bun", "php", "lua", "awk", "gawk", "mawk", "sed",
    "env", "xargs", "find", "git", "vim", "vi", "nvim", "emacs",
    "ssh", "make", "docker", "systemd-run", "nohup", "setsid", "timeout",
    "watch", "script", "expect",
}
EXEC_FLAGS = {
    "-c", "-e", "-E", "--eval", "--exec", "-exec", "-execdir", "--command",
    "-I", "--interactive", "-p", "--program", "-s", "--source",
}
DESTRUCTIVE = re.compile(
    r"\b(rm|rmdir|unlink|shred|srm|truncate|mv|dd|mkfs|"
    r"tee|cp|rsync|install|ln|"
    r"rmtree|remove|move|replace|overwrite|"
    r"chown|chmod|chgrp)\b|>\s*/|--force|-f\b|--hard|-{1,2}delete|-D\b|"
    r"-{1,2}(prune|purge|wipe|destroy)\b",
    re.IGNORECASE,
)
GIT_DESTRUCTIVE = re.compile(
    r"\b(reset\s+--hard|clean\s+-[a-z]*[fd]|push\s+.*--force|push\s+.*-f\b|"
    r"branch\s+-D|checkout\s+--\s|restore\s|filter-branch|reflog\s+expire|"
    r"gc\s+--prune)",
    re.IGNORECASE,
)
ARG_ALIASES = {"file_path": "path", "old_string": "old", "new_string": "new"}
ALIASED_TOOLS = {"Read", "Write", "Edit", "Glob", "Grep"}


class ToolDenied(PermissionError):
    pass


def _disp(sb, p):
    p = Path(p)
    return "/" if str(p) == str(sb.root) else "/" + p.relative_to(sb.root).as_posix()


def _norm(tool, args):
    if tool not in ALIASED_TOOLS or not isinstance(args, dict):
        return args
    out = dict(args)
    for a, r in ARG_ALIASES.items():
        if a in out and r not in out:
            out[r] = out.pop(a)
    return out


def _mark_read(sb, p):
    st = getattr(sb, "_rd", None)
    if st is None:
        st = sb._rd = {}
    try:
        st[str(p)] = p.stat().st_mtime_ns
    except OSError:
        st[str(p)] = None


def _check_fresh(sb, p):
    seen = getattr(sb, "_rd", {}).get(str(p))
    if seen is None:
        return
    try:
        now = p.stat().st_mtime_ns
    except OSError:
        return
    if now != seen:
        raise ToolDenied(f"refused: {_disp(sb, p)} changed on disk since you last read it. "
                         f"Read it again before modifying it.")


def _window(picked, start, total):
    if total == 0:
        return "(empty file)"
    if start > total:
        return f"(file has only {total} lines)"
    rows, chars = [], 0
    for i, line in enumerate(picked, start):
        if len(line) > READ_LINE_CHARS:
            line = line[:READ_LINE_CHARS] + " ...[line truncated]"
        row = f"{i:6d}\t{line}"
        if rows and chars + len(row) > READ_MAX_CHARS:
            break
        rows.append(row)
        chars += len(row) + 1
    end = start + len(rows) - 1
    body = "\n".join(rows)
    if end < total:
        body += f"\n[showing lines {start}-{end} of {total}; call Read again with offset={end + 1} to continue]"
    return body


def _read(sb, a):
    p = sb.resolve(a["path"])
    if not p.exists():
        raise FileNotFoundError(_disp(sb, p))
    if not p.is_file():
        raise ValueError(f"not a file: {_disp(sb, p)}")
    start = max(1, int(a.get("offset") or 1))
    lim = READ_DEFAULT_LIMIT if a.get("limit") is None else max(1, int(a["limit"]))
    lines = p.read_text(errors="replace").splitlines()
    total = len(lines)
    picked = lines[start - 1:start - 1 + lim] if total else []
    _mark_read(sb, p)
    return _window(picked, start, total)


def _write(sb, a):
    p = sb.resolve(a["path"])
    b = str(a["content"]).encode("utf-8")
    existed = p.exists()
    if existed:
        if str(p) not in getattr(sb, "_rd", {}):
            raise ToolDenied(f"refused: {_disp(sb, p)} exists but has not been read this session. "
                             f"Read it first -- overwriting a file blind is how work gets lost.")
        _check_fresh(sb, p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b)
    _mark_read(sb, p)
    return f"tool=Write; path={_disp(sb, p)}; bytes={len(b)}; created={not existed}"


def _edit(sb, a):
    p = sb.resolve(a["path"])
    old, new, rall = a["old"], a["new"], bool(a.get("replace_all", False))
    if not p.is_file():
        raise FileNotFoundError(_disp(sb, p))
    if str(p) not in getattr(sb, "_rd", {}):
        raise ToolDenied(f"refused: read {_disp(sb, p)} before editing it.")
    _check_fresh(sb, p)
    if not old:
        raise ValueError("old string must not be empty")
    if old == new:
        raise ValueError("old and new are identical; nothing to change")
    text = p.read_text()
    n = text.count(old)
    if n == 0:
        hint = ""
        if old.strip() and old.strip() in text:
            hint = " (the text exists but surrounding whitespace/indentation differs)"
        raise ValueError("old string not found (whitespace must match exactly)" + hint)
    if n > 1 and not rall:
        raise ValueError(f"old string appears {n} times; pass replace_all or include more "
                         f"surrounding context to make it unique")
    p.write_text(text.replace(old, new) if rall else text.replace(old, new, 1))
    _mark_read(sb, p)
    return f"tool=Edit; path={_disp(sb, p)}; replaced={n if rall else 1}"


def _expand_braces(pat):
    m = re.search(r"\{([^{}]*)\}", pat)
    if not m:
        return [pat]
    out = []
    for alt in m.group(1).split(","):
        for x in _expand_braces(pat[:m.start()] + alt + pat[m.end():]):
            if x not in out:
                out.append(x)
    return out


def _files_under(base):
    for p in sorted(base.rglob("*")):
        if p.is_file() and ".git" not in p.relative_to(base).parts:
            yield p


def _glob(sb, a, max_results=200):
    base = sb.resolve(a.get("path") or "/")
    pat = a["pattern"].lstrip("/")
    hits = {}
    for alt in _expand_braces(pat):
        for m in base.glob(alt):
            if m.is_file():
                hits[str(m)] = m
    ms = sorted(hits.values(), key=lambda x: (-x.stat().st_mtime_ns, str(x)))
    out = [_disp(sb, m) for m in ms[:max_results]]
    if len(ms) > max_results:
        out.append(f"... {len(ms) - max_results} more not shown; narrow the pattern")
    return "\n".join(out) or "(no matches)"


def _grep(sb, a):
    base = sb.resolve(a.get("path") or "/")
    mode = a.get("output_mode") or "content"
    if mode not in ("content", "files_with_matches", "count"):
        raise ValueError("output_mode must be content, files_with_matches or count")
    flags = re.IGNORECASE if a.get("ignore_case") else 0
    try:
        rx = re.compile(a["pattern"], flags)
    except re.error as e:
        raise RuntimeError(f"regex parse error: {e}")
    lnum = True if a.get("line_numbers") is None else bool(a.get("line_numbers"))
    ctx = {"A": int(a.get("after") or 0), "B": int(a.get("before") or 0)}
    if a.get("context"):
        ctx["A"] = ctx["B"] = int(a["context"])
    paths = [base] if base.is_file() else list(_files_under(base))
    out = []
    for p in paths:
        rel = p.relative_to(base).as_posix() if p != base else p.name
        g = a.get("glob")
        if g and not (fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(p.name, g)):
            continue
        try:
            lines = p.read_text().splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        hit = [i for i, ln in enumerate(lines) if rx.search(ln)]
        if not hit:
            continue
        d = _disp(sb, p)
        if mode == "files_with_matches":
            out.append(d)
        elif mode == "count":
            out.append(f"{d}:{len(hit)}")
        else:
            show = {}
            for i in hit:
                for j in range(max(0, i - ctx["B"]), min(len(lines), i + ctx["A"] + 1)):
                    show[j] = show.get(j) or (j == i)
            prev = None
            for j in sorted(show):
                if prev is not None and j != prev + 1 and (ctx["A"] or ctx["B"]):
                    out.append("--")
                sep = ":" if show[j] else "-"
                out.append(f"{d}{sep}{j + 1}{sep}{lines[j]}" if lnum else f"{d}{sep}{lines[j]}")
                prev = j
    limit = int(a["head_limit"]) if a.get("head_limit") is not None else 250
    off = int(a.get("offset") or 0)
    total = len(out)
    shown = out[off:][:limit]
    text = "\n".join(shown) or "(no matches)"
    if total > len(shown) + off:
        text += f"\n[{total - off - len(shown)} more lines; raise head_limit or use offset]"
    return text


def _guard(argv):
    if not argv:
        raise ToolDenied("empty command")
    exe = os.path.basename(str(argv[0])).strip()
    joined = " ".join(str(x) for x in argv)
    if exe in BLOCKED_EXE:
        raise ToolDenied(f"'{exe}' is not available: commands run as an argv list with no "
                         f"shell, so shells and privilege tools are blocked. Run the "
                         f"program directly instead of via a shell.")
    if exe in INTERPRETERS:
        hit = {x for x in argv[1:] if str(x).startswith("-")} & EXEC_FLAGS
        if hit and (DESTRUCTIVE.search(joined) or GIT_DESTRUCTIVE.search(joined)):
            raise ToolDenied(f"refused: inline code ({sorted(hit)[0]}) that looks destructive. "
                             f"Write it to a file and run that, so it can be read before it executes.")
    if exe == "git" and GIT_DESTRUCTIVE.search(joined):
        raise ToolDenied(f"refused: '{joined[:80]}' rewrites or discards git history. "
                         f"Ask first -- practices.md makes this a human decision.")


_EXIT = {"grep": 2, "ls": 2}


def _bash(sb, a):
    argv = a.get("argv", a.get("command"))
    if isinstance(argv, str):
        argv = shlex.split(argv)
    if not isinstance(argv, list):
        raise TypeError("argv must be a list of strings")
    argv = [str(x) for x in argv]
    _guard(argv)
    cmd = argv[0]
    if cmd not in ALLOWED_COMMANDS:
        return f"[stderr]\n{cmd}: not available in this simulated environment\n[exit 127]"
    try:
        out = run_bash_tool(sb, "Bash", {"argv": argv})
    except ValueError as e:
        msg = str(e)
        m = re.match(r"no such (?:file|path|directory): (.*)", msg)
        if m and cmd == "ls":
            msg = f"cannot access '{m.group(1)}': No such file or directory"
        elif m:
            msg = f"{m.group(1)}: No such file or directory"
        return f"[stderr]\n{cmd}: {msg}\n[exit {_EXIT.get(cmd, 1)}]"
    out = (out or "").rstrip()
    if cmd == "grep" and out in ("", "0"):      # grep exits 1 on no match; -c prints 0 first
        return (out + "\n[exit 1]") if out else "[exit 1]"
    return out or "(no output)"


FS_BASH_TOOLS = {"Read": _read, "Write": _write, "Edit": _edit,
                 "Glob": _glob, "Grep": _grep, "Bash": _bash}


def render_fs_call(sb, tool, args):
    """Run an FS/Bash tool call and return exactly what Domicile's dispatch() would."""
    try:
        if tool not in FS_BASH_TOOLS:
            raise ValueError(f"unknown tool {tool}")
        text = FS_BASH_TOOLS[tool](sb, _norm(tool, args if isinstance(args, dict) else {}))
    except ToolDenied as e:
        return f"DENIED: {e}"
    except Exception as e:  # noqa
        return f"ERROR: {type(e).__name__}: {e}"
    if len(text) > TOOL_RESULT_CHARS:
        keep = TOOL_RESULT_CHARS // 2
        text = (text[:keep] + f"\n\n... [{len(text) - TOOL_RESULT_CHARS} chars trimmed] ...\n\n" + text[-keep:])
    return text
