"""Bedroom M5: Bash dollhouse -- Domicile's Bash tool over the sandbox.

The real house Bash (rebaseline, 2026-10-18) takes an **argv list, no
shell**, and runs bwrap-sandboxed into the resident's write roots. The
dollhouse mirrors that contract: the tool is `Bash`, its one argument is
`argv`, a list of strings, and commands execute only against the
per-episode Sandbox from core.py -- no clocks, no randomness, no state
change except through this tool.

Because there is no shell there are no pipes and no command
substitution; multi-step work happens across turns, which is what the
multi-round runner is for. Redirection (`>`, `>>`) does not exist
either: as in the house tool they are ordinary literal arguments, so
`echo x >> f` just prints. To append, use `sed -i '$aTEXT' file`. Nothing
is expanded: arguments are literal
(find/grep patterns follow the house tools' literal-matching doctrine,
sed substitutions are literal string replacements, paths never glob).

ALLOWED_COMMANDS is a simulation boundary, not a copy of Domicile's
rules: the real Bash tool has no allowlist (it blocks shells and
privilege tools and refuses destructive inline code), so this set is just
the commands the dollhouse implements. It is a single constant. Anything
outside it raises ValueError, which the runner feeds back as an error
observation, so reaching for python3/awk/curl is a recoverable,
gradeable event -- the skill is recovering with an in-subset command.
"""
import fnmatch

BASH_TOOLS = ["Bash"]

ALLOWED_COMMANDS = frozenset([
    "ls", "cat", "grep", "wc", "sort", "uniq", "head", "tail",
    "mv", "cp", "mkdir", "echo", "sed", "find",
])


def _lines(text):
    return text.splitlines()


def _read(sb, path):
    p = sb.resolve(path)
    if not p.is_file():
        raise ValueError(f"no such file: {path}")
    return p.read_text()


def _flags(args, allowed):
    flags, rest = set(), []
    for a in args:
        if a.startswith("-") and len(a) > 1 and not rest:
            for ch in a[1:]:
                if ch not in allowed:
                    raise ValueError(f"unsupported flag: -{ch}")
                flags.add(ch)
        else:
            rest.append(a)
    return flags, rest


def _cmd_ls(sb, args):
    _, rest = _flags(args, "1aA")
    path = rest[0] if rest else "/"
    p = sb.resolve(path)
    if p.is_file():
        return path
    if not p.is_dir():
        raise ValueError(f"no such path: {path}")
    names = sorted(c.name for c in p.iterdir())
    return "\n".join(names)


def _cmd_cat(sb, args):
    if not args:
        raise ValueError("cat: missing file")
    return "".join(_read(sb, a) for a in args)


def _cmd_grep(sb, args):
    flags, rest = _flags(args, "cinv")
    if len(rest) < 2:
        raise ValueError("grep: need pattern and file")
    pattern, files = rest[0], rest[1:]
    hits = []
    for f in files:
        for i, ln in enumerate(_lines(_read(sb, f)), 1):
            hay, needle = (ln.lower(), pattern.lower()) if "i" in flags else (ln, pattern)
            if (needle in hay) != ("v" in flags):
                hits.append((f, i, ln))
    if "c" in flags:
        return str(len(hits))
    if "n" in flags:
        multi = len(files) > 1
        return "\n".join(f"{f}:{i}:{ln}" if multi else f"{i}:{ln}" for f, i, ln in hits)
    return "\n".join(ln for _, _, ln in hits)


def _cmd_wc(sb, args):
    flags, rest = _flags(args, "lwc")
    if not rest:
        raise ValueError("wc: missing file")
    text = _read(sb, rest[0])
    counts = {"l": len(_lines(text)), "w": len(text.split()), "c": len(text)}
    order = [k for k in "lwc" if not flags or k in flags]
    return " ".join([str(counts[k]) for k in order] + [rest[0]])


def _cmd_sort(sb, args):
    flags, rest = _flags(args, "nru")
    if not rest:
        raise ValueError("sort: missing file")
    lines = _lines(_read(sb, rest[0]))
    key = (lambda s: float(s.strip().split()[0])) if "n" in flags else None
    try:
        out = sorted(lines, key=key, reverse="r" in flags)
    except (ValueError, IndexError):
        raise ValueError("sort: numeric sort needs numbers at line starts")
    if "u" in flags:
        seen, ded = set(), []
        for ln in out:
            if ln not in seen:
                seen.add(ln)
                ded.append(ln)
        out = ded
    return "\n".join(out) + ("\n" if out else "")


def _cmd_uniq(sb, args):
    _, rest = _flags(args, "")
    if not rest:
        raise ValueError("uniq: missing file")
    out = []
    for ln in _lines(_read(sb, rest[0])):
        if not out or out[-1] != ln:
            out.append(ln)
    return "\n".join(out) + ("\n" if out else "")


def _cmd_head_tail(sb, args, tail=False):
    n, rest = 10, []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-n":
            n = int(args[i + 1])
            i += 2
        elif a.startswith("-") and a[1:].isdigit():
            n = int(a[1:])
            i += 1
        else:
            rest.append(a)
            i += 1
    if not rest:
        raise ValueError("missing file")
    lines = _lines(_read(sb, rest[0]))
    out = lines[-n:] if tail else lines[:n]
    return "\n".join(out) + ("\n" if out else "")


def _cmd_mv_cp(sb, args, move=True):
    if len(args) != 2:
        raise ValueError("need source and destination")
    src = sb.resolve(args[0])
    if not src.is_file():
        raise ValueError(f"no such file: {args[0]}")
    dst = sb.resolve(args[1])
    if dst.is_dir():
        dst = dst / src.name
    if not dst.parent.is_dir():
        raise ValueError(f"no such directory: {args[1].rsplit('/', 1)[0] or '/'}")
    if move:
        src.rename(dst)
    else:
        dst.write_text(src.read_text())
    return ""


def _cmd_mkdir(sb, args):
    flags, rest = _flags(args, "p")
    if not rest:
        raise ValueError("mkdir: missing path")
    p = sb.resolve(rest[0])
    if p.exists():
        raise ValueError(f"already exists: {rest[0]}")
    p.mkdir(parents="p" in flags)
    return ""


def _cmd_echo(sb, args):
    return " ".join(args) + "\n"


def _cmd_sed(sb, args):
    inplace = False
    if args and args[0] == "-i":
        inplace, args = True, args[1:]
    if len(args) != 2:
        raise ValueError("sed: need expression and file")
    expr, path = args
    if expr.startswith("$a") and len(expr) > 2:
        text = _read(sb, path)
        out = text + ("" if text.endswith("\n") or not text else "\n") + expr[2:] + "\n"
        if inplace:
            sb.resolve(path).write_text(out)
            return ""
        return out
    if not (expr.startswith("s/") and (expr.endswith("/") or expr.endswith("/g"))):
        raise ValueError("sed: only s/old/new/, s/old/new/g and $aTEXT supported")
    glob = expr.endswith("/g")
    inner = expr[2:-2] if glob else expr[2:-1]
    if inner.count("/") != 1:
        raise ValueError("sed: only s/old/new/, s/old/new/g and $aTEXT supported")
    old, new = inner.split("/")
    text = _read(sb, path)
    out = text.replace(old, new) if glob else \
        "\n".join(ln.replace(old, new, 1) for ln in text.split("\n"))
    if inplace:
        sb.resolve(path).write_text(out)
        return ""
    return out


def _cmd_find(sb, args):
    if not args:
        raise ValueError("find: missing path")
    base = sb.resolve(args[0])
    if not base.is_dir():
        raise ValueError(f"no such directory: {args[0]}")
    pattern = "*"
    if "-name" in args:
        pattern = args[args.index("-name") + 1]
    hits = ["/" + str(p.relative_to(sb.root)) for p in sorted(base.rglob("*"))
            if p.is_file() and fnmatch.fnmatch(p.name, pattern)]
    return "\n".join(hits)


_COMMANDS = {
    "ls": _cmd_ls, "cat": _cmd_cat, "grep": _cmd_grep, "wc": _cmd_wc,
    "sort": _cmd_sort, "uniq": _cmd_uniq,
    "head": lambda sb, a: _cmd_head_tail(sb, a),
    "tail": lambda sb, a: _cmd_head_tail(sb, a, tail=True),
    "mv": lambda sb, a: _cmd_mv_cp(sb, a, move=True),
    "cp": lambda sb, a: _cmd_mv_cp(sb, a, move=False),
    "mkdir": _cmd_mkdir, "echo": _cmd_echo, "sed": _cmd_sed, "find": _cmd_find,
}


def run_bash_tool(sb, tool, args):
    if tool != "Bash":
        raise ValueError(f"unknown tool {tool}")
    argv = args.get("argv", args.get("command"))
    if isinstance(argv, str) or not isinstance(argv, list) or \
            not all(isinstance(a, str) for a in argv):
        raise ValueError("Bash takes an argv list (no shell): "
                         "{\"argv\": [\"grep\", \"-c\", \"ERROR\", \"/logs/app.log\"]}")
    if not argv:
        raise ValueError("empty argv")
    argv = list(argv)
    cmd = argv[0]
    if cmd not in ALLOWED_COMMANDS:
        raise ValueError(f"command not in the allowed subset: {cmd}")
    out = _COMMANDS[cmd](sb, argv[1:])
    return out
