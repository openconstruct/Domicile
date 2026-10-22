"""Write sandbox: which directories the agent may modify in a restricted mode.

Pure functions, no I/O beyond path resolution, so the policy is testable on
its own. local_tools.py enforces it in Write and Edit; the Bash tier (bwrap)
mounts the same roots writable.

Rules:
  * Roots are resolved (symlinks followed) once, here. Candidate paths are
    resolved by the caller before the check, so a symlink inside a root that
    points outside resolves to the outside path and is refused.
  * A root may not be "/" or the user's home directory itself -- either one
    makes the sandbox meaningless.
  * Relative root entries are taken relative to `base` (the install dir).
"""

from __future__ import annotations

from pathlib import Path


class SandboxConfigError(ValueError):
    pass


def resolve_roots(entries, base: Path, home: Path | None = None,
                  protected=()) -> list[Path]:
    """Resolve configured entries to absolute roots.

    `protected` lists files that define the sandbox itself (roles.json,
    config.json, .env). A root that is, or contains, any of them is rejected:
    otherwise the agent could rewrite its own sandbox with Write.
    """
    home = (home or Path.home()).resolve()
    prot = [Path(x).resolve() for x in protected]
    out: list[Path] = []
    for raw in entries or []:
        s = str(raw).strip()
        if not s:
            continue
        p = Path(s).expanduser()
        p = (p if p.is_absolute() else Path(base) / p).resolve()
        if p == Path(p.anchor) or p == home:
            raise SandboxConfigError(
                f"sandbox root {raw!r} resolves to {p}, which is too broad "
                f"(the filesystem root or your whole home directory)."
            )
        for f in prot:
            if f == p or p in f.parents:
                raise SandboxConfigError(
                    f"sandbox root {raw!r} ({p}) contains {f}, which defines the "
                    f"sandbox itself. Pick a narrower directory."
                )
        if p not in out:
            out.append(p)
    return out


def is_within(resolved: Path, roots: list[Path]) -> bool:
    for r in roots:
        try:
            resolved.relative_to(r)
            return True
        except ValueError:
            continue
    return False


def refusal(resolved: Path, roots: list[Path]) -> str:
    where = ", ".join(str(r) for r in roots) or "(none configured)"
    return (
        f"refused: {resolved} is outside this mode's write sandbox. "
        f"Writable here: {where}. Tell the user what you wanted to write and "
        f"why; they can allow it by running `python3 allow.py <path>` (the agent cannot do this itself)."
    )


# --- Bash confinement (bubblewrap) -------------------------------------------
#
# Layout: whole filesystem read-only, private /tmp and /run/user (the latter
# hides the user's systemd/dbus sockets, which would otherwise let a command ask
# systemd to run something outside the sandbox), only the roots writable, no
# network, own PID/IPC namespace, dies with its parent. Mount ORDER matters: the
# tmpfs mounts come before the binds so a root that lives under /tmp survives.
# There is deliberately no unsandboxed fallback anywhere in this module.

import shutil
import subprocess


class SandboxUnavailable(RuntimeError):
    pass


def bwrap_argv(argv, roots, cwd, *, network: bool = False, mask=(),
               die_with_parent: bool = True) -> list[str]:
    exe = shutil.which("bwrap")
    if not exe:
        raise SandboxUnavailable(
            "Bash is unavailable in this mode: bubblewrap is not installed "
            "(sudo apt install bubblewrap). It is never run unsandboxed instead.")
    cmd = [exe, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
           "--tmpfs", "/tmp", "--tmpfs", "/run/user"]
    for r in roots:
        Path(r).mkdir(parents=True, exist_ok=True)   # bind sources must exist
        cmd += ["--bind", str(r), str(r)]
    for f in mask:                                   # unreadable inside (e.g. .env)
        if Path(f).exists():
            cmd += ["--ro-bind", "/dev/null", str(f)]
    cmd += ["--unshare-pid", "--unshare-ipc", "--new-session"]
    if die_with_parent:        # off only for ToolWake jobs, which must outlive a restart
        cmd.append("--die-with-parent")
    if not network:
        cmd.append("--unshare-net")
    cmd += ["--chdir", str(cwd), "--"]
    return cmd + [str(a) for a in argv]


def bwrap_probe() -> tuple[bool, str]:
    """Can this machine actually create the sandbox? (WSL2/containers vary.)"""
    exe = shutil.which("bwrap")
    if not exe:
        return False, "bubblewrap is not installed"
    try:
        p = subprocess.run([exe, "--ro-bind", "/", "/", "--unshare-pid", "--", "true"],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"bwrap failed to run: {exc}"
    if p.returncode != 0:
        return False, (p.stderr or "bwrap exited non-zero").strip()[:300]
    return True, "ok"
