#!/usr/bin/env python3
"""Manage the agent's write sandbox. Run by the user; the agent cannot.

  python3 allow.py ~/projects/foo        add a writable directory to chat
  python3 allow.py --list                show the current writable roots
  python3 allow.py --remove ~/projects/foo
  python3 allow.py --profile safe        set chat's permission profile
  python3 allow.py --profile wakeup      safe, plus ToolWake (for unattended jobs)
  python3 allow.py --profile full        unrestricted (asks you to type "yes")
  python3 allow.py --role build ...      other roles (build is "full" by default)

Profiles: safe = writes only inside the roots, Bash in a sandbox with no
network. wakeup = safe + ToolWake (its jobs are sandboxed too, with network).
full = no restrictions at all. Roots only matter for safe and wakeup.

Edits data/roles.json. Every change is validated by the same rules the
sandbox enforces (no filesystem root, no whole home directory, nothing that
contains roles.json / config.json / .env).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import config_loader as CL
import sandbox


def current_entries(role: str, roles_path: Path) -> list[str]:
    """The role's configured roots as written (before a profile rewrites them),
    so adding a root to a role that is on 'full' keeps the defaults."""
    blk = _roles_from(roles_path).get(role) or {}
    entries = blk.get("write_roots")
    return list(entries) if entries else list(CL.DEFAULT_WRITE_ROOTS)


def _roles_from(path: Path) -> dict:
    from copy import deepcopy
    roles = deepcopy(CL.ROLE_DEFAULTS)
    for name, block in CL._read_json(path).items():
        if name.startswith("_") or not isinstance(block, dict):
            continue
        if isinstance(roles.get(name), dict):
            roles[name].update(block)
        else:
            roles[name] = block
    return roles


def _save(role: str, entries: list[str], roles_path: Path, base: Path) -> None:
    sandbox.resolve_roots(entries, base=base, protected=CL.sandbox_protected_files())
    raw = {}
    if roles_path.exists() and roles_path.stat().st_size > 0:
        try:
            raw = json.loads(roles_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"{roles_path} is not valid JSON ({exc}); fix it first, "
                             f"I won't overwrite it.") from exc
    raw.setdefault(role, {})["write_roots"] = entries
    CL._write_json_atomic(roles_path, raw)


def add(role: str, path: str, roles_path: Path, base: Path) -> str:
    absolute = str(Path(path).expanduser().resolve())
    entries = current_entries(role, roles_path)
    resolved = [str(r) for r in sandbox.resolve_roots(entries, base=base)]
    if absolute in resolved:
        return f"{absolute} is already writable for {role}."
    _save(role, entries + [absolute], roles_path, base)
    note = "" if Path(absolute).exists() else " (it doesn't exist yet; it will be created on first write)"
    return f"{role} may now write under {absolute}{note}."


def remove(role: str, path: str, roles_path: Path, base: Path) -> str:
    target = Path(path).expanduser()
    target = target.resolve() if target.is_absolute() else (base / target).resolve()
    entries = current_entries(role, roles_path)
    keep = [e for e in entries
            if (Path(e).expanduser() if Path(e).expanduser().is_absolute()
                else base / e).resolve() != target]
    if len(keep) == len(entries):
        return f"{target} is not in {role}'s writable roots."
    _save(role, keep, roles_path, base)
    return f"{role} can no longer write under {target}."


def set_profile(role: str, profile: str, roles_path: Path, confirm=None) -> str:
    if profile not in CL.PROFILES or profile == "none":
        raise ValueError(f"unknown profile {profile!r}; choose from "
                         f"{', '.join(p for p in CL.PROFILES if p != 'none')}.")
    if role not in _roles_from(roles_path):
        raise ValueError(f"no role named {role!r}.")
    if profile == "full":
        ok = confirm("full removes every restriction: the agent can write anywhere you "
                     "can, run any command, and use the network.\nType yes to continue: ") \
            if confirm else False
        if str(ok).strip().lower() != "yes":
            raise ValueError("not changed: the full profile needs you to type \"yes\".")
    raw = {}
    if roles_path.exists() and roles_path.stat().st_size > 0:
        try:
            raw = json.loads(roles_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"{roles_path} is not valid JSON ({exc}); fix it first, "
                             f"I won't overwrite it.") from exc
    blk = raw.setdefault(role, {})
    blk["profile"] = profile
    # Legacy keys from before profiles would be ignored anyway; remove the ones
    # that could mislead a reader of the file.
    blk.pop("tools", None)
    if blk.get("write_roots") is None:
        blk.pop("write_roots", None)
    CL._write_json_atomic(roles_path, raw)
    return f"{role} now runs under the {profile} profile."


def describe(role: str) -> str:
    prof = CL.profile_for_role(role)
    roots = CL.write_roots_for_role(role)
    if roots is None:
        return f"{role}: profile {prof} -- unrestricted"
    extra = "  (can launch sandboxed ToolWake jobs)" if getattr(roots, "toolwake", False) else ""
    return (f"{role}: profile {prof}{extra}\nmay write under:\n"
            + "\n".join(f"  {r}" for r in roots))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?")
    ap.add_argument("--role", default=None, help="default: chat (--list shows chat and wakeup)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--remove", metavar="PATH")
    ap.add_argument("--profile", metavar="NAME", help="safe | wakeup | full")
    a = ap.parse_args(argv)
    listing_all = a.role is None
    a.role = a.role or "chat"
    base = CL.ROOT
    try:
        if a.profile:
            conf = input if sys.stdin.isatty() else None
            print(set_profile(a.role, a.profile, CL.ROLES_PATH, confirm=conf)); return 0
        if a.list or (not a.path and not a.remove):
            print("\n\n".join(describe(r) for r in (("chat", "wakeup") if listing_all else (a.role,)))); return 0
        if a.remove:
            print(remove(a.role, a.remove, CL.ROLES_PATH, base)); return 0
        print(add(a.role, a.path, CL.ROLES_PATH, base)); return 0
    except (sandbox.SandboxConfigError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
