"""Wizard step 3.7: offline smoke test. No model calls, no spend, no writes.

Each check returns (name, ok, detail). A live provider ping is deliberately not
here: it costs money and needs per-adapter code; the first real conversation is
that test.
"""
from __future__ import annotations

from pathlib import Path

import config_loader as CL
import mem1
import wizard_persona as persona
import wizard_service as service


def run(mem_dir: Path | None = None, identity_path: Path | None = None,
        unit_dir: Path | None = None, cfg: dict | None = None,
        service_skipped: bool = False) -> list[tuple[str, bool, str]]:
    mem_dir = mem_dir if mem_dir is not None else mem1.MEM1_DIR
    identity_path = identity_path if identity_path is not None else CL.IDENTITY_PATH
    unit_dir = unit_dir if unit_dir is not None else service.UNIT_DIR
    out: list[tuple[str, bool, str]] = []

    def add(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    try:
        cfg = cfg if cfg is not None else CL.load_config()
        add("config loads", True)
    except Exception as e:  # any parse/IO failure is the finding
        add("config loads", False, str(e))
        return out
    try:
        name = CL.active_provider(cfg)
        add("active provider set", bool(name), name or "none selected")
        if name:
            ok = CL.is_cli_provider(name, cfg) or CL.key_available(name, cfg)
            if name == "anthropic":
                import shutil
                add("claude CLI installed", bool(shutil.which("claude")),
                    "found" if shutil.which("claude") else "missing: install Claude Code and run `claude` once to log in")
            add("provider credentials", ok,
                "CLI provider (uses its own login)" if CL.is_cli_provider(name, cfg)
                else ("key found" if ok else f"no API key found for {name}"))
    except Exception as e:
        add("active provider set", False, str(e))

    ident_ok = identity_path.is_file()
    add("identity.json", ident_ok, str(identity_path))
    caps = persona.file_caps(cfg)
    used = 0
    for f in persona.CORE:
        p = mem_dir / f
        if not p.is_file():
            add(f, False, "missing")
            continue
        try:
            cost = mem1.text_tokens(p.read_text(encoding="utf-8"))
        except (OSError, UnicodeError) as e:
            add(f, False, str(e))
            continue
        used += cost
        add(f, cost <= caps[f], f"~{cost:,} / {caps[f]:,} tokens")
    add("mem1 total fits", used <= sum(caps.values()), f"~{used:,} / {sum(caps.values()):,}")

    missing = [n for n in ("cchat.service", "cchat-watchdog.service", "cchat-watchdog.timer")
               if not (unit_dir / n).exists()]
    if missing and service_skipped:
        add("service units", True, "skipped - you chose to run cchat by hand")
    else:
        add("service units", not missing,
            "installed" if not missing else "not installed: " + ", ".join(missing))
    return out


def format_report(results) -> str:
    lines = [f"[{'green' if ok else 'red'}]{'PASS' if ok else 'FAIL'}[/] {n}"
             + (f" - {d}" if d else "") for n, ok, d in results]
    bad = sum(1 for _, ok, _ in results if not ok)
    lines.append("\nAll checks passed." if not bad else f"\n{bad} check(s) failed.")
    return "\n".join(lines)
