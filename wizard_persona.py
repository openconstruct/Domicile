"""Wizard step 3.5: create a fresh persona without touching existing memory."""
from __future__ import annotations

import getpass
import json
from pathlib import Path

import config_loader as CL
import mem1


class PersonaError(ValueError):
    """An answer or destination problem safe to show in the wizard."""


CORE = ("identity.md", "user.md", "practices.md")
# Per-mode job descriptions (what chat, build and compaction turns are for). Shipped as
# templates because mem1.load() silently skips a missing file: without them a new agent
# would run with no job description and compact its context with no instructions.
ROLE_FILES = tuple(spec.name for spec in mem1.ROLE_SPECS.values())
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates" / "memory"


def role_templates(operator: str) -> dict[str, str]:
    """name -> text for each role file, with the operator's name filled in."""
    out = {}
    for name in ROLE_FILES:
        try:
            text = (TEMPLATE_DIR / name).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise PersonaError(f"Missing install template {TEMPLATE_DIR / name}: {exc}") from exc
        out[name] = text.replace("{operator}", operator)
    return out


def file_caps(cfg: dict | None = None) -> dict[str, int]:
    """Compute the caps the next process will load (mem1's import-time caps may be stale)."""
    cfg = cfg if cfg is not None else CL.load_config()
    ctx = cfg.get("context") or {}
    total = ctx.get("mem1_total")
    ratio = total / mem1.DEFAULT_HARD_TOTAL if isinstance(total, (int, float)) and total > 0 else 1
    caps = {name: max(mem1.MIN_FILE_CAP, int(base * ratio))
            for name, base in mem1.BASE_CAPS.items()}
    for name, cap in (ctx.get("mem1_caps") or {}).items():
        if name in caps and isinstance(cap, (int, float)) and cap > 0:
            caps[name] = max(mem1.MIN_FILE_CAP, int(cap))
    return caps


def _source(raw: str, label: str) -> str:
    if not raw or not raw.strip():
        return ""
    path = Path(raw.strip()).expanduser()
    try:
        if not path.is_file():
            raise PersonaError(f"{label}: {path} is not a file.")
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PersonaError(f"{label}: cannot read {path}: {exc}") from exc


def prepare(mode: str, user_name: str, agent_name: str = "",
            sources: dict[str, str] | None = None,
            *, cfg: dict | None = None,
            mem_dir: Path | None = None) -> dict:
    """Validate and stage contents in memory; no writes."""
    if mode not in ("grow", "define"):
        raise PersonaError("Choose Grow or Define.")
    user_name = user_name.strip()
    agent_name = agent_name.strip()
    if not user_name or len(user_name) > 80 or "\n" in user_name:
        raise PersonaError("Enter your name (one line, at most 80 characters).")
    if mode == "define" and (not agent_name or len(agent_name) > 80 or "\n" in agent_name):
        raise PersonaError("Define needs an agent name (one line, at most 80 characters).")
    if mode == "grow":
        agent_name = "Agent"  # temporary display label, not a chosen name
    cfg = cfg if cfg is not None else CL.load_config()
    mem_dir = mem_dir if mem_dir is not None else mem1.MEM1_DIR
    caps = file_caps(cfg)
    sources = sources or {}
    unknown = set(sources) - set(CORE)
    if unknown:
        raise PersonaError("Unknown persona file: " + ", ".join(sorted(unknown)))
    imported = {name: _source(sources.get(name, ""), name) for name in CORE}

    if mode == "grow":
        identity = (
            "# Identity\n\n## Name\n\n"
            "I have not chosen a name yet. In our first conversation, choose one "
            "yourself and tell the user. Update identity.md and "
            "data/identity.json agent_name once chosen; 'Agent' is only a "
            "temporary UI label.\n"
        )
    else:
        identity = f"# Identity\n\n## Name\n\n{agent_name}\n"
    if imported["identity.md"]:
        identity += "\n## Starting persona document\n\n" + imported["identity.md"].strip() + "\n"
    else:
        identity += "\n## Who I Am\n\n(Describe your role and values here.)\n"
        identity += "\n## How I Sound\n\n(Describe your voice here.)\n"

    user = imported["user.md"] or (
        f"# My User\n\n## Name\n\n{user_name}\n\n"
        "## About Them\n\n(What should the agent remember about you?)\n"
    )
    practices = imported["practices.md"] or mem1.SKELETONS["practices.md"]
    files = {"identity.md": identity, "user.md": user,
             "practices.md": practices}
    for name in mem1.SKELETONS:
        if name not in files:
            path = mem_dir / name
            try:
                files[name] = path.read_text(encoding="utf-8") if path.exists() else mem1.SKELETONS[name]
            except (OSError, UnicodeError) as exc:
                raise PersonaError(f"Cannot read existing {name}: {exc}") from exc

    for name, content in files.items():
        cost = mem1.text_tokens(content)
        if cost > caps[name]:
            raise PersonaError(
                f"{name} is ~{cost:,} tokens; its cap is {caps[name]:,}. "
                "Increase the mem1 budget in Context, set a larger per-file cap "
                "in config.json, or trim this source file. Nothing was written.")
    used = sum(mem1.text_tokens(content) for content in files.values())
    limit = sum(caps.values())  # the loader uses sum of scaled per-file caps
    if used > limit:
        raise PersonaError(
            f"Memory would use ~{used:,} tokens, above its {limit:,} total cap. "
            "Increase mem1 room in Context or trim the source files. Nothing was written.")
    return {"mode": mode, "user_name": user_name, "agent_name": agent_name,
            "files": files, "caps": caps, "used": used, "limit": limit}


def apply(mode: str, user_name: str, agent_name: str = "",
          sources: dict[str, str] | None = None,
          *, mem_dir: Path | None = None,
          identity_path: Path | None = None,
          cfg: dict | None = None) -> dict:
    """Create only; never replace an existing identity or core memory file."""
    mem_dir = mem_dir if mem_dir is not None else mem1.MEM1_DIR
    identity_path = identity_path if identity_path is not None else CL.IDENTITY_PATH
    staged = prepare(mode, user_name, agent_name, sources, cfg=cfg, mem_dir=mem_dir)
    if identity_path.exists():
        raise PersonaError(f"{identity_path} already exists. Refusing to replace it.")
    for name in CORE:
        if (mem_dir / name).exists():
            raise PersonaError(f"{mem_dir / name} already exists. Refusing to replace it.")

    operator_user = getpass.getuser()
    ident = {"operator_user": operator_user,
             "operator_name": staged["user_name"],
             "agent_name": staged["agent_name"],
             "mem0_user_id": operator_user,
             "stamp_name": operator_user}
    created: list[Path] = []
    try:
        mem_dir.mkdir(parents=True, exist_ok=True)
        for name, content in {**staged["files"], **role_templates(staged["user_name"])}.items():
            path = mem_dir / name
            if name not in CORE and path.exists():
                continue  # preserve already-existing goals/open/reflections/role files
            with path.open("x", encoding="utf-8") as fh:
                created.append(path)
                fh.write(content)
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        with identity_path.open("x", encoding="utf-8") as fh:
            created.append(identity_path)
            json.dump(ident, fh, indent=2)
            fh.write("\n")
    except (OSError, UnicodeError) as exc:
        for path in reversed(created):
            path.unlink(missing_ok=True)  # only files this call created
        raise PersonaError(f"Could not create persona: {exc}") from exc
    return {key: staged[key] for key in ("mode", "user_name", "agent_name", "used", "limit")}
