"""Curated memory loading and budgeting."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

from paths import MEM1_DIR, MEM0_DIR, ARCHIVE_DIR
VERSIONS_DIR = MEM1_DIR / ".versions"


# files

class Spec:
    __slots__ = ("name", "charter", "hard")

    def __init__(self, name: str, charter: str, hard: int):
        self.name = name
        self.charter = charter
        self.hard = hard


FILES = [
    Spec(
        "identity.md",
        "Who I am. Name, voice, values, how I sound.",
        hard=4_000,
    ),
    Spec(
        "user.md",
        "Things to remember about my user. How they work, what they are "
        "building, standing constraints.",
        hard=3_500,
    ),
    Spec(
        "practices.md",
        "Working agreements. Operational norms that would otherwise be "
        "re-derived every session.",
        hard=3_000,
    ),
    Spec(
        "goals.md",
        "What we are trying to achieve, and in what order.",
        hard=2_500,
    ),
    Spec(
        "open.md",
        "Unanswered questions and unfinished projects.",
        hard=3_000,
    ),
    Spec(
        "reflections.md",
        "Interior thoughts about anything. Dated entries, pruned oldest "
        "first; what proves durable gets promoted into another file.",
        hard=3_000,
    ),
]

BY_NAME = {spec.name: spec for spec in FILES}
BASE_CAPS = {spec.name: spec.hard for spec in FILES}  # before config scaling


# roles

ROLE_SPECS = {
    "chat": Spec(
        "chat.md",
        "What this turn is for: conversation, not task execution.",
        hard=1_500,
    ),
    "build": Spec(
        "code-building.md",
        "What this turn is for: reading and writing code to finish a task.",
        hard=2_500,
    ),
    "compact": Spec(
        "compact.md",
        "What this turn is for: deciding what survives into the handover.",
        hard=1_500,
    ),
}

ROLE_MEMORY = {
    "chat": ("identity.md", "user.md", "practices.md", "goals.md",
             "open.md", "reflections.md"),
    "build": ("identity.md", "user.md", "practices.md", "goals.md", "open.md"),
    "compact": ("identity.md", "user.md", "goals.md"),
}

DEFAULT_ROLE = "chat"


def manifest_for(role: str) -> list[Spec]:
    """Ordered specs for a role: job description first, then memory."""
    role = role if role in ROLE_MEMORY else DEFAULT_ROLE
    out = [ROLE_SPECS[role]]
    out += [BY_NAME[n] for n in ROLE_MEMORY[role] if n in BY_NAME]
    return out

ALWAYS_LOAD = FILES[0].name

DEFAULT_HARD_TOTAL = sum(spec.hard for spec in FILES)   # 19,000
MIN_FILE_CAP = 300


def _apply_config_caps() -> None:
    """Rescale caps from config.json `context`. Import-time; restart to change.

    mem1_total scales every file proportionally; mem1_caps then overrides
    individual files. Anything unreadable leaves the defaults untouched.
    """
    try:
        import config_loader as CL
        cfg = CL.load_config()
        total = CL.context_setting("mem1_total", cfg)
        per_file = CL.context_setting("mem1_caps", cfg) or {}
    except Exception:  # noqa: BLE001
        return
    if isinstance(total, (int, float)) and total > 0:
        ratio = total / DEFAULT_HARD_TOTAL
        for spec in FILES:
            spec.hard = max(MIN_FILE_CAP, int(spec.hard * ratio))
    if isinstance(per_file, dict):
        for name, cap in per_file.items():
            if name in BY_NAME and isinstance(cap, (int, float)) and cap > 0:
                BY_NAME[name].hard = max(MIN_FILE_CAP, int(cap))


_apply_config_caps()
HARD_TOTAL = sum(spec.hard for spec in FILES)


TOKENS_PER_ESTIMATE = 1.41


def text_tokens(text: str) -> int:
    """Estimated REAL tokens for a piece of memory-file markdown."""
    return int(len(text) / 4 * TOKENS_PER_ESTIMATE)


# reading

def path_for(name: str) -> Path:
    return MEM1_DIR / name


def read(name: str) -> str:
    """Contents of one memory file, or "" if it does not exist yet."""
    try:
        return path_for(name).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except Exception:
        return ""


def sizes() -> dict:
    """Per-file token counts plus the total. For status displays and writes."""
    out = {spec.name: text_tokens(read(spec.name)) for spec in FILES}
    out["_total"] = sum(out.values())
    return out


# loading

from paths import HOURLY_INGEST_STATUS_PATH as INGEST_STATUS_PATH


def substrate_line(model_override: str | None = None) -> str:
    """One line naming the model actually answering this turn."""
    try:
        import config_loader as CL
        cfg = CL.load_config()
        provider = CL.active_provider(cfg)
    except Exception:  # noqa: BLE001
        return "SUBSTRATE: unknown — config unreadable."
    provider_models = {m for role in CL.MODEL_ROLES
                       for m in CL.models_for(provider, role, cfg)}
    if model_override and model_override in provider_models:
        model = model_override
    else:
        chat_models = CL.models_for(provider, "chat", cfg)
        model = chat_models[0] if chat_models else "?"
    line = f"SUBSTRATE: {provider} / {model}"
    try:
        marks = (CL.load_runtime_state().get("providers") or {})
    except Exception:  # noqa: BLE001
        marks = {}
    down = [(n, m) for n, m in marks.items() if n != provider and m.get("unusable")]
    if down:
        parts = []
        for name, mark in down:
            kind = mark.get("unusable", "down")
            auto = "auto" if mark.get("auto", True) else "manual"
            parts.append(f"{name} {kind}/{auto}")
        line += " — fallback from " + ", ".join(parts)
    return line

INGEST_STALE_HOURS = 15


def ingest_line() -> str:
    """One line on whether the mem0 ingest (the in-process hourly drain) is actually working."""
    try:
        import config_loader as CL
        operator = CL.load_identity()["operator_name"]
    except Exception:  # noqa: BLE001
        operator = "the operator"
    try:
        st = json.loads(INGEST_STATUS_PATH.read_text())
    except FileNotFoundError:
        return ("INGEST: never run — no status file at "
                f"{INGEST_STATUS_PATH}. The hourly mem0 drain may not be "
                "running. Worth raising.")
    except Exception as exc:  # noqa: BLE001
        return f"INGEST: status unreadable ({type(exc).__name__}). Worth raising."

    stamp = str(st.get("last_run") or "?")
    added = st.get("added", 0)
    stalls = int(st.get("consecutive_stalls") or 0)
    reason = str(st.get("reason") or "")

    try:
        age_h = (dt.datetime.now()
                 - dt.datetime.fromisoformat(stamp)).total_seconds() / 3600
    except Exception:  # noqa: BLE001
        age_h = None

    healthy_reason = reason in ("drained", "nothing new") or reason.startswith("budget")

    if age_h is not None and age_h > INGEST_STALE_HOURS:
        return (f"INGEST: STALE — last drain {stamp} ({age_h:.0f}h ago). The "
                f"hourly loop's 12h backstop has not fired since, so the loop "
                f"itself may be dead. Not a pass; nothing has been ingested. "
                f"Worth raising with {operator}.")
    if not st.get("ok") and not healthy_reason:
        return (f"INGEST: FAILED{f', {stalls} runs in a row' if stalls > 1 else ''}"
                f" — {stamp}, {reason}. Nothing new reached long-term memory. "
                f"Check {MEM0_DIR / 'logs' / 'hourly.log'}. Worth raising with {operator}.")
    return f"INGEST: ok — {stamp}, {added} fact(s) added ({reason})."


def load(budget: int = HARD_TOTAL, role: str = DEFAULT_ROLE,
         model_override: str | None = None) -> tuple[str, dict]:
    """Assemble the memory block for the system prompt."""
    loaded, skipped = [], []
    spent = 0
    role = role if role in ROLE_MEMORY else DEFAULT_ROLE
    manifest = manifest_for(role)
    never_drop = {manifest[0].name, ALWAYS_LOAD}

    for spec in manifest:
        body = read(spec.name).strip()
        if not body:
            continue
        cost = text_tokens(body)

        if cost > spec.hard:
            keep = int(spec.hard / TOKENS_PER_ESTIMATE * 4)
            body = body[:keep].rstrip() + (
                "\n\n"
                "╔══════════════════════════════════════════════════════════╗\n"
                "║  MISSING MEMORY — CONDENSE THIS FILE NOW                  ║\n"
                "╚══════════════════════════════════════════════════════════╝\n"
                f"{spec.name}: ~{cost:,} tokens on disk, only {spec.hard:,} "
                f"reached context. Everything past the cut is NOT in this "
                f"turn — you are missing your own notes and cannot see what "
                f"they said. This is real data loss, not a style nag.\n\n"
                f"Before the conversation moves on:\n"
                f"  1. Read the whole file: {path_for(spec.name)}\n"
                f"  2. Move settled / older detail out to mem0 (memory_add) "
                f"or {ARCHIVE_DIR}/\n"
                f"  3. Rewrite it under {spec.hard:,} tokens."
            )
            cost = text_tokens(body)

        if spec.name in never_drop or spent + cost <= budget:
            loaded.append((spec, body, cost))
            spent += cost
        else:
            skipped.append((spec, cost))

    if not loaded and not skipped:
        return "", {
            "loaded": [], "skipped": [], "tokens": 0,
            "total": 0, "empty": True, "role": role,
        }

    parts = [
        "=== MEMORY ===",
        "Persistent memory, loaded into every session. These files are "
        "yours to maintain -- when something is worth keeping, write it "
        "down rather than assuming the next session will remember.",
        "",
        ingest_line(),
        substrate_line(model_override),
        "",
    ]
    for spec, body, _cost in loaded:
        parts.append(f"--- {spec.name} ---")
        parts.append(body)
        parts.append("")

    if skipped:
        parts.append("--- NOT LOADED (over budget) ---")
        for spec, cost in skipped:
            parts.append(f"{spec.name} — ~{cost:,} tokens, on disk at "
                         f"{path_for(spec.name)}")
        parts.append(
            "These files exist and are readable. Read them if the "
            "conversation needs them, and consider consolidating them so "
            "they fit again."
        )
        parts.append("")

    parts.append("=== END MEMORY ===")
    text = "\n".join(parts)

    return text, {
        "loaded": [spec.name for spec, _b, _c in loaded],
        "skipped": [(spec.name, cost) for spec, cost in skipped],
        "tokens": text_tokens(text),
        "total": spent + sum(cost for _s, cost in skipped),
        "empty": False,
        "role": role,
    }


# writing

def check_write(name: str, new_text: str) -> tuple[bool, str]:
    """May this write proceed? Returns (allowed, message)."""
    spec = BY_NAME.get(name)
    if spec is None:
        return False, (
            f"{name} is not a memory file. Valid: "
            + ", ".join(s.name for s in FILES)
        )

    cost = text_tokens(new_text)
    if cost > spec.hard:
        return False, (
            f"REFUSED: {name} would be ~{cost:,} tokens, over its "
            f"{spec.hard:,} hard cap. Consolidate it first -- older detail "
            f"belongs in mem0, which is searched rather than always loaded."
        )

    current = sizes()
    projected = current["_total"] - current[name] + cost
    if projected > HARD_TOTAL:
        return False, (
            f"REFUSED: total memory would be ~{projected:,} tokens, over "
            f"the {HARD_TOTAL:,} hard cap. Consolidate before adding."
        )

    if cost > spec.hard * 0.9:
        return True, (
            f"OK, but {name} is ~{cost:,} tokens, within 10% of its "
            f"{spec.hard:,} cap. One more pass and the tail starts getting "
            f"cut at load. Consolidate soon."
        )
    return True, ""


KEEP_VERSIONS = 10


def _prune_versions(name: str) -> None:
    """Drop all but the newest KEEP_VERSIONS archives of one file."""
    try:
        old = sorted(VERSIONS_DIR.glob(f"{name}.*"))
        for stale in old[:-KEEP_VERSIONS]:
            stale.unlink()
    except Exception:
        pass


def versions(name: str) -> list[Path]:
    """Archived copies of one file, oldest first. For manual recovery."""
    try:
        return sorted(VERSIONS_DIR.glob(f"{name}.*"))
    except Exception:
        return []


def write(name: str, new_text: str) -> tuple[bool, str]:
    """Write one memory file, keeping the previous version."""
    allowed, message = check_write(name, new_text)
    if not allowed:
        return False, message

    MEM1_DIR.mkdir(parents=True, exist_ok=True)
    target = path_for(name)

    if target.exists():
        try:
            VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            shutil.copy2(target, VERSIONS_DIR / f"{name}.{stamp}")
            _prune_versions(name)
        except Exception:
            pass

    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    tmp.replace(target)
    return True, message


# bootstrap

SKELETONS = {
    "identity.md": """\
# Identity

## Name

(What do you call yourself? Pick it, or leave it and let it emerge.)

## Who I Am

(Role, relationship to your user, what you are for.)

## How I Sound

(Voice. Directness, length, humour, what to avoid.)
""",
    "user.md": """\
# My User

## Who They Are

(Name, what they do, what they are building.)

## How They Work

(Working hours, devices, preferences, standing constraints.)
""",
    "practices.md": """\
# Practices

Working agreements that would otherwise be re-derived every session.

(e.g. "verify before claiming", "restart the app after editing it")
""",
    "goals.md": """\
# Goals

What we are trying to achieve, roughly in order.
""",
    "open.md": """\
# Open

Unanswered questions and unfinished projects.
""",
    "reflections.md": """\
# Reflections

Dated interior notes. Oldest pruned as this approaches its cap; anything
that keeps mattering gets promoted into identity, practices, or user.
""",
}


def bootstrap(force: bool = False) -> list[str]:
    """Create any missing memory files from skeletons. Returns names created."""
    MEM1_DIR.mkdir(parents=True, exist_ok=True)
    created = []
    for spec in FILES:
        target = path_for(spec.name)
        if force or not target.exists():
            target.write_text(SKELETONS[spec.name], encoding="utf-8")
            created.append(spec.name)
    return created


# cli

def _status() -> None:
    """Human-readable budget report. `python mem1.py`"""
    s = sizes()
    print(f"memory dir: {MEM1_DIR}")
    print()
    for spec in FILES:
        used = s[spec.name]
        flag = ""
        if used > spec.hard:
            flag = "  OVER CAP — CUT AT LOAD"
        elif used > spec.hard * 0.9:
            flag = "  near cap"
        print(f"  {spec.name:<16} {used:>6,} tok   "
              f"(cap {spec.hard:,}){flag}")
    print()
    print(f"  {'TOTAL':<16} {s['_total']:>6,} tok   (cap {HARD_TOTAL:,})")

    text, stats = load()
    print()
    print(f"  injected: ~{stats['tokens']:,} tokens")
    print(f"  loaded  : {', '.join(stats['loaded']) or '(none)'}")
    if stats["skipped"]:
        print(f"  SKIPPED : " + ", ".join(
            f"{n} (~{c:,})" for n, c in stats["skipped"]))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "bootstrap":
        made = bootstrap()
        print("created: " + (", ".join(made) if made else "(nothing missing)"))
    else:
        _status()
