"""Wizard step 3.3 / 3.3a: provider + context window.

Pure logic, no UI. The Textual wizard collects answers and calls these;
nothing here prompts or prints. Writes go through credentials.save() (.env)
and config_loader.save_config() (config.json) -- the same paths cchat reads.

Run tests: python3 _test_runner.py _wizard_provider_test.py
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

import config_loader as CL
import credentials

# Presets for the "pick one" list. "api" is the generic OpenAI-compatible case
# and needs a base_url from the user. Model names are NOT preset: they change
# faster than this file, so the user supplies them (blank -> error).
KINDS = ("anthropic", "openai", "api")
# "openai" is ChatGPT-subscription login through the installed `codex` CLI
# (like "anthropic" through `claude`): no API key, no base_url. Users with an
# OpenAI *API key* use the generic "api" kind with base_url api.openai.com/v1.
OPENAI_BASE_URL = "https://api.openai.com/v1"  # hint for the "api" kind

_NAME_RE = re.compile(r"[a-z][a-z0-9_-]{0,31}")


class WizardError(ValueError):
    """A user-correctable answer problem; message is safe to show verbatim."""


def clean_name(raw: str) -> str:
    name = (raw or "").strip().lower()
    if not _NAME_RE.fullmatch(name):
        raise WizardError("Name must be 1-32 chars: lowercase letters, digits, - or _, "
                          "starting with a letter.")
    return name


def check_base_url(raw: str) -> str:
    url = (raw or "").strip().rstrip("/")
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.netloc:
        raise WizardError("base_url must look like https://host/v1")
    return url


def build_provider(kind: str, name: str | None = None, *, base_url: str = "",
                   models: str | dict | None = None, api_key: str = "",
                   vision: bool = False) -> dict:
    """Validate answers and return {name, block, env} without writing anything.

    `models` is one model id (used for chat/code/compact) or a dict with
    those role keys. anthropic uses the CLI adapter: no key, no models.
    openai uses the codex CLI adapter (`codex login`): no key; models are
    optional (blank -> codex's own default).
    """
    if kind not in KINDS:
        raise WizardError(f"Unknown provider kind {kind!r}.")

    if kind == "anthropic":
        name = clean_name(name or "anthropic")
        block = dict(CL.DEFAULTS["providers"]["anthropic"])
        block["models"] = {r: list(m) for r, m in block["models"].items()}
        return {"name": name, "block": block, "env": {}}

    if kind == "openai":
        name = clean_name(name or "codex")
        if isinstance(models, str):
            models = {r: models for r in CL.MODEL_ROLES}
        block = {"adapter": CL.CODEX_ADAPTER, "vision": bool(vision)}
        given = {r: str(m).strip() for r, m in (models or {}).items()
                 if str(m or "").strip()}
        if given:
            block["models"] = {r: [given.get(r) or next(iter(given.values()))]
                               for r in CL.MODEL_ROLES}
        return {"name": name, "block": block, "env": {}}

    name = clean_name(name or kind)
    url = check_base_url(base_url)

    if isinstance(models, str):
        models = {r: models for r in CL.MODEL_ROLES}
    models = {r: (models or {}).get(r, "") for r in CL.MODEL_ROLES}
    if not all(str(m).strip() for m in models.values()):
        raise WizardError("A model id is needed for chat, code and compact "
                          "(one id is fine; it is reused for all three).")

    key = (api_key or "").strip()
    if not key:
        raise WizardError("An API key is needed for this provider.")

    key_env = credentials.key_name(provider=name)
    block = {
        "adapter": "openai_compatible",
        "base_url": url,
        "vision": bool(vision),
        "models": {r: [str(m).strip()] for r, m in models.items()},
        "key_env": key_env,
    }
    return {"name": name, "block": block, "env": {key_env: key}}


def apply_provider(result: dict, *, make_active: bool = False) -> None:
    """Write a build_provider() result. Key first, so config never points at a
    key that was not saved. The new provider joins the rollover list."""
    if result["env"]:
        credentials.save(result["env"])
    name = result["name"]
    cfg = CL._read_json(CL.CONFIG_PATH)
    order = list(cfg.get("provider_order") or [])
    if name not in order:
        order.append(name)
    patch = {"providers": {name: result["block"]}, "provider_order": order}
    if make_active or not cfg.get("active_provider"):
        patch["active_provider"] = name
    CL.save_config(patch)


# --- 3.3a context window -------------------------------------------------
# Heuristic: ceiling ~ half the window, clamped to [floor, 150k];
# mem1 ~ 15% of the window, capped at the 19k default.
CEILING_CAP = 150_000
OUTPUT_MARGIN = 8_000   # room left for the reply itself
FLOOR_RATIO = 0.55      # derived floor as a share of the ceiling


def custom_context(ceiling: int, mem1_total: int | None = None,
                   floor: int | None = None) -> dict:
    """Advanced: user-chosen values instead of the derived heuristic.

    ceiling = history size that triggers a compaction; floor = size of the
    compacted text (summary + older tail) right after one (None leaves the
    default). The floor can't go under the summary + a minimal tail, and the
    ceiling must clear the floor by the headroom, or every turn would
    re-compact. mem1_total None
    leaves the default. Returns {context, fits=None, warning=""}.
    """
    import context_diet as cd
    try:
        ceiling = int(ceiling)
        mem1 = None if mem1_total in (None, "") else int(mem1_total)
        fl = None if floor in (None, "") else int(floor)
    except (TypeError, ValueError):
        raise WizardError("Ceiling, floor and memory budget must be whole "
                          "numbers of tokens.")
    if fl is not None and fl < cd.MIN_COMPACT_FLOOR:
        raise WizardError(f"Floor must be at least {cd.MIN_COMPACT_FLOOR:,} tokens: "
                          "the summary and a minimal tail alone take that much.")
    need = cd.min_ceiling(fl)
    if ceiling < need:
        raise WizardError(f"Ceiling must be at least {need:,} tokens "
                          f"(floor + {cd.COMPACT_HEADROOM:,}); lower and "
                          "every turn would trigger a compaction.")
    if mem1 is not None and mem1 < 1_000:
        raise WizardError("Memory budget under 1,000 tokens leaves nothing to load.")
    ctx = {"compact_ceiling_tokens": ceiling}
    if fl is not None:
        ctx["compact_floor_tokens"] = fl
    if mem1 is not None:
        ctx["mem1_total"] = mem1
    return {"context": ctx, "fits": None, "warning": ""}


def derive_context(window: int) -> dict:
    """Context settings for a model window. Returns {context, fits, warning}."""
    import context_diet
    try:
        window = int(window)
    except (TypeError, ValueError):
        raise WizardError("Context window must be a whole number of tokens.")
    if window < 8_000:
        raise WizardError("Under 8k tokens is too small to run an agent in.")

    floor = context_diet.MIN_COMPACT_CEILING
    mem1 = min(19_000, max(3_000, int(window * 0.15)))
    ceiling = max(floor, min(CEILING_CAP, int(window * 0.5)))
    need = mem1 + ceiling + OUTPUT_MARGIN
    fits = need <= window
    warning = ""
    if not fits:
        warning = (f"This window ({window:,}) is tight: memory {mem1:,} + history "
                   f"ceiling {ceiling:,} + reply room {OUTPUT_MARGIN:,} = {need:,}. "
                   "Compaction can't trigger early enough; expect overflow errors "
                   "on long chats. A larger-window model is the real fix.")
    # Floor ~55% of the ceiling, kept inside [minimum, ceiling - headroom].
    floor = min(ceiling - context_diet.COMPACT_HEADROOM,
                max(context_diet.MIN_COMPACT_FLOOR, int(ceiling * FLOOR_RATIO)))
    return {"context": {"compact_ceiling_tokens": ceiling,
                        "compact_floor_tokens": floor, "mem1_total": mem1},
            "fits": fits, "warning": warning}


def apply_context(window: int | None = None, *, ceiling: int | None = None,
                  mem1_total: int | None = None,
                  floor: int | None = None) -> dict:
    """Simple path: pass `window`. Advanced path: pass `ceiling` (and
    optionally `floor`, `mem1_total`) to set the numbers directly."""
    if ceiling is not None:
        out = custom_context(ceiling, mem1_total, floor)
    else:
        out = derive_context(window)
    CL.save_config({"context": out["context"]})
    return out
