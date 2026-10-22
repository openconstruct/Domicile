"""Provider, model, role, and identity configuration."""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from copy import deepcopy
import file_cache
from pathlib import Path

from paths import CCHAT_DATA, CONFIG_PATH, IDENTITY_PATH, ROOT
import credentials

ROLES_PATH = CCHAT_DATA / "roles.json"
RUNTIME_STATE_PATH = CCHAT_DATA / "runtime-state.json"

# Only used for keys identity.json lacks. The wizard writes all of them, so these are
# a safety net for a half-configured install, and they name nobody in particular.
# mem0_user_id scopes the memory store: once memories exist, changing it orphans them.
IDENTITY_DEFAULTS = {
    "operator_user": "",         # shell username; filled from getpass in load_identity()
    "operator_name": "the user", # what prose and prompts call the operator
    "agent_name": "the agent",   # what prose and prompts call the agent
    "mem0_user_id": "",          # mem0 store scope key; filled from operator_user
    "stamp_name": "",            # name in each turn's prompt prefix; empty ->
}

MODEL_ROLES = ("chat", "code", "compact")

CLI_ADAPTER = "claude_cli"
CODEX_ADAPTER = "codex_cli"

DEFAULTS = {
    "active_provider": "anthropic",
    "provider_order": ["anthropic"],
    "providers": {
        "anthropic": {
            "adapter": CLI_ADAPTER,
            "vision": True,
            "models": {"chat": ["sonnet"], "code": ["opus"],
                       "compact": ["haiku"]},
        },
    },
    "backup": {"keep": 2, "prune_slack": 4},
    "hourly_ingest": True,
    # Install-time economics. Defaults are the original hardcoded values.
    #   compact_ceiling_tokens: history size that triggers a compaction.
    #   compact_floor_tokens: size of the compacted text (summary + older
    #     verbatim tail) right after a compaction; base preamble and the
    #     newest few exchanges are not counted. Ceiling >= floor + 16k.
    #   mem1_total: always-loaded memory budget; per-file caps scale to it.
    #   mem1_caps: optional per-file overrides, {"user.md": 5000}.
    "context": {"compact_ceiling_tokens": 60_000,
                "compact_floor_tokens": 30_000, "mem1_total": None,
                "mem1_caps": {}},
}


def context_setting(key: str, cfg: dict | None = None):
    cfg = cfg or load_config()
    return (cfg.get("context") or {}).get(key, DEFAULTS["context"].get(key))

DEFAULT_WRITE_ROOTS = ["data/memory", "data/notes", "data/uploads", "workspace"]

# A profile is the named answer to "what may this role do". Roles pick one with
# a "profile" key in roles.json; the profile wins over any stray tools /
# write_roots values, so a hand-edited null cannot unlock anything. The only way
# to reach "full" is to name it (python3 allow.py --profile full).
#   safe    read everything; write only inside the roots; Bash runs in bwrap with
#           no network; no ToolWake.
#   wakeup  safe, plus ToolWake. Jobs it starts run in bwrap too (network on, so
#           feeds and APIs work), same writable roots, detached from cchat.
#   full    unrestricted.
#   none    no tools (compaction).
PROFILES = {
    "safe":   {"tools": "sandbox", "toolwake": False},
    "wakeup": {"tools": "wakeup", "toolwake": True},
    "full":   {"tools": "full"},
    "none":   {"tools": "none"},
}
DEFAULT_PROFILE = "safe"   # unknown profile names fail closed to this

ROLE_DEFAULTS = {
    "chat":    {"profile": "safe", "max_tool_rounds": 120,
                "write_roots": list(DEFAULT_WRITE_ROOTS)},
    "build":   {"profile": "full", "max_tool_rounds": 120},
    "compact": {"profile": "none", "max_tool_rounds": 0},
    # Unprompted turns (the wakeup loop) run under this role's permissions, not
    # the chat's. Edit with: python3 allow.py --role wakeup ...
    "wakeup":  {"profile": "wakeup", "max_tool_rounds": 60,
                "write_roots": list(DEFAULT_WRITE_ROOTS)},
}


class Roots(list):
    """Resolved write roots. `.toolwake` says whether the profile may launch
    detached jobs. A plain list has no such attribute, so ToolWake stays
    denied unless a provider explicitly restores the resolved permission."""
    toolwake = False


def _expand_profile(block: dict) -> dict:
    """Apply a block's profile: sets tools, forces write_roots, records toolwake."""
    name = block.get("profile")
    if name is None:
        return block                      # legacy / custom role: tools & roots as written
    prof = PROFILES.get(name)
    if prof is None:
        prof, block["profile_error"] = PROFILES[DEFAULT_PROFILE], f"unknown profile {name!r}; using safe"
    block["tools"] = prof["tools"]
    block["toolwake"] = bool(prof.get("toolwake", False))
    if prof["tools"] in ("sandbox", "wakeup"):
        if not block.get("write_roots"):
            block["write_roots"] = list(DEFAULT_WRITE_ROOTS)   # empty/null never means unconfined
    else:
        block.pop("write_roots", None)    # full/none carry no sandbox roots
    return block


def _load_json(text: str) -> dict:
    return json.loads(text)


def _read_json(path: Path) -> dict:
    try:
        data = file_cache.read(path, _load_json)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# config

def load_config() -> dict:
    cfg = deepcopy(DEFAULTS)
    on_disk = _read_json(CONFIG_PATH)
    for key, val in on_disk.items():
        if key == "providers" and isinstance(val, dict):
            for name, block in val.items():
                if isinstance(block, dict) and isinstance(cfg["providers"].get(name), dict):
                    cfg["providers"][name].update(block)
                else:
                    cfg["providers"][name] = block
        elif isinstance(val, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(val)
        else:
            cfg[key] = val
    for name, block in cfg["providers"].items():
        if isinstance(block, dict) and block.get("adapter") not in (CLI_ADAPTER, CODEX_ADAPTER):
            block["key_env"] = credentials.key_name(
                block.pop("key_file", None), block.get("key_env"), name)
    env = os.environ.get("CCHAT_PROVIDER")
    if env:
        cfg["active_provider"] = env
    return cfg


def load_identity() -> dict:
    ident = dict(IDENTITY_DEFAULTS)
    for key, val in _read_json(IDENTITY_PATH).items():
        if isinstance(val, str) and val.strip():
            ident[key] = val.strip()
    if not ident["operator_user"]:
        import getpass
        ident["operator_user"] = getpass.getuser()
    if not ident["mem0_user_id"]:
        ident["mem0_user_id"] = ident["operator_user"]
    return ident


def provider_block(name: str, cfg: dict | None = None) -> dict:
    cfg = cfg if cfg is not None else load_config()
    return (cfg.get("providers") or {}).get(name) or {}


def adapter_for(name: str, cfg: dict | None = None) -> str:
    return provider_block(name, cfg).get("adapter", "openai_compatible")


def is_cli_provider(name: str, cfg: dict | None = None) -> bool:
    return adapter_for(name, cfg) == CLI_ADAPTER


def supports_vision(name: str, cfg: dict | None = None) -> bool:
    return bool(provider_block(name, cfg).get("vision", False))


def provider_order(cfg: dict | None = None) -> list:
    cfg = cfg if cfg is not None else load_config()
    order = [p for p in (cfg.get("provider_order") or [])
             if p in (cfg.get("providers") or {})]
    for name in (cfg.get("providers") or {}):
        if name not in order:
            order.append(name)
    return order


def active_provider(cfg: dict | None = None) -> str:
    cfg = cfg if cfg is not None else load_config()
    name = cfg.get("active_provider") or "anthropic"
    order = provider_order(cfg)
    return name if name in (cfg.get("providers") or {}) else (
        order[0] if order else "anthropic")


def set_active_provider(name: str) -> None:
    """Persist the live provider so a restart agrees with the running process."""
    raw = _read_json(CONFIG_PATH)
    raw["active_provider"] = name
    _write_json_atomic(CONFIG_PATH, raw)


def _deep_merge(base: dict, patch: dict) -> dict:
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], val)
        else:
            base[key] = deepcopy(val)
    return base


def save_config(patch: dict, path: Path | None = None) -> dict:
    """Deep-merge `patch` into the on-disk config.json and write atomically.

    Merges against the raw file, not load_config(), so DEFAULTS never get
    baked to disk. Nested dicts merge key by key; anything else replaces.
    Returns the merged raw dict. `path` is for tests; wizard steps use the default.
    """
    path = path or CONFIG_PATH
    raw = _read_json(path)
    if not raw and path.exists() and path.stat().st_size > 0:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # Unreadable/corrupt: refuse rather than silently clobber it.
            raise ValueError(f"{path} exists but is not valid JSON: {exc}") from exc
    _deep_merge(raw, patch)
    _write_json_atomic(path, raw)
    return raw


def models_for(provider: str, role: str, cfg: dict | None = None) -> list:
    """The ordered model list for `role` on `provider`."""
    block = provider_block(provider, cfg)
    entry = (block.get("models") or {}).get(role)
    if entry is None:
        return []
    if isinstance(entry, str):
        return [entry]
    return [m for m in entry if isinstance(m, str) and m]


def max_tokens_for(provider: str, cfg: dict | None = None) -> int:
    return int(provider_block(provider, cfg).get("max_tokens", 16384))


# thinking levels
#
# There is no shared wire knob behind this. Anthropic's SDK takes a named
# effort level; DashScope/Qwen takes a numeric thinking_budget in extra_body;
# an OpenAI-style endpoint takes reasoning_effort; some endpoints take nothing
# and reject or silently ignore the rest. So the provider block declares both
# *how* to send it ("kind") and *which* levels that mapping actually covers.
# The UI asks thinking_levels() and offers only those -- a control that renders
# three positions and silently does nothing on one of them is worse than no
# control at all.
#
#   "thinking": {"kind": "effort"}
#   "thinking": {"kind": "reasoning_effort"}
#   "thinking": {"kind": "thinking_budget",
#                "levels": {"low": 1024, "medium": 8192, "high": 32768}}
#
# Absent / empty -> the provider has no capability and the UI says so.

THINKING_LEVELS = ("low", "medium", "high")

# Every level any adapter may declare, in ascending order. THINKING_LEVELS is
# the default offer; a provider widens it with "levels": [...] (list form) --
# e.g. codex, whose catalog goes up to xhigh/max. "ultra" is deliberately not
# here: for codex it means automatic task delegation (native subagents), which
# the lockdown removes.
ALL_THINKING_LEVELS = ("low", "medium", "high", "xhigh", "max")

# Which adapter can actually send each kind. A kind nobody implements is
# treated as no capability at all -- better to render "n/a" than to offer a
# level that silently does nothing.
THINKING_KINDS = {
    "effort": "claude_cli",            # ClaudeAgentOptions.effort
    "reasoning_effort": "openai_compatible",
    "thinking_budget": "openai_compatible",
    "codex_effort": "codex_cli",       # -c model_reasoning_effort=<level>
}


def thinking_block(provider: str, cfg: dict | None = None) -> dict:
    """The provider's declared thinking mapping, or {} if it has none."""
    cfg = cfg if cfg is not None else load_config()
    block = provider_block(provider, cfg).get("thinking")
    if not isinstance(block, dict) or not block.get("kind"):
        return {}
    kind = block["kind"]
    if kind not in THINKING_KINDS:
        return {}
    if adapter_for(provider, cfg) != THINKING_KINDS[kind]:
        return {}   # declared a knob this adapter has no sender for
    return block


def supports_thinking(provider: str, cfg: dict | None = None) -> bool:
    return bool(thinking_block(provider, cfg))


def _declared_levels(block: dict) -> list | None:
    """Levels a block restricts/widens to, or None if it declares none.

    dict form ({"low": 1024, ...}) carries a per-level payload (budgets);
    list form (["low", "high", "xhigh"]) is just the offered set.
    """
    levels = block.get("levels")
    if isinstance(levels, dict):
        return [lv for lv in ALL_THINKING_LEVELS if lv in levels]
    if isinstance(levels, (list, tuple)):
        return [lv for lv in ALL_THINKING_LEVELS if lv in levels]
    return None


def thinking_levels(provider: str, cfg: dict | None = None) -> list:
    """Levels this provider can actually honour, in ascending order.

    [] means no capability -- the caller should render the control as
    unavailable rather than offering levels that do nothing.
    """
    block = thinking_block(provider, cfg)
    if not block:
        return []
    declared = _declared_levels(block)
    if declared is not None:
        return declared
    return list(THINKING_LEVELS)


def thinking_param(provider: str, level: str, cfg: dict | None = None) -> dict:
    """Translate (provider, level) into kwargs for the wire call.

    Returns {} for "auto", an unoffered level, or no capability -- meaning
    send nothing and let the provider use its own default.
    """
    block = thinking_block(provider, cfg)
    if not block:
        return {}
    if level not in thinking_levels(provider, cfg):
        return {}

    kind = block["kind"]
    levels = block.get("levels")
    if kind == "reasoning_effort":
        return {"reasoning_effort": level}
    if kind == "thinking_budget":
        budget = (levels or {}).get(level) if isinstance(levels, dict) else None
        if not budget:
            return {}
        body = {"thinking_budget": int(budget)}
        if block.get("enable_thinking") is not None:
            body["enable_thinking"] = bool(block["enable_thinking"])
        return {"extra_body": body}
    return {}   # "effort"/"codex_effort" are sent by their own adapters


# roles

import contextlib
import contextvars

_PERM_ROLE: contextvars.ContextVar = contextvars.ContextVar("perm_role", default=None)


@contextlib.contextmanager
def permission_role(role: str):
    """For the duration of the block, tool/sandbox/round-limit lookups use `role`'s
    block instead of the chat's mode. Memory, model and prompt still follow the
    mode. Scoped to this task's context, so a concurrent turn is unaffected."""
    tok = _PERM_ROLE.set(role)
    try:
        yield
    finally:
        _PERM_ROLE.reset(tok)


def _perm(role: str) -> str:
    return _PERM_ROLE.get() or role


def load_roles() -> dict:
    roles = deepcopy(ROLE_DEFAULTS)
    for name, block in _read_json(ROLES_PATH).items():
        if name.startswith("_") or not isinstance(block, dict):
            continue  # "_comment" and friends are documentation, not roles
        if isinstance(roles.get(name), dict):
            roles[name].update(block)
        else:
            roles[name] = block
    for blk in roles.values():
        if isinstance(blk, dict):
            _expand_profile(blk)
    return roles


def role_block(role: str, roles: dict | None = None) -> dict:
    roles = roles if roles is not None else load_roles()
    return roles.get(role) or roles.get("chat") or ROLE_DEFAULTS["chat"]


def sandbox_protected_files() -> list[Path]:
    """Files that define the sandbox; no writable root may contain them."""
    return [ROLES_PATH, CONFIG_PATH, credentials.ENV_PATH]


def write_roots_for_role(role: str, roles: dict | None = None):
    """Resolved write-sandbox roots for a role, or None if unrestricted.

    Profiles decide: safe/wakeup are always confined, full is not. A role with
    no profile (custom) falls back to its own write_roots key. A bad root entry
    raises -- a mangled sandbox must not quietly become no sandbox.
    """
    blk = role_block(_perm(role), roles)
    entries = blk.get("write_roots")
    if entries is None:
        if blk.get("tools") in ("sandbox", "wakeup"):
            import sandbox
            raise sandbox.SandboxConfigError(
                f"role {role!r} uses the sandbox tier but has no write_roots")
        return None
    import sandbox
    out = Roots(sandbox.resolve_roots(entries, base=ROOT,
                                      protected=sandbox_protected_files()))
    out.toolwake = bool(blk.get("toolwake", False))
    return out


def profile_for_role(role: str, roles: dict | None = None) -> str:
    """The profile a role runs under, or 'custom' for a role without one."""
    return role_block(_perm(role), roles).get("profile") or "custom"


def tools_for_role(role: str, roles: dict | None = None):
    return role_block(_perm(role), roles).get("tools", "full")


def max_tool_rounds(role: str, roles: dict | None = None) -> int:
    """The per-role ceiling on tool round-trips."""
    return int(role_block(_perm(role), roles).get("max_tool_rounds", 120))


def pin_model_role(role: str, roles: dict | None = None) -> str | None:
    """The model role this mode is locked to, ignoring the F3 pin."""
    pin = role_block(role, roles).get("pin_model_role")
    return pin if pin in MODEL_ROLES else None


def model_role_for(role: str, fallback: str | None = None,
                   roles: dict | None = None) -> str:
    """Which model list a mode draws from, most specific wins:"""
    pin = pin_model_role(role, roles)
    if pin:
        return pin
    if fallback in MODEL_ROLES:
        return fallback
    if role in MODEL_ROLES:
        return role
    return "chat"


# runtime state

def load_runtime_state() -> dict:
    """Provider availability state."""
    st = _read_json(RUNTIME_STATE_PATH)
    provs = st.get("providers")
    if not isinstance(provs, dict):
        st["providers"] = {}
    return st


def save_runtime_state(state: dict) -> None:
    _write_json_atomic(RUNTIME_STATE_PATH, state)


def mark_unusable(name: str, kind: str, *, detail: str = "",
                  reset_at: str | None = None, next_probe_at: str | None = None,
                  auto: bool = True) -> dict:
    """Record that `name` cannot serve a turn right now."""
    st = load_runtime_state()
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    prev = st["providers"].get(name) or {}
    st["providers"][name] = {
        "unusable": kind,
        "since": prev.get("since", now),
        "reset_at": reset_at,
        "attempts": int(prev.get("attempts", 0)),
        "last_probe": prev.get("last_probe"),
        "next_probe_at": next_probe_at,
        "detail": (detail or "")[:400],
        "auto": auto,
    }
    save_runtime_state(st)
    return st


def clear_mark(name: str) -> dict:
    st = load_runtime_state()
    st["providers"].pop(name, None)
    save_runtime_state(st)
    return st


def unusable_providers(state: dict | None = None) -> set:
    st = state if state is not None else load_runtime_state()
    return {n for n, v in (st.get("providers") or {}).items()
            if isinstance(v, dict) and v.get("unusable")}


# key checks

def key_available(name: str, cfg: dict | None = None) -> bool:
    """Can we authenticate to this provider right now?"""
    cfg = cfg if cfg is not None else load_config()
    if is_cli_provider(name, cfg):
        return True
    if adapter_for(name, cfg) == CODEX_ADAPTER:
        return True  # session auth via `codex login`, no key file/env to check
    block = provider_block(name, cfg)
    name = credentials.key_name(block.get("key_file"), block.get("key_env"), name)
    return bool(credentials.get(name))



def usable_providers(cfg: dict | None = None,
                     state: dict | None = None) -> list:
    cfg = cfg if cfg is not None else load_config()
    down = unusable_providers(state)
    return [p for p in provider_order(cfg)
            if p not in down and key_available(p, cfg)]


# the walk

def resolve_chain(role: str, cfg: dict | None = None,
                  state: dict | None = None,
                  start: str | None = None,
                  model_role: str | None = None) -> list:
    """Every (provider, model) worth trying for `role`, best first."""
    cfg = cfg if cfg is not None else load_config()
    order = provider_order(cfg)
    if not order:
        return []

    mrole = model_role_for(role, model_role)

    begin = order.index(start) if start in order else (
        order.index(active_provider(cfg)) if active_provider(cfg) in order else 0)
    rotated = order[begin:] + order[:begin]

    down = unusable_providers(state)
    chain = []
    for prov in rotated:
        if prov in down or not key_available(prov, cfg):
            continue
        for model in models_for(prov, mrole, cfg):
            chain.append((prov, model))
    return chain


def resolve_role_model(role: str, **kw):
    """First usable (provider, model) for `role`, or (None, None)."""
    chain = resolve_chain(role, **kw)
    if not chain:
        return None, None
    prov, model = chain[0]
    return (None if is_cli_provider(prov) else prov), model


__all__ = [
    "MODEL_ROLES", "CLI_ADAPTER", "CODEX_ADAPTER", "CONFIG_PATH", "ROLES_PATH",
    "RUNTIME_STATE_PATH", "IDENTITY_PATH", "IDENTITY_DEFAULTS",
    "load_config", "load_identity",
    "provider_block", "adapter_for", "is_cli_provider",
    "supports_vision", "provider_order", "active_provider",
    "set_active_provider", "save_config", "models_for", "max_tokens_for",
    "THINKING_LEVELS", "ALL_THINKING_LEVELS", "THINKING_KINDS", "thinking_block", "supports_thinking",
    "thinking_levels", "thinking_param",
    "load_roles", "role_block", "permission_role", "tools_for_role", "max_tool_rounds",
    "pin_model_role", "model_role_for",
    "load_runtime_state", "save_runtime_state", "mark_unusable", "clear_mark",
    "unusable_providers", "key_available", "usable_providers",
    "resolve_chain", "resolve_role_model",
]
