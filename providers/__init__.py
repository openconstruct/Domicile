"""Provider construction and instance caching."""

from __future__ import annotations

import config_loader as CL
from config_loader import (  # re-exported: these were imported from here
    CONFIG_PATH, load_config, resolve_chain, resolve_role_model,
    models_for, key_available, active_provider, provider_order,
    supports_vision, max_tool_rounds,
)

from .base import Provider, TruncatedCompletion, EmptyCompletion

_INSTANCE: Provider | None = None
_INSTANCE_NAME: str | None = None


def get_provider(force: str | None = None) -> Provider | None:
    """The process-wide adapter, or None for the built-in Claude Code path."""
    global _INSTANCE, _INSTANCE_NAME

    cfg = CL.load_config()
    name = force or CL.active_provider(cfg)

    if CL.is_cli_provider(name, cfg):
        if force is None:
            _INSTANCE, _INSTANCE_NAME = None, name
        return None

    if _INSTANCE is not None and _INSTANCE_NAME == name and force is None:
        return _INSTANCE

    adapter = CL.adapter_for(name, cfg)
    if adapter == "openai_responses":
        from .openai_responses import OpenAIResponsesProvider
        inst = OpenAIResponsesProvider(name, CL.provider_block(name, cfg))
    elif adapter == CL.CODEX_ADAPTER:
        from .codex_cli import CodexCLIProvider
        inst = CodexCLIProvider(name, CL.provider_block(name, cfg))
    else:
        from .openai_compatible import OpenAICompatibleProvider
        inst = OpenAICompatibleProvider(name, CL.provider_block(name, cfg))

    if force is None:
        _INSTANCE, _INSTANCE_NAME = inst, name
    return inst


def get_model_config(provider: str | None = None) -> dict:
    """The model lists for a provider, as {chat: [...], code: [...], ...}."""
    cfg = CL.load_config()
    name = provider or CL.active_provider(cfg)
    return {role: CL.models_for(name, role, cfg) for role in CL.MODEL_ROLES}


def first_model(role: str, provider: str | None = None) -> str:
    """The preferred model for a role on a provider -- the head of its list."""
    return (get_model_config(provider).get(role) or [""])[0]


def tools_for_mode(mode: str):
    """Kept as a name because four call sites use it; roles.json owns the answer now."""
    return CL.tools_for_role(mode)


__all__ = ["Provider", "TruncatedCompletion", "EmptyCompletion",
           "get_provider", "get_model_config", "first_model",
           "tools_for_mode", "resolve_role_model", "resolve_chain",
           "load_config", "CONFIG_PATH", "models_for", "key_available",
           "active_provider", "provider_order", "supports_vision",
           "max_tool_rounds"]
