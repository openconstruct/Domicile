#!/usr/bin/env python3
"""OpenRouter mem0 extraction adapter."""

from __future__ import annotations


from mem0.llms.base import LLMBase

import mem0_llm_common as _common

DEFAULT_MODEL = "inclusionai/ling-3.0-flash"
DEFAULT_TIMEOUT = 120


UsageLimitReached = _common.UsageLimitReached
_is_usage_limit = _common.is_usage_limit
_is_hard_cap = _common.is_hard_cap
_is_rate_limit = _common.is_rate_limit


_RATE_LIMIT_BACKOFF = (15.0, 40.0, 75.0)


def _read_key(key_file: str) -> str:
    return _common.read_key(key_file, "OPENROUTER_API_KEY")


class OpenRouterLLM(_common.ExtractionResponse, LLMBase):
    """mem0 LLM provider hitting OpenRouter's chat/completions directly."""

    def __init__(self, config=None):
        super().__init__(config)
        from openai import OpenAI
        model = getattr(self.config, "model", None)
        self.model = model or DEFAULT_MODEL
        key_file = getattr(self.config, "key_file", None)  # identifier only; the key comes from .env
        self._key_source = (key_file, "OPENROUTER_API_KEY")
        self._client_factory = OpenAI
        self._client_options = dict(
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/openconstruct/Domicile",
                "X-Title": "Domicile mem0",
            },
        )
        self._client = OpenAI(api_key=_common.read_key(*self._key_source), **self._client_options)

    def _validate_config(self):
        return

    def _create_retrying(self, **kwargs):
        """chat.completions.create with sync backoff on a transient 429."""
        return _common.retry_chat(self._client.chat.completions.create,
                                  _RATE_LIMIT_BACKOFF, **kwargs)

    def _request(self, messages, budget, options):
        return self._create_retrying(
            model=self.model, messages=messages, max_tokens=budget,
            timeout=options.get("timeout", DEFAULT_TIMEOUT))


def register() -> None:
    """Make the provider available to mem0 as `openrouter_ling`."""
    from mem0.configs.llms.base import BaseLlmConfig
    from mem0.utils.factory import LlmFactory

    LlmFactory.provider_to_class["openrouter_ling"] = (
        "mem0_or_llm.OpenRouterLLM",
        BaseLlmConfig,
    )
