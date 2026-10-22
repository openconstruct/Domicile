#!/usr/bin/env python3
"""OpenAI-compatible mem0 extraction adapter."""

from __future__ import annotations

import json

from mem0.llms.base import LLMBase

import mem0_llm_common as _common

DEFAULT_MODEL = "muse-spark-1.2-contributor"

_RATE_LIMIT_BACKOFF = (15.0, 40.0, 75.0)

_NO_THINKING_ENDPOINTS = ("maas.aliyuncs.com",)


def _parses_as_json(text: str) -> bool:
    """True if mem0's extraction parser would get facts out of this reply."""
    from mem0.memory.utils import remove_code_blocks, extract_json
    cleaned = remove_code_blocks((text or "").strip())
    if not cleaned:
        return False
    for candidate in (cleaned, extract_json(cleaned)):
        try:
            json.loads(candidate, strict=False)
            return True
        except Exception:
            continue
    return False


UsageLimitReached = _common.UsageLimitReached
_is_usage_limit = _common.is_usage_limit
_read_key = _common.read_key


class OpenAILLM(_common.ExtractionResponse, LLMBase):
    """mem0 LLM provider hitting any OpenAI-compatible /chat/completions."""

    def __init__(self, config=None):
        raw = dict(config or {})
        base_url = raw.get("base_url") or "https://api.openai.com/v1"
        key_file = raw.get("key_file")
        key_env = raw.get("key_env")
        super().__init__({"model": raw.get("model") or DEFAULT_MODEL})
        from openai import OpenAI
        self.model = getattr(self.config, "model", None) or DEFAULT_MODEL
        self._no_thinking = any(
            host in (base_url or "") for host in _NO_THINKING_ENDPOINTS)
        self._key_source = (key_file, key_env)
        self._client_factory = OpenAI
        self._client_options = dict(
            base_url=base_url,
            default_headers={
                "x-opencode-session": "cchat-mem0-extract",
                "x-opencode-client": "cchat-mem0",
            },
        )
        self._client = OpenAI(api_key=_common.read_key(*self._key_source), **self._client_options)

    def _chat(self, payload: dict):
        """One chat call with rate-limit backoff."""
        if self._no_thinking:
            payload = dict(payload)
            payload["extra_body"] = {"enable_thinking": False}
        return _common.retry_chat(self._client.chat.completions.create,
                                  _RATE_LIMIT_BACKOFF, **payload)

    validate_json = staticmethod(_parses_as_json)

    def _request(self, messages, budget, options):
        return self._chat({"model": self.model, "messages": messages, "max_tokens": budget})
