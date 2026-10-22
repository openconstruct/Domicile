#!/usr/bin/env python3
"""Shared mem0 provider errors and credentials."""

from __future__ import annotations

import random
import time
from credentials import read_key


class UsageLimitReached(RuntimeError):
    """The subscription's/quota's rolling usage window is exhausted."""


HTTP_LIMIT_MARKERS = (
    "429", "rate limit", "rate-limit", "insufficient credit",
    "insufficient_quota", "402", "quota", "balance",
    "exceeded your", "throttl", "requests per minute",
    # dead subscription / revoked access: ali returned 403
    # AccessDenied.Unpurchased for days and ingest silently starved.
    "unpurchased", "accessdenied", "access denied",
)

HARD_CAP_MARKERS = (
    "insufficient credit", "insufficient_quota", "402",
    "quota exceeded", "exceeded your", "balance",
    "unpurchased", "accessdenied", "access denied",
)


def is_usage_limit(text: str) -> bool:
    """Any HTTP-backend usage-limit signal, hard cap or soft throttle alike."""
    t = (text or "").lower()
    return any(m in t for m in HTTP_LIMIT_MARKERS)


def is_hard_cap(text: str) -> bool:
    """A spent quota / no credit: retrying the same call is pointless."""
    t = (text or "").lower()
    return any(m in t for m in HARD_CAP_MARKERS)


def is_rate_limit(text: str) -> bool:
    t = (text or "").lower()
    if is_hard_cap(t):
        return False
    return any(s in t for s in (
        "429", "rate limit", "rate-limit", "too many requests",
        "throttl", "requests per minute",
    ))



def retry_chat(create, backoff, **payload):
    """Retry transient throttles, never hard quota failures."""
    for attempt in range(len(backoff) + 1):
        try:
            return create(**payload)
        except Exception as exc:
            if not is_rate_limit(str(exc)) or attempt == len(backoff):
                raise
            time.sleep(backoff[attempt] * (1 + random.uniform(0, 0.3)))


class ExtractionResponse:
    """Shared mem0 response handling; adapters supply their request settings."""
    validate_json = None
    reasoning_headroom = 4096

    def _refresh_credentials(self):
        if not hasattr(self, "_key_source"):
            return
        key = read_key(*self._key_source)
        if key != self._client.api_key:
            replacement = self._client_factory(api_key=key, **self._client_options)
            previous, self._client = self._client, replacement
            previous.close()

    def generate_response(self, messages, response_format=None, tools=None,
                          tool_choice="auto", **kwargs) -> str:
        msgs = [dict(m) for m in (messages or []) if (m.get("content") or "").strip()]
        if not msgs:
            return ""
        self._refresh_credentials()
        wants_json = isinstance(response_format, dict) and response_format.get("type") == "json_object"
        if wants_json:
            msgs[-1]["content"] += (
                "\n\nRespond with raw JSON only. No prose before or after, "
                "no explanation, no markdown fence.")
        budget = int(kwargs.get("max_tokens") or 2048) + self.reasoning_headroom
        for attempt in range(2):
            try:
                response = self._request(msgs, budget, kwargs)
            except Exception as exc:
                if is_usage_limit(str(exc)):
                    raise UsageLimitReached(str(exc)[:300]) from exc
                raise
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            if text:
                if (wants_json and attempt == 0 and self.validate_json is not None
                        and not self.validate_json(text)):
                    msgs[-1] = dict(msgs[-1])
                    msgs[-1]["content"] += (
                        "\n\nYour previous reply was not valid JSON and was "
                        "discarded. Reply again with ONLY valid JSON.")
                    continue
                return text
            if choice.finish_reason != "length" or attempt:
                return text
            budget += self.reasoning_headroom * 2
        return text
