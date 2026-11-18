#!/usr/bin/env python3
"""Extraction LLM chain for the memory store: which provider extracts facts,
and how to fall through when one hits a usage limit.

Config-driven (the `compact` chain in config.json; MEM0_LLM is a preference
that goes first). Used by mem0_v2 for add(extract=True).
"""

from __future__ import annotations

import os
import time

import config_loader as CL
import mem0_claude_llm
import mem0_llm_common
import mem0_oai_llm


def _llm_spec(prov: str, model) -> tuple | None:
    """One (backend, model, endpoint_cfg) for a provider name, or None."""
    blk = CL.provider_block(prov) or {}
    if not blk:
        return None
    if CL.is_cli_provider(prov):
        return ("claude_cli", model, None)
    if blk.get("base_url"):
        return ("oai", model, {
            "base_url": blk["base_url"],
            "key_file": blk.get("key_file"),
            "key_env": blk.get("key_env"),
        })
    return None


def resolve_llm_chain() -> list:
    """Every extraction backend worth trying, best first.

    MEM0_LLM is a *preference*, not a pin: it goes first (that is what the
    systemd drop-in is for -- keep extraction off the subscription while a
    cheap provider is alive), and the compaction chain for the active
    provider picks up when it caps out or its key dies.
    """
    specs, seen = [], set()

    def _push(spec):
        if not spec:
            return
        key = (spec[0], spec[1])
        if key in seen:
            return
        seen.add(key)
        specs.append(spec)

    ovr = os.environ.get("MEM0_LLM")
    if ovr:
        blk = CL.provider_block(ovr) or {}
        _push(_llm_spec(ovr, (blk.get("models", {}).get("compact")
                              or [None])[0]))

    for prov, model in CL.resolve_chain("compact"):
        _push(_llm_spec(prov, model))

    return specs or [("claude_cli", None, None)]


MEM0_LLM_CHAIN = resolve_llm_chain()
MEM0_LLM_BACKEND, MEM0_LLM_MODEL, MEM0_LLM_ENDPOINT = MEM0_LLM_CHAIN[0]

_IDENT = CL.load_identity()

CUSTOM_INSTRUCTIONS = (
    f"These memories form the long-term continuity store for an AI agent "
    f"({_IDENT['agent_name']}) and its operator ({_IDENT['operator_name']}). "
    "They are re-read months later, cold, without the "
    "conversation that produced them. Prefer fuller memories that carry the "
    "reasoning behind a decision, not just the decision itself. When a fact came "
    "from debugging or analysis, keep enough of the 'why' and the hard "
    "specifics -- file paths, function names, error strings, version numbers, "
    "config values, dates -- that someone could act on it without the original "
    "context. Record what changed and what it replaced. Attribute decisions to "
    "whoever made them. Err toward the upper end of the length guidance; "
    "terseness that drops the rationale is a failure here, not concision. "
    "When the conversation contains a correction, a mistake together with its "
    "cause, or a hard-won rule about how something must be done, also record "
    "it as its own memory that begins with 'Lesson:' and states what to do or "
    "avoid, in which situation, and why -- for example 'Lesson: when X, do Y "
    "rather than Z, because ...'. Only for things that would change what "
    "someone does next time; an ordinary fact is not a lesson."
)


# extraction LLM: a chain, not a pin

DEMOTION_RETRY_SECONDS = 1800.0   # how long a capped link stays demoted


def _build_llm(spec):
    """One extraction adapter from a (backend, model, endpoint) spec."""
    backend, model, ep = spec
    if backend == "claude_cli":
        return mem0_claude_llm.ClaudeCliLLM(
            {"model": model or mem0_claude_llm.DEFAULT_MODEL})
    if backend == "oai":
        if not ep:
            raise RuntimeError(
                "no OAI endpoint resolved for mem0 extraction -- "
                "every provider is marked down or missing its key")
        return mem0_oai_llm.OpenAILLM({
            "model": model or mem0_oai_llm.DEFAULT_MODEL,
            "base_url": ep["base_url"],
            "key_file": ep.get("key_file"),
            "key_env": ep.get("key_env"),
        })
    raise RuntimeError(
        f"MEM0_LLM={backend!r} unrecognized; use 'claude_cli' or 'oai'")


class ChainLLM:
    """Walks the extraction chain when a link reports a usage limit.

    A capped link is demoted, not discarded: after DEMOTION_RETRY_SECONDS
    the walk restarts from the top, so a rolling-window quota recovers on
    its own instead of pinning us to the fallback for the process's life.
    """

    def __init__(self, specs: list):
        self.specs = list(specs)
        self._i = 0
        self._llm = None
        self._demoted_at = 0.0

    @property
    def spec(self):
        return self.specs[min(self._i, len(self.specs) - 1)]

    def _active(self):
        while self._llm is None and self._i < len(self.specs):
            try:
                self._llm = _build_llm(self.spec)
            except Exception as exc:  # noqa: BLE001  -- missing key, bad cfg
                self._demote(f"unusable: {type(exc).__name__}: {exc}")
        if self._llm is None:
            raise RuntimeError(
                "no usable mem0 extraction provider in "
                f"{[(s[0], s[1]) for s in self.specs]}")
        return self._llm

    def _demote(self, why: str) -> None:
        import sys
        sys.stderr.write(
            f"mem0: extraction link {self.spec[0]}/{self.spec[1]} capped "
            f"({why[:120]}); falling through\n")
        self._i += 1
        self._llm = None
        self._demoted_at = time.monotonic()

    def _maybe_reset(self) -> None:
        if self._i and (time.monotonic() - self._demoted_at
                        > DEMOTION_RETRY_SECONDS):
            self._i, self._llm = 0, None

    def generate_response(self, *args, **kwargs) -> str:
        self._maybe_reset()
        last = None
        while self._i < len(self.specs):
            try:
                return self._active().generate_response(*args, **kwargs)
            except mem0_llm_common.UsageLimitReached as exc:
                last, _ = exc, self._demote(str(exc))
            except Exception as exc:  # noqa: BLE001
                if not mem0_llm_common.is_usage_limit(str(exc)):
                    raise
                last, _ = exc, self._demote(str(exc))
        self._i = len(self.specs) - 1   # stay on the last link; reset later
        raise mem0_llm_common.UsageLimitReached(
            f"every extraction provider capped; last: {str(last)[:200]}")
