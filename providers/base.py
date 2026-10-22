"""Provider interface and completion errors."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TruncatedCompletion(RuntimeError):
    """finish_reason == 'length': the model ran out of output budget."""


class EmptyCompletion(RuntimeError):
    """The call succeeded and returned nothing usable."""


class Provider(ABC):
    """One backend. Instantiated once at import and reused for the process."""

    name: str = "base"

    @abstractmethod
    async def astream_turn(self, chat, text, on_delta=None, images=None,
                           source=""):
        """Run one turn. Returns (reply_text, status)."""

    @abstractmethod
    async def asummarize(self, prompt: str, max_tokens: int,
                         system: str | None = None,
                         model: str | None = None) -> str:
        """Write a handover note. No tools, no session, no memory."""

    def context_tokens(self, chat) -> int:
        """Best estimate of what the next turn will send. Drives the badge."""
        return 0

    async def aclose(self) -> None:
        """Release anything long-lived (HTTP clients, MCP subprocesses)."""
