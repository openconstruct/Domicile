#!/usr/bin/env python3
"""Claude CLI adapter for mem0 extraction."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, List, Optional

from mem0.llms.base import LLMBase

import mem0_llm_common as _common

NESTED_ENV = "CCHAT_MEM0_NESTED"

DEFAULT_MODEL = "haiku"
DEFAULT_TIMEOUT = 300

CHILD_FLAGS = [
    "-p",
    "--setting-sources", "",
    "--strict-mcp-config",
    "--max-turns", "1",
    "--disallowed-tools", "Bash,Read,Write,Edit,Glob,Grep,WebSearch,WebFetch,Task",
]


_EXTRACTION_TEMPLATE = """\
You extract durable facts from conversation for a long-term memory store.

Read the "## New Messages" section. Emit each distinct fact that will still
matter months from now: preferences, decisions and the reasoning behind them,
biography, health, project details, commitments, and conclusions worth not
re-deriving.

Rules:
- One self-contained fact per entry. It must stand alone, read cold, with no
  access to this conversation.
- Name the subject ("%s prefers ...", "%s decided ..."). Never "he",
  "the user", or "we".
- Skip anything already covered in "## Existing Memories".
- Skip pleasantries, transient state, and anything you are inferring rather
  than being told.
- Preserve specifics: numbers, filenames, versions, dates, reasons.
- If nothing is worth storing, return {"memory": []}.

Return raw JSON and nothing else, in exactly this shape:
{"memory": [{"text": "..."}, {"text": "..."}]}"""

try:
    import config_loader as _CL
    _ident = _CL.load_identity()
    COMPACT_EXTRACTION_PROMPT = _EXTRACTION_TEMPLATE % (
        _ident["operator_name"], _ident["agent_name"])
except Exception:
    COMPACT_EXTRACTION_PROMPT = _EXTRACTION_TEMPLATE % ("the user", "the agent")


UsageLimitReached = _common.UsageLimitReached


_LIMIT_MARKERS = (
    "usage limit",
    "rate limit",
    "limit reached",
    "out of credit",
    "insufficient credit",
    "quota",
    "too many requests",
    "resets at",
    "upgrade to",
)


def _is_usage_limit(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _LIMIT_MARKERS)


def _compact(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Swap mem0's oversized extraction system prompt for the compact one."""
    if os.environ.get("CCHAT_MEM0_FULL_PROMPT") == "1":
        return messages
    out = []
    for m in messages:
        content = m.get("content") or ""
        if m.get("role") == "system" and len(content) > 20000:
            out.append({"role": "system", "content": COMPACT_EXTRACTION_PROMPT})
        else:
            out.append(m)
    return out


class ClaudeCliLLM(LLMBase):
    """mem0 LLM provider backed by the local `claude` binary."""

    def __init__(self, config=None):
        super().__init__(config)
        self.binary = os.environ.get("CCHAT_CLAUDE_BIN") or shutil.which("claude")
        if not self.binary:
            raise RuntimeError(
                "claude CLI not found on PATH; set CCHAT_CLAUDE_BIN to its path"
            )
        model = getattr(self.config, "model", None)
        self.model = model if model in ("haiku", "sonnet", "opus") else DEFAULT_MODEL

    def _validate_config(self):
        return

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> str:
        """Run one non-interactive `claude -p` and return its text."""
        parts = []
        for m in _compact(messages or []):
            content = (m.get("content") or "").strip()
            if not content:
                continue
            parts.append(content)
        prompt = "\n\n".join(parts)
        if not prompt:
            return ""

        wants_json = isinstance(response_format, dict) and \
            response_format.get("type") == "json_object"
        if wants_json:
            prompt += (
                "\n\nRespond with raw JSON only. No prose before or after, "
                "no explanation, no markdown fence."
            )

        env = dict(os.environ)
        env[NESTED_ENV] = "1"

        proc = subprocess.run(
            [self.binary, *CHILD_FLAGS, "--model", self.model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=kwargs.get("timeout", DEFAULT_TIMEOUT),
            env=env,
            cwd="/tmp",
        )
        if proc.returncode != 0:
            err = ((proc.stderr or "").strip() or (proc.stdout or "").strip())
            if _is_usage_limit(err):
                raise UsageLimitReached(err[:300])
            raise RuntimeError(f"claude -p failed ({proc.returncode}): {err[:300]}")
        return (proc.stdout or "").strip()


def register() -> None:
    """Make the provider available to mem0 as `claude_cli`."""
    from mem0.configs.llms.base import BaseLlmConfig
    from mem0.utils.factory import LlmFactory

    LlmFactory.provider_to_class["claude_cli"] = (
        "mem0_claude_llm.ClaudeCliLLM",
        BaseLlmConfig,
    )


if __name__ == "__main__":
    register()
    llm = ClaudeCliLLM()
    print("binary:", llm.binary, "| model:", llm.model)
    out = llm.generate_response(
        messages=[
            {"role": "system", "content": "You extract facts as JSON."},
            {"role": "user", "content":
             'Extract facts from: "Alex uses Arch and hates yaml." '
             'Return {"facts": ["..."]}'},
        ],
        response_format={"type": "json_object"},
    )
    print("reply:", out)
