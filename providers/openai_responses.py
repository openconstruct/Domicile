"""Responses wire adapter for the shared conversation runner."""

from __future__ import annotations

import json
from types import SimpleNamespace

from .openai_compatible import (
    OpenAICompatibleProvider,
    TruncatedCompletion,
    EmptyCompletion,
    _log,
    _RATE_LIMIT_BACKOFF,
)


def _part_to_input(part: dict) -> dict | None:
    """One chat-completions content part -> one Responses input part."""
    ptype = part.get("type")
    if ptype == "text":
        return {"type": "input_text", "text": part.get("text", "")}
    if ptype == "image_url":
        url = part.get("image_url")
        url = url.get("url") if isinstance(url, dict) else url
        if not url:
            return None
        return {"type": "input_image", "image_url": url}
    return None


def _user_content(content) -> list[dict]:
    """User-side content: string or parts -> input parts."""
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    out = []
    for part in content or []:
        mapped = _part_to_input(part) if isinstance(part, dict) else None
        if mapped:
            out.append(mapped)
    return out or [{"type": "input_text", "text": ""}]


def _tool_result_output(content):
    if isinstance(content, str):
        return content
    out = []
    for part in content or []:
        mapped = _part_to_input(part) if isinstance(part, dict) else None
        if mapped:
            out.append(mapped)
    if not out:
        return ""
    return out


class OpenAIResponsesProvider(OpenAICompatibleProvider):
    """One provider speaking the OpenAI Responses wire format."""

    # translate

    def _to_responses_input(self, messages: list[dict]) -> tuple[str | None,
                                                                  list[dict]]:
        """chat-completions messages -> (instructions, input items)."""
        instructions: str | None = None
        items: list[dict] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                c = m.get("content")
                instructions = c if isinstance(c, str) else json.dumps(c)
                continue
            if role == "user":
                items.append({"role": "user",
                              "content": _user_content(m.get("content"))})
                continue
            if role == "assistant":
                text = m.get("content")
                if isinstance(text, str) and text:
                    items.append({
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    })
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    items.append({
                        "type": "function_call",
                        "call_id": tc.get("id") or "",
                        "name": fn.get("name") or "",
                        "arguments": fn.get("arguments") or "{}",
                    })
                continue
            if role == "tool":
                items.append({
                    "type": "function_call_output",
                    "call_id": m.get("tool_call_id") or "",
                    "output": _tool_result_output(m.get("content")),
                })
                continue
        return instructions, items

    @staticmethod
    def _flat_tools(schemas: list[dict]) -> list[dict]:
        """chat-completions nested tool schema -> Responses flat schema."""
        flat = []
        for s in schemas or []:
            if s.get("type") != "function":
                continue
            fn = s.get("function") or s
            flat.append({
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description") or "",
                "parameters": fn.get("parameters")
                or {"type": "object", "properties": {}},
            })
        return flat

    # turn call

    async def _one_call(self, messages, on_delta, parts):
        """One streamed Responses turn. Same contract as the base's _one_call: returns (text, tool_calls, usage, reasoning)."""
        instructions, items = self._to_responses_input(messages)
        tools = self._flat_tools(self._tool_schemas())

        stream = await self._client.responses.create(
            model=self.model,
            input=items,
            instructions=instructions,
            tools=tools or None,
            stream=True,
            max_output_tokens=max(self.max_tokens, 16),
        )

        text = ""
        tool_calls: list[dict] = []
        finish = None
        usage = None
        completed = False

        async for event in stream:
            if completed:
                continue
            etype = getattr(event, "type", "")
            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                text += delta
                if on_delta is not None:
                    on_delta(parts, text)
            elif etype == "response.output_item.done":
                item = getattr(event, "item", None)
                if getattr(item, "type", "") == "function_call":
                    tool_calls.append({
                        "id": getattr(item, "call_id", "") or "",
                        "type": "function",
                        "function": {
                            "name": getattr(item, "name", "") or "",
                            "arguments": getattr(item, "arguments", "") or "{}",
                        },
                    })
            elif etype in ("response.completed", "response.incomplete",
                           "response.failed"):
                resp = getattr(event, "response", None)
                u = getattr(resp, "usage", None)
                if u is not None:
                    usage = SimpleNamespace(
                        prompt_tokens=getattr(u, "input_tokens", 0) or 0,
                        completion_tokens=getattr(u, "output_tokens", 0) or 0,
                        model=getattr(resp, "model", None),
                    )
                reason = getattr(
                    getattr(resp, "incomplete_details", None), "reason", None)
                if etype == "response.incomplete" and reason:
                    finish = "length" if reason == "max_output_tokens" \
                        else reason
                elif etype == "response.failed":
                    finish = "failed"
                completed = True

        if finish == "failed":
            raise RuntimeError(
                f"{self.model} failed the response upstream "
                f"(see provider log).")
        if finish == "length" and not text.strip() and not tool_calls:
            raise TruncatedCompletion(
                f"{self.model} returned nothing and hit the token ceiling "
                f"({self.max_tokens}) -- most likely spent on hidden "
                f"reasoning. Raise max_tokens or shorten the input.")
        if not text.strip() and not tool_calls:
            raise EmptyCompletion(f"{self.model} returned an empty response.")

        return text, tool_calls, usage, None

    # compact

    async def _aresp_retrying(self, **kwargs):
        """Responses completion with rate-limit backoff."""
        return await self._retry_rate_limit(self._client.responses.create,
                                          _RATE_LIMIT_BACKOFF, **kwargs)

    async def asummarize(self, prompt: str, max_tokens: int,
                         system: str | None = None,
                         model: str | None = None) -> str:
        await self._refresh_credentials()
        sub = model or self.submodel
        base = max_tokens + self.reasoning_headroom
        last: Exception | None = None
        for attempt, budget in enumerate((base, base + self.reasoning_headroom * 2)):
            try:
                resp = await self._aresp_retrying(
                    model=sub,
                    input=[{"role": "user", "content": prompt}],
                    instructions=system or
                    "You write factual handover notes for another "
                    "assistant. Output only the note itself -- no "
                    "preamble, no commentary, no offer to help further.",
                    stream=False,
                    max_output_tokens=max(budget, 16),
                )
                text = ""
                for item in getattr(resp, "output", None) or []:
                    if getattr(item, "type", "") != "message":
                        continue
                    for p in getattr(item, "content", None) or []:
                        if getattr(p, "type", "") == "output_text":
                            text += getattr(p, "text", "") or ""
                text = text.strip()
                status = getattr(resp, "status", "")
                reason = getattr(
                    getattr(resp, "incomplete_details", None), "reason", None)
                try:
                    u = resp.usage
                    think = getattr(
                        getattr(u, "output_tokens_details", None),
                        "reasoning_tokens", None) or 0
                    _log(f"summarize {sub}: budget={budget} "
                         f"reasoning={think} visible={text and len(text)} "
                         f"status={status}")
                except Exception:
                    pass
                if status == "incomplete" and reason == "max_output_tokens" \
                        and not text:
                    raise TruncatedCompletion(
                        f"{sub} hit the {budget}-token ceiling with "
                        f"no output (hidden reasoning).")
                if not text:
                    raise EmptyCompletion(
                        f"{sub} returned an empty summary.")
                return text
            except (TruncatedCompletion, EmptyCompletion) as exc:
                last = exc
                if attempt == 0:
                    continue
        raise last
