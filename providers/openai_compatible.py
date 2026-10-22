"""OpenAI-compatible conversation and tool runner."""

from __future__ import annotations

import asyncio
import json
import re

from .base import Provider, TruncatedCompletion, EmptyCompletion

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None

try:
    from openai import RateLimitError as _RateLimitError
except Exception:  # pragma: no cover
    class _RateLimitError(Exception):
        pass

_RATE_LIMIT_BACKOFF = (8.0, 22.0, 45.0)

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


from paths import OPENROUTER_LOG as LOG, MEM1_DIR
import config_loader as CL
import context_diet

# The full text is saved by tool_spill and the marker says how to fetch the
# cut part, so a trim here is not a permanent loss.
TOOL_OUTPUT_MAX_CHARS = 6_000
TOOL_OUTPUT_HEAD = 4_500
TOOL_OUTPUT_TAIL = 1_500

# Substring shape of a provider rejecting an image_url content block (e.g.
# "No endpoints found that support image input", seen as both 400 and 404).
# Kept broad rather than pinned to one wording: over-matching costs a wasted
# retry, under-matching loses the whole turn.
_IMAGE_REJECT_RE = re.compile(
    r"image.{0,40}(not support|unsupported|no endpoint|does not|cannot|"
    r"invalid|reject|not allow)"
    r"|(not support|unsupported|no endpoint|does not|cannot|invalid|reject|"
    r"not allow).{0,40}image",
    re.IGNORECASE,
)


def _truncate_tool_output(text: str, call_id=None) -> str:
    """Shorten a tool result at creation time, marking the gap.

    Anything over tool_spill's minimum is saved whole first (even when it is
    short enough to deliver whole now: the wire diet trims older results harder
    later and needs the original then).
    """
    key = None
    try:
        import tool_spill
        key = tool_spill.spill(call_id, text)
    except Exception:
        key = None
    if len(text) <= TOOL_OUTPUT_MAX_CHARS:
        return text
    head = TOOL_OUTPUT_HEAD
    tail = TOOL_OUTPUT_TAIL
    cut = len(text) - head - tail
    _log(f"tool_output_truncated: {len(text):,} -> {TOOL_OUTPUT_MAX_CHARS:,} chars "
         f"(head={head}, tail={tail}, spilled={bool(key)})")
    try:
        import tool_spill
        hint = tool_spill.recovery_hint(key, head, cut)
    except Exception:
        hint = ""
    note = hint if hint else " Re-read the source for full content."
    return (
        text[:head]
        + f"\n\n… [{cut:,} characters trimmed at source.{note}] …\n\n"
        + text[-tail:]
    )


def _log(line: str) -> None:
    """Append-only breadcrumb. Logging must never take a turn down."""
    try:
        import time as _t
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"{_t.strftime('%Y-%m-%d %H:%M:%S')}  {line}\n")
    except Exception:  # pragma: no cover
        pass


def _read_key(key_file: str, env_var: str = "OPENROUTER_API_KEY") -> str:
    from credentials import read_key
    return read_key(key_file, env_var)


# Exactly the params CL.thinking_param can put on a request. Kept narrow on
# purpose: a broad match would swallow unrelated 400s and retry them into a
# second identical failure.
_THINKING_PARAM_NAMES = ("reasoning_effort", "thinking_budget", "enable_thinking")


def _is_thinking_param_rejection(exc: Exception) -> bool:
    """True when a request error is about the reasoning knob specifically.

    Strict gateways name the offending param -- zen/go answers an unknown
    extra_body key with 400 {'error': {'param': 'variant', ...}}. Only that
    shape counts here.
    """
    text = str(exc)
    if "400" not in text and "invalid_request_error" not in text:
        return False
    return any(name in text for name in _THINKING_PARAM_NAMES)


# Pinned thinking level for the compaction seam (asummarize). Compaction is
# transformation, not search: a deep reasoning budget buys little and every
# token of it is output-priced and serial, which is where the 4+ minute
# compactions came from -- the compact model's own default. Translated
# per-provider like the chat path; {} = provider has no mapping, keep default.
_COMPACT_THINKING = "low"


class OpenAICompatibleProvider(Provider):
    """One provider speaking the OpenAI chat-completions wire format."""

    def __init__(self, name: str, cfg: dict):
        if AsyncOpenAI is None:
            raise RuntimeError("pip install openai")
        _ = _log  # fail at construction, not inside a swallowing except
        self.name = name
        self.cfg = cfg
        base_url = cfg.get("base_url")
        if not base_url:
            raise RuntimeError(
                f"provider {name!r} has no base_url in config.json -- an "
                f"OpenAI-compatible provider cannot be constructed without one")
        self.model = (CL.models_for(name, "chat") or [""])[0]
        self.submodel = (CL.models_for(name, "compact") or [self.model])[0]
        if not self.model:
            raise RuntimeError(
                f"provider {name!r} has no models.chat entry in config.json")
        self.max_tokens = int(cfg.get("max_tokens", 16384))
        self.reasoning_headroom = int(cfg.get("reasoning_headroom", 12288))
        self.tool_phase = cfg.get("tool_phase", "readonly")
        self.mcp_allow = cfg.get("mcp_allow")
        self._mcp = None
        self._mcp_lock = asyncio.Lock()
        self.use_mcp = bool(cfg.get("use_mcp", False))
        self._image_cap = None
        _timeout = (httpx.Timeout(90.0, connect=10.0)
                    if httpx is not None else 90.0)
        self._key_source = (cfg.get("key_file", ""), cfg.get("key_env", ""))
        self._client_options = dict(
            base_url=base_url,
            timeout=_timeout,
            max_retries=1,
            default_headers={
                "HTTP-Referer": "https://github.com/openconstruct/Domicile",
                "X-Title": "Domicile",
                "x-opencode-session": "cchat-unset",
                "x-opencode-client": "cchat",
            },
        )
        self._client = AsyncOpenAI(api_key=_read_key(*self._key_source), **self._client_options)

    async def _refresh_credentials(self):
        if not hasattr(self, "_key_source"):
            return  # Test doubles can provide an already-constructed client.
        key = _read_key(*self._key_source)
        if key == self._client.api_key:
            return
        replacement = AsyncOpenAI(api_key=key, **self._client_options)
        previous, self._client = self._client, replacement
        try:
            await previous.close()
        except Exception as exc:
            _log(f"{self.name}: retired client close failed ({type(exc).__name__})")


    # tools

    def _max_rounds(self) -> int:
        """Tool-round ceiling for the turn in flight, from roles.json."""
        try:
            return CL.max_tool_rounds(getattr(self, "_turn_mode", "chat"))
        except Exception:
            return 120

    def _enabled_tools(self) -> list[str]:
        import local_tools as LT
        mode = getattr(self, "_turn_mode", None)
        if mode is not None:
            try:
                return LT.resolve_phase(CL.tools_for_role(mode))
            except Exception:
                pass
        return {
            "none": [],
            "readonly": LT.PHASE_READONLY,
            "write": LT.PHASE_WRITE,
            "full": LT.PHASE_FULL,
        }.get(self.tool_phase, LT.PHASE_READONLY)

    def _tool_schemas(self) -> list[dict]:
        import local_tools as LT
        enabled = self._enabled_tools()
        out = LT.schemas(enabled) if enabled else []
        if self._mcp is not None:
            out += self._mcp.schemas()
        if not self._supports_image():
            out = [s for s in out if (s.get("function") or {}).get("name")
                   != "ReadImage"]
        return out

    async def _ensure_mcp(self):
        """Start MCP servers once, on the first turn that could use them."""
        async with self._mcp_lock:
            if self._mcp is not None or not self.use_mcp:
                return
            try:
                from .mcp_client import MCPManager
                import cchat
                mgr = MCPManager(cchat.mcp_servers(), allow=self.mcp_allow)
                if not mgr.available:
                    self.use_mcp = False
                    return
                await mgr.start()
                self._mcp = mgr
                _log(mgr.status())
            except Exception as exc:  # noqa: BLE001
                _log(f"MCP unavailable: {type(exc).__name__}: {exc}")
                self.use_mcp = False

    # prompt

    _ROLE_PROMPTS = {
        "chat": "chat.md",
        "build": "code-building.md",
    }

    def _system_prompt(self, chat) -> str:
        """Role prompt + SYSTEM_PROMPT + mem1, rebuilt every turn."""
        import cchat
        from pathlib import Path

        mode = chat.get("mode", "chat")
        system = cchat.SYSTEM_PROMPT

        role_file = self._ROLE_PROMPTS.get(mode)
        if role_file:
            role_path = MEM1_DIR / role_file
            try:
                role_text = role_path.read_text().strip()
                if role_text:
                    system = f"{role_text}\n\n{system}"
            except FileNotFoundError:
                pass  # file missing — fall back to base prompt

        if cchat.MEM1 is not None:
            try:
                block, stats = cchat.MEM1.load()
                if block:
                    system = f"{system}\n\n{block}"
                chat["_mem1"] = stats
            except Exception as exc:
                chat["_mem1"] = {"error": repr(exc)}
        return system

    def _maybe_diet_wire(self, chat) -> None:
        """Trim stale tool results / old images before paying for them again."""
        try:
            import context_diet as cd
            wire = chat.get("wire")
            if not isinstance(wire, list) or not wire:
                return
            if cd.wire_chars(wire) < cd.WIRE_MAX_CHARS_BEFORE_DIET:
                return
            new_wire, st = cd.diet_wire(wire)
            if st["saved_chars"] <= 0:
                return
            chat["wire"] = new_wire
            cd.log_compaction(
                f"{chat.get('id')}: wire diet {st['before_chars']:,} -> "
                f"{st['after_chars']:,} chars (~{st['before_tokens']:,} -> "
                f"~{st['after_tokens']:,} tok, saved ~{st['saved_tokens']:,} "
                f"tok; shrunk {st['shrunk']} results"
                + (f", floor-shrunk {st['floor_shrunk']}" if st.get("floor_shrunk") else "")
                + (f", spared {st['mem_results_spared']}m/{st['reads_spared']}r"
                   if st.get("mem_results_spared") or st.get("reads_spared") else "")
                + (f", dropped {st['images_dropped']} images)" if st["images_dropped"] else ")")
            )
        except Exception:
            pass  # trimming must never take a turn down

    @staticmethod
    def _sanitize_tool_calls(tool_calls):
        """Coerce every tool_call's arguments into a JSON object string, IN PLACE. Returns how many calls were rewritten."""
        fixed = 0
        for tc in tool_calls:
            fn = tc.get("function") or {}
            raw = fn.get("arguments")
            if isinstance(raw, str):
                try:
                    ok = isinstance(json.loads(raw), dict)
                except ValueError:
                    ok = False
            else:
                ok = False
            if not ok:
                fn["arguments"] = "{}"
                tc["function"] = fn
                fixed += 1
        if fixed:
            _log(f"sanitized {fixed} tool_call(s) with "
                 f"non-JSON-object arguments -> '{{}}'")
        return fixed

    def _write_roots(self):
        """Sandbox roots for this turn's mode; a broken config means nothing writable."""
        try:
            return CL.write_roots_for_role(getattr(self, "_turn_mode", "chat"))
        except Exception:
            return []

    def _supports_image(self) -> bool:
        """Does the live model accept image input? Cached, never raises."""
        ov = self.cfg.get("supports_image")
        if ov is not None:
            self._image_cap = bool(ov)
        if self._image_cap is not None:
            return self._image_cap
        if not CL.supports_vision(self.name):
            self._image_cap = False
            return False
        cap = True
        try:
            base = self.cfg.get("base_url", "https://openrouter.ai/api/v1")
            if httpx is not None:
                _vh = getattr(self, "_client", None)
                _vh = dict(_vh.default_headers) if _vh is not None else {}
                resp = httpx.get(f"{base}/models", timeout=8.0, headers=_vh)
                if resp.is_success:
                    import itertools as _it
                    for m in resp.json().get("data", []):
                        if m.get("id") == self.model:
                            mods = m.get("input_modalities")
                            flat = list(_it.chain.from_iterable(
                                ([v] if isinstance(v, str) else v)
                                for v in (mods.values() if isinstance(mods, dict)
                                          else mods or [])))
                            cap = "image" in flat
                            break
        except Exception as exc:
            try:
                _log(f"image-cap lookup failed ({type(exc).__name__}: {exc}); "
                     f"assuming vision-capable")
            except Exception:
                pass
            cap = True
        self._image_cap = cap
        try:
            _log(f"image-cap {self.model!r}: {cap}")
        except Exception:
            pass
        return cap

    @staticmethod
    def _is_image_rejection(exc: Exception) -> bool:
        """Best-guess match on a provider's own text rejecting image input.

        Matches the message, not the exception type: the same rejection text
        arrives under different exception classes (400 and 404).
        """
        return bool(_IMAGE_REJECT_RE.search(str(exc)))

    def _strip_images(self, *collections) -> int:
        """Replace image_url content blocks with a placeholder, in place.

        Called after a provider rejects an image mid-turn, so the retried
        call doesn't resend the same content and the turn doesn't have to be
        thrown away to recover. Returns how many blocks were replaced.
        """
        count = 0
        for coll in collections:
            if not isinstance(coll, list):
                continue
            for m in coll:
                content = m.get("content")
                if (m.get("role") == "user"
                        and isinstance(content, list)
                        and any(isinstance(p, dict)
                                and p.get("type") == "image_url"
                                for p in content)):
                    m["content"] = ("[image omitted -- provider rejected "
                                     f"image input for model {self.model!r}]")
                    count += 1
        return count

    def _heal_wire(self, chat) -> None:
        """Repair bad entries already persisted in wire."""
        try:
            wire = chat.get("wire")
            if not isinstance(wire, list) or not wire:
                return
            fixed = 0
            placeholders = 0
            image = self._supports_image()
            n = len(wire)
            for i, m in enumerate(wire):
                tcs = m.get("tool_calls")
                if tcs:
                    fixed += self._sanitize_tool_calls(tcs)
                if image:
                    continue
                content = m.get("content")
                if (m.get("role") == "user"
                        and isinstance(content, list)
                        and any(isinstance(p, dict)
                                and p.get("type") == "image_url"
                                for p in content)):
                    tail = i == n - 1
                    note = " [image read was never completed -- it could not " \
                        "be delivered in a prior turn]" if tail else ""
                    m["content"] = "[image omitted -- current model does not " \
                        "support image input]" + note
                    placeholders += 1
            if fixed:
                _log(f"healed wire: {fixed} tool_call(s) carried "
                     f"invalid tool arguments")
            if placeholders:
                _log(f"healed wire: {placeholders} image_url block(s) replaced "
                     f"with a placeholder (model {self.model!r} has no vision)")
        except Exception:
            pass  # healing must never take a turn down either

    def _build_messages(self, chat, prompt: str,
                        wire_prompt: str | None = None) -> list[dict]:
        """System + full wire, with this turn's user entry spliced in."""
        self._maybe_diet_wire(chat)
        self._heal_wire(chat)
        wire = chat.setdefault("wire", [])
        wire.append({"role": "user",
                     "content": wire_prompt if wire_prompt is not None
                     else prompt})
        return ([{"role": "system", "content": self._system_prompt(chat)}]
                + wire[:-1] + [{"role": "user", "content": prompt}])

    async def _memory_hints(self, text: str) -> str:
        query = (text or "").strip()
        if not query:
            return ""
        try:
            import asyncio as _aio
            import mem0_core
            hits = await _aio.wait_for(
                _aio.to_thread(mem0_core.hint, query[:2000]),
                timeout=30.0)
        except Exception as exc:  # noqa: BLE001 - degrade, don't die
            try:
                _log(f"hints unavailable: {type(exc).__name__}: {exc}")
            except Exception:
                pass
            return ""
        if not hits:
            return ""
        lines = []
        for score, made, mem in hits:
            stamp = f"{made} " if made else ""
            tag = "[kw] " if score <= 0 else f"[{score:.2f}] "
            lines.append(f"- {tag}{stamp}{mem}")
        try:
            _log(f"hints: {len(hits)} hit(s), top={hits[0][0]:.2f}, "
                 f"q={query[:80]!r}")
        except Exception:
            pass
        return ("[memory hints -- auto-retrieved from long-term memory by "
                "similarity to the message below. Possibly relevant, "
                "possibly noise; call memory_search for full detail before "
                "relying on one.]\n" + "\n".join(lines))

    # turn

    async def astream_turn(self, chat, text, on_delta=None, images=None,
                           source=""):
        # Debug capture (inputlog.py): armed per-turn via data/inputlog/ARM.
        try:
            import inputlog
            inputlog.begin(self.name)
        except Exception:
            inputlog = None
        try:
            return await self._astream_turn(chat, text, on_delta, images,
                                            source)
        finally:
            if inputlog is not None:
                inputlog.end()

    async def _astream_turn(self, chat, text, on_delta=None, images=None,
                            source=""):
        import cchat
        import datetime as dt

        try:
            await self._refresh_credentials()
        except Exception as exc:
            return f"**Error:** `{type(exc).__name__}: {exc}`", "error"

        self._turn_mode = chat.get("mode", "chat")
        self._turn_thinking = chat.get("thinking") or ""

        try:
            self._client.default_headers["x-opencode-session"] = (
                "cchat-" + str(chat.get("id", "anon")))
        except Exception:
            pass

        stamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
        src = f" — {source}" if source else ""
        clean = f"[{stamp} — {cchat.OPERATOR_USER}{src}]\n{text}"

        seed = chat.get("pending_seed")

        hinted = await self._memory_hints(text) if source else ""
        hud = cchat.domicile_hud(chat, self.model)
        prompt = ((f"{seed}\n\n" if seed else "")
                  + (f"{hinted}\n\n" if hinted else "")
                  + clean
                  + (f"\n\n{hud}" if hud else ""))

        await self._ensure_mcp()

        wire_start = len(chat.get("wire") or [])
        messages = self._build_messages(chat, prompt, wire_prompt=clean)
        try:
            import inputlog
            inputlog.write("prompt.txt", prompt)
            inputlog.write("system.txt", messages[0].get("content", ""))
            inputlog.write("meta.json", {
                "provider": self.name, "model": self.model,
                "mode": chat.get("mode"), "wire_len": len(messages),
                "tools": [t.get("function", {}).get("name")
                          for t in (self._tool_schemas() or [])]})
        except Exception:
            pass
        parts: list[str] = []
        status = "ok"
        _midturn = (cchat.midturn_enabled() and chat.get("id") != "api-fresh"
                    and cchat.MIDTURN.open())
        if _midturn and messages and messages[0].get("role") == "system":
            messages[0]["content"] += "\n\n" + cchat._midturn_tool_instruction()

        try:
            max_rounds = self._max_rounds()
            for _round in range(max_rounds):
                try:
                    text_out, tool_calls, usage, reasoning_msg = await self._one_call(
                        messages, on_delta, parts)
                except EmptyCompletion:
                    await asyncio.sleep(3)
                    text_out, tool_calls, usage, reasoning_msg = await self._one_call(
                        messages, on_delta, parts)
                except Exception as exc:
                    if not self._is_image_rejection(exc):
                        raise
                    # Provider rejected an image_url block sent this turn. Don't
                    # let the outer handler wipe the whole turn: record the cap,
                    # scrub the offending content in place, and retry once.
                    # ReadImage leaves _tool_schemas() once _image_cap is False.
                    _log(f"image rejected by provider "
                         f"({type(exc).__name__}): {exc}")
                    self._image_cap = False
                    stripped = self._strip_images(messages, chat.get("wire"))
                    if not stripped:
                        raise  # nothing to fix -- a real error, not this one
                    text_out, tool_calls, usage, reasoning_msg = await self._one_call(
                        messages, on_delta, parts)

                if usage:
                    self._record_usage(chat, usage, messages)

                self._sanitize_tool_calls(tool_calls)

                assistant_msg = {"role": "assistant",
                                 "content": text_out or None}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                if reasoning_msg:
                    # Echo the thinking block back next call under the exact
                    # key the provider used (reasoning_content for Qwen/DeepSeek
                    # thinking models). Required or the provider 400s.
                    assistant_msg.update(reasoning_msg)
                chat["wire"].append(assistant_msg)
                messages.append(assistant_msg)

                if text_out:
                    parts.append(text_out)
                    if on_delta is not None:
                        on_delta(parts, "", force=True)

                if not tool_calls:
                    break

                results = await self._run_tools(tool_calls)
                if _midturn:
                    cchat._attach_midturn_to_tool_results(chat, results)
                chat["wire"].extend(results)
                messages.extend(results)
            else:
                parts.append(
                    f"\n\n*(stopped after {max_rounds} tool rounds)*")

        except asyncio.CancelledError:
            status = "cancelled"
        except Exception as exc:
            parts.append(f"**Error:** `{type(exc).__name__}: {exc}`")
            status = "error"
            try:
                wire = chat.get("wire")
                if isinstance(wire, list) and len(wire) > wire_start:
                    del wire[wire_start:]
                    chat["updated"] = dt.datetime.now().isoformat(
                        timespec="seconds")
            except Exception:
                pass
        finally:
            if _midturn:
                cchat.MIDTURN.close()  # no more tool calls: fall back to next turn

        if status == "cancelled":
            parts.append("*(interrupted)*")

        if seed and status == "ok":
            chat.pop("pending_seed", None)
            chat.pop("precompact_session_id", None)

        return ("\n\n".join(p for p in parts if p) or "*(no reply)*"), status

    async def _one_call(self, messages, on_delta, parts):
        """One streamed completion. Returns (text, tool_calls, usage, reasoning)."""
        tools = self._tool_schemas()
        # Per-turn thinking level, translated by config_loader so the wire
        # shape stays a config question rather than a code one. {} means the
        # chat is on "auto" or this provider declares no mapping -- send
        # nothing and let the provider use its own default.
        extra = {}
        try:
            extra = CL.thinking_param(self.name, getattr(self, "_turn_thinking", ""))
        except Exception as exc:
            _log(f"{self.name}: thinking level not applied "
                 f"({type(exc).__name__}: {exc})")
        request = dict(
            model=self.model,
            messages=messages,
            tools=tools or None,
            stream=True,
            max_tokens=self.max_tokens,
            stream_options={"include_usage": True},
        )
        try:
            import inputlog
            if inputlog.CURRENT is not None:
                inputlog.write(f"round-{inputlog.next_round()}-request.json",
                               {"model": self.model, "messages": messages,
                                "tools": tools, "extra": extra})
        except Exception:
            pass
        try:
            stream = await self._client.chat.completions.create(**request, **extra)
        except Exception as exc:
            if not extra or not _is_thinking_param_rejection(exc):
                raise
            # The capability is declared per provider, but a gateway can
            # validate per upstream model -- so a model swap can make every
            # turn 400 on a knob that worked yesterday. Drop the knob and
            # retry once instead of taking the turn down; the level simply
            # isn't applied, and the log says so.
            _log(f"{self.name}: thinking param rejected ({type(exc).__name__}: "
                 f"{str(exc)[:200]}), retrying without it -- the 'thinking' "
                 f"block in config.json may not fit this model")
            stream = await self._client.chat.completions.create(**request)

        text = ""
        finish = None
        usage = None
        reasoning = ""
        reasoning_key = None   # which delta field carried it; echoed verbatim
        acc: dict[int, dict] = {}

        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue

            # Thinking-mode models (Qwen/DeepSeek on OpenRouter, ali, opencode)
            # stream a hidden chain-of-thought and REQUIRE it back verbatim on
            # the next call -- dropping it 400s the whole conversation. Capture
            # whichever field name the provider used so the echo matches.
            # Keep the field even when the provider streams an EMPTY string:
            # some return reasoning_content="" and still require the key back.
            for _rk in ("reasoning_content", "reasoning"):
                if getattr(delta, _rk, None) is not None:
                    if reasoning_key is None:
                        reasoning_key = _rk
                    reasoning += getattr(delta, _rk) or ""
                    break

            if getattr(delta, "content", None):
                text += delta.content
                if on_delta is not None:
                    on_delta(parts, text)

            for tc in (getattr(delta, "tool_calls", None) or []):
                slot = acc.setdefault(tc.index, {
                    "id": "", "type": "function",
                    "function": {"name": "", "arguments": ""}})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["function"]["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["function"]["arguments"] += tc.function.arguments

        tool_calls = [acc[i] for i in sorted(acc)]

        if finish == "length" and not text.strip() and not tool_calls:
            raise TruncatedCompletion(
                f"{self.model} returned nothing and hit the token ceiling "
                f"({self.max_tokens}) -- most likely spent on hidden "
                f"reasoning. Raise max_tokens or shorten the input."
            )
        if not text.strip() and not tool_calls:
            raise EmptyCompletion(f"{self.model} returned an empty response.")

        # The thinking block, keyed under the exact field the provider used,
        # so the assistant message we persist echoes it back verbatim on the
        # next call. The field must be present even when EMPTY (opencode PR
        # #24146: DeepSeek demands the key on the echo, "" value included).
        reasoning_msg = ({reasoning_key: reasoning}
                         if reasoning_key is not None else None)
        return text, tool_calls, usage, reasoning_msg

    _TOOL_ROUND_TIMEOUT_S = 60.0

    async def _run_tools(self, tool_calls: list[dict]) -> list[dict]:
        """Execute a round; preserve completed results when the deadline expires."""
        import local_tools as LT
        enabled = self._enabled_tools()

        def message(tc, text):
            return {"role": "tool", "tool_call_id": tc["id"], "content": text}

        async def run(tc):
            try:
                name = tc["function"]["name"]
                raw = tc["function"].get("arguments") or "{}"
                try:
                    args = json.loads(raw)
                except (ValueError, TypeError):
                    return [message(tc, f"ERROR: arguments were not valid JSON: {str(raw)[:200]}")]
                if not isinstance(args, dict):
                    return [message(tc, "ERROR: arguments must be a JSON object.")]
                if name.startswith("mcp__"):
                    if self._mcp is None:
                        return [message(tc, f"ERROR: MCP not available for {name!r} (not started or no servers).")]
                    out = await self._mcp.call(name, args)
                elif name == "ReadImage":
                    if name not in enabled:
                        return [message(tc, f"ERROR: tool {name!r} is not enabled in this phase.")]
                    if not self._supports_image():
                        return [message(tc, f"ERROR: ReadImage is unavailable -- model {self.model!r} does not support image input.")]
                    result = await LT.execute(name, args, enabled, write_roots=self._write_roots())
                    return [message(tc, LT._render(result)),
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": result["data_url"]}},
                                {"type": "text", "text": f"[image: {result['path']}]"},
                            ]}]
                else:
                    out = await LT.dispatch(name, args, enabled, write_roots=self._write_roots())
                return [message(tc, _truncate_tool_output(out, tc.get("id")))]
            except Exception as exc:
                prefix = "DENIED" if isinstance(exc, LT.ToolDenied) else "ERROR"
                return [message(tc, f"{prefix}: {type(exc).__name__}: {exc}")]

        if not tool_calls:
            return []
        tasks = [asyncio.create_task(run(tc)) for tc in tool_calls]
        try:
            done, pending = await asyncio.wait(tasks, timeout=self._TOOL_ROUND_TIMEOUT_S)
            if pending:
                _log(f"tool round timeout: {len(pending)}/{len(tasks)} unfinished calls")
            results = []
            for tc, task in zip(tool_calls, tasks):
                if task in done:
                    results.extend(task.result())
                else:
                    results.append(message(tc,
                        f"ERROR: tool round exceeded {self._TOOL_ROUND_TIMEOUT_S:g}s. "
                        "Cancellation requested; the operation may have completed or may still be running. "
                        "Check its outcome before retrying any side effect."))
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return ([m for m in results if m["role"] == "tool"]
                + [m for m in results if m["role"] != "tool"])

    @staticmethod
    def _usage_key(messages) -> tuple[str, int | None]:
        """(session, history chars) for the usage log. Session is a hash of
        the first non-system message -- it changes when compaction reseeds
        the wire. Chars cover everything except the system message, so the
        fitted intercept is system prompt + tool schemas = the base load."""
        try:
            import hashlib
            import json
            hist = [m for m in (messages or []) if m.get("role") != "system"]
            first = json.dumps(hist[0], ensure_ascii=False, default=str) if hist else ""
            sess = hashlib.sha1(first.encode()).hexdigest()[:8]
            return sess, len(json.dumps(hist, ensure_ascii=False, default=str))
        except Exception:
            return "", None

    def _record_usage(self, chat, usage, messages=None) -> None:
        total = getattr(usage, "prompt_tokens", 0) or 0
        try:
            import inputlog
            details = getattr(usage, "prompt_tokens_details", None)
            inputlog.usage_row({
                "round": inputlog._round,
                "prompt_tokens": total,
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "cached_tokens": getattr(details, "cached_tokens", None),
                "messages": len(messages or [])})
        except Exception:
            pass
        try:
            from expression import append_usage
            slug = (getattr(usage, "model", None)
                    or getattr(self, "submodel", None)
                    or getattr(self, "model", None))
            sess, chars = self._usage_key(messages)
            append_usage(self.name, total,
                         getattr(usage, "completion_tokens", 0) or 0, slug,
                         session=sess, prompt_chars=chars)
        except Exception:
            pass
        if context_diet.should_replace_context(chat.get("context_tokens") or 0,
                                              total):
            if chat.get("context_estimated"):
                try:
                    projected = chat.get("context_tokens") or 0
                    if projected:
                        err = (total - projected) / projected
                        context_diet.log_compaction(
                            f"{chat.get('id')}: projection check -- projected "
                            f"{projected:,}, actual {total:,} ({err:+.1%})")
                except Exception:
                    pass
            chat["context_tokens"] = total
            chat.pop("context_estimated", None)

    # compaction

    async def _acreate_retrying(self, **kwargs):
        """chat.completions.create with in-place backoff on rate-limit 429s."""
        return await self._retry_rate_limit(self._client.chat.completions.create,
                                          _RATE_LIMIT_BACKOFF, **kwargs)

    async def _retry_rate_limit(self, create, backoff, **kwargs):
        import random
        for i in range(len(backoff) + 1):
            try:
                return await create(**kwargs)
            except _RateLimitError:
                if i == len(backoff):
                    raise
                delay = backoff[i] * (1 + random.uniform(0, 0.3))
                _log(f"{kwargs.get('model')}: 429 rate-limited, retry "
                     f"{i + 1}/{len(backoff)} in {delay:.0f}s")
                await asyncio.sleep(delay)

    async def asummarize(self, prompt: str, max_tokens: int,
                         system: str | None = None,
                         model: str | None = None) -> str:
        """Handover note on the cheap submodel. No tools, no memory, no state."""
        last: Exception | None = None
        await self._refresh_credentials()
        sub = model or self.submodel
        # Cap reasoning at _COMPACT_THINKING via the same translation the
        # chat path uses (_one_call). The budgets below still carry headroom
        # in case the level isn't honoured or the provider has no mapping.
        extra = {}
        try:
            extra = CL.thinking_param(self.name, _COMPACT_THINKING)
        except Exception as exc:
            _log(f"summarize {sub}: thinking level not applied "
                 f"({type(exc).__name__}: {exc})")
        base = max_tokens + self.reasoning_headroom
        for attempt, budget in enumerate((base, base + self.reasoning_headroom * 2)):
            try:
                resp = await self._acreate_retrying(
                    model=sub,
                    messages=[
                        {"role": "system", "content": system or
                         "You write factual handover notes for another "
                         "assistant. Output only the note itself -- no "
                         "preamble, no commentary, no offer to help further."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=budget,
                    stream=False,
                    **extra,
                )
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()
                try:
                    det = getattr(resp.usage, "completion_tokens_details", None)
                    think = getattr(det, "reasoning_tokens", None) or 0
                    out = getattr(resp.usage, "completion_tokens", 0) or 0
                    _log(f"summarize {sub}: budget={budget} "
                         f"reasoning={think} visible={out - think} "
                         f"finish={choice.finish_reason}")
                except Exception:
                    pass
                if choice.finish_reason == "length" and not text:
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
                raise
            except Exception as exc:
                # Same deal as _one_call: a model swap can make the endpoint
                # 400 on a thinking knob that worked yesterday. Drop the
                # knob and retry bare rather than lose the compaction.
                if extra and _is_thinking_param_rejection(exc):
                    _log(f"summarize {sub}: thinking param rejected "
                         f"({type(exc).__name__}: {str(exc)[:200]}), "
                         f"retrying without it")
                    extra = {}
                    continue
                raise
        raise last  # unreachable

    def context_tokens(self, chat) -> int:
        return chat.get("context_tokens", 0)

    async def aclose(self) -> None:
        if self._mcp is not None:
            await self._mcp.aclose()
        await self._client.close()
