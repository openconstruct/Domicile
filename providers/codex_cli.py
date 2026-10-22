"""Codex CLI adapter: ChatGPT-subscription auth via the installed `codex` binary,
with Domicile's own tools served by codex_tools_server.py (an MCP server).

Mirrors cchat.py's stream_claude, not OpenAICompatibleProvider: codex keeps its
own conversation state (resume by thread id) and runs its own tool loop before
returning one finished turn, so there is no message wire to replay.

Only Domicile's tools are allowed (Read/Write/Edit/Bash/WebSearch/WebFetch/...,
the same local_tools.py code every provider runs, with the same phase gating).
Codex's native tools are removed:
  * shell, browser and computer-use by --disable flags;
  * native web_search by `-c web_search=disabled`;
  * apply_patch and native subagents by a model-catalog override (they are
    per-model catalog properties with no switch of their own), plus more
    --disable flags.
The one residual native surface is `functions.exec`, a locked-down V8 isolate
with no require/process/fetch/console, whose `tools` object holds only our
tools, a read-only clock, and MCP-resource readers that can only see our
server. Fail closed: if the lockdown cannot be applied (no catalog, or a model
missing from it), the turn refuses to run instead of running open.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .base import Provider, EmptyCompletion

import context_diet  # cchat root is on sys.path, same as cchat.py itself


def turn_usage_from_cumulative(chat: dict, cum_in: int, cum_out: int,
                               thread_id: str | None) -> tuple[int, int, bool]:
    """This turn's usage, derived from codex's running thread totals.

    `codex exec` reports `turn.completed.usage` as the *thread's* cumulative
    figures, not the turn's: the same prompt run twice on one thread reports
    roughly double the second time. Treated as per-turn numbers they inflate
    the context figure without bound on a long thread.

    So the values worth keeping are the differences against the totals we saw
    last turn. With no previous sighting of this thread there is nothing to
    difference against: returns zeros and first_sighting=True, so the caller
    marks the context figure estimated rather than publish a thread lifetime as
    a context size. Records the new totals in the chat either way.
    """
    prev_in = chat.get("codex_cum_in") or 0
    prev_out = chat.get("codex_cum_out") or 0
    same_thread = chat.get("codex_cum_thread") == thread_id
    chat["codex_cum_in"] = cum_in
    chat["codex_cum_out"] = cum_out
    chat["codex_cum_thread"] = thread_id
    if not cum_in and not cum_out:
        return 0, 0, False
    if prev_in and same_thread and cum_in >= prev_in:
        return cum_in - prev_in, max(cum_out - prev_out, 0), False
    return 0, 0, True

CODEX_BIN = shutil.which("codex") or "codex"
TOOLS_SERVER = str(Path(__file__).resolve().parent.parent / "codex_tools_server.py")

# Harmless stderr noise on most invocations; never treated as an error.
_STDERR_NOISE = "failed to refresh available models"


def _write_roots_env(chat) -> str:
    """JSON list of writable roots for the tools server, '' = unrestricted.

    A role whose sandbox config is broken yields "[]" (nothing writable),
    never "" -- a bad config must not turn the sandbox off.
    """
    import config_loader as _CL
    try:
        roots = _CL.write_roots_for_role((chat or {}).get("mode", "chat"))
    except Exception:
        return "[]"
    return "" if roots is None else json.dumps([str(r) for r in roots])


def _toolwake_env(chat) -> str:
    """Pass the resolved wakeup permission to Codex's separate tools server."""
    import config_loader as _CL
    try:
        roots = _CL.write_roots_for_role((chat or {}).get("mode", "chat"))
    except Exception:
        return "0"
    return "1" if roots is not None and getattr(roots, "toolwake", False) else "0"


def _toml_str(s: str) -> str:
    """A python str as a valid TOML/JSON double-quoted string literal."""
    return json.dumps(s)


def _toml_list(items: list[str]) -> str:
    return "[" + ", ".join(_toml_str(s) for s in items) + "]"


# --- native-tool lockdown ----------------------------------------------------
# Domicile's rule: every agent uses only Domicile tools. Codex ships native
# tools we can't allow. How the lockdown works (codex-cli 0.156.x):
#   * `apply_patch` has no config switch and no --disable flag. It is a
#     per-model property of codex's model catalog (`apply_patch_tool_type`).
#     It can write files as a native `file_change` even when the model's own
#     tool listing omits it (a model's self-reported tool list is not evidence).
#   * Native subagents (`collaboration.*`) survive `--disable multi_agent`
#     because the catalog sets `multi_agent_version`.
#   * The rest is removable with --disable.
# So: hand codex a catalog copy (`-c model_catalog_json=`) with both fields
# nulled on every model, plus the --disable list below. Result: the model sees
# only functions.wait + functions.exec, and inside exec `tools` holds only
# `clock__curr_time` (read-only; no flag known to remove it) next to our
# domicile_tools MCP server. Fail closed: no catalog, or a model that is not in it
# (codex would use degraded "fallback metadata"), means no turn.

_NATIVE_DISABLES = (
    "shell_tool", "browser_use", "computer_use",
    "multi_agent", "multi_agent_v2", "apps", "goals", "image_generation",
    "in_app_browser", "tool_suggest", "view_image", "sleep_tool",
)
_CATALOG_NULLED = ("apply_patch_tool_type", "multi_agent_version")
_CATALOG_TTL = 600.0  # seconds before re-reading `codex debug models`


def _effort_options(chat, provider: str) -> list[str]:
    """`-c model_reasoning_effort=<level>` for this turn's thinking level.

    Sent only if the level is one config declares for codex -- "auto" or
    anything stale means send nothing and let the model's catalog default
    stand. An unknown value is rejected by codex with a 400.
    """
    level = (chat or {}).get("thinking") or ""
    if not level:
        return []
    try:
        import config_loader as CL
        if level not in CL.thinking_levels(provider):
            return []
    except Exception:
        return []
    return ["-c", f"model_reasoning_effort={_toml_str(level)}"]


def _image_options(paths: list[str]) -> list[str]:
    """One `--image=<path>` per image. NOT `-i <path>`: -i takes a variable
    number of values, so in codex-cli 0.156.1 the positional prompt that follows
    is swallowed as another image path and codex exits 1 with "No prompt
    provided via stdin". `--image=` and `-i p -- prompt` both work, and
    `--image=` also survives the `resume <id>` subcommand."""
    return [f"--image={p}" for p in paths]


class CodexLockdownError(RuntimeError):
    """The native-tool lockdown cannot be applied; refuse rather than run open."""


def _catalog_path() -> Path:
    from paths import CCHAT_DATA
    return CCHAT_DATA / "codex-catalog-lockdown.json"


def build_lockdown_catalog(raw: dict) -> dict:
    """Copy of codex's model catalog with the native-tool fields nulled on EVERY
    model (so a catalog refresh can't bring a model back with a patch tool)."""
    models = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(models, list) or not models:
        raise CodexLockdownError("codex model catalog has no models")
    out = []
    for m in models:
        if not isinstance(m, dict) or not m.get("slug"):
            raise CodexLockdownError("codex model catalog has a malformed entry")
        m = dict(m)
        for key in _CATALOG_NULLED:
            m[key] = None
        out.append(m)
    return {**raw, "models": out}


def _catalog_is_locked(cat) -> bool:
    models = cat.get("models") if isinstance(cat, dict) else None
    return bool(models) and all(
        isinstance(m, dict) and m.get("slug")
        and all(m.get(k) is None for k in _CATALOG_NULLED) for m in models)


_catalog_memo: dict = {"at": 0.0, "slugs": frozenset(), "path": None}


def lockdown_catalog(force: bool = False) -> tuple[str, frozenset]:
    """(path to the lockdown catalog, its model slugs). Blocking (~0.7s when it
    has to call `codex debug models`) -- call via asyncio.to_thread."""
    now = time.monotonic()
    path = _catalog_path()
    memo = _catalog_memo
    if (not force and memo["path"] == str(path) and path.exists()
            and now - memo["at"] < _CATALOG_TTL):
        return memo["path"], memo["slugs"]
    try:
        proc = subprocess.run([CODEX_BIN, "debug", "models"], capture_output=True,
                              text=True, timeout=30, check=True)
        cat = build_lockdown_catalog(json.loads(proc.stdout))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(cat))
        os.replace(tmp, path)
    except Exception as exc:
        # Last good copy is acceptable, but only if it is still verifiably locked.
        try:
            cat = json.loads(path.read_text())
        except Exception:
            cat = None
        if not _catalog_is_locked(cat):
            raise CodexLockdownError(
                f"cannot build codex lockdown catalog ({type(exc).__name__}: {exc})"
            ) from exc
    slugs = frozenset(m["slug"] for m in cat["models"])
    memo.update(at=now, slugs=slugs, path=str(path))
    return str(path), slugs


def lockdown_options(model: str) -> list[str]:
    """`codex exec` options that remove codex's native tools for `model`."""
    path, slugs = lockdown_catalog()
    if model not in slugs:
        raise CodexLockdownError(
            f"model {model!r} is not in codex's catalog; running it would use "
            "degraded fallback metadata with the native-tool lockdown unverified")
    opts = ["-c", f"model_catalog_json={_toml_str(path)}"]
    for feature in _NATIVE_DISABLES:
        opts += ["--disable", feature]
    return opts


class CodexCLIProvider(Provider):
    """Shells out to `codex exec` / `codex exec resume`, JSONL on stdout."""

    name = "codex"

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg or {}

    # config

    def _phase(self, chat) -> str:
        import config_loader as CL
        return CL.tools_for_role(chat.get("mode", "chat"))

    def _model(self, chat) -> str:
        import config_loader as CL
        if chat.get("model"):
            return chat["model"]
        role = CL.model_role_for(chat.get("mode", "chat"))
        models = CL.models_for(self.name, role) or CL.models_for(self.name, "chat")
        if models:
            return models[0]
        block_models = (self.cfg.get("models") or {}).get("chat") or [""]
        return block_models[0]

    def _system_prompt(self, chat) -> str:
        import cchat
        system = cchat.SYSTEM_PROMPT
        if cchat.MEM1 is not None:
            try:
                block, stats = cchat.MEM1.load(role=chat.get("mode", "chat"),
                                               model_override=chat.get("model"))
                if block:
                    system = f"{system}\n\n{block}"
                chat["_mem1"] = stats
            except Exception as exc:
                chat["_mem1"] = {"error": repr(exc)}
        system += (
            "\n\n[codex_cli path] You have no native shell, apply-patch, "
            "browser, or computer-use tool, and no native web search -- "
            "those are disabled for this session. Your only tools are on "
            "the 'domicile_tools' MCP server: the same Read/Glob/Grep/Write/"
            "Edit/Bash/ToolWake/WebSearch/WebFetch/ExpressToPane/SetWakeup "
            "implementations the Claude-backed path uses, gated by the "
            "current role. Use those, not any built-in capability, for "
            "every file, shell, or web action."
        )
        return system

    def _base_options(self, chat, model: str) -> list[str]:
        phase = self._phase(chat)
        opts = [
            "--json", "--skip-git-repo-check",
            "-C", str(Path.home()),
            "-m", model,
            "-c", f"web_search={_toml_str('disabled')}",
            "-c", f"mcp_servers.domicile_tools.command={_toml_str('python3')}",
            "-c", f"mcp_servers.domicile_tools.args={_toml_list([TOOLS_SERVER])}",
            "-c", f"mcp_servers.domicile_tools.env.DOMICILE_TOOL_PHASE={_toml_str(phase)}",
            "-c", f"mcp_servers.domicile_tools.env.DOMICILE_WRITE_ROOTS={_toml_str(_write_roots_env(chat))}",
            "-c", f"mcp_servers.domicile_tools.env.DOMICILE_TOOLWAKE={_toml_str(_toolwake_env(chat))}",
            "-c", f"instructions={_toml_str(self._system_prompt(chat))}",
            *lockdown_options(model),
            "--approve-for-me",
        ]
        opts += _effort_options(chat, self.name)
        return opts

    # images

    def _write_temp_images(self, images) -> tuple[list[str], str | None]:
        if not images:
            return [], None
        tmpdir = tempfile.mkdtemp(prefix="domicile-codex-img-")
        paths = []
        for i, img in enumerate(images):
            ext = (img.media_type or "image/png").split("/")[-1] or "png"
            p = Path(tmpdir) / f"img{i}.{ext}"
            p.write_bytes(img.data)
            paths.append(str(p))
        return paths, tmpdir

    # memory hints (mirrors OpenAICompatibleProvider._memory_hints)

    async def _memory_hints(self, text: str) -> str:
        query = (text or "").strip()
        if not query:
            return ""
        try:
            import mem0_core
            hits = await asyncio.wait_for(
                asyncio.to_thread(mem0_core.hint, query[:2000]), timeout=30.0)
        except Exception:
            return ""
        if not hits:
            return ""
        lines = []
        for score, made, mem in hits:
            stamp = f"{made} " if made else ""
            tag = "[kw] " if score <= 0 else f"[{score:.2f}] "
            lines.append(f"- {tag}{stamp}{mem}")
        return ("[memory hints -- auto-retrieved from long-term memory by "
                "similarity to the message below. Possibly relevant, "
                "possibly noise; call memory_search for full detail before "
                "relying on one.]\n" + "\n".join(lines))

    # subprocess

    async def _run(self, argv: list[str], on_event) -> int:
        """Spawn `codex`, feed each parsed JSONL event to on_event. Returns the exit code."""
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path.home()),
        )

        async def _drain_stderr():
            try:
                async for _ in proc.stderr:
                    pass  # noisy, cosmetic (model-refresh timeouts); discarded
            except Exception:
                pass

        drainer = asyncio.create_task(_drain_stderr())
        try:
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                on_event(event)
            return await proc.wait()
        except asyncio.CancelledError:
            proc.kill()
            await asyncio.gather(proc.wait(), return_exceptions=True)
            raise
        finally:
            drainer.cancel()
            await asyncio.gather(drainer, return_exceptions=True)

    # turn

    async def astream_turn(self, chat, text, on_delta=None, images=None,
                           source=""):
        import cchat

        model = self._model(chat)
        try:  # fail closed before doing any other work for this turn
            await asyncio.to_thread(lockdown_catalog)
            lockdown_options(model)
        except CodexLockdownError as exc:
            return (f"**Error:** codex native-tool lockdown unavailable, "
                    f"refusing to run: {exc}"), "error"
        stamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
        src = f" — {source}" if source else ""
        clean = f"[{stamp} — {cchat.OPERATOR_USER}{src}]\n{text}"

        seed = chat.get("pending_seed")
        if not seed and not chat.get("codex_thread_id") and chat.get("messages") \
                and cchat.COMPACT is not None:
            try:
                seed = cchat.COMPACT.recovery_seed(chat["messages"]) or None
            except Exception:
                pass

        hinted = await self._memory_hints(text) if source else ""
        hud = cchat.domicile_hud(chat, model)
        prompt = ((f"{seed}\n\n" if seed else "")
                  + (f"{hinted}\n\n" if hinted else "")
                  + clean
                  + (f"\n\n{hud}" if hud else ""))

        img_paths, img_tmpdir = self._write_temp_images(images)

        opts = self._base_options(chat, model)
        opts += _image_options(img_paths)

        bridge = None
        if cchat.midturn_enabled() and chat.get("id") != "api-fresh" \
                and cchat.MIDTURN.open():
            from midturn import MidTurnSocketBridge
            try:
                bridge = await MidTurnSocketBridge(
                    cchat.MIDTURN, cchat._midturn_tool_notice,
                    lambda e: cchat.append_message(
                        chat, "user", f"[mid-turn · {e.source}] {e.text}"),
                ).start()
                opts += ["-c", "mcp_servers.domicile_tools.env.DOMICILE_MIDTURN_SOCKET="
                         + _toml_str(bridge.path)]
                for i, option in enumerate(opts):
                    if option.startswith("instructions="):
                        base = json.loads(option.split("=", 1)[1])
                        opts[i] = "instructions=" + _toml_str(
                            base + "\n\n" + cchat._midturn_tool_instruction())
                        break
            except Exception:
                if bridge is not None:
                    await bridge.close()
                    bridge = None
                cchat.MIDTURN.close()  # no bridge: callers take normal path

        # All options -- including --approve-for-me and -C -- must precede
        # the `resume <id>` positional: --approve-for-me is a flag on `codex exec`
        # itself, not on `resume`. Placed after `resume <id>` the flags either
        # hard-error (-C, --approve-for-me) or silently fail to unblock MCP
        # tool-call approval.
        thread_id = chat.get("codex_thread_id")
        if thread_id:
            argv = [CODEX_BIN, "exec"] + opts + ["resume", thread_id, prompt]
        else:
            argv = [CODEX_BIN, "exec"] + opts + [prompt]

        parts: list[str] = []
        turn_in = turn_out = cum_in = 0
        status = "ok"
        errors: list[str] = []

        def on_event(ev: dict) -> None:
            nonlocal turn_in, turn_out, cum_in
            etype = ev.get("type")
            if etype == "thread.started":
                tid = ev.get("thread_id")
                if tid:
                    chat["codex_thread_id"] = tid
            elif etype == "item.completed":
                item = ev.get("item") or {}
                itype = item.get("type")
                if itype == "agent_message":
                    txt = item.get("text") or ""
                    if txt:
                        parts.append(txt)
                        if on_delta is not None:
                            on_delta(parts, "", force=True)
                elif itype == "error":
                    msg = item.get("message") or ""
                    if msg and "unrecognized configuration" not in msg.lower():
                        errors.append(msg)
            elif etype == "turn.completed":
                u = ev.get("usage") or {}
                # Thread-cumulative, so keep the running maximum and difference
                # it once the stream ends (see turn_input_from_cumulative).
                cum_in = max(cum_in, (u.get("input_tokens", 0)
                                      + u.get("cached_input_tokens", 0)))
                turn_out = max(turn_out, (u.get("output_tokens", 0)
                                          + u.get("reasoning_output_tokens", 0)))
            elif etype == "error":
                msg = ev.get("message") or ""
                if msg and "unrecognized configuration" not in msg.lower():
                    errors.append(msg)

        try:
            rc = await self._run(argv, on_event)
            if rc != 0 and not parts:
                parts.append(
                    f"**Error:** `codex exec` exited {rc}"
                    + (f": {errors[-1]}" if errors else ""))
                status = "error"
        except asyncio.CancelledError:
            status = "cancelled"
        except Exception as exc:
            parts.append(f"**Error:** `{type(exc).__name__}: {exc}`")
            status = "error"
        finally:
            if bridge is not None:
                await bridge.close()
                cchat.MIDTURN.close()  # undelivered entries fall back normally
            if img_tmpdir:
                try:
                    shutil.rmtree(img_tmpdir, ignore_errors=True)
                except Exception:
                    pass

        turn_in, turn_out, first_sighting = turn_usage_from_cumulative(
            chat, cum_in, turn_out, chat.get("codex_thread_id"))
        if turn_in and context_diet.plausible_context(turn_in):
            chat["context_tokens"] = turn_in
            chat.pop("context_estimated", None)
        elif cum_in:
            # First turn on this thread as far as this chat knows: the running
            # total covers the whole thread, so it says nothing about us. If
            # what's stored is a thread lifetime from before this fix, drop it
            # rather than leave a wrong number on screen.
            if not context_diet.plausible_context(
                    chat.get("context_tokens") or 0):
                chat["context_tokens"] = 0
            chat["context_estimated"] = True

        try:
            from expression import append_usage
            append_usage(self.name, turn_in, turn_out, model,
                         session=(chat.get("codex_thread_id") or "")[:8])
        except Exception:
            pass

        if seed and status == "ok":
            chat.pop("pending_seed", None)
            chat.pop("precompact_session_id", None)

        if status == "cancelled":
            parts.append("*(interrupted)*")

        return ("\n\n".join(p for p in parts if p) or "*(no reply)*"), status

    # compaction

    async def asummarize(self, prompt: str, max_tokens: int,
                         system: str | None = None,
                         model: str | None = None,
                         effort: str | None = None) -> str:
        import config_loader as CL
        sub = model or (CL.models_for(self.name, "compact") or [""])[0]
        if not sub:
            raise RuntimeError(f"provider {self.name!r} has no compact model configured")

        await asyncio.to_thread(lockdown_catalog)
        instructions = system or (
            "You write factual handover notes for another assistant. "
            "Output only the note itself -- no preamble, no commentary, no "
            "offer to help further."
        )
        opts = [
            "--json", "--skip-git-repo-check",
            "-C", str(Path.home()),
            "-m", sub,
            "-c", f"web_search={_toml_str('disabled')}",
            "-c", f"instructions={_toml_str(instructions)}",
            *lockdown_options(sub),
            *_effort_options({"thinking": effort}, self.name),
            "-s", "read-only",
        ]
        argv = [CODEX_BIN, "exec"] + opts + [prompt]

        parts: list[str] = []
        errors: list[str] = []

        def on_event(ev: dict) -> None:
            etype = ev.get("type")
            if etype == "item.completed":
                item = ev.get("item") or {}
                if item.get("type") == "agent_message":
                    txt = item.get("text") or ""
                    if txt:
                        parts.append(txt)
                elif item.get("type") == "error":
                    msg = item.get("message") or ""
                    if msg:
                        errors.append(msg)

        rc = await self._run(argv, on_event)
        text = "\n\n".join(p for p in parts if p).strip()
        if not text:
            detail = f": {errors[-1]}" if errors else f" (exit {rc})"
            raise EmptyCompletion(f"{sub} returned an empty summary{detail}.")
        return text

    def context_tokens(self, chat) -> int:
        return chat.get("context_tokens", 0)

    async def aclose(self) -> None:
        pass  # each turn is a fresh subprocess; nothing long-lived to release


# --- codex options this adapter relies on --------------------------------
#   -m <model>                       model override (exec and resume)
#   -c web_search="disabled"         removes the hosted web_search tool; the value
#                                    is a string (disabled/cached/indexed/live)
#   -c instructions="..."            full system-prompt override
#   -c mcp_servers.<n>.command/.args/.env.*    per-call MCP server injection
#   --disable shell_tool/browser_use/computer_use/multi_agent/apps/goals/
#     image_generation/in_app_browser/tool_suggest/view_image/sleep_tool
#   --approve-for-me                 the only non-interactive way to get MCP tool
#                                    calls approved; sets workspace-write and
#                                    codex's own auto-review, exclusive with -s.
#                                    Must precede `resume <id>` (see above).
#   -s read-only                     denies native writes but also every MCP call
#   -c model_catalog_json="<file>"   replaces the model catalog. Nulling
#                                    apply_patch_tool_type and multi_agent_version
#                                    on every model removes native apply_patch and
#                                    subagents; a model missing from the catalog
#                                    gets degraded metadata, so the turn is refused.
#   codex debug models               prints the raw catalog JSON
#   --image=<path>                   not `-i <path>`: -i is variadic and swallows
#                                    the prompt that follows
# Not supported by codex (ignored or rejected): per-server approval_policy or
# trusted keys, approval_policy=granular, tools.web_search, mcp_approval_policy,
# mcp_tool_approval_policy, tool_approval_policy, mcp_servers_trust. Re-test
# against a newer codex-cli before relying on any of them.
