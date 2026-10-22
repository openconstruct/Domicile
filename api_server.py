#!/usr/bin/env python3
"""Domicile HTTP API, uploads, and web UI."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import time
import uuid

from aiohttp import web

import config_loader as _CL

from paths import CCHAT_DATA, EXPRESSION_FILE, UPLOADS_DIR
from pathlib import Path

HOST = "0.0.0.0"
PORT = 8765
WEBUI_DIR = Path(__file__).resolve().parent / "webui"

MAX_UPLOAD = 25 * 1024 * 1024
def _nocache(resp):
    """Revalidate every load (ETag still gives 304s) so UI edits land without a hard refresh."""
    resp.headers["Cache-Control"] = "no-cache"
    return resp


_STATIC_FILES = ("index.html", "logo.svg", "marked.min.js", "purify.min.js")

OUTBOX_DIR = CCHAT_DATA / "outbox"
MAX_OUTBOUND_FILES = 4
MAX_OUTBOUND_BYTES = 200 * 1024 * 1024


SEND_FILE_RE = re.compile(
    r"\[\[\s*send_file\s*:\s*(?P<path>[^\]]+?)"
    r"(?:\s+as\s+(?P<name>[^\]/]+?))?\s*\]\]",
    re.IGNORECASE,
)


def _outbox_state(turn_id: str) -> dict:
    return {"buf": "", "files": [], "probs": [], "bytes": 0, "turn": turn_id}


def _stage_file(st: dict, match) -> None:
    raw = (match.group("path") or "").strip().strip("\"'")
    src = Path(raw).expanduser()
    name = (match.group("name") or "").strip() or src.name
    try:
        if len(st["files"]) >= MAX_OUTBOUND_FILES:
            raise ValueError(f"over the {MAX_OUTBOUND_FILES}-file limit")
        if not src.is_file():
            raise ValueError("no such file")
        size = src.stat().st_size
        if st["bytes"] + size > MAX_OUTBOUND_BYTES:
            raise ValueError(f"over the {MAX_OUTBOUND_BYTES // (1024 * 1024)} MB outbox cap")
        directory = OUTBOX_DIR / st["turn"]
        directory.mkdir(parents=True, exist_ok=True)
        dest = directory / name
        i = 1
        while dest.exists():
            stem, dot, ext = name.partition(".")
            dest = directory / (f"{stem}-{i}.{ext}" if dot else f"{name}-{i}")
            i += 1
        shutil.copy2(src, dest)
        st["bytes"] += size
        st["files"].append(dest.name)
    except (OSError, ValueError) as exc:
        st["probs"].append(f"{raw}: {exc}")


def _stage_attachments(text: str, turn_id: str):
    st = _outbox_state(turn_id)

    def stage(match):
        _stage_file(st, match)
        return ""

    clean = SEND_FILE_RE.sub(stage, text or "")
    return clean, st["files"], st["probs"]


def _resolve_history_files(chat: dict) -> None:
    """Read-time [[send_file]] resolution for /api/chat."""
    msgs = chat.get("messages") or []
    for msg in msgs:
        if msg.get("role") != "assistant":
            continue
        text = msg.get("text") or ""
        if "[[" not in text:
            continue
        out: list[str] = []
        last = 0
        for m in SEND_FILE_RE.finditer(text):
            out.append(text[last:m.start()])
            last = m.end()
            key = hashlib.sha1(m.group(0).encode("utf-8")).hexdigest()[:12]
            dest_dir = OUTBOX_DIR / key
            raw = (m.group("path") or "").strip().strip("'\"")
            src_f = Path(raw).expanduser()
            served: str | None = None
            try:
                if not src_f.is_file():
                    raise OSError(f"{raw}: no such file")
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / src_f.name
                if not dest.is_file():
                    shutil.copy2(src_f, dest)
                served = dest.name
            except OSError as exc:
                out.append(f"[send_file: {exc}]")
            if served:
                out.append(f"/files/{key}/{served}")
        if last == 0:
            continue
        out.append(text[last:])
        msg["text"] = "".join(out)

def _outbox_gc(max_age_s: int = 48 * 3600) -> None:
    now = time.time()
    for child in OUTBOX_DIR.iterdir():
        try:
            if now - child.stat().st_mtime > max_age_s:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        except OSError:
            continue

def _model_slug(agent_name: str) -> str:
    """API model id for an agent name: lowercase, letters/digits/dashes only."""
    slug = re.sub(r"[^a-z0-9]+", "-", agent_name.lower()).strip("-")
    return slug or "agent"


MODEL_LIVE = _model_slug(_CL.load_identity()["agent_name"])
MODEL_FRESH = MODEL_LIVE + "-fresh"
MODELS = (MODEL_LIVE, MODEL_FRESH)


def _now() -> int:
    return int(time.time())


def _prompt_from(messages: list, attached: bool) -> str:
    """Turn an OpenAI `messages` array into one prompt string."""
    if attached:
        for m in reversed(messages):
            if m.get("role") == "user":
                return _flatten(m.get("content"))
        return ""
    _ident = _CL.load_identity()
    names = {"system": "System",
             "assistant": _ident["agent_name"],
             "user": _ident["operator_name"]}
    lines = []
    for m in messages:
        role = m.get("role", "user")
        text = _flatten(m.get("content"))
        if not text:
            continue
        who = names.get(role, role)
        lines.append(f"{who}: {text}")
    return "\n\n".join(lines)


def _flatten(content) -> str:
    """OpenAI allows content as a string or a list of typed parts."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                if isinstance(part.get("text"), str):
                    out.append(part["text"])
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out).strip()
    return ""


def _completion(text: str, model: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _chunk(delta: str, model: str, cid: str, finish=None) -> str:
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": ({"content": delta} if delta else {}),
            "finish_reason": finish,
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"


class ApiServer:
    """Owns the aiohttp runner. Talks to cchat through the app object."""

    def __init__(self, app, host: str = HOST, port: int = PORT):
        self.app = app                      # the CCChat instance
        self.host = host
        self.port = port
        self._runner: web.AppRunner | None = None


    async def _models(self, request: web.Request) -> web.Response:
        return web.json_response({
            "object": "list",
            "data": [
                {"id": m, "object": "model", "created": _now(), "owned_by": "Domicile"}
                for m in MODELS
            ],
        })

    async def _completions(self, request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"error": {"message": "body is not valid JSON", "type": "invalid_request_error"}},
                status=400)

        if not isinstance(body, dict):
            return web.json_response({"error": {"message": "body must be a JSON object"}}, status=400)
        model = body.get("model") or MODEL_LIVE
        fresh = str(model).strip().lower() == MODEL_FRESH
        messages = body.get("messages") or []
        if (not isinstance(messages, list) or not messages
                or not all(isinstance(m, dict) for m in messages)):
            return web.json_response(
                {"error": {"message": "messages must be a non-empty array",
                           "type": "invalid_request_error"}}, status=400)

        prompt = _prompt_from(messages, attached=not fresh)
        if not prompt:
            return web.json_response(
                {"error": {"message": "no usable text in messages",
                           "type": "invalid_request_error"}}, status=400)

        if body.get("stream"):
            return await self._stream(request, prompt, fresh, model)
        text = await self.app.api_turn(prompt, fresh=fresh)
        turn_id = uuid.uuid4().hex[:12]
        text, files, problems = _stage_attachments(text, turn_id)
        if problems:
            text += "".join(f"\n\n[{prob}]" for prob in problems)
        if files:
            text += "".join(f"\n\n/files/{turn_id}/{n}" for n in files)
        return web.json_response(_completion(text, model))

    def _note_file(self, st, m) -> None:
        _stage_file(st, m)

    def _feed_delta(self, st, chunk: str, final: bool) -> str:
        buf = st["buf"] + chunk
        out: list[str] = []
        while True:
            m = SEND_FILE_RE.search(buf)
            if m:
                out.append(buf[:m.start()])
                self._note_file(st, m)
                buf = buf[m.end():]
                continue
            if final:
                out.append(buf)
                st["buf"] = ""
                break
            i = buf.rfind("[[")
            if i == -1 and buf.endswith("["):
                i = len(buf) - 1
            if i != -1 and len(buf) - i < 4096:
                out.append(buf[:i])
                st["buf"] = buf[i:]
                break
            out.append(buf)
            st["buf"] = ""
            break
        return "".join(out)

    async def _stream(self, request, prompt, fresh, model) -> web.StreamResponse:
        resp = web.StreamResponse(status=200, headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })
        await resp.prepare(request)
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        queue: asyncio.Queue = asyncio.Queue()
        sent = 0
        turn_id = uuid.uuid4().hex[:12]
        st = _outbox_state(turn_id)
        client_gone = False

        async def safe_write(data: bytes):
            nonlocal client_gone
            if client_gone:
                return
            try:
                await resp.write(data)
            except (ConnectionResetError, ConnectionError, BrokenPipeError):
                client_gone = True

        def on_delta(parts, pending, force=False):
            nonlocal sent
            full = "\n\n".join(list(parts) + ([pending] if pending else []))
            if len(full) > sent:
                emit = self._feed_delta(st, full[sent:], False)
                if emit:
                    queue.put_nowait(emit)
                sent = len(full)

        task = asyncio.ensure_future(
            self.app.api_turn(prompt, fresh=fresh, on_delta=on_delta))
        drain = None
        try:
            while True:
                drain = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait({drain, task},
                                             return_when=asyncio.FIRST_COMPLETED)
                if drain in done:
                    await safe_write(_chunk(drain.result(), model, cid).encode())
                    continue
                drain.cancel()
                while not queue.empty():
                    await safe_write(_chunk(queue.get_nowait(), model, cid).encode())
                final = task.result()
                if sent == 0 and final:
                    final = self._feed_delta(st, final, True)
                    await safe_write(_chunk(final, model, cid).encode())
                break
        except Exception as exc:
            await safe_write(_chunk(f"\n\n**API error:** {exc}", model, cid).encode())
        finally:
            pending = [task] + ([drain] if drain is not None else [])
            for future in pending:
                if not future.done():
                    future.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        tail = self._feed_delta(st, "", True)
        if tail:
            await safe_write(_chunk(tail, model, cid).encode())
        if st["files"] or st["probs"]:
            note = "".join(f"\n\n/files/{st['turn']}/{n}" for n in st["files"])
            note += "".join(f"\n\n[{prob}]" for prob in st["probs"])
            await safe_write(_chunk(note, model, cid).encode())
        await safe_write(_chunk("", model, cid, finish="stop").encode())
        await safe_write(b"data: [DONE]\n\n")
        return resp


    async def _index(self, request: web.Request) -> web.StreamResponse:
        return _nocache(web.FileResponse(WEBUI_DIR / "index.html"))

    async def _identity(self, request: web.Request) -> web.Response:
        ident = _CL.load_identity()
        return web.json_response({
            "operator_name": ident["operator_name"],
            "agent_name": ident["agent_name"],
            "model": MODEL_LIVE,
        })

    async def _expression(self, request: web.Request) -> web.Response:
        try:
            text = EXPRESSION_FILE.read_text(encoding="utf-8")
        except OSError:
            text = ""
        return web.Response(text=text, content_type="text/plain")

    async def _telemetry(self, request: web.Request) -> web.Response:
        try:
            import expression
            snap = await asyncio.get_event_loop().run_in_executor(
                None, expression.telemetry_snapshot)
        except Exception as exc:
            return web.json_response(
                {"error": {"message": f"telemetry failed: {exc}"}}, status=500)
        return web.json_response(snap)

    async def _upload(self, request: web.Request) -> web.Response:
        """Accept file(s) from the web UI, save them under UPLOADS_DIR, then hand the agent a turn announcing them."""
        try:
            reader = await request.multipart()
        except Exception:
            return web.json_response(
                {"error": {"message": "expected multipart/form-data"}}, status=400)

        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        note = ""
        saved: list[dict] = []
        try:
            async for part in reader:
                if part.name == "note":
                    note = (await part.text())[:2000]
                    continue
                if part.name != "file":
                    continue

                raw = Path(part.filename or "").name.strip()
                if not raw or raw.startswith(".") or "/" in raw or "\\" in raw:
                    return web.json_response(
                        {"error": {"message": f"rejected filename: {part.filename!r}"}},
                        status=400)

                dest = (UPLOADS_DIR / raw).resolve()
                if dest.parent != UPLOADS_DIR.resolve():
                    return web.json_response(
                        {"error": {"message": "path escapes uploads dir"}}, status=400)

                stamp = time.strftime("%Y%m%d-%H%M%S")
                dest = UPLOADS_DIR / f"{stamp}-{raw}"
                n = 1
                while dest.exists():
                    dest = UPLOADS_DIR / f"{stamp}-{n}-{raw}"
                    n += 1

                size = 0
                try:
                    with open(dest, "wb") as fh:
                        while True:
                            chunk = await part.read_chunk(64 * 1024)
                            if not chunk:
                                break
                            size += len(chunk)
                            if size > MAX_UPLOAD:
                                fh.close()
                                dest.unlink(missing_ok=True)
                                return web.json_response(
                                    {"error": {"message":
                                     f"{raw} exceeds {MAX_UPLOAD // (1024*1024)} MB"}},
                                    status=413)
                            fh.write(chunk)
                    dest.chmod(0o600)
                except OSError as exc:
                    return web.json_response(
                        {"error": {"message": f"write failed: {exc}"}}, status=500)

                ctype = part.headers.get("Content-Type") or "application/octet-stream"
                saved.append({"path": dest, "size": size, "media_type": ctype})
        except Exception as exc:
            return web.json_response(
                {"error": {"message": f"upload failed: {exc}"}}, status=500)

        if not saved:
            return web.json_response(
                {"error": {"message": "no file part in request"}}, status=400)

        try:
            reply = await self.app.upload_turn(saved, note=note)
        except Exception as exc:
            return web.json_response(
                {"error": {"message": f"turn failed: {exc}"},
                 "files": [p["path"].name for p in saved]}, status=500)

        return web.json_response({
            "ok": True,
            "files": [{"name": p["path"].name, "size": p["size"]} for p in saved],
            "reply": reply,
        })

    async def _session(self, request: web.Request) -> web.Response:
        try:
            return web.json_response(self.app.session_state())
        except Exception as exc:
            return web.json_response(
                {"error": {"message": f"session read failed: {exc}"}}, status=500)

    async def _session_set(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"error": {"message": "body is not valid JSON"}}, status=400)
        if not isinstance(body, dict):
            return web.json_response(
                {"error": {"message": "body must be a JSON object"}}, status=400)
        try:
            state = self.app.set_session(
                provider=body.get("provider"),
                model=body.get("model"),
                mode=body.get("mode"),
                thinking=body.get("thinking"),
            )
        except (ValueError, RuntimeError) as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=400)
        except Exception as exc:
            return web.json_response(
                {"error": {"message": f"switch failed: {exc}"}}, status=500)
        return web.json_response(state)

    async def _chat(self, request: web.Request) -> web.Response:
        """Return the current live chat, resolved exactly like a turn would."""
        try:
            chat, _screen = self.app._bridge_target()
        except Exception as exc:
            return web.json_response(
                {"error": {"message": f"failed to resolve chat: {exc}"}}, status=500)
        payload = {
            "id": chat.get("id"),
            "title": chat.get("title"),
            "model": chat.get("model"),
            "created": chat.get("created"),
            "updated": chat.get("updated"),
            "messages": [dict(m) for m in chat.get("messages", [])],
        }
        _resolve_history_files(payload)
        return web.json_response(payload)

    async def _file(self, request: web.Request) -> web.StreamResponse:
        turn = request.match_info["turn"]
        name = request.match_info["name"]
        if (not re.fullmatch(r"[0-9a-f]{12}", turn) or not name
                or "/" in name or "\\" in name or name in (".", "..")):
            raise web.HTTPNotFound
        base = (OUTBOX_DIR / turn).resolve()
        target = (base / name).resolve()
        if base not in target.parents:
            raise web.HTTPNotFound
        if not target.is_file():
            raise web.HTTPNotFound
        return web.FileResponse(target)


    async def start(self) -> None:
        server = web.Application(client_max_size=MAX_UPLOAD + 1024 * 1024)
        server.router.add_get("/", self._index)
        server.router.add_get("/v1/models", self._models)
        server.router.add_post("/v1/chat/completions", self._completions)
        server.router.add_get("/api/identity", self._identity)
        server.router.add_get("/api/expression", self._expression)
        server.router.add_get("/api/chat", self._chat)
        server.router.add_get("/api/session", self._session)
        server.router.add_post("/api/session", self._session_set)
        server.router.add_get("/api/telemetry", self._telemetry)
        server.router.add_post("/api/upload", self._upload)
        server.router.add_get("/ui", self._index)
        for name in _STATIC_FILES:
            p = WEBUI_DIR / name
            if p.is_file():
                server.router.add_get(f"/{name}", lambda r, p=p: _nocache(web.FileResponse(p)))
        OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
        _outbox_gc()
        server.router.add_get("/files/{turn}/{name}", self._file)
        server.router.add_get("/health", lambda r: web.json_response({"status": "ok"}))
        self._runner = web.AppRunner(server)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
