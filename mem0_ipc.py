#!/usr/bin/env python3
"""Client and server for memory IPC."""

from __future__ import annotations

import asyncio
import json
import os
import socket
from pathlib import Path

from paths import SOCKET_PATH

CALL_TIMEOUT = 120.0


class Mem0IpcDown(RuntimeError):
    """The IPC call did not complete. cchat may or may not be alive."""


class Mem0IpcNoListener(Mem0IpcDown):
    """connect() itself failed: nothing is bound to the socket at all."""


# server

def _dispatch_to_core(op: str, args: dict):
    import mem0_core as core

    if op == "hint":
        return core.hint(
            args.get("text", ""),
            threshold=float(args.get("threshold", core.HINT_THRESHOLD)),
            limit=int(args.get("limit", core.HINT_LIMIT)),
        )
    if op == "search":
        return core.search(args.get("query", ""), int(args.get("limit", 5)))
    if op == "add":
        return core.add(
            args.get("text", ""),
            extract=bool(args.get("extract", True)),
            metadata=args.get("metadata"),
            idempotency_key=args.get("idempotency_key"),
            guard=bool(args.get("guard", False)),
        )
    if op == "list":
        return core.list_memories(int(args.get("limit", 20)))
    if op == "list_facts":
        return core.list_facts(include_superseded=bool(args.get("include_superseded", False)))
    if op == "delete":
        return core.delete(args.get("memory_id", ""))
    if op == "add_verbatim":
        return core.add_verbatim(args.get("text", ""), metadata=args.get("metadata"))
    if op == "mark_superseded":
        return core.mark_superseded(
            args.get("drop_id", ""), args.get("keep_id", ""),
            reason=args.get("reason", ""),
        )
    raise ValueError(f"unknown op {op!r}")


async def serve(dispatch=_dispatch_to_core) -> asyncio.Server:
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SOCKET_PATH.exists():
        if await _endpoint_alive():
            raise RuntimeError(
                f"{SOCKET_PATH} already served by another Domicile instance")
        SOCKET_PATH.unlink()  # crash leftover: probed dead above, safe to clear

    async def handle(reader: asyncio.StreamReader,
                     writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                reply = await _handle_line(line, dispatch)
                writer.write(reply)
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass  # client vanished or shutdown: nothing to report to
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    server = await asyncio.start_unix_server(handle, path=str(SOCKET_PATH))
    os.chmod(SOCKET_PATH, 0o600)
    return server


async def _handle_line(line: bytes, dispatch) -> bytes:
    req_id = None
    try:
        req = json.loads(line)
        req_id = req.get("id")
        result = await asyncio.to_thread(dispatch, req.get("op"),
                                         dict(req.get("args") or {}))
        resp = {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001 - errors travel as data
        resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if req_id is not None:
        resp["id"] = req_id
    return (json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8")


async def _endpoint_alive() -> bool:
    try:
        _, writer = await asyncio.open_unix_connection(str(SOCKET_PATH))
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        return False
    writer.close()
    return True


# client

def call(op: str, args: dict | None = None, timeout: float = CALL_TIMEOUT,
         path: Path | None = None):
    """One synchronous request over one connection."""
    target = Path(path or SOCKET_PATH)
    payload = (json.dumps({"op": op, "args": args or {}, "id": 1})
               + "\n").encode("utf-8")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        try:
            sock.connect(str(target))
        except OSError as exc:
            raise Mem0IpcNoListener(f"{type(exc).__name__}: {exc}") from exc
        sock.sendall(payload)
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
    except OSError as exc:
        raise Mem0IpcDown(f"{type(exc).__name__}: {exc}") from exc
    finally:
        sock.close()
    if not buf.strip():
        raise Mem0IpcDown("empty response from memory IPC")
    try:
        resp = json.loads(buf)
    except ValueError as exc:
        raise Mem0IpcDown(f"malformed response: {buf[:200]!r}") from exc
    if not isinstance(resp, dict) or "ok" not in resp:
        raise Mem0IpcDown(f"unrecognized response shape: {buf[:200]!r}")
    if not resp["ok"]:
        raise RuntimeError(resp.get("error") or "unknown store error")
    return resp.get("result")
