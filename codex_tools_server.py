#!/usr/bin/env python3
"""Stdio MCP server exposing cchat's own local tools to the Codex CLI.

Codex has its own native shell/apply-patch/browser/computer-use tools; those
are disabled by the codex_cli provider (see providers/codex_cli.py) so that
tool calls go through here instead -- the same fs_read/fs_write/exec_argv/
web_search/web_fetch/express_to_pane implementations the Claude path uses via
local_tools.py, with the same phase gating (readonly/write/full/none) and the
same mutation lock and result-size trimming.

Which tools are registered is fixed at process start from DOMICILE_TOOL_PHASE
(same values as roles.json's `tools` field: "full", "write", "readonly",
"none", or a comma-separated explicit list). One subprocess per turn, spawned
fresh by codex_cli.py with the phase for that turn.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MEM0_TELEMETRY", "false")

from mcp.server import MCPServer

import local_tools as LT
from midturn import take_from_socket

_phase_env = os.environ.get("DOMICILE_TOOL_PHASE", "readonly")
# Fail closed: a set-but-unparseable DOMICILE_WRITE_ROOTS means "sandboxed, nowhere
# writable", never "unrestricted". Unset means unrestricted (build mode).
_wr = os.environ.get("DOMICILE_WRITE_ROOTS")
if _wr is None or _wr == "":
    WRITE_ROOTS = None
else:
    import json as _json
    from pathlib import Path as _Path
    try:
        _entries = _json.loads(_wr)
        if not isinstance(_entries, list) or not all(isinstance(x, str) for x in _entries):
            raise ValueError("DOMICILE_WRITE_ROOTS must be a list of paths")
        from config_loader import Roots as _Roots
        WRITE_ROOTS = _Roots(_Path(x) for x in _entries)
        WRITE_ROOTS.toolwake = os.environ.get("DOMICILE_TOOLWAKE") == "1"
    except Exception:
        WRITE_ROOTS = []

if "," in _phase_env:
    ENABLED = LT.resolve_phase([t.strip() for t in _phase_env.split(",") if t.strip()])
else:
    ENABLED = LT.resolve_phase(_phase_env)

mcp = MCPServer("domicile_tools")


def _register(name):
    """Build typed MCP wrappers from the same schema used by chat providers."""
    import inspect

    spec = LT.schemas([name])[0]["function"]
    schema = spec["parameters"]
    required = schema.get("required", ())
    types = {"string": str, "integer": int, "number": float,
             "boolean": bool, "array": list[str]}
    defaults = {"offset": 1, "count": 5, "max_chars": 12000, "replace_all": False}
    parameters = []
    for key, field in schema["properties"].items():
        annotation = types[field["type"]]
        default = inspect.Parameter.empty if key in required else defaults.get(key)
        if default is None:
            annotation = annotation | None
        parameters.append(inspect.Parameter(
            key, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation))

    async def call(**args):
        if name == "SetWakeup":
            args = {k: v for k, v in args.items() if v is not None}
        out = await LT.dispatch(name, args, ENABLED, write_roots=WRITE_ROOTS)
        socket = os.environ.get("DOMICILE_MIDTURN_SOCKET")
        if socket:
            notices = await take_from_socket(socket)
            if notices:
                out += "\n\n" + "\n\n".join(notices)
        return out

    call.__name__ = name
    call.__signature__ = inspect.Signature(parameters, return_annotation=str)
    mcp.add_tool(call, name=name, description=spec["description"])
    return call


for _name in ENABLED:
    # ReadImage uses multimodal output on the chat path; not a text MCP tool.
    if _name != "ReadImage":
        globals()[_name] = _register(_name)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print(f"phase={_phase_env!r} enabled={ENABLED}")
        raise SystemExit(0)
    mcp.run()
