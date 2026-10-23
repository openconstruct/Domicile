"""LFM student with the full Bedroom surface, for bench-scale runs.

Same llama-server path as the per-milestone adapters (model's own
chat template, --jinja, tools array; whole observation history
replayed each turn), with every tool the multi runner offers: the
file five, Bash, the memory seven (mcp__-prefixed as the resident
sees them), WebSearch/WebFetch, and the social nine. Used by the
bench calibration and campaign baseline/final runs, where one
student must face every family.
"""
import json, urllib.request

URL = "http://127.0.0.1:8091/v1/chat/completions"
def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}
S = {"type": "string"}
# Domicile's real model-facing schemas (vendored; see bedroom/sync_domicile_schemas.py)
import json as _json
from pathlib import Path as _Path
_DOMICILE = _json.loads((_Path(__file__).resolve().parent.parent / "bedroom" / "domicile_tools.json").read_text())
TOOLS = _DOMICILE + [
    fn("mcp__memory__memory_search", "Search long-term memory.", {"query": S, "limit": {"type": "integer"}}, ["query"]),
    fn("mcp__memory__memory_add", "Store one self-contained fact in long-term memory.", {"text": S}, ["text"]),
    fn("mcp__memory__memory_list", "List stored memories, newest first.", {"limit": {"type": "integer"}}, []),
    fn("mcp__memory__memory_delete", "Delete one memory by its id.", {"memory_id": S}, ["memory_id"]),
    fn("mcp__mem2__mem2_search", "Search past conversations.", {"query": S, "limit": {"type": "integer"}}, ["query"]),
    fn("mcp__mem2__mem2_list_dates", "List the dates that have conversation logs.", {}, []),
    fn("mcp__mem2__mem2_get_chunk", "Read one conversation chunk in full.",
       {"source_file": S, "chunk_index": {"type": "integer"}}, ["source_file", "chunk_index"]),
]

def lfm_student_full(obs):
    content = obs["situation"]
    if "memory_listing" in obs:
        content += "\n\nCurrent memory listing:\n" + obs["memory_listing"]
    messages = [{"role": "user", "content": content}]
    for h in obs.get("history", []):
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": "call_0", "type": "function",
                                         "function": {"name": h["tool"],
                                                      "arguments": json.dumps(h.get("args", {}))}}]})
        messages.append({"role": "tool", "tool_call_id": "call_0", "content": str(h["result"])})
    body = {"messages": messages, "tools": TOOLS, "tool_choice": "auto",
            "temperature": 0.1, "max_tokens": 300}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=240))
    msg = r["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls:
        return {"text": (msg.get("content") or "")[:500]}
    f = calls[0]["function"]
    try:
        args = json.loads(f["arguments"])
    except Exception:
        args = {}
    return {"tool": f["name"], "args": args}
