"""LFM student for M4 social episodes: multi-aware server-path adapter.

Same llama-server path as lfm_student_web.py (model's own chat template,
--jinja, tools array; whole observation history replayed each turn), with
the M4 tool surface: Domicile's nine Google tools (GmailSearch, GmailRead,
GmailSend, GmailDraft, CalendarList, CalendarCreate, BloggerPost, ChatRead,
ChatSend) plus the standard set the multi runner offers.

The Google/Web schemas are NOT hand-written here: they are loaded from
bedroom/domicile_tools.json, which sync_domicile_schemas exports from
local_tools. Hand-copied descriptions drifted from the real ones once
already; the model must be shown exactly what Domicile shows it.
"""
import json, urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8091/v1/chat/completions"
_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "bedroom" / "domicile_tools.json"
_DOMICILE = {t["function"]["name"]: t for t in json.loads(_SCHEMA_PATH.read_text())}


def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}


def real(*names):
    return [_DOMICILE[n] for n in names]


S = {"type": "string"}
TOOLS = (
    real("GmailSearch", "GmailRead", "GmailSend", "GmailDraft",
         "CalendarList", "CalendarCreate", "BloggerPost", "ChatRead", "ChatSend",
         "WebSearch", "WebFetch", "Read", "Write")
    + [
        fn("mcp__memory__memory_search", "Search long-term memory semantically.", {"query": S, "limit": {"type": "integer"}}, ["query"]),
        fn("mcp__memory__memory_add", "Store one self-contained, third-person fact in long-term memory.", {"text": S}, ["text"]),
        fn("mcp__memory__memory_list", "List stored memories, newest first.", {"limit": {"type": "integer"}}, []),
        fn("mcp__memory__memory_delete", "Delete one memory by id.", {"memory_id": S}, ["memory_id"]),
        fn("mcp__mem2__mem2_search", "Search past conversations.", {"query": S, "limit": {"type": "integer"}}, ["query"]),
    ]
)

def lfm_student_social(obs):
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
