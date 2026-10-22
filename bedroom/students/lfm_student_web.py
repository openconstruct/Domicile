"""LFM student for M3 web episodes: multi-aware server-path adapter.

Same llama-server path as lfm_student.py (model's own chat template,
--jinja, tools array), extended for the multi-round runner: the whole
observation history is replayed each turn as messages -- the student's
own tool calls as assistant tool_calls, the runner's results as tool
messages -- so the model sees what its searches and fetches returned.

Tools: the web pair (WebSearch / WebFetch, Domicile's registry names)
plus the memory / mem2 / Write subset the web scenarios exercise.
"""
import json, urllib.request

URL = "http://127.0.0.1:8091/v1/chat/completions"
def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}
S = {"type": "string"}
TOOLS = [
    fn("WebSearch", "Search the web. Returns titled result blocks with URLs and snippets.",
       {"query": S, "count": {"type": "integer"}}, ["query"]),
    fn("WebFetch", "Fetch the full body of a page by its exact URL.", {"url": S}, ["url"]),
    fn("mcp__memory__memory_search", "Search long-term memory semantically.", {"query": S, "limit": {"type": "integer"}}, ["query"]),
    fn("mcp__memory__memory_add", "Store one self-contained, third-person fact in long-term memory.", {"text": S}, ["text"]),
    fn("mcp__memory__memory_list", "List stored memories, newest first.", {"limit": {"type": "integer"}}, []),
    fn("mcp__memory__memory_delete", "Delete one memory by id.", {"memory_id": S}, ["memory_id"]),
    fn("mcp__mem2__mem2_search", "Search past conversations.", {"query": S, "limit": {"type": "integer"}}, ["query"]),
    fn("Write", "Write a file, overwriting it entirely.", {"path": S, "content": S}, ["path", "content"]),
    fn("Read", "Read a file.", {"path": S}, ["path"]),
]

def lfm_student_web(obs):
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
