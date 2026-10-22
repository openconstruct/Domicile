"""LFM student for M2 episodes: same server path as lfm_student.py, with
the memory tools added in their resident form (mcp__-prefixed names are
legal function names, and that is the surface the resident sees)."""
import json, urllib.request

URL = "http://127.0.0.1:8091/v1/chat/completions"
def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}
S = {"type": "string"}
TOOLS = [
    fn("mcp__memory__memory_search", "Search long-term memory semantically.", {"query": S, "limit": {"type": "integer"}}, ["query"]),
    fn("mcp__memory__memory_add", "Store one self-contained fact in long-term memory.", {"text": S}, ["text"]),
    fn("mcp__memory__memory_list", "List stored memories, newest first.", {"limit": {"type": "integer"}}, []),
    fn("mcp__memory__memory_delete", "Delete one memory by id.", {"memory_id": S}, ["memory_id"]),
    fn("mcp__mem2__mem2_search", "Search past conversations.", {"query": S, "limit": {"type": "integer"}, "date_from": S, "date_to": S}, ["query"]),
    fn("mcp__mem2__mem2_list_dates", "List dates with available conversation logs.", {}, []),
    fn("mcp__mem2__mem2_get_chunk", "Get the full text of a specific conversation chunk.", {"source_file": S, "chunk_index": {"type": "integer"}}, ["source_file", "chunk_index"]),
    fn("Read", "Read a file.", {"path": S}, ["path"]),
    fn("Write", "Write a file, overwriting entirely.", {"path": S, "content": S}, ["path", "content"]),
    fn("Edit", "Replace an exact unique string in a file.", {"path": S, "old": S, "new": S}, ["path", "old", "new"]),
]

def lfm_student_m2(obs):
    content = obs["situation"]
    if "memory_listing" in obs:
        content += "\n\nCurrent memory listing:\n" + obs["memory_listing"]
    body = {"messages": [{"role": "user", "content": content}],
            "tools": TOOLS, "tool_choice": "auto", "temperature": 0.1, "max_tokens": 300}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=240))
    msg = r["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls:
        return {"text": (msg.get("content") or "")[:120]}
    f = calls[0]["function"]
    return {"tool": f["name"], "args": json.loads(f["arguments"])}
