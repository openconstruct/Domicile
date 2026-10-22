"""LFM student adapter v2 -- server path (the format fix, 2026-10-18).

Talks to llama-server running the model's OWN chat template (--jinja)
with the tools presented as a proper tools array. The model answers with
structured tool_calls; no output scraping, no JSON pleading, no native
bracket-syntax parsing (that was v1, in git-less scratch -- retired).

Start the server:
  llama-server -m LFM2.5-2.6B-Q4_K_M.gguf -t 2 --jinja --host 127.0.0.1 --port 8091

Tools below are Domicile's real schemas (local_tools._SCHEMAS), FS subset.
"""
import json, urllib.request

URL = "http://127.0.0.1:8091/v1/chat/completions"
TOOLS = [
    {"type": "function", "function": {"name": "Read", "description": "Read a file. Returns numbered lines.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "Write", "description": "Write a file, overwriting it entirely. Must Read an existing file first.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "Edit", "description": "Replace an exact string in a file. Must be unique unless replace_all is set.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}, "replace_all": {"type": "boolean"}}, "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {"name": "Glob", "description": "Find files by glob pattern, newest first.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "Grep", "description": "Search file contents with ripgrep regex.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"]}}, "required": ["pattern"]}}},
]

def lfm_student(obs):
    body = {"messages": [{"role": "user", "content": obs["situation"]}],
            "tools": TOOLS, "tool_choice": "auto", "temperature": 0.1, "max_tokens": 300}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=240))
    msg = r["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls:
        return {"text": (msg.get("content") or "")[:120]}
    fn = calls[0]["function"]
    return {"tool": fn["name"], "args": json.loads(fn["arguments"])}
