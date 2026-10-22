"""LFM student for M5 bash episodes: multi-aware server-path adapter.

Same llama-server path as lfm_student_social.py (model's own chat
template, --jinja, tools array; whole observation history replayed each
turn), with the M5 tool surface: Domicile's Bash (an argv list, no
shell) plus the file tools and the simulated Google/Discord nine the
multi runner offers, so the demo beat can run a bash scenario and a
social scenario back-to-back through the curriculum runner.
"""
import json, urllib.request

URL = "http://127.0.0.1:8091/v1/chat/completions"
def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}
S = {"type": "string"}
TOOLS = [
    fn("Bash", "Run one command from the house's allowed subset (ls, cat, grep, wc, sort, uniq, head, tail, mv, cp, mkdir, echo, sed, find) in the sandbox. Takes an argv list with no shell, so no pipes, redirects, globs or &&; arguments are literal.",
       {"argv": {"type": "array", "items": S}}, ["argv"]),
    fn("Read", "Read a file.", {"path": S}, ["path"]),
    fn("Write", "Write a file, overwriting it entirely.", {"path": S, "content": S}, ["path", "content"]),
    fn("Edit", "Replace one unique passage in a file.", {"path": S, "old": S, "new": S}, ["path", "old", "new"]),
    fn("gmail_search", "Search the inbox. Returns message blocks with ids, senders, subjects and snippets.",
       {"query": S, "count": {"type": "integer"}}, ["query"]),
    fn("gmail_read", "Read one inbox message in full, by its id.", {"id": S}, ["id"]),
    fn("gmail_send", "Send an email.", {"to": S, "subject": S, "body": S}, ["to", "subject", "body"]),
    fn("gmail_draft", "Save an email as a draft for review instead of sending it.",
       {"to": S, "subject": S, "body": S}, ["to", "subject", "body"]),
    fn("calendar_list", "List calendar events with ids, titles and start/end times.", {}, []),
    fn("calendar_create", "Create a calendar event. Start and end are ISO strings like 2026-10-23T12:30.",
       {"title": S, "start": S, "end": S}, ["title", "start", "end"]),
    fn("blogger_post", "Publish a post to the Blogger blog.", {"title": S, "body": S}, ["title", "body"]),
    fn("discord_read", "Read recent messages from a Discord channel.", {"channel": S, "limit": {"type": "integer"}}, ["channel"]),
    fn("discord_post", "Post a message to a Discord channel.", {"channel": S, "text": S}, ["channel", "text"]),
]

def lfm_student_bash(obs):
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
