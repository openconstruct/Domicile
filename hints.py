"""The one ambient-hint renderer.

Every memory hint that reaches a prompt is rendered here. There were three
copies of this code -- `cchat._ambient_hints` for the Claude path and one in
each non-Claude provider -- and they had already drifted: cchat clipped each
memory to a 20-word teaser, the providers sent the full text, so the same query
produced a different prompt depending on which provider happened to be active.
On 2026-11-08 that drift cost a day: the mem0-v2 shadow hook was added to
cchat's copy, which never runs on deepseek, and nothing looked broken because
nothing errored.

The clipped form wins. It was the deliberate one (a hint block is a teaser, not
an answer; the full text is a `memory_search` call away) and it is what keeps
per-turn overhead bounded as the store grows.
"""

HINT_SNIPPET_WORDS = 20

HEADER = ("[memory hints — auto-retrieved from long-term memory by similarity "
          "to the message below. Possibly relevant, possibly noise; call "
          "memory_search for full detail before relying on one.]")


def snippet(mem: str, limit: int = HINT_SNIPPET_WORDS) -> str:
    """Clip a stored memory to a short teaser."""
    words = (mem or "").split()
    if len(words) <= limit:
        return mem or ""
    return " ".join(words[:limit]) + " …"


def render(hits) -> str:
    """The hint block for a list of (score, made_date, memory); '' when empty."""
    hits = list(hits or [])
    if not hits:
        return ""
    lines = []
    for score, made, mem in hits:
        stamp = f"{made} " if made else ""
        tag = "[kw] " if score <= 0 else f"[{score:.2f}] "
        lines.append(f"- {tag}{stamp}{snippet(mem)}")
    return HEADER + "\n" + "\n".join(lines)
