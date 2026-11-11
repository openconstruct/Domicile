# Fact extraction and reconciliation (Rook R1: the prompts carry the
# quality — ported verbatim in prompts.py). The llm is any callable
# (system_prompt, user_prompt) -> str; Domicile wires its own inference.
import json
import re

from .prompts_v3 import (
    ADDITIVE_EXTRACTION_PROMPT,
    AGENT_CONTEXT_SUFFIX,
    generate_additive_extraction_prompt,
)
from .prompts import (
    AGENT_MEMORY_EXTRACTION_PROMPT,
    DEFAULT_UPDATE_MEMORY_PROMPT,
    USER_MEMORY_EXTRACTION_PROMPT,
)


def parse_messages(messages):
    if isinstance(messages, str):
        return messages
    out = ""
    for msg in messages:
        content = msg.get("content")
        if content is None:
            continue
        out += f"{msg.get('role', 'user')}: {content}\n"
    return out


def parse_json(text):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def extract_facts(messages, llm, agent_scoped=False, custom_instructions=None):
    system = AGENT_MEMORY_EXTRACTION_PROMPT if agent_scoped else USER_MEMORY_EXTRACTION_PROMPT
    if custom_instructions:
        system += f"\n\nCustom instructions:\n{custom_instructions}"
    if "json" not in system.lower():
        system += ("\n\nYou must return your response in valid JSON format "
                   "with a 'facts' key containing an array of strings.")
    response = llm(system, f"Input:\n{parse_messages(messages)}")
    data = parse_json(response)
    return [str(f) for f in data.get("facts", [])]


def reconcile(facts, existing, llm):
    # existing: [{"id":..., "text":...}] retrieved as context for the facts.
    current = [{"id": e["id"], "text": e["text"]} for e in existing]
    user = (
        "Current memory:\n"
        + json.dumps(current, indent=2)
        + "\n\nNew extracted facts:\n"
        + json.dumps(facts, indent=2)
        + "\n\nReturn the operations as JSON with a 'memory' key: a list of "
        '{"id", "text", "event" (ADD/UPDATE/DELETE/NONE), "old_memory"} objects. '
        "For ADD use a fresh placeholder id."
    )
    response = llm(DEFAULT_UPDATE_MEMORY_PROMPT, user)
    data = parse_json(response)
    return data.get("memory", [])


# --- verbatim from installed mem0 2.0.19 mem0/memory/utils.py
def remove_code_blocks(content: str) -> str:
    """
    Removes enclosing code block markers ```[language] and ``` from a given string.

    Remarks:
    - The function uses a regex pattern to match code blocks that may start with ``` followed by an optional language tag (letters or numbers) and end with ```.
    - If a code block is detected, it returns only the inner content, stripping out the markers.
    - If no code block markers are found, the original content is returned as-is.
    """
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        content = "".join(parts)
    if not isinstance(content, str):
        return ""
    pattern = r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$"
    match = re.match(pattern, content.strip())
    match_res=match.group(1).strip() if match else content.strip()
    return re.sub(r"<think>.*?</think>", "", match_res, flags=re.DOTALL).strip()


def extract_json(text):
    """
    Extracts JSON content from a string, removing enclosing triple backticks and optional 'json' tag if present.
    If no code block is found, attempts to locate JSON by finding the first '{' and last '}'.
    If that also fails, returns the text as-is.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx : end_idx + 1]
        else:
            json_str = text
    return json_str


def extract_additive(parsed_messages, existing_rows, llm, agent_scoped=False,
                     custom_instructions=None, last_k_messages=None):
    """2.0.19's single ADD-only extraction call. existing_rows are shown to
    the model under integer ids (upstream's anti-hallucination mapping).
    LLM failures propagate (Domicile's local edit: callers do provider
    fallback); only unparseable output degrades to no facts."""
    existing = [{"id": str(i), "text": r["text"]} for i, r in enumerate(existing_rows)]
    system = ADDITIVE_EXTRACTION_PROMPT + (AGENT_CONTEXT_SUFFIX if agent_scoped else "")
    user = generate_additive_extraction_prompt(
        existing_memories=existing, new_messages=parsed_messages,
        last_k_messages=last_k_messages or [], custom_instructions=custom_instructions)
    response = llm(system, user)
    try:
        response = remove_code_blocks(response)
        if not response or not response.strip():
            return []
        try:
            return json.loads(response, strict=False).get("memory", [])
        except json.JSONDecodeError:
            return json.loads(extract_json(response), strict=False).get("memory", [])
    except Exception:
        return []
