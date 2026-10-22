"""Conversation compaction and tool-result budgeting."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from pathlib import Path

# tuning

KEEP_THINKING_LAST_N_TURNS = 3

TOOL_RESULT_MAX_CHARS = 2000   # the wire path (non-Claude providers) still uses these
TOOL_RESULT_HEAD = 1200
TOOL_RESULT_TAIL = 600

# Claude/SDK resume path only. Cuts on this path carry a recovery command,
# see fetch_result.py.
SDK_TOOL_RESULT_MAX_CHARS = 6000
SDK_TOOL_RESULT_HEAD = 4500
SDK_TOOL_RESULT_TAIL = 1500

MEM_TOOL_RESULT_MAX_CHARS = 10_000
MEM_TOOL_RESULT_HEAD = 8_000
MEM_TOOL_RESULT_TAIL = 2_000
def _is_read_tool(name) -> bool:
    """The built-in `Read` or our in-process `mcp__cchat__Read`."""
    return name == "Read" or (isinstance(name, str) and name.endswith("__Read"))


_MEM_TOOL_MARKERS = (
    "memory_search", "memory_list", "memory_add",
    "mem2_search", "mem2_get_chunk", "mem2_list_dates",
)

KEEP_TOOL_RESULTS_LAST_N_TURNS = 6

KEEP_ATTACHMENTS_LAST_N = {
    "edited_text_file": 2,
    "hook_success": 1,
    "total_tokens_reminder": 1,
    "deferred_tools_delta": 2,
    "skill_listing": 1,
    "agent_listing_delta": 1,
}

KEEP_ATTACHMENTS_DEFAULT = 4

from paths import SESSIONS_DIR as ROOT, CONTEXT_DIET_LOG as LOG, \
    MEM0_LOG_DIR as DAILY_LOG_DIR


WIRE_TOOL_RESULT_KEEP_LAST = 8   # tool results to leave whole; older get 2k head+tail
WIRE_IMAGE_KEEP_LAST = 1       # image_url blocks to keep; older become a placeholder
WIRE_MAX_CHARS_BEFORE_DIET = 40_000  # ~10k tokens; diet is cheap enough to run often
WIRE_MAX_CHARS_AFTER_DIET = 10_000  # ~2.5k tokens; aggressive second pass if still too big


def enabled() -> bool:
    """Whether to filter on load. Off means this is a plain mirror."""
    return os.environ.get("CCHAT_CONTEXT_DIET", "1") not in ("0", "false", "no")


def _approx_tokens(obj) -> int:
    """Rough token count for reporting. Bytes/4, the usual English estimate."""
    return len(json.dumps(obj, default=str)) // 4


# filtering

FETCH_SCRIPT = Path(__file__).resolve().parent / "fetch_result.py"


def _truncate(text: str, max_chars: int = TOOL_RESULT_MAX_CHARS,
              head: int = TOOL_RESULT_HEAD, tail: int = TOOL_RESULT_TAIL,
              tool_use_id: str | None = None) -> str:
    """Shorten one oversized tool_result body, marking the gap.

    With a tool_use_id the marker says how to get the missing middle back:
    the offsets are into the original text, so the command fetches exactly
    what was cut.
    """
    if len(text) <= max_chars:
        return text
    cut = len(text) - head - tail
    recover = ""
    if tool_use_id:
        recover = (f" Recover the cut part: python3 {FETCH_SCRIPT} "
                   f"{tool_use_id} --start {head} --len {cut}")
    return (
        text[:head]
        + f"\n\n… [{cut:,} characters trimmed from context; "
          f"the full output is in the transcript on disk.{recover}] …\n\n"
        + text[-tail:]
    )


def _shrink_tool_result(block: dict, mem: bool = False) -> dict:
    """Truncate a tool_result's content, whatever shape it arrived in."""
    max_chars = MEM_TOOL_RESULT_MAX_CHARS if mem else SDK_TOOL_RESULT_MAX_CHARS
    head = MEM_TOOL_RESULT_HEAD if mem else SDK_TOOL_RESULT_HEAD
    tail = MEM_TOOL_RESULT_TAIL if mem else SDK_TOOL_RESULT_TAIL
    tid = block.get("tool_use_id")
    content = block.get("content")

    if isinstance(content, str):
        if len(content) <= max_chars:
            return block
        return {**block, "content": _truncate(content, max_chars, head, tail, tid)}

    if isinstance(content, list):
        out = []
        changed = False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if len(text) > max_chars:
                    out.append({**part, "text": _truncate(text, max_chars, head, tail, tid)})
                    changed = True
                    continue
            out.append(part)
        return {**block, "content": out} if changed else block

    return block


def _filter_message_content(content, drop_thinking: bool, shrink_results: bool,
                            spare_ids: frozenset = frozenset()):
    """Apply the two filters to one message's content list."""
    if not isinstance(content, list):
        return None

    out = []
    changed = False
    for block in content:
        if not isinstance(block, dict):
            out.append(block)
            continue

        kind = block.get("type")

        if drop_thinking and kind in ("thinking", "redacted_thinking"):
            changed = True
            continue

        if shrink_results and kind == "tool_result":
            shrunk = _shrink_tool_result(
                block, mem=block.get("tool_use_id") in spare_ids)
            if shrunk is not block:
                changed = True
            out.append(shrunk)
            continue

        out.append(block)

    if not changed:
        return None

    if not out:
        out = [{"type": "text", "text": "[reasoning omitted from context]"}]

    return out


def _is_turn_start(entry) -> bool:
    """A real conversational turn boundary."""
    if not isinstance(entry, dict) or entry.get("type") != "user":
        return False
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") != "tool_result"
            for b in content
        )
    return True


def _mem_tool_use_ids(entries) -> frozenset:
    """tool_use ids whose tool is a memory-retrieval tool (_MEM_TOOL_MARKERS)."""
    out = set()
    for e in entries:
        if not isinstance(e, dict) or e.get("type") != "assistant":
            continue
        content = (e.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if (isinstance(b, dict) and b.get("type") == "tool_use"
                    and b.get("id")
                    and any(m in (b.get("name") or "") for m in _MEM_TOOL_MARKERS)):
                out.add(b["id"])
    return frozenset(out)


def _recent_read_result_ids(entries) -> frozenset:
    """tool_use ids of the most recent Read of each distinct file path."""
    latest: dict[str, str] = {}
    for e in entries:
        if not isinstance(e, dict) or e.get("type") != "assistant":
            continue
        content = (e.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if (isinstance(b, dict) and b.get("type") == "tool_use"
                    and _is_read_tool(b.get("name")) and b.get("id")):
                inp = b.get("input") or {}
                fpath = inp.get("file_path") or inp.get("path")
                if fpath:
                    latest[fpath] = b["id"]  # later entry wins -> most recent
    return frozenset(latest.values())


def diet(entries: list[dict]) -> tuple[list[dict], dict]:
    """Trim a loaded transcript down to what a resume actually needs."""
    before = _approx_tokens(entries)

    user_positions = [
        i for i, e in enumerate(entries) if _is_turn_start(e)
    ]
    mem_ids = _mem_tool_use_ids(entries)
    read_ids = _recent_read_result_ids(entries)
    spare_ids = mem_ids | read_ids
    think_cutoff = (
        user_positions[-KEEP_THINKING_LAST_N_TURNS]
        if len(user_positions) >= KEEP_THINKING_LAST_N_TURNS else len(entries)
    )
    result_cutoff = (
        user_positions[-KEEP_TOOL_RESULTS_LAST_N_TURNS]
        if len(user_positions) >= KEEP_TOOL_RESULTS_LAST_N_TURNS else len(entries)
    )

    quota = dict(KEEP_ATTACHMENTS_LAST_N)
    seen: dict[str, int] = {}
    drop_indices: set[int] = set()
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        if not isinstance(entry, dict) or entry.get("type") != "attachment":
            continue
        kind = (entry.get("attachment") or {}).get("type", "?")
        limit = quota.get(kind, KEEP_ATTACHMENTS_DEFAULT)
        count = seen.get(kind, 0)
        if count >= limit:
            drop_indices.add(i)
        else:
            seen[kind] = count + 1

    remap: dict[str, str] = {}
    last_kept: str | None = None
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        uuid = entry.get("uuid")
        if i in drop_indices:
            if uuid:
                remap[uuid] = last_kept
        elif uuid:
            last_kept = uuid

    def reparent(entry: dict) -> dict:
        """Point an entry past any dropped ancestors."""
        parent = entry.get("parentUuid")
        if parent is None or parent not in remap:
            return entry
        seen_uuids = set()
        while parent in remap and parent not in seen_uuids:
            seen_uuids.add(parent)
            parent = remap[parent]
        return {**entry, "parentUuid": parent}

    out = []
    for i, entry in enumerate(entries):
        if i in drop_indices:
            continue

        if not isinstance(entry, dict):
            out.append(entry)
            continue

        entry = reparent(entry)
        message = entry.get("message")
        if not isinstance(message, dict):
            out.append(entry)
            continue

        new_content = _filter_message_content(
            message.get("content"),
            drop_thinking=i < think_cutoff,
            shrink_results=i < result_cutoff,
            spare_ids=spare_ids,
        )
        if new_content is None:
            out.append(entry)
        else:
            out.append({**entry, "message": {**message, "content": new_content}})

    after = _approx_tokens(out)
    return out, {
        "entries": len(entries),
        "kept": len(out),
        "turns": len(user_positions),
        "mem_results_spared": len(mem_ids),
        "reads_spared": len(read_ids),
        "attachments_dropped": len(drop_indices),
        "before": before,
        "after": after,
        "saved": before - after,
        "pct": round(100 * (before - after) / before) if before else 0,
    }


# store

class DietSessionStore:
    """A transcript mirror that serves a slimmed copy back on resume."""

    def __init__(self, root: Path = ROOT):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.last_stats: dict | None = None

    def _path(self, key) -> Path:
        project = str(key.get("project_key") or "default").replace("/", "-")
        session = str(key.get("session_id") or "unknown")
        subpath = key.get("subpath")
        directory = self.root / project
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{session}__{str(subpath).replace('/', '-')}" if subpath else session
        return directory / f"{name}.jsonl"

    async def append(self, key, entries) -> None:
        """Mirror a batch of transcript lines verbatim."""
        path = self._path(key)
        try:
            with path.open("a", encoding="utf-8") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:  # pragma: no cover - mirror is best-effort
            self._log(f"append failed for {path.name}: {exc!r}")

    async def load(self, key):
        """Return the session for resume, trimmed unless the diet is off."""
        path = self._path(key)
        if not path.exists():
            return None

        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        if not entries:
            return None

        if not enabled():
            self.last_stats = {"entries": len(entries), "skipped": True}
            self._log(f"{path.name}: diet disabled, loaded {len(entries)} entries")
            return entries

        trimmed, stats = diet(entries)
        self.last_stats = stats
        try:  # debug capture: exactly what the CLI is handed on resume
            import inputlog
            inputlog.write_jsonl("resumed.jsonl", trimmed)
        except Exception:
            pass
        self._log(
            f"{path.name}: {stats['entries']} entries, "
            f"~{stats['before']:,} -> ~{stats['after']:,} tokens "
            f"(saved ~{stats['saved']:,}, {stats['pct']}%)"
        )
        return trimmed

    async def list_sessions(self, project_key: str):
        directory = self.root / str(project_key).replace("/", "-")
        if not directory.is_dir():
            return []
        return [
            {
                "session_id": path.stem,
                "mtime": int(path.stat().st_mtime * 1000),
            }
            for path in directory.glob("*.jsonl")
            if "__" not in path.stem  # subagent transcripts are not sessions
        ]

    async def delete(self, key) -> None:
        """Remove a session, cascading to its subagent transcripts."""
        path = self._path(key)
        path.unlink(missing_ok=True)
        if not key.get("subpath"):
            for sub in path.parent.glob(f"{path.stem}__*.jsonl"):
                sub.unlink(missing_ok=True)

    async def list_subkeys(self, key):
        path = self._path({**key, "subpath": None})
        prefix = f"{path.stem}__"
        return [
            sub.stem[len(prefix):]
            for sub in path.parent.glob(f"{prefix}*.jsonl")
        ]

    def _log(self, line: str) -> None:
        try:
            with LOG.open("a", encoding="utf-8") as handle:
                handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {line}\n")
        except Exception:  # pragma: no cover - logging must never raise
            pass


# cli

# compaction

# Non-memory preamble (system prompt + tool defs). Memory is added separately
# via preamble_reserve(memory_tokens). A typical Claude base is 18-22k, of which
# mem1 is ~7k, so ~13k here is about right.
COMPACT_BASE_PREAMBLE = 13_000

COMPACT_SUMMARY_TOKENS = 6_000
COMPACT_TAIL_TOKENS = 5_000

DEFAULT_COMPACT_CEILING = 60_000

# The floor is the size of the COMPACTED text right after a compaction:
# summary + older verbatim tail. The base preamble (system prompt, tools,
# mem1) is not counted, and neither are the newest few exchanges, which
# ride on top (see COMPACT_PROTECTED_*). The tail is the variable part.
DEFAULT_COMPACT_FLOOR = 30_000
MIN_TAIL_TOKENS = 2_000
MIN_COMPACT_FLOOR = COMPACT_SUMMARY_TOKENS + MIN_TAIL_TOKENS      # 8_000

# The newest exchanges are always kept verbatim and never counted against
# the floor. Capped so a few huge turns can't bloat the post-compaction
# context; at least the newest exchange is kept whatever it weighs.
COMPACT_PROTECTED_EXCHANGES = 4
COMPACT_PROTECTED_CAP = 8_000

# Margin so a fresh post-compaction history is not already at the trigger
# and every turn compacts again: room for the protected block at its cap,
# plus real headroom for new turns. Ceiling >= floor + this.
COMPACT_HEADROOM = COMPACT_PROTECTED_CAP + 8_000                  # 16_000


def min_ceiling(floor: int | None = None) -> int:
    """Lowest loop-safe ceiling for a given floor (default floor if None)."""
    return (DEFAULT_COMPACT_FLOOR if floor is None else floor) + COMPACT_HEADROOM


MIN_COMPACT_CEILING = min_ceiling()                      # 30_000


def _floor_from_config() -> int:
    try:
        import config_loader as CL
        val = CL.context_setting("compact_floor_tokens")
    except Exception:  # noqa: BLE001
        return DEFAULT_COMPACT_FLOOR
    if not isinstance(val, (int, float)) or isinstance(val, bool) or val <= 0:
        return DEFAULT_COMPACT_FLOOR
    return max(MIN_COMPACT_FLOOR, int(val))


def compact_tail_tokens() -> int:
    """Older verbatim-tail budget implied by the configured floor.

    Floor minus the summary; the protected newest exchanges are extra.
    """
    return max(MIN_TAIL_TOKENS,
               _floor_from_config() - COMPACT_SUMMARY_TOKENS)


def _ceiling_from_config() -> int:
    floor_ceiling = min_ceiling(_floor_from_config())
    try:
        import config_loader as CL
        val = CL.context_setting("compact_ceiling_tokens")
    except Exception:  # noqa: BLE001
        return max(DEFAULT_COMPACT_CEILING, floor_ceiling)
    if not isinstance(val, (int, float)) or isinstance(val, bool) or val <= 0:
        return max(DEFAULT_COMPACT_CEILING, floor_ceiling)
    return max(floor_ceiling, int(val))


COMPACT_CEILING_TOKENS = _ceiling_from_config()


def preamble_reserve(memory_tokens: int = 0) -> int:
    """Tokens gone before any history: base preamble plus loaded memory."""
    return COMPACT_BASE_PREAMBLE + max(0, memory_tokens)



COMPACT_PROMPT = """\
Below is the earlier portion of a working conversation between {operator_name} \
and {agent_name}. It is being compacted to free context. Write a factual \
handover note, under {budget} words, using exactly these headings:

DECISIONS MADE
  Conclusions reached and settled. Include the reasoning where a later \
reader would otherwise reopen the question.

CURRENT STATE
  Files changed, what now exists, what is running or installed. Exact \
paths and identifiers.

OPEN THREADS
  Started and unfinished. What the next step was going to be.

UNRESOLVED PROBLEMS
  Known bugs, failures, and rejected approaches -- including WHY each was \
rejected, so it is not retried.

Preserve exact names, paths, numbers, error strings and commands. Prefer a \
specific detail over a general statement. Omit pleasantries and any \
narration of who said what. If a heading has no content, write "none".

CONVERSATION:
{conversation}
"""


def compact_enabled() -> bool:
    """On by default. CCHAT_COMPACT=0 disables it."""
    return os.environ.get("CCHAT_COMPACT", "1") not in ("0", "false", "no")


TOKENS_PER_ESTIMATE = 1.41


def text_tokens(text: str) -> int:
    """Estimated REAL tokens for plain prose."""
    return int(len(text) / 4 * TOKENS_PER_ESTIMATE)


def history_tokens(messages: list[dict]) -> int:
    """Estimated tokens of the conversation text alone (no base preamble).

    This is what the compaction ceiling is compared against. It counts
    message text, so tool output that never lands in a message's text is
    not included.
    """
    return sum(text_tokens(m.get("text") or "") for m in messages)


# Sanity bound for a provider-reported last-turn context figure. Generous: the
# biggest real window any provider we use offers is ~1M, plus base preamble.
# A wrong-but-huge number is worse than no number: it makes the HUD meaningless.
# Some providers (codex) report thread-cumulative usage, which can run to tens
# of millions on a long thread.
CONTEXT_SANITY_MAX = 2_000_000


def plausible_context(total: int) -> bool:
    """Whether a reported context figure could be a real last-turn input."""
    return bool(total) and 0 < total <= CONTEXT_SANITY_MAX


def should_replace_context(stored: int, fresh: int) -> bool:
    """Whether a provider's fresh total should replace the stored one.

    Raise-only is the default: on a given provider a smaller number is usually
    a stale or partial report, and lowering the HUD each turn is noise. The
    exception is a stored value that cannot be real: a stuck implausible number
    would otherwise refuse every later, smaller, correct report. A fresh,
    plausible number always beats an implausible stored one.
    """
    if not plausible_context(fresh):
        return False
    if not stored:
        return True
    return fresh > stored or not plausible_context(stored)


def context_split(chat: dict) -> dict | None:
    """Split the provider-reported last-turn total into base + history.

    `total` is chat["context_tokens"] (provider-reported, or a post-compaction
    projection when chat["context_estimated"] is set). `history` is our own
    estimate of the conversation text that turn actually carried: everything
    before the last assistant message, since that reply (and anything after
    it) was not in the last turn's input. `base` is the remainder: system
    prompt, memory files, tool definitions. Returns None with no total yet.
    """
    total = chat.get("context_tokens") or 0
    if not total:
        return None
    msgs = chat.get("messages") or []
    cut = len(msgs)
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "assistant":
            cut = i
            break
    history = min(history_tokens(msgs[:cut]), total)
    return {
        "total": total,
        "history": history,
        "base": total - history,
        "estimated": bool(chat.get("context_estimated")),
    }


def _protected_start(messages: list[dict]) -> int:
    """Index where the protected newest exchanges begin (len(messages) if none).

    Walks back from the end one exchange (user message onward) at a time, up
    to COMPACT_PROTECTED_EXCHANGES, stopping before an exchange that would
    push the block past COMPACT_PROTECTED_CAP. The newest exchange is always
    protected.
    """
    start = len(messages)
    spent = 0
    exchanges = 0
    for i in range(len(messages) - 1, -1, -1):
        spent += text_tokens(messages[i].get("text") or "")
        if messages[i].get("role") != "user":
            continue
        exchanges += 1
        if exchanges > 1 and spent > COMPACT_PROTECTED_CAP:
            break
        start = i
        if exchanges >= COMPACT_PROTECTED_EXCHANGES:
            break
    return start


def _render(messages: list[dict]) -> str:
    """Flatten role/text pairs into the transcript the summarizer reads."""
    import config_loader as _CL
    _ident = _CL.load_identity()
    op_name, ag_name = _ident["operator_name"], _ident["agent_name"]
    out = []
    for m in messages:
        if m.get("synthetic"):
            out.append(
                "[EARLIER HANDOVER NOTE -- this is already a compaction of "
                "everything before it. Carry its facts forward into the new "
                "note; do not drop them.]\n" + (m.get("text", "")))
            continue
        who = op_name if m.get("role") == "user" else ag_name
        out.append(f"{who}: {m.get('text', '')}")
    return "\n\n".join(out)


def split_for_compaction(
    messages: list[dict],
    tail_tokens: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Divide history into (to summarize, to keep verbatim).

    The newest exchanges (see _protected_start) are always kept and do not
    count against the budget. tail_tokens is the budget for the verbatim
    material OLDER than them; it defaults to the budget implied by the
    configured floor, resolved at call time (not import time) so config
    changes apply. The kept list is older tail + protected block.
    """
    if not messages:
        return [], []

    budget = compact_tail_tokens() if tail_tokens is None else tail_tokens
    p = _protected_start(messages)
    cut = p
    for i in range(p - 1, -1, -1):
        budget -= text_tokens(messages[i].get("text", ""))
        if budget <= 0:
            cut = i
            break
    else:
        return [], list(messages)  # whole history fits; nothing to summarize

    while cut < p and messages[cut].get("role") != "user":
        cut += 1

    if cut >= len(messages):
        cut = max(0, len(messages) - 2)

    return list(messages[:cut]), list(messages[cut:])


def append_daily_log(chat_id: str, title: str, head: list[dict]) -> None:
    """Append the about-to-be-summarized head to today's mem0 ingestion log."""
    if not head:
        return
    import config_loader as _CL
    _ident = _CL.load_identity()
    op_name, ag_name = _ident["operator_name"], _ident["agent_name"]
    DAILY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now()
    path = DAILY_LOG_DIR / f"{now.strftime('%Y-%m-%d')}.log"
    lines = [
        f"=== compacted from chat {chat_id!r} ({title!r}) at "
        f"{now.isoformat(timespec='seconds')} ==="
    ]
    for m in head:
        if m.get("synthetic"):
            continue  # a prior compaction's digest, not a raw turn -- mem0
        text = (m.get("text") or "").strip()
        if not text:
            continue
        who = op_name if m.get("role") == "user" else ag_name
        ts = (m.get("ts") or "")[:16].replace("T", " ")
        lines.append(f"[{ts}] {who}: {text}" if ts else f"{who}: {text}")
    if len(lines) == 1:
        return  # nothing but empty messages in this head
    with path.open("a") as f:
        f.write("\n".join(lines) + "\n\n")


async def build_seed(
    messages: list[dict], summarize, memory_tokens: int = 0
) -> tuple[str, dict]:
    """Summarize old history and return the seed for the next session."""
    head, tail = split_for_compaction(messages)
    if not head:
        raise ValueError("nothing to compact: history fits in the tail budget")

    conversation = _render(head)
    import config_loader as _CL
    _ident = _CL.load_identity()
    prompt = COMPACT_PROMPT.format(
        budget=int(COMPACT_SUMMARY_TOKENS * 0.75),
        conversation=conversation,
        operator_name=_ident["operator_name"],
        agent_name=_ident["agent_name"],
    )
    summary = await summarize(prompt, COMPACT_SUMMARY_TOKENS)

    seed = (
        "The earlier part of this conversation was compacted to free "
        "context. Here is the handover note, followed by the most recent "
        "exchanges verbatim. Continue as though you remember all of it.\n\n"
        "=== HANDOVER NOTE ===\n"
        f"{summary.strip()}\n\n"
        "=== RECENT EXCHANGES (verbatim) ===\n"
        f"{_render(tail)}\n"
        "=== END OF COMPACTED HISTORY ===\n"
    )

    stats = {
        "summarized": len(head),
        "kept": len(tail),
        "summary": summary.strip(),
        "head_tokens": text_tokens(conversation),
        "summary_tokens": text_tokens(summary),
        "tail_tokens": text_tokens(_render(tail)),
        "seed_tokens": text_tokens(seed),
        "memory_tokens": memory_tokens,
        "projected": preamble_reserve(memory_tokens) + text_tokens(seed),
    }
    return seed, stats


def recovery_seed(messages: list[dict], budget_tokens: int | None = None) -> str:
    """Rebuild conversational context from chat["messages"] when the normal continuity carrier is gone."""
    if not messages:
        return ""
    hist = list(messages)
    if hist and hist[-1].get("role") == "user":
        hist = hist[:-1]
    if not hist:
        return ""

    budget = budget_tokens or COMPACT_TAIL_TOKENS
    _, tail = split_for_compaction(hist, tail_tokens=budget)
    if not tail:
        tail = hist[-6:]
    dropped = len(hist) - len(tail)

    note = ""
    if dropped > 0:
        note = (f"[{dropped} earlier message(s) are omitted -- the session "
                f"lost its context mid-conversation and only the recent tail "
                f"could be recovered from disk.]\n\n")

    return (
        "This conversation continues from earlier. The session's live context "
        "was lost (a provider switch or an interrupted turn), so here are the "
        "most recent exchanges verbatim. Continue as though you remember the "
        "whole conversation.\n\n"
        "=== RECENT EXCHANGES (verbatim) ===\n"
        f"{note}{_render(tail)}\n"
        "=== END RECOVERED HISTORY ===\n"
    )


def collapse_after_compaction(
    messages: list[dict], summary: str, tail_keep: int
) -> list[dict]:
    """Fold the just-summarized head into one synthetic summary message."""
    if not summary or tail_keep <= 0 or len(messages) <= tail_keep:
        return list(messages)
    synth = {
        "role": "assistant",
        "text": summary,
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "synthetic": True,
    }
    return [synth] + list(messages[-tail_keep:])


def _is_image_content(content) -> bool:
    """True if this string/list content carries an image_url."""
    if isinstance(content, list):
        return any(isinstance(p, dict) and p.get("type") == "image_url" for p in content)
    if isinstance(content, str):
        return content.startswith("data:image/") or "data:image/" in content[:120]
    return False


def wire_tool_result_indices(wire: list[dict]) -> list[int]:
    """Indices of wire entries that are tool results (string content on role tool)."""
    out = []
    for i, m in enumerate(wire):
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            out.append(i)
        elif m.get("role") == "user" and _is_image_content(m.get("content")):
            out.append(i)
    return out


def wire_chars(wire: list[dict]) -> int:
    return sum(len(str(m.get("content") or "")) + len(str(m.get("tool_calls") or "")) for m in wire)


def _wire_truncate(m: dict, text: str, max_chars: int, head: int, tail: int) -> str:
    """_truncate for the wire path, with a recoverable marker.

    Cuts the ORIGINAL text so the offsets in the recovery command are exact:
    a result already trimmed at source is re-cut from its saved full copy.
    """
    key = None
    original = text
    try:
        import tool_spill
        found = tool_spill.KEY_IN_MARKER.search(text)
        if found:
            full = tool_spill.load(found.group(1))
            if full is not None:
                original, key = full, found.group(1)
        else:
            key = tool_spill.spill(m.get("tool_call_id"), text)
    except Exception:
        key = None
    return _truncate(original, max_chars, head, tail, key)


def _wire_spare_call_ids(wire: list[dict]) -> tuple[frozenset, int, int]:
    mem_ids: set[str] = set()
    latest_read: dict[str, str] = {}
    for m in wire:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            cid = tc.get("id")
            if not cid:
                continue
            if any(marker in name for marker in _MEM_TOOL_MARKERS):
                mem_ids.add(cid)
            elif _is_read_tool(name):
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (ValueError, TypeError):
                    args = {}
                fpath = args.get("file_path") or args.get("path")
                if fpath:
                    latest_read[fpath] = cid  # later entry wins -> most recent
    read_ids = set(latest_read.values())
    return frozenset(mem_ids | read_ids), len(mem_ids), len(read_ids)


def diet_wire(wire: list[dict]) -> tuple[list[dict], dict]:
    """Trim an OpenRouter wire in place-shaped, without breaking tool_use/tool_result pairing."""
    if not wire:
        return wire, {"before_chars": 0, "after_chars": 0, "saved_chars": 0,
                      "saved_tokens": 0, "before_tokens": 0, "after_tokens": 0,
                      "shrunk": 0, "images_dropped": 0, "kept_tail": 0,
                      "floor_shrunk": 0, "mem_results_spared": 0, "reads_spared": 0}

    before = wire_chars(wire)
    idxs = wire_tool_result_indices(wire)
    spare_ids, n_mem_spared, n_read_spared = _wire_spare_call_ids(wire)
    keep_set = set(idxs[-WIRE_TOOL_RESULT_KEEP_LAST:] if len(idxs) > WIRE_TOOL_RESULT_KEEP_LAST else idxs)
    image_replace_cutoff_idx = -1
    if WIRE_IMAGE_KEEP_LAST > 0:
        image_idxs = [i for i in idxs if _is_image_content(wire[i].get("content"))]
        keep_images = set(image_idxs[-WIRE_IMAGE_KEEP_LAST:] if image_idxs else [])
    else:
        keep_images = set()

    shrunk = 0
    images_dropped = 0
    out: list[dict] = []
    for i, m in enumerate(wire):
        if i in keep_set or i not in idxs:
            out.append(m)
            continue
        content = m.get("content")
        if _is_image_content(content):
            if i in keep_images:
                out.append(m)
            else:
                images_dropped += 1
                placeholder = "[earlier image omitted from context -- see transcript on disk]"
                if isinstance(content, list):
                    placeholder_content: object = [{"type": "text", "text": placeholder}]
                else:
                    placeholder_content = placeholder
                out.append({**m, "content": placeholder_content})
            continue
        text = str(content or "")
        if m.get("tool_call_id") in spare_ids:
            if len(text) > MEM_TOOL_RESULT_MAX_CHARS:
                shrunk += 1
                out.append({**m, "content": _wire_truncate(
                    m, text, MEM_TOOL_RESULT_MAX_CHARS,
                    MEM_TOOL_RESULT_HEAD, MEM_TOOL_RESULT_TAIL)})
            else:
                out.append(m)
        elif len(text) > TOOL_RESULT_MAX_CHARS:
            shrunk += 1
            out.append({**m, "content": _wire_truncate(
                m, text, TOOL_RESULT_MAX_CHARS, TOOL_RESULT_HEAD, TOOL_RESULT_TAIL)})
        else:
            out.append(m)

    after = wire_chars(out)
    floor_shrunk = 0

    if after > WIRE_MAX_CHARS_AFTER_DIET:
        _AGGRESSIVE_HEAD = 500
        _AGGRESSIVE_TAIL = 500
        result: list[dict] = []
        for i, m in enumerate(out):
            if i in keep_set or i not in idxs or m.get("tool_call_id") in spare_ids:
                result.append(m)  # keep-whole tail, or a spared mem/read result
                continue
            content = m.get("content")
            if _is_image_content(content):
                result.append(m)  # already handled in first pass
                continue
            text = str(content or "")
            if len(text) > _AGGRESSIVE_HEAD + _AGGRESSIVE_TAIL:
                floor_shrunk += 1
                result.append({**m, "content": _wire_truncate(
                    m, text, _AGGRESSIVE_HEAD + _AGGRESSIVE_TAIL,
                    _AGGRESSIVE_HEAD, _AGGRESSIVE_TAIL)})
            else:
                result.append(m)
        out = result
        after = wire_chars(out)

    stats = {
        "before_chars": before,
        "after_chars": after,
        "saved_chars": before - after,
        "saved_tokens": (before - after) // 4,
        "before_tokens": before // 4,
        "after_tokens": after // 4,
        "shrunk": shrunk,
        "images_dropped": images_dropped,
        "kept_tail": len(keep_set),
        "floor_shrunk": floor_shrunk,
        "mem_results_spared": n_mem_spared,
        "reads_spared": n_read_spared,
    }
    return out, stats


def log_compaction(line: str) -> None:
    """Compaction events share the diet log; both answer 'why is it this big'."""
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {line}\n")
    except Exception:
        pass


def _report(path: Path) -> None:
    """Show what the diet would do to one transcript, without doing it."""
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    _, stats = diet(entries)
    print(f"{path.name}")
    print(f"  entries : {stats['entries']:,} -> {stats['kept']:,} "
          f"({stats['attachments_dropped']:,} stale attachments dropped)")
    print(f"  before  : ~{stats['before']:,} tokens")
    print(f"  after   : ~{stats['after']:,} tokens")
    print(f"  saved   : ~{stats['saved']:,} tokens ({stats['pct']}%)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python context_diet.py <transcript.jsonl> [...]")
        raise SystemExit(2)
    for arg in sys.argv[1:]:
        _report(Path(arg))
