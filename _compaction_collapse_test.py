#!/usr/bin/env python3
"""Focused test for the post-compaction head collapse (found 2026-08-28:
chat["messages"] was only trimmed on the OpenRouter path, so the Anthropic
path re-summarised and re-logged the whole conversation from its origin on
every compaction).

Covers context_diet.collapse_after_compaction, the _render synthetic label,
append_daily_log's synthetic skip, and a multi-cycle integration run proving
messages + seed stay bounded and the daily log never re-logs a message.

No network, no qdrant, no API. Run: python3 _compaction_collapse_test.py
"""
import asyncio
import re
import tempfile
from pathlib import Path

import context_diet as cd

# These fixtures are ~113 tokens per message, sized for a small tail. Since the
# floor counts compacted text only (30k default -> 26k tail), pin a small tail
# and ceiling here so the history still has a real head to summarize.
cd.compact_tail_tokens = lambda: 5_000
cd.COMPACT_CEILING_TOKENS = 10_000


def mk(role, i):
    # ~320 chars each so a 120-message history has a real head above the
    # 5k-token verbatim tail budget.
    return {"role": role, "text": f"MSG{i:04d} " + f"{role} line {i}. " * 18,
            "ts": f"2026-08-28T10:{i % 60:02d}:00"}


def history(n, start=0):
    return [mk("user" if k % 2 == 0 else "assistant", k)
            for k in range(start, start + n)]


def msg_ids(text):
    return set(re.findall(r"MSG\d{4}", text))


# --- 1. collapse_after_compaction: shape ------------------------------------
msgs = history(50)
out = cd.collapse_after_compaction(msgs, "HANDOVER NOTE ONE", tail_keep=10)
assert len(out) == 11, len(out)
assert out[0]["synthetic"] is True
assert out[0]["role"] == "assistant"
assert out[0]["text"] == "HANDOVER NOTE ONE"
assert out[1:] == msgs[-10:], "tail must be preserved verbatim"
assert "ts" in out[0]
print("1. collapse shape OK")

# --- 2. collapse_after_compaction: no-op guards ---------------------------
assert cd.collapse_after_compaction(msgs, "", 10) == msgs          # no summary
assert cd.collapse_after_compaction(msgs, "x", 0) == msgs          # bad keep
assert cd.collapse_after_compaction(msgs, "x", 999) == msgs        # keep >= len
# returns a copy, not the same list object, when it no-ops
assert cd.collapse_after_compaction(msgs, "", 10) is not msgs
print("2. collapse no-op guards OK")

# --- 3. collapse does not accumulate synthetics -------------------------
once = cd.collapse_after_compaction(history(50), "NOTE A", 8)
grown = once + history(20, start=50)
twice = cd.collapse_after_compaction(grown, "NOTE B", 8)
assert sum(1 for m in twice if m.get("synthetic")) == 1, \
    "old synthetic must fall into the head and be replaced, not pile up"
assert twice[0]["text"] == "NOTE B"
print("3. no synthetic accumulation OK")

# --- 4. _render labels the synthetic entry ------------------------------
r = cd._render([{"role": "assistant", "text": "PRIOR", "synthetic": True},
                {"role": "user", "text": "hello"}])
assert "EARLIER HANDOVER NOTE" in r
assert "PRIOR" in r
assert "Assistant: PRIOR" not in r, "synthetic must not render as a normal turn"
import config_loader as _CL
assert f"{_CL.load_identity()['operator_name']}: hello" in r     # whoever this install's operator is
print("4. _render synthetic label OK")

# --- 5. append_daily_log skips synthetic --------------------------------
with tempfile.TemporaryDirectory() as td:
    cd.DAILY_LOG_DIR = Path(td)
    head = [{"role": "assistant", "text": "DIGEST TEXT", "synthetic": True,
             "ts": "2026-08-28T10:00:00"},
            {"role": "user", "text": "MSG9001 real question",
             "ts": "2026-08-28T10:01:00"},
            {"role": "assistant", "text": "MSG9002 real answer",
             "ts": "2026-08-28T10:02:00"}]
    cd.append_daily_log("chatX", "titleX", head)
    logs = list(Path(td).glob("*.log"))
    assert len(logs) == 1, logs
    body = logs[0].read_text()
    assert "DIGEST TEXT" not in body, "synthetic digest must not be logged"
    assert "MSG9001" in body and "MSG9002" in body
print("5. append_daily_log skips synthetic OK")


# --- 6. integration: many compaction cycles ----------------------------
async def integration():
    seen_by_summariser = []      # ids the fake summariser was shown, per call
    prompts = []

    async def fake_summarise(prompt, max_tokens):
        prompts.append(prompt)
        ids = msg_ids(prompt)
        seen_by_summariser.append(ids)
        # A fixed-size note (like a real bounded handover) that still names
        # a couple of ids so we can watch the chain carry forward.
        sample = " ".join(sorted(ids)[:3])
        return (f"HANDOVER after cycle. Covered {len(ids)} messages "
                f"incl {sample}. " + "filler. " * 40)

    with tempfile.TemporaryDirectory() as td:
        cd.DAILY_LOG_DIR = Path(td)

        messages = history(120)
        next_id = 120
        seed_tokens_seen = []
        max_len = 0

        for cycle in range(6):
            head, _tail = cd.split_for_compaction(messages)
            cd.append_daily_log("c1", "Main", head)
            seed, stats = await cd.build_seed(messages, fake_summarise, 5000)
            seed_tokens_seen.append(cd.text_tokens(seed))
            messages = cd.collapse_after_compaction(
                messages, stats["summary"], stats["kept"])
            max_len = max(max_len, len(messages))
            assert messages[0].get("synthetic") is True
            assert sum(1 for m in messages if m.get("synthetic")) == 1
            # simulate ~15 new turns before the next compaction
            messages += history(15, start=next_id)
            next_id += 15

        # a) messages never grows without bound
        assert max_len < 90, f"messages grew unbounded: {max_len}"

        # b) seed size stays bounded across cycles (no re-summarising the
        #    whole history each time)
        assert max(seed_tokens_seen) < 12_000, seed_tokens_seen
        spread = max(seed_tokens_seen) - min(seed_tokens_seen)
        assert spread < 4_000, f"seed size drifting: {seed_tokens_seen}"

        # c) from cycle 2 on the summariser is handed the prior note
        for p in prompts[1:]:
            assert "EARLIER HANDOVER NOTE" in p

        # d) the daily log logged every real message exactly once -- no
        #    re-logging of already-compacted history
        body = "\n".join(p.read_text() for p in Path(td).glob("*.log"))
        all_ids = sorted(msg_ids(body))
        # every raw message that was ever compacted away should appear
        assert "MSG0000" in all_ids and "MSG0119" in all_ids
        for mid in all_ids:
            count = body.count(mid)
            assert count == 1, f"{mid} logged {count} times (should be 1)"
        print(f"   cycles=6  max messages len={max_len}  "
              f"seed tokens={seed_tokens_seen}  logged ids={len(all_ids)}")

asyncio.run(integration())
print("6. integration: bounded + no re-logging OK")


# --- 7. maybe_compact wires messages + wire on BOTH providers ----------
def maybe_compact_case(provider_obj):
    """Run the real cchat.maybe_compact against a fake chat and return it."""
    import cchat

    async def fake_summarise(prompt, max_tokens):
        return "HANDOVER: " + "note. " * 30

    with tempfile.TemporaryDirectory() as td:
        cd.DAILY_LOG_DIR = Path(td)
        chat_path = Path(td) / "c.json"
        msgs = history(120)
        chat = {
            "id": "cInt", "title": "Main", "messages": msgs,
            # cap-safety mirror, as append_message would have built it on the
            # Anthropic path
            "wire": [{"role": m["role"], "content": m["text"]} for m in msgs],
            "context_tokens": 120_000, "session_id": "old-sess",
            "_path": chat_path, "_mem1": {"tokens": 5000},
        }
        orig_prov = cchat.PROVIDER
        orig_sum = cchat.summarize_for_compaction
        cchat.PROVIDER = provider_obj
        cchat.summarize_for_compaction = fake_summarise
        try:
            asyncio.run(cchat.maybe_compact(chat, screen=None))
        finally:
            cchat.PROVIDER = orig_prov
            cchat.summarize_for_compaction = orig_sum
        return chat


# Anthropic path (PROVIDER is None): messages collapsed, wire re-mirrored
# from the collapsed messages -> bounded, and a later flip to OpenRouter
# still has the handover note + tail.
chat_a = maybe_compact_case(None)
assert chat_a["pending_seed"], "seed must be staged"
assert chat_a["session_id"] is None
assert chat_a["messages"][0].get("synthetic") is True
assert len(chat_a["messages"]) < 90, len(chat_a["messages"])
assert 0 < len(chat_a["wire"]) < 90, len(chat_a["wire"])
assert chat_a["wire"][0]["content"].startswith("HANDOVER:"), \
    "wire must carry the handover note first so an OR flip loses nothing"
assert len(chat_a["wire"]) == len(chat_a["messages"]), \
    "wire mirrors the collapsed messages on the Anthropic path"
print("7a. maybe_compact Anthropic: messages + wire both bounded OK")

# OpenRouter path (PROVIDER set): messages collapsed the same way, wire
# emptied (OR bakes the seed into the next turn's prompt prefix itself).
chat_o = maybe_compact_case(object())
assert chat_o["messages"][0].get("synthetic") is True
assert len(chat_o["messages"]) < 90
assert chat_o["wire"] == [], chat_o["wire"]
print("7b. maybe_compact OpenRouter: messages bounded, wire emptied OK")

print("\nALL OK")
