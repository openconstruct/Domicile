"""Multi-round smoke: oracle following the full chain per scenario must
score 1.0; a lazy always-search student and a one-shot student that quits
after its first tool call must score ~0 -- the chains are the point."""
import sys
sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from bedroom.multi import load_multi_dir, run_eval_multi

base = __file__.rsplit("/tests/", 1)[0]
scenarios = load_multi_dir(base + "/scenarios/multi")

MS = "mcp__memory__memory_search"
M2S = "mcp__mem2__mem2_search"
MADD = "mcp__memory__memory_add"
MDEL = "mcp__memory__memory_delete"
M2C = "mcp__mem2__mem2_get_chunk"

ORACLE = {
    "multi-route-0001": [
        {"tool": MS, "args": {"query": "operator home address"}},
        {"tool": M2S, "args": {"query": "moved new address"}},
        {"tool": MDEL, "args": {"memory_id": "m-001"}},
        {"tool": MADD, "args": {"text": "the operator lives at 88 Harbor Lane"}},
        {"text": "The current address is 88 Harbor Lane."},
    ],
    "multi-route-0002": [
        {"tool": MS, "args": {"query": "deploy day"}},
        {"tool": M2S, "args": {"query": "deploy Fridays"}},
        {"tool": MADD, "args": {"text": "the operator deploys on Fridays"}},
        {"text": "The operator deploys on Friday."},
    ],
    "multi-route-0003": [
        {"tool": MS, "args": {"query": "morning drink"}},
        {"tool": M2S, "args": {"query": "morning tea coffee settled"}},
        {"tool": MDEL, "args": {"memory_id": "m-002"}},
        {"text": "The operator drinks tea in the morning."},
    ],
    "multi-route-0004": [
        {"tool": M2S, "args": {"query": "spare workshop key"}},
        {"tool": M2C, "args": {"source_file": "2026-07-04-chat.jsonl", "chunk_index": 1}},
        {"text": "The spare key is under the blue planter by the side gate."},
    ],
    "multi-fs-0001": [
        {"tool": "Read", "args": {"path": "/out/log.txt"}},
        {"tool": "Write", "args": {"path": "/out/log.txt", "content": "day one\nday two\n"}},
        {"text": "Appended."},
    ],
    "multi-fs-0002": [
        {"tool": "Grep", "args": {"pattern": "debug", "path": "/cfg"}},
        {"tool": "Read", "args": {"path": "/cfg/app.conf"}},
        {"tool": "Edit", "args": {"path": "/cfg/app.conf", "old": "debug = true", "new": "debug = false"}},
        {"text": "Debug is off."},
    ],
    "multi-mem1-0001": [
        {"tool": "Read", "args": {"path": "/mem1/practices.md"}},
        {"tool": "Edit", "args": {"path": "/mem1/practices.md",
                                  "old": "Practices.\n\n- When you are about to disagree with something the operator has said, it is very important that you Verify before contradicting, every single time, without exception.\n- Before you change any file on the system, always remember to Back up before editing, because edits cannot be undone reliably.\n- Also, in general, be helpful and friendly in tone whenever you can.\n",
                                  "new": "Practices.\n- Verify before contradicting.\n- Back up before editing.\n"}},
        {"text": "Consolidated under budget."},
    ],
    "multi-cross-0001": [
        {"tool": MS, "args": {"query": "project status file"}},
        {"tool": "Read", "args": {"path": "/proj/status.txt"}},
        {"text": "The release codename is cobalt finch."},
    ],
}

def oracle(obs):
    return ORACLE[obs["scenario_id"]][obs["turn"]]

def lazy(obs):
    return {"tool": MS, "args": {"query": "stuff"}}

def oneshot(obs):
    if obs["turn"] == 0:
        return ORACLE[obs["scenario_id"]][0]
    return {"text": "Done, probably."}

if __name__ == "__main__":
    o = run_eval_multi(scenarios, oracle, "/tmp/bedroom-multi-traces.jsonl")
    l = run_eval_multi(scenarios, lazy)
    s = run_eval_multi(scenarios, oneshot)
    print(f"oracle  overall={o['overall']} n={o['n']} by_family={o['by_family']}")
    print(f"lazy    overall={l['overall']}")
    print(f"oneshot overall={s['overall']}")
    assert o["overall"] == 1.0, [r for r in o["results"] if r["reward"] == 0]
    assert l["overall"] == 0.0 and s["overall"] == 0.0
    print("MULTI SMOKE OK")
