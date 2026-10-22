"""Smoke for the GRPO regimen: pure-math checks always; the full loop
(tiny model, 1 round, 2 scenarios, G=2) when torch is available.

Run: python3 tests/smoke_test_grpo.py [--tiny]
Bedroom rule: the training step must run end to end and move a
checkpoint to disk before this regimen is called built.
"""
import json, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bedroom.train.grpo_loop import (group_advantages, parse_action)  # noqa: E402

fails = []
def check(name, cond):  # noqa: E306
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

# --- pure math / parsing -------------------------------------------------
advs = group_advantages([1, 0, 1, 0])
check("advantages split sign", advs[0] > 0 and advs[1] < 0)
check("advantages zero-sum", abs(sum(advs)) < 1e-9)
check("dead group is all zero", group_advantages([1, 1, 1]) == [0.0, 0.0, 0.0])
check("parse tool json", parse_action('{"tool": "Read", "args": {"path": "/x"}}')
      == {"tool": "Read", "args": {"path": "/x"}})
check("parse openai-ish", parse_action('{"name": "Read", "arguments": "{\\"path\\": \\"/x\\"}"}')
      == {"tool": "Read", "args": {"path": "/x"}})
check("parse LFM pythonic", parse_action("hmm</think><|tool_call_start|>[Read(path='/x')]<|tool_call_end|>")
      == {"tool": "Read", "args": {"path": "/x"}})
check("parse LFM list takes first", parse_action("<|tool_call_start|>[Glob(pattern='*a*'), Glob(pattern='*b*')]<|tool_call_end|>")
      == {"tool": "Glob", "args": {"pattern": "*a*"}})
check("parse LFM argv list", parse_action("<|tool_call_start|>[Bash(argv=['ls', '-l', '/d'])]<|tool_call_end|>")
      == {"tool": "Bash", "args": {"argv": ["ls", "-l", "/d"]}})
check("parse Qwen xml", parse_action("<tool_call>\n<function=Bash>\n<parameter=argv>\n[\"ls\", \"/d\"]\n</parameter>\n</function>\n</tool_call>")
      == {"tool": "Bash", "args": {"argv": ["ls", "/d"]}})
check("think block stripped from answer", parse_action("<think>x</think>Ana\nMarta") == {"text": "Ana\nMarta"})
check("malformed native call is text, not crash", "tool" not in parse_action("<|tool_call_start|>[Read(path=)]<|tool_call_end|>"))
def _step(sched, rnd):  # mirrors run_campaign's lookup
    ks = [k for k in sched if k <= rnd]
    return sched[max(ks)] if ks else 3
check("difficulty step schedule", [_step({0: 1, 10: 2, 20: 3}, r) for r in (0, 5, 10, 19, 20, 30)] == [1, 1, 2, 2, 3, 3])
check("parse text fallback", parse_action("just an answer") == {"text": "just an answer"})

if "--tiny" in sys.argv:
    from bedroom.multi import load_multi_dir
    from bedroom.train.grpo_loop import load_policy, make_optimizer, run_campaign
    from students.lfm_student_full import TOOLS
    spec = [t["function"] for t in TOOLS]
    tools_spec = [{"type": "function", "function": f} for f in spec]
    policy, ref = load_policy("HuggingFaceTB/SmolLM2-135M-Instruct",
                              TOOLS, device="cpu")
    scenarios = load_multi_dir(Path(__file__).resolve().parent.parent / "bench" / "dev")[:4]
    opt = make_optimizer(policy.model, lr=1e-5)
    with tempfile.TemporaryDirectory(prefix="grpo-smoke-") as td:
        stats = run_campaign(policy, ref, opt, scenarios, td, rounds=1,
                             scenarios_per_round=2, group_size=2,
                             temperature=0.8, log=lambda *a: None)
        ckpt = Path(td) / "round-000"
        check("campaign produced stats", len(stats) == 1 and "mean_reward" in stats[0])
        check("loss finite", stats[0]["loss"] == stats[0]["loss"] and abs(stats[0]["loss"]) < 1e6)
        check("checkpoint saved", (ckpt / "config.json").exists())
    print(json.dumps(stats, indent=1))

print("\nSMOKE", "FAILED" if fails else "OK", fails or "")
sys.exit(1 if fails else 0)
