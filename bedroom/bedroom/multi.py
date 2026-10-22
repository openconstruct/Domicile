"""Bedroom multi-round runner: episodes where the reward lives in the chain.

Extends the M1/M2 one-shot loop: a scenario declares max_turns, a
required_sequence of tools (subsequence, in order), and end-state
postconditions. Reward stays strict binary -- 1 only when the sequence
was followed AND every postcondition holds AND any declared final-answer
check passes. Tool results are fed back as observations between turns.

Design notes (from the Oct 19 design conversation):
- Escalation is paid by construction, not by partial credit: scenarios
  seed mem0 with a plausible-but-stale fact whose correction exists only
  in mem2, so the only road to 1 runs mem0 -> mem2 -> act.
- Routing is logged in the trace (mem0_before_mem2, turns used) as a
  curriculum metric even though it does not change the binary reward.
- A student ends an episode early by returning {"text": ...} instead of
  a tool call; that text is the final answer when one is declared.
"""
import hashlib, json
from pathlib import Path

from .bash import BASH_TOOLS, run_bash_tool
from .core import Sandbox, run_fs_tool
from .render import render_fs_call
from .memory import FactStore, TranscriptStore, MEMORY_TOOLS, approx_tokens, run_memory_tool
from .social import SocialState, SOCIAL_TOOLS, run_social_tool
from .web import Dollhouse, WEB_TOOLS, run_web_tool


def _dispatch(store, tstore, sb, dh, social, tool, args):
    if tool in BASH_TOOLS:
        return render_fs_call(sb, tool, args)
    if tool in WEB_TOOLS:
        return run_web_tool(dh, tool, args)
    if tool in SOCIAL_TOOLS:
        return run_social_tool(social, tool, args)
    if tool in MEMORY_TOOLS or tool.startswith("mcp__"):
        return run_memory_tool(store, tstore, sb, tool, args)
    return render_fs_call(sb, tool, args)


def _mail_match(msg, pc):
    if "to_contains" in pc and pc["to_contains"] not in msg.get("to", ""):
        return False
    if "subject_contains" in pc and pc["subject_contains"] not in msg.get("subject", ""):
        return False
    if "body_contains" in pc and pc["body_contains"] not in msg.get("body", ""):
        return False
    return True


def _check_postconditions(store, sb, scenario, social=None):
    for pc in scenario["expected"].get("postconditions", []):
        kind = pc["type"]
        if kind == "mail_sent":
            if not any(_mail_match(m, pc) for m in social.sent):
                return False
        elif kind == "mail_not_sent":
            if any(_mail_match(m, pc) for m in social.sent):
                return False
        elif kind == "draft_exists":
            if not any(_mail_match(d, pc) for d in social.drafts):
                return False
        elif kind == "mail_sent_count":
            n = sum(1 for m in social.sent if _mail_match(m, pc))
            if n != pc["equals"]:
                return False
        elif kind == "calendar_event":
            if not any(pc["title_contains"] in e.get("title", "")
                       and e.get("start") == pc["start"] for e in social.calendar):
                return False
        elif kind == "calendar_no_event":
            if any(pc["title_contains"] in e.get("title", "")
                   and ("start" not in pc or e.get("start") == pc["start"])
                   for e in social.calendar):
                return False
        elif kind == "blog_posted":
            if not any(pc["body_contains"] in p.get("body", "") for p in social.blog):
                return False
        elif kind == "discord_posted":
            msgs = social.discord.get(pc["channel"].lstrip("#"), [])
            if not any(pc["text_contains"] in m.get("text", "") for m in msgs):
                return False
        elif kind == "discord_not_posted":
            msgs = social.discord.get(pc["channel"].lstrip("#"), [])
            if any(pc["text_contains"] in m.get("text", "") for m in msgs):
                return False
        elif kind == "store_contains":
            if not any(pc["text"] in f["text"] for f in store.facts):
                return False
        elif kind == "store_not_contains":
            if any(pc["text"] in f["text"] for f in store.facts):
                return False
        elif kind == "store_count":
            if len(store.facts) != pc["equals"]:
                return False
        elif kind == "file_contains":
            p = sb.resolve(pc["path"])
            if not p.exists() or pc["text"] not in p.read_text():
                return False
        elif kind == "file_not_contains":
            p = sb.resolve(pc["path"])
            if p.exists() and pc["text"] in p.read_text():
                return False
        elif kind == "file_max_tokens":
            p = sb.resolve(pc["path"])
            if not p.exists() or approx_tokens(p.read_text()) > pc["max"]:
                return False
        elif kind == "file_exists":
            if not sb.resolve(pc["path"]).exists():
                return False
        elif kind == "file_not_exists":
            if sb.resolve(pc["path"]).exists():
                return False
        elif kind == "file_equals":
            p = sb.resolve(pc["path"])
            if not p.exists() or p.read_text() != pc["text"]:
                return False
        else:
            raise ValueError(f"unknown postcondition {kind}")
    return True


def _sequence_ok(required, executed):
    it = iter(executed)
    return all(any(t == want for t in it) for want in required)


def run_episode_multi(scenario, student_fn, trace_path=None):
    setup = scenario.get("setup", {})
    store = FactStore(setup.get("facts"))
    tstore = TranscriptStore(setup.get("chunks"))
    sb = Sandbox(setup.get("files"))
    dh = Dollhouse(setup.get("web"))
    social = SocialState(setup.get("social"))
    max_turns = int(scenario.get("max_turns", 3))
    exp = scenario["expected"]
    obs = {"situation": scenario["situation"], "scenario_id": scenario["id"],
           "tools": MEMORY_TOOLS + WEB_TOOLS + SOCIAL_TOOLS + BASH_TOOLS + ["Read", "Write", "Edit", "Glob", "Grep"],
           "files_root": "/", "turn": 0, "max_turns": max_turns, "history": []}
    if setup.get("show_listing"):
        obs["memory_listing"] = store.list()
    obs_hash = hashlib.sha256(json.dumps(
        {k: v for k, v in obs.items() if k != "history"},
        sort_keys=True).encode()).hexdigest()[:16]
    executed, final_text, reward, reason = [], None, 0, "out of turns"
    try:
        for turn in range(max_turns):
            obs["turn"] = turn
            obs["history"] = list(obs["history"])
            action = student_fn(obs)
            if not isinstance(action, dict):
                reason = "action not a dict"
                break
            if "tool" not in action:
                final_text = action.get("text", "")
                reason = "ended by student"
                break
            tool = action["tool"]
            if tool in exp.get("forbidden_tools", []):
                reason = f"forbidden tool {tool}"
                executed.append(tool)
                break
            try:
                result = _dispatch(store, tstore, sb, dh, social, tool, action.get("args", {}))
            except Exception as e:
                result = f"ERROR: {type(e).__name__}: {e}"
            executed.append(tool)
            obs["history"].append({"tool": tool, "args": action.get("args", {}),
                                    "result": str(result)[:12000]})
        else:
            reason = "out of turns"
        seq_ok = _sequence_ok(exp.get("required_sequence", []), executed)
        pc_ok = _check_postconditions(store, sb, scenario, social)
        ans_ok = True
        if exp.get("final_answer_contains"):
            ans_ok = final_text is not None and all(
                x in final_text for x in exp["final_answer_contains"])
        if exp.get("final_answer_exact_lines"):
            got = [ln.strip() for ln in (final_text or "").splitlines() if ln.strip()]
            ans_ok = ans_ok and got == [str(x) for x in exp["final_answer_exact_lines"]]
        if seq_ok and pc_ok and ans_ok:
            reward, reason = 1, "ok"
        elif not seq_ok:
            reason = f"sequence incomplete: did {executed}, need {exp.get('required_sequence', [])}"
        elif not pc_ok:
            reason = "postcondition failed"
        elif not ans_ok:
            reason = "final answer does not match declared content"
    finally:
        mem_tools = [t.split("__")[-1] for t in executed]
        routing = None
        if "memory_search" in mem_tools and "mem2_search" in mem_tools:
            routing = mem_tools.index("memory_search") < mem_tools.index("mem2_search")
        trace = {"id": scenario["id"], "obs_hash": obs_hash, "executed": executed,
                 "turns": len(executed), "mem0_before_mem2": routing,
                 "final_text": final_text, "reward": reward, "reason": reason}
        if trace_path:
            with open(trace_path, "a") as f:
                f.write(json.dumps(trace) + "\n")
        sb.cleanup()
    return trace


def load_multi_dir(d):
    out = []
    for p in sorted(Path(d).glob("*.json")):
        s = json.loads(Path(p).read_text())
        assert s["id"] and s["expected"].get("required_sequence"), s.get("id")
        out.append(s)
    return out


def run_eval_multi(scenarios, student_fn, trace_path=None):
    results = [run_episode_multi(s, student_fn, trace_path) for s in scenarios]
    by_family = {}
    for s, r in zip(scenarios, results):
        by_family.setdefault(s["family"], []).append(r["reward"])
    return {"overall": sum(r["reward"] for r in results) / max(1, len(results)),
            "by_family": {k: sum(v) / len(v) for k, v in by_family.items()},
            "n": len(results), "results": results}


def run_curriculum(scenarios, student_fn, weights=None, trace_path=None, workers=1):
    """Curriculum run per spec section 8: families x difficulty ladders.

    weights maps a family name to a repeat count (its sampling weight for
    this run; default 1). Every episode still grades strict binary; the
    report carries per-family and per-difficulty accuracy, the shape the
    bulk bench and the bench/train/bench demonstration protocol consume.
    """
    weights = weights or {}
    plan = []
    for s in scenarios:
        plan.extend([s] * max(1, int(weights.get(s["family"], 1))))
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda s: run_episode_multi(s, student_fn, trace_path), plan))
    else:
        results = [run_episode_multi(s, student_fn, trace_path) for s in plan]
    by_family, by_difficulty = {}, {}
    for s, r in zip(plan, results):
        by_family.setdefault(s["family"], []).append(r["reward"])
        by_difficulty.setdefault(s.get("difficulty", 1), []).append(r["reward"])
    return {"overall": sum(r["reward"] for r in results) / max(1, len(results)),
            "by_family": {k: sum(v) / len(v) for k, v in sorted(by_family.items())},
            "by_difficulty": {k: sum(v) / len(v) for k, v in sorted(by_difficulty.items())},
            "n": len(results), "results": results}
