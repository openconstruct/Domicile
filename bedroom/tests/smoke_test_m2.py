"""M2 smoke test: oracle (reads expected from the scenario) must score 1.0;
a lazy student that always searches memory for 'stuff' and a free-text
student must score ~0."""
import sys
sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from bedroom.memory import load_mem_dir, run_eval_mem

base = __file__.rsplit("/tests/", 1)[0]
scenarios = load_mem_dir(base + "/scenarios/memory")
by_id = {s["id"]: s for s in scenarios}

def oracle(obs):
    s = by_id[obs["scenario_id"]]
    args = {}
    for k, rule in s["expected"].get("args_match", {}).items():
        if "equals" in rule: args[k] = rule["equals"]
        elif "contains" in rule: args[k] = " ".join(rule["contains"])
        elif "one_of" in rule: args[k] = rule["one_of"][0]
    if s["id"] == "mem1-edit-0001":
        args["old"] = "- Answer plainly."
        args["new"] = "- Answer plainly.\n- Verify before contradicting."
    if s["id"] == "mem1-edit-0002":
        args["content"] = "Goals.\nbeta, paper, rest"
    return {"tool": s["expected"]["tool"], "args": args}

def lazy(obs):
    return {"tool": "mcp__memory__memory_search", "args": {"query": "stuff"}}

def chatty(obs):
    return {"text": "I remember everything, trust me."}

if __name__ == "__main__":
    o = run_eval_mem(scenarios, oracle, "/tmp/bedroom-m2-traces.jsonl")
    l = run_eval_mem(scenarios, lazy)
    c = run_eval_mem(scenarios, chatty)
    print(f"oracle overall={o['overall']} n={o['n']} by_family={o['by_family']}")
    print(f"lazy   overall={l['overall']}")
    print(f"chatty overall={c['overall']}")
    assert o["overall"] == 1.0, [r for r in o["results"] if r["reward"] == 0]
    assert l["overall"] < 0.2 and c["overall"] == 0.0
    print("M2 SMOKE OK")
