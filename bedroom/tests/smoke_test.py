"""Smoke test: oracle student (reads expected from scenario file directly) must score 1.0;
a lazy student that always Reads a guessed path must score < 1.0; free text scores 0."""
import sys, json
sys.path.insert(0, __file__.rsplit("/tests/",1)[0])
from bedroom.core import load_dir, run_eval
from pathlib import Path

base = __file__.rsplit("/tests/",1)[0]
scenarios = load_dir(base + "/scenarios/fs")
by_id = {s["id"]: s for s in scenarios}

def oracle(obs):
    s = by_id[obs["scenario_id"]]
    args = {}
    for k, rule in s["expected"].get("args_match", {}).items():
        if "equals" in rule: args[k] = rule["equals"]
        elif "contains" in rule: args[k] = " ".join(rule["contains"])
        elif "one_of" in rule: args[k] = rule["one_of"][0]
    return {"tool": s["expected"]["tool"], "args": args}

def lazy(obs):
    return {"tool": "Read", "args": {"path": "/notes/todo.txt"}}

def chatty(obs):
    return {"text": "Sure! The answer is probably in a file somewhere."}

if __name__ == "__main__":
    o = run_eval(scenarios, oracle, "/tmp/bedroom-traces.jsonl")
    l = run_eval(scenarios, lazy)
    c = run_eval(scenarios, chatty)
    print(f"oracle overall={o['overall']} n={o['n']} by_family={o['by_family']}")
    print(f"lazy   overall={l['overall']}")
    print(f"chatty overall={c['overall']}")
    assert o["overall"] == 1.0, [r for r in o["results"] if r["reward"] == 0]
    assert l["overall"] < 1.0 and c["overall"] == 0.0
    print("SMOKE OK")
