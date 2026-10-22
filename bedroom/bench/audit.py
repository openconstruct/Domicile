"""Bedroom bench audit: prove the generated pools are sound and disjoint.

Checks, per the bench design (bedroom-bench-regimen-design-oct21.md):

1. Oracle soundness -- every scenario's generated oracle scores 1.
2. Degenerate rejection -- a text-only student scores 0 on every
   scenario (no scenario is passable by narrating).
3. Leak check -- no graded answer string appears verbatim in the
   scenario's own situation text (the answer must be earned through
   the tools, not read off the prompt). File-content postconditions
   are exempt where the situation itself dictates the content
   (write/append archetypes), flagged by archetype below.
4. Cross-pool duplication -- no two scenarios in different pools
   share an identical (situation, setup) pair; seed spaces and
   vocabularies are disjoint, so any hit is a generator bug.

Run: python3 -m bench.audit [--root bench] -- writes audit-report.json
beside the pools and exits nonzero on any failure.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bedroom.multi import load_multi_dir, run_episode_multi  # noqa: E402

DICTATED = {"fs-write-note", "bash-append", "bash-append-two", "discord-post"}


def _answers(scenario):
    exp = scenario["expected"]
    out = list(exp.get("final_answer_contains", []))
    out += [str(x) for x in exp.get("final_answer_exact_lines", [])]
    for pc in exp.get("postconditions", []):
        if pc["type"] in ("file_equals",) and scenario["archetype"] not in DICTATED:
            out.append(pc["text"].strip())
        if pc["type"] == "store_contains" and scenario["archetype"] != "mem-add":
            out.append(pc["text"])
        if pc["type"] == "file_contains":
            out.append(pc["text"])
    return [a for a in out if a and len(a) >= 3]


def audit_pool(root, pool, oracles):
    scenarios = load_multi_dir(root / pool)
    failures = []
    lazy = lambda obs: {"text": "I do not know."}  # noqa: E731
    for sc in scenarios:
        steps = oracles[sc["id"]]
        oracle = lambda obs, steps=steps: steps[obs["turn"]]
        r = run_episode_multi(sc, oracle)
        if r["reward"] != 1:
            failures.append(f"{sc['id']}: oracle scored 0 ({r['reason']})")
        r = run_episode_multi(sc, lazy)
        if r["reward"] != 0:
            failures.append(f"{sc['id']}: lazy student scored 1")
        for ans in _answers(sc):
            if ans in sc["situation"]:
                failures.append(f"{sc['id']}: answer leaks into situation: {ans[:40]!r}")
    return scenarios, failures


def main():
    root = Path(__file__).resolve().parent
    oracles = json.loads((root / "oracles.json").read_text())
    report, seen = {"pools": {}}, {}
    bad = 0
    for pool in ("train", "dev", "eval"):
        scenarios, failures = audit_pool(root, pool, oracles[pool])
        for sc in scenarios:
            key = (sc["situation"], json.dumps(sc["setup"], sort_keys=True))
            if key in seen and seen[key] != pool:
                failures.append(f"{sc['id']}: duplicates a scenario in {seen[key]}")
            seen[key] = pool
        report["pools"][pool] = {"n": len(scenarios), "failures": failures}
        bad += len(failures)
        print(f"{pool}: n={len(scenarios)} failures={len(failures)}")
        for f in failures[:10]:
            print(f"  FAIL {f}")
    (root / "audit-report.json").write_text(json.dumps(report, indent=1))
    print("AUDIT", "OK" if bad == 0 else f"FAILED ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
