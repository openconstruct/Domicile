"""Measure the non-compactable context base from Claude Agent SDK session logs.

The first assistant turn of a session has no history, so its provider-reported
input total (input + cache_creation + cache_read) is base + that turn's
injections (hints, HUD, user message). Across many sessions the minimum is the
best estimate of the pure base; the high percentiles are the working headroom.

Read-only over ~/.claude/projects/<home dir, slashes as dashes>/*.jsonl. Writes
data/base_measure.json (atomic). Direct-API providers are not covered:
token_usage.csv has no session boundaries, so a first turn can't be found.

    python3 measure_base.py            # measure, print, write json
    python3 measure_base.py --dry      # print only
"""
import glob
import json
import os
import statistics
import sys
import time

SESS_GLOB = os.path.join(os.path.expanduser("~/.claude/projects"),
                         os.path.expanduser("~").replace("/", "-"), "*.jsonl")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "base_measure.json")
# First turns above this are continuations/seeded sessions, not fresh ones.
OUTLIER_CEIL = 80_000
RECENT_N = 60  # newest sessions only; the base drifts as tools/mem1 change


def first_turn(path):
    with open(path, errors="replace") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            m = d.get("message") or {}
            u = m.get("usage")
            if d.get("type") == "assistant" and u:
                return {
                    "ts": d.get("timestamp", ""),
                    "model": m.get("model"),
                    "total": (u.get("input_tokens", 0)
                              + u.get("cache_creation_input_tokens", 0)
                              + u.get("cache_read_input_tokens", 0)),
                    "cache_read": u.get("cache_read_input_tokens", 0),
                }
    return None


def measure():
    files = sorted(glob.glob(SESS_GLOB), key=os.path.getmtime, reverse=True)
    rows, dropped = [], 0
    for f in files:
        if len(rows) >= RECENT_N:
            break
        r = first_turn(f)
        if not r or not r["model"] or r["model"].startswith("<"):
            continue
        if r["total"] > OUTLIER_CEIL:
            dropped += 1
            continue
        rows.append(r)
    if not rows:
        return None
    tot = sorted(r["total"] for r in rows)
    q = lambda p: tot[min(len(tot) - 1, int(p * len(tot)))]
    # Stable cached prefix = harness prompt + tool schemas + MCP instructions.
    prefix = [r["cache_read"] for r in rows if r["cache_read"]]
    return {
        "provider": "anthropic",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sessions": len(rows),
        "outliers_dropped": dropped,
        "base_min": tot[0],
        "base_median": int(statistics.median(tot)),
        "base_p90": q(0.9),
        "base_max": tot[-1],
        "stable_prefix": int(statistics.median(prefix)) if prefix else None,
        "note": "first-turn totals; min ~ pure base, max ~ base plus a "
                "heavy first-turn injection. mem1 loaded size varies by "
                "role; add (mem1_cap - mem1_loaded) for worst case.",
    }


def main():
    res = measure()
    if not res:
        print("no sessions found")
        return 1
    print(json.dumps(res, indent=2))
    if "--dry" in sys.argv:
        return 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=2)
    os.replace(tmp, OUT)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
