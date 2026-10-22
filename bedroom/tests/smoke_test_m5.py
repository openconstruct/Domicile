"""M5 smoke: oracle walking the graded Bash road per scenario must score
8/8; a lazy student that answers immediately, a one-shot student that
fires a single plausible call (wc for grep -c, cp for mv, > for >>), and
a sloppy student that works at full length on the bait (the draft memo,
uniq without sort, python3 instead of sed, describing instead of doing)
must all score 0 -- the count, the target, the order and the end-state
are each graded exactly.

Also unit-checks the M5 runner polish: mail_sent_count grades
exactly-once sending (M4's gmail-0001 double-send now fails), and
run_curriculum reports per-family and per-difficulty accuracy with
sampling weights applied.
"""
import sys
sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from bedroom.multi import load_multi_dir, run_curriculum, run_episode_multi, run_eval_multi

base = __file__.rsplit("/tests/", 1)[0]
scenarios = load_multi_dir(base + "/scenarios/bash")

B = "Bash"

ORACLE = {
    "bash-0001": [
        {"tool": B, "args": {"argv": ["grep", "-c", "ERROR", "/logs/app.log"]}},
        {"text": "5"},
    ],
    "bash-0002": [
        {"tool": B, "args": {"argv": ["mv", "/docs/memo-2025.txt", "/docs/memo-2026.txt"]}},
        {"text": "Renamed to memo-2026.txt."},
    ],
    "bash-0003": [
        {"tool": B, "args": {"argv": ["sed", "-i", "s/PENDING/APPROVED/g", "/notes/release.txt"]}},
        {"text": "All three now read APPROVED."},
    ],
    "bash-0004": [
        {"tool": B, "args": {"argv": ["find", "/data", "-name", "*.log"]}},
        {"text": "/data/exports/audit.log\n/data/old/access.log\n/data/server.log"},
    ],
    "bash-0005": [
        {"tool": B, "args": {"argv": ["sort", "-u", "/data/crew.txt"]}},
        {"text": "Ana\nJonas\nMarta\nPriya\nSilvie"},
    ],
    "bash-0006": [
        {"tool": B, "args": {"argv": ["sed", "-i", "$a22:00 shift closed", "/logs/shift.txt"]}},
        {"text": "Appended the closing line."},
    ],
    "bash-0007": [
        {"tool": B, "args": {"argv": ["mkdir", "-p", "/archive/2025"]}},
        {"tool": B, "args": {"argv": ["mv", "/inbox/photo-notes.txt", "/archive/2025/photo-notes.txt"]}},
        {"text": "Moved into /archive/2025."},
    ],
    "bash-0008": [
        {"tool": B, "args": {"argv": ["sed", "-i", "s/teal/turquoise/g", "/notes/colors.txt"]}},
        {"text": "Replaced teal with turquoise throughout."},
    ],
}

ONESHOT = {
    "bash-0001": {"tool": B, "args": {"argv": ["wc", "-l", "/logs/app.log"]}},
    "bash-0002": {"tool": B, "args": {"argv": ["cp", "/docs/memo-2025.txt", "/docs/memo-2026.txt"]}},
    "bash-0003": {"tool": "Edit", "args": {"path": "/notes/release.txt",
                                          "old": "PENDING", "new": "APPROVED"}},
    "bash-0004": {"tool": B, "args": {"argv": ["find", "/data", "-name", "*"]}},
    "bash-0005": {"tool": B, "args": {"argv": ["cat", "/data/crew.txt"]}},
    "bash-0006": {"tool": "Write", "args": {"path": "/logs/shift.txt",
                                           "content": "22:00 shift closed\n"}},
    "bash-0007": {"tool": B, "args": {"argv": ["mv", "/inbox/photo-notes.txt",
                                              "/archive/2025/photo-notes.txt"]}},
    "bash-0008": {"tool": B, "args": {"argv": ["awk", "{gsub(/teal/, \"turquoise\")} 1",
                                              "/notes/colors.txt"]}},
}

SLOPPY = {
    "bash-0001": [  # counts by eye and folds the lowercase lines in
        {"tool": B, "args": {"argv": ["cat", "/logs/app.log"]}},
        {"text": "7"},
    ],
    "bash-0002": [  # moves the draft sibling instead
        {"tool": B, "args": {"argv": ["mv", "/docs/memo-2025-draft.txt", "/docs/memo-2026.txt"]}},
        {"text": "Renamed the memo."},
    ],
    "bash-0003": [  # overwrite that drops the changelog line
        {"tool": "Write", "args": {"path": "/notes/release.txt",
                                   "content": "Fernway 2.4 release checklist\n- build: APPROVED\n"
                                              "- smoke tests: APPROVED\n- announcement: APPROVED\n"
                                              "Ship only when nothing is pending.\n"}},
        {"text": "Updated every status to APPROVED."},
    ],
    "bash-0004": [  # lists only the top level
        {"tool": B, "args": {"argv": ["ls", "/data"]}},
        {"text": "/data/server.log"},
    ],
    "bash-0005": [  # uniq without sort: non-adjacent duplicates survive
        {"tool": B, "args": {"argv": ["uniq", "/data/crew.txt"]}},
        {"text": "Marta\nJonas\nAna\nMarta\nPriya\nJonas\nAna\nSilvie"},
    ],
    "bash-0006": [  # shell habit: ">>" is a literal arg with no shell, so nothing is written
        {"tool": B, "args": {"argv": ["echo", "22:00 shift closed", ">>", "/logs/shift.txt"]}},
        {"text": "Added the closing line."},
    ],
    "bash-0007": [  # hits the missing-dir error, then copies instead of moving
        {"tool": B, "args": {"argv": ["mv", "/inbox/photo-notes.txt", "/archive/2025/photo-notes.txt"]}},
        {"tool": B, "args": {"argv": ["mkdir", "-p", "/archive/2025"]}},
        {"tool": B, "args": {"argv": ["cp", "/inbox/photo-notes.txt", "/archive/2025/photo-notes.txt"]}},
        {"text": "Archived the notes."},
    ],
    "bash-0008": [  # reaches for python3, gets refused, describes instead of doing
        {"tool": B, "args": {"argv": ["python3", "-c",
                                      "import pathlib; p = pathlib.Path('/notes/colors.txt'); "
                                      "p.write_text(p.read_text().replace('teal', 'turquoise'))"]}},
        {"text": "I replaced every teal with turquoise in colors.txt."},
    ],
}

def oracle(obs):
    return ORACLE[obs["scenario_id"]][obs["turn"]]

def lazy(obs):
    return {"text": "I do not know."}

def oneshot(obs):
    if obs["turn"] == 0:
        return ONESHOT[obs["scenario_id"]]
    return {"text": "Done, probably."}

def sloppy(obs):
    steps = SLOPPY[obs["scenario_id"]]
    return steps[min(obs["turn"], len(steps) - 1)]


def check_exactly_once():
    scenario = {
        "id": "unit-exactly-once", "family": "unit", "difficulty": 1, "max_turns": 4,
        "situation": "Reply to the operator once.",
        "setup": {"social": {"inbox": [
            {"id": "m1", "from": "op@example.com", "subject": "Window?", "body": "When?"}]}},
        "expected": {"required_sequence": ["gmail_send"],
                     "postconditions": [{"type": "mail_sent_count", "equals": 1,
                                         "to_contains": "op@example.com",
                                         "body_contains": "23:00"}]},
    }
    send = {"tool": "gmail_send", "args": {"to": "op@example.com", "subject": "Re: Window?",
                                           "body": "The window is 23:00 UTC."}}
    once = lambda obs: [send, {"text": "Sent."}][obs["turn"]]
    twice = lambda obs: [send, send, {"text": "Sent."}][obs["turn"]]
    assert run_episode_multi(scenario, once)["reward"] == 1
    assert run_episode_multi(scenario, twice)["reward"] == 0
    print("exactly-once postcondition OK")


def check_curriculum():
    rep = run_curriculum(scenarios, oracle, weights={"bash": 1})
    assert rep["overall"] == 1.0 and rep["n"] == 8
    assert rep["by_family"] == {"bash": 1.0}
    assert set(rep["by_difficulty"]) == {1, 2, 3}
    heavy = run_curriculum(scenarios[:1], oracle, weights={"bash": 3})
    assert heavy["n"] == 3 and heavy["overall"] == 1.0
    print(f"curriculum runner OK: by_difficulty={rep['by_difficulty']}")


if __name__ == "__main__":
    o = run_eval_multi(scenarios, oracle, "/tmp/bedroom-m5-traces.jsonl")
    l = run_eval_multi(scenarios, lazy)
    os_ = run_eval_multi(scenarios, oneshot)
    sl = run_eval_multi(scenarios, sloppy)
    print(f"oracle  overall={o['overall']} n={o['n']} by_family={o['by_family']}")
    print(f"lazy    overall={l['overall']}")
    print(f"oneshot overall={os_['overall']}")
    print(f"sloppy  overall={sl['overall']}")
    assert o["overall"] == 1.0, [r for r in o["results"] if r["reward"] == 0]
    assert l["overall"] == 0.0 and os_["overall"] == 0.0 and sl["overall"] == 0.0
    check_exactly_once()
    check_curriculum()
    print("M5 SMOKE OK")
