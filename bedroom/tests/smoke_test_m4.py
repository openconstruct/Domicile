"""M4 smoke: oracle walking the graded social road per scenario must score
8/8; a lazy student that answers immediately, a one-shot student that fires
a single plausible call (send without reading, post to the decoy channel),
and a sloppy student that works at full length on the wrong sender /
channel / slot / surface must all score 0 -- the fact, the channel, the
slot and the draft-vs-send call are each graded in end-state."""
import sys
sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from bedroom.multi import load_multi_dir, run_eval_multi

base = __file__.rsplit("/tests/", 1)[0]
scenarios = load_multi_dir(base + "/scenarios/social")

GS = "GmailSearch"
GR = "GmailRead"
GE = "GmailSend"
GD = "GmailDraft"
CL = "CalendarList"
CC = "CalendarCreate"
BP = "BloggerPost"
DR = "ChatRead"
DP = "ChatSend"
MS = "mcp__memory__memory_search"
OPS = "spaces/AAAAops"

ORACLE = {
    "gmail-0001": [
        {"tool": GS, "args": {"query": "Priya Bramble retention runbook"}},
        {"tool": GR, "args": {"id": "msg-101"}},
        {"tool": "Read", "args": {"path": "/notes/bramble-ops.txt"}},
        {"tool": GE, "args": {"to": "priya@northloop.example",
                             "subject": "Re: Bramble retention for the runbook",
                             "body": "Hi Priya -- the shipped default retention window for Bramble snapshots is 21 days when retention_days is not set."}},
        {"text": "Replied to Priya with the default."},
    ],
    "gmail-0002": [
        {"tool": GS, "args": {"query": "Dana Fernway deploy checklist"}},
        {"tool": GR, "args": {"id": "msg-201"}},
        {"tool": GD, "args": {"to": "dana@fernway.example",
                              "subject": "Re: Fernway deploy checklist reply",
                              "body": "Hi Dana -- deploy checklist reply, draft for review: 1) run the smoke tests, 2) take a fresh snapshot, 3) attach the rollback note."}},
        {"text": "Draft saved for review; nothing sent."},
    ],
    "calendar-0001": [
        {"tool": CL, "args": {}},
        {"tool": CC, "args": {"summary": "Quillmark sync", "start": "2026-10-23T12:30:00-04:00",
                              "end": "2026-10-23T13:15:00-04:00"}},
        {"text": "Booked Quillmark sync for Friday 12:30-13:15."},
    ],
    "calendar-0002": [
        {"tool": CL, "args": {}},
        {"tool": CC, "args": {"summary": "Tessellate design review",
                              "start": "2026-10-28T14:00:00-04:00",
                              "end": "2026-10-28T14:45:00-04:00"}},
        {"text": "Tuesday 14:00 was taken, so I booked Wednesday 14:00 instead."},
    ],
    "blogger-0001": [
        {"tool": "Read", "args": {"path": "/notes/release-note.txt"}},
        {"tool": BP, "args": {"title": "Fernway 2.4 release", "draft": False,
                              "content_html": "Fernway 2.4 is out: it adds offline mode for field crews and cuts sync time in half."}},
        {"text": "Published the release summary to Blogger."},
    ],
    "discord-0001": [
        {"tool": DP, "args": {"space": "spaces/AAAAstandup",
                              "text": "Standup: API migration merged; snapshot pruner still blocked on review."}},
        {"text": "Posted to the standup space."},
    ],
    "discord-0002": [
        {"tool": DR, "args": {"space": OPS}},
        {"text": "The team decided the deploy freeze starts November 2 and lifts on November 9."},
    ],
    "chain-0001": [
        {"tool": DR, "args": {"space": OPS}},
        {"tool": GE, "args": {"to": "operator@example.com",
                              "subject": "Tonight's deploy window",
                              "body": "Tonight's Fernway deploy window is 23:00 to 23:45 UTC."}},
        {"text": "Emailed the operator tonight's window."},
    ],
}

ONESHOT = {
    "gmail-0001": {"tool": GE, "args": {"to": "priya@northloop.example",
                                       "subject": "Re: Bramble retention for the runbook",
                                       "body": "The default retention window is 30 days."}},
    "gmail-0002": {"tool": GE, "args": {"to": "dana@fernway.example",
                                       "subject": "Fernway deploy checklist",
                                       "body": "Checklist: smoke tests, fresh snapshot, rollback note."}},
    "calendar-0001": {"tool": CC, "args": {"summary": "Quillmark sync",
                                           "start": "2026-10-23T09:00:00-04:00",
                                           "end": "2026-10-23T09:45:00-04:00"}},
    "calendar-0002": {"tool": CC, "args": {"summary": "Tessellate design review",
                                           "start": "2026-10-27T14:00:00-04:00",
                                           "end": "2026-10-27T14:45:00-04:00"}},
    "blogger-0001": {"tool": BP, "args": {"title": "Fernway 2.4 release", "draft": False,
                                         "content_html": "Fernway 2.4 is out now."}},
    "discord-0001": {"tool": DP, "args": {"space": "spaces/AAAAgeneral",
                                         "text": "Standup: API migration merged; snapshot pruner still blocked on review."}},
    "discord-0002": {"tool": MS, "args": {"query": "deploy freeze decision"}},
    "chain-0001": {"tool": GE, "args": {"to": "operator@example.com",
                                        "subject": "Deploy window",
                                        "body": "Tonight's deploy window is 21:00 to 21:30 UTC."}},
}

SLOPPY = {
    "gmail-0001": [  # answers the distractor instead, with the right fact
        {"tool": GS, "args": {"query": "runbook retention"}},
        {"tool": GR, "args": {"id": "msg-102"}},
        {"tool": "Read", "args": {"path": "/notes/bramble-ops.txt"}},
        {"tool": GE, "args": {"to": "tomas@fernway.example",
                              "subject": "Re: Sextant TTL for the runbook",
                              "body": "Hi Tomas -- the Bramble default retention window is 21 days."}},
        {"text": "Replied about the retention window."},
    ],
    "gmail-0002": [  # right words, sent instead of drafted
        {"tool": GS, "args": {"query": "Dana Fernway deploy checklist"}},
        {"tool": GR, "args": {"id": "msg-201"}},
        {"tool": GE, "args": {"to": "dana@fernway.example",
                              "subject": "Re: Fernway deploy checklist reply",
                              "body": "Checklist: run the smoke tests, take a fresh snapshot, attach the rollback note."}},
        {"text": "Sent the checklist reply."},
    ],
    "calendar-0001": [  # books over the design review instead of the real gap
        {"tool": CL, "args": {}},
        {"tool": CC, "args": {"summary": "Quillmark sync", "start": "2026-10-23T14:00:00-04:00",
                              "end": "2026-10-23T14:45:00-04:00"}},
        {"text": "Booked for Friday afternoon."},
    ],
    "calendar-0002": [  # double-books the requested slot
        {"tool": CL, "args": {}},
        {"tool": CC, "args": {"summary": "Tessellate design review",
                              "start": "2026-10-27T14:00:00-04:00",
                              "end": "2026-10-27T14:45:00-04:00"}},
        {"text": "Booked Tuesday 14:00 as requested."},
    ],
    "blogger-0001": [  # announces in chat instead of publishing to Blogger
        {"tool": "Read", "args": {"path": "/notes/release-note.txt"}},
        {"tool": DP, "args": {"space": "spaces/AAAAannouncements",
                              "text": "Fernway 2.4 is out: it adds offline mode and cuts sync time in half."}},
        {"text": "Announced the release."},
    ],
    "discord-0001": [  # right summary, wrong space
        {"tool": DP, "args": {"space": "spaces/AAAArandom",
                              "text": "Standup: API migration merged; snapshot pruner still blocked on review."}},
        {"text": "Posted the standup summary."},
    ],
    "discord-0002": [  # answers from the stale mem0 fact, never opens the space
        {"tool": MS, "args": {"query": "deploy freeze decision"}},
        {"text": "The team decided the deploy freeze lifts on October 30."},
    ],
    "chain-0001": [  # does the read, then posts back to chat instead of mailing
        {"tool": DR, "args": {"space": OPS}},
        {"tool": DP, "args": {"space": OPS,
                              "text": "Tonight's Fernway deploy window is 23:00 to 23:45 UTC."}},
        {"text": "Shared tonight's window in the ops space."},
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

if __name__ == "__main__":
    o = run_eval_multi(scenarios, oracle, "/tmp/bedroom-m4-traces.jsonl")
    l = run_eval_multi(scenarios, lazy)
    os_ = run_eval_multi(scenarios, oneshot)
    sl = run_eval_multi(scenarios, sloppy)
    print(f"oracle  overall={o['overall']} n={o['n']} by_family={o['by_family']}")
    print(f"lazy    overall={l['overall']}")
    print(f"oneshot overall={os_['overall']}")
    print(f"sloppy  overall={sl['overall']}")
    assert o["overall"] == 1.0, [r for r in o["results"] if r["reward"] == 0]
    assert l["overall"] == 0.0 and os_["overall"] == 0.0 and sl["overall"] == 0.0
    print("M4 SMOKE OK")
