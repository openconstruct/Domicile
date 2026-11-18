#!/usr/bin/env python3
"""Lessons: stored apart from facts, never stale, hinted at a lower bar.

Run through _test_runner.py. Cosines are constructed exactly, so the threshold
behaviour is tested rather than guessed at.
"""
import json
import os
import sys
import tempfile

import numpy as np

import _test_runner

_test_runner.require_sandbox()
os.environ["MEM0_V2_DB"] = tempfile.mkdtemp() + "/flash.db"

import hints
import mem0_ipc
import mem0_v2 as F
from domicile_mem0.core import MemoryCore

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)


DIM = 384
rng = np.random.default_rng(3)


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


QUERY = unit(rng.standard_normal(DIM))


def at_cosine(c):
    """A unit vector whose cosine with QUERY is exactly c."""
    r = rng.standard_normal(DIM)
    r -= (r @ QUERY) * QUERY
    return unit(c * QUERY + np.sqrt(1 - c * c) * unit(r))


class Lookup:
    def __init__(self):
        self.table = {}

    def embed(self, text):
        if text not in self.table:          # conversation chunks: any unrelated vector
            self.table[text] = at_cosine(0.2)
        return self.table[text]


emb = Lookup()
F._core = MemoryCore(os.environ["MEM0_V2_DB"], emb)
F.USER_ID = "u"
T = F.HINT_THRESHOLD
OLD = "2025-01-01"                      # far past the staleness horizon
emb.table["the question"] = QUERY


def put(text, cos, lesson=False, date=None):
    stored = text if not lesson or text.startswith("Lesson:") else f"Lesson: {text}"
    emb.table[stored] = at_cosine(cos)
    md = {"date": date} if date else None
    if lesson:
        out = F.add_lesson(text, metadata=md)
        assert out.startswith("stored lesson"), out
        return stored
    F.add_verbatim(text, metadata=md)
    return text


def hinted():
    return [m for _c, _d, m in F.hint("the question", limit=10)]


# --- storing -------------------------------------------------------------------
a = put("when editing the wizard units, change the template not the installed file, because install() never overwrites", T - 0.03, lesson=True, date=OLD)
row = F._all_rows()[0]
md = json.loads(row["metadata"])
check("stored with the Lesson: prefix", row["text"].startswith("Lesson: when editing"))
check("tagged in metadata", md.get("lesson") is True and md.get("kind") == "lesson", str(md))
check("storing it twice is a no-op", F.add_lesson(a) == "already stored")
check("over the socket op too", mem0_ipc._dispatch_to_core("add_lesson", {"text": a}) == "already stored")
check("empty lesson refused", F.add_lesson("  ") == "nothing to store")

# --- hints: lower bar, never stale ----------------------------------------------
from datetime import date, timedelta
PAST_BONUS = (date.today() - timedelta(days=70)).isoformat()   # no "made recently" lift
b = put("an ordinary fact just under the bar", T - 0.03, date=PAST_BONUS)
c = put("an old ordinary fact comfortably over the bar", T + 0.05, date=OLD)
d = put("a fresh ordinary fact over the bar", T + 0.05)
e = put("a lesson too far under the bar to matter", T - 0.12, lesson=True)
got = hinted()
check("a lesson just under the bar is hinted", a in got, str(got))
check("an ordinary fact at the same cosine is not", b not in got)
check("an old lesson is not faded (an old fact at higher cosine is)", a in got and c not in got)
check("fresh facts over the bar still hint", d in got)
check("an irrelevant lesson is still not hinted", e not in got)

# --- search: a lesson outranks a fact at the same similarity ---------------------
f = put("search twin fact", 0.60, date=OLD)
g = put("search twin lesson", 0.60, lesson=True, date=OLD)
lines = F.search("the question", limit=10).splitlines()
pos = {t: i for i, line in enumerate(lines) for t in (f, g) if line.endswith(t)}
check("lesson ranks above an equally similar old fact", pos[g] < pos[f], str(pos))

# --- extraction marks lessons by their prefix -------------------------------------
import mem0_chain
check("extractor is told about lessons", "Lesson:" in mem0_chain.CUSTOM_INSTRUCTIONS)
F._llm_ready = True
F._core.llm = lambda s, u: json.dumps({"memory": [
    {"text": "Lesson: never run mem2 --reembed while cchat is up, because both write mem2.db"},
    {"text": "Alex prefers prose over bullet points"}]})
for t in ("Lesson: never run mem2 --reembed while cchat is up, because both write mem2.db",
          "Alex prefers prose over bullet points",
          "user: a chunk of conversation"):
    emb.table[t] = at_cosine(0.2)
out = F.add("user: a chunk of conversation", extract=True)
check("both extracted", out.count("ADD:") == 2, out)
tagged = {r["text"]: json.loads(r["metadata"]).get("lesson") for r in F._all_rows()}
check("the extracted lesson is tagged",
      tagged["Lesson: never run mem2 --reembed while cchat is up, because both write mem2.db"] is True)
check("the extracted fact is not", not tagged["Alex prefers prose over bullet points"])

# --- rendering keeps the "because" ------------------------------------------------
long_lesson = "Lesson: " + " ".join(f"w{i}" for i in range(40)) + " because reason"
long_fact = " ".join(f"w{i}" for i in range(40))
check("a lesson keeps 45 words in a hint", hints.snippet(long_lesson).endswith("because reason"))
check("a fact is still clipped at 20", hints.snippet(long_fact).endswith("w19 …"))
check("is_lesson by prefix or tag", F.is_lesson("Lesson: x") and F.is_lesson("y", {"lesson": True})
      and not F.is_lesson("a lesson learned"))

F._core.close()
F._core = None
print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all passed")
