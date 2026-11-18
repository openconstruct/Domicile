#!/usr/bin/env python3
"""memory_source: from a stored fact back to the conversation it came from.

Run through _test_runner.py. The model is a bag-of-words fake, so a fact and
the passage it paraphrases land close together without loading anything.
"""
import hashlib
import os
import sys
import tempfile

import numpy as np

import _test_runner

_test_runner.require_sandbox()
os.environ["MEM0_V2_DB"] = tempfile.mkdtemp() + "/flash.db"

from domicile_mem0 import embedders as E

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)


class BagOfWords:
    def embed(self, texts):
        for t in texts:
            v = np.zeros(E.EMBED_DIMS, dtype=np.float32)
            for tok in "".join(ch if ch.isalnum() else " " for ch in t.lower()).split():
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % E.EMBED_DIMS] += 1.0
            yield v


E._model = BagOfWords()

import mem0_ipc
import mem0_v2 as F
import mem2_core as M
import paths

F.USER_ID = "u"
paths.INGESTED_DIR.mkdir(parents=True, exist_ok=True)
for stale in paths.INGESTED_DIR.glob("*.log"):
    stale.unlink()


def marker(day, hour):
    return f"=== compacted from chat 'abc' ('Main Chat') at {day}T{hour}:00:00 ===\n\n"


(paths.INGESTED_DIR / "2026-10-01.log").write_text(
    marker("2026-10-01", "02")
    + "### [user]\nWhere should the spare key live now that the lockbox rusted shut?\n\n"
      "### [assistant]\nAlex decided the spare key goes under the blue pot by the back "
      "door, because the lockbox rusted shut and the pot is out of sight from the road.\n\n"
    + marker("2026-10-01", "09")
    + "### [user]\nThe nightly backup failed again.\n\n### [assistant]\nThe push timed out "
      "against the stalled remote; we added a thirty minute git timeout.\n")
(paths.INGESTED_DIR / "2026-10-02.log").write_text(
    marker("2026-10-02", "03")
    + "### [user]\nMove the spare key somewhere else.\n\n### [assistant]\nA different day: "
      "the spare key under the blue pot was discussed again, nothing changed.\n")

fid = F.add_verbatim("Alex keeps the spare key under the blue pot by the back door",
                     metadata={"kind": "compaction-log", "date": "2026-10-01", "source": "hourly"})
bid = F.add_verbatim("Backup pushes have a thirty minute git timeout",
                     metadata={"kind": "compaction-log", "date": "2026-10-01", "source": "hourly"})

# --- search shows the id that memory_source takes --------------------------------
line = F.search("where is the spare key", limit=1)
check("search shows the short id", f"({fid[:8]})" in line, line)

# --- by id: the right day, the right passage --------------------------------------
out = F.source(fid[:8])
check("source: names the memory", out.startswith(f"memory ({fid[:8]}, 2026-10-01)"), out[:80])
check("source: names the log", "source: 2026-10-01.log" in out)
first = out.split("[1] ")[1].split("[2] ")[0]
check("source: best passage is where it was said",
      "lockbox rusted shut" in first and "blue pot" in first, first[:160])
check("source: carries the reasoning the fact dropped", "out of sight from the road" in out)
check("source: only that day's transcript", "A different day" not in out)
check("source: gives a ref mem2_get_chunk can read", "ref=" in first and "#" in first.split("ref=")[1].split()[0])
ref = first.split("ref=")[1].split()[0]
src, _, idx = ref.rpartition("#")
check("source: the ref resolves", "lockbox" in M.dispatch("mem2_get_chunk",
                                                          {"source_file": src, "chunk_index": int(idx)}))
other = F.source(bid[:8], limit=1)
check("source: a different fact from the same day finds its own passage",
      "git timeout" in other.split("[1] ")[1] and "lockbox" not in other.split("[1] ")[1], other[-200:])

# --- by words, and over the socket op ---------------------------------------------
check("source: found by its words when not an id", "lockbox rusted shut" in F.source("spare key blue pot"))
check("source: socket op", mem0_ipc._dispatch_to_core("source", {"ref": fid[:8], "limit": 1}).count("[1] ") == 1)
check("source: empty ref asks for one", F.source(" ").startswith("give a memory id"))

# --- a fact from today's log, which full memory has not indexed yet ----------------
(paths.MEM0_LOG_DIR / "2026-11-12.log").write_text(
    marker("2026-11-12", "01")
    + "### [assistant]\nThe agent raised the wire diet floor to twenty eight thousand characters "
      "after the review showed the old floor fired every time.\n")
tid = F.add_verbatim("The wire diet floor is twenty eight thousand characters",
                     metadata={"kind": "compaction-log", "date": "2026-11-12", "source": "hourly"})
pend = F.source(tid[:8])
check("pending log: falls back to the log file itself",
      "not indexed yet" in pend and "old floor fired every time" in pend, pend[:300])

# --- a fact with no transcript behind it --------------------------------------------
nid = F.add_verbatim("An imported note with no conversation behind it",
                     metadata={"kind": "cmem", "date": "2024-01-01"})
none = F.source(nid[:8])
check("no transcript: says so, with the origin", "no transcript for 2024-01-01" in none and "cmem" in none, none)

F._core.close()
F._core = None
print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all passed")
