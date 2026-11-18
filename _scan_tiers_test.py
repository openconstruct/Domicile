#!/usr/bin/env python3
"""The fact store's scan cache: incremental updates and the two scan tiers.

Run through _test_runner.py. Vectors are synthetic (clustered unit vectors), so
no model is loaded.
"""
import json
import sys
import tempfile
import time

import numpy as np

import _test_runner

_test_runner.require_sandbox()

from domicile_mem0 import vectors as V
from domicile_mem0.core import MemoryCore

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)


DIM = 384
rng = np.random.default_rng(7)
centres = rng.standard_normal((40, DIM)).astype(np.float32)


def vec(cluster, jitter=0.6):
    v = centres[cluster] + jitter * rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class Lookup:
    """Embedder double: returns the vector registered for a text."""
    def __init__(self):
        self.table = {}

    def embed(self, text):
        return self.table[text]


emb = Lookup()
db = tempfile.mkdtemp() + "/flash.db"
core = MemoryCore(db, emb)
ids = []
for n in range(3000):
    text = f"fact {n} about topic {n % 40}"
    emb.table[text] = vec(n % 40)
    ids.append(core.add_verbatim(text, user_id="u" if n % 5 else "other")["results"][0]["id"])


def state(c):
    rows, index, scan, cols = c._snapshot()
    order = np.argsort([r["id"] for r in rows])
    return ([rows[i]["id"] for i in order], [rows[i]["text"] for i in order],
            [rows[i]["metadata"] for i in order],
            scan.matrix[order] if not scan.tiered else scan.codes[order],
            {n: a[order] for n, a in cols.items()})


def same(a, b):
    return (a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and np.array_equal(a[3], b[3])
            and all(np.array_equal(a[4][n], b[4][n], equal_nan=(n == "created_at"))
                    if a[4][n].dtype != object else list(a[4][n]) == list(b[4][n])
                    for n in a[4]))


# --- 1. incremental cache == a fresh full load, after every kind of write ------
core._snapshot()
base = core.rebuilds
for step in range(120):
    kind = step % 6
    target = ids[int(rng.integers(len(ids)))]
    if kind == 0:
        text = f"new fact {step}"
        emb.table[text] = vec(step % 40)
        ids.append(core.add_verbatim(text, user_id="u")["results"][0]["id"])
    elif kind == 1:
        text = f"rewritten {step}"
        emb.table[text] = vec((step * 7) % 40)
        core.update(target, text=text)
    elif kind == 2:
        core.update(target, metadata={"step": step})
    elif kind == 3:
        core.mark_superseded(target, ids[0], "test")
    elif kind == 4:
        core.delete(target)
        ids.remove(target)
    else:
        core.store.db.execute("UPDATE memories SET superseded_by=NULL WHERE id=?", (target,))
        core.store.db.commit()
        core.store._bump(target)                 # un-supersede, the way a restore would
    if step % 10 == 0:
        core._snapshot()                         # interleave reads with the writes
fresh = MemoryCore(db, emb)
check("incremental: matches a fresh full load", same(state(core), state(fresh)))
check("incremental: no full reload was needed", core.rebuilds == base, f"{core.rebuilds - base} reloads")
q_text = "probe"
emb.table[q_text] = vec(3, jitter=0.3)
a = [r["id"] for r in core.search(q_text, user_id="u", limit=10)["results"]]
b = [r["id"] for r in fresh.search(q_text, user_id="u", limit=10)["results"]]
check("incremental: search agrees with a fresh load", a == b, f"{a[:3]} vs {b[:3]}")

# A reader's snapshot must not change under it.
rows0, _i0, scan0, _c0 = core._snapshot()
n0 = len(rows0)
emb.table["late"] = vec(1)
core.add_verbatim("late", user_id="u")
core._snapshot()
check("copy-on-write: an old snapshot is untouched", len(rows0) == n0 == len(scan0))

# Outrunning the journal falls back to one full reload, and stays correct.
for n in range(core.MAX_INCREMENTAL + 5):
    emb.table[f"burst {n}"] = vec(n % 40)
    core.add_verbatim(f"burst {n}", user_id="u")
before = core.rebuilds
core._snapshot()
check("burst of writes: one full reload", core.rebuilds == before + 1)
check("burst of writes: still matches a fresh load", same(state(core), state(MemoryCore(db, emb))))

# --- 2. what a write costs the next read ---------------------------------------
emb.table["timed"] = vec(2)
core.add_verbatim("timed", user_id="u")
t = time.perf_counter(); core._snapshot(); inc_ms = 1e3 * (time.perf_counter() - t)
t = time.perf_counter(); core._rebuild(); full_ms = 1e3 * (time.perf_counter() - t)
print(f"     next read after a write: {inc_ms:.2f} ms incremental vs {full_ms:.1f} ms full reload "
      f"({len(core._cache_rows)} facts)")
check("incremental is much cheaper than a reload", inc_ms * 5 < full_ms, f"{inc_ms:.2f} vs {full_ms:.2f}")

# --- 3. tiered scan (bits -> int8 -> exact float) vs the exact scan -------------
exact = MemoryCore(db, emb)
old_max = V.EXACT_MAX
V.EXACT_MAX = 0
tiered = MemoryCore(db, emb)
_r, _i, tscan, _c = tiered._snapshot()       # the tier is chosen when the cache is built
V.EXACT_MAX = old_max
_r, _i, escan, _c = exact._snapshot()
check("tiered: holds bits + int8, no float32 matrix", tscan.tiered and tscan.matrix is None)
check("tiered: uses under a third of the RAM", tscan.nbytes * 3 < escan.nbytes,
      f"{tscan.nbytes} vs {escan.nbytes}")
hits = total = 0
worst = 0.0
for n in range(60):
    qt = f"q{n}"
    emb.table[qt] = vec(n % 40, jitter=0.5)
    e = exact.search(qt, user_id="u", limit=10)["results"]
    t_ = tiered.search(qt, user_id="u", limit=10)["results"]
    hits += len({r["id"] for r in e} & {r["id"] for r in t_})
    total += len(e)
    by_id = {r["id"]: r["similarity"] for r in e}
    worst = max([worst] + [abs(r["similarity"] - by_id[r["id"]]) for r in t_ if r["id"] in by_id])
print(f"     tiered recall@10 vs exact: {hits / total:.3f}; worst cosine difference {worst:.2e}")
check("tiered: recall@10 >= 0.95", hits / total >= 0.95, f"{hits / total:.3f}")
check("tiered: returned scores are the exact cosine", worst < 1e-5, f"{worst:.2e}")
other = [r["user_id"] for r in tiered.search("q1", user_id="other", limit=10)["results"]]
check("tiered: user filter holds", other and set(other) == {"other"}, str(other))
# incremental updates in tiered mode
emb.table["tiered new"] = emb.table["q2"]
nid = tiered.add_verbatim("tiered new", user_id="u")["results"][0]["id"]
top = tiered.search("q2", user_id="u", limit=3)["results"]
check("tiered: a new fact is searchable at once", top and top[0]["id"] == nid)
tiered.delete(nid)
check("tiered: a deleted fact is gone at once",
      nid not in [r["id"] for r in tiered.search("q2", user_id="u", limit=10)["results"]])
check("tiered: no reload for either", tiered.rebuilds == 1, str(tiered.rebuilds))

# --- 4. the flash adapter still works on top (guard uses the scanner) ----------
import mem0_v2 as F
F._core, F.USER_ID = tiered, "u"
emb.table["twin a"] = emb.table["twin b"] = vec(9)
added = (tiered.add_verbatim("twin a", user_id="u")["results"]
         + tiered.add_verbatim("twin b", user_id="u")["results"])
F._guard(added, cutoff=0.9)
check("guard on the tiered scan: one of two twins survives",
      [x.get("event") for x in added].count("GUARDED") == 1, str(added))
F._core = exact
out = F.search("q5", limit=3)
check("adapter search on the exact scan", out.count("\n") == 2 and out.startswith("["), out[:80])
F._core = None

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all passed")
