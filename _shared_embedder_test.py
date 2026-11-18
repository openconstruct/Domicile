#!/usr/bin/env python3
"""One embedder for every store, and mem2 served from the main process.

Run through _test_runner.py. The model is a counting fake, so nothing is
downloaded: what is checked is the wiring -- the fact store and full memory
call the same model object, and mem2's MCP front reaches the engine over the
memory socket instead of loading a store and a model of its own.
"""
import asyncio
import hashlib
import os
import sys
import tempfile
import threading

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


class FakeModel:
    loads = 0

    def __init__(self):
        FakeModel.loads += 1
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        for t in texts:
            v = np.zeros(E.EMBED_DIMS, dtype=np.float32)
            for tok in t.lower().split():
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % E.EMBED_DIMS] += 1.0
            yield v


fake = E._model = FakeModel()          # what _get_model() would have built

import mem0_ipc
import mem0_v2 as V
import mem2_core as M
import mem2_server as S
import paths

# --- one model object behind both stores --------------------------------------
V.USER_ID = "u"
V.add_verbatim("Crows bend wire into hooks to reach food")
after_fact = fake.calls
check("fact store embeds through the shared model", after_fact >= 1)
check("fact store stamps the shared model name",
      V.core().store.get_meta("embedder") == E.EMBED_MODEL)

paths.INGESTED_DIR.mkdir(parents=True, exist_ok=True)
(paths.INGESTED_DIR / "2026-10-01.log").write_text(
    "=== compacted from chat 'abc' ('Main Chat') at 2026-10-01T02:00:00 ===\n\n"
    "We raised the compaction ceiling to forty thousand tokens.\n")
for stale in paths.INGESTED_DIR.glob("2026-09-01.log"):
    stale.unlink()                      # fixture written by the runner; not needed here
out = M.dispatch("mem2_search", {"query": "compaction ceiling", "limit": 3})
check("mem2 finds the chunk", "compaction ceiling" in out, out[:200])
check("mem2 embeds through the same model object", fake.calls > after_fact)
check("mem2 stamps the same model name", M.get_meta(M._get_db(), "embedder") == E.EMBED_MODEL)
check("exactly one model was built", FakeModel.loads == 1 and E._get_model() is fake)
check("vectors are unit length", abs(float(np.linalg.norm(E.embed_one("a b c"))) - 1.0) < 1e-6)

# --- the MCP front forwards over the memory socket -----------------------------
served = []
real_dispatch = mem0_ipc._dispatch_to_core


def spy(op, args):
    served.append((op, threading.current_thread().name))
    return real_dispatch(op, args)


async def over_socket():
    server = await mem0_ipc.serve(spy)
    try:
        a = await asyncio.to_thread(S.mem2_search, "compaction ceiling", 2)
        b = await asyncio.to_thread(S.mem2_list_dates)
        ref = a.split("ref=")[1].split()[0]
        src, _, idx = ref.rpartition("#")
        c = await asyncio.to_thread(S.mem2_get_chunk, src, int(idx))
        d = await asyncio.to_thread(mem0_ipc.call, "search", {"query": "crows wire hooks"})
        many = await asyncio.gather(*[asyncio.to_thread(S.mem2_search, "ceiling", 1)
                                      for _ in range(6)])
        return a, b, c, d, many
    finally:
        server.close()
        await server.wait_closed()

a, b, c, d, many = asyncio.run(over_socket())
ops = [op for op, _ in served]
check("search went over the socket", ops.count("mem2_search") == 7, str(ops))
check("list_dates and get_chunk went over the socket",
      "mem2_list_dates" in ops and "mem2_get_chunk" in ops, str(ops))
check("socket search result", "compaction ceiling" in a, a[:200])
check("list_dates result", "2026-10-01" in b, b)
check("get_chunk result", "forty thousand" in c, c)
check("fact search shares the socket", "Crows bend wire" in d, d)
check("concurrent mem2 calls all answer", all("ceiling" in m for m in many))
check("still one model after all of it", FakeModel.loads == 1)

# --- no listener: the front opens the store itself -----------------------------
paths.SOCKET_PATH.unlink(missing_ok=True)
check("falls back to direct when cchat is down",
      "compaction ceiling" in S.mem2_search("compaction ceiling", 2))
check("the front holds no model of its own", not hasattr(S, "_embed") and not hasattr(S, "np"))

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all passed")
