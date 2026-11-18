#!/usr/bin/env python3
"""Tool results must not lose their middle, and a cut must stay recoverable.

Run through _test_runner.py (disposable data root). Covers the 2026-11-12
review fixes: Read/WebFetch page instead of being cut, the wire diet never
re-cuts an already-cut result, fetch_result's default slice survives the
at-source cap, and the flash-memory guard / extraction parsing bugs.
"""
import asyncio
import os
import re
import sys
import tempfile
from pathlib import Path

import _test_runner

_test_runner.require_sandbox()

import context_diet as cd
import fetch_result
import local_tools as LT
import tool_spill
from providers import openai_compatible as OC

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def line_gaps(text):
    nums = [int(n) for n in re.findall(r"line (\d{4}) ", text)]
    return nums, [(a, b) for a, b in zip(nums, nums[1:]) if b != a + 1]


tmp = Path(tempfile.mkdtemp(prefix="nodrop-", dir=str(Path.home())))
os.environ["CCHAT_SPILL_DIR"] = str(tmp / "spill")

# --- 1. Read pages to the caller's budget, contiguous, with a pointer ----------
big = tmp / "big.txt"
big.write_text("".join(f"line {i:04d} " + "x" * 60 + "\n" for i in range(1, 1201)))

out = asyncio.run(LT.dispatch("Read", {"path": str(big)}, ["Read"]))
nums, gaps = line_gaps(out)
check("read: default dispatch has no hole", not gaps, str(gaps))
check("read: default dispatch fits the cap", len(out) <= LT.TOOL_RESULT_CHARS, str(len(out)))
check("read: pointer names the next line", f"offset={nums[-1] + 1} to continue" in out, out[-120:])

seen, offset, calls = [], 1, 0
while offset and calls < 200:
    calls += 1
    page = asyncio.run(LT.dispatch("Read", {"path": str(big), "offset": offset}, ["Read"],
                                   max_chars=None, page_chars=OC.TOOL_OUTPUT_MAX_CHARS))
    delivered = OC._truncate_tool_output(page, f"call_{calls}")
    check_ok = delivered == page
    if not check_ok:
        check("read: provider leaves a page whole", False, f"page {calls}: {len(page)} chars")
        break
    n, g = line_gaps(page)
    seen += n
    m = re.search(r"offset=(\d+) to continue", page)
    offset = int(m.group(1)) if m else 0
check("read: paging through the provider cap sees every line once",
      seen == list(range(1, 1201)), f"{len(seen)} lines in {calls} calls")

# The trusted (Claude) path has no budget set: one big window, as before.
res = asyncio.run(LT.execute("Read", {"path": str(big)}, ["Read"], trusted=True))
check("read: no budget outside dispatch", len(res["content"]) > 30_000, str(len(res["content"])))

# --- 2. WebFetch pages with `start` instead of dropping the middle -------------
page_text = "\n\n".join(f"para {i:04d} " + "word " * 40 for i in range(400))
LT._WEB_CACHE["https://example.test/long"] = (LT.time.time(), page_text)

got, start, calls = "", 0, 0
while calls < 200:
    calls += 1
    r = asyncio.run(LT.dispatch("WebFetch", {"url": "https://example.test/long", "start": start},
                                ["WebFetch"], max_chars=None,
                                page_chars=OC.TOOL_OUTPUT_MAX_CHARS))
    if OC._truncate_tool_output(r, f"web_{calls}") != r:
        check("webfetch: provider leaves a window whole", False, str(len(r)))
        break
    body, _, footer = r.rpartition("\n[showing chars ")
    got += body if footer else r
    m = re.search(r"start=(\d+) to continue", r)
    if not m:
        break
    start = int(m.group(1))
check("webfetch: windows reassemble to the whole page", got == page_text,
      f"{len(got)} vs {len(page_text)} in {calls} calls")
first = LT.web_fetch("https://example.test/long")["content"]
check("webfetch: no trimmed-middle marker", "trimmed" not in first)
check("webfetch: says where to continue", re.search(r"start=\d+ to continue", first) is not None)
past = LT.web_fetch("https://example.test/long", start=10 ** 9)["content"]
check("webfetch: start past the end is reported", "past the end" in past, past)

# --- 3. a generic cut in dispatch is recoverable -------------------------------
long_out = "".join(f"row {i:05d}\n" for i in range(3000))


async def _fake_execute(name, args, enabled, **kw):
    return long_out

_real_execute = LT.execute
LT.execute = _fake_execute
try:
    cut = asyncio.run(LT.dispatch("Grep", {}, ["Grep"]))
finally:
    LT.execute = _real_execute
m = tool_spill.KEY_IN_MARKER.search(cut)
check("dispatch: cut carries a recovery key", m is not None, cut[5900:6200])
check("dispatch: the saved copy is the full text", bool(m) and tool_spill.load(m.group(1)) == long_out)

# --- 4. the wire diet never cuts a cut -----------------------------------------
orig = "".join(f"line {i:04d} " + "y" * 40 + "\n" for i in range(1, 400))
msg = {"role": "tool", "tool_call_id": "c1"}
once = cd._wire_truncate(msg, orig, 2000, 1200, 600)
again = cd._wire_truncate(msg, once, 1000, 500, 500)
key = tool_spill.KEY_IN_MARKER.search(again)
check("diet: re-cut is made from the saved original",
      bool(key) and again.startswith(orig[:500]) and again.endswith(orig[-500:]))
(Path(os.environ["CCHAT_SPILL_DIR"]) / f"{key.group(1)}.txt").unlink()
check("diet: lost original => cut text left alone", cd._wire_truncate(msg, once, 1000, 500, 500) == once)
legacy = orig[:900] + "\n\n... [9999 chars trimmed] ...\n\n" + orig[-900:]
check("diet: unrecoverable legacy marker is not cut again",
      cd._wire_truncate(msg, legacy, 1000, 500, 500) == legacy)
mid = "z" * 1500                     # between the floor pass (1000) and the old spill minimum (2000)
cut_mid = cd._wire_truncate({"role": "tool", "tool_call_id": "c2"}, mid, 1000, 500, 500)
check("diet: a 1-2k result cut by the floor pass is recoverable",
      tool_spill.KEY_IN_MARKER.search(cut_mid) is not None, cut_mid[480:700])

# --- 5. fetch_result's default slice fits under the at-source cap --------------
k = tool_spill.spill("fr", long_out)
import contextlib
import io
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = fetch_result.main([k])
sl = buf.getvalue()
check("fetch_result: default slice passes the provider cap whole",
      rc == 0 and OC._truncate_tool_output(sl, "fr2") == sl, str(len(sl)))
check("fetch_result: slice is contiguous from 0", sl.startswith(long_out[:5000]))

# --- 5b. search inside a saved result instead of paging through it -------------
needle = long_out.replace("row 01777\n", "row 01777\nTraceback: RetryExhausted backoff max_retries=3 host db-7\n")
k2 = tool_spill.spill("fr3", needle)


def fetch(*args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = fetch_result.main([k2, *args])
    return code, out.getvalue()


rc, g = fetch("--grep", "retryexhausted", "-C", "1")
off = needle.index("row 01777")
check("grep: finds the line, case-insensitively, with context",
      rc == 0 and "RetryExhausted" in g and "row 01777" in g and "row 01778" in g, g[:200])
check("grep: labels the hit with its offset", f"@{off:,}" in g, g[:120])
rc, cont = fetch("--start", str(off), "--len", "40")
check("grep: that offset reads on from the hit", cont.startswith("row 01777"), cont[:40])
rc, f_ = fetch("--find", "what was the backoff host")
check("find: returns the passage with the words", "db-7" in f_ and "has backoff, host" in f_, f_[:200])
check("grep/find: output passes the provider cap whole",
      all(OC._truncate_tool_output(x, "s") == x for x in (g, f_, fetch("--grep", "row")[1])))
check("grep: an invalid regex is taken literally", fetch("--grep", "row 0177[")[1].startswith("no match"))
check("grep: no match says so", "no match" in fetch("--grep", "zzz")[1])
check("markers tell the model it can search",
      "--grep" in cut and "--grep" in once and tool_spill.KEY_IN_MARKER.search(once) is not None)

# --- 6. flash memory: guard and extraction parsing ----------------------------
import numpy as np

from domicile_mem0 import extract as X
from domicile_mem0.core import MemoryCore
from domicile_mem0.embedders import HashEmbedder

check("extract: null memory list", X.extract_additive("u: hi", [], lambda s, u: '{"memory": null}') == [])
check("extract: bare list of strings",
      X.extract_additive("u: hi", [], lambda s, u: '["a fact"]') == [{"text": "a fact"}])
check("extract: mixed junk is dropped",
      X.extract_additive("u: hi", [], lambda s, u: '{"memory": [{"text": "ok"}, 3, null]}') == [{"text": "ok"}])

core = MemoryCore(str(tmp / "flash.db"), HashEmbedder())
core.add_verbatim("the cat sat on the mat", user_id="u")
core.llm = lambda s, u: '{"memory": [{"text": "the cat sat on the mat"}, {"text": "crows bend wire into hooks"}]}'
r = core.add("irrelevant unrelated conversation text about gardening", user_id="u")["results"]
check("add: exact restatement is not stored twice",
      [x["memory"] for x in r] == ["crows bend wire into hooks"], str(r))
check("search: limit=None does not raise", len(core.search("cat mat", user_id="u", limit=None)["results"]) >= 1)

import mem0_v2 as V
V._core = core
V.USER_ID = "u"
a = core.add_verbatim("alex keeps the spare key under the blue pot", user_id="u")["results"]
b = core.add_verbatim("alex keeps the spare key under the blue pot.", user_id="u")["results"]
added = a + b
V._guard(added, cutoff=0.8)
events = [x.get("event") for x in added]
check("guard: of two new restatements exactly one survives",
      events.count("GUARDED") == 1, str(events))
V._core = None
core.close()

import shutil
shutil.rmtree(tmp, ignore_errors=True)
print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all passed")
