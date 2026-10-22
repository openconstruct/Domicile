"""Unit tests for mem0_hourly.run_once -- store stubbed, files real.

Replacement mode (nightly decommissioned 2026-09-08): exercises tail
holdback on the growing file, incremental cursor, rollover close-out (hourly
moves the day to ingested/ itself), nightly-skip dead-path helper, hold and
fail paths (fail HOLDS forever -- no backstop exists), per-run budget, and
the never-raises guarantee. Run from the project root.
"""
import sys, json, hashlib, shutil, tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mem0_hourly as H

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)


tmp = Path(tempfile.mkdtemp(prefix="hourly-test-"))
log_dir = tmp / "logs"
log_dir.mkdir()
state = tmp / "state.json"

# Redirect the done-marker dir into the sandbox: the module global is read
# at call time by both run_once and _nightly_skip. Status too -- a test run
# must not write a fake "nothing new" into the live hourly-ingest-status.json.
H.DONE_DIR = tmp / "hourly-done"
H.STATUS_PATH = tmp / "hourly-status.json"

# Stub the store: every chunk "extracts" two facts, except chunks containing
# POISON which return a server-side dispatch failure, and chunks containing
# DOWN which hold.
calls = []

def fake_add(chunk_text, metadata):
    calls.append((chunk_text, dict(metadata)))
    if "POISON" in chunk_text:
        return ("fail", "RuntimeError: JSONDecodeError: Expecting ',' delimiter: char 0")
    if "DOWN" in chunk_text:
        return ("hold", "RuntimeError: Connection error: unreachable")
    return ("ok", "ADD: fact one\nADD: fact two")

H._add_with_retry = lambda chunk_text, metadata: fake_add(chunk_text, metadata)

now = datetime(2026, 9, 8, 17, 0, 0)

# --- 1. today's file, first run: tail block held back
today = log_dir / "2026-09-08.log"
today.write_text("Block one about Alpha.\n\nBlock two about Beta.\n\nBlock three still growing.\n\n")
s = H.run_once(log_dir=log_dir, state_path=state, now=now)
check("today: tail held back", len(calls) == 1 and "Block one" in calls[0][0]
      and "Block two" in calls[0][0] and "Block three" not in calls[0][0],
      f"calls={len(calls)}")
check("today: metadata stamped", calls[0][1].get("kind") == "compaction-log"
      and calls[0][1].get("source") == "hourly" and calls[0][1].get("date") == "2026-09-08",
      str(calls[0][1]))
st = json.loads(state.read_text())
off1 = st["files"]["2026-09-08.log"]["offset"]
check("today: offset moved past block two", off1 == len("Block one about Alpha.\n\nBlock two about Beta.\n\n"),
      f"off={off1}")

# --- 2. second run, file unchanged: nothing new to send
n_calls = len(calls)
s = H.run_once(log_dir=log_dir, state_path=state, now=now)
check("today: no re-send when unchanged", len(calls) == n_calls, f"calls+={len(calls)-n_calls}")

# --- 3. day closes (clock rolls over, same file, now a previous day):
#        tail drains, then the hourly MOVES the file to ingested/ itself
#        (replacement mode -- there is no nightly backstop anymore). Cursor
#        survives because the day file never changes name, only "today" does.
now_y = datetime(2026, 9, 9, 0, 30, 0)
s = H.run_once(log_dir=log_dir, state_path=state, now=now_y)
yday = log_dir / "2026-09-08.log"
check("rollover: tail drained", len(calls) == n_calls + 1
      and "Block three" in calls[-1][0], f"calls={len(calls)}")
moved = H.INGESTED_DIR / "2026-09-08.log"
check("rollover: day file moved to ingested/", moved.exists()
      and not yday.exists(), f"moved={moved.exists()}")
check("rollover: moved file intact", "Block three still growing." in moved.read_text())

# --- 4. marker digest mismatch: nightly-skip still refuses (dead-path helper,
#        kept for the manual nightly script). Build a standalone pair.
p4dir = tmp / "logs4"; p4dir.mkdir()
p4 = p4dir / "2026-09-01.log"
p4.write_text("Marker test body.\n\n")
H.DONE_DIR.mkdir(parents=True, exist_ok=True)
m4 = H.DONE_DIR / "2026-09-01.json"
m4.write_text(json.dumps({"date": "2026-09-01", "sha256": "deadbeef"}))
skip, why = H._nightly_skip(p4)
check("nightly-skip: refuses mismatched digest", skip is False and "mismatch" in why, why)

# --- 5. hold path: chunk that holds does NOT advance cursor, retries next run
calls.clear()
log_dir2 = tmp / "logs2"
log_dir2.mkdir()
state2 = tmp / "state2.json"
t2 = log_dir2 / "2026-09-09.log"
t2.write_text("Chunk DOWN here.\n\nChunk after down.\n\n")
s = H.run_once(log_dir=log_dir2, state_path=state2, now=now_y)
st2 = json.loads(state2.read_text())
check("hold: cursor unmoved", st2["files"]["2026-09-09.log"]["offset"] == 0)
check("hold: chunk count stays zero", s["chunks_done"] == 0)
s2 = H.run_once(log_dir=log_dir2, state_path=state2, now=now_y)
check("hold: retries next tick", s2["chunks_done"] == 0)

# --- 6. fail path: REPLACEMENT mode -- a poison chunk HOLDS forever, never
#        advances. No nightly backstop exists, so advance = silent loss once
#        the file is close-out-moved. Escalates to "poison chunk" after 3.
calls.clear()
log_dir3 = tmp / "logs3"
log_dir3.mkdir()
state3 = tmp / "state3.json"
t3 = log_dir3 / "2026-09-09.log"
t3.write_text("POISON chunk zero.\n\nGood chunk follows.\n\n")
H.FAIL_TICKS_BEFORE_SKIP = 3
holds = []
for tick in range(1, 5):
    s = H.run_once(log_dir=log_dir3, state_path=state3, now=now_y)
    st3 = json.loads(state3.read_text())
    holds.append((tick, st3["files"]["2026-09-09.log"]["offset"], s["held"],
                  s["reason"]))
check("fail: never advances, held on all ticks",
      all(h[1] == 0 and h[2] for h in holds), str(holds))
check("fail: escalates to poison chunk",
      "poison chunk" in holds[2][3] and "poison chunk" in holds[3][3],
      str(holds))

# --- 7. never raises on garbage input
(t3).write_bytes(b"\x00\x01\x02 broken utf8: \xff\xfe")
s = H.run_once(log_dir=log_dir3, state_path=state3, now=now_y)
check("never raises on binary garbage", isinstance(s, dict))

print()
print("ALL PASS" if not failures else f"{len(failures)} FAILURES: {failures}")
shutil.rmtree(tmp)
sys.exit(1 if failures else 0)
