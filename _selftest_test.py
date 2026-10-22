#!/usr/bin/env python3
"""Tests for selftest.evaluate_for_shout -- the stateful shout-diffing.

The checks themselves read the live box and are exercised by running
`selftest.py`. What needs a test is the part with memory: shout on a new
failure, stay quiet while it stands, re-shout only after the interval,
announce recovery once.
"""
from __future__ import annotations

import datetime as dt
import json
import tempfile
from pathlib import Path

import selftest
from selftest import Check, OK, FAIL


def _redirect(tmp: Path):
    selftest.SELFTEST_STATUS = tmp / "selftest-status.json"
    selftest.SELFTEST_LOG = tmp / "selftest.log"


def _age_status(hours: float) -> None:
    p = json.loads(selftest.SELFTEST_STATUS.read_text())
    p["ts"] = (dt.datetime.now() - dt.timedelta(hours=hours)).isoformat(
        timespec="seconds")
    selftest.SELFTEST_STATUS.write_text(json.dumps(p))


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        _redirect(Path(d))

        all_ok = [Check("a", OK), Check("b", OK)]
        one_bad = [Check("a", OK), Check("b", FAIL, "b broke")]

        # 1. first run, everything ok -> silence, status written
        assert selftest.evaluate_for_shout(all_ok) is None
        assert selftest.SELFTEST_STATUS.exists()
        print("1. clean first run: silent OK")

        # 2. a check newly fails -> shout naming it
        msg = selftest.evaluate_for_shout(one_bad)
        assert msg and "`b`" in msg and "b broke" in msg, msg
        print("2. new failure shouts OK")

        # 3. same failure still standing, interval not elapsed -> silence
        assert selftest.evaluate_for_shout(one_bad) is None
        print("3. standing failure stays quiet OK")

        # 4. same failure, 13h later -> re-shout
        _age_status(13)
        msg = selftest.evaluate_for_shout(one_bad)
        assert msg and "still failing" in msg, msg
        print("4. re-shout after 12h OK")

        # 5. recovery -> announce once, then silent
        msg = selftest.evaluate_for_shout(all_ok)
        assert msg and "recovered" in msg and "b" in msg, msg
        assert selftest.evaluate_for_shout(all_ok) is None
        print("5. recovery announced once OK")

        # 6. two checks fail at once -> both named in one shout
        two_bad = [Check("a", FAIL, "a broke"), Check("b", FAIL, "b broke")]
        msg = selftest.evaluate_for_shout(two_bad)
        assert "`a`" in msg and "`b`" in msg, msg
        print("6. simultaneous failures batched OK")

    print("\nALL OK")


if __name__ == "__main__":
    main()
