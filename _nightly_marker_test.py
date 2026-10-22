"""Tests for the nightly done-marker skip partition (mem0_nightly.main).

Three cases:
  1. dry-run: marked day is logged as would-move, file stays in place.
  2. live + matching marker: file moved to ingested/ BEFORE any extraction.
  3. live + digest mismatch (file grew after the marker): job survives the
     partition, gets re-extracted (stubbed), then committed by the move --
     the backstop path.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import types
from pathlib import Path

import mem0_nightly as N
import mem0_hourly as H

BODY = "Block one.\n\nBlock two.\n\n"


def _make_day(tmp: Path, name: str = "2026-09-07.log") -> Path:
    logs = tmp / "logs"
    logs.mkdir(exist_ok=True)
    p = logs / name
    p.write_text(BODY)
    return p


def _marker(tmp: Path, path: Path, digest: str) -> None:
    done = tmp / "done"
    done.mkdir(exist_ok=True)
    (done / f"{path.stem}.json").write_text(json.dumps(
        {"date": path.stem, "sha256": digest, "at": "2026-09-08T18:00:00",
         "added": 0}, indent=2))


def _install(tmp: Path) -> None:
    N.LOG_DIR = tmp / "logs"
    N.LOG_INGESTED_DIR = tmp / "ingested"
    N.LOG_INGESTED_DIR.mkdir(exist_ok=True)
    N.REMEMBER_DIR = tmp / "remember"
    N.REMEMBER_DIR.mkdir(exist_ok=True)
    N.STATE_PATH = tmp / "nightly-state.json"
    H.DONE_DIR = tmp / "done"
    N.write_status = lambda *a, **k: None
    N.STATE_PATH.write_text(json.dumps({"chats": {}, "files": {}}))


def test_dry_run_moves_nothing():
    tmp = Path(tempfile.mkdtemp(prefix="nightly-marker-"))
    try:
        p = _make_day(tmp)
        _install(tmp)
        _marker(tmp, p, hashlib.sha256(p.read_bytes()).hexdigest())
        N.M = types.SimpleNamespace(
            memory_add=lambda c: (_ for _ in ()).throw(
                AssertionError("extraction attempted in dry run")))
        rc = N.main.__wrapped__() if hasattr(N.main, "__wrapped__") else None
        # argparse would eat sys.argv; call the partition logic via main with
        # argv pinned through the patch below instead.
        import sys
        old = sys.argv
        sys.argv = ["mem0_nightly.py", "--dry-run"]
        try:
            rc = N.main()
        finally:
            sys.argv = old
        assert rc == 0
        assert p.exists(), "dry run must not move the day file"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_live_matching_marker_moves_without_extraction():
    tmp = Path(tempfile.mkdtemp(prefix="nightly-marker-"))
    try:
        p = _make_day(tmp)
        _install(tmp)
        _marker(tmp, p, hashlib.sha256(p.read_bytes()).hexdigest())
        N.M = types.SimpleNamespace(
            memory_add=lambda c: (_ for _ in ()).throw(
                AssertionError("extraction attempted despite marker")))
        import sys
        old = sys.argv
        sys.argv = ["mem0_nightly.py"]
        try:
            rc = N.main()
        finally:
            sys.argv = old
        assert rc == 0
        assert not p.exists(), "live run must move the marked day file"
        moved = N.LOG_INGESTED_DIR / p.name
        assert moved.exists()
        assert moved.read_text() == BODY
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_live_digest_mismatch_falls_back_to_reextract():
    tmp = Path(tempfile.mkdtemp(prefix="nightly-marker-"))
    try:
        p = _make_day(tmp)
        _install(tmp)
        # Marker claims a digest of DIFFERENT bytes: file grew after rollover.
        _marker(tmp, p, hashlib.sha256(b"stale").hexdigest())
        calls = []
        N.M = types.SimpleNamespace(
            memory_add=lambda c: (calls.append(c), "no new facts\n")[1])
        import sys
        old = sys.argv
        sys.argv = ["mem0_nightly.py"]
        try:
            rc = N.main()
        finally:
            sys.argv = old
        assert rc == 0
        assert len(calls) == 1, "mismatch must fall back to re-extraction"
        # Committed: file moved to ingested/ after extraction.
        assert (N.LOG_INGESTED_DIR / p.name).exists()
        assert not p.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_dry_run_moves_nothing()
    print("PASS dry-run: nothing moved")
    test_live_matching_marker_moves_without_extraction()
    print("PASS live+marker: moved before extraction")
    test_live_digest_mismatch_falls_back_to_reextract()
    print("PASS live+mismatch: re-extracted then moved")
    print("ALL PASS")
