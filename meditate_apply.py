#!/usr/bin/env python3
"""Preview or apply memory consolidation proposals.

Defaults to preview; --apply writes the selected proposals.
--store rebuild uses the disposable store; --store live uses memory IPC.
--only 3,7,12 selects proposal indices from the input log.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mem0_ipc
import paths as _P

REBUILD_QDRANT = _P.MEM0_DIR / "qdrant-rebuild"
REBUILD_FTS = _P.MEM0_DIR / "mem0-fts-rebuild.db"



def newest_log() -> Path | None:
    """Most recent meditate_dryrun.py output in data/mem0/, if any."""
    logs = sorted(_P.MEM0_DIR.glob("meditation-*.json"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


class _DirectBackend:

    def __init__(self) -> None:
        import mem0_core as C
        C.QDRANT_PATH = REBUILD_QDRANT
        C.CONFIG["vector_store"]["config"]["path"] = str(REBUILD_QDRANT)
        C.MEM0_FTS_PATH = REBUILD_FTS
        REBUILD_QDRANT.mkdir(parents=True, exist_ok=True)
        self._core = C

    def add_verbatim(self, text: str, metadata: dict | None) -> str:
        return self._core.add_verbatim(text, metadata=metadata)

    def mark_superseded(self, drop_id: str, keep_id: str, reason: str) -> str:
        return self._core.mark_superseded(drop_id, keep_id, reason=reason)


class _IpcBackend:

    def add_verbatim(self, text: str, metadata: dict | None) -> str:
        return mem0_ipc.call("add_verbatim", {"text": text, "metadata": metadata})

    def mark_superseded(self, drop_id: str, keep_id: str, reason: str) -> str:
        return mem0_ipc.call("mark_superseded",
                             {"drop_id": drop_id, "keep_id": keep_id, "reason": reason})


def _backend_for(target: str):
    if target == "rebuild":
        return _DirectBackend()
    if target == "live":
        return _IpcBackend()
    raise ValueError(f"unknown --store {target!r}")


def _short(s: str) -> str:
    return (s or "?")[:8]


def _preview(i: int, p: dict) -> str:
    action = p.get("action")
    conf = p.get("confidence")
    if action == "supersede":
        return (f"[{i}] SUPERSEDE conf={conf} "
                f"keep={_short(p.get('keep_id'))} "
                f"drop={[_short(d) for d in p.get('drop_ids', [])]}\n"
                f"    rationale: {p.get('rationale', '')}")
    if action == "meta":
        return (f"[{i}] META conf={conf} "
                f"members={[_short(m) for m in p.get('member_ids', [])]}\n"
                f"    synthesis: {p.get('synthesis', '')}\n"
                f"    rationale: {p.get('rationale', '')}")
    return f"[{i}] UNKNOWN action {action!r}: {p}"


def _apply_one(backend, p: dict) -> list[str]:
    action = p.get("action")
    reason = p.get("rationale", "")
    out: list[str] = []

    if action == "supersede":
        keep_id = p.get("keep_id")
        drop_ids = p.get("drop_ids") or []
        if not keep_id or not drop_ids:
            raise ValueError(f"malformed supersede proposal: {p}")
        for drop_id in drop_ids:
            out.append(backend.mark_superseded(drop_id, keep_id, reason))

    elif action == "meta":
        member_ids = p.get("member_ids") or []
        synthesis = (p.get("synthesis") or "").strip()
        if not member_ids or not synthesis:
            raise ValueError(f"malformed meta proposal: {p}")
        new_id = backend.add_verbatim(synthesis, {"kind": "meditation-synthesis",
                                                   "source": "meditate_apply",
                                                   "reason": reason})
        out.append(f"added synthesis {new_id}: {synthesis[:100]}")
        for member_id in member_ids:
            out.append(backend.mark_superseded(member_id, new_id, reason))

    else:
        raise ValueError(f"unknown action {action!r} in proposal: {p}")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", default=None,
                     help="meditate_dryrun.py output JSON (default: the newest "
                          "meditation-*.json in data/mem0/)")
    ap.add_argument("--store", choices=("rebuild", "live"), default="rebuild",
                     help="which qdrant store to write into (default: rebuild)")
    ap.add_argument("--apply", action="store_true",
                     help="actually write; without this, preview only")
    ap.add_argument("--only", type=str, default=None,
                     help="comma-separated 1-based proposal numbers "
                          "(default: every proposal in the log)")
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else newest_log()
    if log_path is None:
        ap.error("no meditation-*.json found in data/mem0/; run meditate_dryrun.py first "
                 "or pass a file")
    log = json.loads(log_path.read_text())
    proposals = log.get("proposals", [])
    if not proposals:
        print(f"no proposals in {log_path}")
        return 0

    only = {int(x) for x in args.only.split(",")} if args.only else None
    selected = [(i, p) for i, p in enumerate(proposals, 1)
                if only is None or i in only]

    print(f"{len(selected)}/{len(proposals)} proposal(s) selected from {log_path}")
    print(f"target store: {args.store}"
          + ("" if args.apply else "  (PREVIEW ONLY -- pass --apply to write)"))
    print()
    for i, p in selected:
        print(_preview(i, p))
        print()

    if not args.apply:
        return 0

    backend = _backend_for(args.store)

    print("--- applying ---")
    failures = 0
    for i, p in selected:
        try:
            for line in _apply_one(backend, p):
                print(f"[{i}] {line}")
        except Exception as exc:
            failures += 1
            print(f"[{i}] FAILED: {type(exc).__name__}: {exc}")
    print()
    print(f"done: {len(selected) - failures}/{len(selected)} applied"
          + (f", {failures} failed" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
