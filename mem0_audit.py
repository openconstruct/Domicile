#!/usr/bin/env python3
"""Domicile mem0 audit."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mem0_core  # noqa: E402  (sets telemetry/env before importing mem0)


RULES: list[tuple[str, str, re.Pattern]] = [
    (
        "breakage",
        "asserts something is broken; fixes rarely get written down",
        re.compile(
            r"\b(has a (critical |known )?bug|is broken|does not work|doesn't work"
            r"|fails|failing|blocked|blocking issue|known issue|silently ignores"
            r"|is missing|deadlock|regression|workaround)\b",
            re.I,
        ),
    ),
    (
        "count",
        "quotes a number that drifts (store size, disk, memory totals)",
        re.compile(
            r"\b\d[\d,.]*\s*(memories|vectors|entries|facts|chunks|files|stars"
            r"|packages|%|percent|GB|MB|KB|TB)\b",
            re.I,
        ),
    ),
    (
        "intent",
        "records a plan or a to-do as if it were a fact",
        re.compile(
            r"\b(wants to|plans to|intends to|is deciding|is going to|will be"
            r"|should be (deleted|removed|fixed|added)|needs to be|todo|to-do"
            r"|pending|unresolved|not yet|hasn't|has not yet|next step)\b",
            re.I,
        ),
    ),
    (
        "now",
        "anchored to an unstated present; ages badly without a date",
        re.compile(
            r"\b(currently|right now|at the moment|these days|nowadays|now (has|uses|runs)"
            r"|still (has|uses|runs|exists)|recently|latest|newest|so far)\b",
            re.I,
        ),
    ),
    (
        "version",
        "pins a version or path that upgrades out from under it",
        re.compile(
            r"\bv?\d+\.\d+(\.\d+)?\b|~/\.[\w.-]*venv|\bvenv\b",
            re.I,
        ),
    ),
]

DATED = re.compile(r"\b(as of|on)\s+20\d\d-\d\d-\d\d\b", re.I)


def _created(row: dict) -> datetime | None:
    raw = row.get("created_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _age_days(row: dict, now: datetime) -> float:
    made = _created(row)
    if made is None:
        return float("inf")  # unknown age: treat as old, i.e. worth checking
    if made.tzinfo is None:
        made = made.replace(tzinfo=timezone.utc)
    return (now - made).total_seconds() / 86400


def audit(min_age_days: float, only: str | None) -> list[dict]:
    store = mem0_core.store()
    rows = mem0_core.rows(
        store.get_all(
            filters={"user_id": mem0_core.USER_ID},
            top_k=mem0_core.ALL_ROWS,
        )
    )
    now = datetime.now(timezone.utc)

    suspects = []
    for row in rows:
        text = row.get("memory") or ""
        if DATED.search(text):
            continue
        age = _age_days(row, now)
        if age < min_age_days:
            continue
        hits = [name for name, _, pat in RULES if pat.search(text)]
        if only:
            hits = [h for h in hits if h == only]
        if hits:
            suspects.append(
                {
                    "id": row.get("id", ""),
                    "age": age,
                    "categories": hits,
                    "text": text,
                }
            )
    suspects.sort(key=lambda s: (-len(s["categories"]), -s["age"]))
    return suspects


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--age", type=float, default=14,
                    help="only flag memories at least this many days old (default 14)")
    ap.add_argument("--category", choices=[r[0] for r in RULES],
                    help="restrict to one rule")
    ap.add_argument("--ids", action="store_true", help="print bare ids only")
    args = ap.parse_args()

    try:
        suspects = audit(args.age, args.category)
    except mem0_core.StoreBusy as exc:
        print(f"memory store is busy: {exc}", file=sys.stderr)
        return 1

    if args.ids:
        for s in suspects:
            print(s["id"])
        return 0

    if not suspects:
        print("no stale-looking memories. Either the store is clean or the "
              "rules need widening.")
        return 0

    print(f"{len(suspects)} suspect memories (>= {args.age:g}d old, undated)\n")
    for s in suspects:
        tags = ",".join(s["categories"])
        age = "?" if s["age"] == float("inf") else f"{s['age']:.0f}d"
        print(f"({s['id'][:8]}) [{tags}] {age}")
        print(f"    {s['text']}\n")

    print("Check each one against reality. If it is still true, rewrite it "
          "with an explicit 'as of YYYY-MM-DD' and this will stop flagging "
          "it. If it is not, delete it and store the correction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
