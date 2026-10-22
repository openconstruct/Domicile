#!/usr/bin/env python3
"""Export Domicile's real model-facing tool schemas into bedroom/domicile_tools.json.

The policy must be shown the same tool definitions Domicile shows a native local
model (local_tools.schemas(enabled, trusted=False)). Run this whenever
local_tools._SCHEMAS changes; tests/conformance_domicile.py fails if the
vendored file is stale.

  python3 -m bedroom.sync_domicile_schemas [--domicile ~/domicile]
"""
import argparse, json, sys
from pathlib import Path

NAMES = ["Read", "Write", "Edit", "Glob", "Grep", "Bash", "WebSearch", "WebFetch"]
OUT = Path(__file__).resolve().parent / "domicile_tools.json"


def live(domicile):
    sys.path.insert(0, str(domicile))
    import local_tools
    return local_tools.schemas(NAMES, trusted=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domicile", default=str(Path.home() / "domicile"))
    a = ap.parse_args()
    OUT.write_text(json.dumps(live(Path(a.domicile)), indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT} ({len(NAMES)} tools)")
