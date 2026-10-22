#!/usr/bin/env python3
"""Detached ToolWake runner: record a job's exit code across provider processes."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    exit_path = Path(sys.argv[1])
    argv = sys.argv[2:]
    try:
        code = subprocess.run(argv, check=False).returncode
    except OSError as exc:
        print(f"ToolWake launch failed: {exc}", file=sys.stderr, flush=True)
        code = 127

    # The watcher must never mistake a partially written status for completion.
    tmp = exit_path.with_name(f"{exit_path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(f"{code}\n")
        os.replace(tmp, exit_path)
    except OSError as exc:
        print(f"ToolWake could not record exit status: {exc}", file=sys.stderr, flush=True)
        return 125

    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
