#!/usr/bin/env python3
"""Tiny launcher for ToolWake jobs: run a command, record its exit code.

    python3 bgrun.py EXITFILE ARGV...

Why it exists: whoever spawns a job may not be the process that later wants to
know how it ended. On the Codex path the tools server spawns the job, but
cchat does the watching, and cchat can't wait() on a process that isn't its
child. So the code is written to EXITFILE (atomically) when the job ends, and
the watcher reads that instead of guessing. A job that is killed outright
leaves no file, and the watcher falls back to "code unavailable".

stdout/stderr/cwd/env are inherited, so the job's log is unchanged.
"""
import os
import signal
import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: bgrun.py EXITFILE ARGV...", file=sys.stderr)
        return 2
    exitfile, argv = sys.argv[1], sys.argv[2:]
    try:
        proc = subprocess.Popen(argv)
    except FileNotFoundError:
        print(f"command not found: {argv[0]}", file=sys.stderr)
        code = 127
    except Exception as exc:
        print(f"could not start {argv[0]}: {exc}", file=sys.stderr)
        code = 126
    else:
        def forward(signum, _frame):
            try:
                proc.send_signal(signum)
            except Exception:
                pass
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, forward)
        code = proc.wait()          # negative = killed by that signal
    tmp = exitfile + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(str(code))
    os.replace(tmp, exitfile)
    return code if code >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
