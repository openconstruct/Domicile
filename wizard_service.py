"""Wizard step 3.6: write the cchat systemd user units (service + watchdog timer).

cchat lives in a private tmux server so crashes are
visible to systemd, and a 5-minute watchdog timer recovers a wedged unit.
render()/install() only write files (create-only); enable() is the one function
that touches systemctl, so tests never need it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import paths

UNIT_DIR = Path.home() / ".config/systemd/user"


class ServiceError(ValueError):
    """A problem safe to show in the wizard."""


def render(install_dir: Path, python: str | None = None, tmux: str | None = None,
           gmail: bool = False, rss: bool = False) -> dict[str, str]:
    install_dir = Path(install_dir).resolve()
    python = python or sys.executable or "/usr/bin/python3"
    tmux = tmux or shutil.which("tmux")
    if not tmux:
        raise ServiceError("tmux is not installed. Install it (apt install tmux) and retry.")
    for need in ("cchat.py", "watchdog.py"):
        if not (install_dir / need).is_file():
            raise ServiceError(f"{install_dir / need} not found; is this the cchat folder?")
    # An explicit data directory must reach the services too, or they would run
    # against a different (empty) one than the wizard just configured.
    data_env = f"Environment=DOMICILE_DATA={paths.CCHAT_DATA}\n" if os.environ.get("DOMICILE_DATA") else ""
    units = {
        "cchat.service": f"""[Unit]
Description=cchat TUI, always-on inside a tmux socket
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
Type=forking
WorkingDirectory={install_dir}
Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYTHONUNBUFFERED=1
Environment=TERM=xterm-256color
{data_env}ExecStart={tmux} -L cchat new-session -d -s main {python} {install_dir}/cchat.py
ExecStartPost=-{tmux} -L cchat set-option -g exit-empty on
ExecStop={tmux} -L cchat kill-server
Restart=always
RestartSec=5
RestartPreventExitStatus=3

[Install]
WantedBy=default.target
""",
        "cchat-watchdog.service": f"""[Unit]
Description=cchat liveness watchdog (heartbeat + tmux session check)
After=cchat.service

[Service]
Type=oneshot
WorkingDirectory={install_dir}
{data_env}ExecStart={python} {install_dir}/watchdog.py
""",
        "cchat-watchdog.timer": """[Unit]
Description=Run cchat watchdog every 5 minutes

[Timer]
OnBootSec=3min
OnUnitActiveSec=5min
AccuracySec=30s

[Install]
WantedBy=timers.target
""",
    }
    if gmail:
        if not (install_dir / "google_auth" / "gmail_notify.py").is_file():
            raise ServiceError(f"{install_dir / 'google_auth' / 'gmail_notify.py'} not found.")
        units.update(gmail_units(install_dir, python, data_env))
    if rss:
        if not (install_dir / "rss" / "rss_notify.py").is_file():
            raise ServiceError(f"{install_dir / 'rss' / 'rss_notify.py'} not found.")
        units.update(rss_units(install_dir, python, data_env))
    return units


def gmail_units(install_dir: Path, python: str, data_env: str = "") -> dict[str, str]:
    """Poll the agent's inbox every 5 minutes; new mail reaches the agent as a background task.
    gmail_notify exits 3 for 'nothing new', which is the normal case, not a failure."""
    return {
        "domicile-gmail.service": f"""[Unit]
Description=Check the agent's Gmail for unread mail
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory={install_dir}
{data_env}ExecStart={python} {install_dir}/google_auth/gmail_notify.py
SuccessExitStatus=3
""",
        "domicile-gmail.timer": """[Unit]
Description=Poll the agent's Gmail every 5 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
RandomizedDelaySec=30s
Persistent=true

[Install]
WantedBy=timers.target
""",
    }


def rss_units(install_dir: Path, python: str, data_env: str = "") -> dict[str, str]:
    """Check the feeds in data/rss/feeds.txt once an hour; new items reach the agent."""
    return {
        "domicile-rss.service": f"""[Unit]
Description=Check RSS feeds for new items
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory={install_dir}
{data_env}ExecStart={python} {install_dir}/rss/rss_notify.py
""",
        "domicile-rss.timer": """[Unit]
Description=Check RSS feeds hourly

[Timer]
OnBootSec=5min
OnUnitActiveSec=1h
RandomizedDelaySec=2min
Persistent=true

[Install]
WantedBy=timers.target
""",
    }


def install(install_dir: Path, unit_dir: Path | None = None, **kw) -> list[Path]:
    """Create the unit files; refuse if any already exists (never replaces)."""
    unit_dir = Path(unit_dir) if unit_dir else UNIT_DIR
    units = render(install_dir, **kw)
    clash = [n for n in units if (unit_dir / n).exists()]
    if clash:
        raise ServiceError(f"Already exists in {unit_dir}: {', '.join(clash)}. Refusing to replace.")
    made: list[Path] = []
    try:
        unit_dir.mkdir(parents=True, exist_ok=True)
        for name, text in units.items():
            with (unit_dir / name).open("x", encoding="utf-8") as fh:
                made.append(unit_dir / name)
                fh.write(text)
    except OSError as exc:
        for p in reversed(made):
            p.unlink(missing_ok=True)
        raise ServiceError(f"Could not write units: {exc}") from exc
    return made


def enable(start: bool = True, gmail: bool = False, rss: bool = False) -> str:
    """daemon-reload, enable (and optionally start) the units. Returns a status line."""
    def sc(*args):
        r = subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, timeout=60)
        if r.returncode:
            raise ServiceError(f"systemctl {' '.join(args)}: {(r.stderr or r.stdout).strip()}")
    sc("daemon-reload")
    names = ["cchat-watchdog.timer", "cchat.service"] + (["domicile-gmail.timer"] if gmail else []) \
        + (["domicile-rss.timer"] if rss else [])
    sc("enable", *names)
    if start:
        sc("start", *names)
    return "Enabled and started." if start else "Enabled."
