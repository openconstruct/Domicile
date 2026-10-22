"""Backup setup checks shared by the future TUI and web wizard screens.

Checks never create a repository or start a backup. The GitHub probe uses
read-only requests; backup.py remains responsible for creating a missing
private repository when the first backup runs.
"""
from __future__ import annotations

import json
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

import config_loader as CL
import credentials


class BackupSetupError(ValueError):
    """A setup problem safe to display without exposing the token."""


class GitHubInfo(NamedTuple):
    owner: str
    repo: str
    exists: bool
    private: bool
    detail: str


def repo_name(raw: str) -> str:
    name = (raw or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", name) or name in {".", ".."}:
        raise BackupSetupError(
            "Use a repository name of 1–100 letters, digits, dots, underscores or hyphens.")
    return name


def clean_token(raw: str) -> str:
    token = (raw or "").strip()
    if not token or any(c.isspace() for c in token):
        raise BackupSetupError("Paste a GitHub personal access token without spaces.")
    return token


def check_local(install_dir: Path | None = None) -> list[tuple[str, bool, str]]:
    """Offline prerequisites; no secret reads, network calls, or writes."""
    install_dir = Path(install_dir) if install_dir is not None else Path(__file__).resolve().parent
    return [
        ("git installed", bool(shutil.which("git")),
         "found" if shutil.which("git") else "Install git before enabling backup."),
        ("backup.py present", (install_dir / "backup.py").is_file(),
         str(install_dir / "backup.py")),
    ]


def check_github(token: str, repo: str, *, opener=None) -> GitHubInfo:
    """Verify a token and read a repo's privacy state. Never creates the repo."""
    token, repo = clean_token(token), repo_name(repo)
    opener = opener or urllib.request.urlopen
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "Domicile-backup-setup",
    }

    def get(url: str) -> dict:
        request = urllib.request.Request(url, headers=headers, method="GET")
        with opener(request, timeout=15) as response:
            return json.load(response)

    try:
        user = get("https://api.github.com/user")
        owner = user.get("login")
        if not isinstance(owner, str) or not owner:
            raise BackupSetupError("GitHub did not return an account name for this token.")
        try:
            target = get(f"https://api.github.com/repos/{owner}/{repo}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return GitHubInfo(
                    owner, repo, False, False,
                    "Repository not visible to this token. Create it as PRIVATE if "
                    "the token cannot create repositories; the first backup will try.")
            raise
        if target.get("private") is not True:
            raise BackupSetupError(
                f"{owner}/{repo} is public. Backups contain private data; choose a private repo.")
        return GitHubInfo(owner, repo, True, True, "Private repository found.")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise BackupSetupError(
                "GitHub rejected this token or denied access. Check its permissions.") from None
        raise BackupSetupError(f"GitHub returned HTTP {exc.code}. Try again later.") from None
    except (urllib.error.URLError, TimeoutError):
        raise BackupSetupError("Could not reach GitHub. Check the connection and retry.") from None
    except (ValueError, KeyError) as exc:
        if isinstance(exc, BackupSetupError):
            raise
        raise BackupSetupError("GitHub returned an unexpected response.") from None


def save(token: str, repo: str, *, config_path: Path | None = None) -> None:
    """Persist the chosen repo in config.json and token only in private .env.

    The UI should call check_github first; save itself is offline so retries
    cannot accidentally create a repo or make an API call.
    """
    token, repo = clean_token(token), repo_name(repo)
    # Refuse a corrupt config before touching the credential file.
    path = Path(config_path) if config_path is not None else CL.CONFIG_PATH
    if path.exists() and path.stat().st_size:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("top-level value is not an object")
        except (OSError, UnicodeError, ValueError) as exc:
            raise BackupSetupError(f"Cannot save backup settings: invalid config.json ({exc}).") from None
    credentials.save({"GITHUB_TOKEN": token})
    CL.save_config({"backup": {"repo": repo}}, path=path)
