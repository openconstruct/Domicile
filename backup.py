#!/usr/bin/env python3
"""Domicile backup and restore."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from paths import CONFIG_PATH, MEM0_DIR, MEM0_LOG_DIR, CHAT_DIR, MEM1_DIR, \
    BACKUP_STAGE, BACKUP_LOG, BACKUP_STATE, PRE_APPLY_DIR

HOME = Path.home()


def _cfg(key: str, default):
    """Read one value from the `backup` block of data/config.json."""
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f).get("backup", {}).get(key, default)
    except (OSError, ValueError):
        return default


STAGE = Path(os.environ.get("CCHAT_BACKUP_STAGE", BACKUP_STAGE))
LOG = BACKUP_LOG
STATE = BACKUP_STATE
import credentials

# env var wins, then config.json backup.repo (what the wizard writes), then the old default
REPO_NAME = os.environ.get("CCHAT_BACKUP_REPO") or _cfg("repo", "domicile-backup")

KEEP_COMMITS = int(os.environ.get("CCHAT_BACKUP_KEEP") or _cfg("keep", 2))
PRUNE_SLACK = int(os.environ.get("CCHAT_BACKUP_PRUNE_SLACK") or _cfg("prune_slack", 4))

INTERVAL_SECONDS = int(os.environ.get("CCHAT_BACKUP_INTERVAL", 24 * 3600))

BACKUP_HOUR = os.environ.get("CCHAT_BACKUP_HOUR", "")

SQLITE_SOURCES = [
    (MEM0_DIR / "history.db", "mem0/history.db.gz"),
    (MEM0_DIR / "qdrant" / "collection" / "ccchat" / "storage.sqlite",
     "mem0/qdrant-ccchat-storage.sqlite.gz"),
    (MEM0_DIR / "qdrant-mem2" / "collection" / "cchat-mem2" / "storage.sqlite",
     "mem0/qdrant-mem2-cchat-mem2-storage.sqlite.gz"),
]

PLAIN_DIRS = [
    (MEM1_DIR, "memory", "*.md"),
    (MEM0_LOG_DIR, "mem0/logs", "**/*.log"),
]

PLAIN_FILES = [
    (MEM0_DIR / "config.json", "mem0/config.json"),
    (MEM0_DIR / "qdrant" / "meta.json", "mem0/qdrant-meta.json"),
    (MEM0_DIR / "nightly-state.json", "mem0/nightly-state.json"),
    (MEM0_DIR / "qdrant-mem2" / "meta.json", "mem0/qdrant-mem2-meta.json"),
    (MEM0_DIR / "mem2-index-state.json", "mem0/mem2-index-state.json"),
]

TAR_DIRS = [
    (CHAT_DIR, "chats.tar.gz"),
]

SECRET_PATTERNS = [
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"glpat-[A-Za-z0-9_\-]{15,}"),
    re.compile(r"sk-[A-Za-z0-9\-]{20,}"),
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
]


_PLACEHOLDER_RE = re.compile(r"(.)\1{5,}|xxxx|XXXX|1234567|abcdef|your[_-]?token|example|placeholder", re.I)


def _is_placeholder(match: str) -> bool:
    """True when a secret-shaped hit is an obvious dummy (hf_xxxx..., ghp_0000...)."""
    body = match.split("_", 1)[-1].split("-", 1)[-1]
    return bool(_PLACEHOLDER_RE.search(body))


def log(msg: str) -> None:
    line = f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}  {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tokens() -> dict[str, str]:
    token = credentials.get("GITHUB_TOKEN")
    return {"github": token} if token else {}


# snapshotting

def snapshot_sqlite(src: Path, dest: Path) -> None:
    """Consistent copy of a possibly-live SQLite db, then verify it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "snap.sqlite"
        src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=30)
        try:
            dst_conn = sqlite3.connect(raw)
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()
        finally:
            src_conn.close()

        check = sqlite3.connect(f"file:{raw}?mode=ro", uri=True)
        try:
            result = check.execute("pragma integrity_check").fetchone()[0]
        finally:
            check.close()
        if result != "ok":
            raise RuntimeError(f"{src.name}: integrity_check said {result!r}")

        with raw.open("rb") as fi, gzip.open(dest, "wb", compresslevel=9) as fo:
            shutil.copyfileobj(fi, fo)


def scan_for_secrets(root: Path) -> list[str]:
    hits = []
    for p in root.rglob("*"):
        if not p.is_file() or ".git" in p.parts:
            continue
        if p.suffix in {".gz", ".sqlite", ".db"}:
            continue  # binary; covered by never copying credential files in
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for pat in SECRET_PATTERNS:
            for m in pat.findall(text):
                if _is_placeholder(m):
                    continue
                hits.append(f"{p.relative_to(root)} matches {pat.pattern[:24]}")
                break
    return hits


def build_payload() -> dict:
    """Populate the staging tree and return the manifest."""
    for sub in ("memory", "mem0", "mem0/logs"):
        (STAGE / sub).mkdir(parents=True, exist_ok=True)

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": os.uname().nodename,
        "files": {},
    }

    for src_dir, dest_sub, pattern in PLAIN_DIRS:
        dest_dir = STAGE / dest_sub
        for old in dest_dir.glob(pattern):
            old.unlink()
        if not src_dir.exists():
            continue
        for f in sorted(src_dir.glob(pattern)):
            target = dest_dir / f.relative_to(src_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)

    for src, dest_rel in PLAIN_FILES:
        if src.exists():
            dest = STAGE / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    for src, dest_rel in SQLITE_SOURCES:
        if not src.exists():
            log(f"WARN missing sqlite source {src}")
            continue
        snapshot_sqlite(src, STAGE / dest_rel)

    for src_dir, dest_rel in TAR_DIRS:
        if not src_dir.exists():
            continue
        dest = STAGE / dest_rel
        with tarfile.open(dest, "w:gz", compresslevel=9) as tar:
            for f in sorted(src_dir.rglob("*")):
                if f.is_file():
                    info = tar.gettarinfo(f, arcname=str(f.relative_to(src_dir)))
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with f.open("rb") as fh:
                        tar.addfile(info, fh)

    for p in sorted(STAGE.rglob("*")):
        if p.is_file() and ".git" not in p.parts and p.name != "MANIFEST.json":
            manifest["files"][str(p.relative_to(STAGE))] = {
                "sha256": sha256(p), "bytes": p.stat().st_size,
            }

    manifest["total_bytes"] = sum(v["bytes"] for v in manifest["files"].values())
    (STAGE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (STAGE / "RESTORE.md").write_text(RESTORE_DOC)
    return manifest


RESTORE_DOC = """# Restoring Domicile

A backup that has never been restored is a hypothesis. Test it with the
scratch restore, which touches nothing live:

    python3 backup.py --restore /tmp/restore-test

To actually recover onto a fresh machine:

    git clone <either remote> <install-dir>/data/backup
    python3 <install-dir>/data/backup/../../backup.py --restore ~/restored

then put the pieces where they live. This is automated:

    python3 backup.py --apply          # dry run: prints exactly what it would do
    python3 backup.py --apply --yes    # actually place the files

`--apply` verifies against the manifest, expands into a scratch dir, checks
every restored sqlite with `pragma integrity_check`, then copies each file to
its real path. Anything it would overwrite is moved first into
`<install-dir>/data/pre-apply/<timestamp>/`, so a botched restore is reversible. It
refuses to run while Domicile or the mem0/mem2 servers are alive (torn-database
risk); stop them first, or pass `--force`.

The placement map, for reference / manual recovery:

| in the archive                                | belongs at                                            |
|-----------------------------------------------|-------------------------------------------------------|
| `memory/*.md`                                 | `<install-dir>/data/memory/`                             |
| `chats.tar.gz`                                | untar into `<install-dir>/data/chats/`                      |
| `mem0/history.db.gz`                          | gunzip to `<install-dir>/data/mem0/history.db`              |
| `mem0/qdrant-ccchat-storage.sqlite.gz`        | gunzip to `<install-dir>/data/mem0/qdrant/collection/ccchat/storage.sqlite` |
| `mem0/qdrant-mem2-cchat-mem2-storage.sqlite.gz` | gunzip to `<install-dir>/data/mem0/qdrant-mem2/collection/cchat-mem2/storage.sqlite` |
| `mem0/logs/**/*.log`                          | `<install-dir>/data/mem0/logs/` (preserving the `ingested/` subdir) |
| `mem0/config.json`                            | `<install-dir>/data/mem0/config.json`                       |
| `mem0/qdrant-meta.json`                       | `<install-dir>/data/mem0/qdrant/meta.json`                  |
| `mem0/qdrant-mem2-meta.json`                  | `<install-dir>/data/mem0/qdrant-mem2/meta.json`             |
| `mem0/nightly-state.json`                     | `<install-dir>/data/mem0/nightly-state.json`                |
| `mem0/mem2-index-state.json`                  | `<install-dir>/data/mem0/mem2-index-state.json`             |

`MANIFEST.json` holds a sha256 for every file. `--restore` verifies all of
them and refuses to proceed on a mismatch, so a silently-corrupted archive
fails at restore time rather than being trusted.

Only `memory/*.md` is truly irreplaceable. `mem0/` is derived from `chats/`
and can be rebuilt by re-running nightly ingestion, at the cost of the model
calls. The daily logs (`mem0/logs/*.log`) are the source for mem2's semantic
search index.
"""


# git+push

def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(("git", "-C", str(STAGE)) + args,
                          capture_output=True, text=True, check=check)


def ensure_repo() -> None:
    if not (STAGE / ".git").exists():
        STAGE.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", "-b", "main", str(STAGE)], check=True)
        git("config", "user.name", "cchat-backup")
        git("config", "user.email", "backup@localhost")
        log(f"initialised staging repo at {STAGE}")


def github_ensure_remote(token: str) -> str | None:
    api = "https://api.github.com"
    hdr = {"Authorization": f"Bearer {token}",
           "Accept": "application/vnd.github+json",
           "User-Agent": "cchat-backup"}

    def call(method, url, body=None):
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, headers=hdr, method=method)
        if data:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30) as f:
            return json.load(f)

    user = call("GET", f"{api}/user")["login"]
    try:
        repo = call("GET", f"{api}/repos/{user}/{REPO_NAME}")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        try:
            repo = call("POST", f"{api}/user/repos", {
                "name": REPO_NAME, "private": True,
                "description": "Domicile memory + mem0 snapshots (private)",
            })
            log(f"github: created private repo {user}/{REPO_NAME}")
        except urllib.error.HTTPError as ce:
            if ce.code != 403:
                raise
            log(f"github: {user}/{REPO_NAME} does not exist and this token "
                f"cannot create repos. Create it PRIVATE at "
                f"https://github.com/new (name: {REPO_NAME}), then re-run.")
            return None

    if not repo.get("private", False):
        log(f"REFUSING github: {user}/{REPO_NAME} is PUBLIC. "
            f"These files contain personal health and financial data.")
        return None
    return f"https://x-access-token:{token}@github.com/{user}/{REPO_NAME}.git"


def collect_targets(tokens: dict[str, str]) -> list[tuple[str, str]]:
    """(name, push-url) for every provider we have a usable credential for."""
    targets: list[tuple[str, str]] = []

    if "github" in tokens:
        try:
            url = github_ensure_remote(tokens["github"])
            if url:
                targets.append(("github", url))
        except Exception as e:
            log(f"github: setup failed: {e}")
    return targets


def push_all(tokens: dict[str, str]) -> tuple[int, int]:
    targets = collect_targets(tokens)

    ok = 0
    for name, url in targets:
        try:
            git("remote", "remove", name, check=False)
            git("remote", "add", name, url)
            r = git("push", "-q", name, "main", check=False)
            if r.returncode == 0:
                log(f"pushed to {name}")
                ok += 1
            else:
                err = (r.stderr or "").strip()
                for t in tokens.values():
                    err = err.replace(t, "<TOKEN>")
                lines = [l.strip() for l in err.splitlines() if l.strip()]
                detail = next((l for l in lines if l.startswith("remote:")),
                              lines[-1] if lines else "?")
                log(f"push to {name} FAILED: {detail}")
        except Exception as e:
            log(f"push to {name} errored: {type(e).__name__}")
        finally:
            git("remote", "remove", name, check=False)
    return ok, len(targets)


# restore

def restore(dest: Path, force: bool = False) -> int:
    """Verify the archive against its manifest and expand it into `dest`."""
    manifest_path = STAGE / "MANIFEST.json"
    if not manifest_path.exists():
        log(f"no MANIFEST.json in {STAGE}; nothing to restore")
        return 1
    manifest = json.loads(manifest_path.read_text())

    bad = []
    for rel, meta in manifest["files"].items():
        p = STAGE / rel
        if not p.exists():
            bad.append(f"missing {rel}")
        elif sha256(p) != meta["sha256"]:
            bad.append(f"checksum mismatch {rel}")
    if bad:
        for b in bad:
            log(f"VERIFY FAIL {b}")
        return 1
    log(f"verified {len(manifest['files'])} files against manifest")

    if dest.exists() and any(dest.iterdir()) and not force:
        log(f"refusing to restore into non-empty {dest} (use --force)")
        return 1
    dest.mkdir(parents=True, exist_ok=True)

    for rel in manifest["files"]:
        src, out = STAGE / rel, dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".sqlite.gz") or rel.endswith(".db.gz"):
            out = dest / rel[:-3]
            with gzip.open(src, "rb") as fi, out.open("wb") as fo:
                shutil.copyfileobj(fi, fo)
            check = sqlite3.connect(f"file:{out}?mode=ro", uri=True)
            try:
                res = check.execute("pragma integrity_check").fetchone()[0]
            finally:
                check.close()
            if res != "ok":
                log(f"RESTORED DB CORRUPT {rel}: {res}")
                return 1
            log(f"restored+verified {out.name}")
        elif rel.endswith(".tar.gz"):
            outdir = dest / rel[:-7]
            outdir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(src, "r:gz") as tar:
                for m in tar.getmembers():
                    if m.name.startswith("/") or ".." in Path(m.name).parts:
                        log(f"skipping unsafe tar path {m.name}")
                        continue
                    tar.extract(m, outdir)
            log(f"restored {rel} -> {outdir.name}/")
        else:
            shutil.copy2(src, out)

    log(f"restore complete -> {dest}")
    return 0


# apply into place

def _placement() -> tuple[dict[str, Path], list[tuple[str, Path]]]:
    """Reverse the backup source tables into 'archive path -> live path'."""
    exact: dict[str, Path] = {}
    for src, dest_rel in SQLITE_SOURCES:
        exact[dest_rel[:-3] if dest_rel.endswith(".gz") else dest_rel] = src
    for src, dest_rel in PLAIN_FILES:
        exact[dest_rel] = src

    prefixes: list[tuple[str, Path]] = []
    for src_dir, dest_sub, _pat in PLAIN_DIRS:
        prefixes.append((dest_sub.rstrip("/") + "/", src_dir))
    for src_dir, dest_rel in TAR_DIRS:
        stem = dest_rel[:-7] if dest_rel.endswith(".tar.gz") else dest_rel
        prefixes.append((stem.rstrip("/") + "/", src_dir))
    return exact, prefixes


def _live_path_for(rel: str, exact: dict, prefixes: list) -> Path | None:
    if rel in exact:
        return exact[rel]
    for pref, base in prefixes:
        if rel.startswith(pref):
            return base / rel[len(pref):]
    return None


def _live_processes() -> list[str]:
    found = []
    for pat in ("cchat.py", "mem0_server.py", "mem2_server.py",
                "mem0_nightly.py", "mem0_ipc.py"):
        r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        if r.returncode == 0:
            found.append(pat)
    return found


def apply_to_home(force: bool = False, yes: bool = False) -> int:
    """Verify the archive, then place every file at its real home path."""
    running = _live_processes()
    if running and not force:
        log(f"REFUSING --apply: live process(es) holding the stores: "
            f"{', '.join(running)}. Stop Domicile and the mem0/mem2 servers "
            f"first, or pass --force if you know they are safe.")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="cchat-apply-"))
    try:
        if restore(tmp, force=True) != 0:
            log("apply aborted: staged restore failed verification")
            return 1

        exact, prefixes = _placement()
        plan: list[tuple[Path, Path]] = []
        for p in sorted(tmp.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(tmp))
            if rel in ("RESTORE.md", "MANIFEST.json"):
                continue  # archive metadata, not live state
            live = _live_path_for(rel, exact, prefixes)
            if live is None:
                log(f"WARN no live destination known for {rel}; skipping")
                continue
            plan.append((p, live))

        stash = (PRE_APPLY_DIR
                 / datetime.now().strftime("%Y%m%d-%H%M%S"))
        log(f"apply plan: {len(plan)} files"
            + ("" if yes else "  (DRY RUN -- pass --yes to perform)"))
        placed = stashed = 0
        for src, live in plan:
            exists = live.exists()
            same = exists and sha256(src) == sha256(live)
            tag = "unchanged" if same else ("overwrite" if exists else "new")
            log(f"  {tag:10} {live}")
            if not yes or same:
                continue
            if exists:
                try:
                    rel_live = live.relative_to(HOME)
                except ValueError:                  # installed outside $HOME, e.g. /opt/domicile
                    rel_live = Path(*live.parts[1:])
                keep = stash / rel_live
                keep.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(live), str(keep))
                stashed += 1
            live.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, live)
            placed += 1

        if not yes:
            log("nothing written. Re-run with --yes to apply.")
            return 0
        log(f"apply complete: {placed} files placed, {stashed} stashed under {stash}")
        try:
            import sync_guard
            head = subprocess.run(["git", "-C", str(STAGE), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
            if head:
                sync_guard.record(head, "apply-to-home")
                log(f"sync-state advanced to {head[:10]} (this box is now current)")
        except Exception as e:
            log(f"sync-state not updated after apply: {type(e).__name__}: {e}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# main

def due() -> bool:
    try:
        last = json.loads(STATE.read_text()).get("last_success", 0)
    except (OSError, ValueError):
        return True
    now = time.time()
    if BACKUP_HOUR.strip():
        try:
            target = int(BACKUP_HOUR)
        except ValueError:
            return (now - last) >= INTERVAL_SECONDS
        last_lt = time.localtime(last)
        now_lt = time.localtime(now)
        same_day = (last_lt.tm_year, last_lt.tm_yday) == (now_lt.tm_year, now_lt.tm_yday)
        if same_day and last_lt.tm_hour >= target:
            return False
        return now_lt.tm_hour >= target or (now - last) >= 30 * 3600
    return (now - last) >= INTERVAL_SECONDS


def run() -> int:
    ensure_repo()
    manifest = build_payload()

    leaks = scan_for_secrets(STAGE)
    if leaks:
        for l in leaks:
            log(f"ABORT credential-shaped string in payload: {l}")
        return 1

    git("add", "-A")
    if not git("status", "--porcelain").stdout.strip():
        log("no changes since last snapshot; not committing")
    else:
        kb = manifest["total_bytes"] // 1024
        git("commit", "-q", "-m",
            f"snapshot {manifest['created']} "
            f"({len(manifest['files'])} files, {kb}KB)")
        log(f"committed {len(manifest['files'])} files, {kb}KB")

    tokens = read_tokens()
    ok, total = push_all(tokens) if tokens else (0, 0)

    if total == 0:
        log("snapshot committed locally; no remotes configured")
    else:
        log(f"pushed to {ok}/{total} remotes")

    if ok > 0 or total == 0:
        STATE.write_text(json.dumps({
            "last_success": time.time(), "pushed": ok, "targets": total,
        }))
        if ok > 0:
            try:
                import sync_guard
                sync_guard.record(git("rev-parse", "HEAD").stdout.strip(),
                                  "local push")
            except Exception as e:
                log(f"sync-state not updated: {type(e).__name__}: {e}")
        if total and ok < total:
            log(f"WARNING degraded: {total - ok} of {total} remotes failed")
        try:
            n = int(git("rev-list", "--count", "HEAD").stdout.strip() or "0")
            if n > KEEP_COMMITS + PRUNE_SLACK:
                log(f"prune: {n} commits > {KEEP_COMMITS}+{PRUNE_SLACK}; trimming")
                prune(KEEP_COMMITS)
        except Exception as e:
            log(f"prune: auto-trim skipped: {type(e).__name__}: {e}")
        return 0
    log("no remote accepted the snapshot; will retry on the next check")
    return 1


def pull() -> int:
    """Fast-forward the staging repo to whatever a remote currently holds."""
    ensure_repo()
    tokens = read_tokens()
    if not tokens:
        log("no tokens; cannot pull")
        return 1
    for name, url in collect_targets(tokens):
        try:
            git("remote", "remove", "pullsrc", check=False)
            git("remote", "add", "pullsrc", url)
            r = git("fetch", "-q", "pullsrc", "main", check=False)
            if r.returncode != 0:
                log(f"pull: fetch from {name} failed, trying next")
                continue
            before = git("rev-parse", "HEAD", check=False).stdout.strip()
            git("reset", "--hard", "-q", "FETCH_HEAD")
            after = git("rev-parse", "HEAD").stdout.strip()
            git("remote", "remove", "pullsrc", check=False)
            if before == after:
                log(f"pull: already at {after[:10]} ({name})")
            else:
                log(f"pull: {before[:10]} -> {after[:10]} from {name}")
            return 0
        except Exception as e:
            log(f"pull from {name} errored: {type(e).__name__}: {e}")
        finally:
            git("remote", "remove", "pullsrc", check=False)
    log("pull: no remote could be fetched")
    return 1


def prune(keep: int = KEEP_COMMITS) -> int:
    """Truncate the staging repo's history to the last `keep` commits."""
    ensure_repo()

    count = int(git("rev-list", "--count", "HEAD").stdout.strip() or "0")
    if count <= keep:
        log(f"prune: {count} commit(s) <= keep={keep}; nothing to do")
        return 0

    old_head = git("rev-parse", "HEAD").stdout.strip()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    root_tree = git("rev-parse", f"HEAD~{keep - 1}^{{tree}}").stdout.strip()
    parent = git("commit-tree", root_tree, "-m",
                 f"baseline: history truncated {stamp} "
                 f"(pre-truncation HEAD {old_head})").stdout.strip()
    for i in range(keep - 2, -1, -1):
        tree = git("rev-parse", f"HEAD~{i}^{{tree}}").stdout.strip()
        msg = git("log", "-1", "--format=%B", f"HEAD~{i}").stdout.strip() or "snapshot"
        parent = git("commit-tree", tree, "-p", parent, "-m", msg).stdout.strip()

    git("reset", "--hard", "-q", parent)
    new_head = git("rev-parse", "HEAD").stdout.strip()
    log(f"prune: rewrote {count} -> {keep} commits "
        f"({old_head[:10]} -> {new_head[:10]})")

    tokens = read_tokens()
    targets = collect_targets(tokens) if tokens else []
    ok = 0
    for name, url in targets:
        try:
            git("remote", "remove", name, check=False)
            git("remote", "add", name, url)
            r = git("push", "--force", "-q", name, "main", check=False)
            if r.returncode == 0:
                log(f"prune: force-pushed to {name}")
                ok += 1
            else:
                err = (r.stderr or "").strip()
                for t in tokens.values():
                    err = err.replace(t, "<TOKEN>")
                lines = [l.strip() for l in err.splitlines() if l.strip()]
                detail = next((l for l in lines if l.startswith("remote:")),
                              lines[-1] if lines else "?")
                log(f"prune: force-push to {name} FAILED: {detail}")
        except Exception as e:
            log(f"prune: push to {name} errored: {type(e).__name__}")
        finally:
            git("remote", "remove", name, check=False)

    if targets and ok == 0:
        git("reset", "--hard", "-q", old_head)
        log(f"prune: no remote accepted the rewrite; "
            f"rolled local back to {old_head[:10]}")
        return 1

    git("reflog", "expire", "--expire=now", "--all", check=False)
    git("gc", "--prune=now", "-q", check=False)
    try:
        import sync_guard
        sync_guard.record(new_head, "prune")
    except Exception as e:
        log(f"prune: sync-state not updated: {type(e).__name__}: {e}")

    sz = subprocess.run(["du", "-sh", str(STAGE / ".git")],
                        capture_output=True, text=True).stdout.split()
    tail = ""
    if targets and ok < len(targets):
        tail = f"  ({len(targets) - ok} remote(s) not updated -- retried next prune)"
    log(f"prune: done; .git now {sz[0] if sz else '?'}{tail}")
    return 0


def main(argv: list[str]) -> int:
    if "--pull" in argv:
        return pull()
    if "--prune" in argv:
        i = argv.index("--prune")
        keep = KEEP_COMMITS
        if len(argv) > i + 1 and argv[i + 1].isdigit():
            keep = int(argv[i + 1])
        return prune(keep)
    if "--apply" in argv:
        return apply_to_home(force="--force" in argv, yes="--yes" in argv)
    if "--restore" in argv:
        i = argv.index("--restore")
        dest = Path(argv[i + 1]) if len(argv) > i + 1 else Path("/tmp/cchat-restore")
        return restore(dest, force="--force" in argv)
    if "--if-due" in argv and not due():
        return 0
    return run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
