#!/usr/bin/env python3
"""Domicile runtime health checks."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import CCHAT_DATA, CONFIG_PATH, HEARTBEAT_FILE, CRASH_LOG, BACKUP_STATE, HOURLY_INGEST_STATUS_PATH, QDRANT_PATH, SOCKET_PATH, SELFTEST_LOG, SELFTEST_STATUS

# Adapters providers/__init__.get_provider() can actually construct. Checked
# instead of a provider-*name* allowlist: dispatch is by adapter, names are
# free-form, and any unknown adapter string falls through to
# "openai_compatible" silently -- so a typo there is the failure worth naming.
KNOWN_ADAPTERS = {
    "claude_cli", "codex_cli", "openai_responses", "openai_compatible",
}

OK, WARN, FAIL = "OK", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str          # OK | WARN | FAIL
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status == FAIL


def _now() -> dt.datetime:
    return dt.datetime.now()


def _age_hours(ts: float) -> float:
    return (time.time() - ts) / 3600.0


# checks


def check_config() -> Check:
    """config.json and roles.json parse and load into a usable shape."""
    try:
        raw = json.loads(CONFIG_PATH.read_text())
    except FileNotFoundError:
        return Check("config", WARN, f"{CONFIG_PATH} absent -- using DEFAULTS")
    except ValueError as exc:
        return Check("config", FAIL, f"config.json is not valid JSON: {exc}")
    try:
        import config_loader as CL
        cfg = CL.load_config()
        roles = CL.load_roles()
    except Exception as exc:
        return Check("config", FAIL,
                     f"load_config() raised {type(exc).__name__}: {exc}")
    if not isinstance(cfg.get("providers"), dict) or not cfg["providers"]:
        return Check("config", FAIL, "no 'providers' block after load")
    order = CL.provider_order(cfg)
    if not order:
        return Check("config", FAIL, "provider_order is empty")
    active = CL.active_provider(cfg)
    if active not in cfg["providers"]:
        return Check("config", FAIL,
                     f"active_provider {active!r} has no provider block")
    if CL.ROLES_PATH.exists():
        try:
            json.loads(CL.ROLES_PATH.read_text())
        except ValueError as exc:
            return Check("config", FAIL, f"roles.json is not valid JSON: {exc}")
    return Check("config", OK,
                 f"{len(raw.get('providers', {}))} provider(s), "
                 f"{len([r for r in roles if not r.startswith('_')])} role(s), "
                 f"active={active}")


def check_role_chains() -> Check:
    try:
        import config_loader as CL
        cfg = CL.load_config()
    except Exception as exc:
        return Check("role_chains", FAIL,
                     f"import/load failed: {type(exc).__name__}: {exc}")

    problems: list[str] = []
    notes: list[str] = []

    for name in CL.provider_order(cfg):
        adapter = CL.adapter_for(name, cfg)
        if adapter not in KNOWN_ADAPTERS:
            problems.append(f"provider '{name}': unknown adapter "
                            f"{adapter!r} (falls back to openai_compatible)")

    for role in ("chat", "build", "compact"):
        try:
            chain = CL.resolve_chain(role, cfg)
        except Exception as exc:
            problems.append(f"{role}: resolve raised {type(exc).__name__}: {exc}")
            continue
        if not chain:
            problems.append(f"{role}: no usable provider/model")
            continue
        prov, model = chain[0]
        if not model:
            problems.append(f"{role}: resolved to an empty model")
        elif CL.is_cli_provider(prov, cfg):
            # Anthropic heading the chain is only suspicious if an off-cap
            # provider that *outranks* it in provider_order fell out of the
            # walk -- and only then for a missing key, not a cap mark (a
            # skipped capped provider is rollover working as designed).
            order = CL.provider_order(cfg)
            in_chain = {p for p, _ in chain}
            ahead = order[:order.index(prov)] if prov in order else []
            keyless = [p for p in ahead
                       if p not in in_chain
                       and not CL.is_cli_provider(p, cfg)
                       and not CL.key_available(p, cfg)]
            if keyless:
                notes.append(f"{role}: resolved to Anthropic; "
                             f"{', '.join(keyless)} dropped (missing key)")
            else:
                notes.append(f"{role}->{prov}/{model}")
        else:
            notes.append(f"{role}->{prov}/{model}")

    try:
        cchain = CL.resolve_chain("compact", cfg)
        if cchain and CL.is_cli_provider(cchain[0][0], cfg) and len(cchain) > 1:
            notes.append("compact: Anthropic is first in the walk")
    except Exception:
        pass

    if problems:
        return Check("role_chains", FAIL, "; ".join(problems))
    if any("missing key" in n for n in notes):
        return Check("role_chains", WARN, "; ".join(notes))
    return Check("role_chains", OK, "; ".join(notes) or "chains resolve")


def check_mem1_budget() -> Check:
    try:
        import mem1
        s = mem1.sizes()
        text, stats = mem1.load()
    except Exception as exc:
        return Check("mem1_budget", FAIL,
                     f"mem1.load() raised {type(exc).__name__}: {exc}")

    over_hard = [
        spec.name for spec in mem1.FILES
        if s.get(spec.name, 0) > spec.hard
    ]
    skipped = [n for n, _c in stats.get("skipped", [])]
    truncated = "MISSING MEMORY" in text
    total = s.get("_total", 0)

    bits = []
    if over_hard:
        bits.append(f"over hard cap: {', '.join(over_hard)}")
    if skipped:
        bits.append(f"dropped (over budget): {', '.join(skipped)}")
    if truncated:
        bits.append("a file was truncated at load")
    if total > mem1.HARD_TOTAL:
        bits.append(f"total {total:,} > hard {mem1.HARD_TOTAL:,}")

    if bits:
        return Check("mem1_budget", FAIL, "; ".join(bits))

    near_cap = [
        spec.name for spec in mem1.FILES
        if s.get(spec.name, 0) > spec.hard * 0.9
    ]
    if near_cap:
        return Check("mem1_budget", WARN,
                     f"within 10% of cap: {', '.join(near_cap)} "
                     f"({total:,}/{mem1.HARD_TOTAL:,} tok total)")
    return Check("mem1_budget", OK,
                 f"{len(stats.get('loaded', []))} files, {total:,} tok, "
                 f"all under cap")


def check_hourly_ingest() -> Check:
    """The in-process hourly mem0 drain is running and not stalled."""
    try:
        st = json.loads(HOURLY_INGEST_STATUS_PATH.read_text())
    except FileNotFoundError:
        return Check("hourly_ingest", WARN,
                     "no hourly-ingest-status.json yet (drain never recorded a run)")
    except ValueError as exc:
        return Check("hourly_ingest", FAIL,
                     f"hourly-ingest-status.json corrupt: {exc}")

    reason = str(st.get("reason", "?"))
    last_run = st.get("last_run", "")
    age_h = None
    try:
        age_h = (_now() - dt.datetime.fromisoformat(last_run)).total_seconds() / 3600
    except (TypeError, ValueError):
        pass

    healthy_reason = (reason in ("drained", "nothing new")
                      or reason.startswith("budget"))
    if st.get("ok") is False and not healthy_reason:
        return Check("hourly_ingest", FAIL,
                     f"last drain not ok: {reason} (at {last_run})")
    if age_h is None:
        return Check("hourly_ingest", WARN,
                     f"last_run unparseable: {last_run!r}")
    if age_h > 26:
        return Check("hourly_ingest", FAIL,
                     f"last drain {age_h:.0f}h ago -- hourly loop not running? "
                     f"(the 12h time-backstop should force one)")
    if age_h > 14:
        return Check("hourly_ingest", WARN,
                     f"last drain {age_h:.0f}h ago (backstop is 12h)")
    return Check("hourly_ingest", OK,
                 f"drained {age_h:.0f}h ago, {st.get('added', 0)} added ({reason})")


def check_mem0_store(live: bool = False) -> Check:
    if not QDRANT_PATH.exists():
        return Check("mem0_store", FAIL, f"qdrant dir missing: {QDRANT_PATH}")
    try:
        entries = list(QDRANT_PATH.iterdir())
    except OSError as exc:
        return Check("mem0_store", FAIL, f"qdrant dir unreadable: {exc}")
    if not entries:
        return Check("mem0_store", WARN, "qdrant dir is empty")
    if not SOCKET_PATH.exists():
        return Check("mem0_store", FAIL if live else WARN,
                     f"IPC socket absent ({SOCKET_PATH.name}) -- "
                     f"memory tools unreachable")
    return Check("mem0_store", OK, "qdrant present, socket live"
                 if SOCKET_PATH.exists() else "qdrant present")


def check_backup_fresh() -> Check:
    """A backup succeeded in the last few days. 4-day cycle + slack."""
    try:
        st = json.loads(BACKUP_STATE.read_text())
        last = float(st["last_success"])
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return Check("backup", WARN, f"no readable backup state: {exc}")
    age_d = _age_hours(last) / 24
    if age_d > 6:
        return Check("backup", FAIL,
                     f"last successful backup {age_d:.1f} days ago")
    if age_d > 5:
        return Check("backup", WARN, f"last backup {age_d:.1f} days ago")
    return Check("backup", OK, f"{age_d:.1f} days ago, "
                 f"{st.get('pushed','?')}/{st.get('targets','?')} targets")


def check_disk() -> Check:
    try:
        usage = shutil.disk_usage(CCHAT_DATA)
    except OSError as exc:
        return Check("disk", WARN, f"disk_usage failed: {exc}")
    free_gib = usage.free / 2**30
    if free_gib < 0.5:
        return Check("disk", FAIL, f"{free_gib:.2f} GiB free on data partition")
    if free_gib < 1.0:
        return Check("disk", WARN, f"{free_gib:.2f} GiB free")
    return Check("disk", OK, f"{free_gib:.1f} GiB free")


def check_provider_keys() -> Check:
    import config_loader as CL
    cfg = CL.load_config()
    missing = [name for name in CL.provider_order(cfg) if not CL.key_available(name, cfg)]
    if missing:
        return Check("provider_keys", FAIL, "missing .env credentials: " + ", ".join(missing))
    return Check("provider_keys", OK, "configured provider credentials present")


def check_heartbeat(live: bool = False) -> Check:
    try:
        age = time.time() - HEARTBEAT_FILE.stat().st_mtime
    except FileNotFoundError:
        return Check("heartbeat", WARN if not live else FAIL,
                     "no heartbeat file")
    if not live:
        return Check("heartbeat", OK, f"{age:.0f}s old (not checked in CLI)")
    if age > 180:
        return Check("heartbeat", FAIL, f"{age:.0f}s stale (loop wedged?)")
    return Check("heartbeat", OK, f"{age:.0f}s old")


def check_crash_log() -> Check:
    """Informational: any crash logged in the last 25h."""
    try:
        text = CRASH_LOG.read_text()
    except FileNotFoundError:
        return Check("crash_log", OK, "no crash.log")
    if not text.strip():
        return Check("crash_log", OK, "empty")
    mtime_age = _age_hours(CRASH_LOG.stat().st_mtime)
    if mtime_age < 25:
        last = text.strip().splitlines()[-1][:120]
        return Check("crash_log", WARN,
                     f"written {mtime_age:.0f}h ago: {last}")
    return Check("crash_log", OK, f"last entry {mtime_age/24:.0f}d ago")


# runner


def run_checks(live: bool = False) -> list[Check]:
    fns = [
        check_config,
        check_role_chains,
        check_mem1_budget,
        check_hourly_ingest,
        lambda: check_mem0_store(live),
        check_backup_fresh,
        check_disk,
        check_provider_keys,
        lambda: check_heartbeat(live),
        check_crash_log,
    ]
    out: list[Check] = []
    for fn in fns:
        try:
            out.append(fn())
        except Exception as exc:  # a check must never take the caller down
            out.append(Check(getattr(fn, "__name__", "check"), FAIL,
                             f"check itself raised {type(exc).__name__}: {exc}"))
    return out


def _log(msg: str) -> None:
    try:
        SELFTEST_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SELFTEST_LOG.open("a") as f:
            f.write(f"{_now().isoformat(timespec='seconds')}  {msg}\n")
    except OSError:
        pass


def write_status(checks: list[Check]) -> dict:
    payload = {
        "ts": _now().isoformat(timespec="seconds"),
        "ok": not any(c.failed for c in checks),
        "checks": {c.name: {"status": c.status, "detail": c.detail}
                   for c in checks},
    }
    try:
        SELFTEST_STATUS.write_text(json.dumps(payload, indent=2))
    except OSError as exc:
        _log(f"could not write status: {exc}")
    return payload


def _prev_status() -> dict:
    try:
        return json.loads(SELFTEST_STATUS.read_text())
    except (OSError, ValueError):
        return {}


RESHOUT_HOURS = 12


def evaluate_for_shout(checks: list[Check]) -> str | None:
    prev = _prev_status()
    prev_checks = prev.get("checks", {})
    prev_ts = prev.get("ts", "")
    try:
        since = (_now() - dt.datetime.fromisoformat(prev_ts)).total_seconds() / 3600
    except (TypeError, ValueError):
        since = 1e9

    failed = [c for c in checks if c.failed]
    newly = [c for c in failed
             if prev_checks.get(c.name, {}).get("status") != FAIL]
    recovered = [c.name for c in checks
                 if not c.failed
                 and prev_checks.get(c.name, {}).get("status") == FAIL]

    write_status(checks)

    if newly:
        _log("NEW FAIL: " + "; ".join(f"{c.name}: {c.detail}" for c in newly))
    if recovered:
        _log("recovered: " + ", ".join(recovered))

    msg_parts = []
    if newly:
        msg_parts.append("⚠️ **selftest**: " + "; ".join(
            f"`{c.name}` {c.detail}" for c in newly))
    elif failed and since >= RESHOUT_HOURS:
        msg_parts.append("⚠️ **selftest** still failing: " + "; ".join(
            f"`{c.name}` {c.detail}" for c in failed))
    if recovered and not newly:
        msg_parts.append("✅ **selftest** recovered: " + ", ".join(recovered))

    return "\n".join(msg_parts) if msg_parts else None


def main(argv: list[str]) -> int:
    live = "--live" in argv
    as_json = "--json" in argv
    checks = run_checks(live=live)
    write_status(checks)

    if as_json:
        print(json.dumps(
            {c.name: {"status": c.status, "detail": c.detail} for c in checks},
            indent=2))
    else:
        width = max(len(c.name) for c in checks)
        for c in checks:
            mark = {OK: "  ok", WARN: "WARN", FAIL: "FAIL"}[c.status]
            print(f"  [{mark}] {c.name:<{width}}  {c.detail}")
        n_fail = sum(c.failed for c in checks)
        n_warn = sum(c.status == WARN for c in checks)
        print()
        print(f"  {n_fail} fail, {n_warn} warn, "
              f"{len(checks) - n_fail - n_warn} ok")

    return 1 if any(c.failed for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
