"""Provider fallback, rate limits, and recovery probes."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import re
from pathlib import Path

from paths import ROLLOVER_LOG
import config_loader as CL

# The Claude CLI's OAuth token has an 8h TTL and the CLI refreshes it when it
# spawns on a *dead* token -- not on one with a minute left. So a boundary
# crossed while idle is invisible (next spawn refreshes), while a boundary
# crossed mid-turn 401s the in-flight turn. Every live auth failure in
# rollover.log (11 of them, 09-04 -> 09-24) fits that shape, spaced 8-9.5h
# apart on busy days and absent on quiet ones.
CLAUDE_CREDENTIALS = Path.home() / ".claude" / ".credentials.json"
AUTH_REFRESH_MARGIN_S = 20 * 60


def _tiers() -> list:
    """The rollover order, from config.json."""
    try:
        return CL.provider_order()
    except Exception:
        return ["anthropic"]


def __getattr__(name):
    if name == "TIERS":
        return _tiers()
    raise AttributeError(name)

DEFAULT_WAIT_H = 6

# Checked for EVERY provider, including ones with no entry below. A provider
# that matches nothing is a dead end: run_turn() sees "not capped, not
# throttled, not auth" and hands the raw error to the user instead of rolling
# down the chain. Every provider in provider_order gets at least these.
_COMMON_CAP_SIGNATURES = (
    "usage limit",
    "quota exceeded",
    "insufficient_quota",
    "insufficient credits",
    "insufficient balance",
    "requires more credits",
    "purchase more credits",
    "out of credits",
    "plan limit",
    "monthly limit",
    "weekly limit",
    "error code: 402",
)

_COMMON_AUTH_SIGNATURES = (
    "invalid api key",
    "not authenticated",
    "authentication_error",
    "error code: 401",
)

_CAP_SIGNATURES = {
    "anthropic": (
        "hit your session limit",
        "session limit",
        "usage limit",
        "weekly limit",
        "5-hour limit",
        "exceeded your usage",
        "your limit will reset",
        "reached your usage limit",
    ),
    "openrouter": (
        "insufficient credits",
        "requires more credits",
        "insufficient_quota",
        "quota exceeded",
        "error code: 402",
    ),
    "opencode": (
        "usage limit",
        "quota exceeded",
        "insufficient_quota",
        "insufficient credits",
        "plan limit",
        "monthly limit",
        "error code: 402",
    ),
    "ali": (
        "arrears",
        "free allocated quota exceeded",
        "allocated quota exceeded",
        "quota exceeded",
        "insufficient balance",
        "insufficient_quota",
        "account balance",
        "expired",
    ),
    # codex: "You've hit your usage limit. Upgrade to Pro ... or try again at ..."
    "codex": (
        "hit your usage limit",
        "usage limits",
        "upgrade to pro",
        "codex/settings/usage",
    ),
}

_AUTH_SIGNATURES = {
    "anthropic": (
        "oauth access token has expired",
        "oauth token has expired",
        "failed to authenticate",
        "re-authenticate to continue",
        "authentication_error",
        "invalid bearer token",
        "please run /login",
        "invalid api key",
        "invalid x-api-key",
        "401 oauth",
    ),
    "openrouter": (
        "no auth credentials found",
        "user not found",
        "invalid api key",
        "not authenticated",
        "error code: 401",
    ),
    "opencode": (
        "invalid api key",
        "not authenticated",
        "unauthorized",
        "error code: 401",
        "error code: 403",
    ),
    "ali": (
        "invalid api-key",
        "invalid api key",
        "incorrectapikey",
        "invalidapikey",
        "error code: 401",
        "access denied",
    ),
    "codex": (
        "run `codex login`",
        "codex login",
        "not logged in",
        "unauthorized",
        "error code: 403",
    ),
}


_RATE_LIMIT_SIGNATURES = {
    "anthropic": (),
    "openrouter": (
        "error code: 429",
        "rate limit exceeded",
        "rate-limited",
        "too many requests",
    ),
    "opencode": (
        "error code: 429",
        "rate limit",
        "too many requests",
    ),
    "ali": (
        "error code: 429",
        "too many requests",
        "rate limit",
        "requests per minute",
        "throttling",
        "throttled",
    ),
    "codex": (
        "error code: 429",
        "too many requests",
        "rate limit",
        "stream disconnected",
    ),
}


_MODEL_SIGNATURES = (
    "model not found",
    "invalid model",
    "unknown model",
    "no endpoints found",
    "is not a valid model",
    "does not exist or you do not have access",
    "model_not_found",
    "context length",
    "context_length_exceeded",
    "maximum context",
    "too many tokens",
    "reduce the length",
)


def _log(line: str) -> None:
    try:
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with ROLLOVER_LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"{ts}  {line}\n")
    except Exception:
        pass


# detection

def _error_scope(text: str) -> str:
    """Lowercased error tail of a failed reply, or the whole thing.

    Every adapter marks the terminal failure with `**Error:**` and appends it
    AFTER whatever prose the model already streamed. Matching signatures
    against the full reply means a turn that merely *discusses* usage limits
    and then fails for an unrelated reason reads as a cap and triggers a
    spurious provider roll. Scope to the last marker onward when there is one.
    """
    low = text.lower()
    idx = low.rfind("**error:**")
    return low[idx:] if idx >= 0 else low


def looks_like_cap(provider: str, text: str) -> bool:
    """True if `text` (an error reply) reads as a usage cap for `provider`."""
    if not text:
        return False
    low = _error_scope(text)
    for sig in _CAP_SIGNATURES.get(provider, ()) + _COMMON_CAP_SIGNATURES:
        if sig in low:
            return True
    return False


def looks_like_auth_failure(provider: str, text: str) -> bool:
    if not text:
        return False
    low = _error_scope(text)
    if any(sig in low for sig in _COMMON_AUTH_SIGNATURES):
        return True
    return any(sig in low for sig in _AUTH_SIGNATURES.get(provider, ()))


def looks_like_rate_limit(provider: str, text: str) -> bool:
    if not text:
        return False
    low = _error_scope(text)
    if looks_like_cap(provider, text):
        return False
    # Deliberately per-provider only, no common set: anthropic's tuple is
    # empty on purpose -- a rate-limit-shaped message there IS the session
    # cap, and retrying it with backoff instead of rolling over wastes turns.
    return any(sig in low for sig in _RATE_LIMIT_SIGNATURES.get(provider, ()))


def looks_like_model_error(provider: str, text: str) -> bool:
    """True if `text` blames the MODEL rather than the account."""
    if not text:
        return False
    low = _error_scope(text)
    if (looks_like_cap(provider, text) or looks_like_auth_failure(provider, text)
            or looks_like_rate_limit(provider, text)):
        return False
    return any(sig in low for sig in _MODEL_SIGNATURES)


RATE_LIMIT_RETRY_DELAYS = (5.0, 15.0, 40.0)


def rate_limit_delay(attempt: int) -> float:
    """Backoff (seconds, with jitter) for retry `attempt` (0-indexed)."""
    import random
    base = RATE_LIMIT_RETRY_DELAYS[min(attempt, len(RATE_LIMIT_RETRY_DELAYS) - 1)]
    return base + random.uniform(0, base * 0.3)


def reset_hint_text(text: str) -> str:
    """A short human snippet like 'resets 6:40pm' for the Discord notice."""
    if not text:
        return ""
    m = re.search(r"reset[s]?\b[^.\n]{0,40}", text, re.I)
    return m.group(0).strip() if m else ""


def parse_reset_hint(text: str):
    """Best-effort local-time datetime from 'resets 6:40pm' / 'resets at 10 pm'."""
    if not text:
        return None
    m = re.search(
        r"reset[s]?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m",
        text.lower(),
    )
    if not m:
        return None
    hour = int(m.group(1)) % 12
    if m.group(3) == "p":
        hour += 12
    minute = int(m.group(2) or 0)
    now = dt.datetime.now().astimezone()
    try:
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    except ValueError:
        return None
    if target <= now:
        target += dt.timedelta(days=1)
    return target


# tier chain

def key_available(tier: str) -> bool:
    """Can we actually authenticate to this tier right now?"""
    try:
        return CL.key_available(tier)
    except Exception:
        return False


def next_tier(current: str) -> str | None:
    """The next provider below `current` with a usable key and no live mark."""
    order = _tiers()
    if not order:
        return None
    try:
        idx = order.index(current)
    except ValueError:
        idx = -1
    down = marked_providers()
    for tier in order[idx + 1:] + order[:max(idx, 0)]:
        if tier == current or tier in down:
            continue
        if key_available(tier):
            return tier
    return None


# marker I/O


def read_state() -> dict | None:
    """The whole runtime state, or None when nothing is marked."""
    st = CL.load_runtime_state()
    return st if st.get("providers") else None


def marked_providers() -> set:
    try:
        return CL.unusable_providers()
    except Exception:
        return set()


def provider_mark(name: str) -> dict | None:
    return (CL.load_runtime_state().get("providers") or {}).get(name)


def clear_state(reason: str = "") -> None:
    st = CL.load_runtime_state()
    names = list((st.get("providers") or {}).keys())
    if not names:
        return
    st["providers"] = {}
    CL.save_runtime_state(st)
    tail = f" ({reason})" if reason else ""
    _log(f"marks cleared: {', '.join(names)}{tail}")


def clear_provider(name: str, reason: str = "") -> None:
    """Clear one provider's mark -- it probed healthy again."""
    if not provider_mark(name):
        return
    CL.clear_mark(name)
    tail = f" ({reason})" if reason else ""
    _log(f"mark cleared: {name}{tail}")


def record_fallback(origin: str, to_tier: str, reply_text: str,
                    transient: bool = False) -> dict:
    """Mark `origin` unusable and log the hop onto `to_tier`."""
    now = dt.datetime.now().astimezone()
    reset_at = parse_reset_hint(reply_text)
    if transient and not reset_at:
        first_probe = now + dt.timedelta(minutes=15)
    else:
        ceiling = now + dt.timedelta(hours=DEFAULT_WAIT_H)
        if reset_at and now < reset_at < ceiling:
            first_probe = reset_at
        else:
            first_probe = ceiling

    if looks_like_auth_failure(origin, reply_text):
        kind = "auth"
    elif looks_like_cap(origin, reply_text):
        kind = "cap"
    else:
        kind = "throttle"

    st = CL.mark_unusable(
        origin, kind,
        detail=reset_hint_text(reply_text) or (reply_text or "")[:200],
        reset_at=reset_at.isoformat() if reset_at else None,
        next_probe_at=first_probe.isoformat(),
    )
    _log(f"marked {origin} unusable ({kind}); rolled to {to_tier} "
         f"(probe at {first_probe.isoformat()})")
    return st["providers"][origin]


def due_for_probe(mark: dict) -> bool:
    npa = (mark or {}).get("next_probe_at")
    if not npa:
        return True
    try:
        return dt.datetime.now().astimezone() >= dt.datetime.fromisoformat(npa)
    except ValueError:
        return True


def providers_due_for_probe() -> list:
    """Every marked provider whose probe is due, best-first in config order."""
    st = CL.load_runtime_state()
    marks = st.get("providers") or {}
    return [name for name in _tiers()
            if name in marks and marks[name].get("auto", True)
            and due_for_probe(marks[name])]


def note_probe(name: str, ok: bool) -> None:
    """Record a failed probe and schedule the next with escalating backoff."""
    st = CL.load_runtime_state()
    mark = (st.get("providers") or {}).get(name)
    if not mark:
        return
    now = dt.datetime.now().astimezone()
    mark["last_probe"] = now.isoformat()
    mark["attempts"] = int(mark.get("attempts", 0)) + 1
    hours = min(3, mark["attempts"])  # 1h, 2h, 3h, 3h, ...
    mark["next_probe_at"] = (now + dt.timedelta(hours=hours)).isoformat()
    CL.save_runtime_state(st)
    _log(f"probe #{mark['attempts']} of {name}: still {mark.get('unusable')}; "
         f"next in {hours}h")


# recovery probe

async def probe_recovered(origin: str) -> bool:
    """Cheap check: has `origin` stopped rejecting requests?"""
    if origin in ("anthropic", "claude"):
        ok = await _probe_anthropic()
    else:
        ok = await _probe_openai(origin)
    if ok:
        _log(f"probe of {origin}: recovered")
    return ok


def anthropic_token_seconds_left(now: dt.datetime | None = None) -> float | None:
    """Seconds until the Claude CLI OAuth token expires, or None if unknown.

    None means "don't act": no credentials file, console/API auth instead of
    OAuth, or an unreadable shape. Never guess a number here -- a wrong
    estimate would burn a refresh on every turn.
    """
    try:
        raw = json.loads(CLAUDE_CREDENTIALS.read_text(encoding="utf-8"))
    except Exception:
        return None
    exp = ((raw.get("claudeAiOauth") or {}).get("expiresAt")
           if isinstance(raw, dict) else None)
    if not isinstance(exp, (int, float)):
        return None
    ref = now or dt.datetime.now()
    return exp / 1000.0 - ref.timestamp()


async def refresh_anthropic_token_if_stale(
        margin_s: float = AUTH_REFRESH_MARGIN_S) -> bool:
    """Nudge the CLI to refresh when the token would expire mid-turn.

    Returns True only when the expiry actually moved. Best effort: the
    in-place retry in run_turn() is what actually recovers from a dead token.
    """
    left = anthropic_token_seconds_left()
    if left is None or left > margin_s:
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            "claude", "auth", "status",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.communicate(), timeout=30)
    except Exception as exc:
        _log(f"auth margin: token {left:.0f}s from expiry; "
             f"refresh nudge failed ({type(exc).__name__}: {exc})")
        return False
    after = anthropic_token_seconds_left()
    moved = after is not None and after > left + 60
    _log(f"auth margin: token was {left:.0f}s from expiry; after nudge "
         + (f"{after:.0f}s -- refreshed" if moved
            else f"{after if after is None else round(after)}s -- NOT refreshed"))
    return moved


async def _probe_anthropic() -> bool:
    model = None
    try:
        from providers import first_model
        model = first_model("compact", "anthropic")
    except Exception:
        pass
    cmd = ["claude", "-p", "ping"]
    if model:
        cmd += ["--model", model]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except Exception as exc:
        _log(f"anthropic probe: could not spawn claude ({exc!r})")
        return False
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=90)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        _log("anthropic probe: timed out")
        return False
    text = (out or b"").decode("utf-8", "replace")
    if proc.returncode != 0:
        return False
    return (not looks_like_cap("anthropic", text)
            and not looks_like_auth_failure("anthropic", text))


async def _probe_openai(tier: str) -> bool:
    try:
        from providers import get_provider
        prov = get_provider(force=tier)
        resp = await asyncio.wait_for(
            prov._client.chat.completions.create(
                model=prov.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            ),
            timeout=60,
        )
        return resp is not None
    except Exception as exc:
        _log(f"{tier} probe: {type(exc).__name__}: {exc}")
        return False
