"""Cap/auth/rate signature classification, incl. the codex dead-end bug.

2026-09-24: `codex` was in provider_order with a usable `opencode` tier under
it, hit its ChatGPT usage cap, and the turn DIED with the raw error instead of
rolling over -- because codex had no entry in any signature table, so
run_turn() read the failure as "not capped, not throttled, not auth" and
returned it. These tests pin the two fixes: common signatures that apply to
every provider (no provider is a dead end), and error-scoped matching (prose
about usage limits can't fake a cap).
"""

import provider_rollover as R

failures = []


def check(name, got, want):
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")


# The verbatim reply from the 2026-09-24 failure.
CODEX_CAP = (
    "**Error:** `codex exec` exited 1: You've hit your usage limit. Upgrade "
    "to Pro (https://chatgpt.com/explore/pro), visit "
    "https://chatgpt.com/codex/settings/usage to purchase more credits or try "
    "again at Sep 30th, 2026 7:49 AM."
)

check("codex cap", R.looks_like_cap("codex", CODEX_CAP), True)
check("codex not rate", R.looks_like_rate_limit("codex", CODEX_CAP), False)
check("codex not model", R.looks_like_model_error("codex", CODEX_CAP), False)
check("codex not auth", R.looks_like_auth_failure("codex", CODEX_CAP), False)
print("1. codex usage cap classified as a cap OK")

# A provider nobody wrote a signature table for must still be escapable.
check("unknown cap", R.looks_like_cap("brandnew", CODEX_CAP), True)
check("unknown auth", R.looks_like_auth_failure(
    "brandnew", "**Error:** `error code: 401 invalid api key`"), True)
check("unknown quota", R.looks_like_cap(
    "brandnew", "**Error:** `insufficient_quota: quota exceeded`"), True)
check("unknown plain failure", R.looks_like_cap(
    "brandnew", "**Error:** `ValueError: bad json`"), False)
print("2. unlisted provider is not a dead end OK")

# Prose that merely discusses caps, then an unrelated terminal failure.
PROSE = (
    "We were talking about how the usage limit stall works, and why quota "
    "exceeded on one provider matters.\n\n"
    "**Error:** `TimeoutError: tool round timed out`"
)
check("prose no cap (codex)", R.looks_like_cap("codex", PROSE), False)
check("prose no cap (anthropic)", R.looks_like_cap("anthropic", PROSE), False)
check("prose then real cap", R.looks_like_cap(
    "codex", "some streamed prose first\n\n" + CODEX_CAP), True)
check("no marker falls back to whole text", R.looks_like_cap(
    "anthropic", "you have hit your session limit"), True)
print("3. cap matching scoped to the error, not streamed prose OK")

# Pre-existing providers keep their behaviour.
check("anthropic auth", R.looks_like_auth_failure(
    "anthropic",
    "**Error:** `Failed to authenticate. API Error: 401 OAuth token has "
    "expired`"), True)
check("anthropic rate stays empty", R.looks_like_rate_limit(
    "anthropic", "**Error:** `rate limit exceeded, too many requests`"), False)
check("ali rate", R.looks_like_rate_limit(
    "ali", "**Error:** `error code: 429 too many requests`"), True)
check("openrouter cap", R.looks_like_cap(
    "openrouter", "**Error:** `insufficient credits`"), True)
check("model error survives", R.looks_like_model_error(
    "opencode", "**Error:** `no endpoints found for deepseek`"), True)
print("4. existing provider classification unchanged OK")

if failures:
    print()
    for line in failures:
        print("FAIL", line)
    raise SystemExit(1)
print()
print("ALL OK")
