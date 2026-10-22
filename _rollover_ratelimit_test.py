"""Rate-limit vs cap distinction in provider_rollover + cchat.run_turn.

No network. Monkeypatches cchat.stream_claude / provider_name / set_provider.
Run: python3 _test_runner.py _rollover_ratelimit_test.py  (NOT directly -- this
file writes provider marks and rollover.log lines, and outside the sandbox
those land in live state; see _test_runner.require_sandbox).
"""
import asyncio
import datetime as dt

from _test_runner import require_sandbox

require_sandbox()

import cchat
import provider_rollover as R


def test_signature_classification():
    assert R.looks_like_rate_limit("ali", "Error code: 429 - Too Many Requests")
    assert not R.looks_like_cap("ali", "Error code: 429 - Too Many Requests")
    assert R.looks_like_rate_limit("openrouter", "Rate limit exceeded, retry later")
    assert R.looks_like_cap("openrouter", "Insufficient credits to run this request")
    assert not R.looks_like_rate_limit("openrouter",
                                       "Insufficient credits to run this request")
    # a hard-cap signal always wins even if a throttle word is also present
    assert R.looks_like_cap("ali", "account balance low, error code: 429")
    assert not R.looks_like_rate_limit("ali", "account balance low, error code: 429")
    # anthropic never rate-limits through this path
    assert not R.looks_like_rate_limit("anthropic", "anything")
    assert R.looks_like_cap("anthropic",
                            "You hit your session limit; your limit will reset at 6pm")
    print("1. signature classification OK")


def test_backoff():
    for _ in range(50):
        assert 5.0 <= R.rate_limit_delay(0) <= 5.0 * 1.3
        assert 15.0 <= R.rate_limit_delay(1) <= 15.0 * 1.3
        assert 40.0 <= R.rate_limit_delay(2) <= 40.0 * 1.3
        assert R.rate_limit_delay(9) <= 40.0 * 1.3  # clamps to last
    print("2. backoff bounds OK")


def test_transient_marker_short_probe():
    R.clear_state()
    st = R.record_fallback("anthropic", "openrouter", "Error code: 429", transient=True)
    delta = (dt.datetime.fromisoformat(st["next_probe_at"])
             - dt.datetime.now().astimezone()).total_seconds()
    assert 800 < delta < 1000, delta
    st2 = R.record_fallback("anthropic", "openrouter", "weekly limit reached",
                            transient=False)
    delta2 = (dt.datetime.fromisoformat(st2["next_probe_at"])
              - dt.datetime.now().astimezone()).total_seconds()
    assert delta2 > 3600 * 5, delta2
    R.clear_state()
    print("3. transient marker short-probe OK")


def _patch(stream_fn, start="openrouter"):
    R.clear_state()
    R.RATE_LIMIT_RETRY_DELAYS = (0.01, 0.01)
    box = {"p": start}
    cchat.provider_name = lambda: box["p"]
    cchat.set_provider = lambda p: box.__setitem__("p", p) or p

    async def _shout(app, text):
        return None
    cchat._shout = _shout
    cchat.stream_claude = stream_fn
    return box


def test_inplace_retry_no_rollover():
    seq = iter([("Error code: 429 too many requests", "error"),
                ("Error code: 429 too many requests", "error"),
                ("ok now", "ok")])

    async def stream(chat, text, on_delta, images, source=""):
        return next(seq)

    box = _patch(stream, start="ali")

    async def main():
        reply, status = await cchat.run_turn(object(), {"messages": [], "mode": "chat"}, "hi")
        assert status == "ok" and box["p"] == "ali", (status, box["p"])
        assert R.read_state() is None
    asyncio.run(main())
    print("4. in-place retry, no rollover OK")


def test_persistent_throttle_rolls_over():
    # The chain gained an `opencode` tier (2026-09-01), so derive the
    # expected hop from TIERS + key_available rather than hardcoding the
    # old anthropic -> openrouter -> ali shape. Start on the first tier that
    # can emit a 429 rate-limit signal AND has a usable lower tier to roll to
    # (anthropic can't rate-limit through this path -- see test 1).
    start = next(t for t in R.TIERS
                 if R.looks_like_rate_limit(t, "Error code: 429 rate limit exceeded")
                 and R.next_tier(t) not in (None, t))
    target = R.next_tier(start)

    async def stream(chat, text, on_delta, images, source=""):
        if cchat.provider_name() == start:
            return ("Error code: 429 rate limit exceeded", "error")
        return ("%s ok" % cchat.provider_name(), "ok")

    box = _patch(stream, start=start)

    async def main():
        chat = {"messages": [], "mode": "chat", "session_id": "x"}
        reply, status = await cchat.run_turn(object(), chat, "hi")
        assert status == "ok" and box["p"] == target, (status, box["p"])
        # State is per-provider now: the mark lives under the provider that
        # went down, not in a single from/to pair.
        st = R.provider_mark(start)
        assert st, R.read_state()
        assert start in R.marked_providers(), R.marked_providers()
        delta = (dt.datetime.fromisoformat(st["next_probe_at"])
                 - dt.datetime.now().astimezone()).total_seconds()
        assert 700 < delta < 1000, delta  # transient => short probe
        R.clear_state()
    asyncio.run(main())
    print("5. persistent throttle -> rollover (transient marker) OK")


def test_auth_failure_classification():
    exp = ("Failed to authenticate. API Error: 401 OAuth access token has "
           "expired. Re-authenticate to continue.")
    assert R.looks_like_auth_failure("anthropic", exp)
    assert not R.looks_like_cap("anthropic", exp)
    assert not R.looks_like_rate_limit("anthropic", exp)
    assert R.looks_like_auth_failure("openrouter", "Error code: 401 - No auth credentials found")
    assert not R.looks_like_auth_failure("anthropic", "the weekly limit will reset at 6pm")
    assert not R.looks_like_auth_failure("anthropic", "")
    print("6. auth-failure classification OK")


def test_auth_failure_rolls_over():
    start = "anthropic"
    target = R.next_tier(start)
    assert target and target != start, "need a usable lower tier for this test"

    async def stream(chat, text, on_delta, images, source=""):
        if cchat.provider_name() == start:
            return ("**Error:** `Failed to authenticate. API Error: 401 OAuth "
                    "access token has expired. Re-authenticate to continue.`",
                    "error")
        return ("%s ok" % cchat.provider_name(), "ok")

    box = _patch(stream, start=start)

    async def main():
        chat = {"messages": [], "mode": "chat", "session_id": "x"}
        reply, status = await cchat.run_turn(object(), chat, "hi")
        assert status == "ok" and box["p"] == target, (status, box["p"])
        st = R.provider_mark("anthropic")
        assert st and st["unusable"] == "auth", st
        # auth failure => short probe cadence, not the 6h cap ceiling
        delta = (dt.datetime.fromisoformat(st["next_probe_at"])
                 - dt.datetime.now().astimezone()).total_seconds()
        assert 700 < delta < 1000, delta
        assert chat["session_id"] is None
        R.clear_state()
    asyncio.run(main())
    print("7. auth failure -> rollover OK")


if __name__ == "__main__":
    test_signature_classification()
    test_backoff()
    test_transient_marker_short_probe()
    test_inplace_retry_no_rollover()
    test_persistent_throttle_rolls_over()
    test_auth_failure_classification()
    test_auth_failure_rolls_over()
    R.clear_state()
    print("\nALL OK")
