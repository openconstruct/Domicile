"""Anthropic OAuth expiry: pre-turn margin nudge + in-place retry.

The CLI's token has an 8h TTL and only refreshes when it spawns on a dead one,
so a boundary crossed mid-turn 401s that turn. Rolling over to a lower tier was
the old response and was always wrong -- these tests pin the recovery path.

Run: python3 _test_runner.py _auth_refresh_test.py  (writes provider marks)
"""
import asyncio
import datetime as dt
import json

from _test_runner import require_sandbox

require_sandbox()

import cchat
import provider_rollover as R

AUTH_ERR = ("**Error:** `Failed to authenticate. API Error: 401 OAuth access "
            "token has expired. Re-authenticate to continue.`")


def _write_creds(seconds_left):
    """Fake ~/.claude/.credentials.json with `seconds_left` until expiry."""
    R.CLAUDE_CREDENTIALS.parent.mkdir(parents=True, exist_ok=True)
    exp = (dt.datetime.now().timestamp() + seconds_left) * 1000
    R.CLAUDE_CREDENTIALS.write_text(json.dumps(
        {"claudeAiOauth": {"accessToken": "x", "expiresAt": exp}}))


def test_seconds_left_unknown_shapes():
    R.CLAUDE_CREDENTIALS.parent.mkdir(parents=True, exist_ok=True)
    if R.CLAUDE_CREDENTIALS.exists():
        R.CLAUDE_CREDENTIALS.unlink()
    assert R.anthropic_token_seconds_left() is None, "missing file must be None"
    R.CLAUDE_CREDENTIALS.write_text("not json{")
    assert R.anthropic_token_seconds_left() is None, "garbage must be None"
    # console / API-key auth has no claudeAiOauth block -- must not guess
    R.CLAUDE_CREDENTIALS.write_text(json.dumps({"apiKey": "sk-x"}))
    assert R.anthropic_token_seconds_left() is None, "non-OAuth must be None"
    R.CLAUDE_CREDENTIALS.write_text(json.dumps(
        {"claudeAiOauth": {"expiresAt": "soon"}}))
    assert R.anthropic_token_seconds_left() is None, "non-numeric must be None"
    print("1. unknown credential shapes -> None (no guessing) OK")


def test_seconds_left_reads_expiry():
    _write_creds(3600)
    left = R.anthropic_token_seconds_left()
    assert 3590 < left < 3601, left
    _write_creds(-90)
    left = R.anthropic_token_seconds_left()
    assert -95 < left < -85, left
    print("2. expiry arithmetic OK (negative when already dead)")


def test_margin_gate_skips_healthy_token():
    spawned = []

    async def _boom(*a, **k):
        spawned.append(a)
        raise AssertionError("must not spawn claude on a healthy token")

    orig = asyncio.create_subprocess_exec
    asyncio.create_subprocess_exec = _boom
    try:
        _write_creds(8 * 3600)
        assert asyncio.run(R.refresh_anthropic_token_if_stale()) is False
        if R.CLAUDE_CREDENTIALS.exists():
            R.CLAUDE_CREDENTIALS.unlink()
        assert asyncio.run(R.refresh_anthropic_token_if_stale()) is False
    finally:
        asyncio.create_subprocess_exec = orig
    assert not spawned, spawned
    print("3. margin gate: no nudge when token is healthy/unknown OK")


def test_margin_nudge_reports_honestly():
    """A nudge that doesn't move the expiry must report False, not success."""
    class _Proc:
        async def communicate(self):
            return (b"", b"")

    async def _fake(*a, **k):
        return _Proc()

    orig = asyncio.create_subprocess_exec
    asyncio.create_subprocess_exec = _fake
    try:
        _write_creds(120)  # inside the 20min margin
        assert asyncio.run(R.refresh_anthropic_token_if_stale()) is False, \
            "unchanged expiry must not read as refreshed"
        # now the nudge does move it: simulate by rewriting mid-call
        class _Proc2:
            async def communicate(self):
                _write_creds(8 * 3600)
                return (b"", b"")

        async def _fake2(*a, **k):
            return _Proc2()
        asyncio.create_subprocess_exec = _fake2
        _write_creds(120)
        assert asyncio.run(R.refresh_anthropic_token_if_stale()) is True
    finally:
        asyncio.create_subprocess_exec = orig
    log = R.ROLLOVER_LOG.read_text() if R.ROLLOVER_LOG.exists() else ""
    assert "NOT refreshed" in log and "-- refreshed" in log, log
    print("4. nudge logs refreshed vs NOT refreshed OK")


def _patch(stream_fn, start="anthropic"):
    R.clear_state()
    box = {"p": start}
    cchat.provider_name = lambda: box["p"]
    cchat.set_provider = lambda p: box.__setitem__("p", p) or p

    async def _shout(app, text):
        return None
    cchat._shout = _shout
    cchat.stream_claude = stream_fn
    return box


def test_auth_failure_retries_in_place():
    calls = []

    async def stream(chat, text, on_delta, images, source=""):
        calls.append(cchat.provider_name())
        return (AUTH_ERR, "error") if len(calls) == 1 else ("ok", "ok")

    box = _patch(stream)

    async def main():
        chat = {"messages": [], "mode": "chat", "session_id": "keep-me"}
        reply, status = await cchat.run_turn(object(), chat, "hi")
        assert status == "ok" and reply == "ok", (status, reply)
        assert box["p"] == "anthropic", "must not roll over on a refreshable 401"
        assert calls == ["anthropic", "anthropic"], calls
        assert chat["session_id"] == "keep-me", "in-place retry keeps the session"
        assert R.provider_mark("anthropic") is None, R.read_state()
    asyncio.run(main())
    print("5. auth 401 -> one in-place retry, no mark, session kept OK")


def test_retry_is_once_then_rolls_over():
    target = R.next_tier("anthropic")
    assert target and target != "anthropic", "need a lower tier for this test"
    calls = []

    async def stream(chat, text, on_delta, images, source=""):
        calls.append(cchat.provider_name())
        if cchat.provider_name() == "anthropic":
            return (AUTH_ERR, "error")
        return ("ok", "ok")

    box = _patch(stream)

    async def main():
        chat = {"messages": [], "mode": "chat", "session_id": "x"}
        reply, status = await cchat.run_turn(object(), chat, "hi")
        assert status == "ok" and box["p"] == target, (status, box["p"])
        # exactly two anthropic attempts: the turn and its single retry
        assert calls[:2] == ["anthropic", "anthropic"], calls
        assert calls.count("anthropic") == 2, calls
        st = R.provider_mark("anthropic")
        assert st and st["unusable"] == "auth", st
        R.clear_state()
    asyncio.run(main())
    print("6. persistent 401 -> retried once, then rolled over OK")


if __name__ == "__main__":
    test_seconds_left_unknown_shapes()
    test_seconds_left_reads_expiry()
    test_margin_gate_skips_healthy_token()
    test_margin_nudge_reports_honestly()
    test_auth_failure_retries_in_place()
    test_retry_is_once_then_rolls_over()
    R.clear_state()
    print("\nALL OK")
