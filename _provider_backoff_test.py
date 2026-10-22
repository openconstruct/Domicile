"""In-place 429 backoff for the non-streaming helper calls.

Covers providers/openai_compatible.py:_acreate_retrying (async, compaction path),
mem0_or_llm.py:_create_retrying and mem0_oai_llm.py:OpenAILLM._chat (sync,
nightly extraction path). No network: the openai client is replaced with a
fake that fails N times then succeeds.

Run: python3 _provider_backoff_test.py
"""
import asyncio
import types

import providers.openai_compatible as ORP
import mem0_or_llm as MOR
import mem0_oai_llm as MOAI


class _Rate(ORP._RateLimitError):
    def __init__(self):
        # bypass APIStatusError's response/body-required __init__
        Exception.__init__(self, "Error code: 429 - Too Many Requests")


def _fake_client(fail_times, exc):
    calls = {"n": 0}

    async def acreate(**kw):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise exc()
        return "OK"

    def screate(**kw):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise exc()
        return "OK"

    ns = types.SimpleNamespace()
    ns.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(create=acreate))
    ns._screate = screate
    return ns, calls


def test_async_retries_then_succeeds():
    ORP._RATE_LIMIT_BACKOFF = (0.01, 0.01, 0.01)
    prov = ORP.OpenAICompatibleProvider.__new__(ORP.OpenAICompatibleProvider)
    client, calls = _fake_client(2, _Rate)
    prov._client = client
    out = asyncio.run(prov._acreate_retrying(model="m", messages=[]))
    assert out == "OK" and calls["n"] == 3, (out, calls)
    print("1. async: 2x 429 then success OK")


def test_async_exhausts_and_raises():
    ORP._RATE_LIMIT_BACKOFF = (0.01, 0.01)
    prov = ORP.OpenAICompatibleProvider.__new__(ORP.OpenAICompatibleProvider)
    client, calls = _fake_client(99, _Rate)
    prov._client = client
    try:
        asyncio.run(prov._acreate_retrying(model="m", messages=[]))
    except ORP._RateLimitError:
        assert calls["n"] == 3, calls  # 1 + len(backoff)
        print("2. async: exhausts backoff, re-raises 429 OK")
        return
    raise AssertionError("should have raised")


def test_async_non_429_raises_immediately():
    ORP._RATE_LIMIT_BACKOFF = (0.01, 0.01)
    prov = ORP.OpenAICompatibleProvider.__new__(ORP.OpenAICompatibleProvider)

    class Boom(Exception):
        pass

    client, calls = _fake_client(99, Boom)
    prov._client = client
    try:
        asyncio.run(prov._acreate_retrying(model="m", messages=[]))
    except Boom:
        assert calls["n"] == 1, calls
        print("3. async: non-429 raises on first try OK")
        return
    raise AssertionError("should have raised")


def test_sync_rate_limit_retries_but_hard_cap_does_not():
    MOR._RATE_LIMIT_BACKOFF = (0.01, 0.01, 0.01)
    llm = MOR.OpenRouterLLM.__new__(MOR.OpenRouterLLM)

    client, calls = _fake_client(2, _Rate)
    llm._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=client._screate)))
    assert llm._create_retrying(model="m", messages=[]) == "OK"
    assert calls["n"] == 3, calls

    class Cap(Exception):
        def __str__(self):
            return "Error code: 402 - insufficient credit"

    c2, calls2 = _fake_client(99, Cap)
    llm._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=c2._screate)))
    try:
        llm._create_retrying(model="m", messages=[])
    except Cap:
        assert calls2["n"] == 1, calls2
        print("4. sync: 429 retries, hard-cap raises immediately OK")
        return
    raise AssertionError("hard cap should raise on first try")


def test_sync_oai_rate_limit_retries_but_hard_cap_does_not():
    """mem0_oai_llm's mirror of test 4, added with the mem0_llm_common
    consolidation: before that, this module retried on ANY is_usage_limit
    match (including hard caps) and only mem0_or_llm had the smarter
    is_rate_limit split. Now both go through _common.is_rate_limit."""
    MOAI._RATE_LIMIT_BACKOFF = (0.01, 0.01, 0.01)
    llm = MOAI.OpenAILLM.__new__(MOAI.OpenAILLM)
    llm._no_thinking = False

    client, calls = _fake_client(2, _Rate)
    llm._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=client._screate)))
    assert llm._chat({"model": "m", "messages": []}) == "OK"
    assert calls["n"] == 3, calls

    class Cap(Exception):
        def __str__(self):
            return "Error code: 402 - insufficient credit"

    c2, calls2 = _fake_client(99, Cap)
    llm._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=c2._screate)))
    try:
        llm._chat({"model": "m", "messages": []})
    except Cap:
        assert calls2["n"] == 1, calls2
        print("5. sync oai: 429 retries, hard-cap raises immediately OK")
        return
    raise AssertionError("hard cap should raise on first try")


if __name__ == "__main__":
    test_async_retries_then_succeeds()
    test_async_exhausts_and_raises()
    test_async_non_429_raises_immediately()
    test_sync_rate_limit_retries_but_hard_cap_does_not()
    test_sync_oai_rate_limit_retries_but_hard_cap_does_not()
    print("\nALL OK")
