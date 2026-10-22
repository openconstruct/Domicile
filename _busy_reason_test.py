"""`busy_reason`: telling an out-of-band UI *why* the app is blocked.

The TUI shows a `[yellow]draining[/yellow]` / `[yellow]compacting[/yellow]`
ctx-badge, but the web UI has no view of it -- from there a compaction or an
hourly mem0 drain looks like a send that silently does nothing, because both
hold `turn_lock`. `session_state()` now carries a reason string the web UI
renders as a centered overlay.

The load-bearing detail: the reason is a *module global*, not a screen
attribute. `_set_draining_badge` early-returns when the focused screen isn't a
ChatScreen, so anything stored on the screen is invisible exactly when the
user is most likely to be on the web UI instead of the TUI.

Run under _test_runner.py.
"""
import asyncio
import unittest
import unittest.mock

import cchat


class BusyReasonGlobal(unittest.TestCase):
    def setUp(self):
        cchat.set_busy_reason(None)

    def tearDown(self):
        cchat.set_busy_reason(None)

    def test_set_and_clear(self):
        cchat.set_busy_reason("compacting")
        self.assertEqual(cchat.BUSY_REASON, "compacting")
        cchat.set_busy_reason(None)
        self.assertIsNone(cchat.BUSY_REASON)

    def test_starts_clear(self):
        self.assertIsNone(cchat.BUSY_REASON)


class DrainBadgeSetsReason(unittest.TestCase):
    """The early-return path is the whole point -- no screen, still reported."""

    def setUp(self):
        cchat.set_busy_reason(None)

    def tearDown(self):
        cchat.set_busy_reason(None)

    def test_no_chatscreen_still_sets_reason(self):
        stub = unittest.mock.Mock(spec=["screen"])
        stub.screen = object()          # not a ChatScreen -> early return
        cchat.CCChat._set_draining_badge(stub, True)
        self.assertEqual(cchat.BUSY_REASON, "draining memory")
        cchat.CCChat._set_draining_badge(stub, False)
        self.assertIsNone(cchat.BUSY_REASON)

    def test_screen_access_raising_still_sets_reason(self):
        class Boom:
            @property
            def screen(self):
                raise RuntimeError("no screen stack yet")

        cchat.CCChat._set_draining_badge(Boom(), True)
        self.assertEqual(cchat.BUSY_REASON, "draining memory")

    def test_off_clears_even_when_screen_raises(self):
        class Boom:
            @property
            def screen(self):
                raise RuntimeError("no screen stack yet")

        cchat.set_busy_reason("draining memory")
        cchat.CCChat._set_draining_badge(Boom(), False)
        self.assertIsNone(cchat.BUSY_REASON)


class SessionStateCarriesReason(unittest.TestCase):
    def tearDown(self):
        cchat.set_busy_reason(None)

    def _state(self):
        stub = unittest.mock.Mock(spec=["_bridge_target", "turn_lock"])
        stub._bridge_target.return_value = ({"id": "c1", "mode": "chat"}, None)
        stub.turn_lock = asyncio.Lock()
        with unittest.mock.patch.object(cchat, "provider_name",
                                        return_value="anthropic"), \
             unittest.mock.patch.object(cchat, "effective_model",
                                        return_value="opus"), \
             unittest.mock.patch.object(cchat, "thinking_cycle_for",
                                        return_value=["off", "high"]):
            return cchat.CCChat.session_state(stub)

    def test_key_present_and_none_when_idle(self):
        cchat.set_busy_reason(None)
        st = self._state()
        self.assertIn("busy_reason", st)
        self.assertIsNone(st["busy_reason"])

    def test_reports_current_reason(self):
        cchat.set_busy_reason("compacting")
        self.assertEqual(self._state()["busy_reason"], "compacting")

    def test_busy_and_reason_are_independent(self):
        """Mid-turn is busy with no reason; a drain is busy *with* one."""
        cchat.set_busy_reason(None)
        self.assertIsNone(self._state()["busy_reason"])
        cchat.set_busy_reason("draining memory")
        self.assertEqual(self._state()["busy_reason"], "draining memory")


class CompactionWiring(unittest.TestCase):
    """maybe_compact sets the reason on entry and clears it on every exit."""

    def tearDown(self):
        cchat.set_busy_reason(None)

    def test_cleared_after_build_seed_failure(self):
        if cchat.COMPACT is None:
            self.skipTest("compaction module unavailable")
        seen = []

        async def boom(*a, **k):
            seen.append(cchat.BUSY_REASON)
            raise RuntimeError("synthetic")

        chat = {"id": "c1", "title": "t",
                "messages": [{"role": "user", "text": "x" * 400_000}]}
        with unittest.mock.patch.object(cchat.COMPACT, "build_seed", boom), \
             unittest.mock.patch.object(cchat.COMPACT, "compact_enabled",
                                        return_value=True), \
             unittest.mock.patch.object(cchat, "save_chat", lambda *a, **k: None):
            asyncio.run(cchat.maybe_compact(chat, None))
        self.assertEqual(seen, ["compacting"], "reason not set before the work")
        self.assertIsNone(cchat.BUSY_REASON, "reason leaked after a failure")

    def test_not_set_when_below_ceiling(self):
        if cchat.COMPACT is None:
            self.skipTest("compaction module unavailable")
        with unittest.mock.patch.object(cchat.COMPACT, "compact_enabled",
                                        return_value=True):
            asyncio.run(cchat.maybe_compact({"id": "c1", "context_tokens": 1},
                                            None))
        self.assertIsNone(cchat.BUSY_REASON)


if __name__ == "__main__":
    unittest.main()
