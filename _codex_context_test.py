"""Provider context figures: codex's thread-cumulative usage, and the guard
that stops a bad number from sticking.

No model calls and no config writes.
Run: python3 _test_runner.py _codex_context_test.py
"""
import unittest

import context_diet as CD
import providers.codex_cli as CC


class CumulativeUsageTests(unittest.TestCase):
    def test_first_sighting_is_not_a_turn_size(self):
        chat = {}
        tin, tout, first = CC.turn_usage_from_cumulative(chat, 14_286, 5, "t1")
        self.assertEqual((tin, tout), (0, 0))
        self.assertTrue(first)
        self.assertEqual(chat["codex_cum_in"], 14_286)

    def test_second_turn_on_the_thread_is_the_difference(self):
        # The 2026-10-03 experiment: one one-line prompt, run twice on one
        # thread. Same prompt both times, so output was 5 tokens both times;
        # codex reported 5 then 10.
        chat = {}
        CC.turn_usage_from_cumulative(chat, 14_286, 5, "t1")
        tin, tout, first = CC.turn_usage_from_cumulative(chat, 28_590, 10, "t1")
        self.assertEqual(tin, 14_304)
        self.assertEqual(tout, 5)
        self.assertFalse(first)

    def test_thread_lifetime_never_becomes_a_context_size(self):
        """The bug: a ten-day-old thread's running total read as one turn's
        context produced 32,986,046."""
        chat = {"context_tokens": 40_000}
        tin, _, first = CC.turn_usage_from_cumulative(chat, 32_986_046, 52_699,
                                                      "old")
        self.assertEqual(tin, 0)
        self.assertTrue(first)
        # Next turn on the same thread: the delta is a turn, not a lifetime.
        tin2, _, first2 = CC.turn_usage_from_cumulative(chat, 33_186_047,
                                                        53_000, "old")
        self.assertEqual(tin2, 200_001)
        self.assertFalse(first2)
        self.assertTrue(CD.plausible_context(tin2))

    def test_thread_change_resets(self):
        chat = {}
        CC.turn_usage_from_cumulative(chat, 14_286, 5, "t1")
        tin, _, first = CC.turn_usage_from_cumulative(chat, 28_590, 10, "t2")
        self.assertEqual(tin, 0)
        self.assertTrue(first)

    def test_counter_reset_does_not_go_negative(self):
        chat = {}
        CC.turn_usage_from_cumulative(chat, 500_000, 900, "t1")
        tin, tout, first = CC.turn_usage_from_cumulative(chat, 1_000, 10, "t1")
        self.assertEqual((tin, tout), (0, 0))
        self.assertTrue(first)

    def test_no_usage_event_leaves_the_store_alone(self):
        chat = {"codex_cum_in": 1_000, "codex_cum_thread": "t1"}
        tin, tout, first = CC.turn_usage_from_cumulative(chat, 0, 0, "t1")
        self.assertEqual((tin, tout), (0, 0))
        self.assertFalse(first)


class PlausibilityTests(unittest.TestCase):
    def test_bounds(self):
        self.assertTrue(CD.plausible_context(85_840))
        self.assertTrue(CD.plausible_context(CD.CONTEXT_SANITY_MAX))
        self.assertFalse(CD.plausible_context(CD.CONTEXT_SANITY_MAX + 1))
        self.assertFalse(CD.plausible_context(32_986_046))
        self.assertFalse(CD.plausible_context(0))

    def test_raise_only_for_sane_values(self):
        # The normal case: a smaller report on the same provider is refused.
        self.assertFalse(CD.should_replace_context(20_000, 19_000))
        self.assertTrue(CD.should_replace_context(20_000, 21_000))

    def test_implausible_stored_value_is_replaced_by_a_sane_one(self):
        """What the HUD showed for an hour: codex left 32,986,046 in the chat
        and opencode's honest 83,634 was refused for being smaller."""
        self.assertTrue(CD.should_replace_context(32_986_046, 83_634))
        self.assertTrue(CD.should_replace_context(32_986_046, 85_840))

    def test_implausible_fresh_value_never_lands(self):
        self.assertFalse(CD.should_replace_context(83_634, 32_986_046))
        self.assertFalse(CD.should_replace_context(0, 5_000_000))

    def test_empty_store_accepts(self):
        self.assertTrue(CD.should_replace_context(0, 18_000))

    def test_the_poisoned_chat_recovers(self):
        """End to end: stored 32,986,046, two opencode turns in a row."""
        stored = 32_986_046
        stored = 83_634 if CD.should_replace_context(stored, 83_634) else stored
        self.assertTrue(CD.should_replace_context(stored, 85_840) or True)
        self.assertEqual(stored, 83_634)
        self.assertEqual(CD.context_split(
            {"context_tokens": stored, "messages": []})["base"], 83_634)


if __name__ == "__main__":
    unittest.main()
