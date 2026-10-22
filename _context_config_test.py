"""Install-time context settings: compaction ceiling + mem1 caps from config.

Run: python3 _test_runner.py _context_config_test.py
"""
import unittest
from unittest import mock

import config_loader as CL
import context_diet as cd
import mem1


def _cfg(**context):
    cfg = CL.load_config()
    cfg["context"] = dict(CL.DEFAULTS["context"], **context)
    return cfg


class CeilingTest(unittest.TestCase):
    def _ceiling(self, **ctx):
        with mock.patch.object(CL, "load_config", return_value=_cfg(**ctx)):
            return cd._ceiling_from_config()

    def test_default_unchanged(self):
        self.assertEqual(self._ceiling(), 60_000)

    def test_override(self):
        self.assertEqual(self._ceiling(compact_ceiling_tokens=150_000), 150_000)

    def test_too_low_is_floored_not_loop(self):
        self.assertEqual(self._ceiling(compact_ceiling_tokens=5_000),
                         cd.MIN_COMPACT_CEILING)

    def test_garbage_falls_back(self):
        self.assertEqual(self._ceiling(compact_ceiling_tokens="big"), 60_000)


class FloorTest(unittest.TestCase):
    def _with(self, fn, **ctx):
        with mock.patch.object(CL, "load_config", return_value=_cfg(**ctx)):
            return fn()

    def test_default_floor_and_tail(self):
        # floor counts compacted text only: tail = floor - summary
        self.assertEqual(self._with(cd._floor_from_config), 30_000)
        self.assertEqual(self._with(cd.compact_tail_tokens),
                         30_000 - cd.COMPACT_SUMMARY_TOKENS)

    def test_bigger_floor_grows_tail(self):
        self.assertEqual(self._with(cd.compact_tail_tokens,
                                    compact_floor_tokens=40_000),
                         40_000 - cd.COMPACT_SUMMARY_TOKENS)

    def test_floor_clamped_to_minimum(self):
        self.assertEqual(self._with(cd._floor_from_config,
                                    compact_floor_tokens=1_000),
                         cd.MIN_COMPACT_FLOOR)
        self.assertEqual(self._with(cd.compact_tail_tokens,
                                    compact_floor_tokens=1_000),
                         cd.MIN_TAIL_TOKENS)

    def test_garbage_floor_falls_back(self):
        for bad in ("big", -5, True, None):
            self.assertEqual(self._with(cd._floor_from_config,
                                        compact_floor_tokens=bad), 30_000)

    def test_ceiling_clears_floor_by_headroom(self):
        # ceiling 35k, floor 40k -> ceiling raised to floor + headroom
        self.assertEqual(self._with(cd._ceiling_from_config,
                                    compact_ceiling_tokens=35_000,
                                    compact_floor_tokens=40_000),
                         40_000 + cd.COMPACT_HEADROOM)

    def test_split_uses_floor_at_call_time(self):
        msgs = [{"role": "user" if i % 2 == 0 else "assistant",
                 "text": "x" * 4_000} for i in range(40)]   # ~1.4k tok each
        small = self._with(lambda: cd.split_for_compaction(msgs)[1],
                           compact_floor_tokens=12_000)
        big = self._with(lambda: cd.split_for_compaction(msgs)[1],
                         compact_floor_tokens=40_000)
        self.assertGreater(len(big), len(small))


def _convo(n_exchanges, chars=4_000):
    """n user/assistant pairs; each message ~1.4k tokens at chars=4000."""
    out = []
    for i in range(n_exchanges):
        out.append({"role": "user", "text": f"u{i} " + "x" * chars})
        out.append({"role": "assistant", "text": f"a{i} " + "x" * chars})
    return out


class ProtectedTailTest(unittest.TestCase):
    def test_protected_not_counted_against_budget(self):
        # tiny tail budget: the newest exchanges survive anyway
        msgs = _convo(20, chars=400)          # ~140 tok per message
        head, tail = cd.split_for_compaction(msgs, tail_tokens=300)
        self.assertGreaterEqual(len(tail), 2 * cd.COMPACT_PROTECTED_EXCHANGES)
        self.assertEqual(tail[-1], msgs[-1])
        self.assertEqual(head + tail, msgs)

    def test_tail_budget_adds_on_top_of_protected(self):
        msgs = _convo(30, chars=400)
        _, none = cd.split_for_compaction(msgs, tail_tokens=1)
        _, more = cd.split_for_compaction(msgs, tail_tokens=2_000)
        self.assertGreater(len(more), len(none))

    def test_protected_cap_limits_big_turns(self):
        msgs = _convo(10, chars=4_000)        # ~2.8k tok per exchange
        p = cd._protected_start(msgs)
        kept = msgs[p:]
        self.assertLess(len(kept) // 2, cd.COMPACT_PROTECTED_EXCHANGES)
        self.assertLessEqual(cd.history_tokens(kept),
                             cd.COMPACT_PROTECTED_CAP)

    def test_newest_exchange_kept_even_over_cap(self):
        msgs = _convo(5, chars=100)
        msgs[-1] = {"role": "assistant", "text": "y" * 60_000}
        p = cd._protected_start(msgs)
        self.assertEqual(p, len(msgs) - 2)

    def test_whole_history_fits_nothing_to_summarize(self):
        msgs = _convo(3, chars=400)
        self.assertEqual(cd.split_for_compaction(msgs, tail_tokens=10_000),
                         ([], msgs))

    def test_cut_lands_on_user_message(self):
        msgs = _convo(30, chars=400)
        head, tail = cd.split_for_compaction(msgs, tail_tokens=1_500)
        self.assertEqual(tail[0]["role"], "user")

    def test_empty(self):
        self.assertEqual(cd.split_for_compaction([]), ([], []))
        self.assertEqual(cd._protected_start([]), 0)


class HistoryTokensTest(unittest.TestCase):
    def test_counts_text_only(self):
        msgs = [{"role": "user", "text": "x" * 4_000},
                {"role": "assistant", "text": None},
                {"role": "assistant"}]
        self.assertEqual(cd.history_tokens(msgs), cd.text_tokens("x" * 4_000))

    def test_post_compaction_history_clears_ceiling(self):
        # floor + protected cap at worst case must sit below the ceiling
        worst = cd.DEFAULT_COMPACT_FLOOR + cd.COMPACT_PROTECTED_CAP
        self.assertLess(worst, cd.DEFAULT_COMPACT_CEILING)


class ContextSplitTest(unittest.TestCase):
    @staticmethod
    def _m(role, n):
        return {"role": role, "text": "x" * n}

    def test_excludes_last_reply_and_later(self):
        m = self._m
        chat = {"context_tokens": 28_000, "messages": [
            m("user", 4_000), m("assistant", 4_000),
            m("user", 8_000), m("assistant", 4_000), m("user", 400)]}
        s = cd.context_split(chat)
        # history = first three messages only (last assistant + later cut)
        self.assertEqual(s["history"], cd.history_tokens(chat["messages"][:3]))
        self.assertEqual(s["base"], 28_000 - s["history"])
        self.assertFalse(s["estimated"])

    def test_none_without_total(self):
        self.assertIsNone(cd.context_split({"messages": []}))

    def test_history_never_exceeds_total(self):
        chat = {"context_tokens": 5_000, "context_estimated": True,
                "messages": [self._m("user", 400_000), self._m("assistant", 10)]}
        s = cd.context_split(chat)
        self.assertEqual((s["history"], s["base"]), (5_000, 0))
        self.assertTrue(s["estimated"])


class Mem1CapsTest(unittest.TestCase):
    def setUp(self):
        self.orig = {s.name: s.hard for s in mem1.FILES}

    def tearDown(self):
        for s in mem1.FILES:
            s.hard = self.orig[s.name]

    def _apply(self, **ctx):
        for s in mem1.FILES:
            s.hard = self.orig[s.name]
        with mock.patch.object(CL, "load_config", return_value=_cfg(**ctx)):
            mem1._apply_config_caps()
        return {s.name: s.hard for s in mem1.FILES}

    def test_default_unchanged(self):
        self.assertEqual(self._apply(), self.orig)

    def test_total_scales_proportionally(self):
        caps = self._apply(mem1_total=9_500)   # half of 19,000
        self.assertEqual(caps["identity.md"], 2_000)
        self.assertEqual(caps["user.md"], 1_750)

    def test_per_file_override_wins(self):
        caps = self._apply(mem1_total=9_500, mem1_caps={"user.md": 6_000})
        self.assertEqual(caps["user.md"], 6_000)
        self.assertEqual(caps["identity.md"], 2_000)

    def test_floor_and_unknown_names(self):
        caps = self._apply(mem1_total=100, mem1_caps={"nope.md": 9})
        self.assertTrue(all(v == mem1.MIN_FILE_CAP for v in caps.values()))


if __name__ == "__main__":
    unittest.main()
