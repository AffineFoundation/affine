#!/usr/bin/env python3
"""Host-level PIT shadow e2e: first GREEN applies in memory; later GREEN is merge_blocked."""
import unittest

from .adapters import stub_adapters
from .box import ScriptedBoxIO
from .config import PitConfig
from .host import HostController
from .state import ItemState


H0 = 'a' * 64
M0 = 'b' * 64


class TestHostE2E(unittest.TestCase):
    def _ctl(self, score_fn, *, box_cap=2):
        clock = {'t': 100.0}

        def now():
            clock['t'] += 1.0
            return clock['t']

        cand = {'n': 0}

        def train(**kwargs):
            cand['n'] += 1
            cid = ('%x' % cand['n']).rjust(64, '0')
            return {
                'candidate_id': cid,
                'candidate_path': '/tmp/pit-fixture/cand-%d' % cand['n'],
                'candidate_master_path': '/tmp/pit-fixture/cand-m-%d' % cand['n'],
                'used_fp32_bootstrap_override': False,
            }

        io = ScriptedBoxIO(on_train=train)
        return HostController(
            config=PitConfig(enabled=True, mode='shadow', box_cap=box_cap),
            adapters=stub_adapters(score=score_fn),
            box_io_factory=lambda: io,
            now_fn=now,
        )

    def _drive(self, ctl, steps=40):
        for _ in range(steps):
            ctl.tick()
            if all(i.state in (ItemState.RELEASED, ItemState.ABORTED) for i in ctl.items):
                break

    def test_first_green_becomes_head_second_stale_merge_blocked(self):
        def score(**kwargs):
            return {'decision': 'green', 'z': 1.5}

        ctl = self._ctl(score, box_cap=2)
        ctl.set_head(H0, epoch=10, master_digest=M0)
        ctl.enqueue(registration_index=1, miner_hotkey_prefix='5MinerOne')
        ctl.enqueue(registration_index=2, miner_hotkey_prefix='5MinerTwo')
        self._drive(ctl)

        states = {i.registration_index: i.state for i in ctl.items}
        # Both finished; exactly one APPLIED (then RELEASED) and one MERGE_BLOCKED (then RELEASED).
        decisions = sorted(
            [(i.registration_index, i.decision, i.merge_blocked, i.paid) for i in ctl.items]
        )
        self.assertEqual(len(ctl.items), 2)
        self.assertTrue(all(i.state is ItemState.RELEASED for i in ctl.items))
        paid_greens = [i for i in ctl.items if i.decision == 'green']
        self.assertEqual(len(paid_greens), 2)
        blocked = [i for i in ctl.items if i.merge_blocked]
        applied_like = [i for i in ctl.items if i.decision == 'green' and not i.merge_blocked]
        # One applied (first to finish scoring while head still H0), one merge_blocked.
        self.assertEqual(len(applied_like), 1, decisions)
        self.assertEqual(len(blocked), 1, decisions)
        # Decision log green-time order
        greens = ctl.decision_log.greens_in_order()
        self.assertEqual(len(greens), 2)
        self.assertEqual(greens[0].decision, 'green')
        self.assertEqual(greens[1].decision, 'merge_blocked')
        # Head advanced in controller memory only (shadow; no HEAD-CHAIN write).
        self.assertNotEqual(ctl.current_head, H0)
        self.assertEqual(ctl.current_head, applied_like[0].candidate_id)

    def test_disabled_tick_is_noop(self):
        from .shadow import ShadowGuard
        ctl = self._ctl(lambda **k: {'decision': 'red', 'z': 0.0})
        ctl.config = PitConfig(enabled=False, mode='shadow')
        ctl.shadow = ShadowGuard(ctl.config)
        ctl.enqueue(registration_index=1, miner_hotkey_prefix='5X')
        out = ctl.tick()
        self.assertFalse(out['enabled'])
        self.assertEqual(ctl.items[0].state, ItemState.QUEUED)

    def test_box_cap_default_two(self):
        self.assertEqual(PitConfig().box_cap, 2)
        self.assertTrue(HostController(
            config=PitConfig(enabled=False),
            adapters=stub_adapters(),
            box_io_factory=lambda: ScriptedBoxIO(),
        ).trainer_fallback_available())


if __name__ == '__main__':
    unittest.main()
