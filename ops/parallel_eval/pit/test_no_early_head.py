#!/usr/bin/env python3
"""Blocker 2: head advances only after successful live apply."""
import unittest

from .apply_live import ApplyLiveError, LiveApplicator
from .box import ScriptedBoxIO
from .config import PitConfig
from .controller import Controller
from .pe_lane import LaneService
from .pe_rent import RentService
from .state import ItemState


H0, M0 = 'a' * 64, 'b' * 64


class TestNoEarlyHead(unittest.TestCase):
    def _ctl(self, verify_uploads):
        clock = {'t': 50.0}

        def now():
            clock['t'] += 1.0
            return clock['t']

        cand = {'n': 0}

        def train(**kwargs):
            cand['n'] += 1
            cid = ('%x' % cand['n']).rjust(64, '0')
            return {
                'candidate_id': cid,
                'candidate_path': '/tmp/c',
                'candidate_master_path': '/tmp/m',
                'candidate_master_digest': M0,
                'candidate_sha256': cid,
                'used_fp32_bootstrap_override': False,
            }

        cfg = PitConfig(enabled=True, mode='live', max_boxes=1, cutover=True)
        applicator = LiveApplicator(
            config=cfg,
            verify_uploads=verify_uploads,
            accept_pit_decision=lambda _payload: True,
        )
        return Controller(
            config=cfg,
            rent=RentService(max_boxes=1),
            lane=LaneService(),
            box_io_factory=lambda: ScriptedBoxIO(on_train=train),
            score_fn=lambda **k: {'decision': 'green', 'z': 1.5},
            live_applicator=applicator,
            now_fn=now,
            round_id_fn=lambda e: 'r-e%d' % int(e),
            plan_seed_fn=lambda e: 'p' * 64,
        ), clock

    def test_head_unchanged_at_decide_advances_after_apply(self):
        ctl, _clock = self._ctl(verify_uploads=lambda **k: {'candidate_ok': True, 'master_ok': True})
        ctl.set_head(H0, epoch=5, master_digest=M0)
        ctl.enqueue(epoch=10, registration_index=1)
        seen_at_scored = []
        orig_decide = ctl._decide

        def watch_decide(item):
            out = orig_decide(item)
            if out[0].state is ItemState.DECIDED:
                seen_at_scored.append(ctl.current_head)
            return out

        ctl._decide = watch_decide  # type: ignore[method-assign]
        for _ in range(25):
            ctl.tick()
            if ctl.items and ctl.items[0].state is ItemState.RELEASED:
                break
        self.assertEqual(seen_at_scored, [H0])
        self.assertNotEqual(ctl.current_head, H0)
        self.assertEqual(ctl.items[0].decision, 'green_fresh')

    def test_verify_fail_keeps_head_and_no_paid_green_stuck(self):
        ctl, _clock = self._ctl(verify_uploads=lambda **k: {'candidate_ok': False, 'master_ok': True})
        ctl.set_head(H0, epoch=5, master_digest=M0)
        ctl.enqueue(epoch=11, registration_index=1)
        for _ in range(30):
            ctl.tick()
            if ctl.items and ctl.items[0].state in (ItemState.RELEASED, ItemState.NO_DECISION):
                break
        self.assertEqual(ctl.current_head, H0)
        item = ctl.items[0]
        self.assertIn(item.state, (ItemState.NO_DECISION, ItemState.RELEASED))
        if item.decision == 'no_decision':
            self.assertFalse(item.paid)
        self.assertFalse(item.decision == 'green_fresh' and item.paid and ctl.current_head != H0)


if __name__ == '__main__':
    unittest.main()
