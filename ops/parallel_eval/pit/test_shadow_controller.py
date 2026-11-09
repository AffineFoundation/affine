#!/usr/bin/env python3
import unittest

from .box import ScriptedBoxIO
from .config import PitConfig
from .controller import Controller
from .pe_lane import LaneService
from .pe_rent import RentService
from .shadow import FORBIDDEN_SHADOW_WRITES, ShadowGuard, ShadowWriteError, assert_no_forbidden_writes
from .state import ItemState


H0, M0 = 'a' * 64, 'b' * 64


class TestShadowController(unittest.TestCase):
    def test_guard_refuses_live_artifacts(self):
        g = ShadowGuard(PitConfig(enabled=True, mode='shadow'))
        for name in ('DECISION.json', 'HEAD-CHAIN.json', 'CURRENT.json', 'WEIGHTS.json'):
            with self.assertRaises(ShadowWriteError):
                g.check_path('/tmp/%s' % name)
        for name in ('DECISION.json', 'HEAD-CHAIN.json', 'R2_HEAD_POINTER'):
            self.assertIn(name, FORBIDDEN_SHADOW_WRITES)

    def test_controller_shadow_never_writes_live(self):
        clock = {'t': 0.0}

        def now():
            clock['t'] += 1
            return float(clock['t'])

        cand = {'n': 0}

        def train(**kwargs):
            cand['n'] += 1
            return {
                'candidate_id': ('%x' % cand['n']).rjust(64, '0'),
                'candidate_path': '/tmp/c',
                'candidate_master_path': '/tmp/m',
                'used_fp32_bootstrap_override': False,
            }

        live_decisions = {
            10: {'decision': 'green', 'candidate': 'f' * 64, 'z': 1.4},
        }

        ctl = Controller(
            config=PitConfig(enabled=True, mode='shadow', max_boxes=2),
            rent=RentService(max_boxes=2),
            lane=LaneService(),
            box_io_factory=lambda: ScriptedBoxIO(on_train=train),
            now_fn=now,
            score_fn=lambda **k: {'decision': 'green', 'z': 1.5},
            live_decision_lookup=lambda e: live_decisions.get(e),
        )
        ctl.set_head(H0, epoch=5, master_digest=M0)
        ctl.enqueue(epoch=10, registration_index=1, hotkey_prefix='5One')
        ctl.enqueue(epoch=11, registration_index=2, hotkey_prefix='5Two')

        for _ in range(30):
            ctl.tick()
            if all(i.state is ItemState.RELEASED for i in ctl.items):
                break

        self.assertTrue(all(i.state is ItemState.RELEASED for i in ctl.items))
        greens = [i for i in ctl.items if i.decision and i.decision.startswith('green')]
        self.assertEqual(len(greens), 2)
        blocked = [i for i in greens if i.merge_blocked]
        fresh = [i for i in greens if not i.merge_blocked]
        self.assertEqual(len(fresh), 1)
        self.assertEqual(len(blocked), 1)
        # Shadow comparison used box candidate, not live's.
        for i in ctl.items:
            comp = (i.meta or {}).get('shadow_comparison')
            if comp:
                self.assertEqual(comp['box_candidate'], i.candidate_id)
                self.assertNotEqual(comp['box_candidate'], 'f' * 64)
        self.assertEqual(ctl.rented_count(), 0)
        assert_no_forbidden_writes([])
        attempted = ctl.shadow.attempted
        self.assertTrue(any('decision' in a or a == 'DECISION.json' for a in attempted))

    def test_disabled_is_noop(self):
        ctl = Controller(
            config=PitConfig(enabled=False, mode='shadow'),
            rent=RentService(),
            lane=LaneService(),
            box_io_factory=ScriptedBoxIO,
        )
        ctl.enqueue(epoch=1, registration_index=1)
        out = ctl.tick()
        self.assertFalse(out['enabled'])
        self.assertEqual(ctl.items[0].state, ItemState.QUEUED)
        self.assertEqual(out['rentals'], 0)


if __name__ == '__main__':
    unittest.main()
