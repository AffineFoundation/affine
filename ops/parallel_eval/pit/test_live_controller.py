#!/usr/bin/env python3
"""Live mode: GREEN_FRESH applies; stale paid+merge_blocked; RED unpaid; no promote on timeout."""
import unittest

from .apply_live import LiveApplicator, PromoteRefused, refuse_if_unpromotable
from .box import ScriptedBoxIO
from .config import PitConfig, load_pit_config
from .controller import Controller
from .pe_lane import LaneService
from .pe_rent import RentService
from .state import ItemState


H0, M0 = 'a' * 64, 'b' * 64


class TestLiveController(unittest.TestCase):
    def _ctl(self, **cfg_kw):
        clock = {'t': 100.0}

        def now():
            clock['t'] += 1.0
            return clock['t']

        cand = {'n': 0}
        applied = []
        destroyed = []

        def train(**kwargs):
            cand['n'] += 1
            cid = ('%x' % cand['n']).rjust(64, '0')
            return {
                'candidate_id': cid,
                'candidate_path': '/tmp/c%d' % cand['n'],
                'candidate_master_path': '/tmp/m%d' % cand['n'],
                'candidate_master_digest': ('%x' % (cand['n'] + 100)).rjust(64, '0'),
                'candidate_sha256': cid,
                'used_fp32_bootstrap_override': False,
            }

        cfg = PitConfig(
            enabled=True, mode='live', max_boxes=5, cutover=True,
            master_upload='after_green', **cfg_kw,
        )
        applicator = LiveApplicator(
            config=cfg,
            accept_pit_decision=lambda item: applied.append(dict(item)) or True,
            verify_uploads=lambda **k: {'candidate_ok': True, 'master_ok': True},
            destroy_state=lambda **k: destroyed.append(dict(k)) or {'ok': True},
            upload_master=lambda **k: {'ok': True},
            upload_candidate=lambda **k: {'ok': True},
        )
        ctl = Controller(
            config=cfg,
            rent=RentService(max_boxes=cfg.max_boxes, price_ceiling_usd_h=56.0),
            lane=LaneService(),
            box_io_factory=lambda: ScriptedBoxIO(on_train=train),
            now_fn=now,
            score_fn=lambda **k: {'decision': 'green', 'z': 1.5},
            live_applicator=applicator,
            destroy_box_state_fn=lambda **k: destroyed.append(dict(k)) or {'ok': True},
            round_id_fn=lambda e: 'r-e%d' % int(e),
            plan_seed_fn=lambda e: 'p' * 64,
        )
        ctl._test_applied = applied
        ctl._test_destroyed = destroyed
        return ctl

    def test_live_config_defaults(self):
        cfg = load_pit_config({'pit': {'enabled': True, 'mode': 'live', 'cutover': True}})
        self.assertEqual(cfg.max_boxes, 5)
        self.assertEqual(cfg.master_upload, 'after_green')
        self.assertEqual(cfg.gpu_order, ('B300',))
        self.assertEqual(cfg.price_ceiling_usd_h, 56.0)
        self.assertTrue(cfg.may_write_live_artifacts())
        self.assertTrue(cfg.cutover_active())

    def test_first_green_applies_second_stale(self):
        ctl = self._ctl()
        ctl.set_head(H0, epoch=5, master_digest=M0)
        ctl.enqueue(epoch=10, registration_index=1, hotkey_prefix='5One')
        ctl.enqueue(epoch=11, registration_index=2, hotkey_prefix='5Two')
        for _ in range(40):
            ctl.tick()
            if all(i.state is ItemState.RELEASED for i in ctl.items):
                break
        self.assertTrue(all(i.state is ItemState.RELEASED for i in ctl.items))
        fresh = [i for i in ctl.items if i.decision == 'green_fresh']
        stale = [i for i in ctl.items if i.decision == 'green_stale']
        self.assertEqual(len(fresh), 1)
        self.assertEqual(len(stale), 1)
        self.assertTrue(stale[0].paid and stale[0].merge_blocked)
        self.assertTrue(fresh[0].paid and not fresh[0].merge_blocked)
        self.assertEqual(len(ctl._test_applied), 1)
        self.assertEqual(ctl._test_applied[0]['source'], 'pit')
        self.assertEqual(ctl.rented_count(), 0)

    def test_red_earns_nothing(self):
        ctl = self._ctl()
        ctl.score_fn = lambda **k: {'decision': 'red', 'z': 0.2}
        ctl.set_head(H0, epoch=5, master_digest=M0)
        ctl.enqueue(epoch=20, registration_index=1, hotkey_prefix='5Red')
        for _ in range(20):
            ctl.tick()
            if ctl.items[0].state is ItemState.RELEASED:
                break
        self.assertEqual(ctl.items[0].decision, 'red')
        self.assertFalse(ctl.items[0].paid)

    def test_never_promote_on_timeout(self):
        with self.assertRaises(PromoteRefused):
            refuse_if_unpromotable({'decision': 'green', 'timeout': True})
        with self.assertRaises(PromoteRefused):
            refuse_if_unpromotable({'decision': 'green', 'partial': True})
        with self.assertRaises(PromoteRefused):
            refuse_if_unpromotable({'decision': 'no_decision'})

    def test_live_cap_5(self):
        ctl = self._ctl()
        ctl.set_head(H0, epoch=1, master_digest=M0)
        for i in range(7):
            ctl.enqueue(epoch=100 + i, registration_index=i + 1)
        ctl.tick()
        self.assertEqual(ctl.rented_count(), 5)

    def test_gpu_fallback_refused(self):
        with self.assertRaises(ValueError):
            load_pit_config({
                'pit': {
                    'enabled': True, 'mode': 'live',
                    'gpu_order': ['B300', 'B200'],
                },
            })

    def test_reds_earn_config_refused(self):
        with self.assertRaises(ValueError):
            load_pit_config({'pit': {'enabled': True, 'mode': 'live', 'reds_earn': True}})


if __name__ == '__main__':
    unittest.main()
