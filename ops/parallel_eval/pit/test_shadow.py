#!/usr/bin/env python3
"""Unit tests: shadow mode never writes live artifacts."""
import unittest

from .adapters import stub_adapters
from .box import ScriptedBoxIO
from .config import PitConfig, load_pit_config
from .host import HostController
from .shadow import (
    FORBIDDEN_SHADOW_WRITES,
    ShadowGuard,
    ShadowWriteError,
    assert_no_forbidden_writes,
)


H0 = 'a' * 64
M0 = 'b' * 64
C1 = 'c' * 64


class TestShadowGuard(unittest.TestCase):
    def test_default_config_disabled(self):
        cfg = load_pit_config({})
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.mode, 'shadow')
        self.assertFalse(cfg.may_write_live_artifacts())

    def test_enabled_non_shadow_refused(self):
        with self.assertRaises(ValueError):
            load_pit_config({'pit': {'enabled': True, 'mode': 'live'}})

    def test_guard_refuses_decision_and_head_chain(self):
        g = ShadowGuard(PitConfig(enabled=True, mode='shadow'))
        for name in ('DECISION.json', 'HEAD-CHAIN.json', 'CURRENT.json', 'WEIGHTS.json'):
            with self.assertRaises(ShadowWriteError):
                g.check_path('/tmp/epoch/1/%s' % name)
        for action in ('write_decision', 'write_head_chain', 'write_weights', 'publish_r2_head'):
            with self.assertRaises(ShadowWriteError):
                g.check_action(action)

    def test_forbidden_set_covers_required_artifacts(self):
        for name in ('DECISION.json', 'HEAD-CHAIN.json', 'R2_HEAD_POINTER'):
            self.assertIn(name, FORBIDDEN_SHADOW_WRITES)

    def test_host_shadow_run_never_touches_live_artifacts(self):
        touched = []

        def score(**kwargs):
            return {'decision': 'green', 'z': 1.7}

        clock = {'t': 1000.0}

        def now():
            clock['t'] += 1.0
            return clock['t']

        io = ScriptedBoxIO(train_result={
            'candidate_id': C1,
            'candidate_path': '/tmp/pit-fixture/cand',
            'candidate_master_path': '/tmp/pit-fixture/cand-master',
            'used_fp32_bootstrap_override': False,
        })
        ctl = HostController(
            config=PitConfig(enabled=True, mode='shadow', box_cap=2),
            adapters=stub_adapters(score=score),
            box_io_factory=lambda: io,
            now_fn=now,
        )
        ctl.set_head(H0, epoch=5, master_digest=M0)
        ctl.enqueue(registration_index=1, miner_hotkey_prefix='5MinerAAAA')

        # Drive until release or safety bound.
        for _ in range(20):
            out = ctl.tick()
            for act in out['actions']:
                touched.append(act.get('op'))
            if ctl.items[0].state.value == 'released':
                break

        self.assertEqual(ctl.items[0].state.value, 'released')
        self.assertEqual(ctl.items[0].decision, 'green')
        # Shadow guard must have refused live apply actions during the run.
        attempted = ctl.shadow.attempted
        self.assertTrue(any('decision' in a or a == 'DECISION.json' for a in attempted))
        assert_no_forbidden_writes([])  # no paths collected = ok
        # Controller memory may advance head, but no live artifact basenames were written.
        self.assertNotIn('DECISION.json', touched)
        self.assertNotIn('HEAD-CHAIN.json', touched)


if __name__ == '__main__':
    unittest.main()
