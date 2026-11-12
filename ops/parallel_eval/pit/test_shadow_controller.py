#!/usr/bin/env python3
"""Shadow controller — hermetic (no /home/const / $HOME path defaults)."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from .box import ScriptedBoxIO
from .config import PitConfig
from .controller import Controller
from .pe_lane import LaneService
from .pe_rent import RentService
from .shadow import SHADOW_R2_PREFIXES, ShadowGuard, ShadowWriteError, assert_no_forbidden_writes
from .state import ItemState


H0, M0 = 'a' * 64, 'b' * 64


class TestShadowController(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.pe = self.td / 'pe'
        self.life = self.td / 'life'
        self.pe.mkdir()
        self.life.mkdir()
        (self.life / 'epochs').mkdir()
        (self.pe / 'pit').mkdir()
        # Isolate every host path — training_job_checkpoint / queue_ingest must
        # never read real /home/const epochs (e.g. epoch 10).
        self._env = {
            'PE_STATE': str(self.pe),
            'LIFE_ROOT': str(self.life),
            'PE_LIFE': str(self.life),
            'PE_LIFE_STATE': str(self.life),
            'PIT_STATE': str(self.pe / 'pit'),
            'K1_TURNS': str(self.td / 'K1-TURNS.jsonl'),
            'PIT_FP32_MASTER_INDEX': str(self.td / 'fp32'),
        }
        (self.td / 'K1-TURNS.jsonl').write_text('')
        (self.td / 'fp32').mkdir()
        self._prev = {k: os.environ.get(k) for k in self._env}
        os.environ.update(self._env)
        # Matching TRAINING-JOB so stale-head check does not skip rent.
        for ep in (10, 11):
            ed = self.life / 'epochs' / str(ep)
            ed.mkdir(parents=True, exist_ok=True)
            (ed / 'TRAINING-JOB.json').write_text(json.dumps({
                'payload': {'manifest': {'payload': {'checkpoint': {'id': H0}}}},
            }))
            (ed / 'TRAIN-DISPATCH-FF.json').write_text(json.dumps({
                'signature': 's', 'signer': 'ROOT', 'payload': {},
            }))

    def tearDown(self):
        for k, v in self._prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_guard_refuses_live_artifacts(self):
        g = ShadowGuard(PitConfig(enabled=True, mode='shadow'))
        for name in ('DECISION.json', 'HEAD-CHAIN.json', 'CURRENT.json', 'WEIGHTS.json'):
            with self.assertRaises(ShadowWriteError):
                g.check_path('/tmp/%s' % name)

    def test_guard_is_an_allowlist_of_r2_prefixes_and_of_the_state_root(self):
        self.assertEqual(SHADOW_R2_PREFIXES, ('pit-shadow/',))
        root = self.td / 'shadow-state'
        g = ShadowGuard(PitConfig(enabled=True, mode='shadow'), state_root=root)
        # R2: a key under the allowlist passes. Each other key is refused, also one with a name nobody listed.
        g.check_r2_key('pit-shadow/pending/%s/model.safetensors' % ('c' * 64))
        g.check_r2_key('pit-shadow/fp32-master/%s/model-00001-of-00004.safetensors' % ('c' * 64))
        for key in ('fp32-master/%s/model.safetensors' % ('c' * 64), 'pit-pending/x', 'ckpts/x', 'heads/current',
                    'a-new-live-prefix/x', 'pit-shadow', 'pit-shadowx/y', 'pit-shadow/../fp32-master/x',
                    '/pit-shadow/x', '', None):
            with self.assertRaises(ShadowWriteError, msg=key):
                g.check_r2_key(key)
        # Files: the shadow state root passes. Each other place is refused, with any file name.
        g.check_path(root / 'pit' / 'items' / 'DECISION.json')
        for path in (self.pe / 'DECISION.json', self.pe / 'a-new-live-file.json', self.life / 'CURRENT.json',
                     root / '..' / 'pe' / 'x.json', self.td / 'shadow-state-2' / 'x.json'):
            with self.assertRaises(ShadowWriteError, msg=path):
                g.check_path(path)
        assert_no_forbidden_writes([root / 'pit' / 'x.json'], root)
        with self.assertRaises(ShadowWriteError):
            assert_no_forbidden_writes([root / 'pit' / 'x.json', self.pe / 'HEAD-CHAIN.json'], root)
        # No shadow state root: no file write passes.
        none = ShadowGuard(PitConfig(enabled=True, mode='shadow'))
        os.environ.pop('PIT_SHADOW_STATE', None)
        with self.assertRaises(ShadowWriteError):
            none.check_path(root / 'pit' / 'x.json')
        # Live mode after cutover: the guard is off.
        live = ShadowGuard(PitConfig(enabled=True, mode='live'))
        live.check_r2_key('fp32-master/x')
        live.check_path(self.pe / 'DECISION.json')

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
            round_id_fn=lambda e: 'r-e%d' % int(e),
            plan_seed_fn=lambda e: ('p' * 64),
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
