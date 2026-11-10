#!/usr/bin/env python3
"""Live fail-closed wiring (pitsim): refuse unwired host / embedded scores."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from .box import ScriptedBoxIO
from .config import PitConfig
from .controller import Controller
from .pe_score import ScoreError, score
from .run_host import build_controller
from .durable import DurableStore


H0, M0 = 'a' * 64, 'b' * 64


class TestLiveFailClosed(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self._env = {
            'PE_STATE': str(self.td / 'pe'),
            'LIFE_ROOT': str(self.td / 'life'),
            'PE_LIFE': str(self.td / 'life'),
            'PE_LIFE_STATE': str(self.td / 'life'),
            'PIT_STATE': str(self.td / 'pe' / 'pit'),
            'K1_TURNS': str(self.td / 'K1-TURNS.jsonl'),
            'PIT_FP32_MASTER_INDEX': str(self.td / 'fp32'),
        }
        for p in (self.td / 'pe' / 'pit', self.td / 'life', self.td / 'fp32'):
            p.mkdir(parents=True, exist_ok=True)
        (self.td / 'K1-TURNS.jsonl').write_text('')
        self._prev = {k: os.environ.get(k) for k in self._env}
        os.environ.update(self._env)

    def tearDown(self):
        for k, v in self._prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_from_flag_live_refuses_default_box_io(self):
        with self.assertRaises(ValueError) as cm:
            Controller.from_flag({'pit': {'enabled': True, 'mode': 'live', 'cutover': True}})
        self.assertIn('box_io_factory', str(cm.exception))

    def test_from_flag_live_refuses_scripted_box_io(self):
        with self.assertRaises(ValueError) as cm:
            Controller.from_flag(
                {'pit': {'enabled': True, 'mode': 'live', 'cutover': True}},
                box_io_factory=ScriptedBoxIO,
            )
        self.assertIn('ScriptedBoxIO', str(cm.exception))

    def test_run_host_build_live_without_modules_raises(self):
        td = Path(tempfile.mkdtemp())
        flag = td / 'flag.json'
        flag.write_text(json.dumps({
            'parallel_eval': {'pit': {'enabled': True, 'mode': 'live', 'cutover': True}},
        }))
        store = DurableStore(root=td / 'pit')
        with self.assertRaises(RuntimeError) as cm:
            build_controller(
                parallel_eval={'pit': {'enabled': True, 'mode': 'live', 'cutover': True}},
                store=store,
                dry_run=True,
            )
        msg = str(cm.exception).lower()
        # Live wiring fails closed on the first missing seam (rent/lane/score/
        # box/verify/signer). Accept any of those so hermetic hosts without
        # RA modules still pass.
        self.assertTrue(
            any(tok in msg for tok in (
                'pe_rent', 'wiring', 'import', 'signer', 'delegated',
                'pe_lane', 'pe_score', 'verify', 'box_io',
            )),
            msg='unexpected live-wiring error: %s' % cm.exception,
        )

    def test_pe_score_require_host_refuses_embedded_green(self):
        with self.assertRaises(ScoreError) as cm:
            score(
                epoch=1,
                round_id='r',
                result={'score': {'decision': 'green', 'z': 1.5}},
                candidate_id='c' * 64,
                head_id=H0,
                require_host=True,
            )
        self.assertIn('embedded', str(cm.exception).lower())

    def test_live_decide_refuses_embedded_without_score_fn(self):
        from .apply_live import LiveApplicator
        from .pe_lane import LaneService
        from .pe_rent import RentService
        from .pe_score import ScoreError as PitScoreError
        from .state import ItemRecord, ItemState

        cfg = PitConfig(enabled=True, mode='live', max_boxes=2, cutover=True)
        ctl = Controller(
            config=cfg,
            rent=RentService(max_boxes=2, require_host=False),
            lane=LaneService(),
            box_io_factory=lambda: ScriptedBoxIO(),
            live_applicator=LiveApplicator(
                config=cfg,
                verify_uploads=lambda **k: {'candidate_ok': True, 'master_ok': True},
            ),
        )
        ctl.set_head(H0, epoch=1, master_digest=M0)
        item = ItemRecord(
            item_id='x', registration_index=1, epoch=10, uid=1, hotkey_prefix='hk',
            state=ItemState.SCORED, bound_head=H0, bound_master_digest=M0,
            candidate_id='c' * 64,
            meta={'probe_bundle': {
                'item_id': 'x', 'candidate_id': 'c' * 64,
                'bound_head': H0, 'bound_master_digest': M0,
                'probe_result': {'score': {'decision': 'green', 'z': 2.0}, 'partial': False, 'samples': 1, 'planned_samples': 1},
                'master_upload': {}, 'train_receipt': {}, 'box_public_key_hex': '',
            }},
        )
        with self.assertRaises(Exception) as cm:
            ctl._decide(item)
        self.assertIn('score_fn', str(cm.exception).lower())


if __name__ == '__main__':
    unittest.main()
