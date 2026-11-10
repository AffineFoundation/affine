"""GPU-busy during re-adopt: keep ≤15 min, kill+retry once, then alert+release."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from .config import load_pit_config
from .controller import GPU_BUSY_KEEP_S, Controller
from .pe_lane import LaneService
from .pe_rent import RentService
from .state import ItemRecord, ItemState


class TestGpuBusyKeepBox(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        pe = Path(self.td) / 'pe'
        (pe / 'pit').mkdir(parents=True)
        self.prev = {k: os.environ.get(k) for k in ('PE_STATE', 'PIT_STATE')}
        os.environ['PE_STATE'] = str(pe)
        os.environ['PIT_STATE'] = str(pe / 'pit')
        self.clock = {'t': 1000.0}

    def tearDown(self):
        for k, v in self.prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _ctl(self):
        cfg = load_pit_config({
            'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 2},
        })
        rent = RentService(max_boxes=2)
        rent.ledger.active['pod-1065'] = {
            'box_id': 'pod-1065', 'name': 'affine-pe-pit-1065-a0', 'at': 1.0,
        }
        ctl = Controller(
            config=cfg,
            rent=rent,
            lane=LaneService(),
            box_io_factory=lambda: None,
            now_fn=lambda: self.clock['t'],
        )
        ctl.rent.list_pods = lambda: [{'id': 'pod-1065', 'name': 'affine-pe-pit-1065-a0'}]
        return ctl

    def _item(self):
        return ItemRecord(
            item_id='e1065', registration_index=1, epoch=1065,
            state=ItemState.TRAINING, bound_head='a' * 64,
            candidate_id='c' * 64, box_id='pod-1065',
            box_name='affine-pe-pit-1065-a0',
            meta={'pod': {'id': 'pod-1065'}, 'readopt_attempts': 0},
        )

    def test_gpu_busy_keeps_within_cap(self):
        ctl = self._ctl()
        item = self._item()
        ctl.items.append(item)
        done = {'ok': False, 'error': 'GPUs not free after train: [8192, 8192] MiB'}
        updated, note = ctl._apply_worker_result(item, done)
        self.assertEqual(note, 'readopt-gpu-busy-keep')
        self.assertEqual(updated.box_id, 'pod-1065')
        self.assertIn('pod-1065', ctl.rent.ledger.active)

    def test_gpu_busy_kill_retry_then_release(self):
        ctl = self._ctl()
        item = self._item()
        ctl.items.append(item)
        done = {'ok': False, 'error': 'GPUs not free after train: [8192] MiB'}

        # First hit: start keep window.
        updated, note = ctl._apply_worker_result(item, done)
        self.assertEqual(note, 'readopt-gpu-busy-keep')
        item = updated

        # Past 15 min: kill+retry once.
        self.clock['t'] += float(GPU_BUSY_KEEP_S) + 1.0
        with mock.patch.object(ctl, '_kill_box_job_pids', return_value=[42]):
            updated, note = ctl._apply_worker_result(item, done)
        self.assertEqual(note, 'readopt-gpu-busy-kill-retry')
        self.assertEqual(updated.box_id, 'pod-1065')
        self.assertEqual((updated.meta or {}).get('gpu_busy_kill_retries'), 1)
        item = updated

        # Still busy after kill+retry window → alert + release.
        self.clock['t'] += float(GPU_BUSY_KEEP_S) + 1.0
        with mock.patch.object(ctl, '_kill_box_job_pids', return_value=[]):
            updated, note = ctl._apply_worker_result(item, done)
        self.assertEqual(note, 'box-worker-fail')
        self.assertIsNone(updated.box_id)
        self.assertTrue((Path(os.environ['PIT_STATE']) / 'GPU-BUSY-ALERT.json').is_file())


if __name__ == '__main__':
    unittest.main()
