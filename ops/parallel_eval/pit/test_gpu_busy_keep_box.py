"""GPU-busy during re-adopt must keep the box (never release)."""
from __future__ import annotations

import unittest

from .config import load_pit_config
from .controller import Controller
from .pe_lane import LaneService
from .pe_rent import RentService
from .state import ItemRecord, ItemState


class TestGpuBusyKeepBox(unittest.TestCase):
    def test_gpu_busy_readopt_keeps_box(self):
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
        )
        item = ItemRecord(
            item_id='e1065', registration_index=1, epoch=1065,
            state=ItemState.TRAINING, bound_head='a' * 64,
            candidate_id='c' * 64, box_id='pod-1065',
            box_name='affine-pe-pit-1065-a0',
            meta={'pod': {'id': 'pod-1065'}, 'readopt_attempts': 0},
        )
        ctl.items.append(item)
        ctl.rent.list_pods = lambda: [{'id': 'pod-1065', 'name': 'affine-pe-pit-1065-a0'}]

        done = {'ok': False, 'error': 'GPUs not free after train: [8192, 8192] MiB'}
        updated, note = ctl._apply_worker_result(item, done)
        self.assertEqual(note, 'readopt-retry')
        self.assertEqual(updated.box_id, 'pod-1065')
        self.assertIn('pod-1065', ctl.rent.ledger.active)
        self.assertEqual(updated.state, ItemState.TRAINING)
        self.assertIn('gpus not free', (updated.meta or {}).get('last_readopt_error', '').lower())


if __name__ == '__main__':
    unittest.main()
