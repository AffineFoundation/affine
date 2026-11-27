#!/usr/bin/env python3
"""Failed re-adopt SSH retries before release when the box is still in lium."""
from __future__ import annotations

import unittest

from .config import PitConfig
from .controller import READOPT_MAX_ATTEMPTS, Controller
from .pe_lane import LaneService
from .pe_rent import RentService
from .state import ItemRecord, ItemState


H0, M0 = 'a' * 64, 'b' * 64


class TestReadoptRetry(unittest.TestCase):
    def _item(self) -> ItemRecord:
        return ItemRecord(
            item_id='e1056' + '0' * 27,
            registration_index=1,
            epoch=1056,
            state=ItemState.TRAINING,
            bound_head=H0,
            bound_master_digest=M0,
            bound_round_id='r1',
            bound_plan_seed='p' * 64,
            box_id='pod-1056',
            box_name='l1056-a0',
            attempt=0,
            meta={'pod': {'id': 'pod-1056', 'name': 'l1056-a0'}},
        )

    def _ctl(self, *, list_pods=None) -> Controller:
        rent = RentService(max_boxes=2)
        rent.ledger.register('pod-1056', item_id='e1056', name='l1056-a0', at=1.0)
        if list_pods is not None:
            rent.list_pods = list_pods
        return Controller(
            config=PitConfig(enabled=True, mode='shadow', max_boxes=2),
            rent=rent,
            lane=LaneService(),
            box_io_factory=lambda: None,
            round_id_fn=lambda e: 'r1',
            plan_seed_fn=lambda e: 'p' * 64,
        )

    def test_ssh_fail_retries_while_box_listed(self):
        listed = [{'id': 'pod-1056', 'name': 'l1056-a0'}]
        ctl = self._ctl(list_pods=lambda: listed)
        item = self._item()
        ctl.items = [item]

        updated, note = ctl._apply_worker_result(
            item, {'ok': False, 'error': 'ssh rc=255 connection reset'},
        )
        self.assertEqual(note, 'readopt-retry')
        self.assertEqual(updated.state, ItemState.PROVISIONING)
        self.assertEqual(updated.box_id, 'pod-1056')
        self.assertEqual(updated.meta.get('readopt_attempts'), 1)
        self.assertIn('pod-1056', ctl.rent.ledger.active)

    def test_releases_after_max_readopt_attempts(self):
        ctl = self._ctl(list_pods=lambda: [{'id': 'pod-1056', 'name': 'l1056-a0'}])
        item = self._item()
        item.meta['readopt_attempts'] = READOPT_MAX_ATTEMPTS
        ctl.items = [item]

        updated, note = ctl._apply_worker_result(
            item, {'ok': False, 'error': 'ssh rc=255'},
        )
        self.assertEqual(note, 'box-worker-fail')
        self.assertNotEqual(updated.state, ItemState.PROVISIONING)
        self.assertNotIn('pod-1056', ctl.rent.ledger.active)

    def test_releases_immediately_when_box_gone_from_lium(self):
        ctl = self._ctl(list_pods=lambda: [])
        item = self._item()
        ctl.items = [item]

        updated, note = ctl._apply_worker_result(
            item, {'ok': False, 'error': 'ssh rc=255'},
        )
        self.assertEqual(note, 'box-worker-fail')
        self.assertNotIn('pod-1056', ctl.rent.ledger.active)

    def test_non_transport_error_does_not_retry(self):
        ctl = self._ctl(list_pods=lambda: [{'id': 'pod-1056', 'name': 'l1056-a0'}])
        item = self._item()
        ctl.items = [item]

        updated, note = ctl._apply_worker_result(
            item, {'ok': False, 'error': 'Fp32MasterError: digest mismatch'},
        )
        self.assertEqual(note, 'box-worker-fail')
        self.assertIsNone((updated.meta or {}).get('readopt_attempts'))


if __name__ == '__main__':
    unittest.main()
