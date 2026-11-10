#!/usr/bin/env python3
"""lost_infra + green_waiting + apply_busy_check + lium unreachable + fakepass cap."""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, Optional
from unittest import mock

from .apply_live import LiveApplyResult
from .box import ScriptedBoxIO
from .config import PitConfig
from .controller import GREEN_APPLY_LOST_INFRA_S, Controller
from .cutover_ops import apply_busy_check
from .pe_lane import LaneService
from .pe_rent import RentService, is_fakepass_name, is_live_pit_name
from .state import ItemRecord, ItemState


H = 'a' * 64
C1 = 'b' * 64
C2 = 'c' * 64


def _decided(
    *,
    epoch: int,
    candidate: str,
    decision: str = 'green_fresh',
    box_id: Optional[str] = 'box-1',
    meta: Optional[Dict[str, Any]] = None,
) -> ItemRecord:
    rec = ItemRecord(
        item_id='item-%d' % epoch,
        epoch=epoch,
        uid=1,
        hotkey_prefix='5abc',
        registration_index=epoch,
        state=ItemState.DECIDED,
        bound_head=H,
        bound_master_digest='f' * 64,
        bound_round_id='r1',
        candidate_id=candidate,
        decision=decision,
        paid=decision.startswith('green') or decision == 'lost_infra',
        fresh=decision == 'green_fresh',
        result_stored=True,
        z=1.5,
        box_id=box_id,
        meta=dict(meta or {}),
    )
    rec.meta.setdefault('decided_at_mono', float(epoch))
    return rec


class TestLostInfraWaiting(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.pit = self.root / 'pit'
        self.pit.mkdir()
        self._prev = {k: os.environ.get(k) for k in ('PIT_STATE',)}
        os.environ['PIT_STATE'] = str(self.pit)
        self.cfg = PitConfig(enabled=True, mode='live', cutover=True, max_boxes=5)

    def tearDown(self):
        for k, v in self._prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.td.cleanup()

    def _ctl(self, items, *, pods=None):
        rent = RentService(max_boxes=5)
        if pods is not None:
            rent.list_pods = lambda: list(pods)
        return Controller(
            config=self.cfg,
            rent=rent,
            lane=LaneService(),
            box_io_factory=lambda: ScriptedBoxIO(),
            items=list(items),
            current_head=H,
            current_master_digest='f' * 64,
            now_fn=lambda: 1_000_000.0,
        )

    def test_waiting_then_promote_after_lost(self):
        leader = _decided(epoch=1073, candidate=C1, box_id=None, meta={
            'decided_at_mono': 10.0,
            'green_apply_since': 10.0,
        })
        waiter = _decided(epoch=1076, candidate=C2, decision='green_waiting', box_id=None, meta={
            'decided_at_mono': 20.0,
            'waiting_on_leader': leader.item_id,
            'live_apply': {
                'candidate_verified': True,
                'master_verified': True,
                'master_complete': True,
            },
        })
        ctl = self._ctl([leader, waiter], pods=[])
        ctl.now_fn = lambda: 10.0 + float(GREEN_APPLY_LOST_INFRA_S) + 1.0
        updated, _note = ctl.declare_lost_infra(leader, reason='test-timeout')
        self.assertEqual(updated.decision, 'lost_infra')
        self.assertTrue(updated.paid)
        promoted = [i for i in ctl.items if i.item_id == waiter.item_id][0]
        self.assertEqual(promoted.decision, 'green_fresh')
        self.assertTrue(promoted.meta.get('promoted_from_waiting'))

    def test_lium_unreachable_declares_lost(self):
        item = _decided(epoch=1, candidate=C1, meta={
            'decided_at_mono': 1.0,
            'green_apply_since': 1.0,
            'pod': {'id': 'box-1', 'name': 'affine-pe-pit-x'},
        })
        ctl = self._ctl([item], pods=[{
            'id': 'box-1', 'name': 'affine-pe-pit-x',
            'status': 'Node is not reachable',
        }])
        self.assertFalse(ctl._box_still_in_lium(item))
        advanced, _note = ctl._finish_decision(item)
        self.assertEqual(advanced.decision, 'lost_infra')

    def test_lium_unreachable_applies_from_upload(self):
        item = _decided(epoch=2, candidate=C1, meta={
            'decided_at_mono': 1.0,
            'candidate_master_digest': 'd' * 64,
            'candidate_sha256': C1,
            'live_apply': {
                'candidate_verified': True,
                'master_verified': True,
                'master_complete': True,
                'pe_apply_accepted': True,
            },
            'green_apply_pending': True,
            'pod': {'id': 'box-1'},
        })
        calls = []

        class FakeApp:
            def apply(self, it, scored=None):
                calls.append(it.item_id)
                return LiveApplyResult(
                    applied=True,
                    decision='green_fresh',
                    merge_blocked=False,
                    paid=True,
                    destroyed_box_state=False,
                    head_advanced=True,
                    pe_apply_accepted=True,
                    master_verified=True,
                    candidate_verified=True,
                    detail={'ok': True},
                )

        ctl = self._ctl([item], pods=[])
        ctl.live_applicator = FakeApp()
        with mock.patch('ops.parallel_eval.pit.controller.master_index_is_complete', return_value=True):
            with mock.patch('ops.parallel_eval.pit.controller.confirm_head_advanced', return_value=True):
                with mock.patch.object(ctl, '_candidate_upload_complete', return_value=True):
                    advanced, note = ctl._finish_decision(item)
        self.assertIn(note, (
            'green_fresh-applied', 'green-head-pending', 'decided', 'released',
        ))
        self.assertIsNone(advanced.box_id)

    def test_apply_busy_excludes_lost_and_dead_backoff(self):
        live = _decided(epoch=1, candidate=C1, meta={'green_apply_pending': True})
        lost = _decided(epoch=2, candidate=C2, decision='lost_infra', meta={'lost_infra': True})
        dead = _decided(epoch=3, candidate='1' * 64, box_id=None, meta={
            'apply_retry_at': time.time() + 3600,
            'green_apply_pending': True,
        })
        busy = apply_busy_check([live, lost, dead])
        self.assertEqual(len(busy), 1)
        self.assertIn('e1', busy[0])

    def test_fakepass_excluded_from_live_cap(self):
        self.assertTrue(is_fakepass_name('affine-pe-pit-fakepass-abc'))
        self.assertFalse(is_live_pit_name('affine-pe-pit-fakepass-abc'))
        self.assertTrue(is_live_pit_name('affine-pe-pit-live1'))

    def test_release_waiters_on_leader_apply(self):
        leader = _decided(epoch=1, candidate=C1, meta={'decided_at_mono': 1.0})
        waiter = _decided(epoch=2, candidate=C2, decision='green_waiting', meta={
            'decided_at_mono': 2.0, 'waiting_on_leader': leader.item_id,
        })
        ctl = self._ctl([leader, waiter])
        ctl._release_waiters_as_stale(leader)
        w = [i for i in ctl.items if i.item_id == waiter.item_id][0]
        self.assertEqual(w.decision, 'green_stale')


class TestCancelRequest(unittest.TestCase):
    def test_write_and_consume_cancel(self):
        from .durable import DurableStore
        from .rerun import write_cancel_request, consume_pending_cancels, RERUN_PRIORITY

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pit = root / 'pit'
            pit.mkdir()
            os.environ['PIT_STATE'] = str(pit)
            store = DurableStore(pit)
            item = ItemRecord(
                item_id='rerun-1', epoch=42, uid=1, hotkey_prefix='5x',
                registration_index=1, state=ItemState.QUEUED,
                priority=RERUN_PRIORITY,
                meta={'rerun': True},
            )
            store.save_item(item)
            doc = write_cancel_request(
                epoch=42, operator='jacob', reason='test-cancel', store_root=pit,
            )
            self.assertEqual(doc['status'], 'requested')
            cfg = PitConfig(enabled=True, mode='live', cutover=True, max_boxes=5)
            ctl = Controller(
                config=cfg,
                rent=RentService(max_boxes=5),
                lane=LaneService(),
                box_io_factory=lambda: ScriptedBoxIO(),
                items=[item],
                current_head=H,
                current_master_digest='f' * 64,
            )
            out = consume_pending_cancels(store=store, ctl=ctl)
            self.assertTrue(out and out[0].get('ok'))
            self.assertEqual(ctl.items[0].state, ItemState.ABORTED)
            self.assertTrue((ctl.items[0].meta or {}).get('rerun_cancelled'))


if __name__ == '__main__':
    unittest.main()
