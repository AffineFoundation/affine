"""After train.done, box-worker failure must upload C before releasing the box."""
from __future__ import annotations

import json
import unittest
from unittest import mock

from .box_job import MemBoxJobStore, write_marker
from .config import load_pit_config
from .controller import Controller
from .pe_lane import LaneService
from .pe_rent import RentService
from .state import ItemRecord, ItemState


CK = 'c' * 64
MD = 'd' * 64
H = 'a' * 64


class TestPostTrainUploadOnFail(unittest.TestCase):
    def _ctl(self, *, upload_fn=None):
        cfg = load_pit_config({
            'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 2},
        })
        rent = RentService(max_boxes=2)
        rent.ledger.active['pod-1067'] = {
            'box_id': 'pod-1067', 'name': 'affine-pe-pit-1067-a0', 'at': 1.0,
        }
        ctl = Controller(
            config=cfg, rent=rent, lane=LaneService(),
            box_io_factory=lambda: None, now_fn=lambda: 1000.0,
        )
        if upload_fn is not None:
            class App:
                def upload_candidate_artifacts(self, item):
                    return upload_fn(item)
            ctl.live_applicator = App()
        return ctl

    def test_uploads_then_releases_when_train_done(self):
        uploads = []

        def upload(item):
            uploads.append(item.candidate_id)
            return {
                'candidate_verified': True,
                'master_verified': True,
                'master_complete': True,
                'detail': {'candidate': item.candidate_id},
            }

        ctl = self._ctl(upload_fn=upload)
        store = MemBoxJobStore()
        item = ItemRecord(
            item_id='e1067', registration_index=1067, epoch=1067,
            state=ItemState.TRAINING, bound_head=H,
            candidate_id=CK, sealed=True, box_id='pod-1067',
            box_name='affine-pe-pit-1067-a0', attempt=0,
            meta={
                'pod': {'id': 'pod-1067'},
                'candidate_path': 'box:/ckpt/' + CK,
                'candidate_master_path': '/tmp/master',
                'candidate_master_digest': MD,
                'candidate_sha256': CK,
                'defer_candidate_upload': True,
                'candidate_on_box': True,
            },
        )
        # train.done on box (controller also trusts sealed+candidate_id).
        from .box_job import BoxJobController, marker_path
        ctrl = BoxJobController(store=store, item_id=item.item_id, sleep_fn=lambda _s: None)
        ctrl.ensure_dir()
        write_marker(store, ctrl.job_dir, 'train', {
            'ok': True, 'checkpoint': CK, 'candidate_id': CK,
            'candidate_path': 'box:/ckpt/' + CK,
            'candidate_master_path': '/tmp/master',
            'candidate_master_digest': MD,
            'candidate_sha256': CK,
        })

        with mock.patch(
            'ops.parallel_eval.pit.box_job.controller_for_pod',
            return_value=ctrl,
        ), mock.patch.object(
            ctl, '_candidate_upload_complete',
            side_effect=lambda it: bool(
                ((it.meta or {}).get('live_apply') or {}).get('candidate_verified')
            ),
        ), mock.patch.object(
            ctl, '_attach_box_job_log_tail', side_effect=lambda it: it,
        ):
            out = ctl._fail_and_release(
                item, 'box-worker:detached stage eval exited without marker',
            )

        self.assertEqual(uploads, [CK])
        self.assertIsNone(out.box_id)
        self.assertNotIn('pod-1067', ctl.rent.ledger.active)
        self.assertTrue(((out.meta or {}).get('live_apply') or {}).get('candidate_verified'))

    def test_keeps_box_when_upload_fails(self):
        def boom(item):
            raise RuntimeError('upload boom')

        ctl = self._ctl(upload_fn=boom)
        item = ItemRecord(
            item_id='e1067b', registration_index=1067, epoch=1067,
            state=ItemState.TRAINING, bound_head=H,
            candidate_id=CK, sealed=True, box_id='pod-1067',
            box_name='affine-pe-pit-1067-a0', attempt=0,
            meta={
                'pod': {'id': 'pod-1067'},
                'candidate_path': 'box:/ckpt/' + CK,
                'candidate_master_path': '/tmp/master',
                'candidate_master_digest': MD,
                'candidate_sha256': CK,
                'defer_candidate_upload': True,
            },
        )
        with mock.patch.object(
            ctl, '_attach_box_job_log_tail', side_effect=lambda it: it,
        ), mock.patch.object(
            ctl, '_candidate_upload_complete', return_value=False,
        ), mock.patch.object(
            ctl, '_train_done_on_box', return_value=True,
        ):
            out = ctl._fail_and_release(item, 'box-worker:eval boom')
        self.assertEqual(out.box_id, 'pod-1067')
        self.assertIn('pod-1067', ctl.rent.ledger.active)
        self.assertIn('upload boom', (out.meta or {}).get('post_train_upload_error') or '')


if __name__ == '__main__':
    unittest.main()
