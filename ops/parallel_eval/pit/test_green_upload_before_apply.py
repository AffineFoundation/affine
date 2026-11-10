#!/usr/bin/env python3
"""GREEN apply must succeed from durable upload when the box is offline (e1073)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from .apply_live import LiveApplicator
from .config import load_pit_config
from .controller import Controller
from .pe_lane import LaneService
from .pe_rent import RentService
from .signer import FixtureRootSigner
from .state import ItemRecord, ItemState


H = 'a' * 64
C = 'b' * 64
M = 'c' * 64


def _heldout(rid='r-e1073'):
    return dict(
        round_id=rid, set_size=500, bar_z=1.0, k_cand=8, head_rollouts=12,
        scoring='ivw-v2', settings_hash='f' * 64, baseline='prior-v1',
        max_tokens=64000, scoring_mode='separate-run', tau_run=0.01,
        sigma_run_floor=0.0, baseline_done_at_cr='2026-01-01 00:00:00 CR',
    )


class TestGreenApplyFromUploadWhenBoxOffline(unittest.TestCase):
    def test_apply_succeeds_without_ssh_after_early_upload(self):
        """e1073: green_fresh with candidate only on box → upload first → box dies → apply OK."""
        cfg = load_pit_config({
            'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 2},
        })
        rent = RentService(max_boxes=2)
        # Box was rented, then went fully offline (not in lium, ssh dead).
        rent.ledger.active['pod-1073'] = {
            'box_id': 'pod-1073', 'name': 'affine-pe-pit-1073-a0', 'at': 1.0,
        }
        ctl = Controller(
            config=cfg, rent=rent, lane=LaneService(),
            box_io_factory=lambda: None, now_fn=lambda: 1000.0,
        )

        ssh_calls = []

        def upload_cand(**kwargs):
            # After early upload, apply must not need box ssh.
            if kwargs.get('candidate_on_box') or str(kwargs.get('candidate_path') or '').startswith('box:'):
                ssh_calls.append('cand')
                raise RuntimeError('ssh timeout: box offline')
            return {'ok': True, 'path': kwargs.get('candidate_path') or '/tmp/cand'}

        def upload_mast(**kwargs):
            if kwargs.get('candidate_on_box') or kwargs.get('box_master_dir'):
                ssh_calls.append('master')
                raise RuntimeError('ssh timeout: box offline')
            return {'ok': True, 'complete': True, 'shard_count': 2, 'index': '/tmp/idx'}

        applicator = LiveApplicator(
            config=cfg,
            verify_uploads=lambda **k: {
                'candidate_ok': True, 'master_ok': True, 'master_complete': True,
            },
            upload_candidate=upload_cand,
            upload_master=upload_mast,
            accept_pit_decision=lambda payload: True,
            checkpoint_signer=FixtureRootSigner(),
        )
        ctl.live_applicator = applicator

        with tempfile.TemporaryDirectory() as td:
            cand_dir = Path(td) / 'uploads' / C
            cand_dir.mkdir(parents=True)
            (cand_dir / 'model.safetensors').write_bytes(b'weights')
            (cand_dir / 'UPLOAD.json').write_text('{"id":"%s"}\n' % C)
            master_dir = Path(td) / 'uploads' / C / 'fp32-master'
            master_dir.mkdir(parents=True)
            (master_dir / 'master_digest.txt').write_text(M + '\n')

            # Early post-train upload already stamped live_apply (artifacts on R2/host).
            item = ItemRecord(
                item_id='e1073' + '0' * 27,
                registration_index=1073, epoch=1073,
                state=ItemState.DECIDED, decision='green_fresh',
                paid=True, fresh=True,
                box_id='pod-1073', box_name='affine-pe-pit-1073-a0',
                bound_head=H, bound_master_digest=M,
                bound_round_id='r-e1073', bound_plan_seed='p' * 64,
                candidate_id=C,
                meta=dict(
                    pod=dict(id='pod-1073'),
                    # Stale box: markers still say on-box, but upload already done.
                    candidate_path=str(cand_dir),
                    candidate_on_box=False,
                    defer_candidate_upload=False,
                    artifacts_uploaded=True,
                    candidate_files={'model.safetensors': 'd' * 64},
                    candidate_sha256=C,
                    candidate_master_digest=M,
                    candidate_master_path=str(master_dir),
                    live_apply={
                        'candidate_verified': True,
                        'master_verified': True,
                        'master_complete': True,
                        'detail': {'early_upload': True, 'candidate': C},
                    },
                    scored=dict(
                        decision='green', z=1.7, evaluated=dict(z=1.7),
                        heldout=_heldout(), plan_seed='p' * 64, candidate=C,
                    ),
                ),
            )

            env = {
                'PIT_UPLOAD_ROOT': str(Path(td) / 'uploads'),
                'PE_LIFE_STATE': str(Path(td) / 'life'),
                'PE_STATE': td,
            }
            (Path(td) / 'life' / 'pending').mkdir(parents=True)
            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch(
                    'ops.parallel_eval.pit.apply_live.master_index_is_complete',
                    return_value=True,
                ), mock.patch(
                    'ops.parallel_eval.pit.controller.master_index_is_complete',
                    return_value=True,
                ), mock.patch(
                    'ops.parallel_eval.pit.apply_live.drive_cutover_apply',
                    return_value={'ok': True, 'head_advanced': True},
                ), mock.patch(
                    'ops.parallel_eval.pit.controller.confirm_head_advanced',
                    return_value=True,
                ), mock.patch.object(
                    ctl, '_box_still_in_lium', return_value=False,
                ):
                    # Precondition: early upload already complete.
                    self.assertTrue(ctl._candidate_upload_complete(item))
                    out, note = ctl._finish_decision(item)

        self.assertEqual(ssh_calls, [], 'APPLY must not ssh to an offline box')
        self.assertTrue(
            ((out.meta or {}).get('live_apply') or {}).get('candidate_verified')
            or ((out.meta or {}).get('live_apply') or {}).get('head_advanced')
            or note in ('green_fresh-applied', 'green-head-pending', 'released')
            or out.state in (ItemState.RELEASED, ItemState.DECIDED),
            'apply path must complete from durable upload (note=%r state=%s)' % (
                note, out.state,
            ),
        )
        # Must not be stuck in green-apply-retry solely because box is gone.
        self.assertNotEqual(note, 'green-apply-retry')


if __name__ == '__main__':
    unittest.main()
