#!/usr/bin/env python3
"""Mid-eval re-adopt with real sparse train.done — never free_gpus/place_cand."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ops.parallel_eval.live import pe_box

from .box_job import (
    STATUS_RUNNING,
    BoxJobController,
    MemBoxJobStore,
    write_marker,
)


CK = 'c' * 64
H = 'a' * 64
M = 'b' * 64


class TestHostBoxIOSparseMidEval(unittest.TestCase):
    def test_sparse_train_done_with_live_eval_never_place_or_free(self):
        store = MemBoxJobStore(default_ticks=5)
        item_id = 'e1070-sparse'
        ctrl = BoxJobController(store=store, item_id=item_id, sleep_fn=lambda _s: None)
        ctrl.ensure_dir()
        write_marker(store, ctrl.job_dir, 'download', {'ok': True})
        # Exact real layout — only checkpoint, events, ok, rc_ok.
        write_marker(store, ctrl.job_dir, 'train', {
            'ok': True,
            'checkpoint': CK,
            'events': {'done': {'event': 'done', 'checkpoint': CK}},
            'rc_ok': True,
        })
        store.arm_launch(ticks=99, auto_marker=False, payload={})
        pid = store.launch_detached(
            ctrl.job_dir, stage='eval', command='setsid nohup pit-box-job-eval',
        )
        self.assertIn(pid, store.alive_pids)

        class FakePod:
            id = 'pod-sparse'

            def run(self, cmd, timeout=60, check=False, **kwargs):
                return ''

        places = []
        frees = []
        signals = []

        def boom_place(*a, **k):
            places.append(1)
            raise AssertionError('place_cand must not run mid-eval')

        def boom_free(*a, **k):
            frees.append(1)
            raise AssertionError('free_gpus must not run mid-eval')

        def tracking_kill(pids):
            signals.extend(int(p) for p in pids)
            store.alive_pids.discard(int(pids[0])) if pids else None

        store.kill_pids = tracking_kill  # type: ignore[method-assign]

        report_doc = {
            'optimizer': {'sgd': {'fp32_master': {
                'output_master_digest': 'd' * 64, 'source': 'disk',
            }}},
            'source_file_hashes': {'trainer': 'a' * 64, 'batch': 'b' * 64},
            'batching': {'tasks': 1, 'rollouts': 1},
            'tokens': {'prompt_tokens': 1, 'output_tokens': 1, 'loss_tokens': 1},
            'update_size': 1.0,
            'gradient_norm': 1.0,
            'engine': 'packed',
        }

        with tempfile.TemporaryDirectory() as td:
            env = {
                'PE_STATE': td,
                'PIT_UPLOAD_ROOT': str(Path(td) / 'uploads'),
                'PIT_FP32_MASTER_INDEX': str(Path(td) / 'fp32'),
            }
            Path(td, 'fp32').mkdir()
            epoch_dir = Path(td) / '1070'
            epoch_dir.mkdir()
            (epoch_dir / 'TRAINING-JOB.json').write_text(json.dumps({
                'payload': {'manifest': {'payload': {
                    'checkpoint': {'id': H, 'files': {'model.safetensors': 'f' * 64}},
                }}},
            }))
            (epoch_dir / 'TRAIN-DISPATCH-FF.json').write_text('{"ok":true}\n')
            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch(
                    'ops.parallel_eval.pit.box_job.controller_for_pod',
                    return_value=ctrl,
                ), mock.patch(
                    'ops.parallel_eval.live.pe_box.place_cand', side_effect=boom_place,
                ), mock.patch(
                    'ops.parallel_eval.live.pe_box.free_gpus', side_effect=boom_free,
                ), mock.patch(
                    'ops.parallel_eval.live.pe_box.place_and_pull',
                    side_effect=AssertionError('place_and_pull must not run'),
                ), mock.patch(
                    'ops.parallel_eval.live.pe_box.pull_report_only',
                    return_value={
                        'candidate_id': CK,
                        'candidate_path': 'box:/ckpt/' + CK,
                        'box_candidate_dir': '/ckpt/' + CK,
                        'candidate_files': {'model.safetensors': 'e' * 64},
                        'candidate_master_path': str(Path(td) / 'master'),
                        'candidate_master_digest': 'd' * 64,
                        'report': str(Path(td) / 'report.json'),
                        'report_doc': report_doc,
                        'descriptor': {'id': CK, 'files': {'model.safetensors': 'e' * 64}},
                    },
                ), mock.patch(
                    'ops.parallel_eval.live.pe_box._published_master',
                    return_value={'master_digest': M, 'bf16_sha256': H},
                ), mock.patch(
                    'ops.parallel_eval.live.pe_box._as_pod',
                    side_effect=lambda p: p,
                ), mock.patch(
                    'ops.parallel_eval.live.pe_box._detect_box_gpu',
                    return_value='B300',
                ):
                    io = pe_box.HostBoxIO()
                    out = io.train_v834e(
                        head_path='r2://head/%s' % H,
                        master_path='r2://master/%s:%s' % (H, M),
                        expected_master_digest=M,
                        batch_handle=str(epoch_dir),
                        item_id=item_id,
                        pod=FakePod(),
                        epoch=1070,
                        epoch_dir=epoch_dir,
                    )

        self.assertEqual(places, [], 'place_cand must not run')
        self.assertEqual(frees, [], 'free_gpus must not run')
        self.assertEqual(signals, [], 'eval process group must not be signalled')
        self.assertTrue(out.get('mid_eval_readopt'))
        self.assertEqual(out.get('candidate_id'), CK)
        self.assertIn(pid, store.alive_pids)
        st = ctrl.status()
        self.assertEqual(st.state, STATUS_RUNNING)
        self.assertEqual(st.next_stage, 'eval')


if __name__ == '__main__':
    unittest.main()
