"""Persisted meta['pod'] dict (no .run) must still drive detached box jobs.

Production writes a plain rent dict into the item record / pods.json. Gating on
hasattr(pod, 'run') made detached eval inert (host-side serve), disabled RESULT
resume, and made _kill_box_job_pids a no-op. These tests use that real shape.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest import mock

from .box import BoxRunner, ScriptedBoxIO, _try_box_job_controller, normalize_meta_pod
from .box_job import (
    BoxJobController,
    MemBoxJobStore,
    log_path,
    write_result,
)
from .config import load_pit_config
from .controller import Controller
from .pe_lane import LaneService
from .pe_rent import RentService
from .state import ItemRecord, ItemState


def persisted_pod_dict(**overrides):
    """Shape written by Controller after rent (and into pods.json)."""
    doc = {
        'id': 'pod-1065',
        'box_id': 'pod-1065',
        'name': 'affine-pe-pit-1065-a0',
        'ip': '10.20.30.40',
        'port': 2222,
        'huid': 'swift-test-aa',
        'gpu': 'B300',
        'price_h': 48.0,
        'price_usd_h': 48.0,
        'ssh_command': 'ssh root@10.20.30.40 -p 2222',
    }
    doc.update(overrides)
    return doc


def _pod_like_from_dict(d, *, store: Optional[MemBoxJobStore] = None):
    """Stand-in for pe_common.pod_of — has .run, never a fake stuffed into meta."""
    assert isinstance(d, dict), 'normalize must receive the persisted dict'
    assert 'ip' in d and 'port' in d and 'id' in d
    pod = SimpleNamespace(
        id=d['id'], ip=d['ip'], port=d['port'],
        huid=d.get('huid'), name=d.get('name'),
        _store=store, _raw=d,
    )

    def _run(cmd, timeout=60, check=False, **kwargs):
        return ''

    pod.run = _run
    return pod


class TestNormalizeMetaPod(unittest.TestCase):
    def test_normalize_calls_as_pod_on_dict(self):
        d = persisted_pod_dict()
        fake = _pod_like_from_dict(d)
        with mock.patch('ops.parallel_eval.pit.box._as_pod', return_value=fake) as as_pod:
            out = normalize_meta_pod(d)
        as_pod.assert_called_once_with(d)
        self.assertIs(out, fake)
        self.assertTrue(callable(out.run))
        self.assertFalse(hasattr(d, 'run'))


class TestPersistedPodBoxJob(unittest.TestCase):
    def setUp(self):
        self.store = MemBoxJobStore()
        self.item_id = 'e1065'
        self.pod_dict = persisted_pod_dict()
        self.pod_like = _pod_like_from_dict(self.pod_dict, store=self.store)

    def _patch_normalize(self):
        return mock.patch(
            'ops.parallel_eval.pit.box._as_pod',
            side_effect=lambda p: _pod_like_from_dict(p, store=self.store),
        )

    def _mem_controller(self, pod, item_id, **kwargs):
        # RemoteBoxJobStore always exposes .pod — Mem must too for detached arming.
        self.store.pod = pod
        clock = {'t': 0.0}

        def _now():
            clock['t'] += 1.0
            return clock['t']

        return BoxJobController(
            store=self.store, item_id=item_id,
            sleep_fn=lambda _s: self.store.advance(),
            now_fn=_now,
        )

    def test_try_box_job_controller_accepts_persisted_dict(self):
        with self._patch_normalize():
            with mock.patch(
                'ops.parallel_eval.pit.box_job.controller_for_pod',
                side_effect=self._mem_controller,
            ):
                ctrl = _try_box_job_controller(self.pod_dict, self.item_id)
        self.assertIsNotNone(ctrl)
        self.assertTrue(hasattr(ctrl.store, 'pod'))
        self.assertFalse(hasattr(self.pod_dict, 'run'))

    def test_detached_eval_taken_with_persisted_pod_dict(self):
        """BoxRunner must arm detached eval (eval-plan.json), not host-side serve."""
        probe = {
            'k': 8, 'samples': 16, 'score': {'decision': 'red', 'z': -1.0},
            'eval_gpu': 'B300', 'partial': False, 'planned_samples': 16, 'dtype': 'bf16',
        }
        self.store.arm_launch(
            ticks=1,
            auto_marker=True,
            payload={
                'ok': True, 'status': 'ok', 'candidate_id': 'c' * 64,
                'probe_result': probe,
            },
        )
        host_serve = {'n': 0}

        def _boot(pod, **kw):
            return {'ok': True}

        def _serve(pod, cid, **kw):
            host_serve['n'] += 1
            raise AssertionError('host must not serve when detached eval is armed')

        def _probe(*a, **k):
            raise AssertionError('host must not run_probe when detached eval is armed')

        # LaneService is a dataclass — set require_host via constructor, not subclass attr.
        host_lane = LaneService(
            require_host=True,
            host_boot=_boot,
            host_serve=_serve,
            host_run_probe=_probe,
        )

        def _ctrl_for_pod(pod, item_id, **kwargs):
            return self._mem_controller(pod, item_id, **kwargs)

        runner = BoxRunner(
            config=load_pit_config({
                'pit': {'enabled': True, 'mode': 'live', 'cutover': True},
            }),
            io=ScriptedBoxIO(),
            lane=host_lane,
        )
        record = ItemRecord(
            item_id=self.item_id, registration_index=1, epoch=1065,
            state=ItemState.PROVISIONING,
            bound_head='a' * 64, bound_master_digest='b' * 64,
            box_id='pod-1065', box_name='affine-pe-pit-1065-a0',
            # Real persisted shape — NOT an object with .run
            meta={'pod': dict(self.pod_dict), 'train_gpu': 'B300'},
        )

        with self._patch_normalize():
            with mock.patch(
                'ops.parallel_eval.pit.box_job.controller_for_pod',
                side_effect=_ctrl_for_pod,
            ):
                rec2, bundle = runner.run(
                    record,
                    batch_handle='opaque-batch',
                    plan_handle='opaque-plan',
                    indices=[1, 2, 3, 4],
                )

        self.assertEqual(host_serve['n'], 0)
        self.assertEqual(rec2.state, ItemState.RESULT)
        self.assertEqual(bundle.candidate_id, 'c' * 64)
        # Detached arm wrote eval-plan.json onto the box job store.
        plan_files = [p for p in self.store.files if p.endswith('eval-plan.json')]
        self.assertTrue(plan_files, 'detached eval must write eval-plan.json on the box')
        plan = json.loads(self.store.files[plan_files[0]])
        self.assertEqual(plan['checkpoint_id'], 'c' * 64)
        self.assertEqual(plan['envs'][0]['indices'], [1, 2, 3, 4])
        # Detached command was launched (not host sync serve).
        self.assertTrue(any(e.get('stage') == 'eval' for e in self.store.launch_log))
        self.assertFalse(hasattr(record.meta['pod'], 'run'))

    def test_result_resume_with_persisted_pod_dict(self):
        prior = {
            'ok': True,
            'candidate_id': 'c' * 64,
            'candidate_path': 'box:/ckpt/' + ('c' * 64),
            'candidate_master_path': '/tmp/master',
            'probe_result': {
                'k': 8, 'samples': 16, 'score': {'decision': 'green', 'z': 2.0},
                'eval_gpu': 'B300', 'partial': False, 'planned_samples': 16, 'dtype': 'bf16',
            },
            'train_receipt': {
                'candidate_id': 'c' * 64,
                'candidate_path': 'box:/ckpt/' + ('c' * 64),
                'candidate_master_path': '/tmp/master',
                'train_gpu': 'B300',
                'training_report_summary': {
                    'source_file_hashes': {'trainer': 'a' * 64, 'batch': 'b' * 64},
                    'batching': {'tasks': 1, 'rollouts': 1},
                    'tokens': {'prompt_tokens': 1, 'output_tokens': 1, 'loss_tokens': 1},
                    'update_size': 1.0, 'gradient_norm': 1.0, 'engine': 'packed',
                },
            },
            'training_report': {
                'source_file_hashes': {'trainer': 'a' * 64, 'batch': 'b' * 64},
                'batching': {'tasks': 1, 'rollouts': 1},
                'tokens': {'prompt_tokens': 1, 'output_tokens': 1, 'loss_tokens': 1},
                'update_size': 1.0, 'gradient_norm': 1.0, 'engine': 'packed',
            },
            'bound_head': 'a' * 64,
            'bound_master_digest': 'b' * 64,
            'box_public_key_hex': 'ab' * 32,
            'train_gpu': 'B300',
            'eval_gpu': 'B300',
        }
        trains = {'n': 0}

        class CountingIO(ScriptedBoxIO):
            def train_v834e(self, **kwargs):
                trains['n'] += 1
                return super().train_v834e(**kwargs)

        def _ctrl_for_pod(pod, item_id, **kwargs):
            c = self._mem_controller(pod, item_id, **kwargs)
            c.ensure_dir()
            write_result(self.store, c.job_dir, prior)
            return c

        runner = BoxRunner(
            config=load_pit_config({'pit': {'enabled': True, 'mode': 'shadow'}}),
            io=CountingIO(),
            lane=LaneService(),
        )
        record = ItemRecord(
            item_id=self.item_id, registration_index=1, epoch=1065,
            state=ItemState.PROVISIONING,
            bound_head='a' * 64, bound_master_digest='b' * 64,
            box_id='pod-1065',
            meta={'pod': dict(self.pod_dict), 'train_gpu': 'B300'},
        )
        with self._patch_normalize():
            with mock.patch(
                'ops.parallel_eval.pit.box_job.controller_for_pod',
                side_effect=_ctrl_for_pod,
            ):
                rec2, bundle = runner.run(
                    record, batch_handle='opaque-batch', plan_handle='opaque-plan',
                )
        self.assertEqual(trains['n'], 0)
        self.assertEqual(rec2.state, ItemState.RESULT)
        self.assertEqual(bundle.candidate_id, 'c' * 64)
        self.assertTrue((bundle.meta or {}).get('readopt_result'))

    def test_kill_box_job_pids_with_persisted_pod_dict(self):
        td = tempfile.mkdtemp()
        pe = Path(td) / 'pe'
        (pe / 'pit').mkdir(parents=True)
        prev = {k: os.environ.get(k) for k in ('PE_STATE', 'PIT_STATE')}
        os.environ['PE_STATE'] = str(pe)
        os.environ['PIT_STATE'] = str(pe / 'pit')
        try:
            cfg = load_pit_config({
                'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 2},
            })
            rent = RentService(max_boxes=2)
            ctl = Controller(
                config=cfg, rent=rent, lane=LaneService(),
                box_io_factory=lambda: None, now_fn=lambda: 1000.0,
            )
            # Seed a live job pid the kill path must reach through the dict pod.
            def _ctrl_for_pod(pod, item_id, **kwargs):
                c = self._mem_controller(pod, item_id, **kwargs)
                c.ensure_dir()
                self.store.arm_launch(ticks=99, auto_marker=False)
                pid = self.store.launch_detached(
                    c.job_dir, stage='train', command='train_pit.py',
                )
                self.assertIn(pid, self.store.alive_pids)
                return c

            item = ItemRecord(
                item_id=self.item_id, registration_index=1, epoch=1065,
                state=ItemState.TRAINING, bound_head='a' * 64,
                candidate_id='c' * 64, box_id='pod-1065',
                box_name='affine-pe-pit-1065-a0',
                meta={'pod': dict(self.pod_dict)},
            )
            with self._patch_normalize():
                with mock.patch(
                    'ops.parallel_eval.pit.box_job.controller_for_pod',
                    side_effect=_ctrl_for_pod,
                ):
                    killed = ctl._kill_box_job_pids(item)
            self.assertTrue(killed, 'kill must reach recorded PIDs via persisted dict')
            self.assertFalse(self.store.alive_pids)
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_fail_and_release_copies_job_log_tail(self):
        td = tempfile.mkdtemp()
        pe = Path(td) / 'pe'
        (pe / 'pit').mkdir(parents=True)
        prev = {k: os.environ.get(k) for k in ('PE_STATE', 'PIT_STATE')}
        os.environ['PE_STATE'] = str(pe)
        os.environ['PIT_STATE'] = str(pe / 'pit')
        try:
            cfg = load_pit_config({
                'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 2},
            })
            rent = RentService(max_boxes=2)
            rent.ledger.active['pod-1065'] = {
                'box_id': 'pod-1065', 'name': 'affine-pe-pit-1065-a0', 'at': 1.0,
            }
            ctl = Controller(
                config=cfg, rent=rent, lane=LaneService(),
                box_io_factory=lambda: None, now_fn=lambda: 1000.0,
            )

            lines = ['train line %d' % i for i in range(250)]
            log_body = '\n'.join(lines) + '\n'

            def _ctrl_for_pod(pod, item_id, **kwargs):
                c = self._mem_controller(pod, item_id, **kwargs)
                c.ensure_dir()
                self.store.write_text(log_path(c.job_dir), log_body)
                return c

            item = ItemRecord(
                item_id=self.item_id, registration_index=1, epoch=1066,
                state=ItemState.TRAINING, bound_head='a' * 64,
                candidate_id='c' * 64, box_id='pod-1065',
                box_name='affine-pe-pit-1066-a0', attempt=0,
                meta={'pod': dict(self.pod_dict)},
            )
            with self._patch_normalize():
                with mock.patch(
                    'ops.parallel_eval.pit.box_job.controller_for_pod',
                    side_effect=_ctrl_for_pod,
                ):
                    out = ctl._fail_and_release(
                        item, 'box-worker:detached stage train exited without marker',
                    )
            tail = (out.meta or {}).get('box_job_log_tail') or ''
            self.assertTrue(tail, 'job.log tail must be stamped before release')
            self.assertIn('train line 249', tail)
            self.assertNotIn('train line 0', tail)  # only last ~200 lines
            self.assertIsNone(out.box_id)
            self.assertNotIn('pod-1065', ctl.rent.ledger.active)
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == '__main__':
    unittest.main()
