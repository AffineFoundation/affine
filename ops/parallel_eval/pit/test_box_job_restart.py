"""Restart-safe box jobs: resume from markers / RESULT; never re-run completed stages."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from .box import BoxRunner, ScriptedBoxIO, _bundle_from_box_result
from .box_job import (
    STATUS_DONE,
    STATUS_NEED_LAUNCH,
    STATUS_RUNNING,
    BoxJobController,
    MemBoxJobStore,
    inspect_job,
    job_dir_for,
    kill_stale_job,
    marker_path,
    write_marker,
    write_result,
)
from .config import PitConfig
from .pe_lane import LaneService
from .signer import generate_ephemeral_box_key
from .state import ItemRecord, ItemState


def _ctrl(store: MemBoxJobStore, item_id: str = 'item-restart-1') -> BoxJobController:
    return BoxJobController(
        store=store,
        item_id=item_id,
        job_root='/workspace/pst/pit-jobs',
        sleep_fn=lambda _s: None,
        now_fn=lambda: 0.0,
    )


class TestBoxJobRestart(unittest.TestCase):
    def test_restart_during_download_reruns_only_download_after_kill_stale(self):
        store = MemBoxJobStore(default_ticks=2)
        ctrl = _ctrl(store)
        ctrl.ensure_dir()
        jd = ctrl.job_dir

        # Simulate a host crash mid-download: a stale gate_fetch-like job is alive,
        # no download.done marker yet.
        store.arm_launch(ticks=99, auto_marker=False, payload={'stale': True})
        stale_pid = store.launch_detached(jd, stage='download', command='gate_fetch.py head')
        self.assertIn(stale_pid, store.alive_pids)
        st = ctrl.status()
        self.assertEqual(st.state, STATUS_RUNNING)
        self.assertEqual(st.next_stage, 'download')
        self.assertNotIn('download', st.markers)

        launches = {'n': 0}

        def download_runner():
            launches['n'] += 1
            return {'ok': True, 'phase': 'download'}

        # Re-adopted host: kill stale, re-run download only, write marker.
        out = ctrl.run_stage_once('download', download_runner)
        self.assertEqual(out.get('phase'), 'download')
        self.assertEqual(launches['n'], 1)
        self.assertNotIn(stale_pid, store.alive_pids)
        self.assertIn('download', ctrl.status().markers)

        # Second host pass must NOT re-run download.
        out2 = ctrl.run_stage_once('download', download_runner)
        self.assertEqual(launches['n'], 1)
        self.assertEqual(out2.get('phase'), 'download')

    def test_restart_during_train_resumes_polling_without_relaunch(self):
        store = MemBoxJobStore(default_ticks=3)
        ctrl = _ctrl(store, item_id='item-train-mid')
        ctrl.ensure_dir()
        write_marker(store, ctrl.job_dir, 'download', {'ok': True})

        store.arm_launch(
            ticks=3,
            auto_marker=True,
            payload={'ok': True, 'checkpoint': 'c' * 64},
        )
        # First host: launch detached train, then "crash" before marker appears.
        pid = store.launch_detached(
            ctrl.job_dir, stage='train', command='setsid nohup train_pit.py',
        )
        self.assertIn(pid, store.alive_pids)
        st = inspect_job(store, ctrl.job_dir)
        self.assertEqual(st.state, STATUS_RUNNING)
        self.assertEqual(st.next_stage, 'train')
        self.assertNotIn('train', st.markers)
        launches_before = len(store.launch_log)

        # Re-adopted host: run_stage_once sees running train, polls until marker.
        clock = {'t': 0.0}

        def now():
            clock['t'] += 1.0
            return clock['t']

        ctrl.now_fn = now
        ctrl.sleep_fn = lambda _s: store.advance()

        def must_not_run():
            raise AssertionError('host must not re-run train body while detached job lives')

        out = ctrl.run_stage_once(
            'train', must_not_run,
            detached_command='train_pit.py',
            wait_for_marker=True,
            poll_s=0.0,
            timeout_s=30.0,
        )
        self.assertEqual(out.get('checkpoint'), 'c' * 64)
        self.assertIn('train', ctrl.status().markers)
        # No additional launch — resumed the in-flight pid.
        self.assertEqual(len(store.launch_log), launches_before)

    def test_restart_mid_eval_resumes_to_result_without_release(self):
        """e1065 class: download.done + train.done, mid-probe → re-adopt polls eval."""
        store = MemBoxJobStore(default_ticks=3)
        ctrl = _ctrl(store, item_id='item-eval-mid')
        ctrl.ensure_dir()
        write_marker(store, ctrl.job_dir, 'download', {'ok': True})
        write_marker(store, ctrl.job_dir, 'train', {
            'ok': True, 'checkpoint': 'c' * 64,
            'candidate_id': 'c' * 64,
            'candidate_path': 'box:/ckpt/' + ('c' * 64),
            'report': '/tmp/report.json',
        })

        store.arm_launch(
            ticks=3,
            auto_marker=True,
            payload={
                'ok': True,
                'candidate_id': 'c' * 64,
                'probe_result': {'samples': 32, 'score': {'decision': 'red'}},
            },
        )
        pid = store.launch_detached(
            ctrl.job_dir, stage='eval', command='setsid nohup pit-box-job-eval',
        )
        self.assertIn(pid, store.alive_pids)
        st = inspect_job(store, ctrl.job_dir)
        self.assertEqual(st.state, STATUS_RUNNING)
        self.assertEqual(st.next_stage, 'eval')
        self.assertNotIn('eval', st.markers)
        launches_before = len(store.launch_log)

        clock = {'t': 0.0}

        def now():
            clock['t'] += 1.0
            return clock['t']

        ctrl.now_fn = now
        ctrl.sleep_fn = lambda _s: store.advance()

        def must_not_rerun():
            raise AssertionError('host must not re-run eval while detached job lives')

        out = ctrl.run_stage_once(
            'eval', must_not_rerun,
            detached_command='pit-box-job-eval',
            wait_for_marker=True,
            poll_s=0.0,
            timeout_s=30.0,
        )
        self.assertEqual(out.get('candidate_id'), 'c' * 64)
        self.assertEqual((out.get('probe_result') or {}).get('samples'), 32)
        self.assertIn('eval', ctrl.status().markers)
        self.assertEqual(len(store.launch_log), launches_before)

        # Publish RESULT as the detached job would; re-adopt fetches without release.
        write_result(store, ctrl.job_dir, {
            'ok': True,
            'candidate_id': 'c' * 64,
            'probe_result': out.get('probe_result'),
            'candidate_path': 'box:/ckpt/' + ('c' * 64),
        })
        fetched = ctrl.resume_or_fetch()
        self.assertIsNotNone(fetched)
        assert fetched is not None
        self.assertEqual(fetched['candidate_id'], 'c' * 64)
        # No release semantics in box_job layer — controller must keep box on GPU-busy.
        self.assertNotIn('release', str(fetched).lower())

    def test_restart_after_result_fetches_without_rerun(self):
        store = MemBoxJobStore()
        ctrl = _ctrl(store, item_id='item-done')
        ctrl.ensure_dir()
        write_marker(store, ctrl.job_dir, 'download', {'ok': True})
        write_marker(store, ctrl.job_dir, 'train', {'ok': True, 'checkpoint': 'c' * 64})
        write_marker(store, ctrl.job_dir, 'eval', {'ok': True})
        result = {
            'ok': True,
            'candidate_id': 'c' * 64,
            'candidate_path': 'box:/ckpt/' + ('c' * 64),
            'probe_result': {'score': {'decision': 'red'}, 'samples': 8},
            'train_receipt': {
                'candidate_id': 'c' * 64,
                'candidate_path': 'box:/ckpt/' + ('c' * 64),
                'candidate_master_path': '/tmp/master',
            },
            'bound_head': 'a' * 64,
            'bound_master_digest': 'b' * 64,
            'box_public_key_hex': 'ab' * 32,
        }
        write_result(store, ctrl.job_dir, result)

        st = ctrl.status()
        self.assertEqual(st.state, STATUS_DONE)
        fetched = ctrl.resume_or_fetch()
        self.assertIsNotNone(fetched)
        assert fetched is not None
        self.assertEqual(fetched['candidate_id'], 'c' * 64)
        self.assertEqual(fetched['probe_result']['samples'], 8)

        def boom():
            raise AssertionError('must not re-run any stage after RESULT')

        # Any stage call returns RESULT, never invokes runner / launch.
        out = ctrl.run_stage_once('download', boom)
        self.assertEqual(out['candidate_id'], 'c' * 64)
        self.assertEqual(store.launch_log, [])

    def test_kill_stale_before_rerun_when_marker_missing(self):
        store = MemBoxJobStore(default_ticks=99)
        ctrl = _ctrl(store, item_id='item-stale')
        ctrl.ensure_dir()
        write_marker(store, ctrl.job_dir, 'download', {'ok': True})
        store.arm_launch(ticks=99, auto_marker=False)
        pid = store.launch_detached(ctrl.job_dir, stage='train', command='train_pit.py')
        # Job died without marker (host sees need_launch).
        store.alive_pids.discard(pid)
        store.pending.pop(pid, None)
        st = ctrl.status()
        self.assertEqual(st.state, STATUS_NEED_LAUNCH)
        self.assertEqual(st.next_stage, 'train')

        killed = kill_stale_job(store, ctrl.job_dir, extra_patterns=('train_pit.py',))
        # pid file cleared even if process already dead.
        self.assertIsNone(store.read_text(marker_path(ctrl.job_dir, 'train')))
        self.assertTrue(isinstance(killed, list))

        store.arm_launch(ticks=1, auto_marker=True, payload={'ok': True, 'checkpoint': 'd' * 64})
        clock = {'t': 0.0}

        def _now():
            clock['t'] += 1.0
            return clock['t']

        ctrl.now_fn = _now
        ctrl.sleep_fn = lambda _s: store.advance()
        out = ctrl.run_stage_once(
            'train', lambda: {},
            detached_command='train_pit.py',
            wait_for_marker=True,
            poll_s=0.0,
            timeout_s=20.0,
        )
        self.assertEqual(out.get('checkpoint'), 'd' * 64)

    def test_box_runner_resumes_from_result_without_train(self):
        store = MemBoxJobStore()
        item_id = 'item-runner-resume'
        ctrl = BoxJobController(store=store, item_id=item_id, sleep_fn=lambda _s: None)
        ctrl.ensure_dir()
        prior = {
            'ok': True,
            'candidate_id': 'c' * 64,
            'candidate_path': 'box:/ckpt/' + ('c' * 64),
            'candidate_master_path': '/tmp/master',
            'probe_result': {
                'k': 8, 'samples': 16, 'score': {'decision': 'red', 'z': -1.0},
                'eval_gpu': 'B300',
            },
            'train_receipt': {
                'candidate_id': 'c' * 64,
                'candidate_path': 'box:/ckpt/' + ('c' * 64),
                'candidate_master_path': '/tmp/master',
                'train_gpu': 'B300',
            },
            'training_report': {
                'source_file_hashes': {'trainer': 'a' * 64, 'batch': 'b' * 64},
                'batching': {'tasks': 1, 'rollouts': 1},
                'tokens': {'prompt_tokens': 1, 'output_tokens': 1, 'loss_tokens': 1},
                'update_size': 1.0,
                'gradient_norm': 1.0,
                'engine': 'packed',
            },
            'bound_head': 'a' * 64,
            'bound_master_digest': 'b' * 64,
            'box_public_key_hex': 'cd' * 32,
            'train_gpu': 'B300',
            'eval_gpu': 'B300',
        }
        write_result(store, ctrl.job_dir, prior)

        class Pod:
            def run(self, cmd, timeout=60, check=False, **kwargs):
                # Minimal remote surface so _try_box_job_controller accepts us;
                # BoxRunner resume path uses a patched controller via store inject.
                return ''

        # Drive _bundle_from_box_result directly (RESULT path unit).
        record = ItemRecord(
            item_id=item_id,
            registration_index=1,
            epoch=1057,
            state=ItemState.PROVISIONING,
            bound_head='a' * 64,
            bound_master_digest='b' * 64,
            shadow=True,
        )
        key = generate_ephemeral_box_key()
        rec2, bundle = _bundle_from_box_result(
            record, prior, box_key=key, train_gpu='B300', eval_gpu=None,
        )
        self.assertEqual(rec2.state, ItemState.RESULT)
        self.assertEqual(bundle.candidate_id, 'c' * 64)
        self.assertEqual(bundle.probe_result.get('samples'), 16)
        self.assertTrue((bundle.meta or {}).get('readopt_result'))

    def test_job_dir_has_no_host_paths(self):
        jd = job_dir_for('abc123', job_root='/workspace/pst/pit-jobs')
        self.assertTrue(jd.startswith('/workspace/'))
        self.assertNotIn('/home/', jd)
        self.assertNotIn('192.', jd)
        self.assertNotIn('10.', jd)


class TestBoxRunnerMidEvalReadopt(unittest.TestCase):
    """Documented host-restart check (eval phase).

    After a host restart with cap=5, nearly every box is mid-eval. The host must:

    * read ``markers/`` (download.done + train.done), ``job.pid``, and eventually
      ``RESULT.json``;
    * **not** re-run train;
    * **not** kill / re-launch the detached eval while ``job.pid`` is alive;
    * **not** release the box.

    See also ``test_restart_mid_eval_resumes_to_result_without_release`` (box_job
    layer) and this BoxRunner-level coverage.
    """

    def test_host_restart_mid_eval_skips_train_and_polls_pid(self):
        """Real sparse train.done (checkpoint/events/ok/rc_ok only) + live eval pid."""
        trains = {'n': 0}
        places = {'n': 0}
        frees = {'n': 0}

        class CountingIO(ScriptedBoxIO):
            def train_v834e(self, **kwargs):
                trains['n'] += 1
                # If the mid-eval path wrongly falls through, HostBoxIO would
                # call place_and_pull → free_gpus. Scripted path still counts.
                places['n'] += 1
                frees['n'] += 1
                return super().train_v834e(**kwargs)

            def pull_training_report(self, **kwargs):
                return {
                    'source_file_hashes': {'trainer': 'a' * 64, 'batch': 'b' * 64},
                    'batching': {'tasks': 1, 'rollouts': 1},
                    'tokens': {'prompt_tokens': 1, 'output_tokens': 1, 'loss_tokens': 1},
                    'update_size': 1.0,
                    'gradient_norm': 1.0,
                    'engine': 'packed',
                }

        store = MemBoxJobStore(default_ticks=3)
        item_id = 'item-mid-eval-readopt'
        ctrl = BoxJobController(store=store, item_id=item_id, sleep_fn=lambda _s: None)
        ctrl.ensure_dir()
        write_marker(store, ctrl.job_dir, 'download', {'ok': True})
        # Exact real-box train.done shape — no candidate_path / master_path.
        write_marker(store, ctrl.job_dir, 'train', {
            'ok': True,
            'checkpoint': 'c' * 64,
            'events': {
                'done': {'event': 'done', 'checkpoint': 'c' * 64},
                'native-grpo-complete': {'event': 'native-grpo-complete'},
            },
            'rc_ok': True,
        })
        store.arm_launch(
            ticks=3,
            auto_marker=True,
            payload={
                'ok': True,
                'candidate_id': 'c' * 64,
                'probe_result': {
                    'k': 8, 'samples': 16, 'score': {'decision': 'red', 'z': -1.0},
                    'eval_gpu': 'B300', 'partial': False,
                    'planned_samples': 16, 'dtype': 'bf16',
                },
            },
        )
        pid = store.launch_detached(
            ctrl.job_dir, stage='eval', command='setsid nohup pit-box-job-eval',
        )
        self.assertIn(pid, store.alive_pids)
        st = inspect_job(store, ctrl.job_dir)
        self.assertEqual(st.state, STATUS_RUNNING)
        self.assertEqual(st.next_stage, 'eval')
        self.assertIn('train', st.markers)
        self.assertNotIn('eval', st.markers)
        launches_before = len(store.launch_log)
        killed: list[int] = []
        _orig_kill = store.kill_pids

        def _tracking_kill(pids):
            killed.extend(int(p) for p in pids)
            return _orig_kill(pids)

        store.kill_pids = _tracking_kill  # type: ignore[method-assign]

        clock = {'t': 0.0}

        def now():
            clock['t'] += 1.0
            return clock['t']

        ctrl.now_fn = now
        ctrl.sleep_fn = lambda _s: store.advance()

        io = CountingIO()
        runner = BoxRunner(config=PitConfig(enabled=True, mode='shadow'), io=io, lane=LaneService())
        record = ItemRecord(
            item_id=item_id,
            registration_index=1,
            epoch=1069,
            state=ItemState.PROVISIONING,
            bound_head='a' * 64,
            bound_master_digest='b' * 64,
            shadow=True,
            box_id='pod-mid-eval',
            meta={'pod': {'id': 'pod-mid-eval', 'ip': '10.0.0.1', 'port': 22}, 'train_gpu': 'B300'},
        )

        import ops.parallel_eval.pit.box as box_mod
        orig = box_mod._try_box_job_controller

        def _fake_try(pod, iid):
            if iid == item_id:
                return ctrl
            return orig(pod, iid)

        box_mod._try_box_job_controller = _fake_try
        try:
            with tempfile.TemporaryDirectory() as td:
                with mock.patch.dict(os.environ, {'PE_STATE': td}, clear=False):
                    rec2, bundle = runner.run(
                        record,
                        batch_handle='opaque-batch',
                        plan_handle='opaque-plan',
                    )
        finally:
            box_mod._try_box_job_controller = orig

        self.assertEqual(trains['n'], 0, 'must not re-run train on mid-eval re-adopt')
        self.assertEqual(places['n'], 0, 'must not place_cand while eval pid alive')
        self.assertEqual(frees['n'], 0, 'must not free_gpus while eval pid alive')
        self.assertEqual(len(store.launch_log), launches_before, 'must not re-launch eval')
        self.assertNotIn(pid, killed, 'must not kill/signal the live eval pid')
        self.assertEqual(rec2.state, ItemState.RESULT)
        self.assertEqual(bundle.candidate_id, 'c' * 64)
        self.assertEqual(bundle.probe_result.get('samples'), 16)
        # Real report pulled — not an all-zero invented summary.
        self.assertEqual(
            (bundle.training_report or {}).get('source_file_hashes', {}).get('trainer'),
            'a' * 64,
        )
        self.assertTrue(rec2.result_stored)


class TestBoxRunnerResultResume(unittest.TestCase):
    def test_runner_short_circuits_on_box_result(self):
        trains = {'n': 0}

        class CountingIO(ScriptedBoxIO):
            def train_v834e(self, **kwargs):
                trains['n'] += 1
                return super().train_v834e(**kwargs)

        class FakePod:
            def __init__(self, store: MemBoxJobStore, item_id: str):
                self._store = store
                self._item_id = item_id

            def run(self, cmd, timeout=60, check=False, **kwargs):
                # Delegate to Remote-like Mem store by operating through controller helpers.
                # BoxRunner only needs hasattr(run); RESULT is fetched via controller_for_pod
                # which uses RemoteBoxJobStore — so patch _try_box_job_controller instead.
                return ''

        store = MemBoxJobStore()
        item_id = 'item-short-circuit'
        prior = {
            'ok': True,
            'candidate_id': 'c' * 64,
            'candidate_path': 'box:/ckpt/' + ('c' * 64),
            'candidate_master_path': '/tmp/master',
            'probe_result': {
                'k': 8, 'samples': 16, 'score': {'decision': 'green', 'z': 2.0},
                'eval_gpu': 'B300', 'partial': False,
                'planned_samples': 16, 'dtype': 'bf16',
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
                    'update_size': 1.0,
                    'gradient_norm': 1.0,
                    'engine': 'packed',
                },
            },
            'training_report': {
                'source_file_hashes': {'trainer': 'a' * 64, 'batch': 'b' * 64},
                'batching': {'tasks': 1, 'rollouts': 1},
                'tokens': {'prompt_tokens': 1, 'output_tokens': 1, 'loss_tokens': 1},
                'update_size': 1.0,
                'gradient_norm': 1.0,
                'engine': 'packed',
            },
            'bound_head': 'a' * 64,
            'bound_master_digest': 'b' * 64,
            'box_public_key_hex': 'ef' * 32,
            'train_gpu': 'B300',
            'eval_gpu': 'B300',
        }
        ctrl = BoxJobController(store=store, item_id=item_id, sleep_fn=lambda _s: None)
        ctrl.ensure_dir()
        write_result(store, ctrl.job_dir, prior)

        io = CountingIO()
        runner = BoxRunner(config=PitConfig(enabled=True, mode='shadow'), io=io, lane=LaneService())
        record = ItemRecord(
            item_id=item_id,
            registration_index=1,
            epoch=1058,
            state=ItemState.PROVISIONING,
            bound_head='a' * 64,
            bound_master_digest='b' * 64,
            shadow=True,
            meta={'pod': FakePod(store, item_id), 'train_gpu': 'B300'},
        )

        # Inject Mem controller so resume_or_fetch sees RESULT without SSH.
        import ops.parallel_eval.pit.box as box_mod
        orig = box_mod._try_box_job_controller

        def _fake_try(pod, iid):
            if iid == item_id:
                return ctrl
            return orig(pod, iid)

        box_mod._try_box_job_controller = _fake_try
        try:
            rec2, bundle = runner.run(
                record,
                batch_handle='opaque-batch',
                plan_handle='opaque-plan',
            )
        finally:
            box_mod._try_box_job_controller = orig

        self.assertEqual(trains['n'], 0)
        self.assertEqual(rec2.state, ItemState.RESULT)
        self.assertEqual(bundle.candidate_id, 'c' * 64)
        self.assertTrue((bundle.meta or {}).get('readopt_result'))


if __name__ == '__main__':
    unittest.main()
