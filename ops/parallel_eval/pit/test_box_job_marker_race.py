"""Race: stage writes <.done> then exits between marker read and pid-alive check.

inspect_job must check pid-alive first, then markers. When not running with no
marker yet, re-read markers/RESULT (~3× over ~10s). Never fail/release on
'exited without marker' without checking artifacts + job-log rc==0; otherwise
retry the stage once on the same box.
"""
from __future__ import annotations

import json
import unittest
from typing import Any, Dict, Optional, Set

from .box_job import (
    STATUS_NEED_LAUNCH,
    STATUS_RUNNING,
    BoxJobController,
    BoxJobError,
    MemBoxJobStore,
    inspect_job,
    job_log_rc_ok,
    marker_path,
    write_marker,
    write_result,
)


CK = 'c' * 64


class RaceInterleaveStore(MemBoxJobStore):
    """On first pid_alive check while still alive: write marker, then die.

    Old inspect order (markers → pid): markers empty, then pid dies after
    writing .done → NEED_LAUNCH without marker.
    New order (pid → markers): pid check writes .done + dies, then markers
    read sees train.done.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pid_check_count = 0
        self.marker_list_count = 0
        self.race_stage = 'train'
        self.race_payload: Dict[str, Any] = {
            'ok': True, 'checkpoint': CK,
        }
        self.race_armed = True

    def list_marker_stages(self, job_dir: str) -> Set[str]:
        self.marker_list_count += 1
        return super().list_marker_stages(job_dir)

    def pid_alive(self, pid: int) -> bool:
        self.pid_check_count += 1
        alive = int(pid) in self.alive_pids
        if self.race_armed and alive:
            # Stage completed: write .done then exit (between the two inspect reads
            # under the old marker-first order).
            jd = None
            for job in self.pending.values():
                if job.get('stage') == self.race_stage:
                    jd = job.get('job_dir')
                    break
            if jd is None and self.pending:
                jd = next(iter(self.pending.values())).get('job_dir')
            if jd:
                path = marker_path(str(jd), self.race_stage)
                self.files[path] = json.dumps(self.race_payload, sort_keys=True) + '\n'
            self.alive_pids.discard(int(pid))
            self.pending.pop(int(pid), None)
            self.files.pop('%s/job.pid' % str(jd or '').rstrip('/'), None)
            self.race_armed = False
            return False
        return alive


class LateMarkerStore(MemBoxJobStore):
    """Marker appears only after N list_marker_stages calls (re-read recovery)."""

    def __init__(self, *args: Any, appear_after: int = 2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.appear_after = appear_after
        self.list_calls = 0
        self.late_stage = 'train'
        self.late_payload = {'ok': True, 'checkpoint': CK}
        self.late_job_dir: Optional[str] = None

    def list_marker_stages(self, job_dir: str) -> Set[str]:
        self.list_calls += 1
        self.late_job_dir = job_dir
        found = super().list_marker_stages(job_dir)
        if self.late_stage in found:
            return found
        if self.list_calls > self.appear_after:
            path = marker_path(job_dir, self.late_stage)
            self.files[path] = json.dumps(self.late_payload, sort_keys=True) + '\n'
            found = super().list_marker_stages(job_dir)
        return found


class TestInspectJobPidFirst(unittest.TestCase):
    def test_pid_check_before_markers_wins_race(self):
        store = RaceInterleaveStore(default_ticks=99)
        ctrl = BoxJobController(
            store=store, item_id='e1067-race',
            sleep_fn=lambda _s: None, now_fn=lambda: 0.0,
        )
        ctrl.ensure_dir()
        write_marker(store, ctrl.job_dir, 'download', {'ok': True})
        store.arm_launch(ticks=99, auto_marker=False, payload={})
        pid = store.launch_detached(
            ctrl.job_dir, stage='train', command='train_pit.py',
        )
        self.assertIn(pid, store.alive_pids)

        # Single inspect: pid-alive first triggers write+.done+exit, then markers.
        st = inspect_job(store, ctrl.job_dir)
        self.assertNotEqual(st.state, STATUS_RUNNING)
        self.assertIn('train', st.markers)
        self.assertEqual(st.next_stage, 'eval')
        self.assertGreaterEqual(store.pid_check_count, 1)
        # Markers listed only after pid check (dead path).
        self.assertGreaterEqual(store.marker_list_count, 1)

    def test_old_order_would_miss_marker(self):
        """Document the race: markers-first snapshot misses a concurrent .done."""
        store = RaceInterleaveStore(default_ticks=99)
        ctrl = BoxJobController(
            store=store, item_id='e1067-old',
            sleep_fn=lambda _s: None, now_fn=lambda: 0.0,
        )
        ctrl.ensure_dir()
        write_marker(store, ctrl.job_dir, 'download', {'ok': True})
        store.arm_launch(ticks=99, auto_marker=False)
        pid = store.launch_detached(
            ctrl.job_dir, stage='train', command='train_pit.py',
        )
        # Emulate OLD order: list markers while alive → empty for train.
        markers_first = store.list_marker_stages(ctrl.job_dir)
        self.assertNotIn('train', markers_first)
        # Then pid check writes .done and dies.
        self.assertFalse(store.pid_alive(pid))
        # Stale markers_first still missing train — that was the bug.
        self.assertNotIn('train', markers_first)
        # Fresh read after death sees it.
        self.assertIn('train', store.list_marker_stages(ctrl.job_dir))


class TestLateMarkerReread(unittest.TestCase):
    def test_reread_recovers_train_marker(self):
        store = LateMarkerStore(appear_after=2, default_ticks=1)
        clock = {'t': 0.0}
        sleeps: list[float] = []

        def sleep(s: float) -> None:
            sleeps.append(s)
            clock['t'] += float(s)

        ctrl = BoxJobController(
            store=store, item_id='e1067-late',
            sleep_fn=sleep, now_fn=lambda: clock['t'],
        )
        ctrl.ensure_dir()
        write_marker(store, ctrl.job_dir, 'download', {'ok': True})
        # Process already dead, no marker yet.
        store.arm_launch(ticks=1, auto_marker=False)
        pid = store.launch_detached(
            ctrl.job_dir, stage='train', command='train_pit.py',
        )
        store.alive_pids.discard(pid)
        store.pending.pop(pid, None)
        st = ctrl.status()
        self.assertEqual(st.state, STATUS_NEED_LAUNCH)
        self.assertNotIn('train', st.markers)

        # Direct recovery path: late marker appears on re-read.
        out = ctrl._await_late_marker('train', attempts=3, gap_s=3.0)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.get('checkpoint'), CK)
        self.assertEqual(sleeps, [3.0, 3.0, 3.0][: len(sleeps)])
        self.assertLessEqual(len(sleeps), 3)

    def test_reread_recovers_eval_result(self):
        store = MemBoxJobStore()
        clock = {'t': 0.0}
        ctrl = BoxJobController(
            store=store, item_id='e1067-eval-late',
            sleep_fn=lambda s: clock.__setitem__('t', clock['t'] + s),
            now_fn=lambda: clock['t'],
        )
        ctrl.ensure_dir()
        write_marker(store, ctrl.job_dir, 'download', {'ok': True})
        write_marker(store, ctrl.job_dir, 'train', {'ok': True, 'checkpoint': CK})

        # RESULT appears on second fetch during re-read.
        fetches = {'n': 0}
        real_fetch = store.read_text

        def read_text(path: str):
            text = real_fetch(path)
            if path.endswith('RESULT.json'):
                fetches['n'] += 1
                if fetches['n'] >= 2:
                    doc = {
                        'ok': True, 'candidate_id': CK,
                        'probe_result': {'samples': 8},
                    }
                    store.files[path] = json.dumps(doc) + '\n'
                    return store.files[path]
            return text

        store.read_text = read_text  # type: ignore[method-assign]
        out = ctrl._await_late_marker('eval', attempts=3, gap_s=1.0)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.get('candidate_id'), CK)


class TestArtifactAndRetry(unittest.TestCase):
    def test_artifacts_rc0_treats_train_done(self):
        store = MemBoxJobStore(
            has_train_artifacts=True,
            train_recovery_payload={'ok': True, 'checkpoint': CK},
        )
        ctrl = BoxJobController(
            store=store, item_id='e1067-art',
            sleep_fn=lambda _s: None, now_fn=lambda: 0.0,
        )
        ctrl.ensure_dir()
        write_marker(store, ctrl.job_dir, 'download', {'ok': True})

        store.arm_launch(ticks=1, auto_marker=False)
        pid = store.launch_detached(
            ctrl.job_dir, stage='train', command='train_pit.py',
        )
        store.alive_pids.discard(pid)
        store.pending.pop(pid, None)
        # Stamp rc=0 onto the attempt log *after* launch (launch creates job.log.N).
        store.write_text('%s/job.log' % ctrl.job_dir, 'train done\nrc=0\n')
        store.write_text('%s/job.log.0' % ctrl.job_dir, 'train done\nrc=0\n')
        self.assertTrue(job_log_rc_ok(store, ctrl.job_dir))

        clock = {'t': 0.0}
        ctrl.now_fn = lambda: clock.__setitem__('t', clock['t'] + 1.0) or clock['t']
        # Make late-marker re-reads find nothing (no auto write).
        out = ctrl._handle_exited_without_marker('train')
        self.assertEqual(out.get('checkpoint'), CK)
        self.assertIn('train', ctrl.status().markers)

    def test_retry_once_on_same_box_then_succeed(self):
        store = MemBoxJobStore(default_ticks=2)
        clock = {'t': 0.0}

        def now() -> float:
            clock['t'] += 1.0
            return clock['t']

        ctrl = BoxJobController(
            store=store, item_id='e1067-retry',
            sleep_fn=lambda _s: store.advance(),
            now_fn=now,
        )
        ctrl.ensure_dir()
        write_marker(store, ctrl.job_dir, 'download', {'ok': True})

        # First launch: die immediately without marker / artifacts / rc=0.
        store.arm_launch(ticks=1, auto_marker=False)
        launches_before = 0

        def runner() -> dict:
            raise AssertionError('host must not sync-run detached train')

        # Arm second launch (the retry) to succeed with a marker.
        # First arm is consumed by the first launch inside run_stage_once;
        # we re-arm after detecting the first death via a custom store tick.
        class DieThenSucceed(MemBoxJobStore):
            def __init__(self) -> None:
                super().__init__(default_ticks=2)
                self.launches = 0

            def launch_detached(self, job_dir: str, *, stage: str, command: str) -> int:
                self.launches += 1
                if self.launches == 1:
                    self.arm_launch(ticks=1, auto_marker=False)
                else:
                    self.arm_launch(
                        ticks=2, auto_marker=True,
                        payload={'ok': True, 'checkpoint': CK},
                    )
                return super().launch_detached(job_dir, stage=stage, command=command)

        store2 = DieThenSucceed()
        # Copy download marker into new store.
        write_marker(store2, ctrl.job_dir, 'download', {'ok': True})
        ctrl.store = store2
        ctrl.sleep_fn = lambda _s: store2.advance()
        # Skip long re-read sleeps for the no-artifact path.
        ctrl._await_late_marker = (  # type: ignore[method-assign]
            lambda stage, attempts=3, gap_s=3.0: None
        )

        out = ctrl.run_stage_once(
            'train', runner,
            detached_command='train_pit.py',
            wait_for_marker=True,
            poll_s=0.0,
            timeout_s=60.0,
        )
        self.assertEqual(out.get('checkpoint'), CK)
        self.assertEqual(store2.launches, 2)  # initial + one retry
        self.assertIn('train', ctrl.status().markers)

    def test_hard_fail_after_retry_exhausted(self):
        class AlwaysDie(MemBoxJobStore):
            def __init__(self) -> None:
                super().__init__()
                self.launches = 0

            def launch_detached(self, job_dir: str, *, stage: str, command: str) -> int:
                self.launches += 1
                self.arm_launch(ticks=1, auto_marker=False)
                return super().launch_detached(job_dir, stage=stage, command=command)

        store = AlwaysDie()
        clock = {'t': 0.0}
        ctrl = BoxJobController(
            store=store, item_id='e1067-fail',
            sleep_fn=lambda _s: store.advance(),
            now_fn=lambda: clock.__setitem__('t', clock['t'] + 1.0) or clock['t'],
        )
        ctrl.ensure_dir()
        write_marker(store, ctrl.job_dir, 'download', {'ok': True})
        ctrl._await_late_marker = (  # type: ignore[method-assign]
            lambda stage, attempts=3, gap_s=3.0: None
        )

        with self.assertRaises(BoxJobError) as ctx:
            ctrl.run_stage_once(
                'train', lambda: {},
                detached_command='train_pit.py',
                wait_for_marker=True,
                poll_s=0.0,
                timeout_s=60.0,
            )
        self.assertIn('exited without marker', str(ctx.exception))
        self.assertNotIn('retry once', str(ctx.exception))
        self.assertEqual(store.launches, 2)

    def test_run_stage_race_interleave_recovers_without_relaunch(self):
        """e1067 class: train finishes (write .done + exit) during inspect → done."""
        store = RaceInterleaveStore(default_ticks=99)
        clock = {'t': 0.0}
        ctrl = BoxJobController(
            store=store, item_id='e1067-live',
            sleep_fn=lambda _s: None,
            now_fn=lambda: clock.__setitem__('t', clock['t'] + 1.0) or clock['t'],
        )
        ctrl.ensure_dir()
        write_marker(store, ctrl.job_dir, 'download', {'ok': True})
        store.arm_launch(ticks=99, auto_marker=False)
        # Pre-launch a running train; run_stage_once should poll → race → recover.
        store.launch_detached(
            ctrl.job_dir, stage='train', command='train_pit.py',
        )
        launches_before = len(store.launch_log)

        def boom() -> dict:
            raise AssertionError('must not sync-run')

        out = ctrl.run_stage_once(
            'train', boom,
            detached_command='train_pit.py',
            wait_for_marker=True,
            poll_s=0.0,
            timeout_s=30.0,
        )
        self.assertEqual(out.get('checkpoint'), CK)
        # Recovered from in-flight race — no relaunch.
        self.assertEqual(len(store.launch_log), launches_before)


class TestEvalArtifactRecovery(unittest.TestCase):
    def test_eval_result_plus_rc0_recovers(self):
        store = MemBoxJobStore()
        ctrl = BoxJobController(
            store=store, item_id='e1067-eval-art',
            sleep_fn=lambda _s: None, now_fn=lambda: 0.0,
        )
        ctrl.ensure_dir()
        write_marker(store, ctrl.job_dir, 'download', {'ok': True})
        write_marker(store, ctrl.job_dir, 'train', {'ok': True, 'checkpoint': CK})
        store.write_text('%s/job.log' % ctrl.job_dir, 'eval done\nrc=0\n')
        write_result(store, ctrl.job_dir, {
            'ok': True, 'candidate_id': CK,
            'probe_result': {'samples': 16, 'score': {'decision': 'red'}},
        })
        out = ctrl._recover_from_artifacts('eval')
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out['candidate_id'], CK)
        self.assertEqual(out['probe_result']['samples'], 16)


if __name__ == '__main__':
    unittest.main()
