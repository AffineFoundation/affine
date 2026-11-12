#!/usr/bin/env python3
"""Two starters of one detached stage: one starts the job, the other adopts it. A job that is alive is not an error.

The box is this machine. The real ``RemoteBoxJobStore`` runs its real shell commands (the flock, the claim file,
setsid and nohup) in a local shell. The "trainer" is a real detached process: it writes one line when it starts,
sleeps, and writes the stage marker. ``FencedBox`` keeps ``pgrep`` inside the processes of this test, so the
stale-job kill of the code under test cannot touch a different process of this machine.

The cases with a fixed order of events use ``MemBoxJobStore``, which has the same claim rule.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ops.parallel_eval.live import pe_box
from ops.parallel_eval.pit.box_job import (
    CLAIM_LOCK_NAME,
    CLAIM_NAME,
    PID_NAME,
    START_CLAIM_TTL_S,
    STATUS_RUNNING,
    BoxJobController,
    BoxJobError,
    MemBoxJobStore,
    RemoteBoxJobStore,
    marker_path,
    stage_path,
)
from ops.parallel_eval.pit.r2_fixture import ShellBox

PAYLOAD = {'ok': True, 'checkpoint': 'c' * 64}


def forbidden() -> Dict[str, Any]:
    raise AssertionError('the host must not run a detached stage')


class FencedBox(ShellBox):
    """A local shell as the box. ``pgrep`` gives only the processes that a box with this mark started."""

    def __init__(self, mark: str) -> None:
        super().__init__()
        self.mark = mark

    def run(self, cmd, inp=None, timeout=600, check=True, **_kwargs):
        self.commands.append(cmd)
        proc = subprocess.run(['bash', '-c', cmd], input=inp, capture_output=True, timeout=timeout,
                              env=dict(os.environ, PIT_TEST_BOX=self.mark))
        out = proc.stdout.decode(errors='replace')
        if check and proc.returncode != 0:
            raise RuntimeError('box command failed rc=%d: %s' % (proc.returncode, proc.stderr.decode(errors='replace')[-400:]))
        if cmd.startswith('pgrep '):
            out = ' '.join(pid for pid in out.split() if self.started_here(pid))
        return out

    def started_here(self, pid: str) -> bool:
        try:
            return ('PIT_TEST_BOX=%s' % self.mark).encode() in Path('/proc/%s/environ' % pid).read_bytes().split(b'\0')
        except OSError:
            return False


class RealBoxCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.mark = uuid.uuid4().hex
        self.addCleanup(self.stop_jobs)

    def stop_jobs(self) -> None:
        box = FencedBox(self.mark)
        for pid in box.run('pgrep -f pit-box-job || true', check=False).split():
            box.run('kill -9 %s 2>/dev/null || true' % pid, check=False)

    def controller(self, item: str = 'item-1', **store_args: Any) -> BoxJobController:
        store = RemoteBoxJobStore(FencedBox(self.mark), **store_args)
        return BoxJobController(store=store, item_id=item, job_root=str(self.tmp / 'jobs'))

    def trainer(self, ctrl: BoxJobController, *, seconds: float = 1.0) -> str:
        """The command of a detached "trainer": one line in ``starts`` when it starts, then the marker of the stage."""
        marker = marker_path(ctrl.job_dir, 'train')
        return "echo start $$ >> {starts}; sleep {s}; echo '{doc}' > {m}.tmp && mv {m}.tmp {m}".format(
            starts=self.starts_file(ctrl), s=seconds, doc=json.dumps(PAYLOAD), m=marker)

    def starts_file(self, ctrl: BoxJobController) -> Path:
        return Path(ctrl.job_dir) / 'starts'

    def starts(self, ctrl: BoxJobController) -> List[str]:
        path = self.starts_file(ctrl)
        return path.read_text().splitlines() if path.exists() else []

    def run_train(self, ctrl: BoxJobController, **args: Any) -> Dict[str, Any]:
        return ctrl.run_stage_once('train', forbidden, detached_command=self.trainer(ctrl, **args),
                                   wait_for_marker=True, poll_s=0.1, timeout_s=60)


class TestTwoStartersOnARealBox(RealBoxCase):
    def test_two_starters_at_one_time_start_one_trainer_and_the_two_get_its_result(self) -> None:
        for round_no in range(6):
            item = 'item-%d' % round_no
            ctrls = [self.controller(item), self.controller(item)]
            self.assertNotEqual(ctrls[0].owner, ctrls[1].owner)
            gate = threading.Barrier(2)
            out: List[Any] = [None, None]

            def start(i: int) -> None:
                gate.wait()
                try:
                    out[i] = self.run_train(ctrls[i])
                except Exception as exc:      # the test reads it from ``out``
                    out[i] = exc

            workers = [threading.Thread(target=start, args=(i,)) for i in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(90)
            self.assertEqual(out, [PAYLOAD, PAYLOAD], 'round %d' % round_no)
            self.assertEqual(len(self.starts(ctrls[0])), 1, 'round %d: one trainer process' % round_no)
            did = sorted(ctrls[0].starts + ctrls[1].starts)
            self.assertEqual(did, [('train', 'adopted'), ('train', 'started')], 'round %d' % round_no)
            self.assertFalse((Path(ctrls[0].job_dir) / CLAIM_NAME).exists(), 'the claim is given back')

    def test_a_starter_that_comes_when_the_trainer_is_alive_adopts_it(self) -> None:
        first, second = self.controller(), self.controller()
        first.ensure_dir()
        pid = first.store.launch_detached(first.job_dir, stage='train', command=self.trainer(first, seconds=1.5))
        status = second.status()
        # No marker of the stage before it: the markers alone do not show that the job is a train job.
        self.assertEqual((status.state, status.next_stage, status.pid), (STATUS_RUNNING, 'download', pid))
        self.assertEqual(self.run_train(second), PAYLOAD)
        self.assertEqual(len(self.starts(first)), 1)
        self.assertEqual(second.starts, [('train', 'adopted')])
        self.assertIsNone(second.store.last_launch, 'the second starter started nothing')

    def test_a_launch_for_a_stage_that_is_alive_starts_nothing(self) -> None:
        ctrl = self.controller()
        ctrl.ensure_dir()
        store = ctrl.store
        pid = store.launch_detached(ctrl.job_dir, stage='train', command=self.trainer(ctrl, seconds=1.0))
        self.assertEqual(store.last_launch, 'started')
        self.assertEqual(Path(stage_path(ctrl.job_dir)).read_text().split(), ['train'])
        for _ in range(3):
            self.assertEqual(store.launch_detached(ctrl.job_dir, stage='train', command=self.trainer(ctrl)), pid)
            self.assertEqual(store.last_launch, 'busy')
        self.assertEqual(len(self.starts(ctrl)), 1)
        # A different stage is not this job: it starts. (The caller kills leftovers before that.)
        other = store.launch_detached(ctrl.job_dir, stage='eval', command='sleep 0.2')
        self.assertNotEqual(other, pid)
        self.assertEqual(store.last_launch, 'started')
        deadline = time.time() + 30
        while (store.pid_alive(pid) or store.pid_alive(other)) and time.time() < deadline:
            time.sleep(0.1)
        # The trainer has ended: a new launch of the stage starts a new process.
        store.pod.run('rm -f %s' % marker_path(ctrl.job_dir, 'train'))
        again = store.launch_detached(ctrl.job_dir, stage='train', command=self.trainer(ctrl, seconds=0.1))
        self.assertNotEqual(again, pid)
        self.assertEqual(store.last_launch, 'started')

    def test_the_kill_of_leftovers_removes_the_stage_of_the_job(self) -> None:
        ctrl = self.controller()
        ctrl.ensure_dir()
        pid = ctrl.store.launch_detached(ctrl.job_dir, stage='train', command=self.trainer(ctrl, seconds=30))
        self.assertIn(pid, ctrl.kill_stale())
        self.assertFalse(ctrl.store.pid_alive(pid))
        for name in (PID_NAME, 'job.stage'):
            self.assertFalse((Path(ctrl.job_dir) / name).exists(), name)

    def test_the_start_claim_is_for_one_starter_and_ends_when_its_starter_died(self) -> None:
        ctrl = self.controller()
        ctrl.ensure_dir()
        store, jd = ctrl.store, ctrl.job_dir
        claim = Path(jd) / CLAIM_NAME

        def take(owner: str, **args: Any) -> bool:
            return store.claim_start(jd, stage='train', owner=owner, ttl_s=START_CLAIM_TTL_S, **args)

        self.assertFalse(take('a', renew=True))                  # no claim: there is nothing to renew
        self.assertFalse(claim.exists())
        self.assertTrue(take('a'))
        self.assertFalse(take('b'))
        self.assertTrue(take('a'))                               # its owner can take it again
        self.assertTrue(take('a', renew=True))
        store.release_start(jd, stage='train', owner='b')        # not the owner: nothing changes
        self.assertFalse(take('b'))
        store.release_start(jd, stage='train', owner='a')
        self.assertFalse(claim.exists())
        self.assertTrue(take('b'))
        self.assertFalse(take('a', renew=True))                  # a claim that went to "b" is not renewed for "a"
        claim.write_text('dead train %d\n' % (int(time.time()) - int(START_CLAIM_TTL_S) - 5))
        self.assertFalse(take('c', renew=True))                  # the time of a claim does not make it ours
        self.assertTrue(take('c'))                               # the starter "dead" did not come back
        self.assertEqual(claim.read_text().split()[:2], ['c', 'train'])
        claim.write_text('x y not-a-time\n')
        self.assertTrue(take('d'))                               # a claim without a time has ended

    @unittest.skipUnless(shutil.which('flock'), 'flock is not on this machine')
    def test_a_start_lock_that_does_not_come_free_is_no_claim_and_no_launch(self) -> None:
        ctrl = self.controller(lock_wait_s=1)
        ctrl.ensure_dir()
        store, jd = ctrl.store, ctrl.job_dir
        lock = str(Path(jd) / CLAIM_LOCK_NAME)
        holder = subprocess.Popen(['flock', lock, 'sleep', '4'])
        self.addCleanup(holder.kill)
        deadline = time.time() + 10
        while subprocess.run(['flock', '-n', lock, 'true']).returncode == 0 and time.time() < deadline:
            time.sleep(0.05)
        self.assertFalse(store.claim_start(jd, stage='train', owner='a', ttl_s=START_CLAIM_TTL_S))
        with self.assertRaises(BoxJobError) as caught:
            store.launch_detached(jd, stage='train', command=self.trainer(ctrl))
        self.assertIn('start lock', str(caught.exception))
        self.assertEqual(self.starts(ctrl), [])
        holder.wait(20)
        self.assertTrue(store.claim_start(jd, stage='train', owner='a', ttl_s=START_CLAIM_TTL_S))

    def test_two_starters_put_the_same_file_at_one_time(self) -> None:
        dest = str(self.tmp / 'pst' / 'train_pit.py')
        body = b'print("trainer")\n' * 2000
        gate = threading.Barrier(2)
        errors: List[Exception] = []

        def put() -> None:
            box = FencedBox(self.mark)
            gate.wait()
            try:
                for _ in range(15):
                    pe_box.put(box, dest, body, '700')
            except Exception as exc:
                errors.append(exc)

        workers = [threading.Thread(target=put) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(120)
        self.assertEqual(errors, [])
        self.assertEqual(Path(dest).read_bytes(), body)
        self.assertEqual(sorted(p.name for p in Path(dest).parent.iterdir()), ['train_pit.py'])


class OrderStore(MemBoxJobStore):
    """The in-memory store, with one hook before the pid file of the job is removed and a count of live jobs."""

    before_pid_remove: Optional[Callable[[], None]] = None
    most_alive = 0

    def remove(self, path: str) -> None:
        if path.endswith('/' + PID_NAME) and self.before_pid_remove is not None:
            hook, self.before_pid_remove = self.before_pid_remove, None
            hook()
        super().remove(path)

    def launch_detached(self, job_dir: str, *, stage: str, command: str) -> int:
        pid = super().launch_detached(job_dir, stage=stage, command=command)
        self.most_alive = max(self.most_alive, len(self.alive_pids))
        return pid


class TestStartOrder(unittest.TestCase):
    """Fixed orders of events, on the in-memory store. No marker of an earlier stage is there."""

    def setUp(self) -> None:
        self.now = 1000.0
        self.store = OrderStore(clock=lambda: self.now)
        self.on_sleep: List[Any] = []

        def sleep(seconds: float) -> None:
            self.now += float(seconds)
            if self.on_sleep:
                self.on_sleep.pop(0)()
            self.store.advance()

        self.ctrl = BoxJobController(store=self.store, item_id='item-1', sleep_fn=sleep, now_fn=lambda: self.now)
        self.jd = self.ctrl.job_dir

    def run_train(self, timeout_s: float = 600.0) -> Dict[str, Any]:
        return self.ctrl.run_stage_once('train', forbidden, detached_command='train-cmd', wait_for_marker=True,
                                        poll_s=1.0, timeout_s=timeout_s)

    def other_starts(self, ticks: int = 3) -> int:
        """A different starter starts the train job, with the same store rules."""
        self.store.arm_launch(ticks=ticks, payload=PAYLOAD)
        return self.store.launch_detached(self.jd, stage='train', command='train-cmd')

    def launched(self) -> List[int]:
        return [row['pid'] for row in self.store.launch_log]

    def test_a_starter_without_the_claim_waits_and_adopts_the_job_of_the_claim_owner(self) -> None:
        """The other starter holds the claim and has not started its job. Its job comes one poll later."""
        self.store.claims[self.jd] = {'owner': 'other', 'stage': 'train', 'at': self.now}
        pids: List[int] = []

        def other() -> None:
            pids.append(self.other_starts())
            self.store.release_start(self.jd, stage='train', owner='other')

        self.on_sleep.append(other)
        self.assertEqual(self.run_train(), PAYLOAD)
        self.assertEqual(self.launched(), pids)
        self.assertEqual(self.ctrl.starts, [('train', 'adopted')])

    def test_the_claim_of_a_starter_that_died_ends_and_the_start_goes_on(self) -> None:
        self.store.claims[self.jd] = {'owner': 'dead', 'stage': 'train', 'at': self.now - START_CLAIM_TTL_S - 1}
        self.store.arm_launch(ticks=2, payload=PAYLOAD)
        self.assertEqual(self.run_train(), PAYLOAD)
        self.assertEqual(len(self.store.launch_log), 1)
        self.assertEqual(self.ctrl.starts, [('train', 'started')])
        self.assertEqual(self.store.claims, {})

    def test_a_claim_with_no_job_past_the_deadline_is_an_error_and_nothing_is_killed(self) -> None:
        self.store.claims[self.jd] = {'owner': 'other', 'stage': 'train', 'at': self.now}
        self.on_sleep.extend([lambda: self.store.claims[self.jd].update(at=self.now)] * 40)
        with self.assertRaises(BoxJobError) as caught:
            self.run_train(timeout_s=30.0)
        self.assertIn('a different starter holds the start claim', str(caught.exception))
        self.assertEqual(self.store.launch_log, [])
        self.assertEqual(self.store.claims[self.jd]['owner'], 'other')

    def test_a_job_that_came_alive_before_our_claim_is_adopted_and_not_killed(self) -> None:
        """The other starter started its job and gave the claim back after this starter saw no job."""
        seen: List[str] = []
        pids: List[int] = []
        real_status = self.ctrl.status

        def status():
            st = real_status()
            seen.append(st.state)
            if len(seen) == 1:                       # after the first look: the other starter starts its job
                pids.append(self.other_starts())
            return st

        self.ctrl.status = status
        self.assertEqual(self.run_train(), PAYLOAD)
        self.assertEqual(self.launched(), pids)
        self.assertEqual(self.ctrl.starts, [('train', 'adopted')])

    def test_a_job_of_the_code_before_the_stage_file_is_adopted_by_its_markers(self) -> None:
        """No ``job.stage``: the first stage without a marker is the stage of the job that is alive."""
        self.store.files[marker_path(self.jd, 'download')] = '{}\n'
        seen: List[str] = []
        pids: List[int] = []
        real_status = self.ctrl.status

        def status():
            st = real_status()
            seen.append(st.state)
            if len(seen) == 1:
                pids.append(self.other_starts())
                del self.store.files[stage_path(self.jd)]
            return st

        self.ctrl.status = status
        self.assertEqual(self.run_train(), PAYLOAD)
        self.assertEqual(self.launched(), pids)
        self.assertEqual(self.ctrl.starts, [('train', 'adopted')])

    def test_a_job_that_is_alive_at_the_launch_is_adopted(self) -> None:
        """A starter that does not use the claim started the job after our kill of leftovers: the launch is BUSY."""
        pids: List[int] = []
        real_kill = self.ctrl.kill_stale

        def kill(**args: Any):
            out = real_kill(**args)
            pids.append(self.other_starts())
            return out

        self.ctrl.kill_stale = kill
        self.assertEqual(self.run_train(), PAYLOAD)
        self.assertEqual(self.launched(), pids)
        self.assertEqual(self.store.last_launch, 'busy')
        self.assertEqual(self.ctrl.starts, [('train', 'adopted')])
        self.assertEqual(self.store.claims, {})

    def test_a_starter_that_was_slower_than_its_claim_does_not_start_a_second_trainer(self) -> None:
        """Our kill of leftovers takes longer than the claim lives. In that time a different starter takes the claim,
        starts its job and gives the claim back. Then the end of our kill removes the pid file of that job."""
        def other() -> None:
            self.now += START_CLAIM_TTL_S + 1
            self.assertTrue(self.store.claim_start(self.jd, stage='train', owner='other', ttl_s=START_CLAIM_TTL_S))
            self.other_starts(ticks=50)
            self.store.release_start(self.jd, stage='train', owner='other')

        self.store.before_pid_remove = other
        self.assertEqual(self.run_train(), PAYLOAD)
        self.assertEqual(self.store.most_alive, 1, 'two trainers were alive at one time')
        self.assertEqual(len(self.store.launch_log), 2)
        self.assertEqual(self.ctrl.starts, [('train', 'started')])

    def test_a_release_that_fails_is_not_an_error_of_the_start(self) -> None:
        def broken(job_dir: str, *, stage: str, owner: str) -> None:
            raise RuntimeError('the box did not answer')

        self.store.release_start = broken
        self.store.arm_launch(ticks=2, payload=PAYLOAD)
        self.assertEqual(self.run_train(), PAYLOAD)
        self.assertEqual(self.ctrl.starts, [('train', 'started')])


if __name__ == '__main__':
    unittest.main()
