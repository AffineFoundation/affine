import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops" / "heldout_monitor"))

import base64
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from nacl.signing import SigningKey
import serial_worker as worker
from monitor import Monitor, select_step


class SerialMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'monitor'; self.root.mkdir()
        self.key = SigningKey.generate(); self.authority = self.key.verify_key.encode().hex()
        self.worker = self.root / 'serial_worker.py'
        self.worker.write_bytes(Path(worker.__file__).read_bytes())

    def sign(self, body):
        return dict(payload=body, signer=self.authority,
                    signature=base64.b64encode(self.key.sign(worker.canonical(body)).signature).decode())

    def job(self, name='step-2-fixture', delay=0.2):
        d = self.root / 'jobs' / name; d.mkdir(parents=True)
        files = {'a.safetensors': worker.digest(b'alpha'), 'config.json': worker.digest(b'{}')}
        cp = worker.digest(worker.canonical(files)); model = self.root.parent / 'checkpoints' / cp
        p = dict(directory=str(d), checkpoint=dict(id=cp, path=str(model), files=files))
        evaluate = ("import argparse,json,time,fcntl,os\nfrom pathlib import Path\n"
                    "p=argparse.ArgumentParser();p.add_argument('--plan');a=p.parse_args();d=Path(a.plan).parent\n"
                    "f=open(d/'evaluation.lease','a+b');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
                    "(d/'output').mkdir();(d/'started').write_text(str(os.getpid()))\n"
                    "time.sleep(" + str(delay) + ")\n(d/'output/result.json').write_text('{}')\n") .encode()
        hydrate = b"import sys,json\nfrom pathlib import Path\nPath(sys.argv[sys.argv.index('--output')+1]).write_text('{}')\n"
        p['program_sha256'] = worker.digest(evaluate)
        payloads = {'evaluate.py': evaluate, 'hydrate.py': hydrate,
                    'plan.json': worker.canonical(self.sign(p)),
                    'read-plan.json': worker.canonical(self.sign(dict(destination=str(model), checkpoint=cp)))}
        for n, raw in payloads.items(): (d / n).write_bytes(raw)
        assignment = self.sign(dict(version='serial-heldout128-assignment-v1', root=str(self.root), job_id=name,
                                    checkpoint=cp, checkpoint_path=str(model), retained_UUID='fixture',
                                    step=2, attempt=1,
                                    worker_sha256=worker.file_hash(self.worker),
                                    program_files={n: worker.digest(raw) for n, raw in payloads.items()}))
        (d / 'assignment.json').write_bytes(worker.canonical(assignment))
        return d, p

    def dispatch(self, d):
        return worker.dispatch(self.root, d / 'assignment.json', self.authority, sys.executable)

    def wait(self, fn, timeout=8):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            value = fn()
            if value: return value
            time.sleep(.03)
        self.fail('fixture did not finish')

    def retirement(self):
        d, plan = self.job(); (d / 'evaluation.lease').touch()
        path = Path(plan['checkpoint']['path']); path.mkdir(parents=True)
        (path / 'a.safetensors').write_bytes(b'alpha'); (path / 'config.json').write_bytes(b'{}')
        envelope = self.sign(plan)
        archive = self.sign(dict(checkpoint=plan['checkpoint']['id'], task_count=128,
                                 full_readback_verified=True, plan_sha256=worker.digest(worker.canonical(envelope))))
        grant = self.sign(dict(version='archived-heldout128-model-retirement-v1', root=str(self.root),
                               checkpoint=plan['checkpoint']['id'], evaluation_plan=envelope, archive=archive))
        p = self.root / 'grant.json'; p.write_bytes(worker.canonical(grant))
        return p, path, d

    def test_selection_starts_at_two_coalesces_and_waits_two_steps(self):
        self.assertFalse(select_step(1, None)); self.assertTrue(select_step(2, None))
        self.assertFalse(select_step(3, 2)); self.assertTrue(select_step(4, 2)); self.assertTrue(select_step(17, 2))

    def test_restart_observes_original_dispatch_without_relaunch(self):
        d, _ = self.job(); first = self.dispatch(d)
        self.wait(lambda: (d / 'status.json').exists() and json.loads((d / 'status.json').read_bytes())['phase'] == 'complete')
        self.wait(lambda: not worker.observe(self.root, d)['lease_busy'])
        second = self.dispatch(d)
        self.assertTrue(second['reused_original_dispatch']); self.assertNotIn('pid', second)
        self.assertEqual(first['pid'], json.loads((d / 'launch.json').read_bytes())['pid'])

    def test_child_keeps_global_lease_after_supervisor_crash(self):
        d, _ = self.job(delay=1.5); first = self.dispatch(d)
        self.wait(lambda: (d / 'started').exists()); os.kill(first['pid'], signal.SIGKILL)
        self.assertTrue(worker.observe(self.root, d)['lease_busy'])
        other, _ = self.job(name='step-4-fixture')
        with self.assertRaises(BlockingIOError): self.dispatch(other)
        self.wait(lambda: (d / 'output/result.json').exists())
        self.wait(lambda: not worker.observe(self.root, d)['lease_busy'])
        self.assertEqual(worker.observe(self.root, d)['phase'], 'complete-recovered')
        self.assertTrue(self.dispatch(d)['reused_original_dispatch'])

    def test_crash_after_durable_intent_never_repeats_ambiguous_launch(self):
        d, _ = self.job()
        with patch.object(worker.subprocess, 'Popen', side_effect=OSError('injected')):
            with self.assertRaises(OSError): self.dispatch(d)
        with patch.object(worker.subprocess, 'Popen') as spawn:
            self.assertTrue(self.dispatch(d)['reused_original_dispatch']); spawn.assert_not_called()
        self.assertEqual(worker.observe(self.root, d)['phase'], 'abandoned')

    def test_result_written_before_final_status_failure_is_routed_to_full_validation(self):
        d, _ = self.job(); (d / 'output').mkdir(); (d / 'output/result.json').write_text('{}')
        worker.save(d / 'dispatch.json', dict(at=time.time()))
        worker.save(d / 'status.json', dict(phase='failed'))
        observed = worker.observe(self.root, d)
        self.assertEqual(observed['phase'], 'complete-recovered')
        self.assertNotIn('scientific_success', observed)

    def recovery_monitor(self, d, max_attempts):
        """Use real dispatch/process locks; isolate bucket archive transport."""
        monitor = object.__new__(Monitor)
        monitor.root = self.root; monitor.remote = str(self.root); monitor.authority = self.authority
        monitor.config = dict(max_attempts=max_attempts, failed_cleanup_program=str(self.worker))
        monitor.sign = self.sign
        monitor.archive_failure = lambda job: self.sign(dict(job_id=job, full_readback_verified=True, scientific_success=False))
        monitor.stage = lambda directory, files: [Path(directory, name).write_bytes(raw) for name, raw in files.items()]
        def action(name, job=None, grant=None):
            if name == 'retire-failed':
                self.assertFalse(worker.observe(self.root, d)['lease_busy'])
                return dict(phase='complete')
            return self.dispatch(self.root / 'jobs' / job)
        monitor.worker = action; monitor.upload_job = lambda job: None
        def prepare(latest, attempt, previous):
            self.assertEqual(attempt, 2); self.assertEqual(previous['job_id'], d.name)
            other, _ = self.job(name=d.name + '-attempt-2')
            return other.name
        monitor.prepare = prepare
        (d / 'production-completion.json').write_text('{}')
        return monitor

    def test_transient_hydration_failure_retries_as_new_attempt_without_duplicate_evaluation(self):
        d, _ = self.job()
        (d / 'hydrate.py').write_text("raise OSError('transient fixture transport failure')\n")
        body = json.loads((d / 'assignment.json').read_bytes())['payload']
        body['program_files']['hydrate.py'] = worker.file_hash(d / 'hydrate.py')
        (d / 'assignment.json').write_bytes(worker.canonical(self.sign(body)))
        self.dispatch(d)
        self.wait(lambda: (d / 'status.json').exists() and json.loads((d / 'status.json').read_bytes())['phase'] == 'failed')
        self.wait(lambda: not worker.observe(self.root, d)['lease_busy'])
        self.assertFalse((d / 'started').exists())
        monitor = self.recovery_monitor(d, 3); state = dict(active=d.name, completed=[], completed_step=None)
        monitor.recover_failed(state, d.name)
        new = self.root / 'jobs' / state['active']
        self.assertNotEqual(new, d)
        self.wait(lambda: (new / 'output/result.json').exists())
        self.wait(lambda: not worker.observe(self.root, new)['lease_busy'])
        self.assertTrue((new / 'started').exists()); self.assertFalse((d / 'started').exists())
        self.assertEqual(json.loads((d / 'status.json').read_bytes())['phase'], 'failed')

    def test_exhausted_failure_is_recorded_and_allows_later_step_without_success(self):
        d, _ = self.job(); monitor = self.recovery_monitor(d, 1)
        state = dict(active=d.name, completed=[], completed_step=None)
        monitor.recover_failed(state, d.name)
        self.assertIsNone(state['active']); self.assertEqual(state['completed'], [])
        self.assertFalse(state['failed'][0]['scientific_success'])
        self.assertFalse(select_step(3, state['last_failed_step']))
        self.assertTrue(select_step(4, state['last_failed_step']))

    def test_program_tamper_fails_before_gpu_dispatch(self):
        d, _ = self.job(); (d / 'evaluate.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'pinned'): self.dispatch(d)
        self.assertFalse((d / 'dispatch.json').exists())

    def test_completed_cache_retirement_and_idempotent_retry(self):
        grant, model, d = self.retirement()
        with patch.object(worker, 'unused_model'):
            first = worker.retire(self.root, grant, self.authority)
            second = worker.retire(self.root, grant, self.authority)
        self.assertEqual(first, second); self.assertFalse(model.exists())
        self.assertTrue((d / 'plan.json').exists())

    def test_unarchived_cache_is_preserved(self):
        grant, model, _ = self.retirement(); raw = json.loads(grant.read_bytes()); p = raw['payload']
        a = p['archive']['payload']; a['full_readback_verified'] = False; p['archive'] = self.sign(a)
        grant.write_bytes(worker.canonical(self.sign(p)))
        with self.assertRaises(ValueError): worker.retire(self.root, grant, self.authority)
        self.assertTrue(model.exists())

    def test_global_lease_prevents_retirement(self):
        grant, model, _ = self.retirement()
        with worker.lease(self.root):
            with self.assertRaises(BlockingIOError): worker.retire(self.root, grant, self.authority)
        self.assertTrue(model.exists())

    def test_evaluation_lease_prevents_retirement(self):
        grant, model, d = self.retirement()
        with (d / 'evaluation.lease').open('rb') as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError): worker.retire(self.root, grant, self.authority)
        self.assertTrue(model.exists())

    def test_current_different_job_cannot_lose_same_checkpoint(self):
        grant, model, _ = self.retirement()
        worker.save(self.root / 'active.json', dict(job_id='step-4-other', checkpoint=model.name))
        with self.assertRaisesRegex(ValueError, 'another current job'): worker.retire(self.root, grant, self.authority)
        self.assertTrue(model.exists())

    def test_partial_retirement_recovers_without_deleting_unknown_files(self):
        grant, model, _ = self.retirement(); original = Path.unlink; count = 0
        def interrupted(p, *args, **kwargs):
            nonlocal count
            if p.parent == model:
                count += 1
                if count == 2: raise OSError('crash after first unlink')
            return original(p, *args, **kwargs)
        with patch.object(worker, 'unused_model'), patch.object(Path, 'unlink', interrupted):
            with self.assertRaises(OSError): worker.retire(self.root, grant, self.authority)
        with patch.object(worker, 'unused_model'):
            result = worker.retire(self.root, grant, self.authority)
        self.assertEqual(result['phase'], 'complete'); self.assertFalse(model.exists())

    def test_changed_or_extra_cache_file_preserved(self):
        grant, model, _ = self.retirement(); (model / 'unexpected').write_text('keep')
        with self.assertRaises(ValueError): worker.retire(self.root, grant, self.authority)
        self.assertTrue((model / 'unexpected').exists())

    def test_base_model_outside_scoped_directory_is_never_retired(self):
        grant, model, _ = self.retirement(); raw = json.loads(grant.read_bytes())['payload']
        plan = raw['evaluation_plan']['payload']; plan['checkpoint']['path'] = str(self.root.parent / 'base')
        raw['evaluation_plan'] = self.sign(plan); archive = raw['archive']['payload']
        archive['plan_sha256'] = worker.digest(worker.canonical(raw['evaluation_plan'])); raw['archive'] = self.sign(archive)
        grant.write_bytes(worker.canonical(self.sign(raw)))
        with self.assertRaisesRegex(ValueError, 'base is protected'): worker.retire(self.root, grant, self.authority)
        self.assertTrue(model.exists())


if __name__ == '__main__': unittest.main()
