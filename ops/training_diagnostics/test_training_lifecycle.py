import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import train_archive as archive
import train_contract as c
import serial_worker as worker
from monitor import Monitor, health_status
from test_training_eval import AUTH, fixture, sign


def evidence():
    p, rows = fixture(); envelope = sign(p)
    files = {'task-' + str(r['index']) + '.json': c.canonical(r) for r in rows}
    result = dict(version=p['version'], plan_sha256=c.digest(envelope), checkpoint=p['checkpoint']['id'],
                  tasks=128, cohort_sha256=p['cohort_sha256'], profile_sha256=p['profile_sha256'],
                  seeds_sha256=p['seeds_sha256'], raw_artifacts={n: archive.sha(b) for n, b in files.items()},
                  correct=96, errors=0, unresolved=0, accuracy=96/128, completed_at=1791792000.)
    files['result.json'] = c.canonical(result)
    return envelope, files


class MemoryBucket:
    name = 'synthetic'
    def __init__(self): self.client = self; self.objects = {}; self.fail = False; self.read = []
    def put(self, key, raw, content_type): self.objects[key] = raw
    def get_object(self, *, Bucket, Key):
        raw = self.objects[Key]
        if self.fail and Key.endswith('task-7.json'): raw = b'corrupt'
        self.read.append(Key)
        return dict(ContentLength=len(raw), Body=io.BytesIO(raw))


class LifecycleTests(unittest.TestCase):
    def test_held_queue_health_is_not_green(self):
        for flag in ('baseline_failed','queue_failed','dispatch_hold'):
            h=health_status({flag:True})
            self.assertFalse(h['healthy']);self.assertEqual(h['status'],'held')
        self.assertEqual(health_status({'active':'original'})['status'],'active')
        self.assertTrue(health_status({'active':None})['healthy'])

    def test_archive_rejects_changed_or_selectively_missing_record(self):
        envelope, files = evidence(); archive.validate(envelope, files, AUTH)
        bad = dict(files); bad['task-3.json'] += b' '
        with self.assertRaisesRegex(ValueError, 'raw task hash'): archive.validate(envelope, bad, AUTH)
        files.pop('task-3.json')
        with self.assertRaisesRegex(ValueError, 'no selective exclusions'): archive.validate(envelope, files, AUTH)

    def make_capture(self, root):
        envelope, files = evidence(); local = root / 'jobs/step-0-test'; local.mkdir(parents=True)
        (local / 'plan.json').write_bytes(c.canonical(envelope))
        for name in ('evaluate.py', 'train_contract.py'): (local/name).write_bytes(b'# synthetic source\n')
        bucket = MemoryBucket()
        monitor = types.SimpleNamespace(root=root, remote='/synthetic/queue', authority=AUTH, bucket=bucket,
            config=dict(archive_prefix='synthetic-only', aggregate_directory=str(root/'public')),
            full_readback=archive.full_readback, sign=sign,
            ssh_code=lambda code: dict(files={n:base64.b64encode(raw).decode() for n,raw in files.items()}))
        return monitor

    def test_publish_only_after_every_raw_full_get_and_repeat_idempotently(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = self.make_capture(Path(tmp)); ack, aggregate = archive.capture_and_archive(monitor, 'step-0-test')
            self.assertEqual(len(list((Path(tmp)/'public').glob('*.signed.json'))), 1)
            self.assertTrue(aggregate['durable_result']['full_get_verified'])
            self.assertEqual(len([k for k in monitor.bucket.read if Path(k).stem.removeprefix('task-').isdecimal()]),128)
            before = next((Path(tmp)/'public').glob('*.signed.json')).read_bytes()
            archive.capture_and_archive(monitor, 'step-0-test')
            self.assertEqual(before,next((Path(tmp)/'public').glob('*.signed.json')).read_bytes())

    def test_one_failed_full_get_prevents_public_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = self.make_capture(Path(tmp)); monitor.bucket.fail = True
            with self.assertRaisesRegex(ValueError,'full byte readback'):
                archive.capture_and_archive(monitor, 'step-0-test')
            self.assertFalse((Path(tmp)/'public').exists())
            self.assertFalse((Path(tmp)/'jobs/step-0-test/archive.json').exists())

    def test_dispatch_crash_intent_never_launches_another_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); directory=root/'jobs/step-4-test'; directory.mkdir(parents=True)
            path=directory/'assignment.json'; path.write_text('{}')
            p=dict(job_id=directory.name,checkpoint='synthetic')
            with mock.patch.object(worker,'assignment',return_value=(p,directory)), mock.patch.object(worker.subprocess,'Popen',side_effect=RuntimeError('ambiguous launch')) as launch:
                with self.assertRaises(RuntimeError):worker.dispatch(root,path,AUTH,'/synthetic/python')
                result=worker.dispatch(root,path,AUTH,'/synthetic/python')
                self.assertTrue(result['reused_original_dispatch']);self.assertEqual(launch.call_count,1)

    def test_serial_lease_excludes_second_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with worker.lease(Path(tmp)):
                with self.assertRaises(BlockingIOError):
                    with worker.lease(Path(tmp)):pass

    def test_baseline_error_archived_but_prevents_current_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); monitor=Monitor.__new__(Monitor);monitor.root=root;monitor.remote='/synthetic'
            monitor.config=dict(worker_program=__file__,failed_cleanup_program=__file__)
            monitor.stage=lambda *a:None;monitor.upload_job=lambda *a:None
            monitor.worker=lambda *a,**kw:dict(phase='complete',result_present=True,lease_busy=False)
            monitor.capture=lambda job:({},dict(phase='baseline',errors=1,comparability_passed=True,checkpoint='synthetic',optimizer_step=0))
            worker.save(root/'state.json',dict(active='step-0-test',completed_step=None,completed=[]))
            monitor.tick(); state=json.loads((root/'state.json').read_bytes())
            self.assertTrue(state['baseline_failed']); self.assertIsNone(state['active'])
            monitor.prepare=lambda *a: self.fail('unusable reference must not launch latest')
            monitor.tick()

    def test_original_upstream_reference_cannot_be_retired(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p,rows=fixture()
            p['checkpoint']['id']='2173b3e0119380ab813c9588bedff6b3edd5d3dcb01565490fa33ac67b2debf6'
            p['checkpoint']['path']=str(root/'checkpoints'/p['checkpoint']['id'])
            grant=sign(dict(version='archived-training128-model-retirement-v1',root=str(root),
                            evaluation_plan=sign(p),archive=sign({})))
            path=root/'grant.json';path.write_bytes(c.canonical(grant))
            with self.assertRaisesRegex(ValueError,'reference remains protected'):worker.retire(root,path,AUTH)

    def test_expiring_pod_does_not_dispatch_new_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); monitor=Monitor.__new__(Monitor);monitor.root=root;monitor.remote='/synthetic'
            monitor.config=dict(worker_program=__file__,failed_cleanup_program=__file__,baseline_checkpoint={},lease_expires_at=0)
            monitor.template=dict(epoch='synthetic-epoch91');monitor.stage=lambda *a:None
            monitor.prepare=lambda *a:self.fail('dispatch after bounded lease')
            monitor.tick();state=json.loads((root/'state.json').read_bytes())
            self.assertIn('lease',state['dispatch_hold']);self.assertIsNone(state['active'])

    def test_any_failure_hold_blocks_future_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);monitor=Monitor.__new__(Monitor);monitor.root=root;monitor.remote='/synthetic'
            monitor.config=dict(worker_program=__file__,failed_cleanup_program=__file__)
            monitor.stage=lambda *a:None
            monitor.latest=lambda:self.fail('no current checkpoint dispatch after uncertain cleanup')
            worker.save(root/'state.json',dict(active=None,completed_step=67,completed=[],baseline_job='original-base',queue_failed=True))
            monitor.tick()

    def test_failed_original_archived_once_then_holds_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);local=root/'jobs/step-67-test';local.mkdir(parents=True)
            assignment=sign(dict(attempt=1,checkpoint='synthetic',step=67));worker.save(local/'assignment.json',assignment)
            worker.save(local/'read-plan.json',{})
            monitor=Monitor.__new__(Monitor);monitor.root=root;monitor.remote='/synthetic';monitor.authority=AUTH
            monitor.config=dict(max_attempts=1,failed_cleanup_program=__file__)
            monitor.sign=sign;monitor.stage=lambda *a:None;monitor.worker=lambda *a,**kw:dict(phase='complete')
            monitor.archive_failure=lambda job:sign(dict(full_readback_verified=True))
            state=dict(active='step-67-test',completed_step=None,completed=[])
            monitor.recover_failed(state,'step-67-test')
            self.assertTrue(state['queue_failed']);self.assertIsNone(state['active'])


if __name__=='__main__':unittest.main()
