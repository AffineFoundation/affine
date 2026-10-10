import ast
import base64
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest import mock

from dashboard import test_training_evidence_projection as fixtures
from ops import collect_finalized_trainer_logs as collector


class CollectorTests(unittest.TestCase):
    setUp = fixtures.TrainingEvidenceTests.setUp
    tearDown = fixtures.TrainingEvidenceTests.tearDown
    signed = fixtures.TrainingEvidenceTests.signed
    write = fixtures.TrainingEvidenceTests.write
    save = fixtures.TrainingEvidenceTests.save

    def prepare(self):
        self.cache = self.state/'training-cache'
        self.logs = self.state/'private-logs'
        self.write(self.state/(self.epoch+'-first-signed-manifest.json'), self.record['manifest_envelope'])
        self.write(self.state/(self.epoch+'-signed-learner-completion.json'), self.record['completion_envelope'])
        self.write(self.cache/(self.epoch+'.signed.json'), self.record['training_envelope'])
        self.remote = {'workspace': '/isolated/workspace', 'known_hosts': '/isolated/known_hosts',
                       'port': 22, 'host': 'example.invalid', 'python': '/isolated/python'}
        self.calls = 0

    def transport(self, argv, **kwargs):
        self.calls += 1
        self.assertEqual(kwargs['timeout'], 40)
        self.assertIn('StrictHostKeyChecking=yes', argv)
        data = ast.literal_eval(kwargs['input'].splitlines()[0].removeprefix('DATA = '))
        result = []
        for row in data['rows']:
            raw = b'private warning token=SECRET\n'
            result.append(dict(row, status='available', retrieved_at=1, raw_sha256=collector.sha(raw),
                               raw_size=len(raw), original_job_bytes_match=True,
                               raw_base64=base64.b64encode(raw).decode()))
        return SimpleNamespace(returncode=0, stdout=json.dumps(result))

    def collect(self, transport=None):
        return collector.collect(self.state, self.cache, self.logs, self.remote, self.authority,
                                 run=transport or self.transport)

    def test_exact_finalized_original_cached_once_and_private(self):
        self.prepare()
        self.assertEqual(self.collect()['retrieved'], 1)
        self.assertEqual(self.collect()['retrieved'], 0)
        self.assertEqual(self.calls, 1)
        self.assertEqual((self.logs/(self.epoch+'.worker.log')).stat().st_mode & 0o777, 0o600)

    def test_no_signed_completion_means_no_remote_read(self):
        self.prepare()
        (self.state/(self.epoch+'-signed-learner-completion.json')).unlink()
        self.assertEqual(self.collect()['retrieved'], 0)
        self.assertEqual(self.calls, 0)

    def test_missing_log_backoff_does_not_block_publication(self):
        self.prepare()
        def missing(argv, **kwargs):
            data = ast.literal_eval(kwargs['input'].splitlines()[0].removeprefix('DATA = '))
            return SimpleNamespace(returncode=0, stdout=json.dumps([
                dict(row, status='not_retained_on_current_trainer') for row in data['rows']]))
        result = self.collect(missing)
        self.assertEqual(result['retrieved'], 0)
        self.assertEqual(len(result['missing']), 1)
        self.assertEqual(self.collect()['status'], 'no_uncached_finalized_logs')
        self.assertEqual(self.calls, 0)

    def test_timeout_is_availability_not_epoch_failure(self):
        self.prepare()
        def failed(*args, **kwargs):
            raise subprocess.TimeoutExpired('private command', 40)
        result = self.collect(failed)
        self.assertEqual(result['status'], 'remote_log_read_unavailable')
        self.assertNotIn('private command', json.dumps(result))

    def test_wrong_job_response_cannot_enter_cache(self):
        self.prepare()
        def wrong(argv, **kwargs):
            result = self.transport(argv, **kwargs)
            rows = json.loads(result.stdout)
            rows[0]['job_id'] += '-wrong'
            result.stdout = json.dumps(rows)
            return result
        self.assertEqual(self.collect(wrong)['retrieved'], 0)
        self.assertFalse((self.logs/(self.epoch+'.worker.log')).exists())

    def test_stale_failed_role_pointer_does_not_choose_log(self):
        self.prepare()
        self.write(self.state/'roles'/(self.epoch+'-train.json'), {'job_id': 'failed-original'})
        self.assertEqual(self.collect()['retrieved'], 1)
        receipt = json.loads((self.logs/(self.epoch+'.receipt.json')).read_bytes())
        self.assertEqual(receipt['job_id'], self.job_id)

    def test_invalid_receipt_is_skipped_without_printing_private_details(self):
        self.prepare()
        document = self.record['training_envelope']
        document['payload']['remote_job_id'] = '../SECRET'
        self.write(self.cache/(self.epoch+'.signed.json'), document)
        result = self.collect()
        self.assertEqual(result['retrieved'], 0)
        self.assertNotIn('SECRET', json.dumps(result))
        self.assertEqual(self.calls, 0)

    def test_existing_private_log_never_overwritten(self):
        self.prepare()
        self.logs.mkdir()
        path = self.logs/(self.epoch+'.worker.log')
        path.write_bytes(b'original-private-bytes')
        self.assertEqual(self.collect()['retrieved'], 0)
        self.assertEqual(path.read_bytes(), b'original-private-bytes')

    def test_atomic_cache_publishes_only_complete_bytes(self):
        self.prepare()
        self.logs.mkdir()
        path = self.logs/'test.worker.log'
        original_link = collector.os.link
        def observe(source, destination):
            self.assertFalse(Path(destination).exists())
            self.assertEqual(Path(source).read_bytes(), b'complete-log')
            original_link(source, destination)
        with mock.patch.object(collector.os, 'link', side_effect=observe):
            collector.immutable_file(path, b'complete-log')
        self.assertEqual(path.read_bytes(), b'complete-log')
        self.assertFalse(list(self.logs.glob('.*.tmp')))

    def test_concurrently_created_different_final_is_preserved(self):
        self.prepare()
        self.logs.mkdir()
        path = self.logs/'test.worker.log'
        original_link = collector.os.link
        def concurrent(source, destination):
            Path(destination).write_bytes(b'other-writer')
            original_link(source, destination)
        with mock.patch.object(collector.os, 'link', side_effect=concurrent):
            with self.assertRaisesRegex(ValueError, 'immutable_log_cache_collision'):
                collector.immutable_file(path, b'new-log')
        self.assertEqual(path.read_bytes(), b'other-writer')
        self.assertFalse(list(self.logs.glob('.*.tmp')))

    def test_concurrently_created_identical_final_is_idempotent(self):
        self.prepare()
        self.logs.mkdir()
        path = self.logs/'test.worker.log'
        original_link = collector.os.link
        def concurrent(source, destination):
            Path(destination).write_bytes(b'same-log')
            original_link(source, destination)
        with mock.patch.object(collector.os, 'link', side_effect=concurrent):
            collector.immutable_file(path, b'same-log')
        self.assertEqual(path.read_bytes(), b'same-log')
        self.assertFalse(list(self.logs.glob('.*.tmp')))


if __name__ == '__main__':
    unittest.main()
