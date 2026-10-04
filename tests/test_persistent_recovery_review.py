"""Independent synthetic recovery controls; no remote commands or HTTP calls."""
import copy
import io
import json
import unittest
from unittest.mock import Mock, patch

import test_persistent_training_integration as fixtures
from subnet.persistent_cpu_adamw import sha
from subnet.persistent_training_controller import train
from subnet.persistent_training_protocol import independently_commit, read_json, state_pointer
from subnet.remote_backend import RemoteJobs, save


class PersistentRecoveryReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PersistentIntegrationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def prepared_controller(self):
        f = self.fixture
        report, job = f.report()
        epoch = f.manifest['epoch']
        save(f.root/(epoch+'-scores.json'), {'receipts': f.receipts})
        save(f.root/(epoch+'-audit-challenge.json'), f.challenge)
        record = dict(job_id=job['job_id'], role='train', job_sha256=sha(job),
                      manifest_sha256=sha(f.manifest), source_files=job['source_files'],
                      runtime_versions=job['runtime_versions'])
        save(f.root/'roles'/(epoch+'-train.json'), record)
        save(f.root/'roles'/(job['job_id']+'-job.json'), f.sign(job))
        save(f.root/'roles'/(job['job_id']+'-report.json'), report)
        jobs = RemoteJobs.__new__(RemoteJobs)
        jobs.controller = f.controller
        jobs.state = f.root/'roles'
        jobs.command = Mock(side_effect=AssertionError('recovery must not execute a remote command'))
        jobs.remote_status = Mock(side_effect=AssertionError('completed original report needs no remote probe'))
        f.controller.jobs = jobs

        def publish(manifest, path):
            save(f.root/(epoch+'-checkpoint-publication.json'),
                 dict(checkpoint=f.cp['id'], operator_independent_hashes=True,
                      objects={name: {'sha256': digest} for name, digest in f.cp['files'].items()}))
            return copy.deepcopy(f.cp)

        f.controller.publish_remote_checkpoint = Mock(side_effect=publish)
        f.controller.checkpoint_with_reads = lambda cp: cp
        return report, job, {'miner': report['audits'][0]}, jobs

    def test_interruption_after_state_journal_before_metrics_reuses_original_job(self):
        f = self.fixture
        report, job, reports, jobs = self.prepared_controller()
        metrics_path = f.root/(f.manifest['epoch']+'-training-metrics.json')

        def fail_metrics(path, value):
            if path == metrics_path:
                raise OSError('synthetic interrupted metrics publication')
            return save(path, value)

        with patch('subnet.remote_backend.save', side_effect=fail_metrics):
            with self.assertRaisesRegex(OSError, 'synthetic interrupted'):
                train(f.controller, f.manifest, reports, '/synthetic/input', steps=3)
        self.assertFalse(metrics_path.exists())
        first_pointer = json.loads((f.root/'latest-trainer-state.json').read_text())
        self.assertEqual(first_pointer['optimizer_steps'], 3)
        self.assertEqual(first_pointer['inference_checkpoint'], f.cp['id'])

        checkpoint, recovered = train(f.controller, f.manifest, reports, '/synthetic/input', steps=3)
        self.assertEqual(recovered['trainer_state'], first_pointer)
        self.assertEqual(recovered['original_job_sha256'], sha(job))
        self.assertEqual(checkpoint['id'], f.cp['id'])
        self.assertFalse(recovered['weights_changed'])
        self.assertEqual(recovered['trainer_state']['optimizer_steps'], 3)
        jobs.command.assert_not_called()
        jobs.remote_status.assert_not_called()
        self.assertEqual(json.loads(metrics_path.read_text())['trainer_state'], first_pointer)
        authoritative_writes = [event for event in f.bucket.events
                                if event == ('authority-write', first_pointer['descriptor_key'])]
        self.assertEqual(len(authoritative_writes), 1)

    def test_incomplete_independent_readback_cannot_sign_state_or_commit_journal(self):
        f = self.fixture
        report, job = f.report()
        namespace = job['persistent_training']['output_namespace']

        def truncated(key):
            data = f.bucket.get(key)
            yield data[:-1]

        with self.assertRaisesRegex(ValueError, 'independent trainer state integrity'):
            independently_commit(f.controller, report, job, f.manifest, read_chunks=truncated)
        self.assertNotIn(namespace+'/authority-state.json', f.bucket.objects)
        self.assertFalse((f.root/'latest-trainer-state.json').exists())
        # Retained durable shards can be read again; retry is publication only.
        pointer = independently_commit(f.controller, report, job, f.manifest)
        self.assertEqual(pointer['optimizer_steps'], 3)
        self.assertEqual(pointer['inference_checkpoint'], f.cp['id'])

    def test_descriptor_length_mismatch_is_detected_and_body_closed(self):
        body = io.BytesIO(b'{}')
        bucket = Mock()
        bucket.name = 'synthetic'
        bucket.client.get_object.return_value = {'Body': body, 'ContentLength': 3}
        with self.assertRaisesRegex(ValueError, 'truncated persistent descriptor'):
            read_json(bucket, 'synthetic/state.json')
        self.assertTrue(body.closed)

    def test_cached_state_publication_must_be_from_the_original_signed_job(self):
        f = self.fixture
        _, job, reports, jobs = self.prepared_controller()
        _, metrics = train(f.controller, f.manifest, reports, '/synthetic/input', steps=3)
        original_pointer = metrics['trainer_state']
        publication = json.loads(f.bucket.get(original_pointer['descriptor_key']))['payload']
        # Model/descriptor/counters are identical; only original job provenance
        # differs. A restored local pointer must not silently choose this state.
        alternate = copy.deepcopy(publication)
        alternate['job_id'] = 'separate-synthetic-request'
        alternate['job_sha256'] = 'e'*64
        alternate['namespace'] = 'private/trainer-state/'+f.manifest['epoch']+'/'+alternate['job_id']
        alternate_pointer = state_pointer(alternate)
        f.bucket.json(alternate_pointer['descriptor_key'], f.sign(alternate))
        changed = dict(metrics, trainer_state=alternate_pointer)
        save(f.root/(f.manifest['epoch']+'-training-metrics.json'), changed)
        save(f.root/'latest-trainer-state.json', alternate_pointer)
        with self.assertRaises(ValueError):
            train(f.controller, f.manifest, reports, '/synthetic/input', steps=3)
        jobs.command.assert_not_called()
        self.assertEqual(original_pointer['descriptor_sha256'], alternate_pointer['descriptor_sha256'])
        self.assertNotEqual(original_pointer['namespace'], alternate_pointer['namespace'])


if __name__ == '__main__':
    unittest.main()
