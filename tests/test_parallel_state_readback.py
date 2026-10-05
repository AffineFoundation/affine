"""Transport concurrency controls; scientific report validation is a boundary."""
import hashlib
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from subnet.persistent_training_protocol import independently_commit


class ParallelStateReadbackTests(unittest.TestCase):
    def fixture(self):
        data = {f'state-{n:06}.safetensors': bytes([n]) * 32 for n in range(12)}
        descriptor = {'shards': [dict(name=name, size=len(body),
                         sha256=hashlib.sha256(body).hexdigest())
                         for name, body in data.items()]}
        controller = SimpleNamespace(bucket=Mock(), authority=SimpleNamespace(id='approved'),
                                     signed=lambda value: {'payload': value})
        job = {'job_id': 'original', 'persistent_training': {'output_namespace': 'original-state'}}
        return data, descriptor, controller, job

    def run_commit(self, descriptor, controller, job, chunks, workers=4):
        # This test supplies a prevalidated descriptor. Real lineage/source and
        # signature validation remains covered by persistent integration tests.
        from botocore.exceptions import ClientError
        missing = ClientError({'Error': {'Code': 'NoSuchKey'}}, 'GetObject')
        def read(bucket, key):
            if key.endswith('/staged-state.json'):
                return descriptor
            if controller.bucket.json.call_count == 0:
                raise missing
            return controller.bucket.json.call_args.args[1]
        with patch('subnet.persistent_training_protocol.validate_report', return_value=descriptor), \
             patch('subnet.persistent_training_protocol.read_json', side_effect=read), \
             patch('subnet.backend_jobs.signed', side_effect=lambda obj, authority: obj['payload']), \
             patch('subnet.persistent_training_protocol.state_pointer', side_effect=lambda obj: obj):
            return independently_commit(controller, {}, job, {}, read_chunks=chunks,
                                        readback_workers=workers)

    def test_bounded_concurrency_reads_every_shard_before_signing(self):
        data, descriptor, controller, job = self.fixture()
        lock = threading.Lock()
        first_wave = threading.Barrier(4, timeout=3)
        active = maximum = 0
        done = set()
        def chunks(key):
            nonlocal active, maximum
            name = key.rsplit('/', 1)[-1]
            with lock:
                active += 1
                maximum = max(maximum, active)
            try:
                if name in list(data)[:4]:
                    first_wave.wait()
                self.assertEqual(controller.bucket.json.call_count, 0)
                yield data[name][:16]
                yield data[name][16:]
            finally:
                with lock:
                    done.add(name)
                    active -= 1
        self.run_commit(descriptor, controller, job, chunks)
        self.assertEqual(maximum, 4)
        self.assertEqual(done, set(data))
        self.assertEqual(active, 0)
        controller.bucket.json.assert_called_once()

    def test_corrupt_shard_never_signs_partial_success(self):
        data, descriptor, controller, job = self.fixture()
        def chunks(key):
            name = key.rsplit('/', 1)[-1]
            yield b'corrupt' if name == list(data)[-1] else data[name]
        with self.assertRaisesRegex(ValueError, 'state integrity'):
            self.run_commit(descriptor, controller, job, chunks)
        controller.bucket.json.assert_not_called()

    def test_concurrency_bound_is_strict(self):
        for value in (0, 9, True, 4.0):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'concurrency'):
                independently_commit(None, None, None, None, readback_workers=value)


if __name__ == '__main__':
    unittest.main()
