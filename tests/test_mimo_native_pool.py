"""Scheduling controls; real native originals are benchmarked separately.

Thread fixtures below use only the signal-free fake math session. Production
uses spawn processes, exercised here with main-thread signal isolation.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import os
import signal
import time
import unittest

from subnet.audit_policy import InvalidSample
from subnet.mimo_native_pool import NativeGradingPool, bounded_ordered_map
from test_mimo_coordinator import CoordinatorIntegrationTests


def isolated_probe(request):
    # A native grader must be free to install process-local alarms; this fails
    # in a worker thread and must not alter the coordinator's signal handlers.
    original = signal.signal(signal.SIGALRM, signal.SIG_IGN)
    start = time.monotonic()
    time.sleep(request['sleep'])
    signal.signal(signal.SIGALRM, original)
    return dict(slot=request['slot'], pid=os.getpid(), start=start, end=time.monotonic())


class NativePoolSchedulingTests(unittest.TestCase):
    def test_missing_tokenizer_file_or_manifest_pin_fails_before_loading(self):
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from subnet.mimo_native_pool import TOKENIZER_FILES, authenticated_tokenizer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in TOKENIZER_FILES:
                (root/name).write_text('fixture')
            pins = {name: '0'*64 for name in TOKENIZER_FILES}
            with patch('transformers.AutoTokenizer.from_pretrained', side_effect=AssertionError('must not load')):
                missing = dict(pins); missing.pop('tokenizer.json')
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    authenticated_tokenizer(root, missing, '0'*64)
                (root/'tokenizer.json').unlink()
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    authenticated_tokenizer(root, pins, '0'*64)

    def test_spawn_isolation_parallel_overlap_and_original_result_order(self):
        requests = [dict(slot=i, sleep=.5 if i == 0 else .15) for i in range(6)]
        original = signal.getsignal(signal.SIGALRM)
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
            rows = list(bounded_ordered_map(pool, isolated_probe, iter(requests), 3))
        self.assertEqual([request for request, _ in rows], requests)
        results = [result for _, result in rows]
        self.assertEqual([r['slot'] for r in results], list(range(6)))
        self.assertGreaterEqual(len({r['pid'] for r in results}), 2)
        self.assertTrue(any(a['pid'] != b['pid'] and max(a['start'], b['start']) < min(a['end'], b['end'])
                            for a in results for b in results))
        self.assertEqual(signal.getsignal(signal.SIGALRM), original)

    def test_prefetch_is_bounded_and_exception_stops_population_consumption(self):
        consumed = []
        def inputs():
            for i in range(100):
                consumed.append(i)
                yield i
        def grade(i):
            if i == 0:
                raise OSError('native unavailable')
            return i
        with ThreadPoolExecutor(max_workers=3) as pool:
            mapped = bounded_ordered_map(pool, grade, inputs(), 3)
            with self.assertRaisesRegex(OSError, 'native unavailable'):
                next(mapped)
        self.assertEqual(consumed, [0, 1, 2])

    def test_early_close_does_not_consume_remaining_documents(self):
        consumed = []
        def inputs():
            for i in range(100):
                consumed.append(i)
                yield i
        with ThreadPoolExecutor(max_workers=2) as pool:
            mapped = bounded_ordered_map(pool, lambda i: i, inputs(), 2)
            self.assertEqual(next(mapped), (0, 0))
            mapped.close()
        self.assertEqual(consumed, [0, 1])

    def test_worker_count_is_explicit_bounded_integer(self):
        for value in (0, 33, True, 2.5):
            with self.assertRaises(ValueError):
                NativeGradingPool('/not-loaded', {}, '0'*64, {}, workers=value)
        self.assertEqual(NativeGradingPool('/not-loaded', {}, '0'*64, {}).workers, 8)


class ParallelCoordinatorIntegrationTests(CoordinatorIntegrationTests):
    """Run all original seal/recovery/collision controls through ordered mapping."""
    def make_coordinator(self):
        coordinator = super().make_coordinator()
        def mapped(inputs, manifest):
            def grade(request):
                miner, _, batch = request
                try:
                    return coordinator.grade_batch(batch, manifest, miner=miner)
                except InvalidSample as error:
                    return error
            with ThreadPoolExecutor(max_workers=3) as pool:
                yield from bounded_ordered_map(pool, grade, inputs, 3)
        coordinator.grade_batches = mapped
        return coordinator

    def test_parallel_dispatch_equals_serial_byte_for_byte_with_same_frozen_population(self):
        import json
        import shutil
        from unittest.mock import patch
        from subnet.mimo_coordinator import Coordinator
        first = self.actor(self.miners[0]); second = self.actor(self.miners[1])
        for index in (0, 1, 2):
            self.upload(first, index)
        self.upload(second, 0)
        self.upload(second, 3, lie=True)
        # Freeze exact population/selection through a temporary deferred grader.
        with patch.object(self.coordinator, 'grade_batch', side_effect=OSError('defer')):
            with self.assertRaises(OSError):
                self.seal(max_tasks=1)
        serial_root = self.root/'serial'
        shutil.copytree(self.coordinator.root, serial_root)
        serial = Coordinator(serial_root, self.bucket, self.authority,
            resolve_task=self.coordinator.resolve_task, grade_batch=self.coordinator.grade_batch)
        serial_result = serial.seal(max_tasks=1, now=self.opened['deadline'])
        parallel_result = self.coordinator.seal(max_tasks=1, now=self.opened['deadline'])
        self.assertEqual(serial_result, parallel_result)
        self.assertEqual((serial_root/'DISPATCH.json').read_bytes(),
                         (self.coordinator.root/'DISPATCH.json').read_bytes())
        self.assertEqual(len(parallel_result['payload']['excluded']), 4)


if __name__ == '__main__':
    unittest.main()
