"""Integrity, race, retry and memory bounds for parallel frozen publication."""
import datetime
import hashlib
import io
import threading
import time
import unittest
from botocore.exceptions import ClientError
from subnet.storage import Bucket, Gateway, Identity, sha


class Body(io.BytesIO):
    def __init__(self, data):
        super().__init__(data); self.read_sizes = []

    def read(self, size=-1):
        assert 0 < size <= 1024 * 1024, 'must stream bounded chunks'
        self.read_sizes.append(size)
        return super().read(size)


class Client:
    def __init__(self):
        self.objects = {}; self.bodies = []; self.copies = []
        self.mutate_before_copy = False; self.copy_failure = False
        self.get_barrier = None; self.active = 0; self.peak = 0
        self.lock = threading.Lock()

    def put_object(self, **args):
        self.objects[args['Key']] = args['Body']

    def get_object(self, **args):
        data = self.objects[args['Key']]; body = Body(data); self.bodies.append(body)
        if self.get_barrier:
            with self.lock:
                self.active += 1; self.peak = max(self.peak, self.active)
            self.get_barrier.wait(timeout=3)
            with self.lock: self.active -= 1
        return dict(Body=body, ContentLength=len(data), ETag=sha(data))

    def copy_object(self, **args):
        source = args['CopySource']['Key']
        if self.mutate_before_copy: self.objects[source] = b'changed after read'
        if sha(self.objects[source]) != args['CopySourceIfMatch']:
            raise ClientError({'Error': {'Code': 'PreconditionFailed'}}, 'CopyObject')
        if self.copy_failure: raise IOError('copy unavailable')
        self.objects[args['Key']] = self.objects[source]
        self.copies.append((source, args['Key']))

    def generate_presigned_url(self, *args, **kwargs):
        return 'https://unused.invalid/frozen'


def bucket():
    b = Bucket.__new__(Bucket); b.name = 'test'; b.client = Client(); return b


class FrozenPublication(unittest.TestCase):
    def test_large_body_is_streamed_and_closed_before_conditional_copy(self):
        b = bucket(); data = b'ab' * (1024 * 1024 + 11)
        b.put('frozen', data); b.verified_copy('frozen', 'public', sha(data), len(data))
        self.assertEqual(b.client.objects['public'], data)
        self.assertTrue(b.client.bodies[0].closed)
        self.assertGreater(len(b.client.bodies[0].read_sizes), 2)

    def test_corrupted_body_never_copied_and_stream_closes(self):
        b = bucket(); b.put('frozen', b'wrong')
        with self.assertRaisesRegex(ValueError, 'receipt hash changed'):
            b.verified_copy('frozen', 'public', sha(b'right'), 5)
        self.assertEqual(b.client.copies, []); self.assertTrue(b.client.bodies[0].closed)

    def test_size_change_refused_before_copy(self):
        b = bucket(); b.put('frozen', b'data')
        with self.assertRaisesRegex(ValueError, 'receipt size changed'):
            b.verified_copy('frozen', 'public', sha(b'data'), 5)
        self.assertEqual(b.client.copies, []); self.assertTrue(b.client.bodies[0].closed)

    def test_read_copy_race_fails_condition_instead_of_publishing_replacement(self):
        b = bucket(); b.put('frozen', b'original'); b.client.mutate_before_copy = True
        with self.assertRaises(ClientError):
            b.verified_copy('frozen', 'public', sha(b'original'), 8)
        self.assertNotIn('public', b.client.objects)

    def test_truncated_stream_refused(self):
        b = bucket(); b.put('frozen', b'data'); original = b.client.get_object
        def truncated(**args):
            response = original(**args); response['ContentLength'] = 5; return response
        b.client.get_object = truncated
        with self.assertRaisesRegex(ValueError, 'receipt hash changed'):
            b.verified_copy('frozen', 'public', sha(b'data'), 5)
        self.assertEqual(b.client.copies, [])

    def prepared(self, workers=2):
        b = bucket(); g = Gateway(b, direct_r2=True, publication_workers=workers)
        epoch = 'nonpayable-publish'; ids = [Identity().id for _ in range(4)]
        g.open(epoch, ids, int(time.time()) + 100)
        snapshots = {}
        for miner in ids:
            data = miner.encode(); key = 'private/' + epoch + '/frozen/' + miner + '/' + sha(data) + '.zip'
            b.put(key, data)
            snapshots[miner] = dict(key='mutable-staging', snapshot_key=key,
                sha256=sha(data), size=len(data), received_at=g.epochs[epoch]['start'])
        g.epochs[epoch]['snapshots'] = snapshots
        return b, g, epoch

    def test_parallel_reads_are_bounded_and_all_frozen_hashes_retained(self):
        b, g, epoch = self.prepared()
        try:
            b.client.get_barrier = threading.Barrier(2)
            receipts = g.freeze(epoch)
            self.assertEqual(b.client.peak, 2); self.assertEqual(len(receipts), 4)
            for miner, receipt in receipts.items():
                self.assertEqual(sha(b.client.objects[receipt['frozen_key']]), receipt['sha256'])
                self.assertEqual(receipt['received_at'], g.epochs[epoch]['start'])
            self.assertTrue(all('/frozen/' in source for source, _ in b.client.copies))
        finally: g.stop()

    def test_copy_fault_keeps_original_snapshots_and_retry_does_not_read_staging(self):
        b, g, epoch = self.prepared()
        try:
            snapshots = dict(g.epochs[epoch]['snapshots']); b.client.copy_failure = True
            with self.assertRaises(IOError): g.freeze(epoch)
            self.assertNotIn('frozen_receipts', g.epochs[epoch])
            self.assertEqual(g.epochs[epoch]['snapshots'], snapshots)
            b.client.copy_failure = False
            receipts = g.freeze(epoch)
            self.assertEqual(set(receipts), set(snapshots))
            self.assertEqual(g.freeze(epoch), receipts)
        finally: g.stop()

    def test_worker_bound_rejected_before_server_or_bucket_work(self):
        for workers in (0, 9, True, 1.5):
            with self.assertRaises(ValueError): Gateway(None, publication_workers=workers)


if __name__ == '__main__': unittest.main()
