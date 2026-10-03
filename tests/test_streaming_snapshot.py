"""Atomic GET/conditional-copy freezing, bounded memory and durable partial work."""
import datetime
import tempfile
import time
import threading
import unittest
from pathlib import Path
from botocore.exceptions import ClientError
from test_frozen_publication import Body,Client,bucket
from subnet.storage import Gateway,Identity,SubmissionPolicyError,sha


class SnapshotClient(Client):
    def __init__(self):
        super().__init__();self.completed={};self.fail_source=None
    def get_object(self,**args):
        if args['Key'] not in self.objects:raise ClientError({'Error':{'Code':'NoSuchKey'}},'GetObject')
        response=super().get_object(**args);response['LastModified']=datetime.datetime.fromtimestamp(self.completed.get(args['Key'],int(time.time())),datetime.timezone.utc);return response
    def copy_object(self,**args):
        if args['CopySource']['Key']==self.fail_source:raise IOError('copy unavailable')
        return super().copy_object(**args)


def snapshot_bucket():
    b=bucket();b.client=SnapshotClient();return b


class StreamingSnapshot(unittest.TestCase):
    def test_large_atomic_body_is_streamed_and_copied_without_operator_upload(self):
        b=snapshot_bucket();data=b'x'*(2*1024*1024+3);b.put('staging',data);b.client.completed['staging']=100
        receipt=b.freeze_snapshot('staging','frozen',99,101,limit=len(data))
        self.assertEqual(receipt['sha256'],sha(data));self.assertEqual(receipt['received_at'],100);self.assertEqual(receipt['size'],len(data))
        self.assertEqual(b.client.objects[receipt['snapshot_key']],data);self.assertEqual(len(b.client.copies),1)
        self.assertTrue(b.client.bodies[0].closed);self.assertGreater(len(b.client.bodies[0].read_sizes),2)
        self.assertEqual(set(b.client.objects),{'staging',receipt['snapshot_key']})

    def test_missing_and_out_of_window_or_oversized_uploads_do_not_copy(self):
        b=snapshot_bucket();self.assertIsNone(b.freeze_snapshot('missing','frozen',99,101))
        b.put('staging',b'abc')
        for timestamp in [98,101,102]:
            b.client.completed['staging']=timestamp
            with self.assertRaises(SubmissionPolicyError):b.freeze_snapshot('staging','frozen',99,101)
            self.assertEqual(b.client.bodies[-1].read_sizes,[]);self.assertTrue(b.client.bodies[-1].closed)
        b.client.completed['staging']=100
        with self.assertRaises(SubmissionPolicyError):b.freeze_snapshot('staging','frozen',99,101,limit=2)
        self.assertEqual(b.client.copies,[])

    def test_truncated_stream_and_read_copy_race_never_freeze_substitution(self):
        b=snapshot_bucket();b.put('staging',b'original');b.client.completed['staging']=100;original=b.client.get_object
        def truncated(**args):
            response=original(**args);response['ContentLength']+=1;return response
        b.client.get_object=truncated
        with self.assertRaisesRegex(ValueError,'size mismatch'):b.freeze_snapshot('staging','frozen',99,101)
        self.assertTrue(b.client.bodies[-1].closed);self.assertEqual(b.client.copies,[])
        b.client.get_object=original;b.client.mutate_before_copy=True
        with self.assertRaises(ClientError):b.freeze_snapshot('staging','frozen',99,101)
        self.assertEqual(set(b.client.objects),{'staging'})

    def test_parallel_freeze_has_bounded_reads_and_persists_siblings_on_copy_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            b=snapshot_bucket();g=Gateway(b,direct_r2=True,publication_workers=2,state_path=Path(directory)/'gateway.json');epoch='nonpayable-stream-freeze';ids=[Identity().id for _ in range(4)]
            try:
                g.open(epoch,ids,int(time.time())+100)
                for miner in ids:b.put('private/'+epoch+'/staging/'+miner+'.zip',miner.encode())
                b.client.get_barrier=threading.Barrier(2);bad='private/'+epoch+'/staging/'+ids[0]+'.zip';b.client.fail_source=bad
                with self.assertRaisesRegex(IOError,'copy unavailable'):g.freeze(epoch)
                self.assertEqual(b.client.peak,2);self.assertEqual(set(g.epochs[epoch]['snapshots']),set(ids[1:]));self.assertEqual(g.epochs[epoch]['rejections'],{})
                self.assertNotIn('frozen_receipts',g.epochs[epoch])
                g.stop();b.client.get_barrier=None;b.client.fail_source=None
                for miner in ids[1:]:b.put('private/'+epoch+'/staging/'+miner+'.zip',b'late replacement')
                g=Gateway(b,direct_r2=True,publication_workers=2,state_path=Path(directory)/'gateway.json');receipts=g.freeze(epoch)
                self.assertEqual(set(receipts),set(ids))
                for miner in ids:self.assertEqual(b.client.objects[receipts[miner]['frozen_key']],miner.encode())
                copies=len(b.client.copies);self.assertEqual(g.freeze(epoch),receipts);self.assertEqual(len(b.client.copies),copies)
            finally:g.stop()


if __name__=='__main__':unittest.main()
