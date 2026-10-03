import json,tempfile,time,unittest
from pathlib import Path
from subnet.storage import Gateway,Identity,canonical,sha,SubmissionPolicyError

class MemoryR2:
    def __init__(self):self.objects={};self.completed={};self.after_read=None
    def put(self,key,data,*args):self.objects[key]=data;self.completed[key]=time.time()
    def json(self,key,value):self.put(key,canonical(value))
    def get(self,key):return self.objects[key]
    def presign(self,key,operation='get_object',expires=604800):return 'https://r2.example.invalid/'+key+'?operation='+operation
    def snapshot(self,key,limit=100000000):
        if key not in self.objects:return None
        data=self.objects[key]
        if not 0<len(data)<=limit:raise SubmissionPolicyError('R2 upload size')
        value=dict(data=data,size=len(data),etag=sha(data),completed_at=self.completed[key])
        if self.after_read:self.after_read(key)
        return value

class DirectR2Freeze(unittest.TestCase):
    def test_server_copy_publishes_verified_frozen_bytes_and_retries_copy_fault(self):
        from unittest.mock import patch
        class CopyingR2(MemoryR2):
            def __init__(self):super().__init__();self.copies=[]
            def copy(self,source,destination):
                self.copies.append((source,destination));self.objects[destination]=self.objects[source]
        b=CopyingR2();i=Identity();g=Gateway(b,direct_r2=True)
        try:
            epoch='nonpayable-copy';g.open(epoch,[i.id],int(time.time())+100)
            staging=f'private/{epoch}/staging/{i.id}.zip';b.put(staging,b'original')
            with patch.object(b,'copy',side_effect=IOError('copy unavailable')):
                with self.assertRaises(IOError):g.freeze(epoch)
            self.assertNotIn('frozen_receipts',g.epochs[epoch])
            frozen=g.epochs[epoch]['snapshots'][i.id]['snapshot_key'];b.put(staging,b'late replacement')
            with patch.object(b,'snapshot',side_effect=AssertionError('reuse frozen bytes')):
                receipts=g.freeze(epoch)
            self.assertEqual(b.copies,[(frozen,receipts[i.id]['frozen_key'])])
            self.assertEqual(b.get(receipts[i.id]['frozen_key']),b'original')
            self.assertEqual(g.freeze(epoch),receipts);self.assertEqual(len(b.copies),1)
        finally:g.stop()

    def test_corrupt_frozen_object_refuses_before_server_copy(self):
        from unittest.mock import Mock
        b=MemoryR2();b.copy=Mock();i=Identity();g=Gateway(b,direct_r2=True)
        try:
            epoch='nonpayable-copy-corrupt';g.open(epoch,[i.id],int(time.time())+100)
            frozen=f'private/{epoch}/frozen/{i.id}/original.zip';b.put(frozen,b'changed')
            g.epochs[epoch]['snapshots']={i.id:dict(key='unused',snapshot_key=frozen,sha256=sha(b'original'))}
            with self.assertRaisesRegex(ValueError,'receipt hash changed'):g.freeze(epoch)
            b.copy.assert_not_called();self.assertNotIn('frozen_receipts',g.epochs[epoch])
        finally:g.stop()

    def test_direct_checkpoint_urls_fail_closed_before_network(self):
        from subnet.client import checkpoint_download,direct_r2_url
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d,patch('subnet.client.requests.get',side_effect=AssertionError('network should not run')):
            manifest=dict(transport_policy='direct-r2-v1',checkpoint=dict(files={'config.json':'a'*64},base_url='https://tunnel.invalid'))
            with self.assertRaises(ValueError):checkpoint_download(manifest,d)
            manifest['checkpoint']['read_urls']={'config.json':'https://tunnel.invalid/config.json'}
            with self.assertRaises(ValueError):checkpoint_download(manifest,d)
        for url in ('http://account.r2.cloudflarestorage.com/file?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=a','https://tunnel.invalid/file'):
            with self.assertRaises(ValueError):direct_r2_url(url)

    def test_private_capability_and_late_inflight_overwrite_cannot_change_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            b=MemoryR2();i=Identity();g=Gateway(b,state_path=Path(d)/'gateway.json',direct_r2=True)
            try:
                deadline=int(time.time())+100;caps=g.open('nonpayable-direct',[i.id],deadline);cap=i.decrypt(caps[i.id])
                self.assertEqual(cap['transport'],'direct-r2-v1');self.assertIn('/private/nonpayable-direct/staging/'+i.id,cap['put_url'])
                self.assertEqual(cap['headers'],{'Content-Type':'application/octet-stream'})
                key=f'private/nonpayable-direct/staging/{i.id}.zip';b.put(key,b'first');b.put(key,b'latest complete')
                def late(k):b.put(k,b'late inflight replacement');b.completed[k]=deadline+1
                b.after_read=late
                receipts=g.freeze('nonpayable-direct');r=receipts[i.id]
                self.assertEqual(r['sha256'],sha(b'latest complete'))
                self.assertEqual(b.get(r['frozen_key']),b'latest complete')
                b.after_read=None;b.put(key,b'another late replacement')
                self.assertEqual(g.freeze('nonpayable-direct'),receipts)
                self.assertEqual(b.get(r['snapshot_key']),b'latest complete')
                self.assertTrue(json.loads((Path(d)/'gateway.json').read_text())['epochs']['nonpayable-direct']['closed'])
            finally:g.stop()

    def test_completion_deadline_and_epoch_start_are_enforced(self):
        b=MemoryR2();ids=[Identity() for _ in range(3)];g=Gateway(b,direct_r2=True)
        try:
            deadline=int(time.time())+100;g.open('nonpayable-late',[i.id for i in ids],deadline)
            for i,t in zip(ids,[deadline,deadline+1,g.epochs['nonpayable-late']['start']-1]):
                key=f'private/nonpayable-late/staging/{i.id}.zip';b.put(key,b'late');b.completed[key]=t
            self.assertEqual(g.freeze('nonpayable-late'),{})
            self.assertEqual(len(g.epochs['nonpayable-late']['rejections']),3)
        finally:g.stop()

    def test_bad_size_peer_is_rejected_without_blocking_honest_peer(self):
        b=MemoryR2();bad,good=Identity(),Identity();g=Gateway(b,direct_r2=True)
        try:
            g.open('nonpayable-size',[bad.id,good.id],int(time.time())+100)
            b.put(f'private/nonpayable-size/staging/{bad.id}.zip',b'')
            b.put(f'private/nonpayable-size/staging/{good.id}.zip',b'honest')
            result=g.freeze('nonpayable-size')
            self.assertEqual(set(result),{good.id});self.assertEqual(b.get(result[good.id]['frozen_key']),b'honest')
            self.assertEqual(g.epochs['nonpayable-size']['rejections'][bad.id],'R2 upload size')
        finally:g.stop()

    def test_stream_integrity_fault_remains_retryable_not_miner_rejection(self):
        from unittest.mock import patch
        b=MemoryR2();i=Identity();g=Gateway(b,direct_r2=True)
        try:
            g.open('nonpayable-fault',[i.id],int(time.time())+100)
            with patch.object(b,'snapshot',side_effect=ValueError('stream truncated')):
                with self.assertRaisesRegex(ValueError,'stream truncated'):g.freeze('nonpayable-fault')
            self.assertEqual(g.epochs['nonpayable-fault']['rejections'],{})
        finally:g.stop()

    def test_receipt_publication_failure_recovers_without_resnapshot(self):
        from unittest.mock import patch
        b=MemoryR2();i=Identity();g=Gateway(b,direct_r2=True)
        try:
            g.open('nonpayable-recover',[i.id],int(time.time())+100)
            key=f'private/nonpayable-recover/staging/{i.id}.zip';b.put(key,b'accepted')
            original=b.json
            def unavailable(name,value):
                if name.endswith('/receipts.json'):raise IOError('transient publish fault')
                original(name,value)
            with patch.object(b,'json',side_effect=unavailable):
                with self.assertRaises(IOError):g.freeze('nonpayable-recover')
            b.put(key,b'late overwrite')
            with patch.object(b,'snapshot',side_effect=AssertionError('must reuse persisted frozen receipt')):
                receipts=g.freeze('nonpayable-recover')
            self.assertEqual(b.get(receipts[i.id]['frozen_key']),b'accepted')
            self.assertEqual(json.loads(b.get('public/nonpayable-recover/receipts.json')),receipts)
        finally:g.stop()

if __name__=='__main__':unittest.main()
