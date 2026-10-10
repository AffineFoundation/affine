"""Actual Bucket shape, metadata above token-GET cap, no network or signing."""
import io,json,pathlib,sys,types,unittest
HERE=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from subnet.storage import Bucket
from ops.native_task_representative_lifecycle import _archive,_read_metadata
from botocore.exceptions import ClientError
class Body(io.BytesIO):
    def read(self,n=-1):self.last_read=n;return super().read(n)
class Client:
    def __init__(self):self.values={};self.bodies=[];self.calls=[];self.puts=0;self.override=None
    def get_object(self,**kw):
        self.calls.append(kw);key=kw['Key']
        if key not in self.values:raise ClientError({'Error':{'Code':'NoSuchKey'}},'GetObject')
        data=self.values[key];length=len(data)
        if self.override:data,length=self.override(data,length)
        body=Body(data);self.bodies.append(body)
        return {'Body':body,'ContentLength':length}
    def put_object(self,**kw):self.puts+=1;self.values[kw['Key']]=kw['Body']
class MetadataReads(unittest.TestCase):
    def setUp(self):
        self.bucket=object.__new__(Bucket);self.bucket.name='test-only';self.bucket.client=Client()
        self.controller=types.SimpleNamespace(bucket=self.bucket);self.data=b'x'*2_000_001;self.key='private/epoch/native-task-representatives/pool.json'
    def test_real_small_document_api_still_rejects_large_metadata(self):
        with self.assertRaisesRegex(ValueError,'bounded document GET size'):self.bucket.get_bounded(self.key,limit=len(self.data))
        self.assertEqual(self.bucket.client.calls,[])
    def test_large_metadata_actual_bucket_put_full_get_and_idempotent_replay(self):
        result=_archive(self.controller,self.key,self.data)
        self.assertTrue(result['full_get_verified']);self.assertEqual(result['size'],len(self.data));self.assertEqual(self.bucket.client.puts,1)
        self.assertEqual(result,_archive(self.controller,self.key,self.data));self.assertEqual(self.bucket.client.puts,1)
        self.assertTrue(all(b.closed and b.last_read==len(self.data)+1 for b in self.bucket.client.bodies))
    def test_metadata_collision_never_overwrites(self):
        self.bucket.client.values[self.key]=b'y'*len(self.data)
        with self.assertRaisesRegex(ValueError,'immutable'): _archive(self.controller,self.key,self.data)
        self.assertEqual(self.bucket.client.puts,0);self.assertTrue(self.bucket.client.bodies[-1].closed)
    def test_content_length_str_bool_wrong_refuse_and_close_before_read(self):
        self.bucket.client.values[self.key]=self.data
        for value in (str(len(self.data)),True,len(self.data)-1,len(self.data)+1):
            self.bucket.client.override=lambda data,length,value=value:(data,value)
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'ContentLength'):_read_metadata(self.bucket,self.key,len(self.data))
            body=self.bucket.client.bodies[-1];self.assertTrue(body.closed);self.assertFalse(hasattr(body,'last_read'))
    def test_truncation_and_extra_body_refuse_after_bounded_read(self):
        self.bucket.client.values[self.key]=self.data
        for change in (-1,1):
            self.bucket.client.override=lambda data,length,change=change:(data[:-1]if change<0 else data+b'x',length)
            with self.assertRaisesRegex(ValueError,'complete byte count'):_read_metadata(self.bucket,self.key,len(self.data))
            self.assertTrue(self.bucket.client.bodies[-1].closed)
    def test_corruption_after_put_never_acknowledged(self):
        self.bucket.client.override=lambda data,length:(b'y'*length,length)
        with self.assertRaisesRegex(ValueError,'full metadata readback'):_archive(self.controller,self.key,self.data)
        self.assertEqual(self.bucket.client.puts,1)
    def test_size_bound_is_separate_and_checked_before_transport(self):
        for size in (0,-1,True,64*1024**2+1):
            with self.assertRaisesRegex(ValueError,'metadata bound'):_read_metadata(self.bucket,self.key,size)
        self.assertEqual(self.bucket.client.calls,[])
    def test_transport_failure_is_not_missing_and_never_triggers_put(self):
        def fail(**kw):raise TimeoutError('GET unavailable')
        self.bucket.client.get_object=fail
        with self.assertRaises(TimeoutError):_archive(self.controller,self.key,self.data)
        self.assertEqual(self.bucket.client.puts,0)
    def test_malformed_response_is_not_missing_and_never_triggers_put(self):
        for response in ({},{'ContentLength':len(self.data)}):
            self.bucket.client.get_object=lambda **kw:response
            with self.subTest(response=response),self.assertRaises(KeyError):_archive(self.controller,self.key,self.data)
            self.assertEqual(self.bucket.client.puts,0)
if __name__=='__main__':unittest.main()
