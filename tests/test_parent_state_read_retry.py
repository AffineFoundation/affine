import hashlib,json,tempfile,time,unittest,os
from pathlib import Path
from unittest.mock import patch,Mock
import requests
from subnet import backend_jobs as backend

class Response:
 def __init__(self,parts,status=200,error=None):self.parts=parts;self.status_code=status;self.error=error;self.closed=False
 def __enter__(self):return self
 def __exit__(self,*args):self.closed=True
 def close(self):self.closed=True
 def iter_content(self,n):
  yield from self.parts
  if self.error:raise self.error

class ParentStateRetryControls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.out=Path(self.tmp.name)/'jobs/original';self.transfer=self.out/'.fp32-state-transfer-unique';self.transfer.mkdir(parents=True);self.path=self.transfer/'state-000000.safetensors';self.raw=b'full approved FP32 shard bytes';self.row={'name':self.path.name,'size':len(self.raw),'sha256':hashlib.sha256(self.raw).hexdigest()};self.binding={'row':self.row,'expires_at':time.time()+60,'out':self.out}
 def read(self,**kw):
  with patch.object(backend,'r2_url',side_effect=lambda url,op:url):return backend._retry_parent_object('https://storage.invalid/original',self.binding,self.path,sleep=lambda n:None,**kw)
 def journal(self):return json.loads((self.out/'parent-state-read-retries'/(self.path.name+'.json')).read_bytes())
 def test_streamed_connection_timeout_retries_byte_zero_and_full_SHA(self):
  replies=[Response([b'prefix'],error=requests.ConnectionError('ReadTimeoutError')),Response([self.raw])]
  with patch('requests.get',side_effect=replies)as get:self.read()
  self.assertEqual(get.call_count,2);self.assertEqual(self.path.read_bytes(),self.raw);self.assertFalse(self.path.with_suffix('.safetensors.partial').exists());r=self.journal();self.assertTrue(r['verified']);self.assertEqual([x['status']for x in r['attempts']],['transient_transport_failure','verified_complete']);self.assertFalse(r['reused_existing_bytes'])
 def test_429_503_retry_but_403_is_not_retried(self):
  with patch('requests.get',side_effect=[Response([],429),Response([],503),Response([self.raw])])as get:self.read()
  self.assertEqual(get.call_count,3);self.assertEqual(self.path.read_bytes(),self.raw)
  self.path.unlink()
  with patch('requests.get',return_value=Response([],403))as get,self.assertRaisesRegex(ValueError,'R2 GET status 403'):self.read()
  self.assertEqual(get.call_count,1)
 def test_digest_mismatch_and_oversize_never_retried(self):
  for data in [b'corrupt',self.raw+b'excess']:
   with self.subTest(data=data),patch('requests.get',return_value=Response([data]))as get,self.assertRaises(backend.ArtifactRejected):self.read()
   self.assertEqual(get.call_count,1);self.assertFalse(self.path.exists());self.assertFalse(self.path.with_suffix('.safetensors.partial').exists())
 def test_three_transient_failures_neutral_exhaustion_no_admitted_bytes(self):
  with patch('requests.get',side_effect=requests.ConnectionError('timeout'))as get,self.assertRaises(backend.StateReadInfrastructureDeferred):self.read()
  self.assertEqual(get.call_count,3);self.assertEqual(self.journal()['status'],'infrastructure_deferred_retry_exhausted');self.assertFalse(self.path.exists());self.assertFalse(self.journal()['verified'])
 def test_original_expiry_no_clock_extension_or_network_attempt(self):
  self.binding['expires_at']=time.time()-1
  with patch('requests.get')as get,self.assertRaises(backend.StateReadInfrastructureDeferred):self.read()
  get.assert_not_called();self.assertEqual(self.journal()['status'],'infrastructure_deferred_original_expiry')
 def test_full_verified_read_crossing_expiry_is_not_admitted_for_update(self):
  self.binding['expires_at']=15
  with patch.object(backend.time,'time',side_effect=[10,10,16,16]),patch('requests.get',return_value=Response([self.raw])),self.assertRaises(backend.StateReadInfrastructureDeferred):self.read()
  self.assertEqual(self.path.read_bytes(),self.raw);self.assertTrue(self.journal()['verified']);self.assertEqual(self.journal()['status'],'infrastructure_deferred_original_expiry')
 def test_actual_complete_existing_bytes_rehashed_not_receipt_promoted(self):
  self.path.write_bytes(self.raw)
  with patch('requests.get')as get:self.read()
  get.assert_not_called();self.assertTrue(self.journal()['reused_existing_bytes']);self.assertEqual(self.path.read_bytes(),self.raw)
  self.path.write_bytes(b'corrupt')
  with patch('requests.get')as get,self.assertRaises(backend.ArtifactRejected):self.read()
  get.assert_not_called();self.assertEqual(self.path.read_bytes(),b'corrupt')
 def test_hardlink_existing_file_refused(self):
  self.path.write_bytes(self.raw);os.link(self.path,self.out/'foreign')
  with patch('requests.get')as get,self.assertRaisesRegex(ValueError,'single-link'):self.read()
  get.assert_not_called();self.assertEqual((self.out/'foreign').read_bytes(),self.raw)
 def test_scope_routes_only_original_parent_URL_SHA_size_owned_job_path(self):
  job={'job_id':'original','role':'train','training_policy':'bf16-cpu-fp32-master-task-normalized-persistent-v4','expires_at':self.binding['expires_at'],'persistent_training':{'parent_read_urls':{self.path.name:'original-cap'}}};parent={'shards':[self.row]};context=(job,{},'AUTH',str(self.out.parent.parent))
  with patch.object(backend,'_PARENT_READ_CONTEXT',context),patch('subnet.persistent_training_protocol.validate_job',return_value=({},parent)):
   self.assertEqual(backend._parent_read_binding('original-cap',self.row['sha256'],self.path,self.row['size'])['row'],self.row)
   self.assertIsNone(backend._parent_read_binding('unrelated-cap',self.row['sha256'],self.path,self.row['size']))
   for sha,size,path in [('bad',self.row['size'],self.path),(self.row['sha256'],self.row['size']+1,self.path),(self.row['sha256'],self.row['size'],self.out/'arbitrary')]:
    with self.subTest(sha=sha,size=size,path=path),self.assertRaises(ValueError):backend._parent_read_binding('original-cap',sha,path,size)
 def test_unrelated_artifact_download_retains_historical_one_attempt(self):
  with patch.object(backend,'_PARENT_READ_CONTEXT',None),patch.object(backend,'_get_object_once',side_effect=requests.ConnectionError('transport'))as get,self.assertRaises(requests.ConnectionError):backend.get_object('artifact',self.row['sha256'],self.path,self.row['size'])
  self.assertEqual(get.call_count,1)
if __name__=='__main__':unittest.main()
