import base64,copy,io,json,time,unittest,zipfile
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet import long_context_backend_jobs as jobs

class AuthorizationTests(unittest.TestCase):
 def setUp(self):
  self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex();now=time.time()
  self.manifest={'checkpoint':{'files':{'config.json':'1'*64,'model.safetensors':'2'*64}},'model_runtime_revision':jobs.REVISION,'numerical_policy':jobs.NUMERICAL_POLICY,'backend_profile':jobs.BACKEND_PROFILE,'transport_policy':'direct-r2-v1','artifact_policy':jobs.TRANSPORT_POLICY,'audit_policy':{'mode':'full'},'K':1,'L':1,'start':now-5,'deadline':now+60}
  self.manifest['checkpoint']['id']=jobs.file_map(self.manifest['checkpoint']['files'])
  self.job={'schema':1,'job_id':'test','role':'mine','created_at':now-5,'expires_at':now+60,'interpreter_sha256':'c'*64,'source_files':{name:'a'*64 for name in jobs.SOURCE_FILES},'runtime_versions':dict(torch='x',transformers='x',toploc='x',numpy='x'),'manifest':self.sign(self.manifest),'miner_id':'b'*64,'seed_start':0,'search_budget':2,'capability':{'put_url':self.url(),'headers':{'Content-Type':'application/octet-stream'}},'artifact_policy':jobs.TRANSPORT_POLICY,'chain_transactions':False,'payable':False,'resource_policy':{'min_free_vram_bytes':12*1024**3,'wait_seconds':1800}}
 def url(self):return 'https://bucket.r2.cloudflarestorage.com/object?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc'
 def sign(self,payload):return {'payload':copy.deepcopy(payload),'signer':self.authority,'signature':base64.b64encode(self.key.sign(jobs.canonical(payload)).signature).decode()}
 def test_authentication_rejects_before_any_artifact_or_factory(self):
  envelope=self.sign(self.job);envelope['payload']['role']='upload'
  with patch.object(jobs.Path,'mkdir',side_effect=AssertionError('artifact access')),self.assertRaises(Exception):jobs.execute(envelope,self.authority,'/unused',session_factory=lambda _:self.fail('factory accessed'))
 def test_valid_manifest_and_exact_profile(self):
  job,manifest=jobs.validate(self.sign(self.job),self.authority)
  self.assertEqual(manifest['backend_profile']['max_context'],32768)
  self.assertEqual(job['artifact_policy']['compressed_bytes'],250000000)
 def test_numeric_tolerance_relaxation_rejected(self):
  self.manifest['numerical_policy']={**jobs.NUMERICAL_POLICY,'logprob_atol':.01};self.job['manifest']=self.sign(self.manifest)
  with self.assertRaisesRegex(ValueError,'numerical policy'):jobs.validate(self.sign(self.job),self.authority)
 def test_unbounded_or_old_transport_rejected(self):
  self.job['artifact_policy']={'compressed_bytes':999999999,'raw_bytes':500000000}
  with self.assertRaisesRegex(ValueError,'transport policy'):jobs.validate(self.sign(self.job),self.authority)
 def test_chain_or_unsigned_factory_execution_is_not_a_role(self):
  self.job['role']='set_weights'
  with self.assertRaisesRegex(ValueError,'role/schema'):jobs.validate(self.sign(self.job),self.authority)
  self.job['role']='mine';self.job['chain_transactions']=True
  with self.assertRaisesRegex(ValueError,'nonpayable'):jobs.validate(self.sign(self.job),self.authority)
 def test_full_training_requires_exact_signed_optimizer_and_resources(self):
  self.job.update(role='train',steps=1,training_policy=jobs.TRAINING_POLICY,training_parameters=jobs.TRAINING_PARAMETERS,submissions=[{'url':self.url(),'sha256':'a'*64}],resource_policy={'min_free_vram_bytes':20*1024**3,'wait_seconds':1800})
  jobs.validate(self.sign(self.job),self.authority)
  self.job['training_parameters']={**jobs.TRAINING_PARAMETERS,'master_parameters':True}
  with self.assertRaisesRegex(ValueError,'optimizer policy'):jobs.validate(self.sign(self.job),self.authority)
 def test_expired_epoch_or_reduced_vram_rejected(self):
  self.job['resource_policy']['min_free_vram_bytes']=1
  with self.assertRaisesRegex(ValueError,'resources'):jobs.validate(self.sign(self.job),self.authority)
  self.job['resource_policy']['min_free_vram_bytes']=12*1024**3;self.manifest['deadline']=time.time()-1;self.job['manifest']=self.sign(self.manifest)
  with self.assertRaisesRegex(ValueError,'window closed'):jobs.validate(self.sign(self.job),self.authority)

 def test_multistep_training_rejected_before_any_optimizer_or_artifact(self):
  self.job.update(role='train',steps=2,training_policy=jobs.TRAINING_POLICY,training_parameters=jobs.TRAINING_PARAMETERS,submissions=[{'url':self.url(),'sha256':'a'*64}],resource_policy={'min_free_vram_bytes':20*1024**3,'wait_seconds':1800})
  with patch.object(jobs.Path,'mkdir',side_effect=AssertionError('artifact access')),self.assertRaisesRegex(ValueError,'single training step'):
   jobs.execute(self.sign(self.job),self.authority,'/unused',session_factory=lambda _:self.fail('factory accessed'))
 def test_exact_policies_reject_bool_integer_and_float_substitution(self):
  for field,replacement in [('tf32',0),('max_context',32768.0),('torch_threads',2.0)]:
   manifest=copy.deepcopy(self.manifest);manifest['backend_profile'][field]=replacement
   job=copy.deepcopy(self.job);job['manifest']=self.sign(manifest)
   with self.subTest(field=field),self.assertRaisesRegex(ValueError,'profile or numerical policy'):jobs.validate(self.sign(job),self.authority)
  job=copy.deepcopy(self.job);job['artifact_policy']['raw_bytes']=500000000.0
  with self.assertRaisesRegex(ValueError,'transport policy'):jobs.validate(self.sign(job),self.authority)
  job=copy.deepcopy(self.job);job['resource_policy']['wait_seconds']=1800.0
  with self.assertRaisesRegex(ValueError,'resources'):jobs.validate(self.sign(job),self.authority)
 def test_optimizer_boolean_substitution_rejected(self):
  self.job.update(role='train',steps=1,training_policy=jobs.TRAINING_POLICY,training_parameters={**jobs.TRAINING_PARAMETERS,'master_parameters':0},submissions=[{'url':self.url(),'sha256':'a'*64}],resource_policy={'min_free_vram_bytes':20*1024**3,'wait_seconds':1800})
  with self.assertRaisesRegex(ValueError,'optimizer policy'):jobs.validate(self.sign(self.job),self.authority)

class TransportTests(unittest.TestCase):
 def test_roundtrip_retains_full_float32_arrays(self):
  import numpy as np
  array=np.arange(24,dtype=np.float32).reshape(3,8)
  data=jobs.pack([({'index':0},[[array]])]);result=jobs.unpack(data)
  self.assertEqual(result[0][0],{'index':0});np.testing.assert_array_equal(result[0][1][0][0],array)
 def test_archive_duplicate_entries_rejected(self):
  buf=io.BytesIO()
  with zipfile.ZipFile(buf,'w') as z:z.writestr('manifest.json','[]');z.writestr('manifest.json','[]')
  with self.assertRaisesRegex(ValueError,'duplicate'):jobs.unpack(buf.getvalue())
 def test_tensor_header_prevents_unbounded_allocation(self):
  import numpy as np
  buf=io.BytesIO();np.lib.format.write_array_header_1_0(buf,{'descr':'<f4','fortran_order':False,'shape':(513,151936)})
  with self.assertRaisesRegex(ValueError,'shape'):jobs.bounded_tensor(buf.getvalue())
