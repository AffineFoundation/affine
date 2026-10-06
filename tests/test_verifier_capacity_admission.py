import hashlib,json,os,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet.backend_jobs import signed
from subnet.storage import canonical
from subnet.training_receipts import sha as descriptor_sha
from subnet.distributed_worker import Worker,VerifierCapacityDeferred
from subnet.cache_lifecycle import CacheLifecycle
from ops.verifier_capacity_admission import VERSION,admit,CapacityDeferred
from test_failed_training_evidence_retention import FailedEvidenceRetention

class VerifierCapacityAdmission(unittest.TestCase):
 def setUp(self):
  self.fx=FailedEvidenceRetention();self.fx.setUp();self.addCleanup(self.fx.doCleanups);self.sign=self.fx.sign;self.authority=self.fx.authority
  self.root=self.fx.root/'verifier-cache';self.cache=CacheLifecycle(self.root);self.cp='a'*64;self.bytes=b'model-size';self.sha=hashlib.sha256(self.bytes).hexdigest()
  m=json.loads(json.dumps(self.fx.job['manifest']['payload']));m['checkpoint']=dict(id=self.cp,files={'model.safetensors':self.sha});m.pop('artifact_policy',None)
  self.job=dict(role='verify',job_id='original-verifier',manifest=self.sign(m),submissions=[{'sha256':'b'*64}],expires_at=100)
  self.value=dict(version=VERSION,checkpoint_inventories={self.cp:dict(descriptor_sha256=descriptor_sha(dict(id=self.cp,files={'model.safetensors':self.sha})),files={'model.safetensors':dict(size=len(self.bytes),sha256=self.sha)})},disk_floor_bytes=2*1024**3,extra_temporary_bytes=100,max_submissions=4,poll_seconds=2)
  self.policy=self.sign(self.value)
 def call(self,free=10**11,**kw):return admit(self.job,self.policy,self.authority,lifecycle=self.cache,free_bytes=lambda:free,**kw)
 def model(self,cp=None):
  cp=cp or self.cp;p=self.root/'checkpoints'/cp/'model.safetensors';p.parent.mkdir(parents=True);p.write_bytes(self.bytes)
  with self.cache.lease_checkpoint(cp):self.cache.record_checkpoint(cp,{'model.safetensors':self.sha})
  return p
 def test_default_off(self):self.assertEqual(admit(None,None,None,lifecycle=None),{'status':'disabled'})
 def test_cold_admits_exact_sizes_enforceable_input_cap_plus_temp_and_floor(self):
  r=self.call();self.assertEqual(r['model_missing_bytes'],len(self.bytes));self.assertEqual(r['input_cap_bytes'],100_000_000);self.assertEqual(r['temporary_bytes'],100_000_100);self.assertEqual(r['required_free_bytes'],len(self.bytes)+200_000_100+2*1024**3)
 def test_lowspace_defers_no_model_or_backend(self):
  r=self.call(free=0);self.assertEqual(r['status'],'deferred');self.assertFalse((self.root/'checkpoints').exists())
 def test_warm_receipt_uses_inode_metadata_without_model_read(self):
  self.model()
  original=Path.open
  def guarded(path,*args,**kwargs):
   if path.name=='model.safetensors':raise AssertionError('no model read')
   return original(path,*args,**kwargs)
  with patch.object(Path,'open',guarded):r=self.call()
  self.assertEqual(r['authenticated_reusable_bytes'],len(self.bytes));self.assertEqual(r['model_missing_bytes'],0)
 def test_changed_member_partial_and_unowned_sameID_no_credit(self):
  p=self.model();p.write_bytes(b'changed');self.assertEqual(self.call()['authenticated_reusable_bytes'],0)
  p.unlink();(p.parent/'model.safetensors.partial').write_bytes(self.bytes);self.assertEqual(self.call()['authenticated_reusable_bytes'],0)
 def test_obsolete_owned_active_lease_protected_then_eviction_can_admit(self):
  old='d'*64;path=self.model(old);required=self.call()['required_free_bytes']
  def free():return required-1 if path.exists()else required
  with patch('subnet.cache_lifecycle.os.statvfs',return_value=SimpleNamespace(f_bavail=0,f_frsize=1)),self.cache.lease_checkpoint(old):
   result=admit(self.job,self.policy,self.authority,lifecycle=self.cache,free_bytes=free);self.assertEqual(result['status'],'deferred');self.assertTrue(path.exists())
  with patch('subnet.cache_lifecycle.os.statvfs',return_value=SimpleNamespace(f_bavail=0,f_frsize=1)):result=admit(self.job,self.policy,self.authority,lifecycle=self.cache,free_bytes=free)
  self.assertEqual(result['status'],'admitted');self.assertEqual(result['retired_owned_checkpoints'],[old])
 def test_incoming_checkpoint_never_evicted_even_below_floor(self):
  p=self.model();self.assertEqual(self.call(free=0)['status'],'deferred');self.assertTrue(p.exists())
 def test_unsigned_size_or_wrongSHA_and_optimistic_submission_ref_refused(self):
  v=json.loads(json.dumps(self.value));v['checkpoint_inventories'][self.cp]['files']['model.safetensors']['sha256']='f'*64
  with self.assertRaises(ValueError):admit(self.job,self.sign(v),self.authority,lifecycle=self.cache)
  self.job['submissions'][0]['commitment_ref']={'size':1};self.assertEqual(self.call()['input_cap_bytes'],100_000_000)
 def test_missing_size_grant_is_deferrable(self):
  v=json.loads(json.dumps(self.value));v['checkpoint_inventories']={'e'*64:v['checkpoint_inventories'][self.cp]}
  with self.assertRaises(CapacityDeferred):admit(self.job,self.sign(v),self.authority,lifecycle=self.cache)
 def test_bounded_shared_owned_retry_cache_metadata_credit(self):
  self.model();other=CacheLifecycle(self.fx.root/'retry');r=admit(self.job,self.policy,self.authority,lifecycle=other,selected_cache=self.root/'checkpoints'/self.cp,credit_lifecycle=self.cache,free_bytes=lambda:10**11);self.assertEqual(r['model_missing_bytes'],0)
 def test_worker_waits_same_original_lease_no_failreport_or_subprocess(self):
  path=self.fx.root/'policy.json';path.write_bytes(canonical(self.policy));worker=SimpleNamespace(capacity_policy_path=path,authority=self.authority);attempt=self.fx.root/'attempt';attempt.mkdir();claim=dict(lease_until=100,job_sha256='f'*64,attempt=1);seen=[];results=iter([dict(status='deferred'),dict(status='admitted')])
  with patch('ops.verifier_capacity_admission.admit',side_effect=lambda *a,**k:next(results)),patch('subprocess.run',side_effect=AssertionError('no child')):
   Worker.wait_for_capacity(worker,self.job,claim,threading.Event(),self.cache,None,attempt,pause=seen.append,clock=lambda:10)
  self.assertEqual(seen,[2]);self.assertEqual(len(list(attempt.glob('capacity-*.json'))),2)
 def test_original_lease_expiry_never_executes_backend_or_scientific_fail(self):
  worker=SimpleNamespace(capacity_policy_path=self.fx.root/'not-read',authority=self.authority)
  with self.assertRaises(VerifierCapacityDeferred):Worker.wait_for_capacity(worker,self.job,dict(lease_until=5),threading.Event(),self.cache,None,self.fx.root,pause=lambda _:None,clock=lambda:10)
 def test_transport_only_tightens_exact_model_and_input_mapping(self):
  from ops.capacity_bounded_verifier_backend import bind_transport
  import ops.verifier_capacity_admission as module
  calls=[];backend=SimpleNamespace(get_object=lambda *a,**k:calls.append(a))
  job=json.loads(json.dumps(self.job));job['manifest']['payload']['checkpoint']['read_urls']={'model.safetensors':'model-url'};job['manifest']=self.sign(job['manifest']['payload']);job['submissions'][0]['url']='input-url'
  bind_transport(backend,job,self.authority,self.root,self.policy,module)
  backend.get_object('model-url',self.sha,self.root/'checkpoints'/self.cp/'model.safetensors',20_000_000_000)
  self.assertEqual(calls[-1][-1],len(self.bytes))
  backend.get_object('input-url','b'*64,self.root/'jobs'/job['job_id']/'submission-0.zip',2_000_000_000);self.assertEqual(calls[-1][-1],100_000_000)
  with self.assertRaises(ValueError):backend.get_object('model-url',self.sha,self.root/'unowned',20_000_000_000)
  backend.get_object('other-url','e'*64,self.root/'other',42);self.assertEqual(calls[-1][-1],42)
 def test_validated_child_size_caps_download_below_LONG_budget(self):
  from ops.verifier_capacity_admission import input_limits
  # The isolated signed admission fixture exercises validator-before-credit;
  # forged children must propagate rejection rather than claim cheap reserve.
  m=self.job['manifest']['payload'];m['submission_transport_policy']='example'
  with patch('subnet.distributed_roles.validate_frozen_submissions',side_effect=ValueError('unauthenticated child')):
   with self.assertRaises(ValueError):input_limits(self.job,m)
  self.job['submissions'][0]['commitment_ref']={'size':2902}
  with patch('subnet.distributed_roles.validate_frozen_submissions')as check:self.assertEqual(input_limits(self.job,m),[2902]);check.assert_called_once_with(m,self.job['submissions'])
