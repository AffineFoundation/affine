import base64,hashlib,json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet.storage import canonical
from subnet.distributed_roles import digest
from subnet.distributed_worker import Worker,ExpiredCompletedLease
from subnet.cache_lifecycle import CacheLifecycle

class PostAckDisposal(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);key=SigningKey.generate();self.authority=key.verify_key.encode().hex();self.worker=Worker('http://127.0.0.1:19081',bytes(SigningKey.generate()),self.authority,self.root);self.files={n:hashlib.sha256(b).hexdigest()for n,b in [('config.json',b'{}'),('model.safetensors',b'weights')]};self.job={'job_id':'owned-job','role':'verify','manifest':{'payload':{'checkpoint':{'id':'CP','files':self.files}}}};envelope={'signer':self.authority,'payload':self.job,'signature':base64.b64encode(key.sign(canonical(self.job)).signature).decode()};self.claim={'job':envelope,'job_sha256':digest(self.job),'token':'opaque','lease_until':time.time()+300,'attempt':1};self.cache=self.root/'backend/checkpoints/CP';self.ack=False
 def hydrate(self,args,**kw):
  workspace=Path(args[args.index('--workspace')+1]);cache=workspace/'checkpoints/CP';cache.mkdir(parents=True,exist_ok=True);(cache/'config.json').write_bytes(b'{}');(cache/'model.safetensors').write_bytes(b'weights');lc=CacheLifecycle(workspace);lc.record_checkpoint('CP',self.files);out=workspace/'jobs/owned-job';out.mkdir(parents=True);(out/'report.json').write_bytes(canonical({'job_id':'owned-job','success':True}));return SimpleNamespace(returncode=0)
 def request(self,action,**kw):
  if action=='claim':return {'claim':self.claim}
  if action=='report':
   self.assertTrue(self.cache.exists());self.assertEqual(CacheLifecycle(self.root/'backend').evict_checkpoints(only=['CP'],keep=0),[]);self.ack=True;return {'accepted':True}
  if action=='renew':return {'lease_until':self.claim['lease_until']}
  return {'status':'failed'}
 def run_worker(self,backend=None):
  self.worker.request=self.request
  with patch('subnet.distributed_worker.subprocess.run',side_effect=backend or self.hydrate):return self.worker.once()
 def test_successful_ack_releases_lease_then_deletes_only_owned_checkpoint(self):
  self.assertTrue(self.run_worker());self.assertTrue(self.ack);self.assertFalse(self.cache.exists());attempt=self.root/'owned-job/attempt-1';self.assertTrue((attempt/'job.json').exists());self.assertTrue((attempt/'pending-report.json').exists());self.assertTrue((attempt/'worker.log').exists());self.assertTrue((self.root/'backend/jobs/owned-job/report.json').exists())
 def test_failed_backend_keeps_shared_checkpoint_without_ack(self):
  def fail(args,**kw):self.hydrate(args,**kw);return SimpleNamespace(returncode=1)
  self.assertTrue(self.run_worker(fail));self.assertFalse(self.ack);self.assertTrue(self.cache.exists())
 def test_expired_report_retains_checkpoint_and_diagnostics(self):
  original=self.request
  def request(action,**kw):
   if action=='report':self.claim['lease_until']=time.time()-1;raise ValueError('expired original lease')
   return original(action,**kw)
  self.worker.request=request
  with patch('subnet.distributed_worker.subprocess.run',side_effect=self.hydrate),self.assertRaises(ExpiredCompletedLease):self.worker.once()
  self.assertTrue(self.cache.exists());self.assertTrue((self.root/'owned-job/attempt-1/expired-completed-lease.json').exists())
 def test_another_active_lease_prevents_post_ack_delete(self):
  original=CacheLifecycle.evict_checkpoints
  def evict(lc,*a,**kw):
   if self.ack:
    with lc.lease_checkpoint('CP'):return original(lc,*a,**kw)
   return original(lc,*a,**kw)
  with patch.object(CacheLifecycle,'evict_checkpoints',evict):self.assertTrue(self.run_worker())
  self.assertTrue(self.cache.exists());self.assertEqual(CacheLifecycle(self.root/'backend').evict_checkpoints(only=['CP'],keep=0),['CP'])
 def test_changed_owned_inode_refuses_delete(self):
  original=CacheLifecycle.evict_checkpoints
  def evict(lc,*a,**kw):
   if self.ack:(self.cache/'model.safetensors').write_bytes(b'changed-after-record')
   return original(lc,*a,**kw)
  with patch.object(CacheLifecycle,'evict_checkpoints',evict):self.assertTrue(self.run_worker())
  self.assertTrue(self.cache.exists())
 def test_external_mapped_cache_and_unverified_owned_copy_remain(self):
  external=self.root/'external';external.mkdir();(external/'config.json').write_bytes(b'{}');(external/'model.safetensors').write_bytes(b'weights');self.worker.checkpoint_caches={'CP':str(external)}
  def backend(args,**kw):
   self.assertEqual(args[args.index('--checkpoint-cache')+1],str(external));self.cache.mkdir(parents=True);(self.cache/'config.json').write_bytes(b'{}');(self.cache/'model.safetensors').write_bytes(b'unverified');out=self.root/'backend/jobs/owned-job';out.mkdir(parents=True);(out/'report.json').write_bytes(canonical({'job_id':'owned-job'}));return SimpleNamespace(returncode=0)
  self.assertTrue(self.run_worker(backend));self.assertTrue(external.exists());self.assertTrue(self.cache.exists());self.assertFalse((self.root/'backend/.cache-lifecycle/CP.json').exists())
 def test_cleanup_storage_fault_does_not_turn_ack_into_failed_submission(self):
  original=CacheLifecycle.evict_checkpoints
  def evict(lc,*a,**kw):
   if self.ack:raise OSError('actual storage unavailable')
   return original(lc,*a,**kw)
  with patch.object(CacheLifecycle,'evict_checkpoints',evict),self.assertLogs(level='WARNING'):self.assertTrue(self.run_worker())
  self.assertTrue(self.ack);self.assertTrue(self.cache.exists())
 def test_post_ack_disposal_never_rehashes_model_bytes(self):
  original=Path.open
  def opening(path,*a,**kw):
   if self.ack and path.name=='model.safetensors':raise AssertionError('model read after ACK')
   return original(path,*a,**kw)
  with patch.object(Path,'open',opening):self.assertTrue(self.run_worker())
  self.assertFalse(self.cache.exists())
if __name__=='__main__':unittest.main()
