import hashlib,json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from subnet.cache_lifecycle import CacheLifecycle
from subnet.distributed_worker import Worker
import test_worker_post_ack_checkpoint_disposal as disposal_controls

class BoundedRetention(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.lc=CacheLifecycle(self.root);self.files={'model.safetensors':hashlib.sha256(b'weights').hexdigest()}
 def model(self,cp,origin='authenticated-job-ACK',age=0):
  d=self.root/'checkpoints'/cp;d.mkdir(parents=True);(d/'model.safetensors').write_bytes(b'weights')
  with self.lc.lease_checkpoint(cp):self.lc.record_checkpoint(cp,self.files,origin)
  p=self.lc._receipt(cp);v=json.loads(p.read_text());v['touched']=time.time()-age;p.write_text(json.dumps(v));return d
 def sweep(self,**kwargs):return self.lc.retain_acknowledged_checkpoint(600,0,**kwargs)
 def test_retains_only_latest_full_ack_and_retires_partial_failed(self):
  old=self.model('old',age=2);current=self.model('current');partial=self.model('partial','authenticated-model-map');self.assertCountEqual(self.sweep(),['old','partial']);self.assertTrue(current.exists());self.assertFalse(old.exists());self.assertFalse(partial.exists())
 def test_ttl_expires_without_any_new_job(self):
  d=self.model('current',age=601);self.assertEqual(self.sweep(),['current']);self.assertFalse(d.exists())
 def test_low_disk_floor_retires_even_current(self):
  d=self.model('current')
  with patch('subnet.cache_lifecycle.os.statvfs',return_value=SimpleNamespace(f_bavail=0,f_frsize=1)):
   self.assertEqual(self.lc.retain_acknowledged_checkpoint(600,1),['current'])
  self.assertFalse(d.exists())
 def test_lease_preserves_superseded_until_release(self):
  old=self.model('old',age=2);self.model('current')
  with self.lc.lease_checkpoint('old'):self.assertEqual(self.sweep(),[])
  self.assertTrue(old.exists());self.assertEqual(self.sweep(),['old'])
 def test_changed_inode_and_unowned_never_deleted(self):
  d=self.model('old',age=1000);(d/'model.safetensors').write_bytes(b'changed');external=self.root/'checkpoints/unowned';external.mkdir();(external/'weights').write_bytes(b'legacy');self.assertEqual(self.sweep(),[]);self.assertTrue(external.exists());self.assertTrue(d.exists())
 def test_partial_inventory_cannot_be_hot_current(self):
  d=self.model('partial');p=self.lc._receipt('partial');v=json.loads(p.read_text());v['files']['missing']='a'*64;p.write_text(json.dumps(v));self.assertEqual(self.sweep(),['partial']);self.assertFalse(d.exists())
 def test_policy_bounds(self):
  for ttl,floor in [(0,0),(86401,0),(True,0),(600,-1),(600,True)]:
   with self.subTest(ttl=ttl,floor=floor),self.assertRaises(ValueError):self.lc.retain_acknowledged_checkpoint(ttl,floor)

class WorkerRetention(unittest.TestCase):
 def fixture(self):
  f=disposal_controls.PostAckDisposal();f.setUp();self.addCleanup(f.doCleanups);f.worker.checkpoint_retention={'ttl_seconds':600,'disk_floor_bytes':0};return f
 def test_post_ack_then_idle_retains_and_backend_next_job_reuses_same_owned_files(self):
  f=self.fixture();self.assertTrue(f.run_worker());self.assertTrue(f.cache.exists());f.worker.request=lambda action,**kw:{'claim':None};self.assertFalse(f.worker.once());self.assertTrue(f.cache.exists())
  # A distinct authentic job on the same checkpoint finds retained bytes
  # before the backend runs. Its existing backend full-map checks still apply.
  f.job['job_id']='second';key=None
  from nacl.signing import SigningKey
  from subnet.storage import canonical
  from subnet.distributed_roles import digest
  key=SigningKey.generate();f.worker.authority=key.verify_key.encode().hex()
  import base64
  f.claim['job']={'payload':f.job,'signer':f.worker.authority,'signature':base64.b64encode(key.sign(canonical(f.job)).signature).decode()};f.claim['job_sha256']=digest(f.job);f.worker.request=f.request
  def backend(args,**kwargs):
   self.assertTrue(f.cache.exists());self.assertEqual((f.cache/'model.safetensors').read_bytes(),b'weights');workspace=Path(args[args.index('--workspace')+1]);out=workspace/'jobs/second';out.mkdir(parents=True);(out/'report.json').write_text('{"job_id":"second","success":true}');return SimpleNamespace(returncode=0)
  self.assertTrue(f.run_worker(backend));self.assertTrue(f.cache.exists())
 def test_post_ack_idle_expiry_disposes_automatically(self):
  f=self.fixture();f.run_worker();receipt=CacheLifecycle(f.root/'backend')._receipt('CP');v=json.loads(receipt.read_text());v['touched']=time.time()-601;receipt.write_text(json.dumps(v));f.worker.request=lambda action,**kw:{'claim':None};f.worker.once();self.assertFalse(f.cache.exists())
 def test_invalid_operator_policy_rejected(self):
  with tempfile.TemporaryDirectory() as root:
   with self.assertRaises(ValueError):Worker('http://127.0.0.1:1',b'a'*32,'authority',root,checkpoint_retention={'ttl_seconds':600})
