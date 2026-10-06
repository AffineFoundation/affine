import base64,hashlib,json,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from nacl.signing import SigningKey
from subnet.distributed_worker import Worker
from subnet.distributed_roles import digest
from subnet.cache_lifecycle import CacheLifecycle
from subnet.storage import canonical

class ExpiredInputDisposal(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.key=SigningKey.generate();self.worker=Worker('http://127.0.0.1:19081',bytes(SigningKey.generate()),self.key.verify_key.encode().hex(),self.root);self.worker.request=Mock(return_value={'claim':None});self.attempt=self.root/'original-job/attempt-1';self.attempt.mkdir(parents=True);self.job={'job_id':'original-job','role':'verify','manifest':{'payload':{'checkpoint':{'id':'CP'}}},'submissions':[{'sha256':hashlib.sha256(b'durable proof bytes').hexdigest()}]};self.envelope();self.lifecycle=CacheLifecycle(self.root/'backend');self.input=self.root/'backend/jobs/original-job/submission-0.zip';self.input.parent.mkdir(parents=True);self.input.write_bytes(b'durable proof bytes');self.lifecycle.record_download(self.input,self.job['submissions'][0]['sha256']);self.diag={'job_id':'original-job','job_sha256':digest(self.job),'attempt':1,'stage':'report_acknowledgment','backend_terminal':True,'backend_exit':0,'lease_until':time.time()-10,'observed_at':time.time()-1,'report_acknowledged':False};self.save_diag()
  for path in (self.attempt/'pending-report.json',self.attempt/'worker.log',self.input.parent/'report.json'):path.write_bytes(b'retain original diagnostics');path.chmod(0o600)
 def envelope(self):
  e={'payload':self.job,'signer':self.key.verify_key.encode().hex(),'signature':base64.b64encode(self.key.sign(canonical(self.job)).signature).decode()};p=self.attempt/'job.json';p.write_bytes(canonical(e));p.chmod(0o600)
 def save_diag(self):
  p=self.attempt/'expired-completed-lease.json';p.write_bytes(canonical(self.diag));p.chmod(0o600)
 def test_idle_retirement_preserves_all_terminal_evidence_and_never_hashes_proof(self):
  original=Path.open
  def open_file(path,*a,**k):
   if path==self.input:raise AssertionError('no whole proof reread')
   return original(path,*a,**k)
  with patch.object(Path,'open',open_file):self.assertFalse(self.worker.once())
  self.assertFalse(self.input.exists());self.assertFalse((self.lifecycle.meta/'download-original-job.json').exists())
  for p in (self.attempt/'job.json',self.attempt/'expired-completed-lease.json',self.attempt/'pending-report.json',self.attempt/'worker.log',self.input.parent/'report.json'):self.assertTrue(p.exists())
 def test_other_active_inherited_checkpoint_lease_defers_then_retires(self):
  with self.lifecycle.lease_checkpoint('CP'):self.assertFalse(self.worker.once());self.assertTrue(self.input.exists())
  self.assertFalse(self.worker.once());self.assertFalse(self.input.exists())
 def test_mismatched_original_job_or_digest_refuses(self):
  for field,value in [('job_id','foreign'),('job_sha256','a'*64),('attempt',2),('backend_terminal',1),('backend_exit',True),('report_acknowledged',True),('stage','still_running')]:
   old=self.diag[field];self.diag[field]=value;self.save_diag()
   with self.assertLogs(level='WARNING'):self.assertFalse(self.worker.once())
   self.assertTrue(self.input.exists());self.diag[field]=old
 def test_wrong_signed_submission_sha_leaves_owned_copy(self):
  self.job['submissions'][0]['sha256']='b'*64;self.envelope();self.diag['job_sha256']=digest(self.job);self.save_diag();self.assertFalse(self.worker.once());self.assertTrue(self.input.exists())
 def test_untracked_user_file_and_changed_inode_preserved(self):
  user=self.input.parent/'submission-1.zip';user.write_bytes(b'user file');self.input.write_bytes(b'changed input inode');self.assertFalse(self.worker.once());self.assertTrue(self.input.exists());self.assertTrue(user.exists())
 def test_symlink_and_hardlink_alias_refuse(self):
  alias=self.root/'protected';os.link(self.input,alias);self.assertFalse(self.worker.once());self.assertTrue(self.input.exists());alias.unlink();original=self.root/'original';self.input.rename(original);self.input.symlink_to(original);self.assertFalse(self.worker.once());self.assertTrue(original.exists());self.assertTrue(self.input.is_symlink())
 def test_missing_terminal_diagnostics_and_unauthenticated_job_do_not_delete(self):
  (self.attempt/'expired-completed-lease.json').unlink();self.assertFalse(self.worker.once());self.assertTrue(self.input.exists());self.save_diag();p=self.attempt/'job.json';e=json.loads(p.read_bytes());e['payload']['job_id']='foreign';p.write_bytes(canonical(e))
  with self.assertLogs(level='WARNING'):self.assertFalse(self.worker.once())
  self.assertTrue(self.input.exists())
 def test_live_or_nonfinite_time_diagnostic_rejected(self):
  self.diag['lease_until']=time.time()+100;self.save_diag()
  with self.assertLogs(level='WARNING'):self.assertFalse(self.worker.once())
  self.assertTrue(self.input.exists())
 def test_storage_fault_does_not_kill_polling(self):
  with patch.object(CacheLifecycle,'retire_downloads',side_effect=OSError('storage offline')),self.assertLogs(level='WARNING'):self.assertFalse(self.worker.once())
  self.assertTrue(self.input.exists());self.assertFalse(self.worker.once());self.assertFalse(self.input.exists())
 def test_future_report_expiry_retires_after_backend_and_lease_close(self):
  from types import SimpleNamespace
  from subnet.distributed_worker import ExpiredCompletedLease
  root=self.root/'future';worker=Worker('http://127.0.0.1:19081',bytes(SigningKey.generate()),self.worker.authority,root);claim={'job':json.loads((self.attempt/'job.json').read_bytes()),'job_sha256':digest(self.job),'token':'opaque','attempt':1,'lease_until':time.time()+100};download=root/'backend/jobs/original-job/submission-0.zip'
  def request(action,**kw):
   if action=='claim':return {'claim':claim}
   if action=='report':claim['lease_until']=time.time()-1;raise ValueError('original expired lease')
   return {'lease_until':claim['lease_until']}
  def backend(args,**kw):
   download.parent.mkdir(parents=True);download.write_bytes(b'durable proof bytes');CacheLifecycle(root/'backend').record_download(download,self.job['submissions'][0]['sha256']);(download.parent/'report.json').write_bytes(canonical({'job_id':'original-job'}));return SimpleNamespace(returncode=0)
  worker.request=request
  with patch('subnet.distributed_worker.subprocess.run',side_effect=backend),self.assertRaises(ExpiredCompletedLease):worker.once()
  self.assertFalse(download.exists());self.assertTrue((root/'original-job/attempt-1/pending-report.json').exists());self.assertTrue((root/'original-job/attempt-1/expired-completed-lease.json').exists());self.assertTrue((download.parent/'report.json').exists())
 def test_idle_inventory_storage_error_does_not_stop_worker(self):
  with patch.object(self.worker,'sweep_expired_inputs',side_effect=OSError('inventory unavailable')),self.assertLogs(level='WARNING'):self.assertFalse(self.worker.once())
  self.assertTrue(self.input.exists())
 def test_other_receipted_download_outside_signed_submission_inventory_preserved(self):
  foreign=self.input.parent/'submission-7.zip';foreign.write_bytes(b'foreign signed scope');self.lifecycle.record_download(foreign,hashlib.sha256(b'foreign signed scope').hexdigest());self.assertFalse(self.worker.once());self.assertFalse(self.input.exists());self.assertEqual(foreign.read_bytes(),b'foreign signed scope')
if __name__=='__main__':unittest.main()
