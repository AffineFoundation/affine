import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from ops.recovery_checkpoint_retention import retire,VERSION
from subnet.storage import canonical
from subnet.training_receipts import sha
from test_training_parent_restore_recovery import ParentRestoreRecovery

class RecoveryModelRetention(unittest.TestCase):
 def setUp(self):
  f=ParentRestoreRecovery();f.setUp();self.addCleanup(f.doCleanups);self.sign=f.sign;self.authority=f.authority;self.job=f.original
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name);self.old=self.root/'checkpoint';self.old.mkdir();files={};metadata={}
  for n,data in [('config.json',b'{}'),('model.safetensors',b'qualified-owned-model')]:
   (self.old/n).write_bytes(data);files[n]=hashlib.sha256(data).hexdigest();metadata[n]=dict(sha256=files[n],size=len(data))
  self.cp=sha(files);self.path=self.root/'checkpoints'/self.cp;self.state=canonical(dict(checkpoint=f.old['checkpoint']));self.failure=canonical(f.terminal)
  self.v=dict(version=VERSION,execute_allowed=True,created_at=1,expires_at=100,controller_state_sha256=hashlib.sha256(self.state).hexdigest(),original_failure_sha256=hashlib.sha256(self.failure).hexdigest(),original_signed_job=self.sign(self.job),original_job_sha256=sha(self.job),held_controller={'unit':'failed-original','MainPID':0},protected_checkpoints=[f.old['checkpoint']['id']],active_checkpoints=[],required_free_bytes=512*1024**3,max_retire_bytes=1024,candidates=[dict(directory=str(self.path),legacy_directory=str(self.old),checkpoint_document=self.sign(dict(id=self.cp,files=files)),full_readback=self.sign(dict(version='complete-checkpoint-full-readback-v1',checkpoint=self.cp,complete=True,files=metadata,bytes_read=sum(x['size']for x in metadata.values()))))])
 def call(self,v=None,guard=lambda *a:True):
  with patch('ops.checkpoint_retention.gpu_processes',return_value=[]),patch('ops.checkpoint_retention.processes',return_value=[]):return retire(self.sign(v or self.v),self.authority,state_bytes=self.state,failure_bytes=self.failure,held_guard=guard,now=10)
 def test_exact_old_model_retirement_keeps_failure_current_and_parent_history(self):
  result=self.call();self.assertTrue(result['current_checkpoint_preserved']);self.assertEqual(result['retired_bytes'],23);self.assertFalse(self.old.exists());self.assertFalse(self.path.exists());self.assertEqual(self.call()['retired_bytes'],0)
 def test_failed_controller_not_held_or_wrong_state_blocks_before_mutation(self):
  with self.assertRaises(ValueError):self.call(guard=lambda *a:False)
  d=copy.deepcopy(self.v);d['controller_state_sha256']='0'*64
  with self.assertRaises(ValueError):self.call(d)
  self.assertTrue(self.old.exists())
 def test_current_active_or_wrong_readback_never_retired(self):
  for kind in ('protected','active','HEAD','hash','bytes','budget','expiry'):
   d=copy.deepcopy(self.v)
   if kind=='protected':d['protected_checkpoints'].append(self.cp)
   if kind=='active':d['active_checkpoints'].append(self.cp)
   if kind in ('HEAD','hash','bytes'):
    receipt=d['candidates'][0]['full_readback']['payload']
    if kind=='HEAD':receipt['complete']=False
    if kind=='hash':receipt['files']['config.json']['sha256']='0'*64
    if kind=='bytes':receipt['bytes_read']=0
    d['candidates'][0]['full_readback']=self.sign(receipt)
   if kind=='budget':d['max_retire_bytes']=1
   if kind=='expiry':d['expires_at']=9
   with self.subTest(kind=kind),self.assertRaises(ValueError):self.call(d)
  self.assertTrue(self.old.exists())
 def test_tampered_model_file_no_rename(self):
  (self.old/'config.json').write_bytes(b'tampered')
  with self.assertRaises(ValueError):self.call()
  self.assertTrue(self.old.exists())
 def test_bounded_floor_stops_without_discovery_or_deletion(self):
  d=copy.deepcopy(self.v);d['required_free_bytes']=1
  self.assertEqual(self.call(d)['retired_bytes'],0);self.assertTrue(self.old.exists())

 def test_actual_GPU_work_or_open_model_fd_refuses_before_rename(self):
  with patch('ops.checkpoint_retention.gpu_processes',return_value=['123']),self.assertRaises(ValueError):retire(self.sign(self.v),self.authority,state_bytes=self.state,failure_bytes=self.failure,held_guard=lambda *a:True,now=10)
  proc=self.root/'proc'/'123';(proc/'fd').mkdir(parents=True);(proc/'fd'/'0').symlink_to(self.old/'config.json');(proc/'maps').write_text('')
  with patch('ops.checkpoint_retention.gpu_processes',return_value=[]),patch('ops.checkpoint_retention.processes',return_value=[proc]),self.assertRaises(ValueError):retire(self.sign(self.v),self.authority,state_bytes=self.state,failure_bytes=self.failure,held_guard=lambda *a:True,now=10)
  self.assertTrue(self.old.exists());self.assertFalse(self.path.exists())
 def test_new_unowned_member_symlink_or_budget_changed_refused(self):
  (self.old/'unowned.txt').write_text('external')
  with self.assertRaises(ValueError):self.call()
  (self.old/'unowned.txt').unlink();p=self.old/'config.json';data=p.read_bytes();p.unlink();external=self.root/'external';external.write_bytes(data);p.symlink_to(external)
  with self.assertRaises(ValueError):self.call()
  self.assertTrue(external.exists())
