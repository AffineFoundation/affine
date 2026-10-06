import copy,hashlib,json,os,tempfile,unittest
from pathlib import Path
from subnet.storage import canonical
from subnet.training_receipts import sha
from subnet.optimizer_state_cache import identifier
from ops.retire_failed_training_candidate import retire,VERSION
from test_training_parent_restore_recovery import ParentRestoreRecovery

class FailedCandidateRetirement(unittest.TestCase):
 def setUp(self):
  fx=ParentRestoreRecovery();fx.setUp();self.addCleanup(fx.doCleanups);self.sign=fx.sign;self.authority=fx.authority;self.job=copy.deepcopy(fx.original);self.job['persistent_training']['output_shards']['state-000001.safetensors']={}
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.owner=self.root/'jobs'/self.job['job_id'];self.owner.mkdir(parents=True)
  self.transfer=self.owner/('.fp32-state-transfer-'+'a'*32);self.transfer.mkdir();self.cache=self.root/'.optimizer-state-cache';self.cache.mkdir();(self.cache/'lease').touch();self.candidate=self.cache/('candidate-'+identifier(self.job['job_id']));self.candidate.mkdir()
  rows=[];receipts=[];files={}
  for i in range(2):
   name='state-%06d.safetensors'%i;p=(self.transfer if i==0 else self.candidate)/name;p.write_bytes(b'original-failed-byte'+bytes([i]));st=p.stat();h=hashlib.sha256(p.read_bytes()).hexdigest()
   row=dict(name=name,size=st.st_size,sha256=h,device=st.st_dev,inode=st.st_ino,uid=st.st_uid,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,kind='original_transfer'if i==0 else'pending_owned');rows.append(row)
   key=('private/failed-training-evidence/'+self.job['job_id']+'/'+name)if i==0 else self.job['persistent_training']['output_namespace']+'/'+name
   receipts.append(self.sign(dict(version='full-object-readback-evidence-v1',original_job_sha256=sha(self.job),key=key,sha256=h,size=st.st_size,bytes_read=st.st_size,complete=True)))
   if i:files[name]=dict(sha256=h,size=st.st_size)
  pending=dict(job_id=self.job['job_id'],job_sha256=sha(self.job),source_sha256=self.job['manifest']['payload']['source_bundle']['sha256'],descriptor_sha256=None,files=files);raw=canonical(pending);(self.cache/'pending.json').write_bytes(raw)
  self.value=dict(retire_failed_zero=True,version=VERSION,execute_allowed=True,created_at=1,expires_at=100,original_signed_job=self.sign(self.job),original_job_sha256=sha(self.job),original_terminal=fx.terminal,failure_evidence=dict(original_optimizer_updates=1,candidate_committed=False,original_report_absent=True,complete_candidate_descriptor_absent=True,original_processes_absent=True,no_active_checkpoint_lease=True,preserve_failure_history=True),durable_failure_evidence=self.sign(dict(version='full-object-readback-evidence-v1',original_job_sha256=sha(self.job),key='private/failed-training-evidence/'+self.job['job_id']+'/failure-evidence.private.json',sha256='f'*64,size=1024,bytes_read=1024,complete=True)),transfer_directory=str(self.transfer),candidate_directory=str(self.candidate),pending_sha256=hashlib.sha256(raw).hexdigest(),inventory=rows,full_readbacks=receipts,journal=str(self.owner/'failed-candidate-retirement.json'))
 def run_retire(self,value=None,guard=lambda *args:True):return retire(self.sign(value or self.value),self.authority,workspace=self.root,guard=guard,now=10)
 def test_genuine_complete_readbacks_preserve_failed_zero_and_history_before_delete(self):
  result=self.run_retire();self.assertEqual(result['retired_files'],2);self.assertFalse(result['optimizer_candidate_promoted']);self.assertFalse(self.transfer.exists());self.assertFalse(self.candidate.exists());journal=json.loads(Path(self.value['journal']).read_bytes());self.assertEqual(journal['grant'],self.sign(self.value));self.assertEqual(journal['preserved_pending_catalogue']['job_id'],self.job['job_id'])
 def test_missing_fullreadback_head_only_or_failedzero_not_preserved_refuses(self):
  for field,new in [('complete',False),('bytes_read',0),('sha256','0'*64),('key','private/unrelated')]:
   d=copy.deepcopy(self.value);row=d['full_readbacks'][0]['payload'];row[field]=new;d['full_readbacks'][0]=self.sign(row)
   with self.subTest(field=field),self.assertRaises(ValueError):self.run_retire(d)
  self.assertTrue(self.transfer.exists());self.assertFalse(Path(self.value['journal']).exists())
 def test_unowned_inode_mutation_or_symlink_no_delete(self):
  for key,new in [('inode',0),('uid',-1),('sha256','0'*64)]:
   d=copy.deepcopy(self.value);d['inventory'][1][key]=new
   with self.subTest(key=key),self.assertRaises(ValueError):self.run_retire(d)
  p=self.candidate/'state-000001.safetensors';p.unlink();p.symlink_to(self.transfer/'state-000000.safetensors')
  with self.assertRaises(Exception):self.run_retire()
  self.assertTrue(self.transfer.exists())
 def test_actual_active_original_or_lease_blocks(self):
  with self.assertRaises(ValueError):self.run_retire(guard=lambda *a:False)
  import fcntl
  with (self.cache/'lease').open('r+')as f:
   fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   with self.assertRaises(BlockingIOError):self.run_retire()
  self.assertTrue(self.transfer.exists())
 def test_scope_expiry_wrong_namespace_or_complete_candidate_refused(self):
  for key,new in [('expires_at',9),('execute_allowed',False),('candidate_directory',str(self.cache/'current')),('pending_sha256','0'*64)]:
   d=copy.deepcopy(self.value);d[key]=new
   with self.subTest(key=key),self.assertRaises(Exception):self.run_retire(d)
  raw=json.loads((self.cache/'pending.json').read_bytes());raw['descriptor_sha256']='a'*64;(self.cache/'pending.json').write_bytes(canonical(raw))
  with self.assertRaises(ValueError):self.run_retire()

 def test_crash_after_first_unlink_resumes_same_grant_and_replay_completed(self):
  from unittest.mock import patch
  original=Path.unlink;count=[0]
  def unlink(path,*args,**kwargs):
   count[0]+=1
   if count[0]==2:raise RuntimeError('simulated crash')
   return original(path,*args,**kwargs)
  with patch.object(Path,'unlink',unlink),self.assertRaises(RuntimeError):self.run_retire()
  self.assertTrue(Path(self.value['journal']).exists());result=self.run_retire();self.assertEqual(result['retired_files'],2);self.assertEqual(self.run_retire(),result)
 def test_resume_changed_grant_or_unowned_new_file_refused(self):
  from unittest.mock import patch
  with patch.object(Path,'unlink',side_effect=RuntimeError('crash')),self.assertRaises(RuntimeError):self.run_retire()
  changed=copy.deepcopy(self.value);changed['expires_at']=101
  with self.assertRaisesRegex(ValueError,'grant changed'):self.run_retire(changed)
  (self.candidate/'unowned.txt').write_text('external')
  with self.assertRaisesRegex(ValueError,'unowned'):self.run_retire()
 def test_thirteen_only_mode_preserves_failed_zero_locally(self):
  self.value['retire_failed_zero']=False;self.value['inventory']=self.value['inventory'][1:];self.value['full_readbacks']=self.value['full_readbacks'][1:]
  result=self.run_retire();self.assertEqual(result['retired_files'],1);self.assertTrue((self.transfer/'state-000000.safetensors').exists());self.assertFalse(self.candidate.exists());self.assertEqual(self.run_retire(),result)
