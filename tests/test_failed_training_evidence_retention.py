import copy,hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from subnet.storage import canonical
from subnet.training_receipts import sha
from ops.failed_training_evidence_retention import retire,VERSION
from test_training_post_update_recovery import PostUpdateRecovery

class FailedEvidenceRetention(unittest.TestCase):
 def setUp(self):
  f=PostUpdateRecovery();f.setUp();self.addCleanup(f.doCleanups);self.f=f;self.sign=f.sign;self.authority=f.authority
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name);self.owner=self.root/'jobs'/f.original['job_id'];self.owner.mkdir(parents=True);self.transfer=self.owner/('.fp32-state-transfer-'+'a'*32);self.transfer.mkdir();(self.root/'.optimizer-state-cache').mkdir();(self.root/'.optimizer-state-cache'/'lease').touch();(self.root/'runner-status').mkdir()
  (self.root/(f.original['job_id']+'.json')).write_bytes(canonical(self.sign(f.original)));(self.root/'runner-status'/(f.original['job_id']+'.json')).write_bytes(canonical(f.terminal));(self.owner/'worker.log').write_text('original failure preserved')
  rows=[]
  for name,data in [('state-000000.safetensors',b'original uncommitted failedzero'),('failure-000000.json',b'{"failure":"original"}')]:
   p=self.transfer/name;p.write_bytes(data);st=p.stat();rows.append(dict(name=name,size=st.st_size,sha256=hashlib.sha256(data).hexdigest(),device=st.st_dev,inode=st.st_ino,uid=st.st_uid,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns))
  job=copy.deepcopy(f.job);m=job['manifest']['payload'];report,_=f.fx.report();state=report['persistent_training_state'];d=state['descriptor'];d.update(epoch=m['epoch'],optimizer_steps=job['persistent_training']['global_step_after'],input_checkpoint=m['trainer_state_binding']['input_checkpoint'],parent_state_sha256=m['trainer_state_binding']['parent']['descriptor_sha256']);d['parameter_steps']={n:d['optimizer_steps']for n in d['parameter_steps']};state['descriptor_sha256']=sha(d);state['namespace']=job['persistent_training']['output_namespace']
  report.update(epoch=m['epoch'],checkpoint=m['checkpoint']['id'],job_id=job['job_id'],job_sha256=sha(job),success=True)
  self.report=report;self.job=job;self.newroot=self.root/'jobs'/job['job_id'];self.newroot.mkdir();(self.root/(job['job_id']+'.json')).write_bytes(canonical(self.sign(job)));(self.newroot/'report.json').write_bytes(canonical(report));(self.root/'runner-status'/(job['job_id']+'.json')).write_bytes(canonical(dict(job_id=job['job_id'],phase='complete',exit_code=0,runner_pid=999997,runner_pid_ticks='1001',child_pid=999998,child_pid_ticks='1002')))
  ack=dict(version='durable-original-trainer-cache-ACK-v1',job_id=job['job_id'],job_sha256=sha(job),report_sha256=sha(report),input_checkpoint=m['checkpoint'],input_cache=None,new_checkpoint=report['new_checkpoint'],trainer_state=dict(descriptor_sha256=state['descriptor_sha256'],namespace=state['namespace'],optimizer_steps=d['optimizer_steps']),authority_state_committed=True)
  self.value=dict(version=VERSION,execute_allowed=True,created_at=1,expires_at=100,original_signed_job=self.sign(f.original),original_terminal=f.terminal,durable_recovery_ACK=self.sign(ack),directory=str(self.transfer),files=rows,journal=str(self.owner/'evidence-retention.json'))
  self.archives={}
 def archive(self,path,key,row):
  data=path.read_bytes();self.archives[key]=data;return dict(version='full-object-readback-evidence-v1',key=key,sha256=hashlib.sha256(data).hexdigest(),size=len(data),bytes_read=len(data),complete=True)
 def call(self,value=None,archive=None):
  with patch('ops.training_retention.gpu_processes',return_value=[]),patch('ops.training_retention.processes',return_value=[]):return retire(self.sign(value or self.value),self.authority,workspace=self.root,archive=archive or self.archive,policy={'version':VERSION},now=10)
 def test_default_off_never_reads_or_mutates(self):
  self.assertEqual(retire(None,None,workspace='/nonexistent',archive=None)['status'],'disabled');self.assertTrue(self.transfer.exists())
 def test_genuine_durable_replacement_archives_all_bytes_then_retires_and_replays(self):
  result=self.call();self.assertEqual(result['status'],'complete');self.assertFalse(result['incomplete_optimizer_promoted']);self.assertFalse(self.transfer.exists());self.assertEqual(len(self.archives),2);self.assertTrue((self.owner/'worker.log').exists());self.assertTrue((self.newroot/'report.json').exists());self.assertEqual(self.call(),result)
 def test_uncommitted_or_forged_ack_or_report_wrong_original_blocks(self):
  for field,new in [('authority_state_committed',False),('report_sha256','0'*64),('job_sha256','0'*64)]:
   v=copy.deepcopy(self.value);ack=v['durable_recovery_ACK']['payload'];ack[field]=new;v['durable_recovery_ACK']=self.sign(ack)
   with self.subTest(field=field),self.assertRaises(ValueError):self.call(v)
  self.assertTrue(self.transfer.exists());self.assertFalse(Path(self.value['journal']).exists())
 def test_HEAD_partial_GET_or_wrongarchive_leaves_every_unarchived_file(self):
  def bad(path,key,row):
   result=self.archive(path,key,row);result['bytes_read']=0;return result
  with self.assertRaises(ValueError):self.call(archive=bad)
  self.assertTrue((self.transfer/'state-000000.safetensors').exists());self.assertFalse(Path(self.value['journal']+'.complete.json').exists());self.call()
 def test_ownership_change_unlistedmember_and_symlink_refuse(self):
  d=copy.deepcopy(self.value);d['files'][0]['inode']=0
  with self.assertRaises(ValueError):self.call(d)
  (self.transfer/'unowned.txt').write_text('keep')
  with self.assertRaisesRegex(ValueError,'unowned'):self.call()
  self.assertTrue((self.transfer/'unowned.txt').exists())
 def test_crash_after_first_unlink_resumes_archived_allowlist_only(self):
  original=Path.unlink;count=[0]
  def unlink(path,*a,**k):
   if path.name=='state-000000.safetensors':count[0]+=1;original(path,*a,**k);raise RuntimeError('crash after original unlink')
   return original(path,*a,**k)
  with patch.object(Path,'unlink',unlink),self.assertRaises(RuntimeError):self.call()
  self.assertEqual(count[0],1);result=self.call();self.assertEqual(result['retired_files'],2)
 def test_active_replacement_or_inherited_lease_never_unlinks(self):
  import fcntl
  with (self.root/'.optimizer-state-cache'/'lease').open('r+')as f:
   fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   with self.assertRaises(BlockingIOError):self.call()
  ticks=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19];(self.root/'runner-status'/('other-job.json')).write_bytes(canonical(dict(child_pid=os.getpid(),child_pid_ticks=ticks)))
  self.assertEqual(self.call()['status'],'deferred');self.assertTrue(self.transfer.exists())

 def test_bucket_adapter_streams_private_immutable_PUT_and_full_GET(self):
  import io
  from types import SimpleNamespace
  from ops.failed_training_evidence_retention import bucket_archive
  objects={};reads=[]
  class Body(io.BytesIO):
   def read(self,n=-1):reads.append(n);return super().read(n)
  class Client:
   def put_object(self,**kwargs):
    self.assert_private=kwargs['Key'].startswith('private/failed-training-evidence/');objects[kwargs['Key']]=kwargs['Body'].read();assert kwargs['IfNoneMatch']=='*'
   def get_object(self,**kwargs):return {'Body':Body(objects[kwargs['Key']]),'ResponseMetadata':{'HTTPStatusCode':200}}
  bucket=SimpleNamespace(name='private-test',client=Client());row=self.value['files'][0];path=self.transfer/row['name'];key='private/failed-training-evidence/'+self.f.original['job_id']+'/'+row['name'];receipt=bucket_archive(bucket)(path,key,row)
  self.assertTrue(bucket.client.assert_private);self.assertEqual(receipt['sha256'],row['sha256']);self.assertEqual(receipt['bytes_read'],row['size']);self.assertTrue(all(x==8*1024**2 for x in reads))
 def test_wrong_replacement_original_and_tampered_report_or_expired_scope_refuse(self):
  v=copy.deepcopy(self.value);v['expires_at']=9
  with self.assertRaises(ValueError):self.call(v)
  reportpath=self.newroot/'report.json';report=json.loads(reportpath.read_bytes());report['success']=False;reportpath.write_bytes(canonical(report))
  with self.assertRaises(ValueError):self.call()
  self.assertTrue(self.transfer.exists());self.assertFalse(Path(self.value['journal']).exists())

 def test_wrong_execution_source_cannot_reuse_genuine_durability(self):
  job=copy.deepcopy(self.job);m=job['manifest']['payload'];m['source_bundle']['sha256']='0'*64;job['manifest']=self.sign(m)
  (self.root/(job['job_id']+'.json')).write_bytes(canonical(self.sign(job)))
  report=copy.deepcopy(self.report);report['job_sha256']=sha(job);(self.newroot/'report.json').write_bytes(canonical(report));v=copy.deepcopy(self.value);ack=v['durable_recovery_ACK']['payload'];ack.update(job_sha256=sha(job),report_sha256=sha(report));v['durable_recovery_ACK']=self.sign(ack)
  with self.assertRaises(ValueError):self.call(v)
  self.assertTrue(self.transfer.exists());self.assertFalse(Path(self.value['journal']).exists())

 def test_default_off_asynchronous_hook_uses_exact_ROOT_blueprint_after_real_ACK(self):
  from types import SimpleNamespace
  from ops.failed_training_evidence_retention import schedule_after_durability
  controller=SimpleNamespace(authority=SimpleNamespace(id=self.authority),signed=self.sign);seen=[]
  self.assertEqual(schedule_after_durability(controller,None,None,dispatch=seen.append),[])
  blueprint={k:self.value[k]for k in ('version','original_signed_job','original_terminal','directory','files','journal')}
  policy=dict(version=VERSION,approved_failures=[self.sign(blueprint)])
  threads=schedule_after_durability(controller,self.value['durable_recovery_ACK'],policy,dispatch=seen.append)
  for t in threads:t.join(2)
  self.assertEqual(len(seen),1);grant=seen[0]['payload'];self.assertEqual(grant['original_signed_job'],self.value['original_signed_job']);self.assertEqual(grant['durable_recovery_ACK'],self.value['durable_recovery_ACK']);self.assertTrue(grant['execute_allowed'])
  bad=copy.deepcopy(self.value['durable_recovery_ACK']['payload']);bad['authority_state_committed']=False
  with self.assertRaises(ValueError):schedule_after_durability(controller,self.sign(bad),policy,dispatch=seen.append)
  self.assertEqual(len(seen),1);self.assertTrue(self.transfer.exists())
