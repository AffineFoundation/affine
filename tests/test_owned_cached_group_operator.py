"""CPU fixtures only: actual files/locks/disposal, no GPU or native score claims."""
import base64,copy,hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet.backend_jobs import canonical,file_map
from subnet.backend_profiles import profile,HOPPER_FP32_REVISION
from subnet.cache_lifecycle import CacheLifecycle
from subnet.owned_cached_evaluation import cohort,POLICY
from ops.owned_cached_group_retention import VERSION,CACHE_SHA
from ops.owned_cached_group_operator import GroupOperator,GroupACKPublisher,private_json,validate_originals
from ops.owned_cached_larger_cohort import SOURCE
class Bucket:
 def __init__(self):self.objects={};self.bad=False
 def put(self,key,data):self.objects[key]=data
 def get(self,key):return b'corrupt'if self.bad else self.objects[key]
class Transport:
 def __init__(self,test):self.test=test;self.launched=[];self.statuses={};self.reports={};self.acks={};self.fail_launch=False;self.reserved=False
 def free_bytes(self):return 25*1024**3
 def idle(self):return not self.reserved
 def launch(self,envelope):
  job=envelope['payload'];self.launched.append(job['job_id'])
  if self.fail_launch:raise TimeoutError('lost launch reply')
 def status(self,jid):return self.statuses.get(jid,{'phase':'not_launched'})
 def live(self,status):return status.get('test_live',False)
 def report(self,jid):return self.reports[jid]
 def ack(self,jid):return self.acks.get(jid)
class OperatorTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex();self.now=10;self.files={f'subnet/f{i}.py':hashlib.sha256(str(i).encode()).hexdigest()for i in range(176)};self.files['subnet/cache_lifecycle.py']=CACHE_SHA;cpfiles={'model.safetensors':hashlib.sha256(b'cpu-fixture').hexdigest(),'config.json':hashlib.sha256(b'{}').hexdigest()};self.cp={'files':cpfiles,'id':file_map(cpfiles)};directory=self.root/'checkpoints'/self.cp['id'];directory.mkdir(parents=True);(directory/'model.safetensors').write_bytes(b'cpu-fixture');(directory/'config.json').write_bytes(b'{}');self.cache=CacheLifecycle(self.root)
  with self.cache.lease_checkpoint(self.cp['id']):self.cache.record_checkpoint(self.cp['id'],cpfiles)
  rev,backend,numerical=profile(HOPPER_FP32_REVISION);self.groups=[dict(group=g,env_id='affine_math',indices=list(range(6746+g*32,6746+(g+1)*32)),seeds=[20261002+i*1000 for i in range(6746+g*32,6746+(g+1)*32)],harness=dict(version='text-tools-long-kv-v3',policy='autoregressive',max_output_tokens=1024,temperature=.7,top_p=1.))for g in range(4)];self.jobs=[];bindings={};self.cohorts={}
  for g in self.groups:
   manifest=dict(epoch='cpu-fixture',checkpoint=self.cp,source_bundle={'sha256':SOURCE},environments=[dict(env_id='affine_math',indices=list(range(6746)),spec=dict(num_samples=7496))],model_runtime_revision=rev,backend_profile=backend,numerical_policy=numerical)
   job=dict(job_id='original-group-'+str(g['group']),role='evaluate',source_files=self.files,manifest=self.sign(manifest),created_at=1,expires_at=1000,owned_evaluation_policy=POLICY,runtime_versions={'fixture':'CPU'},heldout=[{k:v for k,v in g.items()if k!='group'}]);self.jobs.append(self.sign(job));bindings[job['job_id']]={'group':g['group'],'job_sha256':self.digest(job)};_,self.cohorts[job['job_id']]=cohort(manifest['environments'][0],job['heldout'][0],manifest,self.files)
  self.scope=dict(runtime_versions={'fixture':'CPU'},version=VERSION,execute_allowed=True,created_at=1,expires_at=1000,workspace=str(self.root),source_sha256=SOURCE,source_files=self.files,experiment_id='owned-cached-native-heldout128-cap1024-v1',groups=self.groups,original_jobs=bindings,checkpoint=self.cp,minimum_free_cold_bytes=20000000000,minimum_free_warm_margin_bytes=5000000000);self.transport=Transport(self);self.bucket=Bucket();(self.root/'runner-status').mkdir()
 def digest(self,v):return hashlib.sha256(canonical(v)).hexdigest()
 def sign(self,v):return dict(payload=v,signer=self.authority,signature=base64.b64encode(self.key.sign(canonical(v)).signature).decode())
 def op(self):return GroupOperator(self.sign(self.scope),self.authority,self.root,self.files,self.jobs,self.transport,clock=lambda:self.now)
 def publisher(self):return GroupACKPublisher(self.sign(self.scope),self.authority,self.jobs,self.bucket,self.sign,clock=lambda:self.now)
 def complete(self,n,ack=True):
  job=self.jobs[n]['payload'];manifest=job['manifest']['payload'];jid=job['job_id'];g=self.groups[n];status=dict(job_id=jid,phase='complete',exit_code=0,runner_pid=900000+n*2,runner_pid_ticks='123',child_pid=900001+n*2,child_pid_ticks='124',started_at=2,finished_at=3);self.transport.statuses[jid]=status;(self.root/'runner-status'/(jid+'.json')).write_bytes(canonical(status));(self.root/'runner-status'/(jid+'.json')).chmod(0o600)
  rows=[dict(env_id='affine_math',index=i,seed=s,checkpoint=self.cp['id'],cohort_sha256=self.cohorts[jid],task_hash=hashlib.sha256(str(i).encode()).hexdigest(),native_graded=True,verified=False,proof_verification_performed=False,trust_scope=POLICY['trust_scope'],classification='positive',reward=1)for i,s in zip(g['indices'],g['seeds'])]
  report=dict(job_id=jid,job_sha256=self.digest(job),operator=self.authority,role='evaluate',checkpoint=self.cp['id'],epoch=manifest['epoch'],source_files=self.files,runtime_versions=job['runtime_versions'],chain_transactions=False,success=True,completed_at=3,backend_profile=manifest['backend_profile'],numerical_policy=manifest['numerical_policy'],owned_cached_evaluation={'policy':POLICY},heldout=rows,heldout_failures=[]);self.transport.reports[jid]=report
  if ack:self.transport.acks[jid]=self.publisher().publish(jid,report,status,physical_original_absent=True)
 def test_four_distinct_original_jobs_and_full_ACKs_retire_once(self):
  with self.op()as op:
   for n in range(4):
    r=op.step();self.assertEqual(r['job_id'],'original-group-'+str(n));self.complete(n);self.assertEqual(op.step()['status'],'original-ACK-complete');self.assertTrue((self.root/'checkpoints'/self.cp['id']).exists())
   r=op.step();self.assertEqual(r['status'],'complete');self.assertEqual(r['score']['count'],128);self.assertFalse((self.root/'checkpoints'/self.cp['id']).exists());self.assertEqual(op.step()['status'],'complete')
  self.assertEqual(self.transport.launched,['original-group-'+str(n)for n in range(4)])
 def test_lost_launch_reply_and_reopened_operator_never_redispatch(self):
  self.transport.fail_launch=True
  with self.op()as op:
   with self.assertRaises(TimeoutError):op.step()
  self.transport.fail_launch=False
  with self.op()as op:self.assertEqual(op.step()['status'],'observing-original');self.assertEqual(len(self.transport.launched),1)
 def test_missing_ACK_blocks_next_job_and_keeps_model(self):
  with self.op()as op:
   op.step();self.complete(0,ack=False)
   for _ in range(2):self.assertEqual(op.step()['status'],'awaiting-full-R2-ACK')
   self.assertEqual(len(self.transport.launched),1);self.assertTrue((self.root/'checkpoints'/self.cp['id']).exists())
 def test_corrupt_bucket_readback_cannot_sign_ACK(self):
  self.complete(0,ack=False);self.bucket.bad=True;jid=self.jobs[0]['payload']['job_id']
  with self.assertRaises(ValueError):self.publisher().publish(jid,self.transport.reports[jid],self.transport.statuses[jid],physical_original_absent=True)
  self.assertIsNone(self.transport.ack(jid))
 def test_expired_scope_and_expired_unissued_job_do_not_launch(self):
  with self.op()as op:
   self.now=1000
   with self.assertRaises(ValueError):op.step()
  self.assertFalse(self.transport.launched)
  self.now=1000;self.scope['expires_at']=1100
  with self.op()as op:self.assertEqual(op.step()['status'],'expired-unissued')
  self.assertFalse(self.transport.launched)
 def test_actual_checkpoint_lease_defers_final_disposal(self):
  with self.op()as op:
   for n in range(4):op.step();self.complete(n);op.step()
   with self.cache.lease_checkpoint(self.cp['id']):self.assertEqual(op.step()['status'],'deferred')
   self.assertEqual(op.step()['status'],'complete')
 def test_changed_inode_or_foreign_owner_cannot_dispose(self):
  with self.op()as op:
   for n in range(4):op.step();self.complete(n);op.step()
   p=self.root/'checkpoints'/self.cp['id']/'model.safetensors';q=p.with_suffix('.new');q.write_bytes(p.read_bytes());q.replace(p);self.assertEqual(op.step()['status'],'deferred');self.assertTrue(p.exists())
  p=self.root/'owned.private.json';p.write_bytes(b'{}');p.chmod(0o600)
  with patch('ops.owned_cached_group_operator.os.geteuid',return_value=p.stat().st_uid+1):
   with self.assertRaises(ValueError):private_json(p)
 def test_hardlink_and_symlink_original_input_rejected(self):
  p=self.root/'private.json';p.write_bytes(b'{}');p.chmod(0o600);q=self.root/'other.json';os.link(p,q)
  with self.assertRaises(ValueError):private_json(p)
  q.unlink();q.symlink_to(p)
  with self.assertRaises(ValueError):private_json(q)
 def test_changed_job_group_source_seed_or_external_cache_refused(self):
  for change in('seed','source','external','duplicate'):
   jobs=copy.deepcopy(self.jobs)
   if change=='seed':jobs[0]['payload']['heldout'][0]['seeds'][0]+=1
   if change=='source':jobs[0]['payload']['source_files']={}
   if change=='external':jobs[0]['payload']['checkpoint_cache']='/unowned'
   if change=='duplicate':jobs[1]=jobs[0]
   jobs=[self.sign(j['payload'])for j in jobs]
   with self.assertRaises(ValueError):validate_originals(self.scope,jobs,self.authority)
 def test_changed_ACK_report_or_missing_original_provenance_refused(self):
  with self.op()as op:
   op.step();self.complete(0);jid=self.jobs[0]['payload']['job_id'];a=copy.deepcopy(self.transport.acks[jid]['payload']);a['original_report']['heldout'][0]['seed']+=1;self.transport.acks[jid]=self.sign(a)
   with self.assertRaises(ValueError):op.step()
   self.assertEqual(len(self.transport.launched),1)
 def test_failed_original_and_live_terminal_preserved_no_fake_score(self):
  with self.op()as op:
   op.step();self.complete(0);jid=self.jobs[0]['payload']['job_id'];self.transport.statuses[jid]['test_live']=True;self.assertEqual(op.step()['status'],'terminal-process-still-live');self.transport.statuses[jid].update(test_live=False,phase='failed',exit_code=1);r=op.step();self.assertEqual(r['status'],'original-infrastructure-failure');self.assertIsNone(r['model_reward']);self.assertEqual(len(self.transport.launched),1)
 def test_source_or_request_tampered_journal_cannot_restart(self):
  with self.op()as op:op.step()
  p=self.root/'.owned-cached-group-retention/operator.json';v=json.loads(p.read_bytes());v['binding_sha256']='foreign';p.write_bytes(canonical(v));p.chmod(0o600)
  with self.assertRaises(ValueError):
   with self.op():pass
 def test_changed_runtime_and_future_terminal_are_not_authentic_completion(self):
  jobs=copy.deepcopy(self.jobs);jobs[0]['payload']['runtime_versions']={'other':'CPU'};jobs[0]=self.sign(jobs[0]['payload']);scope=copy.deepcopy(self.scope);scope['original_jobs']['original-group-0']['job_sha256']=self.digest(jobs[0]['payload'])
  with self.assertRaises(ValueError):validate_originals(scope,jobs,self.authority)
  with self.op()as op:
   op.step();self.complete(0);self.transport.statuses['original-group-0']['finished_at']=self.now+1
   with self.assertRaises(ValueError):op.step()
 def test_fresh_scope_can_observe_same_expired_original_never_backdate_or_relaunch(self):
  with self.op()as op:op.step();self.complete(0,ack=False)
  original=copy.deepcopy(self.jobs[0]);self.now=1001;self.scope.update(created_at=1001,expires_at=2000)
  with self.op()as op:self.assertEqual(op.step()['status'],'awaiting-full-R2-ACK')
  self.assertEqual(self.jobs[0],original);self.assertEqual(self.transport.launched,['original-group-0'])
 def test_read_updates_atime_without_false_integrity_failure(self):
  p=self.root/'fresh-private.json';p.write_bytes(b'{"actual":1}');p.chmod(0o600);os.utime(p,ns=(1,p.stat().st_mtime_ns));self.assertEqual(private_json(p),{'actual':1})
 def test_actual_CPU_detached_original_launch_private_log_and_duplicate_refusal(self):
  import time
  from ops.owned_cached_group_operator import LocalOriginalTransport
  source=self.root/'cpu-stub';(source/'subnet').mkdir(parents=True);(source/'subnet/__init__.py').write_text('');(source/'subnet/remote_runner.py').write_text('import pathlib,sys;pathlib.Path(sys.argv[1]+".CPU-marker").write_text("actual CPU child");print("CPU-only")')
  t=LocalOriginalTransport.__new__(LocalOriginalTransport);t.root=self.root;t.code=source;t.authority=self.authority;pid=t.launch(self.jobs[0]);jobfile=self.root/'original-group-0.json';deadline=time.monotonic()+3
  while not Path(str(jobfile)+'.CPU-marker').exists()and time.monotonic()<deadline:time.sleep(.01)
  self.assertTrue(Path(str(jobfile)+'.CPU-marker').exists());self.assertEqual(t.children['original-group-0'].wait(timeout=3),0);self.assertEqual(private_json(jobfile),self.jobs[0]);self.assertEqual((self.root/'original-group-0-runner.log').stat().st_mode&0o777,0o600)
  with self.assertRaises(ValueError):t.launch(self.jobs[0])
if __name__=='__main__':unittest.main()
