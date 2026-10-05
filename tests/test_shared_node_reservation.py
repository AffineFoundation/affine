import base64
import json
import multiprocessing
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from nacl.signing import SigningKey
from subnet.shared_node_reservation import SharedNodeReservation,SharedNodeBusy,VERSION,identity,reserved_evaluation_run
from subnet.backend_jobs import canonical

UUID='c1a01718-a9a1-4726-92fe-a568caf561dd'
def config(root):return dict(version=VERSION,directory=str(root),provider_UUID=UUID)
def barrier_claim(root,barrier,out,n):
 l=SharedNodeReservation(config(root),provider_uuid=UUID);barrier.wait()
 try:l.claim('evaluate',identity(n));out.put('claim')
 except SharedNodeBusy:out.put('blocked')

class LeaseTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'leases';self.l=SharedNodeReservation(config(self.root),provider_uuid=UUID);self.a=identity('eval');self.b=identity('reader');self.sig=identity('signed');self.h=dict(pid=123,ticks='456')
 def tearDown(self):self.tmp.cleanup()
 def terminal(self,key=None):
  self.l.bind(key or self.a,self.sig,self.h);return dict(actual_child_wait_completed=True,exit_code=0,finished_at=time.time())
 def test_actual_multiprocess_claim_barrier(self):
  ctx=multiprocessing.get_context('spawn');bar=ctx.Barrier(2);q=ctx.Queue();ps=[ctx.Process(target=barrier_claim,args=(self.root,bar,q,n))for n in range(2)]
  for p in ps:p.start()
  for p in ps:p.join(10);self.assertEqual(p.exitcode,0)
  self.assertCountEqual([q.get(timeout=2),q.get(timeout=2)],['claim','blocked'])
 def test_reader_priority_does_not_preempt_existing_eval(self):
  self.l.claim('evaluate',self.a)
  with self.assertRaises(SharedNodeBusy):self.l.claim('reader',self.b)
  self.l.claim('evaluate',self.a) # same original adoption allowed
  term=self.terminal();self.l.terminal(self.a,self.sig,self.h,term)
  with self.assertRaises(SharedNodeBusy):self.l.claim('evaluate',identity('new'))
  self.l.claim('reader',self.b)
 def test_actual_terminal_release_and_idempotent_recovery(self):
  self.l.claim('reader',self.b);term=self.terminal(self.b);self.l.terminal(self.b,self.sig,self.h,term)
  self.assertFalse(self.l.blocked());self.l.claim('reader',self.b);self.l.bind(self.b,self.sig,self.h);self.l.terminal(self.b,self.sig,self.h,term);self.assertFalse(self.l.blocked())
 def test_stale_pid_ticks_and_ambiguous_observation_never_release(self):
  self.l.claim('evaluate',self.a);term=self.terminal()
  with self.assertRaises(ValueError):self.l.terminal(self.a,self.sig,dict(pid=123,ticks='999'),term)
  with self.assertRaises(ValueError):self.l.terminal(self.a,self.sig,self.h,dict(actual_child_wait_completed=False,finished_at=time.time()))
  with self.assertRaises(SharedNodeBusy):self.l.claim('reader',self.b)
 def test_crash_recovery_retains_original_owner_no_clock_takeover(self):
  self.l.claim('reader',self.b);self.l.bind(self.b,self.sig,self.h)
  other=SharedNodeReservation(config(self.root),provider_uuid=UUID);a=other.claim('reader',self.b);self.assertEqual(a['handle'],self.h)
  with self.assertRaises(SharedNodeBusy):other.claim('evaluate',self.a)
  with self.assertRaises(ValueError):other.bind(self.b,identity('replacement'),self.h)
 def test_concurrent_same_original_dispatch_blocked_crash_lock_released(self):
  with self.l.executing(self.a):
   other=SharedNodeReservation(config(self.root),provider_uuid=UUID)
   with self.assertRaises(SharedNodeBusy):
    with other.executing(self.a):pass
  with other.executing(self.a):pass
 def test_failed_actual_wait_can_release_without_fake_success(self):
  self.l.claim('reader',self.b);term=self.terminal(self.b);term['exit_code']=1;self.l.terminal(self.b,self.sig,self.h,term);self.assertFalse(self.l.blocked())
  self.assertEqual(json.loads(self.l.path.read_text())['history'][0]['terminal']['exit_code'],1)
 def test_provider_mismatch_and_public_directory_rejected(self):
  with self.assertRaises(ValueError):SharedNodeReservation(config(self.root),provider_uuid='e566fb5f-0000-0000-0000-000000000000')
  self.root.chmod(0o755)
  with self.assertRaises(ValueError):SharedNodeReservation(config(self.root),provider_uuid=UUID)
 def test_symlink_journal_rejected(self):
  self.l.path.symlink_to('/tmp/nonexistent-lease-control')
  with self.assertRaises((ValueError,FileNotFoundError)):self.l.claim('evaluate',self.a)

class EvaluatorAdapterTests(LeaseTests):
 def setup_jobs(self,status,fail=False):
  key=SigningKey.generate();authority=key.verify_key.encode().hex();state=Path(self.tmp.name)/'roles';state.mkdir();manifest=dict(epoch='e');label='eval-original';fields=dict(heldout=[1]);calls=[]
  def sign(p):return dict(payload=p,signer=authority,signature=base64.b64encode(key.sign(canonical(p)).signature).decode())
  job=dict(role='evaluate',job_id='eval-original-abcd',manifest=sign(manifest),heldout=[1]);env=sign(job)
  prior=dict(job_id=job['job_id'],job_sha256=identity(job))
  (state/(label+'.json')).write_text(json.dumps(prior));(state/(job['job_id']+'-job.json')).write_text(json.dumps(env))
  def run(*args,**kw):
   calls.append(job['job_id'])
   if fail:raise TimeoutError('same job observation timeout')
   return dict(success=True)
  def command(code):
   if isinstance(status,Exception):raise status
   return json.dumps(status)
  jobs=SimpleNamespace(state=state,workspace='/original',python='/python',controller=SimpleNamespace(authority=SimpleNamespace(id=authority)),run=run,command=command)
  return jobs,label,manifest,fields,calls
 def test_original_terminal_releases_and_same_request_not_new_nonce(self):
  st=dict(job_id='eval-original-abcd',phase='complete',child_pid=123,child_pid_ticks='456',exit_code=0,finished_at=time.time()+1)
  jobs,label,m,f,c=self.setup_jobs(st);reserved_evaluation_run(self.l,jobs,label,m,None,f);reserved_evaluation_run(self.l,jobs,label,m,None,f)
  self.assertEqual(c,['eval-original-abcd']*2);self.assertFalse(self.l.blocked())
 def test_ssh_ambiguous_timeout_retains_lease_original_adoption(self):
  jobs,label,m,f,c=self.setup_jobs(RuntimeError('SSH timeout'),fail=True)
  for _ in range(2):
   with self.assertRaises(TimeoutError):reserved_evaluation_run(self.l,jobs,label,m,None,f)
  self.assertEqual(c,['eval-original-abcd']*2)
  with self.assertRaises(SharedNodeBusy):self.l.claim('reader',self.b)
 def test_report_without_actual_wait_marker_does_not_release(self):
  jobs,label,m,f,c=self.setup_jobs(dict(job_id='eval-original-abcd',phase='complete'));reserved_evaluation_run(self.l,jobs,label,m,None,f);self.assertTrue(self.l.blocked())

class ReaderRecoveryTests(unittest.TestCase):
 def test_original_reader_recovery_does_not_spawn_or_change_handle(self):
  from subnet.independent_state_dispatch import IndependentStateReader,file_sha
  from unittest.mock import patch
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);lease=SharedNodeReservation(config(root/'lease'),provider_uuid=UUID);jobsha=identity('original-training-job');lease.claim('reader',jobsha)
   request=dict(payload=dict(expires_at=time.time()+30));req=root/'request.ROOT-SIGNED.private.json';req.write_text(json.dumps(request));launch=root/'launch.ROOT-SIGNED.private.json';launch.write_text('{}')
   marker=dict(pid=42,ticks='123',request_file_sha256=file_sha(req),launch_file_sha256=file_sha(launch));child=dict(pid=43,ticks='124');terminal=dict(pid=43,ticks='124',actual_child_wait_completed=True,exit_code=0,timed_out=False,finished_at=time.time())
   status=dict(original_supervisor=marker,child=child,terminal=terminal,result={'actual':'receipt'},supervisor_live=False)
   instance=IndependentStateReader.__new__(IndependentStateReader);instance.shared_lease=lease;instance.authority='authority';instance.config=dict(reader_identity='reader',reader_host={'provider_UUID':UUID},trainer_host={'provider_UUID':'trainer'});calls=[]
   instance.command=lambda code:(calls.append(code)or status)
   binding=dict(job_sha256=jobsha,storage_origin='origin',storage_bucket='bucket',storage_addressing='path')
   with patch('subnet.independent_state_dispatch.reader.validate_receipt'):
    for _ in range(2):
     lease.claim('reader',jobsha);result=instance._wait_original(root,request,launch,binding,[], '/original/unchanged')
     self.assertEqual(result['original_child'],child)
   self.assertFalse(lease.blocked());self.assertEqual(len(calls),2)
   self.assertTrue(all('Popen'not in c and 'subprocess'not in c for c in calls))
