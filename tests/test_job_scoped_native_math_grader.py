"""Real CPU-only isolated native parent/child controls; no Torch/GPU/network."""
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet import job_scoped_native_math_grader as g
from subnet.native_math_grader import ASSET,dependency_binding,isolated_argv

PYTHON=Path('/home/const/.cache/uv/environments-v2/97322a8c223cfacd3691a794fa0c412cc1c6d012ab316bad91ae56ff84e7d34e-12474372e5535b85/bin/python')
ROOT=Path(g.__file__).resolve().parent.parent
ENV='a'*64

def signed(p,key):return dict(payload=p,signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(g.canonical(p)).signature).decode())

def crash():os._exit(9)
def hang():time.sleep(5)
def overflow():print('X'*100000)
def output_a():print('1.0')

class Controls(unittest.TestCase):
 def setUp(self):
  if not PYTHON.exists():self.skipTest('prepared pinned native interpreter unavailable')
  self.directory=tempfile.TemporaryDirectory(prefix='native-job-controls-');self.addCleanup(self.directory.cleanup)
  self.snapshot=Path(self.directory.name)/'tasks.json';self.rows=[{'data':{'answer':'1'}},{'data':{'answer':'2'}}];self.snapshot.write_bytes(g.canonical(self.rows));self.key=SigningKey.generate()
  now=time.time();self.p=dict(version=g.VERSION,execute_allowed=True,job_id='isolated-CPU-control',created_at=now-1,expires_at=now+300,source_root=str(ROOT),source_files={n:g._digest(ROOT/n)for n in g._source_members(ROOT)},asset_files={str(self.snapshot):g._digest(self.snapshot)},grader_path=str(ASSET.resolve()),grader_sha256=g._digest(ASSET),interpreter=str(PYTHON),environment_sha256=ENV,snapshot_path=str(self.snapshot),snapshot_sha256=g._digest(self.snapshot),tasks={str(i):dict(task_hash=str(i+1)*64,row_sha256=hashlib.sha256(g.canonical(r)).hexdigest())for i,r in enumerate(self.rows)},max_requests=8,outer_timeout_seconds=12,memory_bytes=768*1024**2,native_runtime_binding=next(iter(dependency_binding().values())))
 def client(self,p=None,**kw):
  return g.prepare_job_scoped_grader(signed(p or self.p,self.key),authority=self.key.verify_key.encode().hex(),job_id=self.p['job_id'],environment_sha256=ENV,before_model_construction=True,**kw)
 def grade(self,c,reply,index=0):return c.grade(index=index,task_hash=str(index+1)*64,reply=reply)
 def test_default_off_does_not_spawn_or_import_torch(self):
  with patch.object(g.subprocess,'Popen',side_effect=AssertionError('spawned')):self.assertIsNone(g.prepare_job_scoped_grader())
 def test_auth_signature_cross_job_environment_and_pre_model_guards(self):
  scope=signed(self.p,self.key);scope['signature']=base64.b64encode(b'x'*64).decode()
  for params in (dict(scope=scope,authority=self.key.verify_key.encode().hex(),job_id=self.p['job_id'],environment_sha256=ENV,before_model_construction=True),dict(scope=signed(self.p,self.key),authority=self.key.verify_key.encode().hex(),job_id='other',environment_sha256=ENV,before_model_construction=True),dict(scope=signed(self.p,self.key),authority=self.key.verify_key.encode().hex(),job_id=self.p['job_id'],environment_sha256='b'*64,before_model_construction=True),dict(scope=signed(self.p,self.key),authority=self.key.verify_key.encode().hex(),job_id=self.p['job_id'],environment_sha256=ENV,before_model_construction=False)):
   with self.subTest(params=list(params)),self.assertRaises(g.GraderUnavailable):g.JobScopedGrader(**params)
 def test_fake_execute_false_expiry_bounds_refused_before_spawn(self):
  for field,value in [('execute_allowed',False),('expires_at',time.time()-1),('max_requests',129),('max_requests',True),('outer_timeout_seconds',301),('memory_bytes',2*1024**3),('version','unknown')]:
   p=copy.deepcopy(self.p);p[field]=value
   with self.subTest(field=field),patch.object(g.subprocess,'Popen',side_effect=AssertionError('spawned')),self.assertRaises(g.GraderUnavailable):self.client(p)
 def test_cuda_initialized_refuses_spawn_from_worker(self):
  from types import SimpleNamespace
  with patch.dict(g.sys.modules,{'torch':SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True))}),patch.object(g.subprocess,'Popen',side_effect=AssertionError('spawned')),self.assertRaises(g.GraderUnavailable):self.client()
 def test_source_pin_inventory_snapshot_and_task_refusal(self):
  for mutate in ('digest','missing','extra','row','snapshot'):
   p=copy.deepcopy(self.p)
   if mutate=='digest':p['source_files']['subnet/job_scoped_native_math_grader.py']='0'*64
   if mutate=='missing':p['source_files'].pop(next(iter(p['source_files'])))
   if mutate=='extra':p['source_files']['injected.py']='0'*64
   if mutate=='row':p['tasks']['0']['row_sha256']='0'*64
   if mutate=='snapshot':p['snapshot_sha256']='0'*64
   with self.subTest(mutate=mutate),self.assertRaises(g.GraderUnavailable):self.client(p)
 def test_actual_fresh_children_outcomes_and_original_last_boxed(self):
  with self.client()as c:
   rows=[self.grade(c,x)for x in (r'\boxed{1}',r'\boxed{2}','no boxed',r'\boxed{2} then \boxed{1}')]
   self.assertEqual([x['score']for x in rows],[1.,0.,0.,1.]);self.assertEqual(len({x['fresh_child_pid']for x in rows}),4)
   for x in rows:
    with self.assertRaises(ProcessLookupError):os.kill(x['fresh_child_pid'],0)
 def test_actual_reference_per_task_and_wrong_task_context(self):
  with self.client()as c:self.assertEqual(self.grade(c,r'\boxed{2}',1)['score'],1.)
  with self.client()as c:
   with self.assertRaises(g.GraderUnavailable):c.grade(index=1,task_hash='wrong',reply=r'\boxed{2}')
   self.assertIsNotNone(c._process.poll())
 def test_oversized_prediction_fails_closed_and_reaps_parent(self):
  with self.client()as c:
   with self.assertRaises(g.GraderUnavailable):self.grade(c,'X'*g.MAX_REQUEST_BYTES)
   self.assertIsNotNone(c._process.poll())
 def test_request_exhaustion_no_cache_or_job2_reuse(self):
  p=copy.deepcopy(self.p);p['max_requests']=1
  with self.client(p)as c:
   self.grade(c,r'\boxed{1}')
   with self.assertRaises(g.GraderUnavailable):self.grade(c,r'\boxed{1}')
  with self.assertRaises(g.GraderUnavailable):self.grade(c,r'\boxed{1}')
 def test_post_start_snapshot_mutation_and_file_replacement_fail_closed(self):
  for replacement in (False,True):
   with self.subTest(replacement=replacement):
    self.snapshot.write_bytes(g.canonical(self.rows));self.p['asset_files'][str(self.snapshot)]=g._digest(self.snapshot);self.p['snapshot_sha256']=g._digest(self.snapshot)
    with self.client()as c:
     if replacement:
      new=self.snapshot.with_suffix('.replacement');new.write_bytes(self.snapshot.read_bytes());os.replace(new,self.snapshot)
     else:self.snapshot.write_bytes(g.canonical([{'data':{'answer':'9'}},self.rows[1]]))
     with self.assertRaises(g.GraderUnavailable):self.grade(c,r'\boxed{1}')
 def test_unchanged_native_parser_timeout_and_bad_reference_are_infra(self):
  p=copy.deepcopy(self.p);p['outer_timeout_seconds']=.1
  with self.client(p)as c:
   r=self.grade(c,r'\boxed{2^{2^{1000000}}}');self.assertEqual(r['returncode'],75);self.assertIsNone(r['score']);self.assertIn('timeout',r['stderr'])
  self.rows[0]['data']['answer']='';self.snapshot.write_bytes(g.canonical(self.rows));self.p['asset_files'][str(self.snapshot)]=g._digest(self.snapshot);self.p['snapshot_sha256']=g._digest(self.snapshot);self.p['tasks']['0']['row_sha256']=hashlib.sha256(g.canonical(self.rows[0])).hexdigest()
  with self.client()as c:
   r=self.grade(c,r'\boxed{1}');self.assertEqual(r['returncode'],75);self.assertIsNone(r['score'])
 def test_parent_dependency_mutation_guard_detects_added_file(self):
  # Use a private cloned closure fixture, never mutate installed dependencies.
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);(root/'one.py').write_text('x=1');ns={'sysconfig':type('C',(),{'get_path':staticmethod(lambda _:str(root))}),'package_roots':[]};guard=g._runtime_file_guard(ns);self.assertTrue(guard());(root/'new.py').write_text('x=2');self.assertFalse(guard())
 def test_fresh_child_crash_timeout_and_output_overflow_neutral(self):
  for fn,timeout in ((crash,.2),(hang,.05),(overflow,.2)):
   with self.subTest(fn=fn.__name__):
    r,pid,_=g._run_child(fn,{'reply':'x'},'1',timeout,768*1024**2);self.assertEqual(r['returncode'],75)
    with self.assertRaises(ProcessLookupError):os.kill(pid,0)
 def test_fork_refused_in_multithreaded_or_torch_parent(self):
  with patch.dict(g.sys.modules,{'torch':object()}),patch.object(g.os,'fork',side_effect=AssertionError('forked')),self.assertRaises(g.GraderUnavailable):g._run_child(output_a,{'reply':'x'},'1',1,768*1024**2)
 def test_forged_scalar_and_wrong_result_binding_close_parent(self):
  for stdout,job in (('nan',self.p['job_id']),('1.0','other')):
   with self.subTest(stdout=stdout,job=job),self.client()as c:
    fake=dict(job_id=job,sequence=1,returncode=0,stdout=stdout,stderr='',elapsed_seconds=.1,fresh_child_pid=123)
    with patch.object(c,'_receive',return_value=fake),self.assertRaises(g.GraderUnavailable):self.grade(c,r'\boxed{1}')
    self.assertIsNotNone(c._process.poll())

 def test_partial_frame_and_idle_deadline_are_bounded(self):
  readfd,writefd=os.pipe()
  try:
   os.write(writefd,b'{"partial":')
   started=time.monotonic()
   with self.assertRaises(g.GraderUnavailable):g._bounded_line(readfd,128,time.monotonic()+.04)
   self.assertLess(time.monotonic()-started,.5)
  finally:os.close(readfd);os.close(writefd)
 def test_parent_expires_when_no_more_requests_arrive(self):
  p=copy.deepcopy(self.p);p['expires_at']=time.time()+4
  with self.client(p)as c:
   deadline=time.monotonic()+5
   while c._process.poll()is None and time.monotonic()<deadline:time.sleep(.05)
   self.assertIsNotNone(c._process.poll())
   with self.assertRaises(g.GraderUnavailable):self.grade(c,r'\boxed{1}')
 def test_malformed_result_and_parent_death_are_infra(self):
  with self.client()as c:
   os.kill(c._process.pid,signal.SIGKILL);c._process.wait(timeout=5)
   with self.assertRaises((g.GraderUnavailable,BrokenPipeError)):self.grade(c,r'\boxed{1}')
 def test_untrusted_child_state_never_carries_to_next_grade(self):
  global child_only_flag
  child_only_flag=False
  def dirty():
   global child_only_flag
   child_only_flag=True
   print('1.0')
  def clean():print('0.0'if child_only_flag else'1.0')
  a,pid,_=g._run_child(dirty,{'reply':'x'},'1',1,768*1024**2);b,other,_=g._run_child(clean,{'reply':'x'},'1',1,768*1024**2)
  self.assertEqual(a['stdout'].strip(),'1.0');self.assertEqual(b['stdout'].strip(),'1.0');self.assertFalse(child_only_flag);self.assertNotEqual(pid,other)
 def test_real_multithread_parent_refuses_fork(self):
  import threading
  stop=threading.Event();thread=threading.Thread(target=stop.wait);thread.start()
  try:
   with patch.object(g.os,'fork',side_effect=AssertionError('forked')),self.assertRaises(g.GraderUnavailable):g._run_child(output_a,{'reply':'x'},'1',1,768*1024**2)
  finally:stop.set();thread.join()

 def test_same_tick_directory_addition_cannot_hide_inventory_change(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);(p/'a.py').write_text('x=1');guard=g._DirectoryGuard([d]);baseline=guard.directories[d][0]
   (p/'new.py').write_text('x=2')
   with patch.object(g,'_metadata',return_value=baseline):self.assertFalse(guard.valid())
 def test_dependency_replacement_deletion_and_symlink_target_change(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);a=p/'a.py';a.write_text('x=1');link=p/'alias.py';link.symlink_to(a);ns={'sysconfig':type('C',(),{'get_path':staticmethod(lambda _:str(p))}),'package_roots':[]}
   guard=g._runtime_file_guard(ns)
   # Hold the original inode so this test proves a distinct replacement,
   # rather than filesystem inode reuse within one coarse timestamp tick.
   with a.open('rb')as original:
    a.unlink();a.write_text('x=1');self.assertFalse(guard())
   guard=g._runtime_file_guard(ns);a.unlink();self.assertFalse(guard())
 def test_parent_death_kills_its_inflight_cpu_child(self):
  import subprocess
  code="import time;from subnet.job_scoped_native_math_grader import _run_child;_run_child(lambda:time.sleep(30),{'reply':'x'},'1',30,768*1024**2)"
  parent=subprocess.Popen([g.sys.executable,'-c',code],cwd=ROOT,start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  child=None
  try:
   until=time.monotonic()+3
   while time.monotonic()<until:
    children=Path('/proc')/str(parent.pid)/'task'/str(parent.pid)/'children'
    values=children.read_text().split()if children.exists()else[]
    if values:child=int(values[0]);break
    time.sleep(.01)
   self.assertIsNotNone(child)
   # Give the just-created child time to install its kernel parent-death guard.
   time.sleep(.05);parent.kill();parent.wait(timeout=3)
   until=time.monotonic()+2;alive=True
   while time.monotonic()<until:
    status=Path('/proc')/str(child)/'stat'
    if not status.exists()or status.read_text().rsplit(')',1)[1].split()[0]=='Z':alive=False;break
    time.sleep(.01)
   self.assertFalse(alive,'CPU child survived parent crash')
  finally:
   if parent.poll()is None:parent.kill();parent.wait(timeout=3)

if __name__=='__main__':unittest.main()
