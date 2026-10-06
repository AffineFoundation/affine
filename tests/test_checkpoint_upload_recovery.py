import base64,copy,json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet.storage import canonical
from subnet.checkpoint_upload_recovery import select_label,sha,VERSION,computation
class UploadRecoveryTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.state=Path(self.tmp.name);self.roles=self.state/'roles';self.roles.mkdir();self.k=SigningKey.generate();self.authority=self.k.verify_key.encode().hex();self.c=SimpleNamespace(state=self.state,authority=SimpleNamespace(id=self.authority),checkpoint_upload_recovery_files={})
  self.m=dict(epoch='nonpayable-e30',checkpoint=dict(id='a'*64,files={'model.safetensors':'b'*64}),source_bundle={'sha256':'c'*64});self.label='nonpayable-e30-publish-aaaaaaaa';self.jid=self.label+'-00000000';self.now=time.time();self.original=dict(role='upload',job_id=self.jid,created_at=self.now-20,manifest=self.sign(self.m));self.failure=dict(job_id=self.jid,phase='failed',exit_code=1,started_at=self.now-19,finished_at=self.now-10)
  self.write(self.roles/(self.label+'.json'),dict(job_id=self.jid,job_sha256=sha(self.original)));self.write(self.roles/(self.jid+'-job.json'),self.sign(self.original));self.write(self.roles/(self.jid+'-failure.json'),self.failure)
  self.p=dict(version=VERSION,epoch=self.m['epoch'],checkpoint='a'*64,source_sha256='c'*64,original_label=self.label,replacement_label=self.label+'-recovery-1',original_job_id=self.jid,original_job_sha256=sha(self.original),original_failure_sha256=sha(self.failure),remote_path='/owned/exact-export',created_at=self.now-5,expires_at=self.now+3600,computation_sha256=sha(computation(self.m)),transport_path='/owned/CPU-upload.py',transport_operator_sha256='d'*64);self.policy=self.state/'scope.json'
 def sign(self,x):return dict(payload=x,signer=self.authority,signature=base64.b64encode(self.k.sign(canonical(x)).signature).decode())
 def write(self,p,x):p.write_bytes(canonical(x))
 def enable(self):self.write(self.policy,self.sign(self.p));self.c.checkpoint_upload_recovery_files={self.m['epoch']:str(self.policy)}
 def select(self):return select_label(self.c,self.m,'/owned/exact-export',self.label)
 def test_absent_preserves_original_label(self):self.assertEqual(self.select(),self.label)
 def test_terminal_upload_gets_distinct_stable_label_preserving_failure(self):
  before=(self.roles/(self.jid+'-failure.json')).read_bytes();self.enable();self.assertEqual(self.select(),self.p['replacement_label']);self.assertEqual(self.select(),self.p['replacement_label']);self.assertEqual(before,(self.roles/(self.jid+'-failure.json')).read_bytes());self.assertEqual(len(list(self.roles.iterdir())),3)
 def test_changed_scope_unsigned_and_wrong_key_refused(self):
  for key in ('checkpoint','source_sha256','remote_path','original_label','original_job_sha256','original_failure_sha256'):
   with self.subTest(key=key):
    old=self.p[key];self.p[key]='foreign';self.enable()
    with self.assertRaises(ValueError):self.select()
    self.p[key]=old
  self.enable();x=json.loads(self.policy.read_bytes());x['signature']='AAAA';self.write(self.policy,x)
  with self.assertRaises(Exception):self.select()
 def test_failure_is_not_running_zero_exit_or_completed_report(self):
  for change in (dict(phase='running'),dict(exit_code=0)):
   bad=dict(self.failure,**change);self.write(self.roles/(self.jid+'-failure.json'),bad);self.p['original_failure_sha256']=sha(bad);self.enable()
   with self.assertRaises(ValueError):self.select()
  self.write(self.roles/(self.jid+'-failure.json'),self.failure);self.p['original_failure_sha256']=sha(self.failure);self.enable();self.write(self.roles/(self.jid+'-report.json'),{'success':True})
  with self.assertRaises(ValueError):self.select()
 def test_training_cannot_be_reissued_by_upload_authorization(self):
  self.original['role']='train';self.write(self.roles/(self.jid+'-job.json'),self.sign(self.original));self.write(self.roles/(self.label+'.json'),dict(job_id=self.jid,job_sha256=sha(self.original)));self.p['original_job_sha256']=sha(self.original);self.enable()
  with self.assertRaisesRegex(ValueError,'upload request'):self.select()
 def test_expiry_blocks_new_dispatch_but_preserves_existing_original_recovery(self):
  self.enable()
  with patch('subnet.checkpoint_upload_recovery.time.time',return_value=self.p['expires_at']+1):
   with self.assertRaisesRegex(ValueError,'outside authorization'):self.select()
  new=dict(role='upload',job_id=self.p['replacement_label']+'-11111111',created_at=self.now,manifest=self.sign(self.m));self.write(self.roles/(self.p['replacement_label']+'.json'),dict(job_id=new['job_id'],job_sha256=sha(new)));self.write(self.roles/(new['job_id']+'-job.json'),self.sign(new))
  with patch('subnet.checkpoint_upload_recovery.time.time',return_value=self.p['expires_at']+1):self.assertEqual(self.select(),self.p['replacement_label'])
  new['role']='train';self.write(self.roles/(new['job_id']+'-job.json'),self.sign(new))
  with self.assertRaises(ValueError):self.select()
 def test_math_and_unknown_context_drift_refuses(self):
  self.enable()
  for key,value in [('numerical_policy',{'logprob_atol':1}),('unreviewed_semantic_field',True),('K',2)]:
   self.m[key]=value
   with self.assertRaisesRegex(ValueError,'computation'):self.select()
   self.m.pop(key)
 def test_only_read_capability_refresh_is_permitted(self):
  self.enable();self.m['checkpoint']['read_urls']={'model.safetensors':'freshGET'};self.assertEqual(self.select(),self.p['replacement_label'])
 def test_original_filemap_change_refused_even_with_recovery_authorization(self):
  self.enable();self.m['checkpoint']['files']={'other':'d'*64}
  with self.assertRaises(ValueError):self.select()
if __name__=='__main__':unittest.main()

class CPUAdapterTransportTests(unittest.TestCase):
 def exercise(self,failures=0,existing=False,permanent=False,mutate=False):
  import hashlib,os,requests,runpy,sys,types
  from subnet.checkpoint_upload_recovery import transport_program,computation
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name);source=root/'source';(source/'subnet').mkdir(parents=True);(source/'subnet/backend_jobs.py').write_text('# CPU transport fixture only\n');model=root/'model';model.mkdir();member=model/'model.safetensors';member.write_bytes(b'abc');key=SigningKey.generate();authority=key.verify_key.encode().hex();manifest=dict(epoch='CPU-test',checkpoint=dict(id='a'*64,files={'model.safetensors':hashlib.sha256(b'abc').hexdigest()}),source_bundle={'sha256':'c'*64});declfile=root/'declaration.json';program=transport_program(sys.executable,str(source),str(declfile));operator=root/'adapter.py';operator.write_text(program)
  now=time.time();decl=dict(version=VERSION,replacement_label='test-recovery',computation_sha256=sha(computation(manifest)),source_sha256='c'*64,checkpoint='a'*64,remote_path=str(model),transport_path=str(operator),transport_operator_sha256=hashlib.sha256(program.encode()).hexdigest(),created_at=now-1,expires_at=now+30)
  sign=lambda x:dict(payload=x,signer=authority,signature=base64.b64encode(key.sign(canonical(x)).signature).decode())
  job=dict(role='upload',job_id='test-recovery-12345678',created_at=now,expires_at=now+30,manifest=sign(manifest),source_files={'subnet/backend_jobs.py':hashlib.sha256((source/'subnet/backend_jobs.py').read_bytes()).hexdigest()},put_urls={'model.safetensors':'https://example.invalid/model?X-Amz-Signature=SECRET'})
  declfile.write_bytes(canonical(sign(decl)));Path(str(declfile)+'.GET.json').write_bytes(canonical(sign(dict(declaration_sha256=sha(decl),checkpoint='a'*64,read_urls={'model.safetensors':'https://example.invalid/model?X-Amz-Signature=GETSECRET'}))));jobfile=root/'job.json';jobfile.write_bytes(canonical(sign(job)));calls=[]
  class Response:
   headers={}
   def __init__(self,status):self.status_code=status
   def __enter__(self):return self
   def __exit__(self,*a):pass
   def iter_content(self,n):yield b'abc'
  def fake_put(url,**kw):
   calls.append(kw['data'].read());
   if mutate:member.write_bytes(b'bad')
   if permanent:return Response(403)
   if len(calls)<=failures:raise requests.exceptions.SSLError('SSLEOFError EOF with '+url)
   return Response(200)
  def backend(*a,**kw):
   with member.open('rb')as f:requests.put(job['put_urls']['model.safetensors'],data=f)
  namespace={'__name__':'CPU_transport_fixture','__file__':str(operator)}
  with patch.object(sys,'argv',[str(operator),'--backend',str(jobfile),'--authority',authority,'--workspace',str(root),'--checkpoint-cache',str(model)]),patch('requests.get',return_value=Response(200 if existing else 404)),patch('requests.put',side_effect=fake_put),patch('runpy.run_module',side_effect=backend),patch('time.sleep'):
   exec(compile(program,str(operator),'exec'),namespace)
  log=(root/(job['job_id']+'-transport.jsonl')).read_text();self.assertNotIn('SECRET',log);self.assertNotIn('X-Amz',log);return calls,log
 def test_SSL_EOF_retries_same_FD_fullbytes_and_redacts_URL(self):
  calls,log=self.exercise(failures=1);self.assertEqual(calls,[b'abc',b'abc']);self.assertIn('SSLError',log)
 def test_fullGETSHA_matching_existing_skips_PUT(self):
  calls,log=self.exercise(existing=True);self.assertFalse(calls);self.assertIn('skipped_PUT',log)
 def test_403_permanent_and_mutated_FD_refuse(self):
  for options in [dict(permanent=True),dict(mutate=True)]:
   with self.assertRaises((ValueError,RuntimeError)):self.exercise(**options)
 def test_exhausted_SSL_retry_is_bounded(self):
  with self.assertRaises(RuntimeError):self.exercise(failures=5)

class LaunchProfileTests(UploadRecoveryTests):
 def test_detached_launch_preserves_original_CUDA_env_and_cache_argv(self):
  import os,shlex,sys,hashlib
  from subnet.checkpoint_upload_recovery import run_recovery_upload,transport_program
  self.p['transport_path']=str(self.state/'remote/adapter.py');remote_decl=str(self.state/'remote/declaration.ROOT-SIGNED.json');self.p['transport_operator_sha256']=hashlib.sha256(transport_program(sys.executable,'/sealed/source',remote_decl).encode()).hexdigest();self.enable()
  captured=[]
  def command(value,timeout=None):
   parts=shlex.split(value)
   if '-c'in parts:captured.append(parts[parts.index('-c')+1])
   return '{}'
  remote=SimpleNamespace(python=sys.executable,code='/sealed/source',workspace=str(self.state/'workspace'),launch_runner=lambda *a:None,command=command,copy_to=lambda *a:None)
  Path(remote.workspace).mkdir()
  def run(label,role,manifest,cache,**kw):
   remote.launch_runner(label+'-12345678','/remote/job.json',cache);return {'success':True}
  remote.run=run;self.c.jobs=SimpleNamespace(owners={},initial_role='train',roles={'train':remote});self.c.bucket=SimpleNamespace(presign=lambda *a,**k:'CPU-test-only');self.c.signed=self.sign
  self.assertEqual(run_recovery_upload(self.c,self.p['replacement_label'],self.m,self.p['remote_path'],{'model.safetensors':'CPU-test-only'}),{'success':True})
  seen=[]
  class Child:
   pid=os.getpid()
  with patch('subprocess.Popen',side_effect=lambda argv,**kw:(seen.append((argv,kw))or Child())):
   exec(compile(captured[0],'actual detached launcher','exec'),{})
  argv,kw=seen[0];self.assertEqual(kw['env']['CUBLAS_WORKSPACE_CONFIG'],':4096:8');self.assertEqual(argv[argv.index('--checkpoint-cache')+1],self.p['remote_path']);self.assertTrue(kw['start_new_session'])
