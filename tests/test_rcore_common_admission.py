"""Portable CPU metadata/DTO controls. Native/model execution is not simulated as evidence."""
import base64,copy,hashlib,io,json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet.environments import EnvironmentSpec,_source_hash,create_session
from subnet.native_rcore_common import REVISION,VERSION,BINDINGS,CommonRCoreSession,validate_binding,admit_descriptor
from subnet.native_rcore_boundary import PublicActor,canonical,digest
from subnet.native_rcore_role_transport import PINNED,admit_role_binding,role_bindings,execute_role
from subnet.backend_jobs import SOURCE_FILES,file_map,REVISION as GPU_REVISION,NUMERICAL_POLICY,BACKEND_PROFILE
ROOT=Path(__import__('subnet.native_rcore_common',fromlist=['x']).__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def signed(payload,key):return dict(payload=copy.deepcopy(payload),signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())

class PortableRCore(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='affine-rcore-public-test-');self.addCleanup(self.tmp.cleanup);self.stage=Path(self.tmp.name);self.pkg=self.stage/'package';(self.pkg/'operator').mkdir(parents=True)
  self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
  # Explicit mock resources, not the actual native package or a remote qualification.
  (self.stage/'preimport-terminal-worker.py').write_text('# public mock guard; never executed\n')
  (self.pkg/'worker.py').write_text('# public mock worker; never executed\n')
  native=dict(id='affine_rcore',version='prime-v1-1',source_hash='a'*64,max_turns=1)
  (self.pkg/'operator/environment.json').write_bytes(canonical(native));(self.pkg/'operator/affine_rcore.tasks.json').write_bytes(canonical({'public_fixture':'not original taskset'}))
  self.task_hashes=[hashlib.sha256(('public-fixture-'+str(i)).encode()).hexdigest()for i in range(64)]
  self.profile=dict(revision='original-rcore-trusted-terminal-resources-v1',role='trusted-terminal-grader',CPU_only=True,remote=str(self.pkg),model_execution=False,optimizer_ran=False,chain_transactions=False,provider_namespace_controlled=True,full_transitive_closure_claimed=False,environment_version='prime-v1-1',environment_source_hash='a'*64,guard_sha256=sha(self.stage/'preimport-terminal-worker.py'),source_bundle_sha256='b'*64,provider_resource_id='c'*64)
  self.seal_profile()
  binding=dict(authority=self.authority,profile_sha256=sha(self.stage/'signed-resource-profile.json'),guard_sha256=self.profile['guard_sha256'],worker_sha256=sha(self.pkg/'worker.py'),original_environment_sha256=sha(self.pkg/'operator/environment.json'),original_environment_source_hash='a'*64,snapshot_sha256=sha(self.pkg/'operator/affine_rcore.tasks.json'),source_bundle_sha256='b'*64,provider_resource_id='c'*64,task_hashes=self.task_hashes)
  self.spec=EnvironmentSpec(id='affine_rcore',version=VERSION,config=dict(rcore_terminal_revision=REVISION,terminal_public_binding=binding,seed=100),max_turns=1,max_output_tokens=512,num_samples=64,success_reward=1.,source_hash='preparation');self.spec=self.rehash(self.spec)
  self.descriptor=dict(revision='rcore-terminal-role-local-v1',audience='verifier',operator_role='trusted-terminal-grader',stage=str(self.stage))
  self.public=dict(revision='original-rcore-public-text-trusted-terminal-v1',environment_version='prime-v1-1',task_id=self.task_hashes[0],messages=[{'role':'user','content':'Evaluate 1 + 1'}],tools=[],source_bundle_sha256='b'*64,provider_resource_id='c'*64,public_resources=[])
 def seal_profile(self):
  (self.stage/'signed-resource-profile.json').write_bytes(canonical(signed(self.profile,self.key)))
 def rehash(self,spec):return EnvironmentSpec.from_dict(dict(spec.to_dict(),source_hash=_source_hash(spec)))
 def changed_spec(self,**changes):return EnvironmentSpec(**dict(self.spec.to_dict(),**changes))
 def terminal(self,text='2',index=0,seed=100,public=None):
  public=public or self.public;native=dict(observations=[],done=True,reward=1.,classification='positive')
  return dict(done=True,reward=1.,classification='positive',task_id=public['task_id'],public_descriptor_sha256=digest(public),trace_sha256=digest(dict(public_descriptor_sha256=digest(public),index=index,seed=seed,action={'text':text},native_terminal=native)))
 def job(self,role='verify',descriptor=None):
  descriptor=descriptor or self.descriptor;files={'config.json':'1'*64,'model.safetensors':'2'*64}
  manifest=dict(epoch='nonpayable-portable-metadata-test',checkpoint={'id':file_map(files),'files':files},model_runtime_revision=GPU_REVISION,numerical_policy=NUMERICAL_POLICY,backend_profile=BACKEND_PROFILE,K=1,L=1,audit_policy={'mode':'full'},environment=self.spec.to_dict(),indices=[0],harness={'version':'text-tools-v1','policy':'autoregressive','max_output_tokens':512},heldout_indices={'affine_rcore':list(range(32,64))},start=10,deadline=100)
  # Pin real public source bytes and installed versions; no mocked GPU execution.
  pins={name:sha(ROOT/name)for name in set(SOURCE_FILES)|set(PINNED)}
  from importlib.metadata import version
  versions={name:version(name)for name in ['torch','transformers','toploc']}
  job=dict(schema=1,job_id='portable-metadata-test',role=role,created_at=10,expires_at=100,manifest=signed(manifest,self.key),source_files=pins,runtime_versions=versions,submissions=[dict(url='https://account.r2.cloudflarestorage.com/test?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=public-control',sha256='d'*64)],terminal_resource_binding=dict(revision='rcore-terminal-job-binding-v1',environment_sha256=digest(self.spec.to_dict()),profile_sha256=self.spec.config['terminal_public_binding']['profile_sha256'],guard_sha256=self.spec.config['terminal_public_binding']['guard_sha256'],role_local_descriptor_sha256=digest(descriptor)))
  if role=='mine':job.update(miner_id='e'*64,search_budget=16,seed_start=0,capability={'put_url':job['submissions'][0]['url'],'headers':{'Content-Type':'application/octet-stream'}})
  if role=='train':job['steps']=1
  if role=='evaluate':job['heldout']=[dict(env_id='affine_rcore',indices=[32],seeds=[202],harness={'policy':'autoregressive','max_output_tokens':512})]
  return job
 def test_exact_ephemeral_profile_admission_and_private_dto_exclusion(self):
  self.assertEqual(admit_descriptor(self.spec,self.descriptor)[0],self.stage)
  actor=PublicActor(self.public,lambda text:self.terminal(text));self.assertNotIn(str(self.stage),canonical(actor.reset()).decode());self.assertEqual(actor.finish('2')['reward'],1.)
  for name,value in [('public_resources',[{'audience':'verifier','path':'private-grader'}]),('grader_descriptor',self.descriptor),('cache_mount',str(self.pkg))]:
   public=copy.deepcopy(self.public);public[name]=value
   with self.assertRaises(ValueError):PublicActor(public,lambda _:None)
 def test_invalid_adapter_version_seed_and_private_spec_refused_before_dispatch(self):
  for adapter in ['legacy_mastermind','resource_prime_v1','resource_prime_v1_controlled']:
   with patch('subnet.resource_session.create_resource_session')as alternate:
    with self.assertRaises(ValueError):create_session(self.changed_spec(adapter=adapter))
    alternate.assert_not_called()
  with self.assertRaises(ValueError):validate_binding(self.changed_spec(version='wrong'))
  for seed in [True,-1,1.5,'100']:
   config=copy.deepcopy(self.spec.config);config['seed']=seed
   with self.assertRaises(ValueError):validate_binding(self.changed_spec(config=config))
  config=copy.deepcopy(self.spec.config);config['private_grader']=self.descriptor
  with self.assertRaises(ValueError):validate_binding(self.changed_spec(config=config))
 def test_missing_descriptor_and_wrong_audience(self):
  with patch.dict(os.environ,{},clear=True):
   with self.assertRaises(ValueError):create_session(self.spec)
  for field,value in [('audience','operator'),('operator_role','miner'),('revision','wrong')]:
   with self.assertRaises(ValueError):admit_descriptor(self.spec,dict(self.descriptor,**{field:value}))
 def test_forged_profile_and_source_resource_changes_refused(self):
  doc=signed(self.profile,self.key);doc['payload']['environment_source_hash']='f'*64;(self.stage/'signed-resource-profile.json').write_bytes(canonical(doc))
  config=copy.deepcopy(self.spec.config);config['terminal_public_binding']['profile_sha256']=sha(self.stage/'signed-resource-profile.json')
  with self.assertRaises(Exception):admit_descriptor(self.changed_spec(config=config),self.descriptor)
  self.seal_profile()
  for path in [self.stage/'preimport-terminal-worker.py',self.pkg/'worker.py',self.pkg/'operator/environment.json',self.pkg/'operator/affine_rcore.tasks.json']:
   previous=path.read_bytes();path.write_bytes(b'mutated')
   with self.assertRaises(ValueError):admit_descriptor(self.spec,self.descriptor)
   path.write_bytes(previous)
 def test_terminal_trace_crossindex_seed_action_and_spec_refused(self):
  obj=CommonRCoreSession.__new__(CommonRCoreSession);obj.actor=PublicActor(self.public,lambda _:None);obj.index=0;obj.seed=100;result=self.terminal();obj._call=lambda _:result
  self.assertEqual(obj._finish('2'),result)
  for index,seed,text in [(1,100,'2'),(0,101,'2'),(0,100,'wrong')]:
   obj.index=index;obj.seed=seed
   with self.assertRaises(ValueError):obj._finish(text)
  obj.index=0;obj.seed=100;public=dict(self.public,source_bundle_sha256='f'*64);obj.actor=PublicActor(public,lambda _:None)
  with self.assertRaises(ValueError):obj._finish('2')
 def test_native_dto_mock_reset_step_shapes_and_error_refusal(self):
  session=CommonRCoreSession(self.spec,self.descriptor);result=self.terminal();responses=[self.public,result]
  fake=SimpleNamespace(poll=lambda:None,stdin=io.StringIO(),stdout=io.StringIO(),pid=123,wait=lambda **kw:0)
  with patch('subnet.native_rcore_common.subprocess.Popen',return_value=fake),patch.object(session,'_call',side_effect=lambda request:responses.pop(0)):
   initial=session.reset(0,100);self.assertEqual(initial,dict(messages=self.public['messages'],tools=[],task_hash=self.public['task_id']))
   terminal=session.step({'text':'2'});self.assertEqual(terminal,dict(observations=[],done=True,reward=1.,classification='positive'));session.close()
  obj=CommonRCoreSession.__new__(CommonRCoreSession);obj.process=SimpleNamespace(poll=lambda:None,stdin=io.StringIO(),stdout=None)
  with patch('subnet.native_rcore_common.read_protocol_line',return_value='{"ok":false,"error_type":"RuntimeError"}'):
   with self.assertRaises(ValueError):obj._call({'op':'finish','text':'x'})
 def test_all_four_signed_roles_bind_exact_operator_audience(self):
  for role,audience in [('mine','miner'),('verify','verifier'),('train','trainer'),('evaluate','evaluator')]:
   descriptor=dict(self.descriptor,audience=audience);job=self.job(role,descriptor)
   admitted,_=admit_role_binding(signed(job,self.key),self.authority,self.spec,descriptor,now=50);self.assertEqual(admitted['role'],role)
 def test_fresh_launcher_cpu_admission_only_imports_no_runtime(self):
  job=self.job();job['created_at']=time.time()-10;job['expires_at']=time.time()+100
  jobfile=self.stage/'job.json';jobfile.write_bytes(canonical(signed(job,self.key)));descriptorfile=self.stage/'descriptor.json';descriptorfile.write_bytes(canonical(self.descriptor))
  command=[sys.executable,'-I','-B',str(ROOT/'ops/run_rcore_common_role.py'),'--job',str(jobfile),'--authority',self.authority,'--operator-descriptor',str(descriptorfile),'--workspace',str(self.stage/'never-used-model-workspace'),'--cpu-admission-only']
  result=subprocess.run(command,capture_output=True,text=True,timeout=30);self.assertEqual(result.returncode,0,result.stderr)
  report=json.loads(result.stdout);self.assertEqual(report,dict(admitted=True,model_execution=False,gpu_allocation=False,preloaded_runtime_modules=[]));self.assertFalse((self.stage/'never-used-model-workspace').exists())
 def test_concrete_executor_scope_and_refusal_before_invocation(self):
  job=self.job();envelope=signed(job,self.key);calls=[]
  def executor(*args,**kwargs):calls.append(json.loads(Path(os.environ[BINDINGS]).read_bytes()));raise RuntimeError('injected executor error, not model execution')
  with patch('subnet.backend_jobs.time.time',return_value=50),patch('subnet.backend_jobs.execute',side_effect=executor)as backend,patch.dict(os.environ,{BINDINGS:'prior-operator'}):
   with self.assertRaises(RuntimeError):execute_role(envelope,self.authority,self.spec,self.descriptor,'owned-test-workspace')
   self.assertEqual(calls,[self.descriptor]);self.assertEqual(os.environ[BINDINGS],'prior-operator')
   refused=[]
   for field in ['env','environment_overrides','resource_descriptor','terminal_binding_path']:
    change=copy.deepcopy(job);change[field]='miner-path';refused.append(signed(change,self.key))
   for field in ['profile_sha256','environment_sha256','role_local_descriptor_sha256']:
    change=copy.deepcopy(job);change['terminal_resource_binding'][field]='f'*64;refused.append(signed(change,self.key))
   change=copy.deepcopy(job);change['source_files'][PINNED[-1]]='f'*64;refused.append(signed(change,self.key))
   change=copy.deepcopy(job);manifest=copy.deepcopy(change['manifest']['payload']);manifest['heldout_indices']['affine_rcore']=[0];change['manifest']=signed(manifest,self.key);refused.append(signed(change,self.key))
   forged=copy.deepcopy(envelope);forged['payload']['role']='mine';refused.append(forged)
   for rejected in refused:
    with self.assertRaises(Exception):execute_role(rejected,self.authority,self.spec,self.descriptor,'owned-test-workspace')
   with self.assertRaises(ValueError):execute_role(envelope,self.authority,self.spec,dict(self.descriptor,audience='miner'),'owned-test-workspace')
   backend.assert_called_once();self.assertEqual(os.environ[BINDINGS],'prior-operator')
if __name__=='__main__':unittest.main()
