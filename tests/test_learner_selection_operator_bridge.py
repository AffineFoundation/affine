"""Actual frozen-dependency metadata peer tests; synthetic signed docs, no models."""
import copy,hashlib,importlib,json,pathlib,shutil,sys,tempfile,types,unittest
from nacl.signing import SigningKey
import os
REPO=pathlib.Path(__file__).resolve().parents[1]
PROFILE=REPO/'operator_profiles/confirmed_blacklist_cpu_selection_v1'
ENTRY=REPO/'ops/learner_selection_cpu_peer.py';RUNNER=REPO/'ops/learner_selection_cpu_peer_runner.py'
BASE=pathlib.Path(os.environ.get('AFFINE_CPU_PEER_FROZEN_SOURCE',str(REPO))).resolve()
sys.path[:0]=[str(REPO/'tests'),str(BASE)]
from test_committed_training_inputs import LearnerAdmissionTests
from training_receipt_fixtures import sign
if 'AFFINE_CPU_PEER_FROZEN_SOURCE'in os.environ:
 MAP=json.loads((REPO/'tests/fixtures/cpu_selection_peer/f213-runtime177.json').read_bytes());SCIENCE='f21373d7ccb167bcd868f5ed03ce9ac7ef567d894e8ecc9e61f5d7f8645b67b8'
else:
 # Pure fixture inventory; not a scientific approval of this checkout.
 names=sorted(p for p in (BASE/'subnet').glob('*.py')if p.name not in ('learner_blacklist_selection.py','learner_selection_operator_bridge.py'))[:177]
 MAP={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest()for p in names};SCIENCE='f'*64
assert len(MAP)==177
class PeerTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=pathlib.Path(self.tmp.name)
  files=['learner_selection_operator_bridge.py','learner_blacklist_selection.py','committed_training_inputs.py','training_receipts.py'];self.operator=self.root/'operator';(self.operator/'subnet').mkdir(parents=True)
  for f in files:shutil.copyfile(PROFILE/'subnet'/f,self.operator/'subnet'/f)
  package='_peer_'+self.root.name.replace('-','_');mod=types.ModuleType(package);mod.__path__=[str(self.operator/'subnet'),str(BASE/'subnet')];sys.modules[package]=mod
  self.B=importlib.import_module(package+'.learner_selection_operator_bridge');self.C=importlib.import_module(package+'.committed_training_inputs');self.F=importlib.import_module(package+'.learner_blacklist_selection')
  fx=LearnerAdmissionTests();fx.setUp();self.addCleanup(fx.tmp.cleanup);fx.manifest['source_bundle']['sha256']=SCIENCE;fx.build();self.fx=fx;self.key=fx.operator;self.auth=fx.authority
  m=copy.deepcopy(fx.manifest);m['learner_blacklist_selection_round']=39
  policy=dict(version='continuous-probabilistic-audit-v3',recent_epochs=8,decay=.8,prior_alpha=1,prior_beta=1,invalid_multiplier=.1,zero_epoch_after=2,blacklist_after=3,blacklist_epochs=4)
  a=dict(version='hourly-current-miner-assessment-v1',cutoff=0,evidence_cutoff=0,assessment_stale=False,writer_policy_sha256='f'*64,miner_estimates={})
  m[self.F.FIELD]=sign(self.key,dict(version=self.F.VERSION,checkpoint=m['checkpoint']['id'],source_sha256=SCIENCE,target_round=39,maximum_age_seconds=3600,assessment_document=sign(self.key,a),writer_policy_sha256='f'*64,audit_policy=policy))
  m=self.C.coverage_manifest(m,[fx.obj],seed='c'*64,captured_at=21);m['learner_blacklist_selection_snapshot']=self.F.admit(m[self.F.FIELD],m,self.auth,at=21,round_number=39);self.m=m
  self.files={str(p.relative_to(self.operator)):hashlib.sha256(p.read_bytes()).hexdigest()for p in self.operator.rglob('*')if p.is_file()}
  self.approval=sign(self.key,dict(version=self.B.AUTH_VERSION,source_sha256=SCIENCE,scientific_source_files=MAP,operator_files=self.files,minimum_round=39,epoch_prefix=m['epoch'],peer_entry_sha256=hashlib.sha256((ENTRY).read_bytes()).hexdigest(),peer_runner_sha256=hashlib.sha256((RUNNER).read_bytes()).hexdigest(),backend_execution_allowed=False))
  self.job=dict(role='train',job_id='CPU-only-test',training_policy=m['training_policy'],training_input_policy=self.C.VERSION,source_files=MAP,manifest=sign(self.key,m),submissions=[fx.obj])
  self.admitted=self.B.make_admission(self.job,m,self.approval,self.auth,lambda v:sign(self.key,v))
 def peer(self,job=None):return self.B.admit_peer(sign(self.key,job or self.admitted),self.auth,operator_root=self.operator,scientific_root=BASE)
 def test_exact_frozen177_and_separate_enriched_CPU_overrides_pass(self):
  r=self.peer();self.assertEqual(r['operator_files'],self.files);self.assertEqual(self.admitted['source_files'],MAP);self.assertEqual(len(MAP),177);self.assertNotIn('subnet/learner_blacklist_selection.py',MAP);self.assertFalse(r['scientific_operation_started']);self.assertTrue(r['scientific_source_unchanged'])
 def test_admission_and_original_job_retry_are_byte_identical(self):
  self.assertEqual(self.peer(),self.peer())
  with self.assertRaisesRegex(ValueError,'reissued'):self.B.make_admission(self.admitted,self.m,self.approval,self.auth,lambda v:sign(self.key,v))
 def test_missing_admission_and_old_projection_digest_refuse(self):
  with self.assertRaisesRegex(ValueError,'required'):self.peer(self.job)
  j=copy.deepcopy(self.admitted);a=j[self.B.FIELD_ADMISSION]['payload'];a['computation_binding_sha256']='0'*64;j[self.B.FIELD_ADMISSION]=sign(self.key,a)
  with self.assertRaisesRegex(ValueError,'enriched'):self.peer(j)
 def test_round_parent_inventory_jobID_source_tamper_refuse(self):
  for name in ('round','parent','inventory','job','source'):
   j=copy.deepcopy(self.admitted)
   if name=='round':m=copy.deepcopy(self.m);m['learner_blacklist_selection_round']=40;j['manifest']=sign(self.key,m)
   if name=='parent':m=copy.deepcopy(self.m);m['checkpoint']['id']='9'*64;j['manifest']=sign(self.key,m)
   if name=='inventory':j['submissions']=[]
   if name=='job':j['job_id']='different-original'
   if name=='source':j['source_files']=dict(MAP,**{'subnet/model.py':'0'*64})
   with self.subTest(name=name),self.assertRaises(ValueError):self.peer(j)
 def test_remote_helper_laundering_refuses(self):
  j=copy.deepcopy(self.admitted);j['source_files']=dict(MAP,**{'subnet/learner_blacklist_selection.py':self.files['subnet/learner_blacklist_selection.py']})
  with self.assertRaisesRegex(ValueError,'remote177'):self.peer(j)
 def test_module_mutation_symlink_or_unlisted_file_refuse(self):
  p=self.operator/'subnet/learner_blacklist_selection.py';original=p.read_bytes();p.write_bytes(original+b'\n')
  with self.assertRaisesRegex(ValueError,'SHA'):self.peer()
  p.write_bytes(original);p.rename(p.with_suffix('.saved'));p.symlink_to(p.with_suffix('.saved'))
  with self.assertRaisesRegex(ValueError,'symlink'):self.peer()
 def test_actual_entry_authenticates_before_backend_import(self):
  spec=importlib.util.spec_from_file_location('_entry_cpu_test',ENTRY);entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
  package='_authenticated_CPU_selection_peer'
  for k in list(sys.modules):
   if k==package or k.startswith(package+'.'):sys.modules.pop(k)
  receipt,grant,_=entry.prepare(sign(self.key,self.admitted),self.auth,self.operator,BASE)
  self.assertEqual(receipt['original_job_sha256'],self.B.sha(self.admitted));self.assertFalse(grant['backend_execution_allowed'])
  self.assertEqual(receipt['CPU_override_symbols'],['subnet.backend_jobs.FreshSourceFinder'])
 def test_actual_coordinator_dispatch_builder_uses_separate_peer_not_old_runner(self):
  import ast,shlex,subprocess
  auth=copy.deepcopy(self.approval['payload']);auth['backend_execution_allowed']=True;authorization=sign(self.key,auth)
  admitted=self.B.make_admission(self.job,self.m,authorization,self.auth,lambda v:sign(self.key,v));(self.root/'CPU-only-test-job.json').write_text(json.dumps(sign(self.key,admitted)))
  tree=ast.parse((REPO/'subnet/remote_backend.py').read_bytes());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='RemoteJobs');method=next(x for x in cls.body if isinstance(x,ast.FunctionDef)and x.name=='launch_runner')
  module=ast.Module(body=[method],type_ignores=[]);env={'__package__':self.B.__package__,'json':json,'shlex':shlex,'subprocess':subprocess,'RemoteJobTerminalError':RuntimeError};exec(compile(ast.fix_missing_locations(module),'actual_remote_backend.launch_runner','exec'),env)
  calls=[];controller=types.SimpleNamespace(authority=types.SimpleNamespace(id=self.auth));peer=dict(operator_root='/owned/operator',entry='/owned/entry.py',runner='/owned/runner.py',bytecode_prefix_root='/owned/fresh-bytecode',authorization_document=authorization)
  instance=types.SimpleNamespace(python='/owned/python',controller=controller,workspace='/owned/workspace',code='/unchanged/source',state=self.root,config={'learner_selection_cpu_peer':peer},command=lambda c,timeout:calls.append(c))
  env['launch_runner'](instance,'CPU-only-test','/owned/workspace/CPU-only-test.json','/owned/model')
  generated=ast.parse(shlex.split(calls[0])[-1]);popen=next(n for n in ast.walk(generated)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='Popen');argv=ast.literal_eval(popen.args[0]);self.assertEqual(argv[:5],['/owned/python','-I','-B','-X','pycache_prefix=/owned/fresh-bytecode/CPU-only-test']);self.assertEqual(argv[5],peer['runner']);self.assertNotIn('subnet.remote_runner',argv);self.assertIn('--scientific-root',argv);self.assertIn('/unchanged/source',argv)
 def test_persistent_transport_must_be_final_before_admission(self):
  bad=copy.deepcopy(self.admitted);bad['persistent_training']={'new_after_admission':True}
  with self.assertRaisesRegex(ValueError,'enriched'):self.peer(bad)
 def test_real_runner_review_and_defaultoff_refusal_are_pre_model(self):
  import subprocess
  job=self.root/'job.json';job.write_text(json.dumps(sign(self.key,self.admitted)));workspace=self.root/'workspace';workspace.mkdir()
  def command(apply=False):
   args=[sys.executable,'-I','-B','-X','pycache_prefix='+str(self.root/'fresh-bytecode'),str(RUNNER),'--job',str(job),'--authority',self.auth,'--operator-root',str(self.operator),'--scientific-root',str(BASE),'--entry',str(ENTRY),'--workspace',str(workspace),'--fresh-bytecode-prefix',str(self.root/'fresh-bytecode')]
   if apply:args+=['--apply']
   return subprocess.run(args,capture_output=True,text=True,timeout=60)
  review=command();self.assertEqual(review.returncode,0,review.stderr);self.assertTrue(json.loads(review.stdout)['review_only']);self.assertFalse(list(workspace.iterdir()))
  refused=command(True);self.assertNotEqual(refused.returncode,0);self.assertIn('ROOT backend execution grant required',refused.stderr);self.assertFalse(list(workspace.iterdir()))
 def test_runner_SHA_grant_mutation_refuses(self):
  auth=copy.deepcopy(self.approval['payload']);auth['peer_runner_sha256']='0'*64
  badapproval=sign(self.key,auth);j=self.B.make_admission(self.job,self.m,badapproval,self.auth,lambda v:sign(self.key,v))
  self.assertNotEqual(auth['peer_runner_sha256'],hashlib.sha256((RUNNER).read_bytes()).hexdigest())
  # The real subprocess below exercises the signed hash check, not an AST assertion.
  import subprocess
  path=self.root/'bad-runner-job.json';path.write_text(json.dumps(sign(self.key,j)))
  r=subprocess.run([sys.executable,'-I','-B','-X','pycache_prefix='+str(self.root/'prefix'),str(RUNNER),'--job',str(path),'--authority',self.auth,'--operator-root',str(self.operator),'--scientific-root',str(BASE),'--entry',str(ENTRY),'--workspace',str(self.root),'--fresh-bytecode-prefix',str(self.root/'prefix')],capture_output=True,text=True,timeout=60)
  self.assertNotEqual(r.returncode,0);self.assertIn('actual peer runner SHA',r.stderr)
 def test_actual_frozen_loader_reload_keeps_separately_admitted_CPU_parser(self):
  spec=importlib.util.spec_from_file_location('_entry_reload_test',ENTRY);entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
  saved={k:v for k,v in list(sys.modules.items())if k=='subnet'or k.startswith('subnet.')}
  for k in saved:sys.modules.pop(k)
  try:
   backend=entry.bind_backend(self.approval['payload'],self.operator,BASE)
   C=importlib.import_module('subnet.committed_training_inputs');C.validate_job(self.admitted,self.m,self.auth)
   backend.install_source_loader(BASE,('subnet/training_receipts.py','subnet/committed_training_inputs.py'))
   C=importlib.import_module('subnet.committed_training_inputs');self.assertEqual(pathlib.Path(C.__file__),self.operator/'subnet/committed_training_inputs.py')
   C.validate_job(self.admitted,self.m,self.auth)
   bad=copy.deepcopy(self.admitted);bad['job_id']='changed-retry'
   with self.assertRaises(ValueError):C.validate_job(bad,self.m,self.auth)
  finally:
   for k in list(sys.modules):
    if k=='subnet'or k.startswith('subnet.'):sys.modules.pop(k)
   sys.modules.update(saved)
 def test_policy_absent_original_metadata_does_not_read_peer_dependencies(self):
  job=dict(self.job,manifest=sign(self.key,self.fx.manifest))
  self.assertIsNone(self.B.admit_peer(sign(self.key,job),self.auth,operator_root='/DO-NOT-ACCESS',scientific_root='/DO-NOT-ACCESS'))
 def test_signature_tamper_refuses_before_dependency_read(self):
  doc=sign(self.key,self.admitted);doc['payload']['job_id']='unsigned-tamper'
  with self.assertRaises(ValueError):self.B.admit_peer(doc,self.auth,operator_root='/DO-NOT-ACCESS',scientific_root='/DO-NOT-ACCESS')
if __name__=='__main__':unittest.main()
