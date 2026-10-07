"""Actual N8 inventory/native admission; only fixture signing keys and temp docs."""
import copy,hashlib,importlib.util,json,pathlib,sys,tempfile,unittest,base64
from nacl.signing import SigningKey
import shutil
REPO=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'tests'),str(REPO)]
from ops import durable_learner_service as m,durable_audit_services as g
from test_durable_learner_service import LearnerRecovery
PROFILE=REPO/'operator_profiles/confirmed_blacklist_cpu_selection_v1'
class Admission(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=pathlib.Path(self.tmp.name);self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
  fixture=LearnerRecovery('test_restart_preserves_original_and_allows_epoch_advance');fixture.setUp();self.addCleanup(fixture.doCleanups);self.fixture=fixture
  self.policy=copy.deepcopy(fixture.p);source_root=fixture.fixture.runtime
  for old,new in [('file_0.py','committed_training_inputs.py'),('file_1.py','training_receipts.py')]:
   (source_root/'subnet'/old).rename(source_root/'subnet'/new)
  self.source=g.signed(g.read(self.policy['source_approval']['path']),fixture.fixture.auth);self.source['full_source_files']=self.source['runtime_source_files']={str(p.relative_to(source_root)):g.file_hash(p)for p in source_root.rglob('*.py')}
  self.overlay=self.root/'overlay';shutil.copytree(source_root,self.overlay)
  for path in (PROFILE/'subnet').iterdir():shutil.copyfile(path,self.overlay/'subnet'/path.name)
  overrides={str(p.relative_to(self.overlay)):g.file_hash(p)for p in (self.overlay/'subnet').iterdir()if self.source['full_source_files'].get(str(p.relative_to(self.overlay)))!=g.file_hash(p)}
  from test_learner_capture_overlay import CAPTURE
  self.policy['operator_overlay']=dict(version=m.OVERLAY_VERSION,root=str(self.overlay),full_source_files=dict(self.source['full_source_files'],**overrides),overrides=overrides,baseline_source_sha256=fixture.fixture.source,baseline_inventory_sha256=g.digest(self.source['full_source_files']),learner_capture_policy=CAPTURE)
  self.cfg=dict(source_bundle={'sha256':fixture.fixture.source},training_input_policy='committed-unaudited-training-v1',submission_transport_policy='small-commitment-pairs-v2',hourly_execution_policy={},learner_capture_policy=CAPTURE,remote={'roles':{'train':dict(host='retained-trainer.test',port=22,user='root',python='/owned/python',code='/owned/science'),'mine':{}}})
  self.peer_root=self.root/'peer';shutil.copytree(PROFILE,self.peer_root);peer_files={str(p.relative_to(self.peer_root)):g.file_hash(p)for p in self.peer_root.rglob('*.py')}
  self.grant=dict(version='cpu-selection-peer-authorization-v1',source_sha256=fixture.fixture.source,scientific_source_files=self.source['runtime_source_files'],operator_files=peer_files,minimum_round=40,epoch_prefix='prospective-',peer_entry_sha256=g.file_hash(REPO/'ops/learner_selection_cpu_peer.py'),peer_runner_sha256=g.file_hash(REPO/'ops/learner_selection_cpu_peer_runner.py'),backend_execution_allowed=True)
  auth=self.document('fixture-peer-authorization.json',self.grant)
  row=dict(version='durable-CPU-selection-peer-v1',authorization=auth,remote_admission={},peer_root=str(self.peer_root),peer_files=peer_files,entry=str(REPO/'ops/learner_selection_cpu_peer.py'),entry_sha256=self.grant['peer_entry_sha256'],runner=str(REPO/'ops/learner_selection_cpu_peer_runner.py'),runner_sha256=self.grant['peer_runner_sha256']);self.policy[m.PEER_POLICY_FIELD]=row
  role=self.cfg['remote']['roles']['train'];peer=dict(operator_root='/owned/operator',entry='/owned/entry.py',runner='/owned/runner.py',bytecode_prefix_root='/owned/fresh-bytecode',authorization_document=g.read(auth['path']));role['learner_selection_cpu_peer']=peer
  auto=dict(version='automatic-confirmed-blacklist-training-selection-v1',source_sha256=fixture.fixture.source,writer_policy_sha256='a'*64,audit_policy=dict(version='continuous-probabilistic-audit-v3',recent_epochs=8,decay=.8,prior_alpha=1,prior_beta=1,invalid_multiplier=.1,zero_epoch_after=2,blacklist_after=3,blacklist_epochs=4),maximum_age_seconds=600,assessment_path=str(self.root/'assessment.json'))
  self.cfg['learner_blacklist_selection_authorization']=g.read(self.document('fixture-automatic.json',auto)['path'])
  admission=dict(version='CPU-selection-remote-peer-admission-v1',source_sha256=self.policy['source_sha256'],scientific_source_files_sha256=g.digest(self.source['runtime_source_files']),authorization_payload_sha256=g.digest(self.grant),host=role['host'],port=role['port'],user=role['user'],python=role['python'],scientific_root=role['code'],operator_root=peer['operator_root'],entry=peer['entry'],runner=peer['runner'],bytecode_prefix_root=peer['bytecode_prefix_root'],operator_files=row['peer_files'],entry_sha256=row['entry_sha256'],runner_sha256=row['runner_sha256'],CPU_only=True,model_loaded=False,proof_reverification=False,admitted=True)
  row['remote_admission']=self.document('fixture-remote-admission.json',admission)
 def document(self,name,payload):
  p=self.root/name;d=dict(payload=payload,signer=self.authority,signature=base64.b64encode(self.key.sign(g.canonical(payload)).signature).decode());p.write_bytes(g.canonical(d));p.chmod(0o600);return dict(path=str(p),file_sha256=g.file_hash(p),payload_sha256=g.digest(payload))
 def peer(self):return m.validate_cpu_selection_peer(self.policy,self.source,self.cfg,self.authority)
 def test_exact_baseline_overlay177_and_separate_peer_pass(self):
  self.peer();m.validate_operator_overlay(self.policy['operator_overlay'],self.source,self.policy,self.cfg);self.assertEqual(len(self.source['runtime_source_files']),177)
 def test_absent_peer_cannot_expand_old_transport_allowlist(self):
  self.policy.pop(m.PEER_POLICY_FIELD)
  with self.assertRaisesRegex(ValueError,'transport'):m.validate_operator_overlay(self.policy['operator_overlay'],self.source,self.policy,self.cfg)
 def test_bad_null_and_extra_schema_refuse(self):
  for v in (None,{},dict(self.policy[m.PEER_POLICY_FIELD],unapproved=True)):
   with self.subTest(v=type(v).__name__):
    self.policy[m.PEER_POLICY_FIELD]=v
    with self.assertRaisesRegex(ValueError,'exact'):self.peer()
 def test_wrong_source177_and_new_helper_laundering_refuse(self):
  for key,value in [('source_sha256','0'*64),('scientific_source_files',dict(self.source['runtime_source_files'],**{'subnet/learner_blacklist_selection.py':'0'*64}))]:
   grant=copy.deepcopy(self.grant);grant[key]=value;row=self.document('bad-grant-'+key+'.json',grant);self.policy[m.PEER_POLICY_FIELD]['authorization']=row;self.cfg['remote']['roles']['train']['learner_selection_cpu_peer']['authorization_document']=g.read(row['path'])
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'source177'):self.peer()
 def test_remote_endpoint_runtime_path_receipt_and_true_model_claim_refuse(self):
  original=copy.deepcopy(self.cfg['remote']['roles']['train'])
  for key in ('host','port','python','code'):
   self.cfg['remote']['roles']['train']=copy.deepcopy(original);self.cfg['remote']['roles']['train'][key]=999 if key=='port'else'/changed'
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'readiness'):self.peer()
  self.cfg['remote']['roles']['train']=original
  row=self.policy[m.PEER_POLICY_FIELD];a=g.read(row['remote_admission']['path'])['payload'];a['model_loaded']=True;row['remote_admission']=self.document('false-model.json',a)
  with self.assertRaisesRegex(ValueError,'readiness'):self.peer()
 def test_coordinator_peer_parser_divergence_refuses(self):
  self.policy['operator_overlay']['overrides']['subnet/training_receipts.py']='0'*64
  with self.assertRaisesRegex(ValueError,'match'):self.peer()
 def test_cpu_execution_defaultoff_is_not_deployment_grant(self):
  self.grant['backend_execution_allowed']=False;row=self.document('defaultoff.json',self.grant);self.policy[m.PEER_POLICY_FIELD]['authorization']=row
  with self.assertRaisesRegex(ValueError,'execution grant'):self.peer()
 def test_signed_automatic_mechanism_source_age_and_scalar_parser_refuse(self):
  original=self.cfg['learner_blacklist_selection_authorization']['payload']
  for name,value in [('source_sha256','0'*64),('maximum_age_seconds',True),('audit_policy',dict(original['audit_policy'],blacklist_epochs=True))]:
   bad=copy.deepcopy(original);bad[name]=value;row=self.document('bad-automatic-'+name+'.json',bad);self.cfg['learner_blacklist_selection_authorization']=g.read(row['path'])
   with self.subTest(name=name),self.assertRaises(ValueError):self.peer()
 def test_remote_admission_signature_and_file_SHA_refuse(self):
  row=self.policy[m.PEER_POLICY_FIELD]['remote_admission'];original=g.read(row['path']);original['payload']['admitted']=False;path=pathlib.Path(row['path']);path.write_bytes(g.canonical(original));row['file_sha256']=g.file_hash(path)
  with self.assertRaises(Exception):self.peer()
 def test_real_full_policy_null_optin_refuses_before_runtime(self):
  from test_durable_learner_service import LearnerRecovery
  fixture=LearnerRecovery('test_restart_preserves_original_and_allows_epoch_advance');fixture.setUp()
  try:
   cfg=g.read(fixture.fixture.config);cfg['learner_blacklist_selection_authorization']=None;fixture.fixture.config.write_text(json.dumps(cfg));p=copy.deepcopy(fixture.p);p['config']['file_sha256']=g.file_hash(fixture.fixture.config);p['runner_file_sha256']=g.file_hash(pathlib.Path(m.__file__))
   with self.assertRaisesRegex(ValueError,'separately admitted'):m.validate_policy(fixture.fixture.sign(p),fixture.fixture.auth)
  finally:fixture.doCleanups()
 def test_miner_override_refuses(self):
  self.cfg['remote']['roles']['mine']['learner_selection_cpu_peer']=copy.deepcopy(self.cfg['remote']['roles']['train']['learner_selection_cpu_peer'])
  with self.assertRaisesRegex(ValueError,'mining'):self.peer()
if __name__=='__main__':unittest.main()
