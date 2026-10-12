"""Synthetic signed CPU controls: never ROOT-signed permission or GPU evidence."""
from pathlib import Path
import copy,hashlib,importlib.util,json,sys,unittest
from unittest.mock import patch
S=Path(__file__).resolve().parents[1]
from subnet import unaudited_training_execution as u,optimizer_threading as threads
from subnet.training_receipts import sha
from test_unaudited_execution_release import ReleaseTests
from subnet.persistent_cpu_adamw import HYPERPARAMETERS

class OptimizerThreadReleaseTests(unittest.TestCase):
 def setUp(self):
  self.f=ReleaseTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
  self.sign=self.f.f.sign;self.auth=self.f.f.authority
  self.job=self.f.epoch(157,2,1);self.manifest=self.job['manifest']['payload'];m=self.manifest
  self.public=json.loads((self.f.state/(m['epoch']+'-first-signed-manifest.json')).read_text())
  nd=self.f.state/'native-outcome-eligibility'/m['epoch'];self.docs={k:json.loads((nd/(k+'.ROOT-SIGNED.json')).read_text())for k in ('context','grades','subset')}
  r=copy.deepcopy(self.f.release['payload']);r.update(version=u.OBJECTIVE_RELEASE,method=u.OBJECTIVE_METHOD)
  r['original_source_files']['subnet/backend_jobs.py']='a'*64
  before=r['execution_source_files'];before['subnet/backend_jobs.py']='d'*64
  before['subnet/persistent_cpu_adamw.py']=u.QUALIFIED_ADAM_SHA256
  before.update({'subnet/training_policy.py':'1'*64,'subnet/persistent_training_evidence.py':'2'*64})
  r['changed_source_files']={n:h for n,h in before.items()if r['original_source_files'].get(n)!=h}
  r['training_objective']={'version':'sequence-mean-preference-positive-nll-v1','positive_nll_weight':1.0}
  r['training_horizon']=dict(version='retained-adam-unit-positive-nll-16-v1',first_optimizer_step=2,updates=16,initial_checkpoint=m['checkpoint']['id'],initial_parent_descriptor_sha256=m['trainer_state_binding']['parent']['descriptor_sha256'],coefficient_during=1.,coefficient_after=0.)
  q=copy.deepcopy(r['execution_qualification']['payload']);q.update(version=u.OBJECTIVE_QUALIFICATION,method=u.OBJECTIVE_METHOD,execution_source_files_sha256=sha(before),objective_changed=True,training_objective=r['training_objective'],tested_positive_nll_weights=[0.,1.],retained_parent_descriptor_sha256='b'*64,retained_optimizer_step=1,retained_input_checkpoint='c'*64,qualification_scope='private-nonzero-adam-mechanical-continuation-v1',live_optimizer_distribution_equivalence_claimed=False,initial_execution_source_bundle_sha256='d'*64)
  r['execution_qualification']=self.sign(q);self.previous=self.sign(r);u.release(self.previous,self.auth)
  after=copy.deepcopy(before)
  for n in u.THREAD_FILES:after[n]=hashlib.sha256((S/n).read_bytes()).hexdigest()
  self.delta={n:h for n,h in after.items()if before.get(n)!=h}
  self.r=copy.deepcopy(r);self.r.update(version=u.THREAD_RELEASE,optimizer_threading=copy.deepcopy(threads.POLICY),training_source_bundle={'sha256':'6'*64},execution_source_files=after,changed_source_files={n:h for n,h in after.items()if r['original_source_files'].get(n)!=h})
  # Synthetic attestation shape, signed only by the test authority.
  # No production evidence, credentials or private filesystem is required.
  orig=self.sign(dict(version='ROOT-original-actual-state-CPU16-execution-v1',actual_full7B_state_comparison=True,
      all_master_moments_projection_counters_equal=True,original_replica_and_state_unchanged=True,
      production_runtime_qualified=False,production_changed=False,parameter_tensors=339,elements=7615616512,
      scope_sha256='3fb6cb125723fda95d3cd98d25efa6a670856e17c77c33992f9fe4009e1a8ef4',
      original_capture={'sha256':'1e38c6cbb35a25872fe9b1d46c4c41e57b90976d5700ae425648bc49fec9f6b3'}))
  self.e=dict(version='optimizer-only-cpu16-caller-source-equivalence-evidence-v1',previous_release_sha256=sha(self.previous),previous_source_files_sha256=sha(before),source_files_sha256=sha(after),changed_files=self.delta,optimizer_threading=copy.deepcopy(threads.POLICY),optimizer_module_sha256=u.QUALIFIED_ADAM_SHA256,original_execution_sha256=sha(orig),original_result_sha256='0f3a38a4804b703d9175b1a165a4523e0e09973b746961469118b04341c48d9e',all_other_source_files_byte_identical=True,new_numerical_execution=False,production_numerical_qualification_claimed=False,**{k:dict(sha256='f'*64,passed=True) for k in ('cpu_caller_controls','release_report_controls','worker_cpu_readback','original_full_GET_readback')})
  self.q=dict(version=u.THREAD_QUALIFICATION,optimizer_threading=copy.deepcopy(threads.POLICY),qualified_predecessor_release=self.previous,execution_source_bundle_sha256='6'*64,execution_source_files_sha256=sha(after),runtime_versions=r['runtime_versions'],training_runtime_sha256=r['training_runtime_sha256'],original_full_state_execution=orig,equivalence_evidence=self.sign(self.e))
  self.r['execution_qualification']=self.sign(self.q);self.job['source_files']=after
 def release(self):return self.sign(self.r)
 def grant(self):return self.sign(u.derive_preparation(self.release(),self.job,self.public,self.docs,self.auth,created_at=self.job['created_at']))
 def envelope(self):return self.sign(u.attach(self.job,self.grant(),self.auth,self.sign))
 def report(self):
  env=self.envelope();return env,dict(job_sha256=sha(env['payload']),source_files=env['payload']['source_files'],unaudited_training_execution=u.provenance(env,self.auth),training={'updates':[{'optimizer_threading':threads.expected_evidence(threads.POLICY)}]})
 def requote(self):self.q['equivalence_evidence']=self.sign(self.e);self.r['execution_qualification']=self.sign(self.q)
 def test_full_release_preparation_signed_job_worker_report(self):
  r=u.release(self.release(),self.auth);g=self.grant();env,report=self.report();d=u.validate(env,self.auth)
  self.assertEqual(r['version'],u.THREAD_RELEASE);self.assertEqual(g['payload']['version'],u.THREAD_PREPARATION);self.assertEqual(d['version'],u.THREAD_VERSION)
  self.assertEqual(threads.declared_policy(d),threads.POLICY);self.assertEqual(u.required_report(report,env,self.auth)['optimizer_threading'],threads.POLICY)
  self.assertEqual(env['payload']['manifest'],self.job['manifest']);self.assertEqual(env['payload']['submissions'],self.job['submissions'])
 def test_historical_synthetic_absent_policy_shape_unchanged(self):
  self.r=copy.deepcopy(self.previous['payload']);self.job['source_files']=self.r['execution_source_files'];env=self.envelope();d=u.validate(env,self.auth)
  self.assertEqual(d['version'],u.OBJECTIVE_VERSION);self.assertNotIn('optimizer_threading',d);self.assertNotIn('optimizer_threading',u.provenance(env,self.auth))
 def test_historical_release_cannot_opt_in(self):
  self.r=copy.deepcopy(self.previous['payload']);self.r['optimizer_threading']=threads.POLICY
  with self.assertRaises(ValueError):u.release(self.release(),self.auth)
 def test_v6_requires_policy(self):
  self.r.pop('optimizer_threading')
  with self.assertRaises(ValueError):u.release(self.release(),self.auth)
 def test_policy_values_and_types_not_coerced(self):
  for val in (8,16.0,True):
   r=copy.deepcopy(self.r);r['optimizer_threading']['optimizer_intraop']=val
   with self.subTest(val=val),self.assertRaises(ValueError):u.release(self.sign(r),self.auth)
 def test_missing_qualification_cannot_be_reporter_assertion(self):
  self.r['execution_qualification']=self.previous['payload']['execution_qualification']
  with self.assertRaises(ValueError):u.release(self.release(),self.auth)
 def test_sft_permission_does_not_authorize_production(self):
  self.r['execution_qualification']=self.q['original_full_state_execution']
  with self.assertRaises(ValueError):u.release(self.release(),self.auth)
 def test_root_equivalence_signature_required(self):
  self.q['equivalence_evidence']['signature']='bad';self.r['execution_qualification']=self.sign(self.q)
  with self.assertRaises(ValueError):u.release(self.release(),self.auth)
 def test_changed_scientific_file_cannot_sneak_in(self):
  n='subnet/persistent_cpu_adamw.py';self.r['execution_source_files'][n]='8'*64;self.r['changed_source_files'][n]='8'*64
  self.q['execution_source_files_sha256']=sha(self.r['execution_source_files']);self.e['source_files_sha256']=self.q['execution_source_files_sha256'];self.e['changed_files'][n]='8'*64;self.requote()
  with self.assertRaisesRegex(ValueError,'source delta'):u.release(self.release(),self.auth)
 def test_qualification_map_exact(self):
  self.q['execution_source_files_sha256']='9'*64;self.requote()
  with self.assertRaises(ValueError):u.release(self.release(),self.auth)
 def test_evidence_cannot_claim_new_numerical_qualification(self):
  self.e['production_numerical_qualification_claimed']=True;self.requote()
  with self.assertRaises(ValueError):u.release(self.release(),self.auth)
 def test_missing_durable_or_cpu_controls_refused(self):
  for field in ('cpu_caller_controls','release_report_controls','worker_cpu_readback','original_full_GET_readback'):
   q=copy.deepcopy(self.q);e=copy.deepcopy(self.e);e[field]['passed']=False;q['equivalence_evidence']=self.sign(e);r=copy.deepcopy(self.r);r['execution_qualification']=self.sign(q)
   with self.subTest(field=field),self.assertRaises(ValueError):u.release(self.sign(r),self.auth)
 def test_original_real_state_not_synthetic_result(self):
  orig=copy.deepcopy(self.q['original_full_state_execution']['payload']);orig['all_master_moments_projection_counters_equal']=False;self.q['original_full_state_execution']=self.sign(orig);self.e['original_execution_sha256']=sha(self.q['original_full_state_execution']);self.requote()
  with self.assertRaises(ValueError):u.release(self.release(),self.auth)
 def test_missing_report_evidence_refused(self):
  env,report=self.report();report['training']['updates'][0]={}
  with self.assertRaises(ValueError):u.required_report(report,env,self.auth)
 def test_per_update_evidence_count_exact(self):
  env,report=self.report();report['training']['updates']*=2
  with self.assertRaises(ValueError):u.required_report(report,env,self.auth)
 def test_wrong_restored_threads_refused(self):
  env,report=self.report();report['training']['updates'][0]['optimizer_threading']['after']['intraop']=16
  with self.assertRaises(ValueError):u.required_report(report,env,self.auth)
 def test_unsigned_self_asserted_policy_refused(self):
  env,report=self.report();report['unaudited_training_execution']['optimizer_threading']['optimizer_intraop']=8
  with self.assertRaises(ValueError):u.required_report(report,env,self.auth)
 def test_boolean_thread_report_refused(self):
  env,report=self.report();report['training']['updates'][0]['optimizer_threading']['during']['interop']=2.0
  with self.assertRaises(ValueError):u.required_report(report,env,self.auth)
 def test_historical_report_cannot_claim_cpu16(self):
  self.r=copy.deepcopy(self.previous['payload']);self.job['source_files']=self.r['execution_source_files'];env,report=self.report()
  with self.assertRaises(ValueError):u.required_report(report,env,self.auth)
 def test_prepared_grant_cannot_change_source(self):
  grant=self.grant();bad=copy.deepcopy(self.job);bad['source_files']['subnet/backend_jobs.py']='9'*64
  with self.assertRaises(ValueError):u.attach(bad,grant,self.auth,self.sign)
 def test_execution_scope_wrong_role_rejected(self):
  self.job['role']='verify'
  with self.assertRaises(ValueError):self.grant()
 def test_objective_and_lr_not_changed(self):
  env=self.envelope();d=env['payload'][u.FIELD]['payload'];self.assertEqual(d['effective_learning_rate'],5e-7);self.assertEqual(d['training_horizon'],self.previous['payload']['training_horizon']);self.assertEqual(d['training_objective'],self.previous['payload']['training_objective'])
 def test_pure_contract_import_never_imports_torch(self):
  import subprocess
  code="import sys;sys.path.insert(0,"+repr(str(S))+");from subnet import unaudited_training_execution,optimizer_threading;assert 'torch' not in sys.modules"
  out=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=20)
  self.assertEqual(out.returncode,0,out.stderr)
 def test_fresh_source_helper_reloads_after_bootstrap_admission(self):
  import subprocess
  code="""import sys,importlib
sys.path.insert(0,SOURCE)
from subnet import optimizer_threading as before, backend_jobs as b
before.bootstrap_poison=True
b.install_source_loader(__import__('pathlib').Path(SOURCE),('subnet/optimizer_threading.py','subnet/unaudited_training_execution.py'))
after=importlib.import_module('subnet.optimizer_threading')
assert not hasattr(after,'bootstrap_poison')
assert after is not before
assert after.POLICY['optimizer_intraop']==16
""".replace('SOURCE',repr(str(S)))
  out=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=20)
  self.assertEqual(out.returncode,0,out.stderr)

 def test_basic_v11_scope_accepts_v6_without_reward_assessment(self):
  from subnet import learner_selection_operator_bridge as peer
  env=self.envelope();job=env['payload'];manifest=job['manifest']['payload']
  policy=dict(version=peer.BASIC_THREAD_AUTH_VERSION,scientific_source_files=job['source_files'])
  self.assertEqual(peer.validate_execution_scope(job,manifest,policy,self.auth)['optimizer_threading'],threads.POLICY)
  policy['version']=peer.BASIC_AUTH_VERSION
  with self.assertRaises(ValueError):peer.validate_execution_scope(job,manifest,policy,self.auth)

if __name__=='__main__':unittest.main()
