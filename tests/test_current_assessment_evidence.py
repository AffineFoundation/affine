import copy,json,sqlite3,unittest
from pathlib import Path
from unittest.mock import patch
from nacl.signing import SigningKey
import test_committed_training_inputs as learner_fixtures
import test_continuous_audit_policy as policy_fixtures
signed=policy_fixtures.signed
from subnet.continuous_audit_policy import digest,RESOLUTION_VERSION
from subnet.continuous_audit_service import register_population
from ops import current_assessment_evidence as e

class EvidenceControls(unittest.TestCase):
 def setUp(self):
  self.fx=learner_fixtures.LearnerAdmissionTests();self.fx.setUp();self.addCleanup(self.fx.doCleanups)
  self.root=self.fx.root;self.authority=self.fx.authority;self.key=self.fx.operator
  self.workerkey=SigningKey.generate();self.worker=self.workerkey.verify_key.encode().hex()
  self.policy=dict(version=RESOLUTION_VERSION,recent_epochs=8,decay=.8,prior_alpha=1,prior_beta=1,invalid_multiplier=.25,zero_epoch_after=3,blacklist_after=4,blacklist_epochs=2)
  m=copy.deepcopy(self.fx.manifest);m.update(model_runtime_revision='runtime',backend_profile={'version':'test'},numerical_policy={'version':'test'},sampling_contract=None,sampling_source_hash=None,harness_source_hash='5'*64)
  for native in m.get('environments',[]):native['spec']['source_hash']='6'*64
  if 'environment' in m:m['environment']['source_hash']='6'*64
  self.m=m;self.source=m['source_bundle']['sha256'];self.pins={'subnet/backend_jobs.py':'8'*64,'subnet/harness.py':'7'*64}
  original=self.fx.obj['learner_admission']['payload']['original_commitment'];receipt=dict(commitment_document=original,sha256=digest(original));child=original['payload']['batches'][0]
  pair=dict(miner=self.fx.identity,commitment_sha256=digest(original),batch_sha256=child['batch_sha256'],proof_sha256=child['sha256'])
  self.pop=signed(self.key,register_population(signed(self.key,m),{self.fx.identity:receipt},1,10,self.authority,eligible_pairs=[pair]))
  self.row=self.pop['payload']['records'][0];rid=digest(self.row)
  self.state=dict(populations={m['epoch']:self.pop},draws={rid:{'row':self.row}},jobs={'job-1':{'row_sha256':rid}},capture_failures={})
  profile=dict(backend='standard-backend-no-os-resource-enforcement-v1',backend_module_sha256='8'*64,model_runtime_revision='runtime',backend_profile=m['backend_profile'],numerical_policy=m['numerical_policy'],runtime_versions={'torch':'pinned'},execution_resources_enforced=False)
  source=dict(version='continuous-audit-service-sources-v1',approved_sources={self.source:self.pins},job_metadata={self.source:dict(source_files=self.pins,runtime_versions={'torch':'pinned'})},audit_policy=self.policy,execution_evidence_policy=dict(version='explicit-backend-execution-evidence-v1',effective_cutoff=0,sources={self.source:profile}))
  self.config=dict(state=str(self.root),continuous_audit_service=dict(source_admission=signed(self.key,source),policy=self.policy),remote={'verify':[{'identity':'0'*64}]})
  self.cfg=self.root/'config.json';self.directory=self.root/'continuous-audit';self.directory.mkdir();(self.root/'roles').mkdir()
  self.queue=self.make_queue('verified_valid');self.save()
 def make_queue(self,outcome,identifier='job-1',at=20):
  fx=policy_fixtures.PolicyControls();fx.setUp();fx.authority=self.key;fx.root=self.authority;fx.key=self.workerkey;fx.worker=self.worker;fx.row=self.row
  q,_=fx.queue_fixture(outcome);job=q['envelope']['payload'];job.update(job_id=identifier,manifest=signed(self.key,self.m),source_files=self.pins,runtime_versions={'torch':'pinned'});job['submissions'][0]=dict(sha256=self.row['proof_sha256'],commitment_ref={k:self.row[k] for k in ('miner','batch_sha256','commitment_sha256')})
  q.update(id=identifier,digest=digest(job),envelope=signed(self.key,job));report=q['report'];report.update(job_id=identifier,job_sha256=digest(job),epoch=self.m['epoch'],checkpoint=self.m['checkpoint']['id'],source_files=self.pins,completed_at=at,execution_resources_enforced=False);report['audits'][0].update(submission_sha256=self.row['proof_sha256'],epoch=self.m['epoch'])
  if outcome=='numerical_ambiguous':report['audits'][0]['outcomes'][0].update(valid=None,fully_audited=False)
  q.update(report_request=signed(self.workerkey,dict(action='report',job_id=identifier,token=q['token'],report=report)),report_digest=digest(report));return q
 def save(self,queues=None):
  self.cfg.write_text(json.dumps(self.config));(self.directory/'audit-state.json').write_text(json.dumps(self.state))
  db=sqlite3.connect(self.root/'roles/verifier-queue.sqlite3');db.execute('DROP TABLE IF EXISTS jobs');columns=list(self.queue);db.execute('CREATE TABLE jobs ('+','.join(k+' TEXT' for k in columns)+')')
  for q in queues or [self.queue]:db.execute('INSERT INTO jobs VALUES ('+','.join('?' for k in columns)+')',[json.dumps(q[k]) if isinstance(q[k],dict) else q[k] for k in columns])
  db.commit();db.close()
 def load(self,**kw):return e.load_evidence(self.cfg,authority=self.authority,cutoff=30,verifiers=[self.worker],expected_source_admission_sha256=digest(self.config['continuous_audit_service']['source_admission']),**kw)
 def test_actual_signed_join_readonly_without_learner_or_opening(self):
  paths=[self.cfg,self.directory/'audit-state.json',self.root/'roles/verifier-queue.sqlite3'];before=[p.read_bytes() for p in paths]
  r=self.load();self.assertEqual(r['refused'],[]);self.assertEqual(len(r['snapshots']),1);self.assertGreater(r['snapshots'][0]['miners'][self.fx.identity]['validity_probability'],.5);self.assertEqual(before,[p.read_bytes() for p in paths])
 def test_invalid_population_does_not_block_peer(self):
  broken=copy.deepcopy(self.pop);broken['signature']='A'*88;self.state['populations']['corrupt']=broken;self.save();r=self.load();self.assertEqual(len(r['snapshots']),1);self.assertEqual(r['refused'][0]['kind'],'population')
 def test_original_report_after_cutoff_is_excluded(self):
  self.queue=self.make_queue('verified_valid',at=31);self.save();r=self.load();self.assertEqual(r['snapshots'][0]['miners'][self.fx.identity]['validity_probability'],.5);self.assertEqual(r['excluded'][0]['kind'],'job')
 def test_report_binding_tamper_is_refused(self):
  self.queue['token']='other';self.save();r=self.load();self.assertEqual(r['snapshots'][0]['miners'][self.fx.identity]['validity_probability'],.5);self.assertTrue(r['refused'])
 def test_untrusted_config_verifier_does_not_promote(self):
  self.queue['worker']='0'*64;self.save();r=self.load();self.assertTrue(r['refused']);self.assertEqual(r['snapshots'][0]['miners'][self.fx.identity]['validity_probability'],.5)
 def test_source_registry_corruption_is_integrity_error(self):
  self.config['continuous_audit_service']['source_admission']['signature']='A'*88;self.save()
  with self.assertRaises(ValueError):self.load()
 def test_expected_source_digest_not_optional_promotion(self):
  with self.assertRaises(ValueError):e.load_evidence(self.cfg,authority=self.authority,cutoff=30,verifiers=[self.worker],expected_source_admission_sha256='0'*64)
 def test_native_source_mismatch_refuses_population(self):
  p=copy.deepcopy(self.pop['payload']);m=p['manifest_document']['payload'];m['harness_source_hash']='malformed';p['manifest_document']=signed(self.key,m);self.state['populations'][m['epoch']]=signed(self.key,p);self.save();r=self.load();self.assertEqual(r['snapshots'],[]);self.assertTrue(r['refused'])
 def test_unknown_is_not_valid_or_fraud(self):
  self.queue=self.make_queue('numerical_ambiguous');self.save();r=self.load();d=r['snapshots'][0]['miners'][self.fx.identity];self.assertEqual(d['validity_probability'],.5);self.assertEqual(d['confirmed_invalid_recent'],0);self.assertEqual(d['reward_multiplier'],1);self.assertEqual(d['resolution_coverage_factor'],0)
 def test_duplicate_original_evidence_counts_once(self):
  self.queue=self.make_queue('confirmed_invalid');q2=self.make_queue('confirmed_invalid','job-2',21);self.state['jobs']['job-2']=dict(self.state['jobs']['job-1']);self.save([self.queue,q2]);r=self.load();self.assertEqual(r['snapshots'][0]['miners'][self.fx.identity]['confirmed_invalid_recent'],1)
 def test_bad_job_does_not_suppress_good_job(self):
  q2=self.make_queue('verified_valid','job-2',21);q2['report_request']['signature']='A'*88;self.state['jobs']['job-2']=dict(self.state['jobs']['job-1']);self.save([self.queue,q2]);r=self.load();self.assertGreater(r['snapshots'][0]['miners'][self.fx.identity]['validity_probability'],.5);self.assertTrue(r['refused'])
 def test_new_population_does_not_reset_actual_global_penalty(self):
  self.queue=self.make_queue('confirmed_invalid')
  m=copy.deepcopy(self.m);m['epoch']='new-unaudited-epoch'
  old=next(iter(self.pop['payload']['receipts'].values()))['commitment_document']['payload'];c=copy.deepcopy(old);c['epoch']=m['epoch'];c['batches'][0]['index']=1
  original=signed(self.fx.miner,c);child=c['batches'][0]
  pair=dict(miner=self.fx.identity,commitment_sha256=digest(original),batch_sha256=child['batch_sha256'],proof_sha256=child['sha256'])
  p=register_population(signed(self.key,m),{self.fx.identity:dict(commitment_document=original,sha256=digest(original))},2,25,self.authority,eligible_pairs=[pair]);self.state['populations'][m['epoch']]=signed(self.key,p);self.save()
  r=self.load();self.assertEqual(len(r['snapshots']),2);new=r['snapshots'][-1]
  self.assertEqual(new['cohort_miner_details'][self.fx.identity]['confirmed_invalid_current'],0)
  self.assertEqual(new['miners'][self.fx.identity]['confirmed_invalid_recent'],1);self.assertEqual(new['miners'][self.fx.identity]['reward_multiplier'],.25);self.assertAlmostEqual(new['miners'][self.fx.identity]['validity_probability'],1/2.8)
 def test_queue_outage_propagates(self):
  with patch.object(e,'queue_rows',side_effect=TimeoutError('busy')):
   with self.assertRaises(TimeoutError):self.load()

class GlobalControls(unittest.TestCase):
 def setUp(self):self.p=dict(version=RESOLUTION_VERSION,recent_epochs=8,decay=.8,prior_alpha=1,prior_beta=1,invalid_multiplier=.25,zero_epoch_after=3,blacklist_after=4,blacklist_epochs=2)
 def calc(self,rows,round=2):return e.current_estimates(rows,{'m'},round,self.p)['m']
 def obs(self,kind,round=1):return dict(miner='m',round=round,outcome=kind)
 def test_new_unaudited_epoch_preserves_global_bad_evidence(self):
  d=self.calc([self.obs('confirmed_invalid')]);self.assertAlmostEqual(d['validity_probability'],1/2.8);self.assertEqual(d['reward_multiplier'],.25);self.assertEqual(d['confirmed_invalid_recent'],1)
 def test_unknown_and_infra_are_not_fraud_or_success(self):
  d=self.calc([self.obs('numerical_ambiguous'),self.obs('infrastructure_error'),self.obs('verified_valid')]);self.assertAlmostEqual(d['validity_probability'],1.8/2.8);self.assertEqual(d['resolution_coverage_factor'],.5);self.assertEqual(d['reward_multiplier'],1)
 def test_recent_zero_penalty_and_blacklist_duration(self):
  self.assertEqual(self.calc([self.obs('confirmed_invalid')]*3)['reward_multiplier'],0);self.p['zero_epoch_after']=0
  self.assertTrue(self.calc([self.obs('confirmed_invalid')]*4)['blacklisted']);self.assertFalse(self.calc([self.obs('confirmed_invalid')]*4,round=3)['blacklisted'])
 def test_evidence_outside_signed_window_expires(self):
  d=self.calc([self.obs('confirmed_invalid',0)],round=8);self.assertEqual(d['validity_probability'],.5);self.assertEqual(d['reward_multiplier'],1)

if __name__=='__main__':unittest.main()

class IncrementalAdmissionControls(unittest.TestCase):
 def test_all_typed_pairs_match_authoritative_observations(self):
  fx=policy_fixtures.PolicyControls();fx.setUp()
  kinds=('verified_valid','confirmed_invalid','numerical_ambiguous','infrastructure_error')
  for first in kinds:
   for second in kinds:
    for adjudicated in (False,True):
     a=fx.observation(first,job='1'*64);b=fx.observation(second,job='2'*64,at=21)
     jobs={a['job_sha256']:dict(verifier=fx.worker,observations=[a]),b['job_sha256']:dict(verifier=fx.worker,observations=[b])};ptr=[dict(admitted_queue_job_sha256=k) for k in jobs]
     resolution=dict(version='continuous-audit-adjudication-v1',evidence_id=digest(fx.row),original_job_sha256=a['job_sha256'],reference_job_sha256=b['job_sha256'],outcome=second)
     docs=[signed(fx.authority,resolution)] if adjudicated else []
     kwargs=dict(admitted_jobs=jobs,adjudications=docs,authority=fx.root)
     try:expected=e.observations(ptr,[fx.row],{fx.worker:['verify']},30,**kwargs)
     except ValueError:expected=None
     state={}
     try:
      for pointer in ptr:
       one=e.observations([pointer],[fx.row],{fx.worker:['verify']},30,**kwargs)
       state.update(e.candidate_updates(state,one,[resolution] if adjudicated else []))
      actual=list(state.values())
     except ValueError:actual=None
     self.assertEqual(actual,expected,(first,second,adjudicated))
 def test_group_conflict_refuses_without_partial_state_mutation(self):
  old=dict(evidence_id='a',job_sha256='1',outcome='verified_valid');resolved={'a':old}
  candidate=[dict(evidence_id='new',job_sha256='2',outcome='verified_valid'),dict(evidence_id='a',job_sha256='2',outcome='confirmed_invalid')]
  with self.assertRaises(ValueError):e.candidate_updates(resolved,candidate,[])
  self.assertEqual(resolved,{'a':old})
