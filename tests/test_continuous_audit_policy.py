import base64,unittest
from nacl.signing import SigningKey
from subnet.continuous_audit_policy import *

def signed(key,p):return dict(signer=key.verify_key.encode().hex(),payload=p,signature=base64.b64encode(key.sign(canonical(p)).signature).decode())
class PolicyControls(unittest.TestCase):
 def setUp(self):
  self.key=SigningKey.generate();self.worker=self.key.verify_key.encode().hex();self.authority=SigningKey.generate();self.root=self.authority.verify_key.encode().hex()
  self.p=dict(version=VERSION,recent_epochs=4,decay=.8,prior_alpha=1,prior_beta=1,invalid_multiplier=.25,zero_epoch_after=3,blacklist_after=4,blacklist_epochs=2)
  self.row=dict(epoch='e1',round=1,checkpoint='a'*64,miner='b'*64,env_id='math',index=0,batch_sha256='c'*64,proof_sha256='d'*64,commitment_sha256='e'*64,verifier_contract_sha256='f'*64,committed_at=10)
 def observation(self,outcome='verified_valid',job='1'*64,at=20):
  return dict(version='continuous-audit-observation-v1',**{k:self.row[k]for k in ('epoch','checkpoint','miner','batch_sha256','commitment_sha256','verifier_contract_sha256')},outcome=outcome,completed_at=at,job_sha256=job)
 def calculate(self,rows=None,obs=(),**kw):
  jobs={o['job_sha256']:dict(verifier=self.worker,observations=[o])for o in obs};return snapshot(rows or[self.row],[signed(self.key,o)for o in obs],{self.worker:['verify']},epoch='e1',round=1,checkpoint='a'*64,cutoff=30,audit_policy=self.p,admitted_jobs=jobs,**kw)
 def test_hourly_sum_raw_points_not_normalized_shares(self):
  a=self.calculate();a['cutoff']=3600;a['points']={'b'*64:2,'3'*64:1}
  b=dict(a,epoch='e2',round=2,points={'b'*64:0,'3'*64:7})
  r=hourly_aggregate([signed(self.authority,a),signed(self.authority,b)],self.root,3600);self.assertEqual(r['weights']['b'*64],.2);self.assertEqual(r['weights']['3'*64],.8);self.assertEqual(len(r['snapshot_bindings']),2)
  with self.assertRaises(ValueError):hourly_aggregate([signed(self.authority,a),signed(self.authority,a)],self.root,3600)
  self.assertFalse(r['chain_transactions'])
 def test_declared_but_ineligible_batch_does_not_earn_points(self):
  r=self.calculate(eligible_evidence_ids=[]);self.assertEqual(r['points']['b'*64],0.)
  r=self.calculate(obs=[self.observation('confirmed_invalid')],eligible_evidence_ids=[]);self.assertEqual(r['miners']['b'*64]['confirmed_invalid_current'],1);self.assertEqual(r['points']['b'*64],0.)
  with self.assertRaises(ValueError):self.calculate(eligible_evidence_ids=['0'*64])
 def test_infra_then_real_retry_not_conflicting_scientific_evidence(self):
  infra=self.observation('infrastructure_error');good=self.observation(job='2'*64)
  for evidence in ([infra,good],[good,infra]):
   r=self.calculate(obs=evidence);self.assertGreater(r['points']['b'*64],.5);self.assertEqual(r['miners']['b'*64]['confirmed_invalid_current'],0)
 def test_population_and_snapshot_binding_ignore_input_order(self):
  other=dict(self.row,miner='2'*64,index=1,batch_sha256='3'*64)
  self.assertEqual(population([self.row,other]),population([other,self.row]))
  self.assertEqual(self.calculate([self.row,other]),self.calculate([other,self.row]))
 def test_prior_not_a_verified_claim(self):
  r=self.calculate();self.assertEqual(r['points']['b'*64],.5);self.assertFalse(r['unaudited_samples_claimed_verified']);self.assertFalse(r['training_waits_for_audits'])
 def test_probability_falls_and_repeat_deduplicates(self):
  good=self.calculate(obs=[self.observation()]);bad=self.calculate(obs=[self.observation('confirmed_invalid')]);self.assertGreater(good['points']['b'*64],bad['points']['b'*64]);repeat=self.calculate(obs=[self.observation(),self.observation()]);self.assertEqual(good,repeat)
 def test_ambiguity_and_infrastructure_not_fraud(self):
  for kind in ('numerical_ambiguous','infrastructure_error'):
   r=self.calculate(obs=[self.observation(kind)]);self.assertEqual(r['points']['b'*64],.5);self.assertEqual(r['miners']['b'*64]['confirmed_invalid_current'],0)
 def test_duplicate_indices_zero_both(self):
  other=dict(self.row,miner='2'*64,batch_sha256='3'*64);r=self.calculate([self.row,other]);self.assertEqual(set(r['points'].values()),{0.})
 def test_cutoff_and_future_round(self):
  for row in (dict(self.row,committed_at=31),dict(self.row,round=2),dict(self.row,checkpoint='9'*64)):
   with self.assertRaises(ValueError):self.calculate([row])
  r=self.calculate(obs=[self.observation(at=31)]);self.assertEqual(r['points']['b'*64],.5)
 def test_signature_alone_is_not_execution_evidence(self):
  with self.assertRaises(ValueError):observations([signed(self.key,self.observation())],[self.row],{self.worker:['verify']},30)
 def test_no_replacement_and_excludes_prior_draw(self):
  rows=[self.row,dict(self.row,index=1,batch_sha256='9'*64)];first=random_selection(rows,'1'*64,1);second=random_selection(rows,'2'*64,2,[digest(first[0])]);self.assertEqual(len(second),1);self.assertNotEqual(first[0],second[0])
 def test_explicit_reference_adjudication(self):
  a=self.observation('numerical_ambiguous');b=self.observation(job='2'*64)
  with self.assertRaises(ValueError):self.calculate(obs=[a,b])
  resolution=dict(version='continuous-audit-adjudication-v1',evidence_id=digest(self.row),original_job_sha256=a['job_sha256'],reference_job_sha256=b['job_sha256'],outcome=b['outcome'])
  r=self.calculate(obs=[a,b],adjudications=[signed(self.authority,resolution)],authority=self.root);self.assertGreater(r['points']['b'*64],.5)
 def queue_fixture(self,kind='verified_valid'):
  manifest=dict(epoch='e1',checkpoint={'id':'a'*64},sampling_contract={'version':'test'},sampling_source_hash='1'*64,model_runtime_revision='runtime',backend_profile={'version':'test'},numerical_policy={'version':'test'},source_bundle={'sha256':'9'*64})
  row=dict(self.row,verifier_contract_sha256=verifier_contract(manifest));self.row=row
  job=dict(role='verify',job_id='job-1',manifest=signed(self.authority,manifest),source_files={'model.py':'8'*64},runtime_versions={'torch':'pinned'},submissions=[dict(sha256=row['proof_sha256'],commitment_ref={k:row[k]for k in ('miner','batch_sha256','commitment_sha256')})])
  outcome=dict(valid=kind=='verified_valid',fully_audited=kind!='structural_invalid',failure_kind=kind)
  report=dict(success=True,role='verify',job_id='job-1',job_sha256=digest(job),operator=self.root,epoch='e1',checkpoint='a'*64,source_files=job['source_files'],runtime_versions=job['runtime_versions'],backend_profile=manifest['backend_profile'],numerical_policy=manifest['numerical_policy'],execution_resources_enforced=True,completed_at=20,audits=[dict(submission_sha256=row['proof_sha256'],epoch='e1',outcomes=[outcome])])
  request=signed(self.key,dict(action='report',job_id='job-1',token='lease-token',report=report))
  queue=dict(status='complete',role='verify',worker=self.worker,id='job-1',digest=digest(job),envelope=signed(self.authority,job),report=report,report_request=request,report_digest=digest(report),token='lease-token')
  return queue,{'9'*64:job['source_files']}
 def test_real_original_report_and_source_join(self):
  queue,pins=self.queue_fixture();admitted=admit_queue_reports([queue],[self.row],self.root,{self.worker:['verify']},pins)
  r=snapshot([self.row],[dict(admitted_queue_job_sha256=k)for k in admitted],{self.worker:['verify']},epoch='e1',round=1,checkpoint='a'*64,cutoff=30,audit_policy=self.p,admitted_jobs=admitted);self.assertGreater(r['points']['b'*64],.5)
  with self.assertRaises(ValueError):admit_queue_reports([queue],[self.row],self.root,{self.worker:['verify']},{'9'*64:{'model.py':'0'*64}})
  queue['token']='substitution'
  with self.assertRaises(ValueError):admit_queue_reports([queue],[self.row],self.root,{self.worker:['verify']},pins)
 def test_authentic_structurally_invalid_proof_is_invalid(self):
  queue,pins=self.queue_fixture('structural_invalid');admitted=admit_queue_reports([queue],[self.row],self.root,{self.worker:['verify']},pins);self.assertEqual(next(iter(admitted.values()))['observations'][0]['outcome'],'confirmed_invalid')
 def test_artifact_failure_is_authenticated_storage_not_gpu(self):
  failure=dict(version='continuous-artifact-capture-failure-v1',row=self.row,outcome='confirmed_invalid',completed_at=20,reason='NoSuchKey',original_selection_sha256='1'*64,scientific_model_execution_claim=False)
  admitted=admit_artifact_failures([signed(self.authority,failure)],[self.row],self.root);self.assertFalse(next(iter(admitted.values()))['scientific_model_execution_claim'])
  with self.assertRaises(Exception):admit_artifact_failures([signed(self.key,failure)],[self.row],self.root)
if __name__=='__main__':unittest.main()
