import copy,json,time,unittest
from subnet.continuous_audit_service import grouping_policy,admitted_service_config
from subnet.distributed_roles import authenticate,validate_frozen_submissions
from subnet.continuous_audit_policy import digest
from test_parallel_audit_capture import ParallelCapture
from test_commitment_child_queue import sign

class BoundedGrouping(ParallelCapture):
 def setUp(self):
  super().setUp();self.service.sources={'2'*64:{'subnet/model.py':'3'*64}};self.service.group_size=4;self.service._capture=lambda row,p:self.capture_data[digest(row)]
 def jobs(self):return [authenticate(json.loads(p.read_text()),self.authority)for p in sorted(self.service.directory.glob('continuous-audit-group-*-job.json'))]
 def test_multiminer_full_child_queue_binding_and_same_checkpoint(self):
  result=self.service.tick(now=1);self.assertEqual(result['enqueued'],2);jobs=self.jobs();self.assertEqual([len(j['submissions'])for j in jobs],[4,4]);ids=[]
  for j in jobs:
   m=authenticate(j['manifest'],self.authority);validate_frozen_submissions(m,j['submissions']);self.assertEqual(len(m['audit_frozen_receipts']),4);ids+=j['audit_group']['row_sha256s'];self.assertEqual(j['expires_at']-j['created_at'],900)
  self.assertEqual(sorted(ids),sorted(digest(r)for r in self.rows));self.assertEqual(len(ids),len(set(ids)))
 def test_original_group_crash_recovery_never_recaptures_extends_or_redraws(self):
  self.service.tick(now=1);paths=list(self.service.directory.glob('continuous-audit-group-*-job.json'));saved={p:p.read_bytes()for p in paths};draws=copy.deepcopy(self.service.state['draws']);self.service.state['jobs']={}
  for plan in self.service.state['group_plans'].values():plan['resolved']=False
  self.service._capture=lambda *a:(_ for _ in ()).throw(AssertionError('original requests already exist'));self.service.tick(now=10000)
  self.assertEqual(self.service.state['draws'],draws);self.assertEqual(len(self.service.state['jobs']),2)
  for p,b in saved.items():self.assertEqual(p.read_bytes(),b)
 def test_partial_infrastructure_failure_retries_only_failed_draw(self):
  bad=digest(self.rows[0]);old=self.service._capture
  self.service._capture=lambda row,p:(_ for _ in ()).throw(TimeoutError())if digest(row)==bad else old(row,p)
  self.service.tick(now=1);first=self.jobs();draw=copy.deepcopy(self.service.state['draws']);ids=[i for j in first for i in j['audit_group']['row_sha256s']];self.assertEqual(len(ids),7);self.assertNotIn(bad,ids);self.assertEqual(self.service.state['capture_failures'][bad]['kind'],'infrastructure_error')
  self.service._capture=old;self.service.tick(now=2);self.assertEqual(self.service.state['draws'],draw);allids=[i for j in self.jobs()for i in j['audit_group']['row_sha256s']];self.assertEqual(len(allids),8);self.assertEqual(len(set(allids)),8)
 def test_same_miner_two_slots_merge_inventory_and_real_authenticated_ACK(self):
  from subnet.commitment_transport import make,VERSION2
  from subnet.distributed_roles import Coordinator
  from subnet.continuous_audit_policy import verifier_contract,admit_queue_reports
  from types import SimpleNamespace
  m=copy.deepcopy(self.fixture.manifest);m['submission_transport_policy']=VERSION2;m['audit_policy']={'mode':'full','version':1};m['environments'][0]['indices']=[17,18];m.pop('audit_frozen_receipts');m['sampling_contract']={'version':'test'};m['sampling_source_hash']='f'*64
  b0=copy.deepcopy(self.fixture.batch);b1=dict(b0,index=18);env=make(self.fixture.identity,m,[(b0,b'ZIP actual bytes'),(b1,b'ZIP second actual bytes')]);artifacts=[];rows=[]
  for child in env['payload']['batches']:
   frozen='public/'+m['epoch']+'/submissions/'+self.fixture.identity.id+'/'+digest(env)+'/'+str(child['slot'])+'.zip';artifacts.append(dict(child,key='private/'+str(child['slot']),frozen_key=frozen,read_url='https://example.r2.cloudflarestorage.com/bucket/'+frozen,etag='original',received_at=15))
   rows.append(dict(self.rows[0],miner=self.fixture.identity.id,index=child['index'],batch_sha256=child['batch_sha256'],proof_sha256=child['sha256'],commitment_sha256=digest(env),verifier_contract_sha256=verifier_contract(m)))
  receipt=dict(sha256=digest(env),commitment_document=env,artifacts=artifacts);self.service.state['populations'][m['epoch']]=sign(self.root,{'manifest_document':sign(self.root,m)});self.service.dispatch_records=lambda:(rows,0);self.service._capture=lambda row,p:(m,receipt,next(a for a in artifacts if a['index']==row['index']))
  self.service.tick();j=self.jobs()[0];manifest=authenticate(j['manifest'],self.authority);self.assertEqual(len(manifest['audit_frozen_receipts']),1);self.assertEqual(len(j['submissions']),2);validate_frozen_submissions(manifest,j['submissions'])
  worker=self.fixture.worker;wid=worker.verify_key.encode().hex();self.queue.workers[wid]=['verify'];nonce=0
  def request(action,**kw):
   nonlocal nonce
   nonce+=1;return self.queue.request(sign(worker,dict(action=action,at=time.time(),nonce=str(nonce).zfill(32),**kw)))
  claim=request('claim',role='verify')['claim'];self.assertEqual(claim['job']['payload'],j);at=time.time();audits=[]
  for obj in j['submissions']:
   b=b0 if obj['commitment_ref']['index']==17 else b1
   audits.append(dict(epoch=m['epoch'],submission_sha256=obj['sha256'],accepted=[b],outcomes=[dict(valid=True,fully_audited=True)]))
  report=dict(success=True,job_id=j['job_id'],job_sha256=digest(j),operator=self.authority,role='verify',epoch=m['epoch'],checkpoint=m['checkpoint']['id'],source_files=j['source_files'],runtime_versions=j['runtime_versions'],backend_profile=m['backend_profile'],numerical_policy=m['numerical_policy'],execution_resources_enforced=True,chain_transactions=False,completed_at=at,audits=audits)
  self.assertTrue(request('report',job_id=j['job_id'],token=claim['token'],report=report)['accepted'])
  with self.queue.transaction()as db:actual=dict(db.execute('select *from jobs where id=?',(j['job_id'],)).fetchone())
  admitted=admit_queue_reports([actual,actual],rows,self.authority,{wid:['verify']},{m['source_bundle']['sha256']:j['source_files']});self.assertEqual(len(admitted),1);self.assertEqual(len(next(iter(admitted.values()))['observations']),2)
 def test_grouped_scoring_deduplicates_penalties_per_original_child(self):
  from test_continuous_audit_policy import PolicyControls
  from subnet.continuous_audit_policy import admit_queue_reports,snapshot
  fixture=PolicyControls();fixture.setUp();queue,pins=fixture.queue_fixture('confirmed_invalid');row1=fixture.row;row2=dict(row1,index=1,batch_sha256='4'*64,proof_sha256='5'*64)
  j=copy.deepcopy(queue['envelope']['payload']);j['submissions'].append(dict(sha256=row2['proof_sha256'],commitment_ref={k:row2[k]for k in ('miner','batch_sha256','commitment_sha256')}));report=copy.deepcopy(queue['report']);report['job_sha256']=digest(j);report['audits'].append(dict(submission_sha256=row2['proof_sha256'],epoch=row2['epoch'],outcomes=[dict(valid=True,fully_audited=True)]));queue.update(envelope=sign(fixture.authority,j),digest=digest(j),report=report,report_digest=digest(report),report_request=sign(fixture.key,dict(action='report',job_id=j['job_id'],token=queue['token'],report=report)))
  admissions=admit_queue_reports([queue,queue],[row1,row2],fixture.root,{fixture.worker:['verify']},pins);self.assertEqual(len(admissions),1)
  result=snapshot([row1,row2],[dict(admitted_queue_job_sha256=k)for k in admissions],{fixture.worker:['verify']},epoch='e1',round=1,checkpoint='a'*64,cutoff=30,audit_policy=fixture.p,admitted_jobs=admissions);self.assertEqual(result['miners'][row1['miner']]['confirmed_invalid_current'],1);self.assertEqual(len(result['evidence_ids']),2)
 def test_cap_default_and_exact_signed_configuration(self):
  self.assertEqual(grouping_policy(None),1)
  for x in (True,0,1,5,64):
   with self.assertRaises(ValueError):grouping_policy({'version':'bounded-checkpoint-audit-groups-v1','max_submissions':x})
  policy={'version':'bounded-checkpoint-audit-groups-v1','max_submissions':4};cfg={'capture_workers':4,'policy':self.service.policy,'job_grouping_policy':policy};payload={'version':'continuous-audit-service-sources-v1','capture_workers':4,'audit_policy':self.service.policy,'job_grouping_policy':policy};cfg['source_admission']=sign(self.root,payload);admitted_service_config(cfg,self.authority);cfg['job_grouping_policy']=dict(policy,max_submissions=2)
  with self.assertRaises(ValueError):admitted_service_config(cfg,self.authority)
 def test_downgrade_to_default_does_not_duplicate_group_members(self):
  self.service.tick();self.service.group_size=1;self.service._capture=lambda *a:(_ for _ in ()).throw(AssertionError('already grouped members cannot retry'));r=self.service.tick();self.assertEqual(r['enqueued'],0)
 def test_malformed_persisted_group_plan_fails_closed(self):
  self.service.tick();plan=next(iter(self.service.state['group_plans'].values()));plan['resolved']=False;plan['row_sha256s'].append(plan['row_sha256s'][0])
  with self.assertRaisesRegex(ValueError,'bounded original audit group plan'):self.service.tick()
 def test_no_group_spans_different_immutable_openings(self):
  row=self.rows[-1];old=self.capture_data[digest(row)];m,receipt,a=old;m=copy.deepcopy(m);m['epoch']='another-opening';new=dict(row,epoch=m['epoch']);self.rows[-1]=new;self.capture_data[digest(new)]=(m,receipt,a);self.service.state['populations'][m['epoch']]=sign(self.root,{'manifest_document':sign(self.root,m)})
  # Metadata test bypasses queue receipt admission for the changed fixture, but
  # authentic group plans must still segregate openings before any capture.
  self.service._capture=lambda *a:(_ for _ in ()).throw(TimeoutError());self.service.tick()
  for p in self.service.state['group_plans'].values():self.assertEqual({self.service.state['draws'][i]['row']['epoch']for i in p['row_sha256s']},{p['epoch']})
 def test_group_child_mutations_rejected_by_actual_queue_contract(self):
  self.service.tick();j=self.jobs()[0];m=authenticate(j['manifest'],self.authority)
  for field,value in [('miner','f'*64),('batch_sha256','e'*64),('slot',18),('size',999)]:
   bad=copy.deepcopy(j['submissions']);bad[0]['commitment_ref'][field]=value
   with self.assertRaises(ValueError):validate_frozen_submissions(m,bad)
  bad=copy.deepcopy(j['submissions']);bad.append(copy.deepcopy(bad[0]))
  with self.assertRaises(ValueError):validate_frozen_submissions(m,bad)

# The inherited parallel default-off tests should still use group size1.
for name in ('test_four_parallel_captures_owner_signing_full_original_bindings_and_fresh_ttl','test_restart_reuses_exact_original_job_without_recapture_or_extension','test_infra_failure_retains_selection_and_retry_no_redraw'):
 setattr(BoundedGrouping,name,None)
if __name__=='__main__':unittest.main()
