"""CPU metadata/signature/controller hooks only. No real GPU or chain assertions."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
OUT=Path(__file__).parents[1]/'ops'
import base64,copy,json,tempfile,time,threading,unittest,importlib.util
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet import live_reward_bridge as b
from subnet.scoring import score
from subnet.chain import hourly_points
from subnet.storage import Gateway,Identity,canonical
from subnet.controller import Controller
from subnet.remote_backend import RemoteController
from subnet.backend_profiles import profile
sp=importlib.util.spec_from_file_location('reward_worker',OUT/'live_reward_exporter.py');worker=importlib.util.module_from_spec(sp);sp.loader.exec_module(worker)
def sign(payload,key):return dict(payload=payload,signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(b.canonical(payload)).signature).decode())
def fixture():
 key=SigningKey.generate();minerkeys=[SigningKey.generate(),SigningKey.generate()];ids=[k.verify_key.encode().hex() for k in minerkeys];epoch='nonpayable-live-reward-math-v1-TEST-1'
 policy=dict(mode='sampled',version='bounded-random-v1',epoch_budget=2,escalation_budget=2,minimum_per_miner=1,maximum_per_miner=3,penalties=dict(invalid_batch_multiplier=.5,zero_epoch_after=0,penalize_structural=False))
 anchor=dict(version='live-reward-cutover-v1',netuid=120,owner_hotkey='owner',cutover_id='CUTOVER-CPU-TEST',effective_at=3600,compute_epoch_prefix='nonpayable-live-reward-math-v1-',live_epoch_prefix='live-math-reward-v1-',approved_compute_sources=['a'*64])
 manifest=dict(payable=False,epoch=epoch,start=3601,deadline=3900,checkpoint={'id':'b'*64},source_bundle={'sha256':'a'*64},capabilities={i:'CPU-test-ciphertext' for i in ids},K=1,L=1,max_batches=3,audit_policy=policy,environments=[dict(env_id='math',spec={'version':'native-TEST'},indices=[0,1,2])])
 regs={f'hotkey-{i}':dict(uid=i+1,public_key=k,snapshot_block=123) for i,k in enumerate(ids)}
 manifest=b.inject_opening_manifest(manifest,sign(anchor,key),key.verify_key.encode().hex(),regs)
 def batch(index):return dict(schema=2,epoch=epoch,checkpoint='b'*64,env_id='math',index=index,sample_index=index,environment_version='native-TEST',rollouts=[dict(env_id='math',index=index,reward=1.,classification='positive'),dict(env_id='math',index=index,reward=0.,classification='negative')])
 reports={ids[0]:dict(epoch=epoch,submission_sha256='c'*64,remote_job_id='VERIFIER-CPU-TEST-A',accepted=[batch(0)],outcomes=[dict(batch=0,env_id='math',index=0,valid=True,fully_audited=True),dict(batch=1,valid=False,fully_audited=True,failure_kind='confirmed_invalid'),dict(batch=2,valid=False,failure_kind='verification_error')]),ids[1]:dict(epoch=epoch,submission_sha256='d'*64,remote_job_id='VERIFIER-CPU-TEST-B',accepted=[batch(1)],outcomes=[dict(batch=0,env_id='math',index=1,valid=True,fully_audited=True),dict(batch=1,valid=None,fully_audited=False)])}
 scores=dict(score(reports,policy['penalties']),payable=False,epoch_id=epoch,checkpoint='b'*64,finalized_at=4100,receipts={i:{'sha256':reports[i]['submission_sha256']} for i in ids})
 md=sign(manifest,key);opening=sign(dict(version='immutable-first-manifest-v1',epoch=epoch,first_manifest_sha256=b.sha(md),published_at=3602),key);rd=sign(dict(epoch=epoch,snapshot_block=123,registrations=regs),key)
 inputs=dict(manifest_document=md,opening_document=opening,score_document=sign(scores,key),audit_documents={i:sign(r,key) for i,r in reports.items()},registrations_document=rd,anchor_document=sign(anchor,key),authority=key.verify_key.encode().hex())
 return key,inputs,regs,reports
class BridgeTests(unittest.TestCase):
 def test_sampled_verified_subset_penalty_reducer_and_no_historical_reclassification(self):
  key,inputs,regs,reports=fixture();r=b.project(**inputs)
  self.assertEqual(r['adjusted_point_fractions'],{'hotkey-0':[1,2],'hotkey-1':[1,1]});self.assertEqual(r['duplicate_coverage'],'incomplete');self.assertTrue(r['unchecked_duplicate_claims_unresolved'])
  self.assertTrue(r['epoch_id'].startswith('live-'));self.assertTrue(r['compute_epoch_id'].startswith('nonpayable-'));self.assertIs(inputs['score_document']['payload']['payable'],False)
  self.assertEqual(hourly_points([inputs['score_document']['payload']],7200),{})
  result=b.hourly_reward_units([sign(r,key)],inputs['authority'],7200,fresh_registrations=regs);self.assertEqual(result['points'],{'hotkey-0':500000,'hotkey-1':1000000});self.assertEqual(result['points']['hotkey-1']/sum(result['points'].values()),2/3)
 def test_bad_signature_and_historical_contract_added_later_refuse(self):
  key,i,regs,r=fixture();bad=copy.deepcopy(i);bad['manifest_document']['payload']['deadline']+=1
  with self.assertRaises(Exception):b.project(**bad)
  for epoch,start in [('nonpayable-separated-hopper-historical',3601),('nonpayable-live-reward-math-v1-HISTORICAL',3599),('live-loose-prefix',3601)]:
   bad=copy.deepcopy(i);m=bad['manifest_document']['payload'];m.update(epoch=epoch,start=start);m['live_reward_contract'].update(epoch=epoch,starts_at=start);bad['manifest_document']=sign(m,key)
   with self.assertRaises(ValueError):b.project(**bad)
 def test_immutable_opening_and_source_checkpoint_receipt_mutations_refuse(self):
  key,i,regs,r=fixture()
  for field,value in [('first_manifest_sha256','f'*64),('epoch','OTHER'),('published_at',4000)]:
   bad=copy.deepcopy(i);bad['opening_document']['payload'][field]=value;bad['opening_document']=sign(bad['opening_document']['payload'],key)
   with self.assertRaises(ValueError):b.project(**bad)
  bad=copy.deepcopy(i);bad['score_document']['payload']['receipts'][next(iter(r))]['sha256']='f'*64;bad['score_document']=sign(bad['score_document']['payload'],key)
  with self.assertRaises(ValueError):b.project(**bad)
 def test_unaudited_infra_no_reward_and_penalty_tamper_refuse(self):
  key,i,regs,reports=fixture();first=next(iter(reports));reports[first]['outcomes'][0]['fully_audited']=False;i['audit_documents'][first]=sign(reports[first],key)
  with self.assertRaisesRegex(ValueError,'fully audited'):b.project(**i)
  key,i,regs,reports=fixture();i['score_document']['payload']['adjusted_points'][next(iter(reports))]=999;i['score_document']=sign(i['score_document']['payload'],key)
  with self.assertRaisesRegex(ValueError,'adjusted_points'):b.project(**i)
 def test_known_collision_zero_unknown_collision_disclosed(self):
  key,i,regs,reports=fixture();keys=list(reports);reports[keys[1]]['accepted'][0]=copy.deepcopy(reports[keys[0]]['accepted'][0]);reports[keys[1]]['outcomes'][0]['index']=0
  scores=i['score_document']['payload'];scores.update(score(reports,scores['penalty_policy']));i['score_document']=sign(scores,key);i['audit_documents']={k:sign(v,key) for k,v in reports.items()};r=b.project(**i);self.assertEqual(r['raw_unique_observed_points'],{'hotkey-0':0,'hotkey-1':0});self.assertTrue(r['unchecked_duplicate_claims_unresolved'])
 def test_hourly_duplicate_UID_recycling_fraction_policy_refusal(self):
  key,i,regs,reports=fixture();r=b.project(**i);doc=sign(r,key)
  with self.assertRaisesRegex(ValueError,'duplicate reward epoch'):b.hourly_reward_units([doc,doc],i['authority'],7200,fresh_registrations=regs)
  bad=copy.deepcopy(regs);bad['hotkey-0']['uid']=222
  with self.assertRaisesRegex(ValueError,'stale registration'):b.hourly_reward_units([doc],i['authority'],7200,fresh_registrations=bad)
  old=dict(r,epoch_id='nonpayable-HISTORY')
  with self.assertRaises(ValueError):b.hourly_reward_units([sign(old,key)],i['authority'],7200,fresh_registrations=regs)
 def test_explicit_eligibility_excludes_missing_miner_without_changing_ledger(self):
  key,i,regs,reports=fixture();doc=sign(b.project(**i),key);original=b.canonical(doc)
  fresh={'hotkey-1':regs['hotkey-1']}
  out=b.hourly_reward_units([doc],i['authority'],7200,fresh_registrations=fresh,stale_policy='exclude-ineligible-v1')
  self.assertEqual(out['points'],{'hotkey-1':1000000});self.assertEqual(out['registrations'],{'hotkey-1':doc['payload']['identities']['hotkey-1']})
  self.assertEqual(out['excluded_ineligible']['hotkey-0']['fractional_points'],[1,2])
  self.assertEqual(out['excluded_ineligible']['hotkey-0']['reason'],'not_registered')
  self.assertEqual(out['eligibility_registration_sha256'],b.sha(fresh));self.assertEqual(b.canonical(doc),original)
 def test_eligibility_recycled_UID_or_changed_key_never_receives_old_points(self):
  key,i,regs,reports=fixture();doc=sign(b.project(**i),key)
  for field,value in [('uid',222),('public_key','f'*64)]:
   fresh=copy.deepcopy(regs);fresh['hotkey-0'][field]=value
   out=b.hourly_reward_units([doc],i['authority'],7200,fresh_registrations=fresh,stale_policy='exclude-ineligible-v1')
   self.assertEqual(out['points'],{'hotkey-1':1000000});self.assertNotIn('hotkey-0',out['registrations'])
   self.assertEqual(out['excluded_ineligible']['hotkey-0']['reason'],'identity_changed')
 def test_eligibility_all_missing_is_empty_and_signed_inputs_still_required(self):
  key,i,regs,reports=fixture();doc=sign(b.project(**i),key)
  out=b.hourly_reward_units([doc],i['authority'],7200,fresh_registrations={},stale_policy='exclude-ineligible-v1')
  self.assertEqual(out['points'],{});self.assertEqual(len(out['excluded_ineligible']),2)
  bad=copy.deepcopy(doc);bad['payload']['adjusted_point_fractions']['hotkey-0']=[99,1]
  with self.assertRaises(Exception):b.hourly_reward_units([bad],i['authority'],7200,fresh_registrations={},stale_policy='exclude-ineligible-v1')
  with self.assertRaisesRegex(ValueError,'known stale'):b.hourly_reward_units([doc],i['authority'],7200,fresh_registrations=regs,stale_policy='unknown')
 def test_eligibility_aggregates_fractional_exclusions_before_rounding(self):
  key,i,regs,reports=fixture();r=b.project(**i);r['adjusted_point_fractions']['hotkey-0']=[1,3000000]
  records=[]
  for index in range(3):
   nextrow=copy.deepcopy(r);nextrow['epoch_id']='live-math-reward-v1-'+str(index);nextrow['contract']['reward_epoch']=nextrow['epoch_id'];records.append(sign(nextrow,key))
  out=b.hourly_reward_units(records,i['authority'],7200,fresh_registrations={'hotkey-1':regs['hotkey-1']},stale_policy='exclude-ineligible-v1')
  self.assertEqual(out['excluded_ineligible']['hotkey-0']['units'],1)
  self.assertEqual(out['excluded_ineligible']['hotkey-0']['fractional_points'],[1,1000000])
 def test_real_exporter_signs_eligibility_report_and_preserves_earned_ledger(self):
  key,i,regs,reports=fixture();doc=sign(b.project(**i),key)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);compute=root/'compute';compute.mkdir();reward=root/'reward';reward.mkdir()
   ledger=reward/'signed-reward-ledger.json';original=b.canonical([doc]);ledger.write_bytes(original)
   out=worker.run_once(compute,reward,i['anchor_document'],i['authority'],key,{'hotkey-1':regs['hotkey-1']},7200,stale_policy='exclude-ineligible-v1')
   signed_proposal=json.loads((reward/'hour-7200-reward-units.json').read_text())
   self.assertEqual(b.signed(signed_proposal,i['authority']),out)
   self.assertEqual(out['points'],{'hotkey-1':1000000});self.assertEqual(ledger.read_bytes(),original)
   self.assertEqual(out['excluded_ineligible']['hotkey-0']['units'],500000)
 def test_single_writer_status_unknown_enabled_zombie_and_stale_refuse(self):
  key,i,regs,reports=fixture();receipt=dict(version='single-live-reward-writer-v1',netuid=120,boot_id='BOOT',writer_pid=11,writer_start_ticks='12',observed_at=100,global_writer_lock_held=True,legacy_validator_guard_verified=True,writer_process_state='S',old_writers=[dict(unit=u,running=False,enabled=False,status_query_succeeded=True) for u in ('affine-transition-weights.timer','affine-transition-weights.service','affine-hourly-burn.timer','affine-hourly-burn.service')])
  def gate(r):return b.writer_gate(sign(r,key),i['authority'],now=101,boot_id='BOOT',writer_pid=11,writer_ticks='12')
  gate(receipt)
  for field,value in [('writer_process_state','Z'),('writer_process_state','T'),('old_writers',[]),('writer_pid',99),('observed_at',1),('global_writer_lock_held',False),('legacy_validator_guard_verified',False)]:
   bad=copy.deepcopy(receipt);bad[field]=value
   with self.assertRaises(ValueError):gate(bad)
  for change in [{'running':True},{'enabled':True},{'status_query_succeeded':False}]:
   bad=copy.deepcopy(receipt);bad['old_writers'][0].update(change)
   with self.assertRaises(ValueError):gate(bad)
 def test_zero_threshold_and_fraction_accumulate_before_rounding(self):
  key,i,regs,reports=fixture();r=b.project(**i);r['adjusted_point_fractions']={'hotkey-0':[1,3000000],'hotkey-1':[0,1]};r2=copy.deepcopy(r);r2['epoch_id']='live-math-reward-v1-SECOND';r2['contract']['reward_epoch']=r2['epoch_id'];r3=copy.deepcopy(r);r3['epoch_id']='live-math-reward-v1-THIRD';r3['contract']['reward_epoch']=r3['epoch_id']
  out=b.hourly_reward_units([sign(x,key) for x in [r,r2,r3]],i['authority'],7200,fresh_registrations=regs);self.assertEqual(out['points']['hotkey-0'],1)
 def test_adapter_handoff_uses_adjusted_units_and_refuses_bad_writer_before_adapter(self):
  key,i,regs,reports=fixture();r=b.project(**i);hour=b.hourly_reward_units([sign(r,key)],i['authority'],7200,fresh_registrations=regs)
  spec=importlib.util.spec_from_file_location('submit_reward_hour',OUT/'live_reward_submit.py');submit=importlib.util.module_from_spec(spec);spec.loader.exec_module(submit)
  from unittest.mock import Mock
  adapter=Mock();submit.submit_hour(adapter,sign(hour,key),i['authority'],None,now=7201,boot_id='BOOT',writer_pid=1,writer_ticks='2',execute=False)
  self.assertEqual(adapter.submit_hour.call_args.args[0]['hotkey-0'],500000);self.assertIs(adapter.submit_hour.call_args.kwargs['execute'],False)
  adapter.reset_mock()
  with self.assertRaises(ValueError):submit.submit_hour(adapter,sign(hour,key),i['authority'],{'payload':{}},now=7201,boot_id='BOOT',writer_pid=1,writer_ticks='2',execute=True)
  adapter.submit_hour.assert_not_called()
 def test_nonbinary_penalties_share_pure_scorer_and_large_rational_bound(self):
  from fractions import Fraction
  key,i,regs,reports=fixture();penalty=dict(invalid_batch_multiplier=.9,zero_epoch_after=0,penalize_structural=False)
  m=i['manifest_document']['payload'];m['audit_policy']['penalties']=penalty;m['live_reward_contract']['penalties']=penalty
  i['manifest_document']=sign(m,key);i['opening_document']['payload']['first_manifest_sha256']=b.sha(i['manifest_document']);i['opening_document']=sign(i['opening_document']['payload'],key)
  first=next(iter(reports));reports[first]['outcomes']=[reports[first]['outcomes'][0]]+[dict(batch=j+1,valid=False,fully_audited=True,failure_kind='confirmed_invalid') for j in range(32)]
  i['audit_documents']={k:sign(v,key) for k,v in reports.items()};scores=i['score_document']['payload'];scores.update(score(reports,penalty));i['score_document']=sign(scores,key)
  r=b.project(**i);f=Fraction(9,10)**32;self.assertEqual(r['adjusted_point_fractions']['hotkey-0'],[f.numerator,f.denominator]);self.assertGreater(f.numerator,2**63)
  hour=b.hourly_reward_units([sign(r,key)],i['authority'],7200,fresh_registrations=regs);self.assertEqual(hour['points']['hotkey-0'],int(f*1000000))
 def test_hourly_handoff_signature_refuses_before_adapter(self):
  key,i,regs,reports=fixture();r=b.project(**i);hour=b.hourly_reward_units([sign(r,key)],i['authority'],7200,fresh_registrations=regs)
  spec=importlib.util.spec_from_file_location('submit_reward_hour_signature',OUT/'live_reward_submit.py');submit=importlib.util.module_from_spec(spec);spec.loader.exec_module(submit)
  from unittest.mock import Mock
  doc=sign(hour,key);doc['payload']['points']['hotkey-0']+=1;adapter=Mock()
  with self.assertRaises(Exception):submit.submit_hour(adapter,doc,i['authority'],None,now=7201,boot_id='B',writer_pid=1,writer_ticks='1')
  adapter.submit_hour.assert_not_called()
class OpeningIntegration(unittest.TestCase):
 def test_full_audit_live_opening_refuses_before_grants(self):
  key,i,regs,reports=fixture();m=i['manifest_document']['payload'];m['audit_policy']={'mode':'full','version':1}
  with self.assertRaisesRegex(ValueError,'bounded-random-v1 only'):b.contract(m,i['anchor_document']['payload'])
  with self.assertRaisesRegex(ValueError,'bounded-random-v1 only'):b.prevalidate_opening_arguments(m['epoch'],m['checkpoint'],list(m['capabilities']),60,m['audit_policy'],m['source_bundle'],i['anchor_document'],regs,i['authority'])
 def test_bad_anchor_refuses_before_real_Gateway_grants(self):
  bucket=type('Bucket',(),{'json':lambda *a:None})();gateway=type('Gateway',(),{'open':lambda *a,**k:(_ for _ in ()).throw(AssertionError('grant must not occur'))})()
  with tempfile.TemporaryDirectory() as tmp:
   controller=Controller(bucket,gateway,tmp);key=controller.authority.key
   anchor=controller.signed(dict(version='live-reward-cutover-v1',netuid=120,owner_hotkey='OWNER',cutover_id='CPU',effective_at=0,compute_epoch_prefix='nonpayable-live-reward-math-v1-',live_epoch_prefix='live-math-reward-v1-',approved_compute_sources=['a'*64]));anchor['payload']['cutover_id']='UNSIGNED MUTATION'
   with self.assertRaises(Exception):controller.open('nonpayable-live-reward-math-v1-CPU',{'id':'b'*64},[],source_bundle={'sha256':'a'*64},live_reward_anchor_document=anchor,live_reward_registration_snapshot={})
 def test_real_Controller_RemoteController_Gateway_first_signature_and_sidecar_export(self):
  class MemoryBucket:
   def __init__(self):self.objects={}
   def json(self,k,v):self.objects[k]=canonical(v)
   def snapshot(self,k,**kwargs):return None
   def presign(self,k,*a,**kw):return 'https://test.r2.cloudflarestorage.com/'+k+'?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=TEST'
  with tempfile.TemporaryDirectory() as tmp:
   bucket=MemoryBucket();gateway=Gateway.__new__(Gateway);gateway.lock=threading.Lock();gateway.epochs={};gateway.direct_r2=True;gateway.state_path=None;gateway.bucket=bucket;gateway.secret=b'CPU-TEST'
   with patch('subnet.remote_backend.RemoteJobs'):controller=RemoteController(bucket,gateway,Path(tmp)/'compute',{})
   miner=Identity();now=int(time.time());regs={'hk':dict(uid=1,public_key=miner.id,snapshot_block=123)}
   anchor=controller.signed(dict(version='live-reward-cutover-v1',netuid=120,owner_hotkey='OWNER',cutover_id='CPU-TEST',effective_at=now-1,compute_epoch_prefix='nonpayable-live-reward-math-v1-',live_epoch_prefix='live-math-reward-v1-',approved_compute_sources=['a'*64]))
   rev,profile_,policy=profile('cuda-bf16-eager-sm90-v1');spec=dict(id='affine_math',version='prime-v1-1',adapter='prime_v1',config={},max_turns=1,max_output_tokens=256,num_samples=2,success_reward=1.,source_hash='f'*64)
   manifest=controller.open('nonpayable-live-reward-math-v1-CPU-1',{'id':'b'*64,'files':{'config.json':'e'*64}},[miner.id],max_batches=1,duration=60,audit_policy=dict(mode='sampled',version='bounded-random-v1',epoch_budget=2,escalation_budget=2,minimum_per_miner=1,maximum_per_miner=3,penalties=b.penalties()),environments=[dict(spec=spec,indices=[0],harness=dict(version='text-tools-v1',policy='autoregressive',max_output_tokens=256,temperature=.8,top_p=1.))],source_bundle={'sha256':'a'*64,'size':1},model_runtime_revision=rev,backend_profile=profile_,numerical_policy=policy,live_reward_anchor_document=anchor,live_reward_registration_snapshot=regs)
   first=json.loads(bucket.objects['public/'+manifest['epoch']+'/manifest.json']);self.assertEqual(first['payload'],manifest);self.assertIs(manifest['payable'],False);self.assertIs(manifest['live_reward_contract']['payable'],True);self.assertEqual(manifest['max_batches'],1);self.assertEqual(miner.decrypt(manifest['capabilities'][miner.id])['transport'],'direct-r2-v1')
   stored=json.loads((controller.state/(manifest['epoch']+'-first-signed-manifest.json')).read_text());self.assertEqual(stored,first)
   # Metadata-only empty score export through actual hook; no fake audit/inference.
   result=dict(score({},b.penalties()),payable=False,epoch_id=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],finalized_at=manifest['deadline'],receipts={})
   from types import SimpleNamespace
   controller.jobs=SimpleNamespace()
   with patch('subnet.remote_backend.time.time',return_value=manifest['deadline']+1):
    actual,reports=controller.finalize(manifest,Path(tmp)/'unused-checkpoint')
   self.assertEqual(reports,{})
   self.assertTrue((controller.state/(manifest['epoch']+'-signed-compute-scores.json')).exists())
   self.assertIs(actual['payable'],False)
   controller.finalize(manifest,Path(tmp)/'unused-checkpoint')
   opening_path=controller.state/(manifest['epoch']+'-opening-attestation.json');original=opening_path.read_bytes()
   with patch('time.time',return_value=manifest['deadline']+100):b.emit_opening_documents(controller,manifest,regs)
   self.assertEqual(opening_path.read_bytes(),original)
   for field,value in [('epoch','OTHER'),('first_manifest_sha256','0'*64),('published_at',manifest['deadline'])]:
    o=json.loads(original);o['payload'][field]=value;opening_path.write_text(json.dumps(controller.signed(o['payload'])))
    with self.assertRaises(ValueError):b.emit_opening_documents(controller,manifest,regs)
   opening_path.write_bytes(original)
   # Execute actual GPU opening-branch prefix on resumed manifest, before discovery.
   (controller.state/(manifest['epoch']+'-signed-registrations.json')).unlink()
   import ast
   from subnet import gpu_service
   tree=ast.parse(Path(gpu_service.__file__).read_text())
   branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="active['phase'] == 'opening'")
   body=ast.Module(body=[ast.For(target=ast.Name(id='_',ctx=ast.Store()),iter=ast.List(elts=[ast.Constant(0)],ctx=ast.Load()),body=branch.body[:3],orelse=[])],type_ignores=[])
   namespace=dict(__name__='subnet.gpu_service',__package__='subnet',manifestpath=controller.state/(manifest['epoch']+'-manifest.json'),json=json,epoch=manifest['epoch'],config={'max_batches':1},bucket=bucket,controller=controller,active={'registrations':regs})
   with patch('time.time',return_value=manifest['deadline']+100):exec(compile(ast.fix_missing_locations(body),gpu_service.__file__,'exec'),namespace)
   self.assertEqual(bucket.objects['public/'+manifest['epoch']+'/manifest.json'],canonical(first))
   self.assertEqual(opening_path.read_bytes(),original)
   opening_path.unlink()
   with patch('time.time',return_value=manifest['deadline']+100):
    with self.assertRaisesRegex(ValueError,'expired missing opening'):b.emit_opening_documents(controller,manifest,regs)
   opening_path.write_bytes(original)
   key=controller.authority.key;end=(manifest['deadline']//3600+1)*3600
   out=worker.run_once(controller.state,Path(tmp)/'reward',anchor,controller.authority.id,key,regs,end);self.assertEqual(out['points'],{});self.assertEqual(len(out['source_reward_records']),1)
   worker.run_once(controller.state,Path(tmp)/'reward',anchor,controller.authority.id,key,regs,end);ledger=json.loads((Path(tmp)/'reward/signed-reward-ledger.json').read_text());self.assertEqual(len(ledger),1)
if __name__=='__main__':unittest.main()
