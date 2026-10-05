import copy,unittest
from types import SimpleNamespace
from unittest.mock import Mock
from test_committed_training_inputs import LearnerAdmissionTests
from test_continuous_audit_policy import signed
from subnet import continuous_reward_bridge as b
from subnet.continuous_audit_service import register_population
from subnet.continuous_audit_policy import snapshot,hourly_aggregate,VERSION

class BridgeControls(unittest.TestCase):
 def setUp(self):
  fx=LearnerAdmissionTests();fx.setUp();self.addCleanup(fx.doCleanups);self.key=fx.operator;self.authority=fx.authority;self.identity=fx.identity
  self.sign=lambda p:signed(self.key,p)
  self.anchor=self.sign(dict(version=b.ANCHOR,netuid=120,owner_hotkey='owner',activation_id='new-statistical-policy',effective_at=3600,approved_sources=[fx.manifest['source_bundle']['sha256']],units_per_point=b.SCALE,rounding='floor-after-hour-aggregation'))
  self.regs={'miner-hotkey':dict(uid=85,public_key=self.identity,snapshot_block=123)}
  m=copy.deepcopy(fx.manifest);m.update(start=3601,deadline=3900,payable=False,capabilities={self.identity:'encrypted-only'})
  self.manifest=b.inject_opening(m,self.anchor,self.authority,self.regs)
  self.md=self.sign(self.manifest)
  original=fx.obj['learner_admission']['payload']['original_commitment'];receipt=dict(commitment_document=original,sha256=b.sha(original))
  pairs=[dict(miner=self.identity,commitment_sha256=b.sha(original),batch_sha256=fx.obj['learner_admission']['payload']['batch_sha256'],proof_sha256=fx.obj['learner_admission']['payload']['proof_sha256'])]
  self.pop=self.sign(register_population(self.md,{self.identity:receipt},4,3901,self.authority,eligible_pairs=pairs))
  policy=dict(version=VERSION,recent_epochs=4,decay=.8,prior_alpha=1,prior_beta=1,invalid_multiplier=.25,zero_epoch_after=3,blacklist_after=4,blacklist_epochs=2)
  snap=snapshot(self.pop['payload']['records'],[],{},epoch=m['epoch'],round=4,checkpoint=m['checkpoint']['id'],cutoff=7200,audit_policy=policy,eligible_evidence_ids=self.pop['payload']['eligible_evidence_ids'])
  self.snap=self.sign(snap);self.hour=self.sign(hourly_aggregate([self.snap],self.authority,7200));epoch=m['epoch']
  self.openings={epoch:self.sign(dict(version='immutable-first-manifest-v1',epoch=epoch,first_manifest_sha256=b.sha(self.md),published_at=3602))}
  self.regdocs={epoch:self.sign(dict(epoch=epoch,snapshot_block=123,registrations=self.regs))}
  self.completions={epoch:self.sign(dict(epoch=epoch,round=4,checkpoint=m['checkpoint']['id'],next_checkpoint='f'*64,completed_at=4100,input_assurance='unaudited'))}
 def project(self):return b.project(self.hour,[self.snap],[self.pop],self.openings,self.regdocs,self.completions,self.anchor,self.authority,fresh_registrations=self.regs)
 def test_authenticated_statistical_units_are_not_verified_receipts(self):
  result=self.project();self.assertEqual(result['points'],{'miner-hotkey':500000})
  self.assertFalse(result['unaudited_samples_claimed_verified']);self.assertFalse(result['chain_executed']);self.assertNotIn('audits',result)
 def test_tampered_and_resigned_wrong_aggregate_refused(self):
  self.hour['payload']['points'][self.identity]=1
  with self.assertRaises(Exception):self.project()
  self.hour=self.sign(self.hour['payload'])
  with self.assertRaisesRegex(ValueError,'aggregate'):self.project()
 def test_historical_opening_cannot_be_promoted(self):
  self.manifest['start']=3599;self.pop['payload']['manifest_document']=self.sign(self.manifest);self.pop=self.sign(self.pop['payload'])
  epoch=self.manifest['epoch'];self.openings[epoch]=self.sign(dict(version='immutable-first-manifest-v1',epoch=epoch,first_manifest_sha256=b.sha(self.pop['payload']['manifest_document']),published_at=3599))
  with self.assertRaises(ValueError):self.project()
 def test_missing_prospective_opening_contract_refused(self):
  self.manifest.pop('continuous_reward_contract');self.pop['payload']['manifest_document']=self.sign(self.manifest);self.pop=self.sign(self.pop['payload'])
  epoch=self.manifest['epoch'];self.openings[epoch]=self.sign(dict(version='immutable-first-manifest-v1',epoch=epoch,first_manifest_sha256=b.sha(self.pop['payload']['manifest_document']),published_at=3602))
  with self.assertRaisesRegex(ValueError,'contract'):self.project()
 def test_no_live_training_pending_or_outside_hour_rewards(self):
  epoch=self.manifest['epoch']
  for value in (3600,7201):
   self.completions[epoch]['payload']['completed_at']=value;self.completions[epoch]=self.sign(self.completions[epoch]['payload'])
   with self.assertRaisesRegex(ValueError,'window'):self.project()
 def test_stale_hotkey_uid_denies_without_renormalization(self):
  self.regs=copy.deepcopy(self.regs);self.regs['miner-hotkey']['uid']=86
  with self.assertRaisesRegex(ValueError,'stale'):self.project()
 def test_owner_or_duplicate_uid_opening_refused(self):
  m=copy.deepcopy(self.manifest);m.pop('continuous_reward_contract')
  with self.assertRaisesRegex(ValueError,'owner'):b.inject_opening(m,self.anchor,self.authority,{'owner':next(iter(self.regs.values()))})
 def test_missing_population_or_eligibility_refused(self):
  self.pop['payload']['eligible_evidence_ids']=[];self.pop=self.sign(self.pop['payload'])
  with self.assertRaisesRegex(ValueError,'eligibility'):self.project()
 def test_submission_dryrun_and_execute_require_single_writer(self):
  doc=self.sign(self.project());adapter=SimpleNamespace(submit_hour=Mock(return_value={'status':'dry_run'}))
  args=dict(now=7201,boot_id='b',writer_pid=1,writer_ticks='1')
  result=b.submit_hour(adapter,doc,self.authority,None,**args)
  self.assertEqual(result['status'],'dry_run');adapter.submit_hour.assert_called_once()
  with self.assertRaises(Exception):b.submit_hour(adapter,doc,self.authority,None,execute=True,**args)
  self.assertEqual(adapter.submit_hour.call_count,1)

class ProductionOpeningControls(unittest.TestCase):
 def test_real_remote_open_publishes_original_contract_and_immutable_sidecars(self):
  import tempfile,threading,time,json
  from pathlib import Path
  from unittest.mock import patch
  from subnet.storage import Gateway,Identity,canonical
  from subnet.remote_backend import RemoteController
  from subnet.backend_profiles import profile
  class Bucket:
   def __init__(self):self.objects={}
   def json(self,k,v):self.objects[k]=canonical(v)
   def snapshot(self,k,**kwargs):return None
   def presign(self,k,*a,**kw):return 'https://test.r2.cloudflarestorage.com/'+k
  with tempfile.TemporaryDirectory()as tmp:
   bucket=Bucket();gateway=Gateway.__new__(Gateway);gateway.lock=threading.Lock();gateway.epochs={};gateway.direct_r2=True;gateway.state_path=None;gateway.bucket=bucket;gateway.secret=b'CPU-TEST'
   with patch('subnet.remote_backend.RemoteJobs'):controller=RemoteController(bucket,gateway,Path(tmp)/'compute',{})
   miner=Identity();regs={'hk':dict(uid=85,public_key=miner.id,snapshot_block=123)};now=time.time()
   activation=controller.signed(dict(version=b.ANCHOR,netuid=120,owner_hotkey='owner',activation_id='new',effective_at=now-1,approved_sources=['a'*64],units_per_point=b.SCALE,rounding='floor-after-hour-aggregation'))
   rev,profile_,policy=profile('cuda-bf16-eager-sm90-v1');spec=dict(id='affine_math',version='prime-v1-1',adapter='prime_v1',config={},max_turns=1,max_output_tokens=256,num_samples=2,success_reward=1.,source_hash='f'*64)
   m=controller.open('nonpayable-new-statistical-epoch',{'id':'b'*64,'files':{'config.json':'e'*64}},[miner.id],max_batches=1,duration=60,environments=[dict(spec=spec,indices=[0],harness=dict(version='text-tools-v1',policy='autoregressive',max_output_tokens=256,temperature=.8,top_p=1.))],source_bundle={'sha256':'a'*64,'size':1},model_runtime_revision=rev,backend_profile=profile_,numerical_policy=policy,submission_transport_policy='small-commitment-pairs-v2',training_input_policy='committed-unaudited-training-v1',training_policy='bf16-full-adamw-covered-fixed-reference-v3',continuous_reward_activation_document=activation,continuous_reward_registration_snapshot=regs)
   first=json.loads((controller.state/(m['epoch']+'-first-signed-manifest.json')).read_bytes())
   self.assertEqual(first['payload'],m);self.assertEqual(bucket.objects['public/'+m['epoch']+'/manifest.json'],canonical(first));self.assertEqual(m['continuous_reward_contract']['version'],b.VERSION)
   path=controller.state/(m['epoch']+'-opening-attestation.json');before=path.read_bytes()
   with patch('time.time',return_value=m['deadline']+100):b.emit_opening_documents(controller,m,regs)
   self.assertEqual(path.read_bytes(),before)
   path.unlink()
   with patch('time.time',return_value=m['deadline']+100):
    with self.assertRaisesRegex(ValueError,'backdated'):b.emit_opening_documents(controller,m,regs)
