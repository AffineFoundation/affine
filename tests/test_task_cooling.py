import copy,json,tempfile,unittest,base64
from nacl.signing import SigningKey
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from ops import task_cooling as m

POOL=[1553,7121]+[i for i in range(6744)if i!=1553]+[7495]
assert len(POOL)==6746 and len(set(POOL))==6746
P=dict(version=m.VERSION,enabled=False,first_round=154,history_epochs=12,cooldown_count=256,minimum_allowed=4096,env_id='affine_math',epoch_prefix='nonpayable-live-reward-math-v1-',approved_indices_sha256=m.S(POOL),calibration_anchor_indices=[1553,7121],tie_break_domain=m.DOMAIN)
EPOCH=P['epoch_prefix']+'-1791766800-154'
ORIGINAL=dict(environments=[dict(env_id='affine_math',spec=dict(id='affine_math'),indices=POOL,harness=dict(version='text-tools-long-v2',max_output_tokens=2048))],sample_harness_registry=dict(affine_math=dict(indices=POOL,harness=dict(version='text-tools-long-v2',max_output_tokens=2048))),K=4,L=4,max_batches=9,source_bundle=dict(sha256='a'*64),sampling_policy=dict(version='unchanged'),training_task_capacity=dict(max_tasks=512))
def refs(n):return {k:dict(path='/private/'+str(n)+'/'+k+'.json',sha256=m.S([n,k]))for k in ('closure','job','report')}
def evidence():
 return dict(version='finalized-trained-task-exposure-input-v1',assurance='signed-closure-and-job-linked-unsigned-trainer-report',records=[dict(round=n,epoch=P['epoch_prefix']+'-1790000000-'+str(n),input_checkpoint='a'*64,output_checkpoint='b'*64,updates=[dict(global_optimizer_step=n-90,indices=POOL[:1000])],provenance=refs(n))for n in range(142,154)])
def projection(harness,allowed,approved):
 assert set(allowed)<=set(approved);return copy.deepcopy(harness)
KEY=SigningKey(bytes(range(32)))
def sign(value):return dict(payload=value,signer=KEY.verify_key.encode().hex(),signature=base64.b64encode(KEY.sign(m.C(value)).signature).decode())
def verify(envelope):
 if envelope['signer']!=KEY.verify_key.encode().hex():raise ValueError('snapshot signer')
 KEY.verify_key.verify(m.C(envelope['payload']),base64.b64decode(envelope['signature']));return envelope['payload']
def enabled(**kw):return dict(P,enabled=True,**kw)
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.config=dict(state=str(self.root));self.state=dict(round=154,active=dict(epoch=EPOCH,phase='opening'));self.save_state()
 def tearDown(self):self.tmp.cleanup()
 def save_state(self): (self.root/'controller.json').write_bytes(m.C(self.state))
 def service(self,p=None,provider=None,original=None):
  s=SimpleNamespace(contract=lambda c,r:copy.deepcopy(original or ORIGINAL));reader=provider or Mock(return_value=evidence());m.install(s,p or enabled(),reader,project_harness=projection,sign_snapshot=sign,verify_snapshot=verify);return s,reader
 def snapshot(self):return self.root/'task-cooling-v1'/(EPOCH+'.json')
 def test_widths(self):
  for n in (0,256,512):
   x=m.select(POOL,enabled(cooldown_count=n),154,EPOCH,evidence());self.assertEqual(len(x['cooled_indices']),n);self.assertEqual(len(x['allowed_indices']),6746-n);self.assertEqual(x['allowed_indices'][:2],[1553,7121]);self.assertFalse(set(x['cooled_indices'])&{1553,7121})
 def test_count_task_once_per_update(self):
  e=evidence();e['records'][0]['updates'].append(dict(global_optimizer_step=999,indices=[POOL[2]]));x=m.select(POOL,enabled(),154,EPOCH,e);self.assertEqual(x['cooled_indices'][0],POOL[2]);self.assertEqual(x['cooled_exposures'][0]['updates'],13)
 def test_ties_are_deterministic_epoch_bound(self):
  a=m.select(POOL,enabled(),154,EPOCH,evidence());b=m.select(POOL,enabled(),154,EPOCH,evidence());c=m.select(POOL,enabled(),155,EPOCH[:-3]+'155',evidence());self.assertEqual(a,b);self.assertNotEqual(a['cooled_indices'],c['cooled_indices'])
 def test_large_floor(self):
  x=m.select(POOL,enabled(cooldown_count=512,minimum_allowed=6600),154,EPOCH,evidence());self.assertEqual(len(x['allowed_indices']),6600)
 def test_default_off_no_state_or_reader(self):
  s,r=self.service(p=P);self.assertEqual(s.contract(self.config,154),ORIGINAL);r.assert_not_called();self.assertFalse((self.root/'task-cooling-v1').exists())
 def test_prior_round_bypasses(self):
  s,r=self.service();self.assertEqual(s.contract(self.config,153),ORIGINAL);r.assert_not_called()
 def test_only_projected_indices_change(self):
  s,r=self.service();x=s.contract(self.config,154);expected=copy.deepcopy(ORIGINAL);expected['environments'][0]['indices']=x['environments'][0]['indices'];self.assertEqual(x,expected);self.assertEqual(len(x['environments'][0]['indices']),6490);r.assert_called_once()
 def test_snapshot_restart_does_not_read_new_exposure(self):
  s,r=self.service();a=s.contract(self.config,154);r2=Mock(side_effect=AssertionError('must not reread'));s2,_=self.service(provider=r2);self.assertEqual(s2.contract(self.config,154),a);r2.assert_not_called()
 def test_provider_failure_records_original_fallback(self):
  s,r=self.service(provider=Mock(side_effect=ValueError('ambiguous recovery')));self.assertEqual(s.contract(self.config,154),ORIGINAL);v=verify(json.loads(self.snapshot().read_bytes()));self.assertIn('ValueError',v['selection']['fallback_reason']);s2,r2=self.service();self.assertEqual(s2.contract(self.config,154),ORIGINAL);r2.assert_not_called()
 def test_bad_history_full_pool_not_stall(self):
  e=evidence();e['records'].pop();s,r=self.service(provider=Mock(return_value=e));self.assertEqual(s.contract(self.config,154),ORIGINAL)
 def test_open_future_ambiguous_records_refused(self):
  for mutate in [lambda e:e['records'][-1].update(round=154),lambda e:e['records'][-1]['updates'][0]['indices'].append(POOL[0]),lambda e:e['records'][-1]['updates'][0]['indices'].append(99999),lambda e:e.update(assurance='signed-report-not-actually-signed')]:
   e=evidence();mutate(e)
   with self.assertRaises(ValueError):m.select(POOL,enabled(),154,EPOCH,e)
 def test_zero_count_avoids_evidence(self):
  s,r=self.service(p=enabled(cooldown_count=0));self.assertEqual(s.contract(self.config,154),ORIGINAL);r.assert_not_called();self.assertEqual(verify(json.loads(self.snapshot().read_bytes()))['selection']['fallback_reason'],'configured_zero_cooldown')
 def test_changed_policy_cannot_rewrite_snapshot(self):
  s,_=self.service();s.contract(self.config,154);before=self.snapshot().read_bytes();s2,_=self.service(p=enabled(cooldown_count=512))
  with self.assertRaisesRegex(ValueError,'immutable original'):s2.contract(self.config,154)
  self.assertEqual(self.snapshot().read_bytes(),before)
 def test_issued_without_snapshot_refuses_before_evidence(self):
  (self.root/(EPOCH+'-manifest.json')).write_text('{}');s,r=self.service()
  with self.assertRaisesRegex(ValueError,'issued epoch'):s.contract(self.config,154)
  r.assert_not_called()
 def test_issued_original_snapshot_reuses(self):
  s,_=self.service();v=s.contract(self.config,154);(self.root/(EPOCH+'-manifest.json')).write_text('{}');s2,r2=self.service();self.assertEqual(s2.contract(self.config,154),v);r2.assert_not_called()
 def test_tampered_selection_refused(self):
  s,_=self.service();s.contract(self.config,154);v=json.loads(self.snapshot().read_bytes());v['payload']['selection']['allowed_indices'].append(v['payload']['selection']['cooled_indices'][0]);self.snapshot().chmod(0o600);self.snapshot().write_bytes(m.C(v))
  with self.assertRaises(Exception):s.contract(self.config,154)
 def test_changed_original_contract_refused(self):
  s,_=self.service();s.contract(self.config,154);o=copy.deepcopy(ORIGINAL);o['K']=5;s2,_=self.service(original=o)
  with self.assertRaisesRegex(ValueError,'immutable original'):s2.contract(self.config,154)
 def test_concurrent_original_publish_one_snapshot(self):
  services=[self.service()[0]for _ in range(2)]
  with ThreadPoolExecutor(2)as pool:out=list(pool.map(lambda s:s.contract(self.config,154),services))
  self.assertEqual(out[0],out[1]);self.assertEqual(len(list((self.root/'task-cooling-v1').glob('*.json'))),1)
 def test_calibration_anchor_change_refused(self):
  p=enabled();p['calibration_anchor_indices']=[7121,1553];s,_=self.service(p=p)
  with self.assertRaisesRegex(ValueError,'calibration'):s.contract(self.config,154)
 def test_small_pool_policy_refused(self):
  with self.assertRaisesRegex(ValueError,'large minimum'):m.policy(enabled(minimum_allowed=32))
 def test_only_expiring_source_url_can_refresh(self):
  o=copy.deepcopy(ORIGINAL);o['source_bundle'].update(size=123,url='https://original.example/source?time=1');s,_=self.service(original=o);a=s.contract(self.config,154)
  o2=copy.deepcopy(o);o2['source_bundle']['url']='https://original.example/source?time=2';s2,r2=self.service(original=o2);b=s2.contract(self.config,154)
  self.assertEqual(a['environments'],b['environments']);self.assertNotEqual(a['source_bundle']['url'],b['source_bundle']['url']);r2.assert_not_called()
  for field,value in [('sha256','b'*64),('size',124)]:
   changed=copy.deepcopy(o2);changed['source_bundle'][field]=value;s3,_=self.service(original=changed)
   with self.assertRaisesRegex(ValueError,'immutable original'):s3.contract(self.config,154)
 def test_enabled_requires_signed_snapshot_hooks(self):
  with self.assertRaisesRegex(ValueError,'authenticated snapshot hooks'):m.install(SimpleNamespace(contract=lambda c,r:ORIGINAL),enabled(),Mock(),project_harness=projection)
 def test_environment_id_without_spec_has_no_eager_access(self):
  self.assertEqual(m.environment_id({'env_id':'affine_math'}),'affine_math')
  with self.assertRaisesRegex(ValueError,'same environment'):m.environment_id({'env_id':'affine_math','spec':{'id':'other'}})
 def test_correctly_resigned_corrupt_selection_still_refused(self):
  s,_=self.service();s.contract(self.config,154);v=verify(json.loads(self.snapshot().read_bytes()));v['selection']['allowed_indices'].append(v['selection']['cooled_indices'][0]);self.snapshot().chmod(0o600);self.snapshot().write_bytes(m.C(sign(v)))
  with self.assertRaisesRegex(ValueError,'reproduce'):s.contract(self.config,154)
 def test_wrong_signer_winner_refused(self):
  s,_=self.service();s.contract(self.config,154);v=json.loads(self.snapshot().read_bytes());v['signer']='0'*64;self.snapshot().chmod(0o600);self.snapshot().write_bytes(m.C(v))
  with self.assertRaisesRegex(ValueError,'snapshot signer'):s.contract(self.config,154)
if __name__=='__main__':unittest.main()
