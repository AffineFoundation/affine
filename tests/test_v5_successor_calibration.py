"""Real next-checkpoint opening/journals with mocked GPU measurements, CPU only."""
import unittest,json,copy
from types import SimpleNamespace
from unittest.mock import patch
from nacl.signing import VerifyKey
import test_class_quota_opening as opening_fixture
from subnet import successor_calibration as c,fast_prefill_audit as f,forced_sampling as forced
from subnet.harness import normalize
from subnet.backend_profiles import profile,HOPPER_FP32_REVISION
from subnet.storage import canonical
class Controls(unittest.TestCase):
 def setUp(self):
  self.fx=opening_fixture.QuotaOpeningTests();self.fx.setUp();self.addCleanup(self.fx.doCleanups)
  self.row=copy.deepcopy(self.fx.row);self.row['indices']=[0,1];self.rev,self.profile,self.numerical=profile(HOPPER_FP32_REVISION)
  self.h=normalize(self.row['harness']);self.miner=self.fx.miner;self.seen=[]
  cp=self.fx.checkpoint['id'];pred=dict(version=f.CALIBRATION,checkpoint='a'*64,model_runtime_revision=self.rev,backend_profile_sha256=f.digest(self.profile),harness_sha256=f.digest(self.h),report_sha256='c'*64,cdf_abs_error=1e-5,logprob_atol=1e-5,toploc_exp_mismatches=0,toploc_mant_err_mean=0,toploc_mant_err_median=0)
  self.config=dict(sampling_policy=dict(version=forced.MINER_VERSION,max_attempts=1000,calibration=pred,support_adjudication='exact-cached-replay-v1'),owned_miner_identity_files={self.miner:'/not-read'},successor_calibration=dict(version=c.RECALIBRATION_VERSION,env_id='affine_math',max_confirmations=2,deadline_seconds=600))
  self.opening=dict(environments=[self.row],source_bundle=self.fx.config['source_bundle'],model_runtime_revision=self.rev,backend_profile=self.profile,numerical_policy=self.numerical,K=2,L=2,max_batches=3)
  self.status=dict(checkpoint=self.fx.checkpoint)
  self.fx.controller.jobs=SimpleNamespace(run=self.run_job)
  self.fx.controller.checkpoint_with_reads=lambda cp:dict(cp)
 def run_job(self,label,role,manifest,cache,**fields):
  req=fields['successor_calibration'];self.seen.append((label,copy.deepcopy(manifest),copy.deepcopy(req)))
  reports=[dict(version=f.CALIBRATION,checkpoint=manifest['checkpoint']['id'],runtime_revision=self.rev,harness_sha256=f.digest(self.h),used_actual_cached_generation=True,used_actual_teacherforced_prefill=True,actual_model_forwards=9,output_tokens=8,measured_cdf_abs_error=1e-6,measured_logprob_abs_error=1e-6)for _ in req['task_indices']]
  result=dict(version=req['version'],checkpoint=manifest['checkpoint']['id'],request_sha256=c.digest(req),reports=reports,native_controls=[[dict(exp_mismatches=0,mant_err_mean=0.,mant_err_median=0.)]for _ in req['task_indices']],assurance='executed-measurements-not-policy-admission',sampling_miner=req['miner'],sampling_context_sha256=c.digest(c.draw_context(manifest,req)))
  return dict(job_id=label,successor_calibration=result)
 def next_opening(self):return c.before_open(self.fx.controller,self.config,self.status,self.opening)
 def test_new_checkpoint_calibrates_confirms_then_real_signed_K2L2_open(self):
  selected=self.next_opening();self.assertEqual(len(self.seen),2);self.assertEqual(selected['sampling_policy']['calibration']['checkpoint'],self.status['checkpoint']['id']);self.assertNotEqual(self.config['sampling_policy']['calibration']['checkpoint'],self.status['checkpoint']['id'])
  self.assertTrue(all(req['miner']==self.miner and req['draw_contract']['version']==forced.MINER_VERSION and req['draw_contract']['max_attempts']==1000 for _,_,req in self.seen))
  selected=dict(selected);selected.pop('max_batches');manifest=self.fx.controller.open('nonpayable-v5-successor',self.status['checkpoint'],[self.miner],max_batches=3,**selected)
  doc=json.loads(self.fx.bucket.objects['public/nonpayable-v5-successor/manifest.json']);VerifyKey(bytes.fromhex(doc['signer'])).verify(canonical(doc['payload']),__import__('base64').b64decode(doc['signature']));self.assertEqual(doc['payload'],manifest);self.assertEqual((manifest['K'],manifest['L'],manifest['max_batches']),(2,2,3));f.bind(manifest,self.h)
 def test_retry_reobserves_identical_original_requests_and_journal(self):
  first=self.next_opening();calls=copy.deepcopy(self.seen);second=self.next_opening();self.assertEqual(first,second);self.assertEqual(calls,self.seen[2:]);self.assertEqual(len(list((self.fx.controller.state/'successor-calibration').glob('*.json'))),1)
 def test_next_checkpoint_never_reuses_old_journal(self):
  self.next_opening();self.status=copy.deepcopy(self.status);self.status['checkpoint']['id']='d'*64;result=self.next_opening();self.assertEqual(result['sampling_policy']['calibration']['checkpoint'],'d'*64);self.assertNotEqual(self.seen[0][0],self.seen[2][0]);self.assertEqual(len(list((self.fx.controller.state/'successor-calibration').glob('*.json'))),2)
 def test_missing_owned_identity_holds_before_dispatch(self):
  self.config.pop('owned_miner_identity_files')
  with self.assertRaisesRegex(ValueError,'owned miner'):self.next_opening()
  self.assertEqual(self.seen,[])
 def test_request_requires_miner_and_exact1000(self):
  self.next_opening();req=self.seen[0][2]
  for mutation in (dict(miner=True),dict(miner='bad'),dict(draw_contract=dict(req['draw_contract'],max_attempts=128))):
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):c.request(dict(req,**mutation))
  bad=dict(req);bad.pop('miner')
  with self.assertRaises(ValueError):c.request(bad)
 def test_result_wrong_miner_or_context_refuses(self):
  self.next_opening();label,m,req=self.seen[0];result=self.run_job(label,'evaluate',m,None,successor_calibration=req)['successor_calibration']
  for fields in (dict(sampling_miner='e'*64),dict(sampling_context_sha256='f'*64),dict(version=c.VERSION)):
   with self.subTest(fields=fields),self.assertRaises(ValueError):c.admitted_policy(dict(result,**fields),m,req)
 def test_measurement_context_matches_production_formula(self):
  self.next_opening();_,m,req=self.seen[0];context=c.draw_context(m,req);reference=dict(epoch=m['epoch'],checkpoint=m['checkpoint']['id'],contract=req['draw_contract'],miner=self.miner)
  self.assertEqual(context,reference);self.assertEqual(forced.uniform(context,'affine_math','c'*64,0,999,0,0),forced.uniform(reference,'affine_math','c'*64,0,999,0,0))
if __name__=='__main__':unittest.main()
