"""Independent CPU integration; no GPU qualification or activation claims."""
import unittest,tempfile,copy,json,importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from subnet import successor_calibration as c, forced_sampling as fs
spec=importlib.util.spec_from_file_location('legacy_fixture',Path(__import__('subnet').__file__).parent.parent/'tests/test_successor_calibration.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
class Controls(unittest.TestCase):
 def setUp(self):
  fixture=old.Controls();fixture.setUp();self.h=fixture.h;self.base=fixture.result;self.m=fixture.m
  self.policy=c.admitted_policy(self.base,self.m,fixture.req)
  self.sampling=dict(version=fs.MINER_VERSION,max_attempts=1000,support_adjudication='exact-cached-replay-v1',calibration=self.policy)
  self.req=dict(version=c.MINER_CALIBRATION_VERSION,env_id='math',harness=self.h,task_indices=[1,2],max_tokens=self.h['max_output_tokens'],draw_contract=fs.new_contract(self.sampling),miner='5'*64)
  self.manifest=dict(self.m,epoch='nonpayable-v5-qualification',K=2,L=2,max_batches=3)
 def result(self,manifest,req):
  reports=[dict(x,checkpoint=manifest['checkpoint']['id'])for x in self.base['reports']]
  return dict(self.base,version=c.MINER_CALIBRATION_VERSION,checkpoint=manifest['checkpoint']['id'],reports=reports,request_sha256=c.digest(req),sampling_miner=req['miner'],sampling_context_sha256=c.digest(c.draw_context(manifest,req)))
 def test_v5_context_changes_with_miner_and_checkpoint(self):
  a=c.draw_context(self.manifest,self.req);b=c.draw_context(self.manifest,dict(self.req,miner='6'*64));d=c.draw_context(dict(self.manifest,checkpoint={'id':'b'*64}),self.req)
  self.assertNotEqual(fs.uniform(a,'math','f'*64,1,0,0,0),fs.uniform(b,'math','f'*64,1,0,0,0));self.assertNotEqual(fs.uniform(a,'math','f'*64,1,0,0,0),fs.uniform(d,'math','f'*64,1,0,0,0))
 def test_context_wrong_geometry_rejected(self):
  for m in [dict(self.manifest,K=1),dict(self.manifest,max_batches=4)]:
   with self.assertRaises(ValueError):c.draw_context(m,self.req)
 def test_eight_rollout_calibration_uses_same_bound_draw_recipe(self):
  m=dict(self.manifest,K=4,L=4)
  self.assertEqual(c.draw_context(m,self.req),c.draw_context(self.manifest,self.req))
  result=self.result(m,self.req)
  self.assertEqual(c.admitted_policy(result,m,self.req)['checkpoint'],'a'*64)
 def test_result_relabel_miner_checkpoint_or_legacy_rejected(self):
  r=self.result(self.manifest,self.req);self.assertEqual(c.admitted_policy(r,self.manifest,self.req)['checkpoint'],'a'*64)
  for bad in [dict(r,sampling_miner='6'*64),dict(r,checkpoint='b'*64),dict(r,version=c.VERSION),dict(r,sampling_context_sha256='f'*64)]:
   with self.assertRaises(ValueError):c.admitted_policy(bad,self.manifest,self.req)
 def test_prefill_and_native_draws_share_exact_bound_context(self):
  import torch
  class Model(torch.nn.Module):
   def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.zeros(1))
   def forward(self,tokens,**kw):return SimpleNamespace(logits=torch.tensor([[[10.,0.,0.]]]),past_key_values=None)
  runtime=SimpleNamespace(spec=SimpleNamespace(id='math',config={}),harness=self.h,model=Model(),tokenizer=SimpleNamespace(eos_token_id=0),prompt=lambda *x:[1,2],compute=lambda *x:([],[]),build_proofs=lambda *x,**kw:[],verify_proofs=lambda *x:[SimpleNamespace(exp_mismatches=0,mant_err_mean=0.,mant_err_median=0.)])
  session=SimpleNamespace(reset=lambda i,s:dict(messages=[],task_hash='f'*64),close=lambda:None);contexts=[];draws=[];original=fs.uniform
  def measure(rt,*a,**kw):contexts.append(copy.deepcopy(rt.sampling_context));return dict(self.base['reports'][0])
  def uniform(ctx,*a):draws.append(copy.deepcopy(ctx));return original(ctx,*a)
  with patch('subnet.environments.create_session',return_value=session),patch('subnet.fast_prefill_audit.measure_cached_prefill',side_effect=measure),patch('subnet.forced_sampling.uniform',side_effect=uniform):r=c.execute(runtime,self.manifest,self.req)
  expected=c.draw_context(self.manifest,self.req);self.assertEqual(contexts,[expected,expected]);self.assertTrue(draws);self.assertTrue(all(x==expected for x in draws));self.assertEqual(r['sampling_context_sha256'],c.digest(expected));self.assertEqual(r['sampling_miner'],'5'*64)
 def test_before_open_new_checkpoint_and_retry_retains_exact_jobs(self):
  with tempfile.TemporaryDirectory()as tmp:
   opening=dict(environments=[dict(spec={'id':'math'},indices=[1,2],harness=self.h)],source_bundle={'sha256':'c'*64},model_runtime_revision='runtime',backend_profile={'dtype':'float32'},K=2,L=2,max_batches=3)
   config=dict(sampling_policy=self.sampling,owned_miner_identity_files={'5'*64:'/never-read.seed'},successor_calibration={'version':c.VERSION,'env_id':'math'});seen=[]
   def run(label,role,manifest,cache,**fields):
    req=fields['successor_calibration'];seen.append(copy.deepcopy((label,manifest,req)));return dict(job_id=label,successor_calibration=self.result(manifest,req))
   ctrl=SimpleNamespace(state=Path(tmp),jobs=SimpleNamespace(run=run),checkpoint_with_reads=lambda v:v)
   status={'checkpoint':{'id':'a'*64,'files':{'x':'d'*64}}};a=c.before_open(ctrl,config,status,opening);b=c.before_open(ctrl,config,status,opening)
   self.assertEqual(a,b);self.assertEqual(seen[0],seen[2]);self.assertEqual(seen[1],seen[3]);self.assertEqual(seen[0][2]['miner'],'5'*64);self.assertEqual(seen[0][2]['draw_contract']['max_attempts'],1000)
   status['checkpoint']['id']='b'*64;new=c.before_open(ctrl,config,status,opening);self.assertNotEqual(seen[0][0],seen[4][0]);self.assertEqual(new['sampling_policy']['calibration']['checkpoint'],'b'*64);self.assertEqual(len(list((Path(tmp)/'successor-calibration').glob('*.json'))),2)
 def test_missing_owned_identity_prevents_dispatch(self):
  with tempfile.TemporaryDirectory()as tmp:
   opening=dict(environments=[dict(spec={'id':'math'},indices=[1,2],harness=self.h)],K=2,L=2,max_batches=3)
   with self.assertRaisesRegex(ValueError,'owned miner'):c.before_open(SimpleNamespace(state=Path(tmp)),dict(sampling_policy=self.sampling,successor_calibration={'version':c.VERSION,'env_id':'math'}),{'checkpoint':{'id':'a'*64}},opening)
if __name__=='__main__':unittest.main()
