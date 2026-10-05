import unittest,tempfile
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
from subnet import successor_calibration as c, fast_prefill_audit as f
from subnet.harness import normalize
from subnet.forced_sampling import new_contract,VERSION as STRICT
class Controls(unittest.TestCase):
 def setUp(self):
  self.h=normalize({'version':'text-tools-long-v2','policy':'autoregressive','max_output_tokens':1024,'temperature':.8,'top_p':1})
  self.req=dict(version=c.VERSION,env_id='math',harness=self.h,task_indices=[1,2],max_tokens=1024,draw_contract=new_contract({'version':STRICT,'max_attempts':16}))
  self.m=dict(checkpoint={'id':'a'*64},model_runtime_revision='runtime',backend_profile={'dtype':'float32'})
  self.reports=[dict(version=f.CALIBRATION,checkpoint='a'*64,runtime_revision='runtime',harness_sha256=f.digest(self.h),used_actual_cached_generation=True,used_actual_teacherforced_prefill=True,actual_model_forwards=33,output_tokens=32,measured_cdf_abs_error=1e-6,measured_logprob_abs_error=1e-4)for _ in range(2)]
  self.result=dict(version=c.VERSION,checkpoint='a'*64,request_sha256=c.digest(self.req),reports=self.reports,native_controls=[[dict(exp_mismatches=0,mant_err_mean=0.,mant_err_median=0.)]for _ in range(2)],assurance='executed-measurements-not-policy-admission')
 def test_original_actual_report_bound_policy(self):
  p=c.admitted_policy(self.result,self.m,self.req);self.assertEqual(p['checkpoint'],'a'*64);self.assertEqual(p['cdf_abs_error'],4e-6)
 def test_reject_other_checkpoint_or_missing_native(self):
  for result in [dict(self.result,checkpoint='b'*64),dict(self.result,native_controls=[]),dict(self.result,request_sha256='f'*64)]:
   with self.assertRaises(ValueError):c.admitted_policy(result,self.m,self.req)
 def test_refused_bounds_never_widen(self):
  self.reports[0]['measured_cdf_abs_error']=.01196
  with self.assertRaises(ValueError):c.admitted_policy(self.result,self.m,self.req)
 def test_bool_native_is_not_zero_integer(self):
  self.result['native_controls'][0][0]['exp_mismatches']=False
  with self.assertRaises(ValueError):c.admitted_policy(self.result,self.m,self.req)
 def test_legacy_open_no_dispatch(self):
  obj={};self.assertIs(c.before_open(None,{},None,obj),obj)
 def test_static_old_policy_not_sufficient(self):
  with self.assertRaises(ValueError):c.before_open(None,{'sampling_policy':{'version':f.VERSION}},None,{})
 def test_request_budget_and_exact_schema(self):
  for q in [dict(self.req,max_tokens=True),dict(self.req,max_tokens=65),dict(self.req,task_indices=[1]),dict(self.req,extra=1)]:
   with self.assertRaises(ValueError):c.request(q)
 def test_opening_executes_pinned_job_then_reuses_same_manifest(self):
  with tempfile.TemporaryDirectory()as tmp:
   req=self.req;opening=dict(environments=[dict(spec={'id':'math'},indices=[1,2],harness=self.h)],source_bundle={'sha256':'c'*64},model_runtime_revision='runtime',backend_profile={'dtype':'float32'})
   config=dict(sampling_policy={'version':f.VERSION,'max_attempts':16,'calibration':c.admitted_policy(self.result,self.m,self.req)},successor_calibration={'version':c.VERSION,'env_id':'math'})
   seen=[]
   def run(label,role,manifest,cache,**fields):
    seen.append((label,manifest,fields));result=dict(self.result,request_sha256=c.digest(fields['successor_calibration']));return dict(job_id='original',successor_calibration=result)
   controller=SimpleNamespace(state=Path(tmp),jobs=SimpleNamespace(run=run),checkpoint_with_reads=lambda v:dict(v,read_urls={}))
   status=dict(checkpoint={'id':'a'*64,'files':{'x':'d'*64}})
   a=c.before_open(controller,config,status,opening);b=c.before_open(controller,config,status,opening)
   self.assertEqual(a['sampling_policy']['calibration']['checkpoint'],'a'*64);self.assertEqual(a,b)
   self.assertEqual(seen[0],seen[2]);self.assertEqual(seen[1],seen[3]);self.assertEqual(seen[0][1]['sampling_contract']['version'],'forced-inverse-cdf-replay-v1')
 def test_refused_measurement_holds_opening(self):
  with tempfile.TemporaryDirectory()as tmp:
   h=self.h;opening=dict(environments=[dict(spec={'id':'math'},indices=[1,2],harness=h)],source_bundle={'sha256':'c'*64},model_runtime_revision='runtime',backend_profile={'dtype':'float32'})
   config=dict(sampling_policy={'version':f.VERSION,'max_attempts':16,'calibration':c.admitted_policy(self.result,self.m,self.req)},successor_calibration={'version':c.VERSION,'env_id':'math'})
   def run(label,role,manifest,cache,**fields):
    r=dict(self.result,request_sha256=c.digest(fields['successor_calibration']));r['reports'][0]['measured_cdf_abs_error']=.02;return dict(job_id='original',successor_calibration=r)
   ctrl=SimpleNamespace(state=Path(tmp),jobs=SimpleNamespace(run=run),checkpoint_with_reads=lambda v:v)
   with self.assertRaises(ValueError):c.before_open(ctrl,config,{'checkpoint':{'id':'a'*64}},opening)
