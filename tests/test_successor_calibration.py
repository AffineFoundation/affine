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

 def test_native_spec_changes_create_new_job_and_never_reuse_old_manifest(self):
  with tempfile.TemporaryDirectory()as tmp:
   opening=dict(environments=[dict(spec={'id':'math','source_hash':'old'},indices=[1,2],harness=self.h)],source_bundle={'sha256':'c'*64},model_runtime_revision='runtime',backend_profile={'dtype':'float32'})
   config=dict(sampling_policy={'version':f.VERSION,'max_attempts':16,'calibration':c.admitted_policy(self.result,self.m,self.req)},successor_calibration={'version':c.VERSION,'env_id':'math'})
   seen=[]
   def run(label,role,manifest,cache,**fields):
    seen.append((label,manifest));return dict(job_id=label,successor_calibration=dict(self.result,request_sha256=c.digest(fields['successor_calibration'])))
   ctrl=SimpleNamespace(state=Path(tmp),jobs=SimpleNamespace(run=run),checkpoint_with_reads=lambda v:v);status={'checkpoint':{'id':'a'*64}}
   c.before_open(ctrl,config,status,opening)
   opening['environments'][0]['spec']=dict(id='math',source_hash='new')
   c.before_open(ctrl,config,status,opening)
   self.assertNotEqual(seen[0][0],seen[2][0]);self.assertNotEqual(seen[0][1]['epoch'],seen[2][1]['epoch'])
   self.assertEqual(seen[0][1]['environments'][0]['spec']['source_hash'],'old');self.assertEqual(seen[2][1]['environments'][0]['spec']['source_hash'],'new')
   files=list((Path(tmp)/'successor-calibration').glob('*.json'));self.assertEqual(len(files),2)
   import json
   for path in files:
    doc=json.loads(path.read_text());doc['calibration_environment']['spec']['source_hash']='tampered';path.write_text(json.dumps(doc))
   count=len(seen)
   with self.assertRaisesRegex(ValueError,'native environment binding'):c.before_open(ctrl,config,status,opening)
   self.assertEqual(len(seen),count)
 def test_native_preflight_fails_without_model_or_inference(self):
  with patch('subnet.environments.create_session',side_effect=ValueError('trusted environment code or data hash mismatch'))as create:
   with self.assertRaisesRegex(ValueError,'hash mismatch'):c.preflight_native_spec({'id':'math'})
   create.assert_called_once_with({'id':'math'})
  session=SimpleNamespace(close=unittest.mock.Mock())
  with patch('subnet.environments.create_session',return_value=session):c.preflight_native_spec({'id':'math'})
  session.close.assert_called_once()
 def test_worker_rejects_untrusted_native_spec_before_model_factory(self):
  from subnet.backend_jobs import execute
  from unittest.mock import Mock
  job=dict(job_id='bounded-native-preflight',role='evaluate',source_files={},runtime_versions={},successor_calibration=self.req)
  manifest=dict(epoch='nonpayable-calibration',checkpoint={'id':'a'*64,'files':{}},model_runtime_revision='runtime')
  factory=Mock();first={'spec':{'id':'math'},'indices':[1,2],'harness':self.h}
  with tempfile.TemporaryDirectory()as tmp,patch('subnet.backend_jobs._validate',return_value=(job,manifest)),patch('subnet.backend_jobs.install_source_loader'),patch.dict('os.environ',{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}),patch('subnet.backend_profiles.execution_profile',return_value=('runtime',{},{})),patch('subnet.artifact_budget.for_manifest'),patch('subnet.task_assets.hydrate_manifest'),patch('subnet.backend_jobs.checkpoint',return_value=tmp),patch('subnet.protocol.entries',return_value=[first]),patch('subnet.backend_jobs.initial_configuration',return_value=(first,self.h)),patch('subnet.environments.create_session',side_effect=ValueError('trusted environment code or data hash mismatch'))as session:
   with self.assertRaisesRegex(ValueError,'hash mismatch'):execute({},'authority',tmp,runtime_factory=factory)
  session.assert_called_once_with(first['spec']);factory.assert_not_called()
