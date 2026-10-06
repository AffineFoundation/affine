import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from test_successor_calibration import Controls
from subnet import successor_calibration as c,fast_prefill_audit as f

class Recalibration(Controls):
 def run_bounded(self,errors,*,deadline=1800,limit=3,token=False):
  seen=[]; directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
  path=Path(directory.name)/'record.json';record={};opening={}
  if token:opening['token_artifact_policy']={'explicit-test':True}
  seed=copy.deepcopy(self.result)
  # Genuine CP19 observed scalar fixture: proposal .0005950927734375;
  # confirmation .00075531005859375, which needs a fresh confirmation.
  for r in seed['reports']:r['measured_logprob_abs_error']=.000148773193359375;r['measured_cdf_abs_error']=1.588206410230164e-5
  report=dict(job_id='measurement-original',successor_calibration=seed)
  config=dict(sampling_policy={'version':f.VERSION,'max_attempts':16})
  def run(label,role,manifest,cache,**fields):
   seen.append((label,copy.deepcopy(fields)))
   index=int(label.rsplit('-',1)[-1]);r=copy.deepcopy(seed);r['request_sha256']=c.digest(fields['successor_calibration'])
   for row in r['reports']:row.update(measured_logprob_abs_error=errors[index],measured_cdf_abs_error=1.19925e-5)
   return dict(job_id=label,successor_calibration=r)
  args=(None,config,opening,record,path,'c'*64,self.m,self.req,report,SimpleNamespace(run=run),None,dict(max_confirmations=limit,deadline_seconds=deadline))
  return args,seen,path,record
 def test_observed_cp19_failure_seeds_fresh_confirmation(self):
  args,seen,path,record=self.run_bounded([.00075531005859375,.00075531005859375])
  result=c._bounded_confirmation(*args)
  self.assertEqual(len(seen),2);self.assertNotEqual(seen[0][1],seen[1][1])
  self.assertEqual(result['sampling_policy']['calibration']['logprob_atol'],4*.00075531005859375)
  self.assertEqual(len(record['bounded_confirmations']['rounds']),2)
  # Retry observes identical originals, with no new labels or changed draws.
  again=c._bounded_confirmation(*args);self.assertEqual(result,again)
  self.assertEqual([v[0]for v in seen[:2]],[v[0]for v in seen[2:]])
 def test_exhausted_rounds_never_admit(self):
  args,seen,path,record=self.run_bounded([.001,.005],limit=2)
  with self.assertRaisesRegex(ValueError,'exhausted'):c._bounded_confirmation(*args)
  self.assertNotIn('calibration',record);self.assertEqual(len(seen),2)
  with self.assertRaisesRegex(ValueError,'exhausted'):c._bounded_confirmation(*args)
  self.assertEqual(len(record['bounded_confirmations']['rounds']),2)
 def test_deadline_and_native_failure_hold_opening(self):
  args,seen,path,record=self.run_bounded([.0001])
  with patch('time.time',side_effect=[0,0,0,0,2000]):
   with self.assertRaisesRegex(ValueError,'deadline'):c._bounded_confirmation(*args)
  self.assertNotIn('calibration',record)
  args,seen,path,record=self.run_bounded([.0001]);args[8]['successor_calibration']['native_controls'][0][0]['exp_mismatches']=1
  with self.assertRaisesRegex(ValueError,'native calibration proof mismatch'):c._bounded_confirmation(*args)
  self.assertEqual(seen,[])
 def test_saved_report_edit_refuses_and_hard_max_not_widened(self):
  args,seen,path,record=self.run_bounded([.0001]);c._bounded_confirmation(*args)
  record['bounded_confirmations']['rounds'][0]['report']['job_id']='forged'
  with self.assertRaisesRegex(ValueError,'report changed'):c._bounded_confirmation(*args)
  args,seen,path,record=self.run_bounded([.5])
  with self.assertRaises(ValueError):c._bounded_confirmation(*args)
  self.assertNotIn('calibration',record)
 def test_explicit_token_contract_does_not_claim_uploaded_lp(self):
  args,seen,path,record=self.run_bounded([.00075531005859375],token=True)
  with patch('subnet.token_only_protocol.for_manifest',return_value=True)as validate:
   result=c._bounded_confirmation(*args)
  validate.assert_called_once();self.assertEqual(len(seen),1)
  self.assertEqual(result['sampling_policy']['calibration']['logprob_atol'],.0005950927734375)
 def test_invalid_token_contract_rejected_before_confirmation(self):
  args,seen,path,record=self.run_bounded([.0001],token=True)
  with self.assertRaises(ValueError):c._bounded_confirmation(*args)
  self.assertEqual(seen,[])
