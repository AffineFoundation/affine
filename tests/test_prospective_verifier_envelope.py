"""Signed transport admission only; no model, proof result or production keys."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from ops import capacity_bounded_verifier_backend as bootstrap
from ops import verifier_capacity_admission as admission
import test_capacity_selected_source as selected_fixture

class ProspectiveVerifierEnvelope(unittest.TestCase):
 def setUp(self):
  self.f=selected_fixture.CapacitySelectedSource();self.f.setUp();self.addCleanup(self.f.doCleanups)
  self.job=self.f.fx.job;self.authority=self.f.fx.authority
  m=copy.deepcopy(self.job['manifest']['payload'])
  m.pop('training_startup_recovery',None)
  m.update(max_batches=9,K=4,L=4,samples_per_batch=8,training_policy='bf16-cpu-fp32-master-task-normalized-persistent-v4',training_input_policy='committed-unaudited-training-v1',training_task_capacity={'version':'signed-training-task-capacity-v1','max_tasks':512})
  self.job['manifest']=self.f.fx.sign(m)
 def test_exact_signed_scope_only_changes_byte_allowance(self):
  before=copy.deepcopy(self.job)
  for module in (bootstrap,admission):self.assertEqual(module.verify_envelope_limit(self.job,self.authority),32_000_000)
  self.assertEqual(before,self.job)
  changes=[{'max_batches':3},{'max_batches':True},{'K':2},{'L':True},{'samples_per_batch':True},{'training_policy':'other'},{'training_startup_recovery':{}},{'training_task_capacity':None},{'training_task_capacity':{'version':'signed-training-task-capacity-v1','max_tasks':512.0}}]
  for change in changes:
   job=copy.deepcopy(self.job);job['manifest']=self.f.fx.sign(dict(job['manifest']['payload'],**change))
   for module in (bootstrap,admission):
    with self.subTest(change=change,module=module.__name__):self.assertEqual(module.verify_envelope_limit(job,self.authority),4_000_000)
  for module in (bootstrap,admission):self.assertEqual(module.verify_envelope_limit(dict(self.job,role='upload'),self.authority),4_000_000)
 def test_actual_large_source_bound_cpu_subprocess_and_both_bootstrap_checks(self):
  self.job['padding']='x'*5_000_000
  self.assertEqual(self.f.call()[3:],(2902,2902))
  helper=Path(admission.__file__)
  bootstrap.isolated_runtime_inventory(self.job,self.f.fx.policy,self.authority,self.f.source,helper)
  result=bootstrap.isolated_transport_admission(self.job,self.f.fx.policy,self.authority,self.f.source,helper)
  self.assertEqual(result.input_limits(),[2902])
 def test_large_historical_and_forged_scope_reject_before_subprocess(self):
  self.job['padding']='x'*5_000_000
  self.job['manifest']=self.f.fx.sign(dict(self.job['manifest']['payload'],max_batches=3))
  with patch('subprocess.run',side_effect=AssertionError('must not spawn')):
   with self.assertRaisesRegex(ValueError,'bounded signed'):self.f.call()
   with self.assertRaisesRegex(ValueError,'bounded signed'):bootstrap.isolated_runtime_inventory(self.job,self.f.fx.policy,self.authority,self.f.source,Path(admission.__file__))
  self.job['manifest']['payload']['max_batches']=9
  for module in (bootstrap,admission):
   with self.assertRaises(Exception):module.verify_envelope_limit(self.job,self.authority)
 def test_absolute_main_bound_precedes_json_and_child(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'job.json';p.write_bytes(b'x'*32_000_001)
   argv=['bootstrap',str(p),'--authority',self.authority,'--workspace',d,'--capacity-policy',str(Path(d)/'unused')]
   with patch('sys.argv',argv),patch.object(bootstrap.json,'loads')as parse,self.assertRaisesRegex(ValueError,'absolute verify'):
    bootstrap.main()
   parse.assert_not_called()
if __name__=='__main__':unittest.main()
