import copy,unittest
from subnet.environments import build_spec
from subnet.service import definitions
from subnet.model import Runtime
from subnet.protocol import replay_harness_descriptor
class NoneCompatibility(unittest.TestCase):
 def test_real_prime_none_configuration_matches_runtime_default(self):
  spec=build_spec('affine_verbatim',{'taskset':{'num_samples':2,'target_length':8,'content_type':'codes'}},num_samples=2,max_turns=1,max_output_tokens=96)
  rows=definitions({'environments':[{'spec':spec.to_dict(),'harness':None,'indices':[0,1]}]})
  self.assertIsNone(rows[0]['harness'])
  runtime=Runtime.__new__(Runtime);runtime.configure(spec.to_dict(),None)
  self.assertEqual(runtime.harness['policy'],'autoregressive');self.assertEqual(runtime.harness['max_output_tokens'],64)
 def test_real_prime_none_below_default_budget_refused(self):
  spec=build_spec('affine_verbatim',{'taskset':{'num_samples':2,'target_length':8,'content_type':'codes'}},num_samples=2,max_turns=1,max_output_tokens=32)
  with self.assertRaisesRegex(ValueError,'budget'):definitions({'environments':[{'spec':spec.to_dict(),'harness':None,'indices':[0,1]}]})
 def test_none_descriptor_preparation_keeps_legacy_shape(self):
  self.assertEqual(replay_harness_descriptor({'harness':None,'indices':[0]},0),{})
  plain={'version':'text-tools-v1','policy':'autoregressive'}
  self.assertEqual(replay_harness_descriptor({'harness':plain,'indices':[0]},0),{})
 def test_indexed_descriptor_preparation_binds_authorized_copy(self):
  choice={'version':'text-tools-v1','policy':'candidates','candidates':['yes','no']}
  definition={'harness':{'version':'indexed-harness-v1','by_index':{'0':choice}},'indices':[0]}
  fields=replay_harness_descriptor(definition,0);self.assertEqual(fields['resolved_harness']['candidates'],['yes','no'])
  fields['resolved_harness']['candidates'][0]='changed';self.assertEqual(choice['candidates'][0],'yes')
  with self.assertRaises(ValueError):replay_harness_descriptor(definition,1)
if __name__=='__main__':unittest.main()
