import types,unittest
from unittest.mock import patch
import torch
from subnet.long_context_service_training import output_count,rollout_mean,accumulate_pair
from subnet.long_context_service_runtime import LongContextServiceRuntime

class SequentialTests(unittest.TestCase):
 def test_sequential_chain_rule_equals_full_graph_across_unequal_turn_lengths(self):
  pos={'turns':[{'output':[1,2],'factor':2.},{'output':[3],'factor':3.}]};neg={'turns':[{'output':[4],'factor':1.},{'output':[5,6,7],'factor':4.}]}
  runtime=types.SimpleNamespace(weight=torch.tensor(.7,dtype=torch.float64,requires_grad=True))
  def mean(r,t):return (r.weight*t['factor']).sin()
  with patch('subnet.long_context_service_training.turn_mean',mean):
   margin=rollout_mean(runtime,pos)-rollout_mean(runtime,neg)-.13;loss=-torch.nn.functional.logsigmoid(.1*margin);loss.backward();expected=runtime.weight.grad.clone();runtime.weight.grad=None
   coefficient=-.1*float(torch.sigmoid(-.1*margin.detach()));accumulate_pair(runtime,pos,neg,coefficient)
  self.assertTrue(torch.allclose(runtime.weight.grad,expected,atol=1e-14,rtol=0))
 def test_auxiliary_role_cannot_enter_agent_loss(self):
  with self.assertRaisesRegex(ValueError,'all-agent'):output_count({'turns':[{'output':[1],'model_role':'user'}]})

class RuntimeBoundaryTests(unittest.TestCase):
 def test_factory_required_before_gpu_load(self):
  with self.assertRaisesRegex(ValueError,'trusted session factory'):LongContextServiceRuntime('/never-read',{}, {},{},None)
 def test_native_spec_budgets_before_gpu_load(self):
  spec={'id':'affine_eog','adapter':'native_eog_broker','version':'wrong','source_hash':'x'}
  with self.assertRaisesRegex(ValueError,'signed spec'):LongContextServiceRuntime('/never-read',{},spec,{},lambda s:None)
