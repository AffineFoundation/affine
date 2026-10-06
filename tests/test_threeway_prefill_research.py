import unittest,torch,copy
from subnet import threeway_prefill_research as t
from subnet.audit_policy import InvalidSample
from subnet.fast_prefill_audit import NumericalAmbiguity
class Threeway(unittest.TestCase):
 def check(self,p,token,u,error=0.001,top_p=1.):return t.intervals(torch.tensor([p],dtype=torch.float32).log(),[token],[u],1.,top_p,error)
 def test_interior_prescribed_draw_passes(self):self.assertTrue(self.check([.5,.5],0,.25)['all_intervals_verified'])
 def test_wrong_draw_outside_region_rejects(self):
  with self.assertRaises(InvalidSample):self.check([.5,.5],0,.75)
 def test_inside_and_outside_uncertainty_band_are_both_unknown(self):
  for token,u in [(0,.4995),(1,.5005),(0,.5005),(1,.4995)]:
   with self.assertRaises(NumericalAmbiguity):self.check([.5,.5],token,u)
 def test_zero_support_never_passes_near_boundary(self):
  with self.assertRaises(NumericalAmbiguity):self.check([.9,.1],1,.9995,top_p=.8)
 def test_zero_support_far_from_boundary_rejects(self):
  with self.assertRaises(InvalidSample):self.check([.9,.1],1,.5,top_p=.8)
 def test_any_certain_failure_wins_over_other_unknown(self):
  with self.assertRaises(InvalidSample):t.intervals(torch.tensor([[.5,.5],[.9,.1]]).log(),[0,1],[.75,.9995],1.,.8,.001)
 def test_zero_error_preserves_exact_boundary_pick(self):
  self.assertTrue(self.check([.5,.5],1,.5,error=0)['all_intervals_verified'])
  with self.assertRaises(InvalidSample):self.check([.5,.5],0,.5,error=0)
 def test_numeric_boolean_policy_aliases_reject(self):
  for k,v in [('historical_execution_proof',0),('autoregressive_fallback',0.),('numerical_inconclusive_not_valid',1)]:
   p=copy.deepcopy(t.POLICY);p[k]=v
   with self.assertRaises(ValueError):t.validate_policy(p)
 def test_token_and_draw_framing_fail_closed(self):
  for token in (True,-1,2):
   with self.assertRaises(InvalidSample):self.check([.5,.5],token,.25)
  for u in (True,float('nan'),1.):
   with self.assertRaises(ValueError):self.check([.5,.5],0,u)
 def test_no_runtime_model_or_cached_reference_called(self):
  from unittest.mock import patch
  with patch('subnet.fast_prefill_audit.verify_cached_reference',side_effect=AssertionError('forbidden cached replay')):
   self.assertTrue(self.check([.5,.5],0,.25)['all_intervals_verified'])
   with self.assertRaises(NumericalAmbiguity):self.check([.5,.5],0,.4995)
if __name__=='__main__':unittest.main()
