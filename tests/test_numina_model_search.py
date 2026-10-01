import copy,unittest
from ops.probe_numina_model_search import candidates,validate_contract
class NuminaContractTests(unittest.TestCase):
 def setUp(self):
  self.plan=dict(schema=1,experiment='original-numina-public-starter-model-search-v1',payable=False,chain_transactions=False,search_budget=16,indices=[13],environment=dict(id='affine_numina',adapter='prime_v1',num_samples=32,success_reward=1.),harness=dict(version='text-tools-v1',policy='candidates',max_output_tokens=512,temperature=4.,top_p=1.,candidates=candidates()),original_task_snapshot_sha256='bedd979e813cab4c035870e3f38fa2519503f82145a2bd43bb97cbb9dcd974a3')
 def test_exact_qualified_public_target(self):self.assertIs(validate_contract(self.plan),self.plan)
 def test_other_task_not_admitted(self):
  self.plan['indices']=[14]
  with self.assertRaises(ValueError):validate_contract(self.plan)
 def test_unapproved_candidate_not_admitted(self):
  self.plan['harness']['candidates'][1]='pwd'
  with self.assertRaises(ValueError):validate_contract(self.plan)
 def test_payable_not_admitted(self):
  self.plan['payable']=True
  with self.assertRaises(ValueError):validate_contract(self.plan)
 def test_candidates_change_only_public_assertion(self):
  a,b=candidates();self.assertEqual(a.replace('n>0','n<0'),b)
if __name__=='__main__':unittest.main()
