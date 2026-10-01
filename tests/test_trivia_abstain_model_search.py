import copy,unittest
from ops.probe_trivia_abstain_model_search import validate_contract
class Contract(unittest.TestCase):
 def setUp(self):
  self.plan={'schema':1,'experiment':'original-trivia-abstain-model-search-v1','payable':False,'chain_transactions':False,'search_budget':16,'indices':[0,1],'environment':{'id':'affine_trivia_abstain','adapter':'prime_v1','num_samples':32,'success_reward':1.},'harness':{'version':'text-tools-v1','policy':'candidates','max_output_tokens':256,'temperature':4.,'top_p':1.,'candidates':["I don't know",'definitely_wrong_answer']},'original_task_snapshot_sha256':'c08f2397ce161235e6ccefd78cd2294d2249a5e92c1640344e64b25fc528819e'}
 def test_exact_original_scope(self):self.assertIs(validate_contract(self.plan),self.plan)
 def test_no_heldout_duplicate_or_boolean_indices(self):
  for indices in ([16,17],[0,0],[False,1],[0,1,2]):
   with self.subTest(indices=indices),self.assertRaises(ValueError):validate_contract({**self.plan,'indices':indices})
 def test_scope_budget_model_sampling_policy_and_snapshot_changes(self):
  for key,value in [('payable',True),('chain_transactions',True),('search_budget',17),('search_budget',True),('original_task_snapshot_sha256','0'*64)]:
   with self.subTest(key=key),self.assertRaises(ValueError):validate_contract({**self.plan,key:value})
  for field,value in [('candidates',['correct answer','wrong']),('temperature',1.),('max_output_tokens',512),('top_p',False)]:
   p=copy.deepcopy(self.plan);p['harness'][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):validate_contract(p)
 def test_another_environment_is_not_qualified_by_this_probe(self):
  p=copy.deepcopy(self.plan);p['environment']['id']='affine_trivia'
  with self.assertRaises(ValueError):validate_contract(p)
if __name__=='__main__':unittest.main()
