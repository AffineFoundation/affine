import copy
import unittest
from unittest.mock import patch
from ops.probe_wikispeedia_model_search import validate_contract,windowed_config,native_preflight,ARTIFACT_POLICY
from subnet.native_wikispeedia import public_candidate_harness

class WikiModelContract(unittest.TestCase):
    def plan(self):
        tools=[{'function':{'name':'click_link','parameters':{'properties':{'article':{'type':'string'}}}}}]
        candidate=public_candidate_harness('A','C',tools,{'A':['B'],'B':['C'],'C':[]},30)
        return dict(schema=1,experiment='original-wikispeedia-window-model-search-v1',payable=False,chain_transactions=False,search_budget=16,indices=[0,1,2,3],environment=dict(id='affine_wikispeedia',adapter='prime_v1',num_samples=20,max_turns=30,max_output_tokens=256,success_reward=1.),artifact_policy=ARTIFACT_POLICY,shared_GPU_idle_required=True,normal_Numina_Pydantic_recovery_has_priority=True,tasks=[dict(index=i,max_turns=candidate['max_turns'],harness=windowed_config(candidate))for i in range(4)])
    def test_public_route_and_terminal_choices_preserved(self):
        p=self.plan();h=p['tasks'][0]['harness'];self.assertIn('B',h['candidates'][0]);self.assertEqual(h['turn_overrides']['2']['candidates'],['Done.','Finished.']);self.assertEqual(h['history_window_messages'],2);validate_contract(p)
    def test_heldout_or_payable_scope_refused(self):
        for field,value in [('indices',[4,5,6,7]),('search_budget',True),('search_budget',33),('chain_transactions',True),('payable',True),('shared_GPU_idle_required',False),('normal_Numina_Pydantic_recovery_has_priority',False)]:
            p=self.plan();p[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_contract(p)
    def test_legacy_context_or_hidden_budget_change_refused(self):
        for field,value in [('version','text-tools-v1'),('history_window_messages',3),('max_output_tokens',512),('policy','autoregressive')]:
            p=self.plan();p['tasks'][0]['harness'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_contract(p)
        p=self.plan();p['tasks'][0]['index']=False
        with self.assertRaises(ValueError):validate_contract(p)
        p=self.plan();p['environment']['num_samples']=4000
        with self.assertRaises(ValueError):validate_contract(p)
    def test_preloaded_provider_refused_before_resource_read(self):
        with patch.dict('sys.modules',{'wikispeedia_v1.graph':object()}):
            with self.assertRaisesRegex(ValueError,'before resource admission'):native_preflight({})
    def test_wrong_cache_refused_before_provider_or_model_import(self):
        with patch.dict('os.environ',{'WIKISPEEDIA_CACHE_DIR':'/unapproved'}):
            with self.assertRaisesRegex(ValueError,'private cache'):native_preflight({'resource_cache':'/approved'})

if __name__=='__main__':unittest.main()
