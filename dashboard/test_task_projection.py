import unittest,json
from dashboard.task_projection import project
class TaskProjectionTests(unittest.TestCase):
 def fixture(self):
  manifest={'epoch':'e','environments':[{'env_id':'math','spec':{'num_samples':5,'source_hash':'tasks-v1'}}]}
  learner={'task_counts':[{'environment':'math','index':2,'submitted':3,'included':2}]}
  report={'epoch':'e','outcomes':[{'env_id':'math','index':2,'valid':True,'fully_audited':True}], 'accepted':[{'epoch':'e','env_id':'math','index':2,'rollouts':[{'classification':'positive','reward':1,'turns':'PRIVATE'},{'classification':'negative','reward':0}]}]}
  return manifest,learner,report
 def test_repeat_counts_and_audited_solve_are_separate(self):
  m,l,r=self.fixture();x=project(m,l,{'miner':r});self.assertEqual(x['tasks'][0],dict(environment='math',index=2,submitted=3,included=2,solves=1));self.assertTrue(x['solves_available']);self.assertNotIn('PRIVATE',json.dumps(x))
 def test_unaudited_and_replayed_are_not_solved(self):
  m,l,r=self.fixture();r['outcomes'][0]['fully_audited']=False;self.assertFalse(project(m,l,{'miner':r})['solves_available']);r['epoch']='old';self.assertEqual(project(m,l,{'miner':r})['tasks'][0]['solves'],0)
 def test_missing_data_and_taskspace_change(self):
  m,l,r=self.fixture();x=project(m,None,{});self.assertFalse(x['submitted_available']);self.assertFalse(x['solves_available']);m['environments'][0]['spec']['source_hash']='other';self.assertNotEqual(x['corpora'][0]['key'],project(m,None,{})['corpora'][0]['key'])
 def test_invalid_index_unavailable(self):
  m,l,r=self.fixture();l['task_counts'][0]['index']=5;self.assertIsNone(project(m,l,{}))
