import copy,unittest
from ops.matched_quota_trial import task_plan,select_matched,validate_stream,verify_generated

def stream(index=2):
 return [dict(index=index,attempt=i,status='verified',native_verified=True,sampler_verified=True,rollout=dict(index=index,seed=i,classification='positive'if i%3 else'negative',reward=float(bool(i%3)),task_hash='a'*64,turns=[dict(prompt=[1],output=[i+2])]))for i in range(16)]
class TrialTests(unittest.TestCase):
 def test_common_tasks_and_first_zip_not_cartesian(self):
  a,s=select_matched({2:stream()},dict(env_id='math'));self.assertEqual([len(a[x])for x in ('1P1N','2P2N')],[1,2]);self.assertEqual(a['1P1N'][0],a['2P2N'][0]);self.assertEqual(s[0]['first2_prefix8'],4)
 def test_censored_task_excluded_from_both(self):
  x=stream();
  for r in x[1:]:r.update(status='native_error',native_verified=False,sampler_verified=False)
  a,s=select_matched({2:x},{});self.assertEqual(a,{'1P1N':[],'2P2N':[]});self.assertFalse(s[0]['matched_included'])
 def test_unknown_is_not_positive(self):
  x=stream();x[1].update(status='numerical_unknown',sampler_verified=False);a,s=select_matched({2:x},{});self.assertEqual(a['1P1N'][0][1]['seed'],2);self.assertEqual(s[0]['unknown'],1)
 def test_all16_or_fail(self):
  for x in (stream()[:8],stream()[::-1]):
   with self.assertRaises(ValueError):validate_stream(x,2)
 def test_claims_and_bool_alias_not_verification(self):
  for k,v in [('native_verified',1),('sampler_verified',False)]:
   x=stream();x[0][k]=v
   with self.assertRaises(ValueError):validate_stream(x,2)
 def test_duplicate_seed_reward_and_task_reject(self):
  for mutation in ('seed','duplicate','reward','task'):
   x=stream()
   if mutation=='seed':x[0]['rollout']['seed']=3
   if mutation=='duplicate':x[1]['rollout']['turns']=copy.deepcopy(x[0]['rollout']['turns'])
   if mutation=='reward':x[0]['rollout']['reward']=1
   if mutation=='task':x[0]['index']=4
   with self.assertRaises(ValueError):validate_stream(x,2)
 def test_reserved_cohort_excluded_and_deterministic(self):
  a=task_plan(list(range(40)),list(range(20)),'a'*64);self.assertTrue(set(a).isdisjoint(range(20)));self.assertEqual(a,task_plan(list(reversed(range(40))),list(range(20)),'a'*64))
 def test_official_false_does_not_become_native_pass(self):
  class R:
   def rollout(self,i,a):return stream(i)[a]['rollout'],[]
   def verify(self,r,a):return False
  with self.assertRaises(ValueError):verify_generated(R(),2,0)
