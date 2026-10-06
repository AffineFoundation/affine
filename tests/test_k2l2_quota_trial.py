import copy,unittest
from ops.k2l2_quota_trial import prepare,native_outcomes,SOURCE
from subnet.forced_sampling import VERSION

class QuotaTrialControls(unittest.TestCase):
 def manifest(self):return dict(epoch='new-qualification',K=1,L=1,source_bundle={'sha256':SOURCE},checkpoint=dict(id='a'*64,files={'config.json':'b'*64,'model.safetensors':'c'*64}),probability_artifact_policy={'version':'selected-token-logprobs-v1'},sampling_contract=dict(version=VERSION,randomness='d'*64,max_attempts=16,verification='exact-token-replay',generation='uncached-eager-inverse-cdf'),environments=[{'spec':{'native':'immutable'}}])
 def batch(self):
  return dict(index=7,env_id='math',rollouts=[dict(index=7,env_id='math',task_hash='a'*64,classification='positive'if i<2 else 'negative',reward=1 if i<2 else 0,turns=[dict(prompt=[1],output=[i+2])])for i in range(4)])
 def replay(self,row):return dict(done=True,task_hash=row['task_hash'],classification='positive'if row['turns'][0]['output'][0]<4 else 'negative',reward=1 if row['turns'][0]['output'][0]<4 else 0)
 def test_same_sampler_context_only_operational_budget_differs(self):
  m=self.manifest();prior=copy.deepcopy(m);plan=prepare(m);self.assertEqual(m,prior);self.assertEqual(plan['manifest']['K'],2);self.assertEqual(plan['manifest']['L'],2);self.assertFalse(plan['dispatch_allowed']);self.assertFalse(plan['execute_allowed']);self.assertEqual(plan['collection_seconds'],600)
  self.assertEqual([r['search_budget']for r in plan['arms']],[8,16]);self.assertEqual(plan['arms'][0]['sampler_context_sha256'],plan['arms'][1]['sampler_context_sha256']);self.assertEqual(plan['manifest']['sampling_contract'],m['sampling_contract']);self.assertEqual(plan['manifest']['environments'],m['environments'])
 def test_short_signed_draw_budget_not_silently_extended(self):
  m=self.manifest();m['sampling_contract']['max_attempts']=8
  with self.assertRaisesRegex(ValueError,'max16'):prepare(m)
 def test_wrong_source_unknown_compact_policy_and_old_quota_rejected(self):
  for change in [dict(K=2),dict(source_bundle={'sha256':'old'}),dict(probability_artifact_policy=None)]:
   with self.subTest(change=change),self.assertRaises(ValueError):prepare(dict(self.manifest(),**change))
  m=self.manifest();del m['probability_artifact_policy']
  with self.assertRaisesRegex(ValueError,'compact'):prepare(m)
 def test_native_four_outcomes_one_task_point(self):
  result=native_outcomes(self.batch(),self.replay);self.assertEqual(result['counts'],{'positive':2,'negative':2});self.assertEqual(result['valid_task_points'],1);self.assertFalse(result['model_proofs_verified'])
 def test_forged_labels_and_duplicate_or_wrong_task_fail(self):
  for change in ['class','reward','boolreward','duplicate','task','hash','missing']:
   b=self.batch()
   if change=='class':b['rollouts'][0]['classification']='negative'
   if change=='boolreward':b['rollouts'][0]['reward']=True
   if change=='reward':b['rollouts'][0]['reward']=0
   if change=='duplicate':b['rollouts'][1]=copy.deepcopy(b['rollouts'][0])
   if change=='task':b['rollouts'][0]['index']=8
   if change=='hash':b['rollouts'][0]['task_hash']='b'*64
   if change=='missing':b['rollouts'].pop()
   with self.subTest(change=change),self.assertRaises(ValueError):native_outcomes(b,self.replay)
 def test_native_exception_is_infrastructure_not_negative(self):
  def failed(row):raise RuntimeError('grader unavailable')
  with self.assertRaises(RuntimeError):native_outcomes(self.batch(),failed)
 def test_existing_task_weight_does_not_reward_pair_count(self):
  import torch
  from subnet.task_normalized_training import task_groups,accumulate_tasks
  from test_persistent_training_policy import pair
  population=[pair(0,1,2),pair(0,3,4),pair(1,5,6),pair(1,7,8)];pairs,tasks,groups,_=task_groups(population,1,'ab'*32);values=torch.nn.Parameter(torch.zeros(2));observed=accumulate_tasks(torch,lambda i:values[pairs[i][1]['index']],[0.]*4,tasks,groups[0]);torch.testing.assert_close(values.grad,torch.tensor([-.025,-.025]))
  for task in tasks:self.assertAlmostEqual(sum(r['gradient_weight']for r in observed if r['task_sha256']==task['task_sha256']),.5)

if __name__=='__main__':unittest.main()
