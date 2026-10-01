import copy,unittest
from test_verified_replay_pool import ReplayTests
from subnet import verified_replay_pool as p
from subnet.sample_harness import resolve
class IndexedPool(ReplayTests):
 def setUp(self):
  super().setUp()
  for m in [self.historical,self.current]:
   for row in m['environments']:
    plain=row['harness'];row['harness']={'version':'indexed-harness-v1','by_index':{str(i):copy.deepcopy(plain)for i in row['indices']}}
 def fixture(self,*args,**kwargs):
  d,hm,ha,data=super().fixture(*args,**kwargs);v=d['payload'];row=next(r for r in hm['payload']['environments']if r['env_id']==v['environment_id']);h=resolve(row['harness'],v['environment_index'],row['indices']);v.update(resolved_harness=h,resolved_harness_sha256=p.digest(h));return self.sign(v),hm,ha,data
 def test_heldout_overlap_and_unsigned_task_index_rejected(self):
  with self.assertRaises(ValueError):self.validate(self.fixture(index=31))
  self.current['environments'][0]['indices']=[0]
  self.current['environments'][0]['harness']['by_index']={'0':self.current['environments'][0]['harness']['by_index']['0']}
  with self.assertRaises(ValueError):self.validate(self.fixture(index=1))
 def test_descriptor_resolved_binding_required(self):
  for mode in ['missing','changed']:
   d,hm,ha,data=self.fixture();v=d['payload']
   if mode=='missing':v.pop('resolved_harness')
   else:v['resolved_harness']['candidates'][0]='forged'
   with self.assertRaisesRegex(ValueError,'resolved harness'):self.validate((self.sign(v),hm,ha,data))
 def test_other_authorized_index_policy_change_does_not_change_selected_target(self):
  self.current['environments'][0]['harness']['by_index']['1']['candidates'][0]='next-index-policy'
  self.assertEqual(self.validate()['environment_index'],0)
 def test_selected_policy_change_requires_new_qualification(self):
  self.current['environments'][0]['harness']['by_index']['0']['candidates'][0]='changed'
  with self.assertRaisesRegex(ValueError,'resolved replay'):self.validate()
if __name__=='__main__':unittest.main()
