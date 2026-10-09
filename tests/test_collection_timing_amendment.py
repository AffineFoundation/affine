import copy,json,tempfile,unittest
from pathlib import Path
from ops.collection_timing_amendment import validate,VERSION
class TimingTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  h=dict(version='bounded-hourly-phases-v1',mine_seconds=600,freeze_seconds=60,audit_seconds=600,train_publication_seconds=2100,weight_seconds=120,slack_seconds=120)
  self.old=dict(duration=600,hourly_execution_policy=h,state=self.tmp.name,sampling_policy={'max_attempts':1000})
  self.new=copy.deepcopy(self.old);self.new['duration']=1200;self.new['hourly_execution_policy'].update(mine_seconds=1200,train_publication_seconds=1500)
  self.am=dict(version=VERSION,first_round=88,previous_duration=600,duration=1200,previous_hourly_policy=h,hourly_policy=self.new['hourly_execution_policy'])
  self.state=dict(round=88,active=dict(epoch='epoch88',phase='opening'))
 def call(self):return validate(self.am,self.old,self.new,self.state)
 def test_accept_original_future(self):self.assertEqual(self.call(),{'duration','hourly_execution_policy'})
 def test_no_87(self):
  self.state['round']=87
  with self.assertRaises(ValueError):self.call()
 def test_no_other_duration(self):
  self.new['duration']=1800
  with self.assertRaises(ValueError):self.call()
 def test_no_unmatched_mine(self):
  self.new['hourly_execution_policy']=dict(self.new['hourly_execution_policy'],mine_seconds=600)
  with self.assertRaises(ValueError):self.call()
 def test_no_unfunded_total(self):
  self.am['hourly_policy']=dict(self.am['hourly_policy'],train_publication_seconds=2100);self.new['hourly_execution_policy']=self.am['hourly_policy']
  with self.assertRaises(ValueError):self.call()
 def test_no_revised_old_budget(self):
  self.old['duration']=900
  with self.assertRaises(ValueError):self.call()
 def test_no_published_short_epoch(self):
  (Path(self.tmp.name)/'epoch88-manifest.json').write_text(json.dumps(dict(start=100,deadline=700,hourly_execution_policy=self.old['hourly_execution_policy'])))
  with self.assertRaises(ValueError):self.call()
 def test_reobserve_original_long_epoch(self):
  (Path(self.tmp.name)/'epoch88-manifest.json').write_text(json.dumps(dict(start=100,deadline=1300,hourly_execution_policy=self.new['hourly_execution_policy'])))
  self.call()
 def test_scope_excludes_scientific_fields(self):
  self.assertNotIn('sampling_policy',self.call());self.assertNotIn('K',self.call());self.assertNotIn('training_policy',self.call())
 def test_no_unknown_amendment_fields(self):
  self.am['K']=1
  with self.assertRaises(ValueError):self.call()
if __name__=='__main__':unittest.main()
