import copy
from test_replay_training import ReplayTrainingTests
from subnet import verified_replay_pool as r
from subnet.replay_training import admitted,verified_pairs
from subnet.sample_harness import project
class IndexedReplay(ReplayTrainingTests):
 def setUp(self):
  super().setUp();self.full=copy.deepcopy(self.manifest)
  plain=self.full['environments'][0]['harness'];other=dict(plain,candidates=['second','wrong'])
  wrapper=dict(version='indexed-harness-v1',by_index={'0':plain,'1':other})
  self.full['environments'][0]['harness']=wrapper
  self.full['sample_harness_registry']={'e':dict(indices=[0,1],harness=copy.deepcopy(wrapper))}
  current=self.sign(self.full);pool=copy.deepcopy(self.inputs['pool']['payload']);pool['current_manifest_sha256']=r.digest(current)
  pool['entries'][0]['current_manifest_sha256']=r.digest(current);pool['pool_sha256']=r.digest({k:v for k,v in pool.items()if k!='pool_sha256'})
  self.inputs={'manifest':current,'pool':self.sign(pool),'reuse_counts':{}}
  self.manifest=copy.deepcopy(self.full);self.manifest['environments'][0].update(indices=[],harness=project(wrapper,[],[0,1]))
 def test_full_registry_replay_admitted_when_live_row_inactive(self):
  self.assertEqual(len(admitted(self.manifest,self.inputs,self.authority)[1]['selected']),1)
 def test_changed_live_full_registry_cannot_override_trusted_signed_map(self):
  self.manifest['sample_harness_registry']['e']['harness']['by_index']['0']['candidates'][0]='changed'
  with self.assertRaises(ValueError):admitted(self.manifest,self.inputs,self.authority)
 def test_extra_or_missing_registry_refused_even_inactive(self):
  del self.manifest['sample_harness_registry']['e']
  with self.assertRaises(ValueError):admitted(self.manifest,self.inputs,self.authority)
 def test_truthy_nonboolean_replay_verification_refused(self):
  class Runtime:
   def configure(self,*args):pass
   def compute(self,*args):return None,None
   def build_proofs(self,*args,**kwargs):return []
   def verify(self,*args):return 1
  with self.assertRaises(ValueError):verified_pairs(Runtime(),self.manifest,self.traced_inputs(),self.authority)
