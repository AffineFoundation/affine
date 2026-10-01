import copy,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from subnet.harness import source_hash
from subnet.backend_jobs import mine_cumulative,audit
from subnet.verifier import verify
from subnet.batches import unpack,pack
class IndexedPaths(unittest.TestCase):
 def setUp(self):
  self.selected=[];self.uploads=[]
  def choice(text):return dict(version='text-tools-v1',policy='candidates',candidates=[text,'wrong'],max_output_tokens=16,temperature=4.,top_p=1.)
  self.manifest=dict(epoch='nonpayable-indexed',checkpoint=dict(id='approved'),start=0,deadline=1000,K=1,L=1,max_batches=2,audit_policy=dict(mode='full'),harness_source_hash=source_hash(),environments=[dict(env_id='original',spec=dict(id='original',version='native-v1',num_samples=2),indices=[0,1],harness=dict(version='indexed-harness-v1',by_index={'0':choice('taskzero'),'1':choice('taskone')}))])
  self.manifest['sample_harness_registry']={'original':dict(indices=[0,1],harness=copy.deepcopy(self.manifest['environments'][0]['harness']))}
  parent=self
  class Runtime:
   def __init__(self,harness=None):self.harness=harness;self.spec=SimpleNamespace(version='native-v1')
   def for_environment(self,spec,harness):parent.selected.append(harness['candidates'][0]);return Runtime(harness)
   def rollout(self,index,seed):
    positive=seed%2==0
    return dict(schema=2,index=index,sample_index=index,env_id='original',environment_version='native-v1',classification='positive'if positive else'negative',reward=int(positive),turns=[dict(output=[1 if positive else 2],marker=self.harness['candidates'][0])]),[np.zeros((1,2),dtype=np.float32)]
   def verify(self,rollout,arrays):
    if rollout['turns'][0]['marker']!=self.harness['candidates'][0]:raise ValueError('wrong task harness')
    return rollout['turns'][0].get('verification_result',True)
  self.Runtime=Runtime;self.runtime=Runtime()
 def make(self,*args):self.selected.append(args[-1]['candidates'][0]);return self.Runtime(args[-1])
 def mined(self):return mine_cumulative(self.runtime,self.manifest,dict(search_budget=2,seed_start=100),lambda data,timeout:self.uploads.append(data),clock=lambda:10)[0]
 def test_two_same_environment_indices_resolve_through_miner_and_both_auditors(self):
  data=self.mined();self.assertEqual(self.selected,['taskzero','taskone']);self.assertEqual(len(self.uploads),2)
  result,pairs=audit(data,self.manifest,self.runtime);self.assertEqual(len(result['accepted']),2);self.assertEqual([p[1]['index']for p in pairs],[0,1])
  with patch('subnet.verifier.check_runtime_profile'),patch('subnet.verifier.make_runtime',side_effect=self.make):result=verify(data,self.manifest,'unused')
  self.assertEqual(len(result['accepted']),2);self.assertEqual(self.selected[-2:],['taskzero','taskone'])
 def test_local_external_miner_is_lazy_and_routes_each_index(self):
  from subnet.miner import Miner
  identity=SimpleNamespace(id='owned')
  with patch('subnet.miner.time.time',return_value=10),patch('subnet.miner.check_runtime_profile'),patch('subnet.miner.make_runtime',side_effect=self.make)as factory:
   miner=Miner(identity,self.manifest,'unused',capability={'put_url':'unused'})
   factory.assert_not_called();miner.search(0,seed=100,max_attempts=2,env_id='original');miner.search(1,seed=100,max_attempts=2,env_id='original')
   self.assertEqual(self.selected,['taskzero','taskone']);self.assertEqual(len(miner.runtimes),2)
   with self.assertRaises(ValueError):miner.search(2,env_id='original')
 def test_changed_second_task_cannot_reuse_first_task_verifier_cache(self):
  records=unpack(self.mined());records[1][0]['rollouts'][0]['turns'][0]['marker']='taskzero';data=pack(records)
  with patch('subnet.verifier.check_runtime_profile'),patch('subnet.verifier.make_runtime',side_effect=self.make):result=verify(data,self.manifest,'unused')
  self.assertEqual(len(result['accepted']),1);self.assertIn('wrong task harness',result['outcomes'][1]['reason'])
 def test_false_runtime_verification_cannot_be_accepted(self):
  records=unpack(self.mined());records[1][0]['rollouts'][0]['turns'][0]['verification_result']=False;data=pack(records)
  with patch('subnet.verifier.check_runtime_profile'),patch('subnet.verifier.make_runtime',side_effect=self.make):result=verify(data,self.manifest,'unused')
  self.assertEqual(len(result['accepted']),1);self.assertIn('did not succeed',result['outcomes'][1]['reason'])
if __name__=='__main__':unittest.main()
