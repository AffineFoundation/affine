"""Synthetic coordinator contracts, not training or native evaluation."""
import importlib,copy,hashlib,unittest
from subnet import native_tau2_common_bridge as b
from subnet.native_tau2_probe import REVISION
class TestCommonBridge(unittest.TestCase):
 def setUp(self):
  f=importlib.import_module('test_native_tau2_common_search_contract').ContractTests();f.setUp();self.f=f
  f.manifest['tasks']=[{'index':i,'seed':i,'task_hash':hashlib.sha256(str(i).encode()).hexdigest()} for i in range(32)]
  f.manifest.update(mining_indices=list(range(16)),heldout_indices=list(range(16,32)),native_execution_policy={'revision':'original-native-v1','max_steps':12,'max_errors':3})
  f.manifest['roles']['agent'].update(model_runtime_revision='controlled-profile',native_role_revision='controlled-profile',generation_policy={'temperature':.7,'top_p':1},candidate_policy={'revision':'public-only'})
  self.public={'schema':'original-affine-tau2-telecom-public-task-commitments-v1','data_revision':REVISION,'data_inventory_sha256':f.manifest['environment']['data_inventory_sha256'],'split_policy':'original-tau2-user-instruction-group-disjoint-v1','mining_indices':list(range(16)),'heldout_indices':list(range(16,32)),'tasks':[{'index':i,'task_hash':r['task_hash'],'scenario_group_sha256':('a' if i<16 else 'b')*64} for i,r in enumerate(f.manifest['tasks'])]}
  f.manifest['environment']['taskset_sha256']=b.digest(self.public)
 def contract(self):return b.heldout_contract(self.f.sign(self.f.manifest),self.f.authority,self.f.user,self.public)
 def test_fixed16_dataset_excludes_agent_updated_weights(self):
  initial=self.contract();files=self.f.manifest['roles']['agent']['checkpoint']['files'];files['model.safetensors']='f'*64;cp=self.f.manifest['roles']['agent']['checkpoint'];cp['id']=b.digest(files);self.f.manifest['checkpoint']=cp
  self.assertEqual(initial['dataset_id'],self.contract()['dataset_id']);self.assertEqual(len(initial['heldout_tasks']),16);self.assertFalse(initial['evaluation_performed_here'])
 def test_auxiliary_profile_or_runtime_source_changes_dataset_identity(self):
  initial=self.contract();self.f.user['runtime_profile']['threads']=2
  self.assertNotEqual(initial['dataset_id'],self.contract()['dataset_id'])
 def test_overlap_wrong_task_and_type_failclosed(self):
  for mutation in ('overlap','task','type'):
   self.setUp()
   if mutation=='overlap':self.public['tasks'][16]['scenario_group_sha256']='a'*64
   elif mutation=='task':self.public['tasks'][16]['task_hash']='c'*64
   else:self.public['mining_indices'][0]=False
   self.f.manifest['environment']['taskset_sha256']=b.digest(self.public)
   with self.assertRaises(ValueError):self.contract()
 def test_training_cannot_use_qualification_without_new_freeze(self):
  with self.assertRaises(ValueError):b.training_plan({'raw':b'qualification'},{'raw':b'qualification'},self.f.sign(self.f.manifest),self.f.authority,self.f.user,131)
