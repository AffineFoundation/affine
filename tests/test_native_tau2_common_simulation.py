"""Task selection and original orchestration controls; no model/proof claims."""
import copy,json,tempfile,unittest
from pathlib import Path
from subnet.native_tau2_common_simulation import selected_task,loopback_endpoint,execute_original
from subnet.native_tau2_common_contract import digest
from subnet.native_tau2_probe import REVISION

class Task:
 def __init__(self,value):self.value=value;self.id=value['id']
 def model_dump(self,mode):return copy.deepcopy(self.value)
class Simulation(unittest.TestCase):
 def fixture(self,path):
  task={'id':'original-second','user_instructions':'PRIVATE_USER','evaluation_criteria':{'expected':'PRIVATE_GOLD'}}
  collection={'schema':'original-affine-tau2-telecom-private-tasks-v1','data_revision':REVISION,'tasks':[{'index':1,'task':task,'task_hash':digest(task)}]}
  p=path/'private.json';p.write_text(json.dumps(collection))
  config={'endpoint':'http://127.0.0.1:1234/v1','private_collection':str(p),'environment_index':1,'task_hash':digest(task),'models':{'agent':'current-agent','user':'fixed-user'},'max_steps':11,'max_errors':3,'seed':123,'manifest_sha256':'a'*64,'epoch':'nonpayable-1','environment_id':'original-tau2','fixed_user_sha256':'b'*64}
  return task,collection,config
 def test_actual_selected_task_and_fixed_user_model_reach_original_runner(self):
  with tempfile.TemporaryDirectory() as td:
   task,col,cfg=self.fixture(Path(td));calls=[]
   def load(domain,split):return [] if split=='base' else [Task({'id':'first'}),Task(task)]
   def run(**kw):calls.append(kw);return Task({'id':'simulation','reward':0})
   result=execute_original(cfg,load,run,'ORIGINAL_ALL')
   self.assertEqual(calls[0]['task'].id,'original-second');self.assertEqual(calls[0]['llm_agent'],'openai/current-agent');self.assertEqual(calls[0]['llm_user'],'openai/fixed-user');self.assertEqual(calls[0]['seed'],123);self.assertEqual(calls[0]['evaluation_type'],'ORIGINAL_ALL');self.assertEqual(result['task_hash'],digest(task));self.assertFalse(result['payable'])
 def test_private_task_mutation_rejected_before_native_runner(self):
  with tempfile.TemporaryDirectory() as td:
   task,col,cfg=self.fixture(Path(td));col['tasks'][0]['task']['user_instructions']='changed';Path(cfg['private_collection']).write_text(json.dumps(col));calls=[]
   with self.assertRaises(ValueError):execute_original(cfg,lambda *a:[],lambda **kw:calls.append(kw),'ALL')
   self.assertEqual(calls,[])
 def test_original_provider_drift_or_duplicate_rejected_before_runner(self):
  with tempfile.TemporaryDirectory() as td:
   task,col,cfg=self.fixture(Path(td))
   for pool in [[Task(dict(task,user_instructions='changed'))],[Task(task),Task(task)]]:
    calls=[]
    with self.assertRaises(ValueError):execute_original(cfg,lambda d,s:[] if s=='base' else pool,lambda **kw:calls.append(kw),'ALL')
    self.assertEqual(calls,[])
 def test_base_task_exclusion_preserved(self):
  with tempfile.TemporaryDirectory() as td:
   task,col,cfg=self.fixture(Path(td))
   with self.assertRaises(ValueError):execute_original(cfg,lambda *a:[Task(task)],lambda **kw:None,'ALL')
 def test_endpoint_cannot_redirect_private_auxiliary_requests(self):
  for value in ['https://remote.example/v1','http://localhost:123/v1','http://127.0.0.1:123/v1?redirect=remote','http://user@127.0.0.1:123/v1']:
   with self.subTest(value=value),self.assertRaises(ValueError):loopback_endpoint(value)
  self.assertEqual(loopback_endpoint('http://127.0.0.1:123/v1'),'http://127.0.0.1:123/v1')
 def test_duplicate_private_index_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   task,col,cfg=self.fixture(Path(td));col['tasks'].append(copy.deepcopy(col['tasks'][0]))
   with self.assertRaises(ValueError):selected_task(col,1,digest(task))
if __name__=='__main__':unittest.main()
