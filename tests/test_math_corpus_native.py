import hashlib,unittest,copy
from pathlib import Path
from types import SimpleNamespace
from subnet.math_corpus_native import CorpusMathAdapter,VERSION,digest,replay
from subnet.math_corpus import SYSTEM

def spec():
 return SimpleNamespace(id='affine_math',adapter='prime_v1',max_turns=1,success_reward=1.0,num_samples=2,source_hash='a'*64,config={'corpus_qualification':{'version':VERSION,'corpus':'DeepMath-103K','upstream_revision':'b'*40,'catalog_sha256':'c'*64,'adapter_sha256':hashlib.sha256(Path('subnet/math_corpus_native.py').read_bytes()).hexdigest()}})
class Native:
 def reset(self,index,seed):return {'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':'Compute 1+1.'}],'tools':[],'task_hash':'d'*64}
 def step(self,action):return {'done':True,'reward':0.0,'classification':'negative','observations':[]}
 def close(self):pass
class Controls(unittest.TestCase):
 def test_public_shape_lifecycle_and_scalar(self):
  a=CorpusMathAdapter(spec(),session_factory=lambda s:Native());i=a.reset(0,1);self.assertEqual(len(i['messages']),2);self.assertEqual(a.step({'text':'wrong','tool_calls':[]})['reward'],0.)
  with self.assertRaises(ValueError):a.step('again')
 def test_private_action_injection_refused(self):
  a=CorpusMathAdapter(spec(),session_factory=lambda s:Native());a.reset(0,0)
  for action in ({'text':'x','reference':'2'},{'text':'x','grader':{}},{'text':'x','tool_calls':[{}]}):
   with self.assertRaises(ValueError):a.step(action)
 def test_versions_source_adapter_and_seed_refused(self):
  for change in ('version','adapter_sha256','upstream_revision','catalog_sha256'):
   s=spec();s.config['corpus_qualification'][change]='wrong'
   with self.assertRaises(ValueError):CorpusMathAdapter(s,session_factory=lambda s:Native())
  a=CorpusMathAdapter(spec(),session_factory=lambda s:Native())
  for index,seed in ((True,0),(0,True),(2,0),(0,-1)):
   with self.assertRaises(ValueError):a.reset(index,seed)
 def test_native_private_message_injection_refused(self):
  class Bad(Native):
   def reset(self,index,seed):
    r=super().reset(index,seed);r['messages'][1]['reference']='2';return r
  a=CorpusMathAdapter(spec(),session_factory=lambda s:Bad())
  with self.assertRaises(ValueError):a.reset(0,0)
 def test_trace_cross_index_seed_source_and_action_refused_before_grade(self):
  s=spec();t={'version':VERSION,'source_hash':s.source_hash,'index':0,'seed':1,'initial':{},'action':{'text':'x'},'result':{}}
  for f in ('initial','action','result'):t[f+'_sha256']=digest(t[f])
  for field,value in [('index',1),('index',False),('seed',2),('seed',True),('source_hash','e'*64),('action',{'text':'tampered'})]:
   bad=copy.deepcopy(t);bad[field]=value
   with self.assertRaises(ValueError):replay(s,0,1,bad)

 def test_grader_error_is_refused_and_sealed(self):
  class Broken(Native):
   def step(self,action):raise RuntimeError('native grader failed')
  a=CorpusMathAdapter(spec(),session_factory=lambda s:Broken());a.reset(0,0)
  with self.assertRaises(RuntimeError):a.step('x')
  with self.assertRaises(ValueError):a.step('retry')
 def test_boolean_grade_refused(self):
  class Forged(Native):
   def step(self,action):return {'done':True,'reward':True,'classification':'positive'}
  a=CorpusMathAdapter(spec(),session_factory=lambda s:Forged());a.reset(0,0)
  with self.assertRaises(ValueError):a.step('x')
