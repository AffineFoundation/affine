"""CPU/native qualification boundary around the unchanged common MATH task grader.

This does not register new GPU dispatch IDs. It proves question-only reset and
original terminal semantics for source-bound corpus snapshots before admission.
"""
import copy,hashlib,json,re,math
from pathlib import Path
from .math_corpus import SYSTEM
VERSION='question-only-corpus-native-control-v2'

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class CorpusMathAdapter:
 def __init__(self,spec,*,session_factory=None):
  binding=spec.config.get('corpus_qualification',{})
  if spec.id!='affine_math' or spec.adapter!='prime_v1' or spec.max_turns!=1 or type(spec.success_reward) not in (int,float) or spec.success_reward!=1.0:raise ValueError('original MATH native adapter required')
  if binding.get('version')!=VERSION or binding.get('corpus') not in ('DeepMath-103K','NuminaMath-CoT'):raise ValueError('corpus version/identity')
  if not re.fullmatch('[0-9a-f]{40}',binding.get('upstream_revision','')) or not re.fullmatch('[0-9a-f]{64}',binding.get('catalog_sha256','')):raise ValueError('corpus revision/catalog pin')
  if binding.get('adapter_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise ValueError('corpus adapter source pin')
  if session_factory is None:
   from .environments import create_session
   session_factory=create_session
  self.spec=spec;self.native=session_factory(spec);self.started=False;self.finished=False
 def reset(self,index,seed):
  if type(index) is not int or not 0<=index<self.spec.num_samples or type(seed) is not int or not 0<=seed<2**63:raise ValueError('index/seed')
  initial=self.native.reset(index,seed);messages=initial.get('messages')
  if initial.get('tools') or not isinstance(messages,list) or len(messages)!=2 or messages[0]!={'role':'system','content':SYSTEM} or messages[1].get('role')!='user' or set(messages[1])!={'role','content'}:raise ValueError('question-only native reset')
  self.started=True;self.finished=False;return copy.deepcopy(initial)
 def step(self,action):
  if not self.started or self.finished:raise ValueError('terminal lifecycle')
  if isinstance(action,str):action={'text':action}
  if not isinstance(action,dict) or set(action)-{'text','tool_calls'} or not isinstance(action.get('text'),str) or action.get('tool_calls'):raise ValueError('public text action only')
  self.finished=True
  result=self.native.step(action)
  if result.get('done') is not True or type(result.get('reward')) not in (int,float) or not math.isfinite(result['reward']) or result['reward'] not in (0.0,1.0):raise ValueError('original binary single-turn terminal required')
  self.finished=True;return copy.deepcopy(result)
 def close(self):self.native.close()

def record(spec,index,seed,action):
 actor=CorpusMathAdapter(spec)
 try:initial=actor.reset(index,seed);result=actor.step(action)
 finally:actor.close()
 return {'version':VERSION,'source_hash':spec.source_hash,'index':index,'seed':seed,'initial':initial,'initial_sha256':digest(initial),'action':action,'action_sha256':digest(action),'result':result,'result_sha256':digest(result)}

def replay(spec,index,seed,trace):
 if not isinstance(trace,dict) or type(trace.get('index')) is not int or type(trace.get('seed')) is not int or type(index) is not int or type(seed) is not int or trace.get('version')!=VERSION or trace.get('source_hash')!=spec.source_hash or trace.get('index')!=index or trace.get('seed')!=seed:raise ValueError('trace admission binding')
 for field in ('initial','action','result'):
  if trace.get(field+'_sha256')!=digest(trace.get(field)):raise ValueError('trace content commitment')
 fresh=record(spec,index,seed,trace['action'])
 if any(fresh[k]!=trace[k] for k in ('initial','action','result','initial_sha256','action_sha256','result_sha256')):raise ValueError('fresh original native replay mismatch')
 return fresh
