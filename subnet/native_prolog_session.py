"""Prospective common session for the qualified original NQueens actor.

Dispatch only from a new operator-signed source. Original task metadata stays
inside the trusted terminal grader and is absent from public model observations.
"""
import asyncio,hashlib,json,math
from pathlib import Path
from .storage import canonical
from .native_prolog_actor import PublicActor,public_task,grade_original,REVISION,BASE,SHIM_SHA
VERSION='original-nqueens-common-session-v1'
class NativePrologSession:
 def __init__(self,spec):
  from .environments import _taskset
  if spec.id!='affine_prolog' or spec.adapter!='prime_v1' or spec.config.get('prolog_session_revision')!=VERSION:raise ValueError('qualified original Prolog session')
  pins=spec.config.get('prolog_source_files');expected=['subnet/native_prolog_actor.py','subnet/native_prolog_session.py'];root=Path(__file__).resolve().parents[1]
  if not isinstance(pins,dict) or set(pins)!=set(expected):raise ValueError('Prolog session source membership')
  for name,digest in pins.items():
   if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Prolog session source pin')
  runtime=spec.config.get('prolog_runtime')
  if not isinstance(runtime,dict) or runtime.get('revision')!=REVISION or runtime.get('base_image')!=BASE or runtime.get('shim_sha256')!=SHIM_SHA:raise ValueError('Prolog actor runtime pin')
  self.spec=spec;self.runtime=runtime;self.tasks=list(_taskset(spec));self.actor=None;self.done=False
  if len(self.tasks)!=spec.num_samples or any(t.data.kind!='nqueens' for t in self.tasks):raise ValueError('original NQueens task selection')
 def reset(self,index,seed):
  if self.actor is not None or type(index) is not int or not 0<=index<len(self.tasks) or type(seed) is not int:raise ValueError('Prolog reset identity')
  self.task=self.tasks[index];self.public=public_task(self.task);self.turns=0;self.done=False;self.actor=PublicActor(self.runtime,self.public)
  try:self.actor.start()
  except Exception:self.close();raise
  return dict(messages=self.public['messages'],tools=self.public['tools'],task_name=self.task.data.name,task_hash=hashlib.sha256(canonical(dict(public=self.public,version=VERSION,index=index,seed=seed))).hexdigest())
 def step(self,action):
  if self.actor is None or self.done or not isinstance(action,dict) or not isinstance(action.get('text',''),str):raise ValueError('Prolog action state')
  calls=action.get('tool_calls') or []
  if not isinstance(calls,list) or len(calls)>8:raise ValueError('Prolog tool budget')
  observations=[]
  try:
   for ordinal,call in enumerate(calls):
    if not isinstance(call,dict) or call.get('name')!='bash' or not isinstance(call.get('arguments'),dict) or set(call['arguments'])!={'command'}:raise ValueError('original bash action')
    result=self.actor.shell(call['arguments']['command'],timeout=60)
    observations.append(dict(role='tool',tool_call_id=call.get('id',f'call-{self.turns+1}-{ordinal}'),name='bash',content=json.dumps(result)))
   self.turns+=1;self.done=not calls or self.turns>=self.spec.max_turns;reward=0.
   if self.done:
    reward=asyncio.run(grade_original(self.task,self.actor))['reward']
    if type(reward) not in (int,float) or not math.isfinite(reward) or reward not in (0.,1.):raise ValueError('original Prolog binary grade')
   return dict(observations=observations,done=self.done,reward=float(reward),classification='positive' if self.done and reward>=self.spec.success_reward else 'negative' if self.done else 'neutral')
  except Exception:self.done=True;self.close();raise
 def close(self):
  if self.actor is not None:self.actor.close();self.actor=None
