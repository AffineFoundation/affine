"""Owned-worker public text/terminal grader boundary; no hostile-host isolation claim."""
import copy,hashlib,json,math
REVISION='original-rcore-public-text-trusted-terminal-v1'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def sha(v):
 if not isinstance(v,str)or len(v)!=64 or any(c not in'0123456789abcdef'for c in v):raise ValueError('binding SHA')
 return v

def admit_operator(document,authority,root):
 """Authenticate owned-resource bytes before native factory invocation.

 This metadata check is not a provider pre-import guard. Production transport
 must also launch the qualified fresh-I-B worker before calling the factory.
 """
 import base64
 from pathlib import Path
 from nacl.signing import VerifyKey
 if document.get('signer')!=authority:raise ValueError('operator authority')
 VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']),base64.b64decode(document['signature'],validate=True))
 p=document['payload'];fields={'revision','role','environment_version','environment_source_hash','provider_resource_id','source_bundle_sha256','files','inventory_roots','dependency_scope','full_transitive_closure_claimed'}
 if set(p)!=fields or p['revision']!=REVISION or p['role']!='trusted-terminal-grader'or p['dependency_scope']!='provider-namespace-controlled'or p['full_transitive_closure_claimed']is not False:raise ValueError('qualified operator-only scope')
 for k in ['environment_source_hash','provider_resource_id','source_bundle_sha256']:sha(p[k])
 if not isinstance(p['environment_version'],str)or not p['environment_version']:raise ValueError('operator version')
 root=Path(root).resolve();actual={}
 if not isinstance(p['inventory_roots'],list)or not p['inventory_roots']:raise ValueError('operator inventory roots')
 for name in p['inventory_roots']:
  q=Path(name)
  if q.is_absolute()or'..'in q.parts or not q.parts:raise ValueError('relative operator root')
  path=root/q
  if path.is_symlink()or path.resolve()!=path.absolute()or not path.exists():raise ValueError('operator root alias')
  for f in([path]if path.is_file()else path.rglob('*')):
   if f.is_symlink():raise ValueError('operator resource symlink')
   if f.is_file():actual[str(f.relative_to(root))]={'size':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
 if actual!=p['files']:raise ValueError('exact operator resources/source mismatch')
 if p['provider_resource_id']!=digest({'files':p['files'],'scope':p['dependency_scope']}):raise ValueError('operator resource identity')
 return copy.deepcopy(p)

def validate_public(p):
 fields={'revision','environment_version','task_id','messages','tools','source_bundle_sha256','provider_resource_id','public_resources'}
 if not isinstance(p,dict)or set(p)!=fields or p['revision']!=REVISION or p['tools']!=[]or p['public_resources']!=[]:raise ValueError('public text actor excludes grader descriptors/cache resources')
 for k in['task_id','source_bundle_sha256','provider_resource_id']:sha(p[k])
 if not isinstance(p['environment_version'],str)or not p['environment_version']:raise ValueError('public environment version')
 if not isinstance(p['messages'],list)or not p['messages']or any(set(m)!={'role','content'}or m['role']not in('system','user')or not isinstance(m['content'],str)for m in p['messages']):raise ValueError('original public messages')
 return copy.deepcopy(p)

class PublicActor:
 """Receives only public DTO/callable; owner-host reflection is not isolated."""
 def __init__(self,descriptor,terminal):self.descriptor=validate_public(descriptor);self._terminal=terminal;self.sealed=False
 def reset(self):return copy.deepcopy(self.descriptor)
 def finish(self,text):
  if self.sealed or not isinstance(text,str)or len(text)>65536:raise ValueError('bounded unsealed original text')
  self.sealed=True;result=self._terminal(text)
  if set(result)!={'done','reward','classification','task_id','public_descriptor_sha256','trace_sha256'}or result['done']is not True or type(result['reward'])not in(int,float)or not math.isfinite(result['reward'])or result['classification']not in('positive','negative')or result['task_id']!=self.descriptor['task_id']or result['public_descriptor_sha256']!=digest(self.descriptor):raise ValueError('trusted terminal binding')
  sha(result['trace_sha256']);return copy.deepcopy(result)

class OperatorBroker:
 def __init__(self,admission,spec,session_factory):
  if admission.get('revision')!=REVISION or admission.get('role')!='trusted-terminal-grader'or spec.get('id')!='affine_rcore'or spec.get('max_turns')!=1 or spec.get('version')!=admission['environment_version']or spec.get('source_hash')!=admission['environment_source_hash']:raise ValueError('operator admission/environment source/version')
  if type(spec.get('num_samples'))is not int or spec['num_samples']<1 or type(spec.get('success_reward'))not in(int,float)or not math.isfinite(spec['success_reward']):raise ValueError('original task population/reward threshold')
  self.admission=copy.deepcopy(admission);self.spec=copy.deepcopy(spec);self.factory=session_factory;self.session=None;self.sealed=False
 def actor(self,index,seed):
  if self.session is not None or self.sealed or type(index)is not int or not 0<=index<self.spec['num_samples']or type(seed)is not int or seed<0:raise ValueError('single original task/index')
  self.session=self.factory(self.spec)
  try:
   reset=self.session.reset(index,seed)
   p={'revision':REVISION,'environment_version':self.spec['version'],'task_id':reset['task_hash'],'messages':[{'role':m['role'],'content':m['content']}for m in reset['messages']],'tools':reset['tools'],'source_bundle_sha256':self.admission['source_bundle_sha256'],'provider_resource_id':self.admission['provider_resource_id'],'public_resources':[]}
   self.public=validate_public(p);self.index=index;self.seed=seed;return PublicActor(self.public,self._finish)
  except BaseException:self.close();raise
 def _finish(self,text):
  if self.sealed or self.session is None:raise ValueError('sealed original terminal')
  self.sealed=True
  try:
   result=self.session.step({'text':text})
   trace=getattr(self.session,'trace',None)
   if trace is not None and getattr(trace,'info',{}).get('score_error'):raise ValueError('native grader failure cannot be a negative outcome')
   if result.get('done')is not True or result.get('observations')!=[]or type(result.get('reward'))not in(int,float)or not math.isfinite(result['reward']):raise ValueError('original scalar text grader')
   label='positive'if result['reward']>=self.spec['success_reward']else'negative'
   if result.get('classification')!=label:raise ValueError('original reward threshold')
   trace={'public_descriptor_sha256':digest(self.public),'index':self.index,'seed':self.seed,'action':{'text':text},'native_terminal':result}
   return {'done':True,'reward':result['reward'],'classification':label,'task_id':self.public['task_id'],'public_descriptor_sha256':digest(self.public),'trace_sha256':digest(trace)}
  finally:self.close()
 def replay(self,index,seed,text,expected):
  actual=self.actor(index,seed).finish(text)
  if canonical(actual)!=canonical(expected):raise ValueError('fresh original terminal replay')
  return actual
 def close(self):
  if self.session is not None:self.session.close();self.session=None
