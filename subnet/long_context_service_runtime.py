"""Prospective common 32K runtime; trusted deployment supplies native sessions."""
import copy,math
from .long_context_runtime import LongContextRuntime,wait_vram,validate_tokens

SERVICE_REVISION='cuda-bf16-sdpa-flash-sm86-selective-head-common-v1'

class LongContextServiceRuntime(LongContextRuntime):
 def __init__(self,checkpoint,files,environment,harness,session_factory):
  if not callable(session_factory):raise ValueError('trusted session factory required')
  self.session_factory=session_factory;self.configure(environment,harness)
  wait_vram();super().__init__(checkpoint,files)
 def configure(self,environment,harness=None):
  from .environments import EnvironmentSpec
  from . import harness as policy
  if environment.get('adapter')=='native_eog_broker':
   from .native_eog_adapter import VERSION
   self.spec=EnvironmentSpec(**environment)
   if self.spec.version!=VERSION or not self.spec.source_hash or not 1<=self.spec.max_turns<=128 or not 1<=self.spec.max_output_tokens<=512 or not 1<=self.spec.num_samples<=100000 or not math.isfinite(self.spec.success_reward):raise ValueError('native EOG signed spec budgets/version')
  else:self.spec=EnvironmentSpec.from_dict(environment)
  self.harness=policy.normalize(harness)
  if self.harness['max_output_tokens']>self.spec.max_output_tokens:raise ValueError('harness exceeds environment output budget')
  self.env_config=copy.deepcopy(environment);return self
 def for_environment(self,environment,harness=None):return copy.copy(self).configure(environment,harness)
 def prompt(self,messages,tools=()):
  from . import harness as policy
  return policy.render(self.tokenizer,messages,tools,self.harness)
 def profile(self):
  from .long_context_runtime import file_sha
  from importlib.metadata import version
  return {**super().profile(),'service_revision':SERVICE_REVISION,'service_source_sha256':file_sha(__file__),'candidate_score_reduction':'numpy-float32-sum','numpy_version':version('numpy')}
 def sample(self,prompt,seed,messages,turn):
  import torch
  from torch.nn.attention import sdpa_kernel,SDPBackend
  from . import harness as policy
  config=policy.turn_config(self.harness,turn);rng=torch.Generator(device='cuda').manual_seed(seed)
  if config['policy']=='visible-copy-candidates':
   opening,closing=config['input_tags'];visible='\n'.join(m['content'] for m in messages if m['role']=='user');start=visible.rfind(opening)
   if start<0 or closing not in visible[start+len(opening):]:raise ValueError('visible span missing')
   text=visible[start+len(opening):].split(closing,1)[0];before,after=config['output_tags'];config={**config,'policy':'candidates','candidates':[before+text+after,before+text+'!'+after]}
  if config['policy']=='candidates':
   candidates=[self.tokenizer.encode(s,add_special_tokens=False) for s in config['candidates']];scores=[]
   if any(not ids or len(ids)>config['max_output_tokens'] for ids in candidates):raise ValueError('candidate token budget')
   for ids in candidates:
    _,lp=self.compute(prompt,ids);scores.append(float(lp[range(len(ids)),ids].sum()))
   distribution=torch.softmax(torch.tensor(scores,device='cuda',dtype=torch.float32)/config['temperature'],-1)
   return candidates[int(torch.multinomial(distribution,1,generator=rng))]
  if config['policy']!='autoregressive':raise ValueError('unsupported long-context sampling policy')
  output=[]
  with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
   for _ in range(config['max_output_tokens']):
    if len(prompt)+len(output)>=32768:raise ValueError('long-context generation budget')
    hidden=self.model.base_model(torch.tensor([prompt+output],device='cuda'),use_cache=False).last_hidden_state[0,-1:]
    logits=self.model.lm_head(hidden)[0].float()/config['temperature'];probs=torch.softmax(logits,-1)
    if config['top_p']<1:
     values,indices=probs.sort(descending=True);values[values.cumsum(0)-values>config['top_p']]=0;probs=torch.zeros_like(probs).scatter(0,indices,values);probs/=probs.sum()
    token=int(torch.multinomial(probs,1,generator=rng));output.append(token)
    if token==self.tokenizer.eos_token_id:break
  return output
 def rollout(self,index,seed):
  from . import harness as policy
  session=self.session_factory(self.spec)
  try:
   env_seed=int(self.spec.config.get('seed',0));initial=session.reset(index,env_seed);messages=initial['messages'];tools=initial.get('tools',[]);turns=[];arrays=[]
   for i in range(self.spec.max_turns):
    prompt=self.prompt(messages,tools)
    if len(prompt)+self.harness['max_output_tokens']>32768:raise ValueError('signed long-context budget')
    output=self.sample(prompt,seed+i,messages,i);text=self.tokenizer.decode(output,skip_special_tokens=True);acts,lp=self.compute(prompt,output);proofs=self.build_proofs(acts,decode_batching_size=16,topk=128)
    if not proofs or any(p is None for p in proofs):raise ValueError('proof construction')
    result=session.step(policy.action(text,self.harness));turns.append(dict(prompt=prompt,output=output,text=text,proofs=proofs,observations=result['observations'],done=result['done'],reward=result['reward'],classification=result['classification']));arrays.append(lp)
    messages=messages+[dict(role='assistant',content=text)]+policy.observations(result['observations'],self.harness)
    if result['done']:break
   if not result['done']:raise ValueError('environment did not terminate')
   return dict(schema=2,env_id=self.spec.id,environment_version=self.spec.version,index=index,sample_index=index,seed=seed,env_seed=env_seed,task_hash=initial['task_hash'],reward=result['reward'],classification=result['classification'],turns=turns),arrays
  finally:session.close()
 def verify(self,rollout,arrays):
  import numpy as np
  from . import harness as policy
  from .proofs import validate_framing
  if rollout.get('schema')!=2 or rollout.get('sample_index')!=rollout.get('index') or rollout.get('env_id')!=self.spec.id or rollout.get('environment_version')!=self.spec.version:raise ValueError('required environment binding')
  if type(rollout.get('reward')) not in (int,float) or not math.isfinite(rollout['reward']):raise ValueError('reward type/finiteness')
  turns=rollout.get('turns',[])
  if not 0<len(turns)<=self.spec.max_turns or len(turns)!=len(arrays):raise ValueError('turn count')
  session=self.session_factory(self.spec)
  try:
   env_seed=int(self.spec.config.get('seed',0))
   if rollout.get('env_seed')!=env_seed:raise ValueError('environment seed')
   initial=session.reset(rollout['index'],env_seed)
   if rollout.get('task_hash')!=initial['task_hash']:raise ValueError('task hash')
   messages=initial['messages'];tools=initial.get('tools',[])
   for i,(turn,claimed) in enumerate(zip(turns,arrays)):
    prompt=self.prompt(messages,tools);output=turn['output']
    if turn['prompt']!=prompt:raise ValueError('context')
    validate_tokens(prompt,output,self.model.config.vocab_size)
    if len(output)>self.spec.max_output_tokens or len(output)>self.harness['max_output_tokens']:raise ValueError('output budget')
    text=self.tokenizer.decode(output,skip_special_tokens=True)
    if turn['text']!=text:raise ValueError('text')
    if type(turn.get('done')) is not bool or type(turn.get('reward')) not in (int,float) or not math.isfinite(turn['reward']):raise ValueError('turn outcome types')
    acts,lp=self.compute(prompt,output)
    if claimed.dtype!=np.float32 or claimed.shape!=lp.shape or not np.isfinite(claimed).all() or not np.allclose(claimed,lp,atol=1e-5,rtol=0):raise ValueError('probabilities')
    count=1+math.ceil(len(output)/16);validate_framing(turn['proofs'],count);proofs=self.verify_proofs(acts,turn['proofs'],decode_batching_size=16,topk=128)
    if len(proofs)!=count or any(x.exp_mismatches or x.mant_err_mean or x.mant_err_median for x in proofs):raise ValueError('TOPLOC')
    result=session.step(policy.action(text,self.harness))
    if turn['done']!=result['done'] or turn['reward']!=result['reward'] or turn['classification']!=result['classification'] or turn['observations']!=result['observations']:raise ValueError('native environment replay')
    if result['done'] and i!=len(turns)-1:raise ValueError('extra turns')
    messages=messages+[dict(role='assistant',content=text)]+policy.observations(result['observations'],self.harness)
   if not result['done'] or rollout['reward']!=result['reward'] or rollout['classification']!=result['classification']:raise ValueError('incomplete rollout/score')
   return True
  finally:session.close()
 def full_parameter_train(self,pairs,steps=1,lr=5e-5,beta=.1):
  from .long_context_service_training import full_parameter_train
  return full_parameter_train(self,pairs,steps=steps,lr=lr,beta=beta)
