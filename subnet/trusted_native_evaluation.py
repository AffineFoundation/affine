"""Explicit operator-owned native evaluation; generation policy remains unchanged.

These scores are trusted-process diagnostics, never verified miner evidence.
"""
import hashlib,json,math,time
from . import harness
VERSION='trusted-native-generation-evaluation-v1'
POLICY=dict(version=VERSION,trust_scope='operator-owned-process-native-grader',proof_reverification=False,sampling_policy='unchanged-signed-runtime')
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def validate_policy(value):
 if not isinstance(value,dict)or set(value)!=set(POLICY)or any(type(value[k])is not type(v)or value[k]!=v for k,v in POLICY.items()):raise ValueError('explicit trusted native evaluation policy')
 return dict(value)
def rollout(runtime,index,seed,*,create_session,clock=time.monotonic):
 cfg=harness.normalize(runtime.harness)
 if cfg['policy']!='autoregressive'or cfg.get('turn_overrides'):raise ValueError('original free autoregressive evaluation harness')
 session=create_session(runtime.spec);start=clock();turns=[];generation=grading=0.
 try:
  initial=session.reset(index,int(runtime.spec.config.get('seed',0)));messages=initial['messages'];tools=initial.get('tools',[])
  for turn in range(runtime.spec.max_turns):
   prompt=runtime.prompt(messages,tools)
   if len(prompt)+cfg['max_output_tokens']>min(runtime.model.config.max_position_embeddings,8192):raise ValueError('evaluation context budget')
   at=clock();output=runtime.sample_output(prompt,seed,messages,turn,index,initial['task_hash']);generation+=clock()-at
   if not 0<len(output)<=cfg['max_output_tokens']or any(type(v)is not int or not 0<=v<runtime.model.config.vocab_size for v in output):raise RuntimeError('trusted generation token/cap violation')
   text=runtime.tokenizer.decode(output,skip_special_tokens=True)
   at=clock();result=session.step(harness.action(text,cfg));grading+=clock()-at
   if type(result['done'])is not bool or type(result['reward'])not in(int,float)or not math.isfinite(result['reward']):raise RuntimeError('native grader result schema')
   turns.append(dict(prompt_sha256=digest(prompt),output_sha256=digest(output),prompt_tokens=len(prompt),output_tokens=len(output)))
   messages=messages+[dict(role='assistant',content=text)]+harness.observations(result['observations'],cfg)
   if result['done']:break
  if not result['done']:raise RuntimeError('native environment did not terminate within signed turn cap')
  if result['classification']not in('positive','negative')or result['reward']!=(1 if result['classification']=='positive'else 0):raise RuntimeError('binary native math grade schema')
  return dict(env_id=runtime.spec.id,index=index,seed=seed,task_hash=initial['task_hash'],reward=result['reward'],classification=result['classification'],native_graded=True,verified=False,proof_verification_performed=False,trust_scope=POLICY['trust_scope'],sampling_policy=POLICY['sampling_policy'],turns=turns,timings_seconds=dict(total=clock()-start,generation=generation,native_grading=grading),probability_artifact_calls=0,TOPLOC_build_calls=0,TOPLOC_verify_calls=0)
 finally:session.close()
def evaluate(runtime,manifest,job,*,create_session,progress=None):
 from .protocol import entry
 validate_policy(job['trusted_evaluation_policy']);values=[];failures=[]
 for suite in job['heldout']:
  definition=entry(manifest,suite['env_id'])
  if set(suite['indices'])&set(definition['indices']):raise ValueError('heldout/training overlap')
  if len(suite['indices'])!=len(suite['seeds'])or len(set(suite['indices']))!=len(suite['indices']):raise ValueError('exact fixed evaluation cohort')
  selected=runtime.for_environment(definition['spec'],suite['harness'])
  for index,seed in zip(suite['indices'],suite['seeds']):
   if progress is not None:progress(dict(phase='task_started',env_id=suite['env_id'],index=index,seed=seed,completed_tasks=len(values),infrastructure_failures=len(failures)))
   try:values.append(rollout(selected,index,seed,create_session=create_session))
   except(ValueError,RuntimeError,KeyError)as error:failures.append(dict(env_id=suite['env_id'],index=index,seed=seed,error_type=type(error).__name__,error=str(error)[:300],infrastructure_failure=True))
   if progress is not None:progress(dict(phase='task_terminal',env_id=suite['env_id'],index=index,seed=seed,completed_tasks=len(values),infrastructure_failures=len(failures),completed_output_tokens=sum(t['output_tokens']for v in values for t in v['turns'])))
 return values,failures,dict(policy=POLICY,proof_verification_performed=False,miner_reward_evidence=False,historical_execution_proven=False)
