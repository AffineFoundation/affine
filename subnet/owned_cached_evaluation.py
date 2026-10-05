"""Prospective operator-owned cached diagnostics; never miner verification."""
import hashlib,json,math,time
from . import harness
VERSION='owned-cached-native-evaluation-v1'
POLICY=dict(version=VERSION,trust_scope='operator-owned-process-native-grader',proof_reverification=False)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def validate_policy(value):
 if not isinstance(value,dict)or set(value)!=set(POLICY)or any(type(value[k])is not type(v)or value[k]!=v for k,v in POLICY.items()):raise ValueError('explicit owned cached diagnostic policy')
 return dict(value)
def cohort(definition,suite,manifest,source_files):
 cfg=harness.normalize(suite['harness']);indices=suite['indices'];seeds=suite['seeds']
 if (cfg['version']!='text-tools-long-kv-v3'or cfg['policy']!='autoregressive'or cfg.get('turn_overrides')or definition['env_id']!=suite['env_id']):raise ValueError('explicit unmodified cached evaluator harness')
 if not 1<=len(indices)<=32 or len(indices)!=len(seeds)or len(set(indices))!=len(indices)or any(type(i)is not int or i<0 for i in indices+seeds):raise ValueError('fixed evaluation population')
 if set(indices)&set(definition['indices']):raise ValueError('heldout/training overlap')
 value=dict(version=VERSION,env_id=suite['env_id'],environment=definition['spec'],harness=cfg,indices=indices,seeds=seeds,model_runtime_revision=manifest['model_runtime_revision'],backend_profile=manifest['backend_profile'],source_files=source_files)
 return value,digest(value)
def rollout(runtime,index,seed,*,create_session,clock=time.perf_counter):
 from .cached_sampling import sample
 if getattr(runtime,'sampling_context',None)is not None:raise ValueError('owned diagnostic cannot stand in for forced miner evidence')
 cfg=harness.normalize(runtime.harness)
 if cfg['version']!='text-tools-long-kv-v3'or cfg['policy']!='autoregressive'or cfg.get('turn_overrides'):raise ValueError('owned diagnostic cached harness')
 env_seed=int(runtime.spec.config.get('seed',0));session=create_session(runtime.spec);started=clock();timings=dict(environment_reset=0.,prompt_render=0.,cached_generation=0.,decode=0.,native_grading=0.);turns=[];forwards=[]
 try:
  at=clock();initial=session.reset(index,env_seed);timings['environment_reset']+=clock()-at;messages=initial['messages'];tools=initial.get('tools',[])
  for turn in range(runtime.spec.max_turns):
   at=clock();prompt=runtime.prompt(messages,tools);timings['prompt_render']+=clock()-at
   if len(prompt)+cfg['max_output_tokens']>min(runtime.model.config.max_position_embeddings,8192):raise ValueError('evaluation context budget')
   at=clock();output,_=sample(runtime.model,prompt,seed=seed+turn,max_output_tokens=cfg['max_output_tokens'],temperature=cfg['temperature'],top_p=cfg['top_p'],eos_token_id=runtime.tokenizer.eos_token_id,telemetry=forwards.append);timings['cached_generation']+=clock()-at
   at=clock();text=runtime.tokenizer.decode(output,skip_special_tokens=True);timings['decode']+=clock()-at
   at=clock();result=session.step(harness.action(text,cfg));timings['native_grading']+=clock()-at
   if type(result['done'])is not bool or type(result['reward'])not in (int,float)or not math.isfinite(result['reward']):raise ValueError('native grader result types')
   turns.append(dict(prompt_sha256=digest(prompt),output_sha256=digest(output),prompt_tokens=len(prompt),output_tokens=len(output)))
   messages=messages+[dict(role='assistant',content=text)]+harness.observations(result['observations'],cfg)
   if result['done']:break
  if not result['done']:raise ValueError('native environment did not terminate')
  if result['classification']not in ('positive','negative')or result['reward']!=(1 if result['classification']=='positive'else 0):raise ValueError('binary native math outcome')
  return dict(env_id=runtime.spec.id,index=index,seed=seed,task_hash=initial['task_hash'],reward=result['reward'],classification=result['classification'],native_graded=True,verified=False,proof_verification_performed=False,trust_scope=POLICY['trust_scope'],turns=turns,timings_seconds={**timings,'total':clock()-started},generation_forward_calls=len(forwards),generation_input_tokens=sum(f['input_tokens']for f in forwards),cached_decode_calls=sum(f['cache_reused']for f in forwards),prefill_probability_replay_calls=0,TOPLOC_build_calls=0,TOPLOC_verify_calls=0)
 finally:session.close()
def evaluate(runtime,manifest,job,*,create_session):
 from .protocol import entry
 validate_policy(job['owned_evaluation_policy']);records=[];failures=[];cohorts=[]
 for suite in job['heldout']:
  definition=entry(manifest,suite['env_id']);frozen,identifier=cohort(definition,suite,manifest,job['source_files']);cohorts.append(dict(id=identifier,definition=frozen))
  selected=runtime.for_environment(definition['spec'],suite['harness'])
  for index,seed in zip(suite['indices'],suite['seeds']):
   try:row=rollout(selected,index,seed,create_session=create_session);row.update(cohort_sha256=identifier,checkpoint=manifest['checkpoint']['id']);records.append(row)
   except (ValueError,RuntimeError,KeyError)as exc:failures.append(dict(env_id=suite['env_id'],index=index,seed=seed,cohort_sha256=identifier,error_type=type(exc).__name__,error=str(exc)[:300]))
 return records,failures,dict(policy=POLICY,cohorts=cohorts,proof_verification_performed=False,miner_reward_evidence=False,historical_execution_proven=False)
def paired_summary(baseline,learned):
 # Original signed-job/report authentication belongs to the operator caller.
 if not baseline or len(baseline)!=len(learned):raise ValueError('complete equal cached populations')
 def rows(values):
  out={}
  for r in values:
   key=(r['env_id'],r['index'],r['seed'])
   if key in out or r.get('native_graded')is not True or r.get('verified')is not False or r.get('proof_verification_performed')is not False or r.get('trust_scope')!=POLICY['trust_scope']or 'error'in r or r['classification']not in ('positive','negative')or type(r['reward'])not in (int,float)or r['reward']!=(1 if r['classification']=='positive'else 0):raise ValueError('owned native-grade assurance; never verified miner rows')
   out[key]=r
  return out
 b,a=rows(baseline),rows(learned)
 if set(b)!=set(a):raise ValueError('same task indices/seeds required')
 for key in b:
  if b[key]['cohort_sha256']!=a[key]['cohort_sha256']or b[key]['task_hash']!=a[key]['task_hash']:raise ValueError('same cached harness/source/runtime/task required for both checkpoints')
 return dict(count=len(b),baseline_correct=sum(r['classification']=='positive'for r in b.values()),learned_correct=sum(r['classification']=='positive'for r in a.values()),paired_gains=sum(b[k]['classification']=='negative'and a[k]['classification']=='positive'for k in b),paired_losses=sum(b[k]['classification']=='positive'and a[k]['classification']=='negative'for k in b),execution_authenticated_here=False,proofs_verified=False,trust_scope=POLICY['trust_scope'],long_term_convergence_proven=False)
