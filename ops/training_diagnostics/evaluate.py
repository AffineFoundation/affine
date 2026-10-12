"""Fresh generations on approved TRAIN tasks; not independent confirmation or miner evidence."""
import argparse, base64, fcntl, hashlib, importlib.metadata, json, math, os
from pathlib import Path
import signal, subprocess, sys, time

AUTHORITY='3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def file_hash(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as stream:
  while chunk:=stream.read(16*1024**2):h.update(chunk)
 return h.hexdigest()
def signed(d):
 from nacl.signing import VerifyKey
 if set(d)!={'payload','signer','signature'}or d['signer']!=AUTHORITY:raise ValueError('ROOT evaluation scope')
 VerifyKey(bytes.fromhex(AUTHORITY)).verify(canonical(d['payload']),base64.b64decode(d['signature'],validate=True))
 return d['payload']
def save(path,value):
 with Path(path).open('xb')as stream:stream.write(canonical(value))
 Path(path).chmod(0o600)
def wilson(successes,total):
 z=1.959963984540054;p=successes/total;den=1+z*z/total
 center=(p+z*z/(2*total))/den;half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
 return [max(0,center-half),min(1,center+half)]

def run(plan_path,check_only=False):
 envelope=json.loads(Path(plan_path).read_bytes());p=signed(envelope)
 if p['version']!='fresh-approved-training128-plan-v1'or file_hash(__file__)!=p['program_sha256']:raise ValueError('exact evaluation program')
 if p['production_mutation']is not False or p['optimizer_allocation']is not False or p['task_count']!=128:raise ValueError('private bounded evaluation')
 if not p['created_at']<=time.time()<p['expires_at']or not 0<p['expires_at']-p['created_at']<=7200:raise ValueError('bounded evaluation lifetime')
 root=Path(p['directory']);source=Path(p['source']);model=Path(p['checkpoint']['path'])
 for path in (root,source,model):
  if path.resolve()!=path or path.is_symlink():raise ValueError('canonical evaluation paths')
 for name,h in p['source_files'].items():
  path=source/name
  if path.resolve()!=path or file_hash(path)!=h:raise ValueError('pinned source bytes')
 for n,v in p['runtime_versions'].items():
  if importlib.metadata.version(n)!=v:raise ValueError('pinned runtime versions')
 if digest(p['checkpoint']['files'])!=p['checkpoint']['id']or set(x.name for x in model.iterdir())!=set(p['checkpoint']['files']):raise ValueError('checkpoint inventory')
 for name,h in p['checkpoint']['files'].items():
  if Path(name).name!=name or (model/name).is_symlink()or file_hash(model/name)!=h:raise ValueError('checkpoint full SHA')
 import importlib.util
 helper=root/'train_contract.py'
 if file_hash(helper)!=p['contract_sha256']:raise ValueError('exact separate TRAIN policy helper')
 module_spec=importlib.util.spec_from_file_location('train_contract',helper);contract=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(contract)
 training_cohort=contract.validate_plan(p,AUTHORITY)
 manifest=signed(p['manifest']);definition=manifest['environments'][0]
 indices=[i for suite in p['suites']for i in suite['indices']]
 sys.path.insert(0,str(source));sys.dont_write_bytecode=True
 from subnet.backend_jobs import install_source_loader
 install_source_loader(source)
 from subnet.math_corpus import question_key
 rows=json.loads((source/definition['spec']['config']['task_snapshot']).read_bytes())
 for item in training_cohort['tasks']:
  row=rows[item['index']]
  if digest(row)!=item['registry_row_sha256'] or question_key(row['data']['problem'])!=item['normalized_question_sha256']:raise ValueError('actual training registry identity differs')
 from subnet.owned_cached_evaluation import rollout
 from transformers import AutoTokenizer
 from subnet.environments import EnvironmentSpec,create_session
 from subnet import harness
 tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
 spec=EnvironmentSpec.from_dict(definition['spec']);session=create_session(spec);lengths=[]
 try:
  for index in indices:
   initial=session.reset(index,int(spec.config.get('seed',0)))
   lengths.append(len(harness.render(tokenizer,initial['messages'],initial.get('tools',[]),p['suites'][0]['harness'])))
 finally:session.close()
 maximum=min(json.loads((model/'config.json').read_bytes())['max_position_embeddings'],8192)
 if max(lengths)+p['suites'][0]['harness']['max_output_tokens']>maximum:raise ValueError('actual prompt plus output exceeds supported context')
 if check_only:return dict(checked=True,task_count=128,GPU_runs=0,training_membership_verified=True,independent_confirmation=False,max_prompt_tokens=max(lengths),max_output_tokens=p['suites'][0]['harness']['max_output_tokens'])
 lock=os.open(root/'evaluation.lease',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('reserved spare GPU must be idle')
 def expire(*args):raise TimeoutError('bounded evaluation deadline')
 signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(p['expires_at']-time.time())))
 output=root/'output';output.mkdir(mode=0o700)
 from subnet.backend_profiles import execution_profile
 from subnet.gpu_runtime import GPURuntime
 from subnet.environments import create_session
 from subnet import cached_sampling
 revision,_,_=execution_profile(manifest,'evaluate')
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 runtime=GPURuntime(str(model),p['checkpoint']['files'],definition['spec'],p['suites'][0]['harness'],runtime_revision=revision)
 original_sample=cached_sampling.sample;all_rows=[];started=time.monotonic()
 for shard,suite in enumerate(p['suites']):
  for index,seed in zip(suite['indices'],suite['seeds']):
   if time.time()>=p['expires_at']:raise TimeoutError('bounded evaluation deadline')
   traces=[];native=[];initials=[];at=time.monotonic()
   def capture(model,prompt,*args,**kwargs):
    answer=original_sample(model,prompt,*args,**kwargs)
    traces.append(dict(prompt_tokens=list(prompt),output_tokens=list(answer[0]),text=runtime.tokenizer.decode(answer[0],skip_special_tokens=True),seed=kwargs['seed']))
    return answer
   class Session:
    def __init__(self):self.inner=create_session(runtime.spec)
    def __getattr__(self,name):return getattr(self.inner,name)
    def reset(self,*args,**kwargs):
     result=self.inner.reset(*args,**kwargs);initials.append(result);return result
    def step(self,action):
     result=self.inner.step(action);native.append(result);return result
   cached_sampling.sample=capture
   try:
    row=rollout(runtime,index,seed,create_session=lambda spec:Session())
   except Exception as error:
    row=dict(env_id=suite['env_id'],index=index,seed=seed,classification=contract.failed_rollout_classification(native,error),reward=0,
      task_hash=initials[-1].get('task_hash')if initials else None,
      error_type=type(error).__name__,error=str(error)[:500],native_graded=bool(native),verified=False,proof_verification_performed=False)
   finally:cached_sampling.sample=original_sample
   eos=runtime.tokenizer.eos_token_id;eos_ids=set(eos if isinstance(eos,list)else[eos])
   row.update(checkpoint=p['checkpoint']['id'],cohort_sha256=p['cohort_sha256'],raw_turns=traces,native_results=native,
      cap_hit=any(len(t['output_tokens'])>=suite['harness']['max_output_tokens']for t in traces),
      budget_exhausted=any(len(t['output_tokens'])>=suite['harness']['max_output_tokens']and t['output_tokens'][-1]not in eos_ids for t in traces),
      terminal_eos=bool(traces and traces[-1]['output_tokens'] and traces[-1]['output_tokens'][-1] in eos_ids),
      generation_origin='fresh-policy-generation',profile_sha256=p['profile_sha256'],
      elapsed_seconds=time.monotonic()-at)
   save(output/('task-'+str(index)+'.json'),row);all_rows.append(row)
   print(json.dumps(dict(stage='task_completed',count=len(all_rows),correct=sum(r['classification']=='positive'for r in all_rows),unresolved=sum(r['classification']=='unresolved'for r in all_rows))),flush=True)
 correct=sum(r['classification']=='positive'for r in all_rows)
 result=dict(version=p['version'],plan_sha256=digest(envelope),checkpoint=p['checkpoint']['id'],tasks=128,correct=correct,accuracy=correct/128,
  wilson95=wilson(correct,128),unresolved=sum(r['classification']=='unresolved'for r in all_rows),errors=sum(r['classification']=='error'for r in all_rows),
  cap_hits=sum(r['cap_hit']for r in all_rows),budget_exhausted=sum(r['budget_exhausted']for r in all_rows),seconds=time.monotonic()-started,
  cohort_sha256=p['cohort_sha256'],profile_sha256=p['profile_sha256'],seeds_sha256=p['seeds_sha256'],generation_origin='fresh-policy-generation',raw_artifacts={f.name:file_hash(f)for f in output.glob('task-*.json')},production_mutation=False,optimizer_allocation=False,completed_at=time.time())
 save(output/'result.json',result);signal.alarm(0);os.close(lock);return result

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--check',action='store_true');args=parser.parse_args()
 print(json.dumps(run(args.plan,args.check),sort_keys=True))
