"""Bounded original-source task loading and safe reset attempts, no training/chain."""
import argparse,concurrent.futures,importlib,json,os,re,signal,subprocess,sys,time
from pathlib import Path
from subnet.environments import build_spec,source_inventory,_taskset,create_session
PREFIX='RESOURCE_RESULT '
GUARDED={'affine_agent':'native tool registration imports dataset tools.py via general_agent_v1.corpus.load_task_attrs; requires isolated tool server before reset', 'affine_eog':'native enterprise MCP server setup requires separately isolated trusted server/runtime; no host dataset tool execution attempted'}

def flags_config(flags):
 cfg={}
 for i in range(0,len(flags),2):
  key=flags[i].removeprefix('--env.taskset.').replace('-','_').split('.');value=flags[i+1]
  if value in ('True','False'):value=value=='True'
  else:
   try:value=json.loads(value)
   except (ValueError,TypeError):pass
  dest=cfg
  for k in key[:-1]:dest=dest.setdefault(k,{})
  dest[key[-1]]=value
 return cfg

def safe(s):
 s=re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?<redacted-query>',str(s))
 for name,value in os.environ.items():
  if ('TOKEN' in name or 'SECRET' in name or 'PASSWORD' in name) and len(value)>12:s=s.replace(value,'<redacted>')
 return s[:900]

def child(name):
 row={'source':name,'task_loaded':False,'reset':False,'reward':False,'tool_execution':False};start=time.time();session=None
 inv=source_inventory()[name];row.update(module=inv['module'],source_manifest_hash=inv['source_manifest_hash'],source_definition=inv['source'])
 def emit():
  row['seconds']=round(time.time()-start,2);print(PREFIX+json.dumps(row),flush=True)
 try:
  cfg=flags_config(inv['source'].get('extra_flags',[]));row['operator_taskset_config']=cfg
  spec=build_spec(name,{'taskset':cfg},num_samples=1,max_turns=2);row['source_hash']=spec.source_hash
  taskset=_taskset(spec);module=importlib.import_module(inv['module']+'.taskset')
  hints={k:v for k,v in vars(module).items() if k in {'DATASET','DATASET_NAME','REVISION','SPLIT','REPO','CORPUS_DATASET','CORPUS_REPO','DATA_EPOCH'} and isinstance(v,(str,int))};row['resource_constants']=hints
  row['status']='task_loading';emit()
  task=next(iter(taskset),None)
  if task is None:raise RuntimeError('original configured taskset yielded no task')
  row.update(task_loaded=True,task_name=task.data.name,task_hash=task.hash,task_class=type(task).__name__,image=task.data.image,workdir=task.data.workdir,needs_container=task.NEEDS_CONTAINER,status='task_loaded');emit()
  if name in GUARDED:
   row.update(status='safe_runtime_isolation_required',blocker=GUARDED[name]);return
  session=create_session(spec);reset=session.reset(0,0)
  row.update(reset=True,status='actual_original_reset_passed',reset_task_hash=reset['task_hash'],messages=len(reset['messages']),tools=[t['function']['name'] for t in reset['tools']])
  # Never invoke dataset-authored graders in this resource inspection pass.
 except Exception as e:row.update(status='concrete_resource_error',error=type(e).__name__+': '+safe(e))
 finally:
  if session:
   try:session.close()
   except Exception as e:row['cleanup_error']=safe(e)
  emit()

def main():
 p=argparse.ArgumentParser();p.add_argument('--child');p.add_argument('--timeout',type=int,default=55);args=p.parse_args()
 if args.child:child(args.child);return
 target=Path('state/multi-environment');matrixpath=target/'environment-execution-matrix.json'
 existing=json.loads(matrixpath.read_text());names=[r['source'] for r in existing if r['status'] in {'not_yet_probed_original_resources_required','original_resource_budget_pending'}]
 evidence=[]
 def run(name):
  env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(Path.cwd()))
  started=time.time();proc=subprocess.Popen([sys.executable,__file__,'--child',name],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env,start_new_session=True)
  timedout=False
  try:stdout,stderr=proc.communicate(timeout=args.timeout)
  except subprocess.TimeoutExpired:
   timedout=True;os.killpg(proc.pid,signal.SIGTERM)
   try:stdout,stderr=proc.communicate(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);stdout,stderr=proc.communicate()
  results=[json.loads(line[len(PREFIX):]) for line in stdout.splitlines() if line.startswith(PREFIX)]
  row=results[-1] if results else {'source':name,'task_loaded':False,'reset':False}
  if timedout:row.update(status='actual_original_resource_timeout',error=f'original task load/reset exceeded {args.timeout}s',seconds=round(time.time()-started,2))
  elif proc.returncode and row.get('status') not in ('concrete_resource_error',):row.update(status='resource_probe_process_failure',error='exit '+str(proc.returncode))
  # Only container names printed by this private child pipe can be cleaned.
  owncontainers=sorted(set(re.findall(r'\bvf-[a-f0-9]{12}\b',stdout+stderr)))
  if timedout:
   for name_container in owncontainers:subprocess.run(['docker','rm','-f',name_container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
  row['owned_containers_cleanup_on_timeout']=owncontainers if timedout else []
  revisions=sorted(set(re.findall(r'/resolve/([a-f0-9]{40})/',stdout+stderr)));row['observed_dataset_revisions']=revisions
  return row
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  future={pool.submit(run,n):n for n in names}
  for f in concurrent.futures.as_completed(future):
   row=f.result();evidence.append(row);(target/'original-resource-attempts.json').write_text(json.dumps(evidence,indent=2)+'\n')
   # Re-read to preserve parallel root/parent additions.
   matrix=json.loads(matrixpath.read_text())
   for r in matrix:
    if r['source']==row['source']:
     r.update(original_task_loaded=row.get('task_loaded',False),local_reset=row.get('reset',False),status=row['status'],resource_attempt_evidence='state/multi-environment/original-resource-attempts.json',resource_attempt=row)
   matrixpath.write_text(json.dumps(matrix,indent=2)+'\n');print(json.dumps(row),flush=True)
if __name__=='__main__':main()
