"""Longer isolated retries of genuine original resources; no model/source changes."""
import concurrent.futures,json,os,re,signal,subprocess,sys,time,shutil
from pathlib import Path
names=sys.argv[1:] or ['multiswe','swelego','affine_nl2lib','affine_uuidctf','affine_oolong']
target=Path('state/multi-environment/original-resource-retries.json')
existing=json.loads(target.read_text()) if target.exists() else []
def run(name):
 start=time.time();env=dict(os.environ,PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 p=subprocess.Popen([sys.executable,'ops/new_subnet_environment_resources.py','--child',name],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True,env=env)
 timedout=False;disk_guard=False
 try:
  while True:
   try:out,err=p.communicate(timeout=5);break
   except subprocess.TimeoutExpired:
    if shutil.disk_usage(Path.cwd()).free<2*1024**3:
     disk_guard=True;raise
    if time.time()-start>600:raise
 except subprocess.TimeoutExpired:
  timedout=True;os.killpg(p.pid,signal.SIGTERM)
  try:out,err=p.communicate(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);out,err=p.communicate()
 rows=[json.loads(l.removeprefix('RESOURCE_RESULT ')) for l in out.splitlines() if l.startswith('RESOURCE_RESULT ')]
 row=rows[-1] if rows else {'source':name,'status':'process_error','error':err[-1200:]}
 row.update(probe_budget_seconds=600,seconds=round(time.time()-start,2))
 if timedout:
  row.update(status='operator_disk_headroom_guard' if disk_guard else 'longer_original_resource_timeout',error='Operator staging disk below2GiB guard; retry on spacious retained pod with original task preserved' if disk_guard else 'Actual original task load/reset exceeded600seconds; insufficient-budget evidence, not confirmed persistent external block')
  for n in set(re.findall(r'\bvf-[a-f0-9]{12}\b',out+err)):subprocess.run(['docker','rm','-f',n],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20)
 return row
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for row in pool.map(run,names):
  directory=target.parent/'resource-retry-rows';directory.mkdir(exist_ok=True)
  (directory/(row['source']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
  existing=json.loads(target.read_text()) if target.exists() else []
  existing=[r for r in existing if r['source']!=row['source']]+[row];target.write_text(json.dumps(existing,indent=2)+'\n');print(json.dumps(row),flush=True)
  p=Path('state/multi-environment/environment-execution-matrix.json');matrix=json.loads(p.read_text())
  for r in matrix:
   if r['source']==row['source']:r.update(resource_retry=row,resource_retry_evidence=str(target),original_task_loaded=row.get('task_loaded',False),local_reset=row.get('reset',False),status=row['status'])
  p.write_text(json.dumps(matrix,indent=2)+'\n')
