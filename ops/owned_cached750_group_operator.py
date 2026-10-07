"""Default-off 24-original owned evaluator, separate from continuous fixed32.

ROOT signs all 24 requests and the group scope before execution. The GPU host
receives public signatures and full-readback ACKs, never the authority seed.
"""
import argparse,hashlib,json,math,os,stat,subprocess,sys,tempfile,time
from pathlib import Path
from subnet.backend_jobs import canonical,signed
from subnet.owned_cached_evaluation import cohort,validate_policy
from ops.owned_cached750_cohort import SOURCE,aggregate
from ops.owned_cached750_group_retention import validate_ack,GroupRetention,capacity_admission,retire_group,validate_scope
VERSION='owned-cached750-original-operator-v1'

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def private_json(path):
 path=Path(path);s=path.lstat()
 if not stat.S_ISREG(s.st_mode)or s.st_uid!=os.geteuid()or s.st_nlink!=1 or s.st_mode&0o077 or s.st_size>8*1024**2:raise ValueError('private owned single-link group file')
 data=path.read_bytes()
 after=path.lstat()
 fields=('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
 if any(getattr(after,k)!=getattr(s,k)for k in fields):raise ValueError('group file changed during read')
 return json.loads(data)
def save(path,value):
 path=Path(path)
 if path.exists():private_json(path)
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='group-progress-')
 try:
  with os.fdopen(fd,'wb')as f:f.write(canonical(value));f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def finite(value):return type(value)in(int,float)and math.isfinite(value)

def validate_originals(scope,jobs,authority):
 """Bind every exact 32-task request; no new random label/seed on observation."""
 if len(jobs)!=24:raise ValueError('exact 24 presigned originals')
 out={};cohorts={}
 for envelope in jobs:
  job=signed(envelope,authority);manifest=signed(job['manifest'],authority);jid=job['job_id'];binding=scope['original_jobs'].get(jid)
  if(binding is None or jid in out or not isinstance(jid,str)or not jid.replace('-','').replace('_','').isalnum()or
     digest(job)!=binding['job_sha256']or job.get('role')!='evaluate'or job['source_files']!=scope['source_files']or
     manifest['source_bundle']['sha256']!=SOURCE or manifest['checkpoint']!=scope['checkpoint']or job.get('checkpoint_cache')is not None or
     job.get('trusted_evaluation_policy')is not None or job.get('successor_calibration')is not None):raise ValueError('exact original group job/source/checkpoint')
  if manifest.get('tokenizer_binding',{})!={n:v for n,v in manifest['checkpoint']['files'].items()if 'token'in n or 'template'in n}:raise ValueError('actual arm tokenizer/template byte binding')
  validate_policy(job['owned_evaluation_policy'])
  if job['runtime_versions']!=scope['runtime_versions']:raise ValueError('exact signed group runtime versions')
  if any(not finite(job.get(k))for k in('created_at','expires_at'))or not 0<job['expires_at']-job['created_at']<=7200 or job['expires_at']>scope['expires_at']:raise ValueError('bounded original job lifetime')
  group=next(g for g in scope['groups']if g['group']==binding['group']);heldout=[{k:v for k,v in group.items()if k!='group'}]
  if job['heldout']!=heldout:raise ValueError('one distinct original32 group')
  definition=next(e for e in manifest['environments']if e['env_id']=='affine_math')
  if definition['indices']!=scope['mining_indices'] or len(definition['indices'])!=6746 or len(set(definition['indices']))!=6746 or any(type(i)is not int or not 0<=i<7496 for i in definition['indices']):raise ValueError('full native training split')
  if any(type(i)is not int or type(s)is not int for i,s in zip(group['indices'],group['seeds'])):raise ValueError('integer group framing')
  _,cohort_sha=cohort(definition,heldout[0],manifest,job['source_files']);cohorts[jid]=cohort_sha;out[jid]=(envelope,job,manifest)
 if set(out)!=set(scope['original_jobs']):raise ValueError('complete original identity coverage')
 return out,cohorts

def validate_terminal(status,job,now):
 if(status.get('job_id')!=job['job_id']or status.get('phase')not in('complete','failed')or type(status.get('exit_code'))is not int or
    (status['phase']=='complete')!=(status['exit_code']==0)or any(not finite(status.get(k))for k in('started_at','finished_at'))or
    not job['created_at']<=status['started_at']<=status['finished_at']<=now):raise ValueError('actual original terminal/time')
 if status.get('job_sha256',digest(job))!=digest(job):raise ValueError('terminal original digest changed')
 for k in('runner_pid','child_pid'):
  if type(status.get(k))is not int or status[k]<=0 or not str(status.get(k+'_ticks','')).isdigit():raise ValueError('original process-start framing')
 return status

def validate_report(report,job,manifest,cohort_sha):
 """Complete actual native outcomes; infrastructure failures never zero-filled."""
 if(report.get('job_id')!=job['job_id']or report.get('job_sha256')!=digest(job)or report.get('role')!='evaluate'or
    report.get('checkpoint')!=manifest['checkpoint']['id']or report.get('epoch')!=manifest['epoch']or
    report.get('source_files')!=job['source_files']or report.get('runtime_versions')!=job['runtime_versions']or
    report.get('chain_transactions')is not False or report.get('success')is not True or not finite(report.get('completed_at'))or
    not job['created_at']<=report['completed_at']<job['expires_at']):raise ValueError('authentic original report provenance/lifetime')
 from subnet.backend_profiles import execution_profile
 _,profile,policy=execution_profile(manifest,'evaluate')
 if report.get('backend_profile')!=profile or report.get('numerical_policy')!=policy or report.get('operator')!=job['manifest']['signer']:raise ValueError('exact native execution profile/operator')
 validate_policy(report['owned_cached_evaluation']['policy'])
 expected=job['heldout'][0];rows=report['heldout'];keys=[(v['index'],v['seed'])for v in rows]
 if(report.get('heldout_failures')or len(rows)!=len(expected['indices']) or len(set(keys))!=len(expected['indices']) or set(keys)!=set(zip(expected['indices'],expected['seeds']))):raise ValueError('full original32 native cohort, no incomplete score')
 for v in rows:
  if(v.get('env_id')!='affine_math'or v.get('checkpoint')!=manifest['checkpoint']['id']or v.get('cohort_sha256')!=cohort_sha or
     v.get('native_graded')is not True or v.get('verified')is not False or v.get('proof_verification_performed')is not False or
     v.get('trust_scope')!='operator-owned-process-native-grader'or type(v.get('reward'))not in(int,float)or
     v.get('classification')not in('positive','negative')or v['reward']!=(1 if v['classification']=='positive'else 0)or
     not isinstance(v.get('task_hash'),str)or len(v['task_hash'])!=64):raise ValueError('exact native outcome assurance')
 return report

class GroupOperator:
 """Adapter must observe physical originals and authenticate source before launch."""
 def __init__(self,envelope,authority,workspace,source_files,jobs,transport,*,clock=time.time):
  self.envelope=envelope;self.authority=authority;self.root=Path(workspace);self.files=source_files;self.transport=transport;self.clock=clock
  self.scope=validate_scope(envelope,authority,workspace,source_files,now=clock());self.originals,self.cohorts=validate_originals(self.scope,jobs,authority)
  self.binding=digest({k:v for k,v in self.scope.items()if k not in('created_at','expires_at')});self.lease=None
 def __enter__(self):
  self.lease=GroupRetention(self.envelope,self.authority,self.root,self.files,now=self.clock());self.lease.__enter__();self.journal=self.root/'.owned-cached-group-retention'/'operator.json'
  try:
   if self.journal.exists():
    self.progress=private_json(self.journal)
    if self.progress.get('version')!=VERSION or self.progress.get('binding_sha256')!=self.binding:raise ValueError('original group journal changed')
   else:
    self.progress=dict(version=VERSION,binding_sha256=self.binding,originals={j:dict(group=self.scope['original_jobs'][j]['group'],phase='declared')for j in self.originals},production_disposal_changed=False);save(self.journal,self.progress)
  except BaseException:
   self.lease.__exit__(*sys.exc_info());self.lease=None;raise
  return self
 def __exit__(self,*args):
  if self.lease is not None:self.lease.__exit__(*args);self.lease=None
 def step(self):
  if self.lease is None:raise ValueError('active exclusive group lease required')
  validate_scope(self.envelope,self.authority,self.root,self.files,now=self.clock())
  if private_json(self.journal)!=self.progress:raise ValueError('group journal changed')
  for jid in sorted(self.originals,key=lambda j:self.scope['original_jobs'][j]['group']):
   envelope,job,manifest=self.originals[jid];row=self.progress['originals'][jid]
   if row['phase']=='durable_ACK':continue
   if row['phase']=='declared':
    if self.clock()<job['created_at']:return dict(status='waiting-original-start-window',job_id=jid,new_job_started=False)
    if not job['created_at']<=self.clock()<job['expires_at']:return dict(status='expired-unissued',job_id=jid,new_job_started=False)
    capacity=capacity_admission(self.lease,self.transport.free_bytes())
    if not capacity['admitted']:return dict(status='capacity-deferred',job_id=jid,**capacity)
    if not self.transport.idle():return dict(status='physical-reservation-deferred',job_id=jid)
    # Intent precedes the mutating launch. Loss/exception never resets this row.
    row.update(phase='dispatch_attempted',original_job_sha256=digest(job),attempted_at=self.clock());save(self.journal,self.progress)
    self.transport.launch(envelope)
    return dict(status='observing-original',job_id=jid,new_job_started=True)
   status=self.transport.status(jid)
   if status.get('phase')not in('complete','failed'):return dict(status='observing-original',job_id=jid,original_phase=status.get('phase'),new_job_started=False)
   validate_terminal(status,job,self.clock())
   if self.transport.live(status):return dict(status='terminal-process-still-live',job_id=jid)
   # Failed/absent reports need explicit infrastructure reconciliation. Never
   # fabricate a worker report, fill zero outcomes, or launch a replacement.
   if status['exit_code']!=0:return dict(status='original-infrastructure-failure',job_id=jid,model_reward=None,cache_retained=True)
   report=self.transport.report(jid);validate_report(report,job,manifest,self.cohorts[jid]);row.update(phase='terminal-awaiting-ACK',terminal_sha256=digest(status),report_sha256=digest(report));save(self.journal,self.progress)
   ack_envelope=self.transport.ack(jid)
   if ack_envelope is None:return dict(status='awaiting-full-R2-ACK',job_id=jid,cache_retained=True)
   ack=signed(ack_envelope,self.authority)
   validate_ack(ack,envelope,report,status,str(self.root),self.scope['original_jobs'][jid]['group'])
   if(ack.get('version')!='owned-cached-evaluation-durable-ack-v1'or ack.get('durable_report_full_readback')is not True or ack.get('workspace')!=str(self.root)or
      ack.get('original_job')!=envelope or ack.get('original_report')!=report or ack.get('job_sha256')!=digest(job)or ack.get('report_sha256')!=digest(report)or ack.get('checkpoint')!=manifest['checkpoint']):raise ValueError('exact original full durable ACK')
   row.update(phase='durable_ACK',ack_sha256=digest(ack_envelope));save(self.journal,self.progress)
   return dict(status='original-ACK-complete',job_id=jid,cache_retained=True)
  acks=[self.transport.ack(j)for j in self.originals]
  if any(a is None for a in acks):raise ValueError('original ACK disappeared')
  for j,a in zip(self.originals,acks):
   if digest(a)!=self.progress['originals'][j]['ack_sha256']:raise ValueError('original ACK changed before retirement')
  result=retire_group(self.envelope,acks,self.authority,self.root,self.files,now=self.clock(),live=self.transport.live)
  if result['status']!='complete':return result
  reports=[signed(a,self.authority)['original_report']for a in acks]
  plan=dict(dispatch_allowed=False,groups=self.scope['groups'],experiment_id=self.scope['experiment_id'],cohort_sha256=digest(self.scope['groups']))
  score=aggregate(plan,reports);self.progress.update(phase='complete',result=result,score=score,completed_at=self.clock());save(self.journal,self.progress)
  return dict(status='complete',score=score,retirement=result)

class GroupACKPublisher:
 """CPU caller owns bucket/signing; the remote group operator receives ACKs only.

 Caller first observes these exact files on the scope-bound physical host. This
 API authenticates/binds them and performs real complete bucket readbacks; it
 never invents worker outcomes or turns missing originals into zero scores.
 """
 def __init__(self,scope_envelope,authority,jobs,bucket,signer,*,clock=time.time):
  self.scope=signed(scope_envelope,authority);self.authority=authority;self.clock=clock;self.bucket=bucket;self.signer=signer
  validate_scope(scope_envelope,authority,self.scope['workspace'],self.scope['source_files'],now=clock());self.originals,self.cohorts=validate_originals(self.scope,jobs,authority)
 def publish(self,jid,report,status,*,physical_original_absent):
  if physical_original_absent is not True:raise ValueError('physical original terminal/process absence required')
  if not self.scope['created_at']<=self.clock()<self.scope['expires_at']:raise ValueError('fresh group publisher scope')
  envelope,job,manifest=self.originals[jid];validate_terminal(status,job,self.clock())
  if status['exit_code']!=0:raise ValueError('failed original has no invented evaluation report')
  validate_report(report,job,manifest,self.cohorts[jid])
  # Content-addressed objects cannot substitute another report under this ACK.
  records={'original-job':envelope,'original-report':report,'original-terminal':status};bindings={}
  for kind,value in records.items():
   raw=canonical(value)
   if len(raw)>8*1024**2:raise ValueError('bounded original durability object')
   h=hashlib.sha256(raw).hexdigest();key='private/owned-cached-all750/'+digest(job)+'/'+kind+'/'+h+'.json';self.bucket.put(key,raw)
   if self.bucket.get(key)!=raw:raise ValueError('full original R2 readback failed; no ACK')
   bindings[kind]=dict(key=key,sha256=h,bytes=len(raw))
  payload=dict(version='owned-cached-evaluation-durable-ack-v1',workspace=self.scope['workspace'],original_job=envelope,original_report=report,
   durable_report_full_readback=True,job_sha256=digest(job),report_sha256=digest(report),checkpoint=manifest['checkpoint'],
   original_terminal=status,original_terminal_sha256=digest(status),full_readback_objects=bindings,group=self.scope['original_jobs'][jid]['group'])
  ack=self.signer(payload)
  if signed(ack,self.authority)!=payload:raise ValueError('exact approved ROOT ACK signer')
  raw=canonical(ack);key='private/owned-cached-all750/'+digest(job)+'/durable-ACK/'+digest(ack)+'.json';self.bucket.put(key,raw)
  if self.bucket.get(key)!=raw:raise ValueError('full ROOT ACK readback failed')
  return ack

class LocalOriginalTransport:
 """Runs on the ROOT-qualified host, public signed inputs only, no credentials."""
 def __init__(self,scope,authority):
  self.scope=scope;self.root=Path(scope['workspace']);self.code=Path(scope['source_path']);self.authority=authority
  if self.code.resolve()!=self.code or {str(p.relative_to(self.code)):hashlib.sha256(p.read_bytes()).hexdigest()for p in(self.code/'subnet').glob('*.py')}!=scope['source_files']:raise ValueError('exact full frozen177 runtime before launch')
  if scope.get('operator_file_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise ValueError('explicit ROOT operator pin')
  cpu=Path(__file__).parents[1];required=('ops/owned_cached750_group_operator.py','ops/owned_cached750_group_retention.py','ops/owned_cached750_cohort.py','ops/owned_cached_group_frozen_cache.py')
  if set(scope['operator_dependency_pins'])!=set(required)or any(hashlib.sha256((cpu/n).read_bytes()).hexdigest()!=scope['operator_dependency_pins'][n]for n in required):raise ValueError('full qualified CPU group dependency closure')
  if hashlib.sha256(Path('/etc/machine-id').read_bytes()).hexdigest()!=scope['machine_id_sha256']:raise ValueError('ROOT-bound physical group host')
  if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).strip()!=scope['gpu_uuid']:raise ValueError('ROOT-bound physical group GPU')
  from importlib.metadata import version
  if {k:version(k)for k in scope['runtime_versions']}!=scope['runtime_versions']:raise ValueError('ROOT-bound runtime installation')
 def free_bytes(self):return __import__('shutil').disk_usage(self.root).free
 def idle(self):
  if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10).strip():return False
  from subnet.evaluator_cache_lifecycle import live_original
  return not any(live_original(private_json(p))for p in(self.root/'runner-status').glob('*.json')if not p.name.endswith('-cache-retention.json'))
 def launch(self,envelope):
  job=signed(envelope,self.authority);jid=job['job_id'];p=self.root/(jid+'.json')
  if p.exists():raise ValueError('existing original remote request; never launch twice')
  fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
  with os.fdopen(fd,'wb')as f:f.write(canonical(envelope));f.flush();os.fsync(f.fileno())
  # -B suppresses writes; a private empty prefix also prevents reading stale
  # scientific .pyc files from the source checkout. Every original reuses an
  # empty namespace, never a cached execution artifact.
  pycache=self.root/'.scientific-bytecode-unused'
  if pycache.is_symlink():raise ValueError('private scientific bytecode namespace')
  pycache.mkdir(mode=0o700,exist_ok=True)
  if any(pycache.rglob('*')):raise ValueError('scientific bytecode namespace must remain empty')
  # Per-original remote_runner acquires the normal inherited checkpoint lease.
  log=self.root/(jid+'-runner.log')
  with log.open('xb')as out:
   os.fchmod(out.fileno(),0o600);child=subprocess.Popen([sys.executable,'-B','-m','subnet.remote_runner',str(p),'--authority',self.authority,'--workspace',str(self.root)],cwd=self.code,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPYCACHEPREFIX=str(pycache)))
  if not hasattr(self,'children'):self.children={}
  self.children[jid]=child
  return child.pid
 def status(self,jid):
  from subnet.remote_runner import probe
  if jid in getattr(self,'children',{}):self.children[jid].poll()
  return probe(self.root,jid,physical=True)
 def live(self,status):
  from subnet.evaluator_cache_lifecycle import live_original
  return live_original(status)
 def report(self,jid):return private_json(self.root/'jobs'/jid/'report.json')
 def ack(self,jid):
  p=self.root/'durable-evaluation-acks'/(jid+'.json');return private_json(p)if p.exists()else None

def main():
 p=argparse.ArgumentParser();p.add_argument('--scope',required=True);p.add_argument('--authority',required=True);p.add_argument('--original-jobs',nargs=24,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();envelope=private_json(a.scope);scope=signed(envelope,a.authority);jobs=[private_json(p)for p in a.original_jobs]
 if not a.execute:print(json.dumps({'version':VERSION,'dispatch_allowed':False,'reason':'explicit --execute and ROOT scope required'}));return
 transport=LocalOriginalTransport(scope,a.authority)
 with GroupOperator(envelope,a.authority,scope['workspace'],scope['source_files'],jobs,transport)as operator:
  while True:
   result=operator.step();print(json.dumps(result,sort_keys=True),flush=True)
   if result['status']in('complete','original-infrastructure-failure','expired-unissued'):return
   time.sleep(2)
if __name__=='__main__':main()
