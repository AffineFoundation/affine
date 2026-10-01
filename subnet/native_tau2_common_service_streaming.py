"""Isolated operator-broker Tau2 common pipeline. Never submits chain weights.

Only owned UID131 is supported. This is not a general external-miner server.
R2 credentials/signing keys remain operator-local; remote jobs get object caps.
"""
import argparse,base64,copy,gc,hashlib,io,json,math,os,pathlib,shlex,shutil,subprocess,sys,tarfile,time,zipfile
from .long_context_runtime import AUTHORITY,canonical,digest,file_sha,authenticate,wait_vram
from .native_tau2_common_search_contract import preference_pair
from .native_tau2_common_artifacts import pack_sample,unpack_sample,admit_frozen_sample,FREEZE_VERSION
from .native_tau2_common_bridge import heldout_contract,REFERENCE
VERSION='controlled-native-tau2-common-streaming-service-v2'
HOTKEY='5E68nqmVj1gSjoJHbusG2o4SiJ1dq17M7QG5PVzFKK2ic49u'
IDENTITY='598fa5ced6b34e5123ba0033c0af4536c0f53c480e3143bbda14f851486e7d90'
ENV={'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
TRAIN_POLICY={'revision':'native-tau2-same-context-agent-mean-dpo-full-bf16-flash-v1','reference':REFERENCE,'optimizer':'AdamW','steps':1,'learning_rate':5e-5,'beta':.1,'full_model_finetune':True,'auxiliary_tokens_in_loss':False,'min_free_vram_mib':20480,'wait_seconds':1800,'allocator_cap_bytes':8589934592}
ROOT=pathlib.Path(__file__).resolve().parent.parent

def sign(value,key):return {'payload':value,'signer':key.verify_key.encode().hex(),'signature':base64.b64encode(key.sign(canonical(value)).signature).decode()}
def write(path,value):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(canonical(value));path.chmod(0o600)
def private_dir(path):path=pathlib.Path(path);path.mkdir(parents=True,exist_ok=True);path.chmod(0o700);return path
def pid_identity(pid):
 try:
  fields=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return {'pid':pid,'state':fields[0],'start_ticks':fields[19]}
 except FileNotFoundError:return None

def disk_guard(path,minimum=1073741824):
 if shutil.disk_usage(path).free<minimum:raise RuntimeError('waiting-disk-capacity')
def validate_training_report(report,job):
 if report.get('job_sha256')!=digest(job):raise ValueError('training report signed job identity')
 if canonical(report.get('training_policy'))!=canonical(job['training_policy']) or report.get('optimizer_steps')!=1 or report.get('full_model_finetune') is not True or report.get('auxiliary_tokens_in_loss') is not False:raise ValueError('training report approved objective policy')
 return report

def evaluation_completion_gate(before,after,before_contract,after_contract):
 contracts=(before_contract,after_contract)
 for contract in contracts:
  body={k:v for k,v in contract.items() if k!='dataset_id'}
  if contract.get('dataset_id')!=digest(body):return {'complete':False,'reason':'heldout-contract-hash'}
 if before_contract['dataset_id']!=after_contract['dataset_id']:return {'complete':False,'reason':'heldout-dataset-drift'}
 for report,contract in zip((before,after),contracts):
  if report.get('dataset_id')!=contract['dataset_id'] or report.get('fixed_user_sha256')!=digest(contract['fixed_auxiliary_descriptor']):return {'complete':False,'reason':'heldout-plan-binding'}
  if report.get('all_tasks_completed') is not True or type(report.get('completed_count')) is not int or report['completed_count']!=16 or type(report.get('error_count')) is not int or report['error_count']!=0:return {'complete':False,'reason':'heldout-partial-or-errors'}
  rows=report.get('records',[]);expected=contract.get('heldout_tasks',[])
  if len(rows)!=16 or len(expected)!=16 or {r.get('index') for r in rows}!=set(range(16,32)):return {'complete':False,'reason':'heldout-exact16-coverage'}
  by_index={r['index']:r for r in expected}
  for row in rows:
   target=by_index.get(row['index'])
   if type(row.get('index')) is not int or row.get('verified') is not True or not target or row.get('task_hash')!=target['task_hash'] or type(row.get('seed')) is not int or row['seed']!=target['seed'] or type(row.get('reward')) not in (int,float) or not math.isfinite(row['reward']):return {'complete':False,'reason':'heldout-exact-task-seed-or-audit'}
 return {'complete':True,'reason':'both-exact16-full-native-model-audits-same-fixed-dataset'}

def source_guard(files):
 for name,expected in files.items():
  if name.startswith('/') or '..' in name.split('/') or not name.endswith('.py'):raise ValueError('source path')
  path=ROOT/name
  if path.is_symlink() or file_sha(path)!=expected:raise ValueError('approved common source closure')

def remote_job(path):
 """Fixed signed remote train/upload modes; authentication before artifact reads."""
 job=authenticate(json.loads(pathlib.Path(path).read_bytes()),AUTHORITY);source_guard(job['source_files'])
 if job.get('payable') is not False or job.get('chain_transactions') is not False:raise ValueError('nonpayable job')
 role=job.get('role')
 if role=='native-tau2-object-upload-v1':
  import urllib.request,urllib.parse
  for row in job['objects']:
   p=pathlib.Path(row['path']);url=urllib.parse.urlparse(row['put_url'])
   if url.scheme!='https' or url.hostname!=job['approved_r2_host'] or url.username or url.password or p.is_symlink() or file_sha(p)!=row['sha256'] or p.stat().st_size!=row['size']:raise ValueError('minimal exact-object PUT capability')
   with p.open('rb') as body:
    request=urllib.request.Request(row['put_url'],data=body,headers={'Content-Type':'application/octet-stream','Content-Length':str(row['size'])},method='PUT')
    with urllib.request.urlopen(request,timeout=300) as response:
     if response.status not in (200,201,204):raise ValueError('object PUT failed')
  write(job['receipt'],{'job_sha256':digest(job),'completed_at':time.time(),'uploaded':[{'sha256':r['sha256'],'size':r['size']} for r in job['objects']]});return
 if role!='native-tau2-agent-full-train-v1' or canonical(job['training_policy'])!=canonical(TRAIN_POLICY):raise ValueError('qualified full-agent job policy')
 from .long_context_runtime import runtime_environment
 if job['runtime_environment']!=runtime_environment():raise ValueError('training interpreter/packages')
 pair=job['pair']
 if pair['checkpoint']!=job['descriptor']['checkpoint'] or pair['auxiliary_tokens_in_loss'] is not False or pair['objective']!='agent-only-native-outcome-preference-v1':raise ValueError('agent preference current checkpoint/mask')
 wait_vram(TRAIN_POLICY['min_free_vram_mib'],TRAIN_POLICY['wait_seconds'])
 import torch
 import torch.nn.functional as F
 from .native_role_cuda import NativeAgentCUDARuntime
 from .long_context_training import sequence_logprob,configure_full_training
 torch.cuda.set_per_process_memory_fraction(TRAIN_POLICY['allocator_cap_bytes']/torch.cuda.get_device_properties(0).total_memory)
 runtime=NativeAgentCUDARuntime(job['checkpoint_path'],job['descriptor']).runtime
 def parameter_hash(p):return hashlib.sha256(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
 before={n:parameter_hash(p) for n,p in runtime.model.named_parameters()}
 with torch.no_grad():ref_chosen=float(sequence_logprob(runtime,pair['prompt'],pair['chosen']));ref_rejected=float(sequence_logprob(runtime,pair['prompt'],pair['rejected']))
 configure_full_training(runtime);optimizer=torch.optim.AdamW(runtime.model.parameters(),lr=TRAIN_POLICY['learning_rate']);optimizer.zero_grad(set_to_none=True)
 # Two sequential graphs, one reference and AdamW step; all agent output tokens.
 with torch.no_grad():current_chosen=float(sequence_logprob(runtime,pair['prompt'],pair['chosen']));current_rejected=float(sequence_logprob(runtime,pair['prompt'],pair['rejected']))
 beta=TRAIN_POLICY['beta'];coefficient=-beta*float(torch.sigmoid(torch.tensor(-beta*((current_chosen-current_rejected)-(ref_chosen-ref_rejected)))))
 (sequence_logprob(runtime,pair['prompt'],pair['chosen'])*coefficient).backward();(sequence_logprob(runtime,pair['prompt'],pair['rejected'])*-coefficient).backward()
 gradients=sum(p.grad is not None for p in runtime.model.parameters());torch.nn.utils.clip_grad_norm_(runtime.model.parameters(),1.);optimizer.step();optimizer.zero_grad(set_to_none=True);runtime.model.eval()
 with torch.no_grad():after_chosen=float(sequence_logprob(runtime,pair['prompt'],pair['chosen']));after_rejected=float(sequence_logprob(runtime,pair['prompt'],pair['rejected']))
 changes=sum(parameter_hash(p)!=before[n] for n,p in runtime.model.named_parameters())
 out=private_dir(job['out']);cp=out/'checkpoint';cp.mkdir();runtime.model.save_pretrained(cp,safe_serialization=True);runtime.tokenizer.save_pretrained(cp)
 files={p.name:file_sha(p) for p in cp.iterdir() if p.is_file()}
 if len(files)!=6 or not changes:raise ValueError('actual changed complete six-file checkpoint')
 report={'job_sha256':digest(job),'completed_at':time.time(),'checkpoint':{'id':digest(files),'files':files,'path':str(cp)},'training_policy':TRAIN_POLICY,'full_model_finetune':True,'reference_chosen':ref_chosen,'reference_rejected':ref_rejected,'after_chosen':after_chosen,'after_rejected':after_rejected,'preference_margin_delta':after_chosen-after_rejected-ref_chosen+ref_rejected,'gradient_tensors':gradients,'changed_parameter_tensors':changes,'optimizer_steps':1,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'auxiliary_tokens_in_loss':False,'quality_improvement_claimed':False,'payable':False,'chain_transactions':False}
 write(out/'training-report.json',report)

class Coordinator:
 def __init__(self,config):
  from nacl.signing import SigningKey
  from .storage import Bucket
  self.config=config;self.state=private_dir(config['state']);self.key=SigningKey(bytes.fromhex(pathlib.Path(config['authority_seed_file']).read_text().strip()))
  if self.key.verify_key.encode().hex()!=AUTHORITY:raise ValueError('operator authority')
  self.bucket=Bucket(config['bucket']);self.current=json.loads((self.state/'current-checkpoint.json').read_text()) if (self.state/'current-checkpoint.json').exists() else config['initial_checkpoint'];self.round=0
 def status(self,**fields):write(self.state/'status.json',{'version':VERSION,'timestamp':time.time(),'pid':os.getpid(),'process_identity':pid_identity(os.getpid()),'round':self.round,'current_checkpoint':self.current['id'],'payable':False,'chain_transactions':False,'external_general_miners_supported':False,**fields})
 def ssh(self,*args):return ['ssh','-T','-p','20059','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+self.config['known_hosts'],'root@90.95.12.246',*args]
 def scp(self,local,remote):subprocess.run(['scp','-P','20059','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+self.config['known_hosts'],str(local),'root@90.95.12.246:'+remote],check=True,stdout=subprocess.DEVNULL)
 def registration(self,folder):
  from .chain import ChainAdapter
  rows=ChainAdapter(folder/'chain',network='finney',netuid=120).registrations();row=rows.get(HOTKEY)
  if not row or row['uid']!=131 or row['public_key']!=IDENTITY:raise ValueError('current real owned UID131 activation')
  write(folder/'registrations.json',rows);return row
 def open(self,index):
  from ops.stage_native_tau2_mixed_source import prepare,ENV
  from .storage import encrypt
  disk_guard(self.state);stamp=str(int(time.time()));folder=self.state/('epoch-'+stamp);remote='/root/native-tau2-mixed-'+stamp
  prepare(folder,remote,self.config['data'],self.config['public_tasks'],self.config['private_tasks'],self.key,self.config['known_hosts'])
  write(folder/'operator-bucket-config.json',self.config['bucket']);registration=self.registration(folder);manifest=json.loads((folder/'signed-epoch.json').read_text())['payload'];sources=json.loads((folder/'source-inventory.json').read_text())
  for name in ('subnet/native_tau2_common_service.py','subnet/native_tau2_common_artifacts.py','subnet/native_tau2_common_bridge.py','subnet/long_context_training.py','subnet/native_tau2_common_service_streaming.py','subnet/native_tau2_common_role_storage.py','subnet/native_tau2_common_streaming_driver.py','ops/finalize_native_tau2_common_boundary.py'):
   sources[name]=file_sha(ROOT/name);target=folder/'source'/name;target.write_bytes((ROOT/name).read_bytes());target.chmod(0o600)
  manifest['checkpoint']={k:v for k,v in self.current.items() if k in ('id','files')};manifest['roles']['agent']['checkpoint']=manifest['checkpoint']
  for role in manifest['roles'].values():role['source_files']=copy.deepcopy(sources)
  manifest['environment']['source_files']=sources;manifest['environment']['version']='original-tau2-common-controlled-private-streaming-v2';manifest['registration_snapshot']=registration;manifest['artifact_policy']={'compressed_bytes':250000000,'raw_bytes':500000000};manifest['submission_window']={'opens_at':time.time(),'deadline':time.time()+self.config.get('epoch_seconds',1200),'registered_uid':131}
  epoch=sign(manifest,self.key);write(folder/'signed-epoch.json',epoch);write(folder/'fixed-user.json',manifest['roles']['user']);write(folder/'source-inventory.json',sources)
  with tarfile.open(folder/'remote-source.tar.gz','w:gz') as archive:
   for name in sources:archive.add(folder/'source'/name,arcname=name)
  subprocess.run(self.ssh('mkdir -m 700 '+shlex.quote(remote)+' '+shlex.quote(remote+'/source')),check=True)
  self.scp(folder/'remote-source.tar.gz',remote+'/remote-source.tar.gz')
  for name in ('agent','user'):
   descriptor=manifest['roles'][name];path=self.current['path'] if name=='agent' else self.config['fixed_user_checkpoint_path'];write(folder/(name+'-worker.json'),{'version':'native-role-process-json-v1','checkpoint':path,'descriptor':descriptor});self.scp(folder/(name+'-worker.json'),remote+'/'+name+'-worker.json')
  subprocess.run(self.ssh('cd '+shlex.quote(remote+'/source')+' && tar xzf ../remote-source.tar.gz && chmod 600 ../*-worker.json'),check=True)
  staging_key='private/native-tau2-common-live/'+manifest['epoch']+'/uid131.zip';expires=max(1,int(manifest['submission_window']['deadline']-time.time()));put=self.bucket.presign(staging_key,'put_object',expires=expires)
  mailbox=encrypt(IDENTITY,{'method':'PUT','put_url':put,'headers':{'Content-Type':'application/octet-stream'},'epoch':manifest['epoch'],'registered_uid':131,'deadline':manifest['submission_window']['deadline'],'object_key':staging_key})
  self.bucket.json('private/native-tau2-common-live/'+manifest['epoch']+'/challenge.json',{'signed_manifest':epoch,'encrypted_upload_capability':mailbox,'identity':IDENTITY})
  write(folder/'private-upload-capability.json',{'put_url':put,'object_key':staging_key,'approved_r2_host':__import__('urllib.parse',fromlist=['urlparse']).urlparse(self.config['bucket']['endpoint']).hostname});write(folder/'open-receipt.json',{'published_at':time.time(),'manifest_sha256':digest(manifest),'remote':remote,'training_index':index,'registration':registration,'challenge_opened_before_generation':True})
  self.status(phase='open',epoch=manifest['epoch'],folder=str(folder),index=index);return folder,remote,epoch
 def native(self,folder,index,attempt,command,out,epoch=None):
  disk_guard(self.state);epoch_path=folder/'signed-epoch.json'
  if epoch is not None:epoch_path=out.parent/(out.name+'-signed-epoch.json');write(epoch_path,epoch)
  record_path=folder/(out.name+'-'+command+'-process.json');report=out/('generation-report.json' if command=='generate' else 'independent-full-verification.json')
  if record_path.exists():
   old=json.loads(record_path.read_text());current=pid_identity(old['pid'])
   while current and current['state']!='Z' and current['start_ticks']==old['identity']['start_ticks']:time.sleep(5);current=pid_identity(old['pid'])
   if report.exists():return json.loads(report.read_text())
   raise RuntimeError('terminal-existing-native-job-no-retry')
  argv=[sys.executable,'-B','-m','subnet.native_tau2_common_streaming_driver',command,'--bucket-config',str(folder/'operator-bucket-config.json'),'--epoch',str(epoch_path),'--fixed-user',str(folder/'fixed-user.json'),'--worker-configs',str(folder/'worker-configs.json'),'--seed-file',self.config['authority_seed_file'],'--data',self.config['data'],'--public-tasks',str(folder/'public-tasks.json'),'--private-tasks',str(folder/'private-tasks.json'),'--out',str(out),'--authority',AUTHORITY,'--index',str(index),'--attempt',str(attempt)]
  with (folder/(out.name+'-'+command+'.private.log')).open('ab') as log:
   proc=subprocess.Popen(argv,cwd=folder/'source',stdout=log,stderr=subprocess.STDOUT,start_new_session=True);identity=pid_identity(proc.pid);write(record_path,{'pid':proc.pid,'identity':identity,'command':command,'index':index,'attempt':attempt,'out':str(out)});self.status(phase=command,epoch=json.loads(epoch_path.read_text())['payload']['epoch'],index=index,attempt=attempt,child=identity)
   try:code=proc.wait(timeout=3600)
   except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=30);raise
  write(folder/(out.name+'-'+command+'-exit.json'),{'exit_code':code,'completed_at':time.time()})
  if code or not report.exists():raise RuntimeError('original-native-role-job-terminal-failure')
  return json.loads(report.read_text())
 def remote(self,folder,remote,job,label):
  record=folder/(label+'-remote-process.json');exit_path=remote+'/'+label+'.exit';pid_path=remote+'/'+label+'.pid'
  if not record.exists():
   job_path=folder/(label+'-job.json');write(job_path,sign(job,self.key));self.scp(job_path,remote+'/'+job_path.name)
   command='cd '+shlex.quote(remote+'/source')+' && env '+' '.join(k+'='+shlex.quote(v) for k,v in ENV.items())+' /root/miner-venv/bin/python -m subnet.native_tau2_common_service_streaming --remote-job '+shlex.quote(remote+'/'+job_path.name)
   script=command+'; code=$?; echo "$code" > '+shlex.quote(exit_path)
   launch='if test -f '+shlex.quote(pid_path)+'; then cat '+shlex.quote(pid_path)+'; else nohup sh -c '+shlex.quote(script)+' > '+shlex.quote(remote+'/'+label+'.private.log')+' 2>&1 < /dev/null & p=$!; echo "$p" > '+shlex.quote(pid_path)+'; cat '+shlex.quote(pid_path)+'; fi'
   write(record,{'launch_state':'unresolved','job_sha256':digest(job),'remote':remote,'label':label,'pid_path':pid_path,'exit_path':exit_path})
   try:
    pid=int(subprocess.check_output(self.ssh(launch),text=True,timeout=60).strip())
    probe=subprocess.check_output(self.ssh('if test -r /proc/'+str(pid)+'/stat; then cat /proc/'+str(pid)+'/stat; else echo TERMINAL; fi'),text=True,timeout=60).strip();ticks=probe.rsplit(')',1)[1].split()[19] if probe!='TERMINAL' else None
   except (subprocess.CalledProcessError,subprocess.TimeoutExpired,ValueError):raise RuntimeError('unresolved remote launch; operator reconciliation required')
   write(record,{'pid':pid,'start_ticks':ticks,'job_sha256':digest(job),'remote':remote,'label':label})
  prior=json.loads(record.read_text())
  if prior['job_sha256']!=digest(job):raise ValueError('immutable remote job identity')
  if prior.get('launch_state')=='unresolved':raise RuntimeError('unresolved remote launch; operator reconciliation required')
  while True:
   # A lost SSH connection is observation failure, never permission to resubmit.
   try:
    result=subprocess.check_output(self.ssh('if test -f '+shlex.quote(exit_path)+'; then echo EXIT; cat '+shlex.quote(exit_path)+'; elif test -r /proc/'+str(prior['pid'])+'/stat; then cat /proc/'+str(prior['pid'])+'/stat; else echo MISSING; fi'),text=True,timeout=60).strip()
   except (subprocess.CalledProcessError,subprocess.TimeoutExpired):time.sleep(10);continue
   if result.startswith('EXIT'):
    code=int(result.splitlines()[-1]);write(folder/(label+'-remote-exit.json'),{'exit_code':code,'completed_at':time.time(),'job_sha256':digest(job)})
    if code:raise RuntimeError('terminal remote job failed; no automatic retry')
    return
   if result=='MISSING' or prior['start_ticks'] is None or result.rsplit(')',1)[1].split()[19]!=prior['start_ticks']:raise RuntimeError('remote process lost without terminal receipt; no retry')
   time.sleep(5)
 def batch(self,folder,attempts,epoch):
  samples={}
  for attempt in attempts:
   out=folder/('attempt-'+str(attempt));records=json.loads((out/'signed-receipts.json').read_text());arrays={r['payload']['probabilities_file']:__import__('subnet.native_tau2_common_role_storage',fromlist=['read_array']).read_array(self.bucket,out,r['payload']['probabilities_file'],epoch) for r in records};raw,descriptor=pack_sample(epoch,records,json.loads((out/'role-audit.json').read_text()),json.loads((out/'independent-full-verification.json').read_text()),arrays,AUTHORITY,epoch['payload']['roles']['user']);samples['sample-'+str(attempt)+'.zip']=raw
  if sum(sum(i.file_size for i in zipfile.ZipFile(io.BytesIO(raw)).infolist()) for raw in samples.values())>500000000:raise ValueError('cumulative raw transport budget')
  output=io.BytesIO()
  with zipfile.ZipFile(output,'w',zipfile.ZIP_STORED) as archive:
   archive.writestr('batch.json',canonical({'version':VERSION,'epoch':epoch['payload']['epoch'],'registered_uid':131,'samples':list(samples),'payable':False}));
   for name,raw in samples.items():archive.writestr(name,raw)
  if len(output.getvalue())>250000000:raise ValueError('cumulative compressed transport budget')
  return output.getvalue(),samples
 def upload(self,folder,remote,body,label):
  caps=json.loads((folder/'private-upload-capability.json').read_text());path=folder/(label+'.zip');path.write_bytes(body);path.chmod(0o600);self.scp(path,remote+'/'+path.name)
  job={'role':'native-tau2-object-upload-v1','source_files':json.loads((folder/'source-inventory.json').read_text()),'approved_r2_host':caps['approved_r2_host'],'objects':[{'path':remote+'/'+path.name,'put_url':caps['put_url'],'sha256':file_sha(path),'size':len(body)}],'receipt':remote+'/'+label+'-upload.json','payable':False,'chain_transactions':False};self.remote(folder,remote,job,label+'-put')
 def offload(self,folder,out):
  if (out/'signed-private-role-storage.json').exists():
   write(out/'signed-private-storage-admission.json',sign({'storage_inventory_sha256':file_sha(out/'signed-private-role-storage.json'),'full_audit_sha256':file_sha(out/'role-audit.json'),'full_verification_report_sha256':file_sha(out/'independent-full-verification.json'),'private_raw_arrays_retained':True,'payable':False},self.key));return
  files={}
  for path in (out/'roles').glob('*.npy'):
   key='private/native-tau2-common-live/'+folder.name+'/raw/'+out.name+'/'+path.name;expected=file_sha(path);self.bucket.upload(key,path);response=self.bucket.client.get_object(Bucket=self.bucket.name,Key=key);h=hashlib.sha256();length=0
   try:
    for chunk in iter(lambda:response['Body'].read(1048576),b''):h.update(chunk);length+=len(chunk)
   finally:response['Body'].close()
   if h.hexdigest()!=expected or length!=path.stat().st_size:raise ValueError('immutable raw offload verification')
   files[path.name]={'key':key,'sha256':expected,'size':length};path.unlink()
  write(out/'signed-raw-offload.json',sign({'files':files,'uploaded_exact_bytes_verified':True,'local_raw_cache_removed_after_full_audit':True,'payable':False},self.key))
 def evaluate(self,folder,epoch,phase):
  rows=[];manifest=copy.deepcopy(epoch['payload']);manifest['roles']['agent']['candidate_policy']=None;eval_epoch=sign(manifest,self.key);dataset=heldout_contract(eval_epoch,AUTHORITY,manifest['roles']['user'],json.loads((folder/'public-tasks.json').read_text()));write(folder/(phase+'-heldout-contract.json'),dataset)
  # The worker descriptor must match the separately signed autoregressive plan.
  workers=json.loads((folder/'worker-configs.json').read_text());remote=json.loads((folder/'open-receipt.json').read_text())['remote'];config={'version':'native-role-process-json-v1','checkpoint':self.current['path'],'descriptor':manifest['roles']['agent']};write(folder/'agent-eval-worker.json',config);self.scp(folder/'agent-eval-worker.json',remote+'/agent-eval-worker.json');workers['agent']['argv'][-1]=workers['agent']['argv'][-1].replace('agent-worker.json','agent-eval-worker.json');write(folder/'eval-worker-configs.json',workers)
  original=(folder/'worker-configs.json').read_bytes();(folder/'worker-configs.json').write_bytes(canonical(workers))
  try:
   for index in range(16,32):
    out=folder/(phase+'-heldout-'+str(index));start=time.time()
    try:
     self.native(folder,index,0,'generate',out,eval_epoch);report=self.native(folder,index,0,'verify',out,eval_epoch);rows.append({'index':index,'task_hash':next(t['task_hash'] for t in manifest['tasks'] if t['index']==index),'seed':next(t['seed'] for t in manifest['tasks'] if t['index']==index),'reward':report['reward'],'verified':True,'completed_at':time.time()});self.offload(folder,out)
    except Exception as error:rows.append({'index':index,'verified':False,'error_type':type(error).__name__,'completed_at':time.time()})
    completed=[r for r in rows if r['verified']];write(folder/(phase+'-evaluation.json'),{'dataset_id':dataset['dataset_id'],'checkpoint':manifest['checkpoint']['id'],'fixed_user_sha256':digest(manifest['roles']['user']),'phase':phase,'records':rows,'completed_count':len(completed),'error_count':len(rows)-len(completed),'requested_count':16,'all_tasks_completed':len(rows)==16 and len(completed)==16,'mean_reward_completed_only':sum(r['reward'] for r in completed)/len(completed) if completed else None,'autoregressive_agent':True,'original_native_grader':True,'payable':False})
  finally:(folder/'worker-configs.json').write_bytes(original)
 def cycle(self,index):
  folder,remote,epoch=self.open(index);self.round+=1
  # Owned worker configs are regenerated against the newly signed exact roles.
  workers=json.loads((folder/'worker-configs.json').read_text());write(folder/'worker-configs.json',workers)
  views=[]
  for attempt in (0,1):
   out=folder/('attempt-'+str(attempt));self.native(folder,index,attempt,'generate',out);self.native(folder,index,attempt,'verify',out);views.append(json.loads((out/'admitted-sample.json').read_text()));body,samples=self.batch(folder,list(range(attempt+1)),epoch)
   if time.time()>=epoch['payload']['submission_window']['deadline']:raise RuntimeError('no-complete-epoch-window-expired')
   self.upload(folder,remote,body,'cumulative-'+str(attempt))
  preference_pair(views[0],views[1]);caps=json.loads((folder/'private-upload-capability.json').read_text());snapshot=self.bucket.snapshot(caps['object_key'],limit=250000000);received=time.time()
  if snapshot is None or snapshot['data']!=body or received>=epoch['payload']['submission_window']['deadline']:raise ValueError('atomic freeze before deadline')
  for name,raw in samples.items():
   freeze=sign({'version':FREEZE_VERSION,'manifest_sha256':digest(epoch['payload']),'registered_uid':131,'received_at':received,'zip_sha256':hashlib.sha256(raw).hexdigest(),'zip_size':len(raw),'private':True,'payable':False,'chain_transactions':False},self.key)
   admit_frozen_sample(freeze,raw,epoch,AUTHORITY,epoch['payload']['roles']['user'],131,received)
   write(folder/(name+'-freeze.json'),freeze)
  self.bucket.put('private/native-tau2-common-live/'+epoch['payload']['epoch']+'/frozen-'+hashlib.sha256(body).hexdigest()+'.zip',body)
  write(folder/'signed-batch-freeze.json',sign({'version':VERSION,'manifest_sha256':digest(epoch['payload']),'registered_uid':131,'zip_sha256':hashlib.sha256(body).hexdigest(),'zip_size':len(body),'received_at':received,'r2_etag':snapshot['etag'],'payable':False},self.key));self.status(phase='collect-until-deadline',epoch=epoch['payload']['epoch'])
  while time.time()<epoch['payload']['submission_window']['deadline']:time.sleep(min(5,epoch['payload']['submission_window']['deadline']-time.time()))
  final=self.bucket.snapshot(caps['object_key'],limit=250000000)
  from ops.finalize_native_tau2_common_boundary import select
  final_report=select(final,json.loads((folder/'signed-batch-freeze.json').read_text())['payload'],epoch['payload']['submission_window']['deadline'],time.time());write(folder/'signed-final-boundary-selection.json',sign(final_report,self.key))
  # A separate fresh full verifier after freeze. Existing audit files are copied
  # as pre-freeze evidence; post-freeze verification writes a new immutable view.
  fresh_views=[]
  for attempt in (0,1):
   out=folder/('attempt-'+str(attempt));post=folder/('post-freeze-attempt-'+str(attempt));shutil.copytree(out,post);self.native(folder,index,attempt,'verify',post);fresh_views.append(json.loads((post/'admitted-sample.json').read_text()));self.offload(folder,post)
  pair=preference_pair(fresh_views[0],fresh_views[1])
  write(folder/'signed-proposed-score.json',sign({'epoch':epoch['payload']['epoch'],'unique_environment_index':index,'points':{'131':1},'normalized_proposed_weights':{'131':1.},'full_native_and_model_audits':True,'controlled_owned_identity_only':True,'payable':False,'chain_transactions':False},self.key))
  for attempt in (0,1):self.offload(folder,folder/('attempt-'+str(attempt)))
  self.evaluate(folder,epoch,'before')
  job={'role':'native-tau2-agent-full-train-v1','source_files':json.loads((folder/'source-inventory.json').read_text()),'training_policy':TRAIN_POLICY,'descriptor':epoch['payload']['roles']['agent'],'checkpoint_path':self.current['path'],'pair':pair,'runtime_environment':json.loads((ROOT/'state/native-role-cuda/1790862267/job.json').read_text())['payload']['runtime_environment'],'out':remote+'/training','payable':False,'chain_transactions':False};self.status(phase='full-agent-train',epoch=epoch['payload']['epoch']);self.remote(folder,remote,job,'train')
  report_path=folder/'training-report.json';subprocess.run(['scp','-P','20059','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+self.config['known_hosts'],'root@90.95.12.246:'+remote+'/training/training-report.json',str(report_path)],check=True);report=json.loads(report_path.read_text());validate_training_report(report,job);cp=report['checkpoint'];objects=[]
  for name,expected in cp['files'].items():
   # Digest-bound specific objects; only those six PUT caps are delegated.
   response=subprocess.check_output(self.ssh('stat -c %s '+shlex.quote(cp['path']+'/'+name)),text=True);objects.append({'path':cp['path']+'/'+name,'sha256':expected,'size':int(response),'put_url':self.bucket.presign('public/checkpoints/'+cp['id']+'/'+name,'put_object',expires=1200)})
  from urllib.parse import urlparse
  self.remote(folder,remote,{'role':'native-tau2-object-upload-v1','source_files':job['source_files'],'approved_r2_host':urlparse(self.config['bucket']['endpoint']).hostname,'objects':objects,'receipt':remote+'/checkpoint-upload.json','payable':False,'chain_transactions':False},'checkpoint-put')
  from ops.check_r2_checkpoint import check
  independent=check(self.bucket,cp);write(folder/'independent-r2-checkpoint.json',independent);write(folder/'signed-checkpoint-publication.json',sign({'checkpoint':cp,'training_report_sha256':file_sha(report_path),'all_six_R2_hashes_verified':True,'independent_check_sha256':file_sha(folder/'independent-r2-checkpoint.json'),'payable':False},self.key));self.current=cp;write(self.state/'current-checkpoint.json',cp)
  next_manifest=copy.deepcopy(epoch['payload']);next_manifest['epoch']=epoch['payload']['epoch']+'-after-evaluation';next_manifest['submission_window']={'opens_at':time.time(),'deadline':time.time()+self.config.get('epoch_seconds',1200),'registered_uid':131};next_manifest['checkpoint']={k:cp[k] for k in ('id','files')};next_manifest['roles']['agent']['checkpoint']=next_manifest['checkpoint'];next_epoch=sign(next_manifest,self.key)
  self.bucket.json('private/native-tau2-common-live/'+next_manifest['epoch']+'/challenge.json',{'signed_manifest':next_epoch,'controlled_successor_probe':True,'payable':False})
  write(folder/'signed-successor-challenge.json',next_epoch)
  write(folder/'agent-worker.json',{'version':'native-role-process-json-v1','checkpoint':cp['path'],'descriptor':next_manifest['roles']['agent']});self.scp(folder/'agent-worker.json',remote+'/agent-worker.json')
  self.evaluate(folder,next_epoch,'after');next_manifest['epoch']=epoch['payload']['epoch']+'-successor';next_manifest['submission_window']={'opens_at':time.time(),'deadline':time.time()+self.config.get('epoch_seconds',1200),'registered_uid':131};next_epoch=sign(next_manifest,self.key);write(folder/'signed-successor-challenge.json',next_epoch);self.bucket.json('private/native-tau2-common-live/'+next_manifest['epoch']+'/challenge.json',{'signed_manifest':next_epoch,'controlled_successor_probe':True,'payable':False});out=folder/'fresh-successor-native';self.native(folder,index,0,'generate',out,next_epoch);self.native(folder,index,0,'verify',out,next_epoch);self.offload(folder,out)
  gate=evaluation_completion_gate(*(json.loads((folder/name).read_text()) for name in ('before-evaluation.json','after-evaluation.json','before-heldout-contract.json','after-heldout-contract.json')))
  write(folder/'completed.json',{'checkpoint':cp['id'],'epoch':epoch['payload']['epoch'],'full_pipeline_completed':gate['complete'],'completion_gate':gate,'training_completed':True,'successor_native_verified':True,'quality_improvement_claimed':False,'completed_at':time.time(),'payable':False,'chain_transactions':False});self.status(phase='epoch-complete' if gate['complete'] else 'epoch-trained-evaluation-incomplete',epoch=epoch['payload']['epoch'],completion_gate=gate)
 def loop(self):
  for record in self.state.glob('epoch-*/*-remote-process.json'):
   if json.loads(record.read_text()).get('launch_state')=='unresolved':
    self.status(phase='unresolved-remote-launch-hold',journal=str(record));raise RuntimeError('unresolved remote launch; operator reconciliation required')
  for record in self.state.glob('epoch-*/*-process.json'):
   row=json.loads(record.read_text());identity=pid_identity(row.get('pid',-1))
   if identity and identity.get('start_ticks')==row.get('identity',{}).get('start_ticks') and identity['state']!='Z':raise RuntimeError('existing native process active; operator must resume exact epoch')
  self.round=len(list(self.state.glob('epoch-*')))
  while True:
   index=self.round%16
   try:self.cycle(index)
   except Exception as error:
    reason=str(error)
    phase='waiting-capacity' if 'capacity' in reason else 'no-complete-epoch-rollover'
    self.status(phase=phase,index=index,error_type=type(error).__name__,error=reason[:160])
    if 'remote process lost' in reason or 'immutable remote job' in reason or 'unresolved remote launch' in reason:raise
    time.sleep(30)

def native_run():
 from . import native_tau2_common_mixed_driver as driver
 original=driver.CommonRoleEndpoint
 class DiskBoundEndpoint(original):
  def response(self,request):
   roles=[r for r in self.manifest['roles'].values() if r['request_model']==request.get('model')]
   if len(roles)!=1:raise ValueError('unique role')
   role=roles[0];reserve=role['max_output_tokens']*role['vocab_size']*4+128*1024*1024
   disk_guard(ROOT,max(1073741824,reserve+512*1024*1024))
   return super().response(request)
 driver.CommonRoleEndpoint=DiskBoundEndpoint
 sys.argv=[sys.argv[0],*sys.argv[2:]]
 driver.main()

def main():
 if len(sys.argv)>1 and sys.argv[1]=='--native-run':native_run();return
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=pathlib.Path);parser.add_argument('--remote-job',type=pathlib.Path);args=parser.parse_args()
 if args.remote_job:remote_job(args.remote_job);return
 if not args.config:parser.error('operator configuration required')
 Coordinator(json.loads(args.config.read_bytes())).loop()
if __name__=='__main__':main()
