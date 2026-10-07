"""Default-off isolated same-parent study; no public batch or authority writer."""
import argparse,base64,fcntl,hashlib,io,json,os,pathlib,shutil,subprocess,sys,time
from nacl.signing import VerifyKey
AUTH='3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
CP='7369a0b35c34676da76085136c6344e013652df312586d53e8c0335756a5757b'
SCIENCE='f21373d7ccb167bcd868f5ed03ce9ac7ef567d894e8ecc9e61f5d7f8645b67b8'
TASKS=[385,781,2997,3110,4265,4783,4936,6147]
VERSION='same-parent24-completed-negative-research-fresh-v4'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def signed(e,key=AUTH):
 if e.get('signer')!=key:raise ValueError('exact signer')
 VerifyKey(bytes.fromhex(key)).verify(canonical(e['payload']),base64.b64decode(e['signature'],validate=True));return e['payload']
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def file_sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=pathlib.Path(p);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as f:f.write(canonical(v));f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);os.fsync(fd);os.close(fd)
def owned(p,directory=False):
 p=pathlib.Path(p)
 if p.resolve()!=p or p.is_symlink()or p.stat().st_uid!=os.getuid()or (directory and not p.is_dir()):raise ValueError('canonical owned path')
 return p

def fresh_bytecode(s):
 prefix=pathlib.Path(s['bytecode_cache_prefix'])
 if not prefix.is_absolute()or prefix.resolve()!=prefix or prefix.exists()or not sys.dont_write_bytecode or sys.pycache_prefix!=str(prefix)or os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise ValueError('ROOT exact fresh bytecode prefix and write suppression')
 owned(prefix.parent,True)
 return str(prefix)

def parent_admission(packet):
 p=pathlib.Path(packet)/'immutable-parent24';publication=signed(read(p/'ORIGINAL-parent-publication.ROOT-signed.private.json'));descriptor=publication['descriptor'];pointer=read(p/'ORIGINAL-authority-pointer.private.json');request=signed(read(p/'ORIGINAL-independent-reader-request.ROOT-signed.private.json'));actual=read(p/'ORIGINAL-independent-reader-actual-terminal-and-result.private.json');result=signed(actual['result'],request['reader_identity']);terminal=actual['terminal'];objects=[{k:r[k]for k in ('name','sha256','size')}for r in descriptor['shards']]
 if (descriptor['optimizer_steps']!=24 or descriptor['inference_checkpoint']!=CP or len(objects)!=23 or digest(descriptor)!=pointer['descriptor_sha256']or digest(publication)!=pointer['publication_sha256']or result['descriptor_sha256']!=digest(descriptor)or result['objects']!=objects or request['objects']!=objects or result['request_sha256']!=digest(request)or result['all_objects_full_hash']is not True or result['total_bytes']!=sum(x['size']for x in objects)or terminal['actual_child_wait_completed']is not True or terminal['exit_code']!=0 or terminal['timed_out']is not False):raise ValueError('original parent24 full23 ACK and actual wait')
 return descriptor,publication

def pairs_admission(packet,arm,allowed_workers):
 if arm not in ('baseline-cap','completed-EOS'):raise ValueError('explicit branch')
 if not allowed_workers or len(set(allowed_workers))!=len(allowed_workers):raise ValueError('ROOT bound normal verifier admission')
 packet=pathlib.Path(packet);m=signed(read(packet/'ORIGINAL-opening.ROOT-signed.private.json'));commits=read(packet/'ORIGINAL-matched-miner-commitments.private.json');reports=read(packet/'AUTHENTICATED-original-job-worker-reports.private.json');accepted={}
 for record in reports:
  if record['worker_report']['signer']not in allowed_workers:raise ValueError('normal verifier not ROOT authorized')
  j=signed(record['job']);mm=signed(j['manifest']);rq=signed(record['worker_report'],record['worker_report']['signer']);r=record['report']
  if rq.get('action')!='report'or rq['report']!=r or r['job_sha256']!=digest(j)or r['checkpoint']!=CP or r['epoch']!=m['epoch']or r['source_files']!=j['source_files']or len(j['source_files'])!=177 or mm['source_bundle']['sha256']!=SCIENCE:raise ValueError('original authenticated audit context')
  if [a['submission_sha256']for a in r['audits']]!=[o['sha256']for o in j['submissions']]:raise ValueError('original frozen proof binding')
  for a in r['audits']:
   for b in a['accepted']:accepted[digest(b)]=b
 if m['checkpoint']['id']!=CP or m['source_bundle']['sha256']!=SCIENCE:raise ValueError('unchanged original opening')
 groups={}
 for p in sorted(packet.glob('*.json')):
  if '-'not in p.stem or len(p.stem.split('-')[0])!=64:continue
  doc=read(p);miner=doc['miner'];commit=signed(commits[miner],miner);child=next(b for b in commit['batches']if b['slot']==doc['slot']);batch=doc['batch']
  if (file_sha(p)!=child['training_sha256']or p.stat().st_size!=child['training_size']or digest(batch)!=child['batch_sha256']or batch!=accepted.get(digest(batch))or commit['checkpoint']!=CP or commit['source']!=SCIENCE or batch['epoch']!=m['epoch']):raise ValueError('exact original document and accepted verdict')
  key=(batch['env_id'],batch['index']);g=groups.setdefault(key,[])
  batch_refs={digest(r):{'miner':miner,'slot':doc['slot'],'batch_sha256':digest(batch),'document_sha256':file_sha(p),'rollout_sha256':digest(r)}for r in batch['rollouts']}
  g.append((batch,batch_refs))
 if sorted(k[1]for k in groups)!=TASKS or any(k[0]!='affine_math'for k in groups):raise ValueError('exact eight task population')
 pairs=[];bindings=[]
 for index in TASKS:
  batches=groups['affine_math',index];definition=next(e for e in m['environments']if e['env_id']=='affine_math');h=definition['harness']
  if h['max_output_tokens']!=1024 or h['version']!='text-tools-long-v2'or index not in definition['indices']:raise ValueError('original native harness')
  cap=[];complete=[]
  for b,batch_refs in batches:
   pp=[r for r in b['rollouts']if r['classification']=='positive'];nn=[r for r in b['rollouts']if r['classification']=='negative']
   if len(pp)!=1 or len(nn)!=1:raise ValueError('original K1L1')
   n=nn[0];last=n['turns'][-1]['output'];row=(pp[0],n,batch_refs)
   if last[-1]==151645:complete.append(row)
   elif len(last)==1024:cap.append(row)
  if not cap or not complete:raise ValueError('matched native cap/EOS supply')
  selected_cap=min(cap,key=lambda row:(digest(row[1]),row[2][digest(row[1])]['document_sha256']));positive,capped,positive_refs=selected_cap
  selected_negative=selected_cap if arm=='baseline-cap'else min(complete,key=lambda row:(digest(row[1]),row[2][digest(row[1])]['document_sha256']))
  negative=selected_negative[1];negative_refs=selected_negative[2]
  if positive['task_hash']!=negative['task_hash']:raise ValueError('same task content')
  pairs.append((definition,positive,negative));bindings.append({'index':index,'positive':positive_refs[digest(positive)],'negative':negative_refs[digest(negative)],'gradient_task_weight':1/8,'cross_document_pair_authorization_required':arm=='completed-EOS'})
 return m,pairs,bindings

def prospective_budget(s):
 if s.get('optimizer_state_export_allowed')is not False or s.get('optimizer_RAM_disposable_research_only')is not True or s.get('optimizer_payload_GETs_in_training_allowed')is not False or type(s.get('restore_concurrency'))is not int or s['restore_concurrency']!=4:raise ValueError('explicit disposable research state and exact ACKed local four-stream restore')
 if s.get('retained_pod_id')in(None,'ba5ad596-d42a-4bcc-8643-f18e2272ccd5')or s.get('old_interrupted_scope_sha256')!='397f6e23d98b5ae184dd47591fee1b94fa06ee482f6991de03e5d59be44e67c2' or not s.get('workspace','').startswith('/root/affine-CP24-negative-quality-v4-'):raise ValueError('distinct replacement research identity, never original reissue')
 return True

def copy_verified_parent_shard(cached,path,row):
 cached=owned(cached);path=pathlib.Path(path);st=cached.stat()
 if not cached.is_file()or st.st_nlink!=1 or st.st_size!=row['size']or file_sha(cached)!=row['sha256']:raise ValueError('no GET: exact already-ACKed local original parent shard')
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with cached.open('rb')as src,os.fdopen(fd,'wb')as dst:shutil.copyfileobj(src,dst,1024*1024);dst.flush();os.fsync(dst.fileno())

def validate(scope,packet,arm,now):
 s=signed(scope)
 if (s.get('version')!=VERSION or s.get('execute_allowed')is not True or s.get('production_mutations')is not False or s.get('cross_document_original_pair_refs_authorized')is not True or s.get('arm')!=arm or not s['created_at']<=now<s['expires_at']or not 0<s['expires_at']-s['created_at']<=7200):raise ValueError('fresh explicit ROOT research scope')
 prospective_budget(s)
 if type(s['steps'])is not int or type(s['parent_step'])is not int or type(s['output_step'])is not int:raise ValueError('exact integer optimizer budget')
 if s['helper_sha256']!=file_sha(__file__)or s['source_sha256']!=SCIENCE or s['checkpoint_id']!=CP or s['parent_step']!=24 or s['output_step']!=25 or s['steps']!=1 or s['heldout_cohort_sha256']!='20a077180d7cc088669f53bd551c5bf3c4aa51ded6367463f405094ad054b153':raise ValueError('exact helper/source/lineage/cohort')
 packet=owned(packet,True)
 if any(file_sha(owned(packet/name))!=h for name,h in s['evidence_files'].items()):raise ValueError('ROOT evidence file pins')
 parent,_=parent_admission(packet);m,pairs,refs=pairs_admission(packet,arm,s['allowed_report_workers'])
 if digest(parent)!=s['parent_descriptor_sha256']or refs!=s['pair_refs']:raise ValueError('signed original parent and pair references')
 if s['research_full_artifact_ROOT_ACK']is None or s['fresh_physical_ROOT_admission']is None:raise ValueError('ROOT reviewed original evidence and fresh physical qualification')
 ack=signed(s['research_full_artifact_ROOT_ACK'])
 if ack.get('version')!='CP24-matched-eight-original-evidence-ACK-v1'or ack.get('checkpoint_id')!=CP or ack.get('source_sha256')!=SCIENCE or ack.get('task_indices')!=TASKS or ack.get('evidence_files')!=s['evidence_files']or ack.get('full_independent_R2_GET_verified')is not True or ack.get('all16_original_batches_authenticated_ACCEPTED')is not True:raise ValueError('ROOT full original sixteen object ACK context')
 cpu=signed(s['CPU_stage_ROOT_ACK'])
 if cpu.get('version')!='CP24-controlled-research-CPU-stage-ACK-v1'or cpu.get('checkpoint_id')!=CP or cpu.get('source_sha256')!=SCIENCE or cpu.get('evidence_files')!=s['evidence_files']or cpu.get('runtime_package_RECORD_hashes')!=s['runtime_package_RECORD_hashes']or cpu.get('actual_child_wait_exit_code')!=0 or cpu.get('actual_child_wait_completed')is not True or cpu.get('timed_out')is not False or cpu.get('all10_model_fullSHA_verified')is not True or cpu.get('all2121_source_fullSHA_verified')is not True:raise ValueError('ROOT authentic original CPUstage ACK')
 warm=signed(s['parent_state_cache_ROOT_ACK'])
 if warm.get('checkpoint_id')!=CP or warm.get('parent_descriptor_sha256')!=digest(parent)or warm.get('retained_pod_id')!=s['retained_pod_id']or warm.get('cache_path')!=s['parent_state_cache']or warm.get('all23_full_SHA_verified')is not True or warm.get('total_bytes')!=sum(r['size']for r in parent['shards']):raise ValueError('fresh owned local full23 parent warmcache ROOT ACK')
 if cpu.get('retained_pod_id')!=s['retained_pod_id']or cpu.get('machine_id_sha256')!=s['machine_id_sha256']or cpu.get('gpu_uuid')!=s['gpu_uuid']or cpu.get('scope_sha256')!=s['CPU_stage_scope_sha256']:raise ValueError('new physical CPUstage ACK required; old ACK cannot qualify replacement')
 import importlib.metadata
 if any(importlib.metadata.version(k)!=v for k,v in s['runtime_versions'].items()):raise ValueError('actual original package versions')
 if len(s['runtime_package_RECORD_hashes'])<3 or any(file_sha(owned(p))!=h for p,h in s['runtime_package_RECORD_hashes'].items()):raise ValueError('actual admitted runtime RECORD')
 source=owned(s['source_path'],True)
 if len(s['source_files'])!=177 or any(file_sha(source/k)!=v for k,v in s['source_files'].items()):raise ValueError('all original177 module bytes')
 observed={str(p.relative_to(source))for p in source.rglob('*')if p.is_file()and'__pycache__'not in p.parts and p.suffix!='.pyc'}
 if len(s['full_source_files'])!=2121 or observed!=set(s['full_source_files'])or any(file_sha(owned(source/k))!=v for k,v in s['full_source_files'].items()):raise ValueError('full original2121 source membership')
 if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('original CUBLAS profile')
 return s,parent,m,pairs,refs

def fence(s):
 lock=pathlib.Path(s['physical_gpu_lock']);owned(lock.parent,True);fd=os.open(lock,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
 if os.fstat(fd).st_uid!=os.getuid():raise ValueError('owned physical lock')
 fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if hashlib.sha256(pathlib.Path('/etc/machine-id').read_bytes()).hexdigest()!=s['machine_id_sha256']:raise ValueError('new ROOT-bound physical host')
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,compute_cap','--format=csv,noheader'],text=True,timeout=15).strip()
 if gpu!=s['gpu_uuid']+', 9.0'or subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=15).strip():raise ValueError('exclusive original SM90 GPU')
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit()or int(p.name)==os.getpid():continue
  try:argv=(p/'cmdline').read_bytes().split(b'\0');stat=(p/'stat').read_text().rsplit(')',1)[1].split()
  except OSError:continue
  if stat[0]=='Z':continue
  commands=[a.decode(errors='strict')for a in argv if a]
  scientific=any(a in ('subnet.backend_jobs','subnet.distributed_worker','subnet.remote_runner','subnet.cli')or a.startswith(('/root/affine-','/root/affine-learning-'))for a in commands)
  marker=read(owned(s['supervisor_process_marker']))
  own=marker['scope_sha256']==file_sha(s['scope_path'])and int(p.name)==os.getppid()==marker['pid']and s['supervisor_path']in commands and stat[19]==marker['ticks']and file_sha(s['supervisor_path'])==s['supervisor_sha256']
  if scientific and not own:raise ValueError('other scientific preparation actor')
 return fd

def execute(s,parent,m,pairs,refs):
 sys.path.insert(0,s['source_path']);from subnet.gpu_runtime import GPURuntime
 from subnet.model import model_files
 from subnet.persistent_cpu_adamw import parameter_inventory
 from subnet.persistent_training_state import resource_plan,admit_resources,restore_state,available_ram_bytes
 from subnet.task_normalized_training import train_epoch
 root=owned(s['workspace'],True);branch=root/s['arm']
 if branch.exists():raise ValueError('original branch exists; no reissue')
 branch.mkdir(mode=0o700);write(branch/'operation-start.json',{'scope_sha256':digest(s),'started_at':time.time(),'no_repeat':True,'production_mutations':False})
 if model_files(s['checkpoint_path'])!=m['checkpoint']['files']:raise ValueError('actual full CP24 file hashes')
 if type(s['minimum_preload_available_RAM_bytes'])is not int or s['minimum_preload_available_RAM_bytes']<150*1024**3 or available_ram_bytes()<s['minimum_preload_available_RAM_bytes']:raise ValueError('ROOT study premodel effective150GiB RAM floor')
 early_resource=resource_plan(parent['parameters'],bf16_export_bytes=15_242_726_226,concurrency=s['restore_concurrency'],disk_reserve_bytes=8*1024**3);admit_resources(branch,early_resource)
 de=pairs[0][0];runtime=GPURuntime(s['checkpoint_path'],m['checkpoint']['files'],de['spec'],de['harness'],runtime_revision='cuda-bf16-eager-sm90-v1')
 _,inventory=parameter_inventory(runtime.model.named_parameters());resource=resource_plan(inventory,bf16_export_bytes=15_242_726_226,concurrency=s['restore_concurrency'],disk_reserve_bytes=8*1024**3);admission=admit_resources(branch,resource);shards={r['name']:r for r in parent['shards']}
 def fetch(name,path):copy_verified_parent_shard(pathlib.Path(s['parent_state_cache'])/name,path,shards[name])
 restored,evidence=restore_state(parent,digest(parent),CP,inventory,workspace=branch,fetch_shard=fetch,resource_admission=admission,concurrency=s['restore_concurrency'])
 dest,opt,metrics=train_epoch(runtime,pairs,branch,input_checkpoint=CP,epoch=s['research_epoch'],seed=s['selection_seed'],steps=1,restored_state=restored,resource_admission=admission)
 if opt.global_step!=25 or opt.parent_state_sha256!=digest(parent):raise ValueError('same original24 to isolated25')
 files=model_files(dest);output_checkpoint=digest(files)
 result={'version':VERSION,'arm':s['arm'],'checkpoint_files':files,'output_checkpoint':output_checkpoint,'parent_checkpoint':CP,'parent_state_sha256':digest(parent),'parent_step':24,'research_output_step':25,'pair_refs':refs,'restore_evidence':evidence,'training':metrics,'optimizer_state_durably_exported':False,'optimizer_RAM_disposable_research_only':True,'parent_hyperparameters':parent['hyperparameters'],'reference_scope':'immutable-BF16-epoch-input','production_pointer_writes':False,'R2_full_ACK_pending':True,'heldout_gain_claimed':False};write(branch/'result.private.json',result);return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scope',required=True);ap.add_argument('--packet',required=True);ap.add_argument('--arm',choices=['baseline-cap','completed-EOS'],required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args();s,parent,m,pairs,refs=validate(read(a.scope),a.packet,a.arm,time.time());fresh_bytecode(s)
 if not a.execute:print(json.dumps({'contract_check':True,'tasks':len(pairs),'parent_step':24,'branch_step':25,'GPU_started':False}));return
 fd=fence(s)
 try:execute(s,parent,m,pairs,refs)
 finally:os.close(fd)
if __name__=='__main__':main()
