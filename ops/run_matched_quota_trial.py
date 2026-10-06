"""ROOT-scoped research phases; never advances network optimizer authority.

The operator stages a qualified scientific source and authenticated parent model.
This credentialless worker can generate, or run one isolated training fork.
Result publication/ACK and heldout jobs use the existing ROOT CPU relays.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from ops.matched_quota_trial import digest,task_plan,verify_generated,select_matched,train_arm

def private_write(path,data):
 path=Path(path);path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as f:f.write(data);f.flush();os.fsync(f.fileno())

def validate_scope(envelope,authority,phase,now):
 from subnet.backend_jobs import signed
 s=signed(envelope,authority)
 if s['version']!='matched-quota-research-v1' or s['production_state_changes']is not False or s['execute_allowed']is not True:
  raise ValueError('explicit isolated research capability')
 if any(type(s[k])not in(int,float)for k in ('created_at','expires_at'))or not s['created_at']<=now<s['expires_at']or not 0<s['expires_at']-s['created_at']<=7200:raise ValueError('original research lifetime')
 if set(s['phases'])!=({'generate'}if s['stage']=='pilot32'else{'generate','1P1N','2P2N'})or phase not in s['phases']or any(type(s[k])is not int for k in ('training_steps','attempts','K','L','restore_concurrency','output_limit_bytes'))or s['training_steps']!=1 or s['attempts']!=16 or s['K']!=2 or s['L']!=2 or s['restore_concurrency']not in(4,8)or not 0<s['output_limit_bytes']<=512*1024**2:raise ValueError('declared comparison budget')
 if s['task_indices']!=task_plan(s['mining_indices'],s['heldout128_indices']+s['old32_indices'],s['selection_seed'],s['task_count']):raise ValueError('precommitted no-leak task selection')
 m=s['generation_manifest'];c=m['checkpoint'];
 if m.get('probability_artifact_policy')!={'version':'selected-token-logprobs-v1'}:raise ValueError('required compact TOPLOC/selected probability contract')
 if c['id']!=s['checkpoint']or m['source_bundle']['sha256']!=s['source_sha256']or m['sampling_contract']['max_attempts']!=16 or m['K']!=2 or m['L']!=2:raise ValueError('fresh source/checkpoint/quota/draw contract')
 if s['heldout_cohort_sha256']!='20a077180d7cc088669f53bd551c5bf3c4aa51ded6367463f405094ad054b153':raise ValueError('unchanged matched128 cohort')
 if m['sampling_contract']['version']!=s['sampling_version']or s['sampling_version']not in('forced-inverse-cdf-prefill-support-v3','forced-inverse-cdf-prefill-threeway-v4'):raise ValueError('explicit fresh research sampling policy')
 if s['stage']=='pilot32'and(s['generation_task_limit']!=2 or s['task_count']!=8):raise ValueError('predeclared eight-task plan first two only')
 if s['stage']not in('pilot32','matched128')or type(s['generation_task_limit'])is not int or not 1<=s['generation_task_limit']<=s['task_count']:raise ValueError('bounded generation stage')
 for name,expected in s['source_files'].items():
  p=Path(s['source_path'])/name
  if p.is_symlink()or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('qualified scientific module')
 root=Path(s['workspace']);root.mkdir(mode=0o700,parents=True,exist_ok=True)
 if root.resolve()!=root or root.is_symlink()or root.stat().st_uid!=os.getuid():raise ValueError('owned research namespace')
 from subnet.model import model_files
 import subnet.model
 if Path(subnet.model.__file__).resolve().parent.parent!=Path(s['source_path']).resolve():raise ValueError('scientific import path must be exact frozen source')
 if model_files(s['checkpoint_path'])!=c['files']:raise ValueError('exact parent full-model files')
 return s

def generate(s):
 from subnet.protocol import entry,harness_for
 from subnet.runtime_factory import runtime
 from subnet.batches import pack
 from subnet.artifact_budget import for_manifest
 from subnet.storage import canonical
 m=s['generation_manifest'];definition=entry(m,s['env_id']);streams={};R=None;total=0;root=Path(s['workspace'])
 for index in s['task_indices'][:s['generation_task_limit']]:
  if time.time()>=s['expires_at']:raise TimeoutError('original generation expired; no fake censored result')
  R=runtime(s['checkpoint_path'],m,definition['spec'],harness_for(definition,index))if R is None else R.for_environment(definition['spec'],harness_for(definition,index))
  streams[index]=[]
  for attempt in range(16):
   if time.time()>=s['expires_at']:raise TimeoutError('original generation expired')
   started=time.monotonic();row,arrays=verify_generated(R,index,attempt);row['wall_seconds']=time.monotonic()-started
   if arrays is not None:
    batch=dict(schema=2,epoch=m['epoch'],checkpoint=s['checkpoint'],env_id=s['env_id'],environment_version=row['rollout']['environment_version'],index=index,sample_index=index,rollouts=[row['rollout']])
    raw=pack([(batch,[arrays])],budget=for_manifest(m),stable=True);total+=len(raw)
    if total>s['output_limit_bytes']:raise ValueError('bounded original artifacts')
    name=f'jobs/{digest(s)}/submission-{s["task_indices"].index(index)*16+attempt}.zip';private_write(root/name,raw);row['artifact_sha256']=hashlib.sha256(raw).hexdigest();row['artifact_path']=name
    from subnet.cache_lifecycle import CacheLifecycle
    CacheLifecycle(root).record_download(root/name,row['artifact_sha256'])
   streams[index].append(row)
 arms,supply=select_matched(streams,definition)
 result=dict(version=s['version'],scope_sha256=digest(s),streams=streams,supply=supply,artifact_bytes=total,planned_tasks=s['task_count'],executed_tasks=len(streams),stage=s['stage'],matched_tasks=len(arms['1P1N']),arm_pair_counts={k:len(v)for k,v in arms.items()},native_class_claims_used_without_verification=False,production_state_changes=False)
 private_write(root/'generation-result.json',canonical(result));return result

def train(s,arm):
 from subnet.backend_jobs import get_object
 from subnet.gpu_runtime import GPURuntime
 from subnet.protocol import entry,harness_for
 from subnet.storage import canonical
 from subnet.model import model_files
 m=s['generation_manifest'];definition=entry(m,s['env_id']);root=Path(s['workspace']);g=json.loads((root/'generation-result.json').read_bytes())
 if g['scope_sha256']!=digest(s):raise ValueError('exact original generation scope')
 streams={int(k):v for k,v in g['streams'].items()}
 # Verify artifact integrity again; evidence remained under the exclusive owned scope.
 for rows in streams.values():
  for row in rows:
   if row.get('artifact_path'):
    p=root/row['artifact_path']
    if p.is_symlink()or hashlib.sha256(p.read_bytes()).hexdigest()!=row['artifact_sha256']:raise ValueError('unchanged original artifact')
 arms,supply=select_matched(streams,definition);pairs=arms[arm]
 if len(arms['1P1N'])<s['minimum_matched_tasks']:raise ValueError('censored supply: insufficient predeclared matched tasks; no training')
 R=GPURuntime(s['checkpoint_path'],m['checkpoint']['files'],definition['spec'],harness_for(definition,s['task_indices'][0]),runtime_revision='cuda-bf16-eager-sm90-v1')
 parent=s['parent_descriptor'];shards={x['name']:x for x in parent['shards']}
 def fetch(name,path):
  r=shards[name];get_object(s['parent_read_urls'][name],r['sha256'],path,r['size'])
 destination,metrics,evidence=train_arm(R,pairs,root/arm,s,parent,fetch)
 files=model_files(destination);result=dict(arm=arm,checkpoint_files=files,checkpoint=digest(files),parent=s['checkpoint'],parent_optimizer_step=parent['optimizer_steps'],research_optimizer_step=parent['optimizer_steps']+1,optimizer_state_durably_exported=False,production_state_changes=False,metrics=metrics,restore_evidence=evidence)
 private_write(root/(arm+'-training-result.json'),canonical(result));return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--scope',required=True);p.add_argument('--authority',required=True);p.add_argument('--phase',choices=['generate','1P1N','2P2N'],required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();s=validate_scope(json.loads(Path(a.scope).read_bytes()),a.authority,a.phase,time.time())
 if not a.execute:print(json.dumps(dict(CPU_contract_check=True,GPU_started=False,phase=a.phase)));return
 marker=Path(s['workspace'])/(a.phase+'.execution-intent');private_write(marker,json.dumps(dict(scope_sha256=digest(s),created_at=time.time())).encode())
 value=generate(s)if a.phase=='generate'else train(s,a.phase)
 print(json.dumps(dict(phase=a.phase,production_state_changes=False,result_ready=True)))
if __name__=='__main__':main()
