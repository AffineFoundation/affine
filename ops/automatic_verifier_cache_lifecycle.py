"""Operator-only bounded verifier cache housekeeping and future launch maps.

Observe by default. Apply requires a separately pinned root policy; no worker,
controller, queue, identity, protection, model or archive configuration changes.
An uncertain per-role operation blocks further deletions for that role. This is
not installed as a service by importing or running the observation command.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import stat
import sqlite3
import inspect
from types import SimpleNamespace
import subprocess
import time
from pathlib import Path
from ops.retain_checkpoint_caches import (archived_files,endpoints,private_file,
    protections,scopes_for)
from ops.retain_completed_training import sha
from subnet.storage import Bucket,canonical
from subnet.backend_profiles import for_config
from subnet.artifact_budget import for_manifest
from subnet.distributed_roles import authenticate,digest
from subnet.remote_backend import RemoteJobs
from ops.retain_obsolete_training_exports import final_candidate
from ops.verifier_cache_reference_retirement import remaining_queue_references
from ops.verifier_redundant_cache_lifecycle import assert_unreferenced


def save(path,value):
    """Exclusive private fsynced record; no symlink-following or overwrite."""
    path=Path(path);parent=path.parent
    if (parent.resolve()!=parent.absolute() or not stat.S_ISDIR(parent.lstat().st_mode)
            or parent.stat().st_mode&0o077):
        raise ValueError('private canonical journal parent required')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb')as stream:
        actual=os.fstat(stream.fileno())
        if not stat.S_ISREG(actual.st_mode)or actual.st_nlink!=1:raise ValueError('ordinary exclusive journal')
        os.fchmod(stream.fileno(),0o600);stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
    fsync_directory(parent)


def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)


def private_output(path):
    path=Path(path)
    ancestor=path
    while not ancestor.exists():ancestor=ancestor.parent
    if ancestor.resolve()!=ancestor.absolute():raise ValueError('canonical existing output ancestor required')
    path.mkdir(mode=0o700,parents=True,exist_ok=True)
    if path.resolve()!=path.absolute()or not stat.S_ISDIR(path.lstat().st_mode)or path.stat().st_mode&0o077:
        raise ValueError('private canonical output')
    return path

PROBE = '''import json,os,re,stat,subprocess,time
from pathlib import Path
rows=[];mapped=[];scientific=[]
for p in Path('/proc').iterdir():
 if not p.name.isdecimal():continue
 try:s=(p/'stat').read_text().rsplit(')',1)[1].split();a=(p/'cmdline').read_bytes().decode().strip('\\0').split('\\0')
 except(FileNotFoundError,ProcessLookupError):continue
 if s[0]in('Z','X'):continue
 if '-m'in a and a[a.index('-m')+1]=='subnet.distributed_worker':
  for i,v in enumerate(a[:-1]):
   if v=='--checkpoint-cache':
    cp,path=a[i+1].split('=',1);assert re.fullmatch('[0-9a-f]{64}',cp);mapped.append(dict(checkpoint=cp,directory=path,pid=int(p.name),ticks=s[19]))
 elif ('-m'in a and a[a.index('-m')+1]in('subnet.backend_jobs','subnet.remote_runner','subnet.cli'))or any('affine-original-qwen-'in v for v in a):scientific.append(dict(pid=int(p.name),ticks=s[19]))
for value in ROOTS:
 base=Path(value)
 if not base.exists():continue
 assert base.resolve()==base and stat.S_ISDIR(base.lstat().st_mode)
 children=list(base.iterdir());assert len(children)<=128
 for p in children:
  if not re.fullmatch('[0-9a-f]{64}',p.name):continue
  assert p.resolve()==p and stat.S_ISDIR(p.lstat().st_mode)
  members=list(p.iterdir());assert 1<=len(members)<=32
  meta={n.name:dict(size=n.lstat().st_size,links=n.lstat().st_nlink,ordinary=stat.S_ISREG(n.lstat().st_mode),device=n.lstat().st_dev,inode=n.lstat().st_ino,allocated_bytes=n.lstat().st_blocks*512)for n in members}
  rows.append(dict(checkpoint=p.name,directory=str(p),files=meta,single_link=all(m['ordinary']and m['links']==1 for m in meta.values())))
location=Path(CAPACITY_LOCATION if 'CAPACITY_LOCATION'in globals()else ROOTS[0])
while not location.exists():location=location.parent
fs=os.statvfs(location)
print(json.dumps(dict(at=time.time(),candidates=rows,worker_mappings=mapped,scientific=scientific,free_bytes=fs.f_bavail*fs.f_frsize,gpu_busy=bool(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()))))
'''


def select_candidate(observations, protected, blocked_roles=()):
    """One ordinary single-link obsolete cache; mapped paths stay protected."""
    result=[]
    for role,obs in observations.items():
        if role in blocked_roles or obs['gpu_busy'] or obs['scientific']:continue
        mapped={row['checkpoint']for row in obs['worker_mappings']}
        for row in obs['candidates']:
            if row['checkpoint']not in set(protected)|mapped and row['single_link']and not row.get('catalog_only',False):
                result.append((obs['free_bytes'],role,row['directory'],row))
    return min(result,key=lambda r:r[:3])[1:] if result else None



def select_cluster(catalog,observations,protected,blocked_roles=()):
    choices=[]
    for row in catalog:
        role=row['role'];obs=observations[role];cp=row['checkpoint']
        if role in blocked_roles or obs['gpu_busy']or obs['scientific']or cp in set(protected)|{m['checkpoint']for m in obs['worker_mappings']}:continue
        candidates={r['directory']:r for r in obs['candidates']};paths=row['directories']
        if len(paths)!=2 or any(p not in candidates for p in paths):continue
        left,right=[candidates[p]for p in paths]
        if not left['files']or set(left['files'])!=set(right['files']):continue
        if any(not a['ordinary']or not right['files'][n]['ordinary']or a['links']!=2 or right['files'][n]['links']!=2
            or(a['device'],a['inode'],a['size'])!=(right['files'][n]['device'],right['files'][n]['inode'],right['files'][n]['size'])for n,a in left['files'].items()):continue
        selected=dict(left,kind='known-two-alias-cluster',directories=paths)
        choices.append((obs['free_bytes'],role,paths[0],selected))
    return min(choices,key=lambda r:r[:3])[1:]if choices else None

def verified_launch_map(required_files, observations, verify):
    """Future-only exact cache map, keyed by required authenticated checkpoint.

    verify(path,cp,files) must hash full file contents against the signed filemap.
    Existing mappings/incomplete directories never substitute for that check.
    Missing checkpoints are explicit and require capacity-safe scoped hydration.
    """
    if not isinstance(required_files,dict) or not 1<=len(required_files)<=32:
        raise ValueError('bounded required signed checkpoint maps')
    result={};missing=[]
    for cp,files in sorted(required_files.items()):
        if (not isinstance(cp,str)or not re.fullmatch('[0-9a-f]{64}',cp)or not isinstance(files,dict)or not 1<=len(files)<=32
                or 'config.json'not in files or not any(n.endswith('.safetensors')for n in files)
                or any(not isinstance(n,str)or not re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_.-]*',n)or not isinstance(d,str)or not re.fullmatch('[0-9a-f]{64}',d)for n,d in files.items())
                or hashlib.sha256(canonical(files)).hexdigest()!=cp):
            raise ValueError('canonical signed checkpoint filemap')
        candidates=[row for row in observations['candidates']if row['checkpoint']==cp
                    and set(row['files'])==set(files)and all(m['ordinary']for m in row['files'].values())]
        candidates.sort(key=lambda r:r['directory'])
        for row in candidates:
            if verify(row['directory'],cp,files):
                result[cp]=row['directory'];break
        else:missing.append(cp)
    return dict(checkpoint_caches=result,missing_checkpoints=missing,
                future_launch_only=True,live_worker_changed=False,
                cli_arguments=[arg for cp,path in sorted(result.items())for arg in ('--checkpoint-cache',cp+'='+path)])





def future_worker_arguments(original,launch_map):
    """Future boundary only: replace stale maps, preserve every other argument."""
    if launch_map.get('missing_checkpoints')or launch_map.get('capacity',{}).get('admitted')is not True or launch_map.get('future_launch_only')is not True:
        raise ValueError('fully admitted future cache map required')
    result=[];i=0
    while i<len(original):
        value=original[i]
        if value=='--checkpoint-cache':
            if i+1==len(original):raise ValueError('missing old cache argument')
            i+=2;continue
        if value.startswith('--checkpoint-cache='):i+=1;continue
        result.append(value);i+=1
    for cp,path in sorted(launch_map['checkpoint_caches'].items()):
        if not re.fullmatch('[0-9a-f]{64}',cp)or not Path(path).is_absolute()or '..'in Path(path).parts:
            raise ValueError('exact admitted same-host cache mapping')
        result.extend(['--checkpoint-cache',cp+'='+path])
    return result


def journalled_operation(output,cycle,role,record,perform):
    if role not in('verify1','verify2'):raise ValueError('exact verifier role required')
    output=Path(output);cycle=Path(cycle);pending=output/(role+'-pending-operation.private.json')
    save(pending,dict(record,automatic_repeat_forbidden=True))
    try:
        result=perform();save(cycle/'actual-retirement.private.json',result)
        pending.rename(cycle/'completed-operation.private.json');fsync_directory(output);fsync_directory(cycle)
        return result
    except BaseException as error:
        save(cycle/'uncertain-operation.private.json',dict(role=role,error_type=type(error).__name__,at=time.time(),automatic_repeat_forbidden=True,pending_record=str(pending)))
        raise

def capacity_admission(launch_map,readbacks,*,free_bytes,artifact_bytes,reserve_bytes):
    if any(type(v)is not int or v<0 for v in(free_bytes,artifact_bytes,reserve_bytes)):
        raise ValueError('explicit integer capacity budgets')
    needed=0
    for cp in launch_map['missing_checkpoints']:
        files=readbacks[cp]
        if not files or any(type(r.get('size'))is not int or r['size']<=0 or r.get('archive_verified')is not True for r in files.values()):
            raise ValueError('actual verified archive sizes required')
        needed+=sum(r['size']for r in files.values())
    required=needed+artifact_bytes+reserve_bytes
    return dict(admitted=free_bytes>=required,free_bytes=free_bytes,required_free_bytes=required,
        missing_checkpoint_bytes=needed,artifact_bytes=artifact_bytes,reserve_bytes=reserve_bytes,
        reason=None if free_bytes>=required else 'insufficient verifier capacity; no hydration or worker admission')

def non_queue_references(config,authority):
    """Independent guards that a queue-reference ledger may never remove."""
    state=Path(config['state']);roles=state/'roles';status=json.loads((state/'controller.json').read_text())
    protected={status['checkpoint']['id']};active=status.get('active')or{}
    if active.get('next_checkpoint'):protected.add(active['next_checkpoint']['id'])
    if active.get('epoch'):
        manifest=authenticate(json.loads((state/(active['epoch']+'-first-signed-manifest.json')).read_text()),authority)
        if manifest['epoch']!=active['epoch']:raise ValueError('active epoch source binding')
        protected.add(manifest['checkpoint']['id'])
    checker=RemoteJobs.__new__(RemoteJobs);checker.state=roles;checker.controller=SimpleNamespace(authority=SimpleNamespace(id=authority))
    for path in roles.glob('*.json'):
        if path.name.endswith(('-job.json','-report.json','-failure.json')):continue
        prior=json.loads(path.read_text())
        if not isinstance(prior,dict)or prior.get('role')not in('mine','train','evaluate'):continue
        job=authenticate(json.loads((roles/(prior['job_id']+'-job.json')).read_text()),authority);manifest=authenticate(job['manifest'],authority)
        if job['role']!=prior['role']or job['job_id']!=prior['job_id']or digest(job)!=prior['job_sha256']:raise ValueError('original dispatch binding')
        reportpath=roles/(prior['job_id']+'-report.json')
        if not reportpath.exists():protected.add(manifest['checkpoint']['id']);continue
        report=json.loads(reportpath.read_text());checker.checked(report,prior,manifest)
        if job['role']=='train'and manifest['epoch']==active.get('epoch')and active.get('phase')in('train','after'):
            protected.add(final_candidate(job,report)['checkpoint'])
    return protected


def effective_protections(config_path,process_record,authority,reference_ledger,bucket):
    config,principal,active=protections(config_path,process_record,authority)
    if reference_ledger is None:return config,principal,active,None
    ledger_envelope=json.loads(private_file(reference_ledger).read_text())
    ledger=authenticate(ledger_envelope,authority)
    if ledger.get('history_prefix')!=config['remote']['verifier_queue']['history_prefix']:
        raise ValueError('original canonical history prefix required')
    with sqlite3.connect('file:'+str(Path(config['state'])/'roles/verifier-queue.sqlite3')+'?mode=ro',uri=True)as db:
        db.row_factory=sqlite3.Row;rows=[dict(r)for r in db.execute('select * from jobs')]
    verified=remaining_queue_references(rows,ledger_envelope,authority,now=time.time(),max_attempts=config['remote']['verifier_queue']['max_attempts'],fetch_document=bucket.get)
    independent=non_queue_references(config,authority)
    return config,principal,(independent|verified['protected_checkpoints'])-principal,verified

def run_cycle(config_path,scopes_path,authority,output,process_record,*,apply=False,approval=None,prepare_maps=False,map_checkpoints=(),reference_ledger=None,alias_clusters=None):
    os.umask(0o077);config_path=private_file(config_path)
    original_config=json.loads(config_path.read_text());bucket=Bucket(original_config['bucket'])if reference_ledger is not None else None
    c,principal,active,reference_proof=effective_protections(config_path,process_record,authority,reference_ledger,bucket)
    record=json.loads(private_file(process_record).read_text())
    actual_argv=Path('/proc',str(record['child_pid']),'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    if record.get('argv')!=actual_argv or record.get('source_sha256')!=c['source_bundle']['sha256']:
        raise ValueError('actual controller command/source binding')
    scopes=scopes_for(scopes_path,config_path,c)
    scopes={k:v for k,v in scopes.items()if k in ('verify1','verify2')}
    if not scopes:raise ValueError('explicit verifier scopes required')
    out=private_output(output)
    peers=endpoints(c);pins=dict(output_path=str(out.resolve()),config_sha256=sha(config_path),scopes_sha256=sha(scopes_path),helper_sha256=sha(__file__),node_helper_sha256=sha(Path(__file__).with_name('checkpoint_retention.py')),controller_record_sha256=sha(process_record),reference_ledger_sha256=sha(private_file(reference_ledger))if reference_ledger else None,alias_catalog_sha256=sha(private_file(alias_clusters))if alias_clusters else None,reference_helper_sha256=sha(Path(__file__).with_name('verifier_cache_reference_retirement.py')),cluster_helper_sha256=sha(Path(__file__).with_name('verifier_obsolete_alias_cluster.py')),process_barrier_sha256=sha(Path(__file__).with_name('verifier_redundant_cache_lifecycle.py')))
    if apply:
        policy=json.loads(private_file(approval).read_text())
        if (policy.get('approved')is not True or type(policy.get('max_retirements_per_cycle'))is not int or policy.get('max_retirements_per_cycle')!=1
                or policy.get('authority')!=authority or any(policy.get(k)!=v for k,v in pins.items())):
            raise ValueError('explicit pinned automatic retention approval')
    cycle=out/('cycle-'+str(time.time_ns()));cycle.mkdir(mode=0o700)
    def remote(role,code,timeout=240):
        e=peers[role];command=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile='+e['known_hosts'],'-p',str(e['port']),e.get('user','root')+'@'+e['host'],shlex.quote(e['python'])+' -I -B -']
        r=subprocess.run(command,input=code,text=True,capture_output=True,timeout=timeout)
        if r.returncode:
            save(cycle/('transport-'+str(time.time_ns())+'.private.json'),dict(role=role,exit_code=r.returncode,stderr=r.stderr))
            raise RuntimeError('cache operation refused; preserve original evidence')
        return json.loads(r.stdout)
    catalog=[]
    if alias_clusters:
        document=json.loads(private_file(alias_clusters).read_text())
        if document.get('schema')!='known-two-alias-verifier-cache-clusters-v1'or document.get('config_sha256')!=pins['config_sha256']:
            raise ValueError('explicit original alias catalog binding')
        catalog=document.get('clusters')
        if not isinstance(catalog,list)or not 1<=len(catalog)<=8:raise ValueError('bounded known alias catalog')
        for row in catalog:
            if row['role']not in scopes or row['endpoint_sha256']!=hashlib.sha256(canonical(peers[row['role']])).hexdigest():raise ValueError('known role endpoint binding')
            cp=row['checkpoint'];paths=row['directories']
            if not re.fullmatch('[0-9a-f]{64}',cp)or len(paths)!=2 or len(set(paths))!=2:raise ValueError('known exact pair required')
            for path in paths:
                p=Path(path)
                if not p.is_absolute()or p.name!=cp or p.parent.name not in('checkpoint','checkpoints')or '..'in p.parts:raise ValueError('explicit bounded cache alias only')
    def observe_role(role):
        original=set(scopes[role]['roots']);known={p for r in catalog if r['role']==role for p in r['directories']}
        roots=sorted(original|{str(Path(p).parent)for p in known})
        obs=remote(role,'ROOTS='+repr(roots)+'\nCAPACITY_LOCATION='+repr(peers[role]['workspace'])+'\n'+PROBE)
        obs['candidates']=[dict(r,catalog_only=str(Path(r['directory']).parent)not in original)for r in obs['candidates']if str(Path(r['directory']).parent)in original or r['directory']in known]
        return obs
    observations={role:observe_role(role)for role in scopes}
    save(cycle/'observations.private.json',observations)
    protected=principal|active;blocked=[r for r in scopes if(out/(r+'-pending-operation.private.json')).exists()]
    selection=select_candidate(observations,protected,blocked)or select_cluster(catalog,observations,protected,blocked)
    result=dict(applied=False,removed_replicas=0,protected_checkpoints=sorted(protected),blocked_roles=blocked,future_maps={},selected=None)
    if reference_proof:save(cycle/'verified-reference-retirements.private.json',dict(retired_reference_ids=sorted(reference_proof['retired_reference_ids']),protected_checkpoints=sorted(reference_proof['protected_checkpoints'])))
    if prepare_maps:
        bucket=bucket or Bucket(c['bucket']);required={};required_readbacks={}
        requested=set(map_checkpoints)or set(principal)
        if not 1<=len(requested)<=8 or not requested<=protected:raise ValueError('bounded current/protected future map selection')
        for cp in sorted(requested):
            archived=archived_files(bucket,cp,authority)
            if archived is None:raise ValueError('required checkpoint has no authenticated archive')
            save(cycle/('map-readback-'+cp+'.private.json'),archived)
            required[cp]={n:r['sha256']for n,r in archived.items()};required_readbacks[cp]=archived
        for role,obs in observations.items():
            def verify(path,cp,files):
                code='ROOT='+repr(path)+'\nFILES='+repr(files)+'\n'+'''import hashlib,json,stat
from pathlib import Path
p=Path(ROOT);assert p.resolve()==p and stat.S_ISDIR(p.lstat().st_mode)
assert {f.name for f in p.iterdir()}==set(FILES)
for n,d in FILES.items():
 f=p/n;s=f.lstat();assert stat.S_ISREG(s.st_mode);h=hashlib.sha256()
 with f.open('rb')as stream:
  for b in iter(lambda:stream.read(1048576),b''):h.update(b)
 after=f.lstat();assert(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
 assert h.hexdigest()==d
print(json.dumps({'verified':True}))
'''
                return remote(role,code,timeout=600)['verified']
            mapping=verified_launch_map(required,obs,verify)
            revision,profile,policy=for_config(c)
            artifact=for_manifest(dict(c,model_runtime_revision=revision,backend_profile=profile,numerical_policy=policy))['compressed_bytes']
            fresh_capacity=observe_role(role)
            mapping['capacity']=capacity_admission(mapping,required_readbacks,free_bytes=fresh_capacity['free_bytes'],artifact_bytes=artifact,reserve_bytes=2*1024**3)
            result['future_maps'][role]=mapping
    if selection:
        role,path,candidate=selection;result['selected']=dict(role=role,directory=path,checkpoint=candidate['checkpoint'])
    if apply and selection:
        role,path,candidate=selection;cp=candidate['checkpoint'];bucket=bucket or Bucket(c['bucket']);readbacks=archived_files(bucket,cp,authority)
        if readbacks is None:result['deferred']='authenticated archive absent'
        else:
            save(cycle/'complete-archive-readback.private.json',readbacks)
            _,p,a,_=effective_protections(config_path,process_record,authority,reference_ledger,bucket)
            if sha(scopes_path)!=pins['scopes_sha256']or cp in p|a:raise ValueError('scope/reference changed during readback')
            fresh=observe_role(role)
            refreshed=select_candidate({role:fresh},p|a,blocked)or select_cluster([r for r in catalog if r['role']==role],{role:fresh},p|a,blocked)
            if refreshed!=(role,path,candidate):
                result['deferred']='cache/process/reference observation changed'
            else:
                plan=dict(checkpoint=cp,directory=path,protected_checkpoints=sorted(p),active_checkpoints=sorted(a),archive_verified=True,descriptor_authenticated=True,files={n:dict(sha256=r['sha256'],size=r['size'])for n,r in readbacks.items()})
                cluster=candidate.get('kind')=='known-two-alias-cluster'
                if cluster:
                    if reference_ledger is None:raise ValueError('cluster requires signed per-reference retirement policy')
                    plan.update(directories=candidate['directories'],worker_mapped_checkpoints=[m['checkpoint']for m in fresh['worker_mappings']],reference_retirements_verified=True,operation_directory=str(Path(peers[role]['workspace'])/('obsolete-cluster-'+cycle.name)))
                    helper=Path(__file__).with_name('verifier_obsolete_alias_cluster.py').read_text().replace('from ops.verifier_redundant_cache_lifecycle import assert_unreferenced','')
                    helper+='\n'+inspect.getsource(assert_unreferenced)
                else:helper=Path(__file__).with_name('checkpoint_retention.py').read_text()
                pending=out/(role+'-pending-operation.private.json')
                operation=helper+'\nPLAN='+repr(plan)+'\nfrom pathlib import Path\nROOTS='+repr(scopes[role]['roots'])+'\n'
                # Same process/GPU/mapped-cache refusal immediately before deletion.
                operation+='''
import contextlib,io
capture=io.StringIO()
with contextlib.redirect_stdout(capture):
 exec(PROBE)
obs=json.loads(capture.getvalue())
if obs['gpu_busy']or obs['scientific']or PLAN['checkpoint']in {m['checkpoint']for m in obs['worker_mappings']}:raise ValueError('fresh node reference barrier')
print(json.dumps(retire_obsolete_alias_cluster(PLAN,apply=True)if CLUSTER else remove_checkpoint_replica(PLAN)))
'''
                operation='PROBE='+repr(PROBE)+'\nCLUSTER='+repr(cluster)+'\nCAPACITY_LOCATION='+repr(peers[role]['workspace'])+'\n'+operation
                # Any timeout/refusal after this journal leaves pending in place.
                actual=journalled_operation(out,cycle,role,dict(at=time.time(),cycle=str(cycle),role=role,checkpoint=cp,plan=plan,pins=pins),lambda:remote(role,operation,timeout=900))
                result.update(applied=True,removed_replicas=int(actual.get('removed',actual.get('completed',False))),removed_bytes=actual.get('bytes',actual.get('estimated_reclaim_bytes',0)))
    save(cycle/'completion.private.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','scopes','authority','output','controller-process'):p.add_argument('--'+name,required=True)
    p.add_argument('--apply',action='store_true');p.add_argument('--approval');p.add_argument('--prepare-future-maps',action='store_true');p.add_argument('--map-checkpoint',action='append',default=[]);p.add_argument('--reference-retirements');p.add_argument('--known-alias-clusters');a=p.parse_args()
    output=private_output(a.output)
    fd=os.open(output/'automatic-cache.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=run_cycle(a.config,a.scopes,a.authority,output,a.controller_process,apply=a.apply,approval=a.approval,prepare_maps=a.prepare_future_maps,map_checkpoints=a.map_checkpoint,reference_ledger=a.reference_retirements,alias_clusters=a.known_alias_clusters)
        print(json.dumps({k:result[k]for k in ('applied','removed_replicas','blocked_roles')}))
if __name__=='__main__':main()
