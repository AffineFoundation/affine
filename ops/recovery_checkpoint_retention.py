"""Bounded obsolete model retention under an explicit held-failure ROOT scope.

Reuses the existing complete-archive and idle/open-FD/mapping protections.
The usual live-controller guard is not bypassed: this distinct authorization
requires an actual held, failed controller and unchanged original failure/state.
"""
import hashlib,os,re,stat,time,shutil
from pathlib import Path
from subnet.backend_jobs import signed
from subnet.storage import canonical
from ops import checkpoint_retention as retention

VERSION='held-uncommitted-recovery-obsolete-checkpoint-retention-v1'

def retire(envelope,authority,*,state_bytes,failure_bytes,held_guard,now=None):
    v=signed(envelope,authority);now=time.time()if now is None else now
    fields={'version','execute_allowed','created_at','expires_at','controller_state_sha256','original_failure_sha256','original_signed_job','original_job_sha256','held_controller','protected_checkpoints','active_checkpoints','required_free_bytes','max_retire_bytes','candidates'}
    if set(v)!=fields or v['version']!=VERSION or v['execute_allowed']is not True or not v['created_at']<=now<v['expires_at'] or not 0<v['expires_at']-v['created_at']<=3600:raise ValueError('explicit bounded held-failure retention scope')
    if hashlib.sha256(state_bytes).hexdigest()!=v['controller_state_sha256']or hashlib.sha256(failure_bytes).hexdigest()!=v['original_failure_sha256']:raise ValueError('unchanged actual controller/failure required')
    import json
    state=json.loads(state_bytes);failure=json.loads(failure_bytes);original=signed(v['original_signed_job'],authority);manifest=signed(original['manifest'],authority)
    if hashlib.sha256(canonical(original)).hexdigest()!=v['original_job_sha256']or failure.get('job_id')!=original['job_id']or failure.get('phase')!='failed'or failure.get('exit_code')!=1:raise ValueError('exact failed original request')
    current=state['checkpoint']['id']
    if current!=manifest['checkpoint']['id']or current not in v['protected_checkpoints']:raise ValueError('durable/current original input checkpoint protected')
    if type(v['required_free_bytes'])is not int or not 0<v['required_free_bytes']<=512*1024**3 or type(v['max_retire_bytes'])is not int or not 0<v['max_retire_bytes']<=64*1024**3 or not isinstance(v['candidates'],list)or not 1<=len(v['candidates'])<=2:raise ValueError('bounded disk-floor retention budget')
    results=[];total=0
    for row in v['candidates']:
        if set(row)!={'directory','legacy_directory','checkpoint_document','full_readback'}:raise ValueError('exact approved candidate')
        cp=signed(row['checkpoint_document'],authority);receipt=signed(row['full_readback'],authority)
        if receipt.get('version')!='complete-checkpoint-full-readback-v1'or receipt.get('checkpoint')!=cp['id']or receipt.get('complete')is not True or set(receipt.get('files',{}))!=set(cp['files'])or any(receipt['files'][n].get('sha256')!=h for n,h in cp['files'].items()):raise ValueError('authenticated complete original model readback')
        files=receipt['files']
        if not 1<=len(files)<=32 or 'config.json'not in files or not any(n.endswith('.safetensors')for n in files)or any(not re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_.-]*',n)or Path(n).suffix not in {'.json','.safetensors','.txt','.model','.jinja','.tiktoken'}or type(x.get('size'))is not int or not 0<x['size']<=(32 if n.endswith('.safetensors')else 5)*1024**3 for n,x in files.items()):raise ValueError('bounded safe archived model member inventory')
        size=sum(x['size']for x in files.values())
        if receipt.get('bytes_read')!=size:raise ValueError('full model bytes read, not HEAD metadata')
        path=Path(row['directory']);old=Path(row['legacy_directory'])if row['legacy_directory']is not None else path
        if old!=path and (old.name!='checkpoint'or old.parent!=path.parent.parent or path.parent.name!='checkpoints'):raise ValueError('exact legacy qualification model normalization')
        if held_guard(v['held_controller'],original,failure,state)is not True:raise ValueError('actual held failed controller required')
        if shutil.disk_usage(old.parent).free>=v['required_free_bytes']:break
        if total+size>v['max_retire_bytes']:raise ValueError('bounded owned retirement byte limit')
        plan=dict(checkpoint=cp['id'],files=files,directory=str(path),protected_checkpoints=v['protected_checkpoints'],active_checkpoints=v['active_checkpoints'],archive_verified=True,descriptor_authenticated=True)
        if old!=path and old.exists():
            # Full hashes and all idle/protection guards precede even a rename.
            if cp['id']in v['protected_checkpoints']+v['active_checkpoints']or retention.digest(cp['files'])!=cp['id']or path.name!=cp['id']or old.resolve()!=old or old.is_symlink()or path.exists()or {p.name for p in old.iterdir()}!=set(files):raise ValueError('exact obsolete legacy model path/inventory')
            if retention.gpu_processes():raise ValueError('GPU must be idle')
            before={}
            for name,expected in files.items():
                p=old/name;st=p.lstat()
                if not stat.S_ISREG(st.st_mode)or st.st_nlink!=1 or st.st_uid!=os.getuid()or st.st_size!=expected['size']:raise ValueError('owned single-link legacy model')
                h=hashlib.sha256()
                with p.open('rb')as f:
                    for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
                if h.hexdigest()!=expected['sha256']:raise ValueError('full original model bytes')
                before[name]=st
            for process in retention.processes():
                if not process.name.isdecimal():continue
                try:descriptors=list((process/'fd').iterdir())
                except FileNotFoundError:continue
                for fd in descriptors:
                    try:target=fd.readlink()
                    except FileNotFoundError:continue
                    if target.parent==old:raise ValueError('old model open FD')
                try:maps=(process/'maps').read_text()
                except FileNotFoundError:continue
                if any(len(line.split(None,5))==6 and line.split(None,5)[5].startswith(str(old)+'/')for line in maps.splitlines()):raise ValueError('old model memory mapped')
            if held_guard(v['held_controller'],original,failure,state)is not True or retention.gpu_processes():raise ValueError('held original/GPU changed before retention')
            for name,st in before.items():
                if (old/name).lstat()!=st:raise ValueError('legacy model changed before rename')
            path.parent.mkdir(mode=0o700,exist_ok=True);old.rename(path)
        if path.exists()and any(p.lstat().st_uid!=os.getuid()for p in path.iterdir()):raise ValueError('ordinary model ownership changed')
        if held_guard(v['held_controller'],original,failure,state)is not True:raise ValueError('held original changed before removal')
        result=retention.remove_checkpoint_replica(plan);results.append(result);total+=result.get('bytes',0)
    return dict(version=VERSION,retired_bytes=total,results=results,original_failure_preserved=True,current_checkpoint_preserved=True,disk_floor_reached=shutil.disk_usage(Path(v['candidates'][0]['directory']).parent.parent).free>=v['required_free_bytes'])
