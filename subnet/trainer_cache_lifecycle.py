"""CPU-only post-durability cleanup of explicit owned trainer paths."""
import hashlib,json,os
from pathlib import Path
from .backend_jobs import signed
from .storage import canonical
from .cache_lifecycle import CacheLifecycle,identifier
VERSION='durable-original-trainer-cache-ACK-v1'

def live_original(status):
    for field in ('runner_pid','child_pid'):
        pid=status.get(field);ticks=status.get(field+'_ticks')
        if not pid or ticks is None:continue
        p=Path('/proc')/str(pid)/'stat'
        try:fields=p.read_text().rsplit(')',1)[1].split()
        except FileNotFoundError:continue
        if fields[0]!='Z'and fields[19]==str(ticks):return True
    return False

def retire(ack,authority,workspace):
    value=signed(ack,authority);root=Path(workspace).absolute()
    required={'version','job_id','job_sha256','report_sha256','input_checkpoint','input_cache','new_checkpoint','trainer_state','authority_state_committed'}
    if set(value)!=required or value['version']!=VERSION or value['authority_state_committed']is not True:
        raise ValueError('exact durable trainer cleanup ACK')
    jobid=identifier(value['job_id']);job=signed(json.loads((root/(jobid+'.json')).read_bytes()),authority)
    report=json.loads((root/'jobs'/jobid/'report.json').read_bytes())
    if (hashlib.sha256(canonical(job)).hexdigest()!=value['job_sha256']or
        hashlib.sha256(canonical(report)).hexdigest()!=value['report_sha256']or
        job.get('role')!='train'or report.get('success')is not True or
        report.get('job_sha256')!=value['job_sha256']or report.get('job_id')!=jobid or
        report.get('new_checkpoint')!=value['new_checkpoint']):raise ValueError('original completed trainer binding')
    manifest=signed(job['manifest'],authority)
    if manifest['checkpoint']!=value['input_checkpoint']:raise ValueError('original input checkpoint inventory')
    state=report['persistent_training_state']
    if (state['descriptor_sha256']!=value['trainer_state']['descriptor_sha256']or
        state['namespace']!=value['trainer_state']['namespace']or
        state['descriptor']['optimizer_steps']!=value['trainer_state']['optimizer_steps']):
        raise ValueError('committed optimizer state original report binding')
    status=json.loads((root/'runner-status'/(jobid+'.json')).read_bytes())
    if status.get('phase')!='complete'or status.get('exit_code')!=0 or live_original(status):
        return dict(status='deferred',reason='original-child-not-terminal',removed_checkpoints=[],retired_downloads=[])
    # Conservative proc guard covers historical runners that did not inherit leases.
    for p in (root/'runner-status').glob('*.json'):
        if live_original(json.loads(p.read_bytes())):
            return dict(status='deferred',reason='workspace-role-in-flight',removed_checkpoints=[],retired_downloads=[])
    lifecycle=CacheLifecycle(root)
    with lifecycle.lease_checkpoint('trainer-state-retention',blocking=False):
        marker=lifecycle.meta/'trainer-current-state.json'
        previous=json.loads(marker.read_text())if marker.exists()else None
        if previous and previous['optimizer_steps']>value['trainer_state']['optimizer_steps']:
            return dict(status='superseded',removed_checkpoints=[],retired_downloads=[])
        if previous and previous['optimizer_steps']==value['trainer_state']['optimizer_steps']and previous['descriptor_sha256']!=value['trainer_state']['descriptor_sha256']:
            raise ValueError('same counter different cleanup lineage')
        result=_retire_owned(lifecycle,value,ack,jobid,root)
        lifecycle._save(marker,dict(optimizer_steps=value['trainer_state']['optimizer_steps'],descriptor_sha256=value['trainer_state']['descriptor_sha256']))
        return result

def _retire_owned(lifecycle,value,ack,jobid,root):
    new=value['new_checkpoint'];current=new['id'];old=value['input_checkpoint']['id']
    with lifecycle.lease_checkpoint(current,blocking=False):
        lifecycle.adopt_checkpoint(current,new['path'],new['files'],ack)
    if old!=current and value['input_cache']:
        path=Path(value['input_cache']).absolute()
        # Never adopt externally mapped caches. Explicit ownership is required.
        try:path.relative_to(root)
        except ValueError:pass
        else:
            if path.exists():
                with lifecycle.lease_checkpoint(old,blocking=False):
                    lifecycle.adopt_checkpoint(old,path,value['input_checkpoint']['files'],ack)
    downloads=lifecycle.retire_downloads(jobid)
    removed=lifecycle.evict_checkpoints(exclude=(current,),keep=0)
    return dict(status='complete',current_checkpoint=current,removed_checkpoints=removed,retired_downloads=downloads,extra_hashing=False,extra_R2_reads=False)
