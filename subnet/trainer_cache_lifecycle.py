"""CPU-only post-durability cleanup of explicit owned trainer paths."""
import hashlib,json,os,re
from pathlib import Path
from .backend_jobs import signed
from .storage import canonical
from .cache_lifecycle import CacheLifecycle,identifier
VERSION='durable-original-trainer-cache-ACK-v1'
MARKER_VERSION='genesis-bound-trainer-retention-v2'
TRANSITION_VERSION='trainer-cache-run-transition-v1'

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()

def original_ack(ack,authority,workspace):
    """Derive lineage from the original signed ACK/job and hash-bound report."""
    value=signed(ack,authority);root=Path(workspace).absolute()
    required={'version','job_id','job_sha256','report_sha256','input_checkpoint','input_cache','new_checkpoint','trainer_state','authority_state_committed'}
    if set(value)!=required or value['version']!=VERSION or value['authority_state_committed']is not True:
        raise ValueError('exact durable trainer cleanup ACK')
    jobid=identifier(value['job_id']);job=signed(json.loads((root/(jobid+'.json')).read_bytes()),authority)
    report=json.loads((root/'jobs'/jobid/'report.json').read_bytes())
    if (digest(job)!=value['job_sha256']or digest(report)!=value['report_sha256']or
        job.get('role')!='train'or report.get('success')is not True or
        report.get('job_sha256')!=value['job_sha256']or report.get('job_id')!=jobid or
        report.get('new_checkpoint')!=value['new_checkpoint']):raise ValueError('original completed trainer binding')
    manifest=signed(job['manifest'],authority)
    if manifest['checkpoint']!=value['input_checkpoint']:raise ValueError('original input checkpoint inventory')
    state=report['persistent_training_state'];descriptor=state['descriptor'];pointer=value['trainer_state']
    genesis=descriptor.get('genesis_sha256');step=descriptor.get('optimizer_steps')
    if (not isinstance(genesis,str) or re.fullmatch('[0-9a-f]{64}',genesis)is None or
        type(step)is not int or step<0 or digest(descriptor)!=state['descriptor_sha256']or
        state['descriptor_sha256']!=pointer['descriptor_sha256']or state['namespace']!=pointer['namespace']or
        step!=pointer['optimizer_steps']or genesis!=pointer.get('genesis_sha256')or
        descriptor.get('inference_checkpoint')!=value['new_checkpoint']['id']or
        genesis!=manifest.get('trainer_state_binding',{}).get('genesis_sha256')):
        raise ValueError('committed optimizer state original report lineage binding')
    lineage=dict(genesis_sha256=genesis,optimizer_steps=step,descriptor_sha256=state['descriptor_sha256'],
                 inference_checkpoint=descriptor['inference_checkpoint'])
    return value,manifest,report,lineage

def _previous(previous,ack,authority,root,lineage):
    """An old two-field marker carries no independent genesis assertion."""
    if previous.get('version')==MARKER_VERSION:
        fields={'version','genesis_sha256','optimizer_steps','descriptor_sha256','inference_checkpoint','ROOT_ack','retired_geneses'}
        if set(previous)!=fields:raise ValueError('exact genesis-bound trainer retention marker')
        _,_,_,bound=original_ack(previous['ROOT_ack'],authority,root)
        retired=previous['retired_geneses']
        if (any(previous[k]!=v for k,v in bound.items()) or not isinstance(retired,list)or
            len(retired)!=len(set(retired))or bound['genesis_sha256']in retired or
            any(not isinstance(g,str)or re.fullmatch('[0-9a-f]{64}',g)is None for g in retired)):
            raise ValueError('authenticated trainer retention marker lineage')
        return bound,list(retired),previous['ROOT_ack']
    if set(previous)!={'optimizer_steps','descriptor_sha256'} or type(previous['optimizer_steps'])is not int:
        raise ValueError('unknown legacy trainer retention marker')
    if all(previous[k]==lineage[k]for k in previous):return lineage,[],ack
    # A legacy foreign marker needs an explicitly authenticated original ACK;
    # its larger numeric counter is never enough to supersede a fresh run.
    transition=signed(json.loads((root/'.cache-lifecycle'/'trainer-run-transition.json').read_bytes()),authority)
    oldack=transition['previous_ack'];_,_,_,bound=original_ack(oldack,authority,root)
    if any(previous[k]!=bound[k]for k in previous):raise ValueError('legacy marker original ACK binding')
    return bound,[],oldack

def _transition(previous,previous_ack,ack,authority,root,before,after,retired):
    path=root/'.cache-lifecycle'/'trainer-run-transition.json'
    value=signed(json.loads(path.read_bytes()),authority)
    fields={'version','workspace','previous_marker_sha256','previous_ack','next_ack_sha256','from_genesis_sha256','to_genesis_sha256'}
    if (set(value)!=fields or value['version']!=TRANSITION_VERSION or value['workspace']!=str(root)or
        value['previous_marker_sha256']!=digest(previous)or value['previous_ack']!=previous_ack or
        value['next_ack_sha256']!=digest(ack)or value['from_genesis_sha256']!=before['genesis_sha256']or
        value['to_genesis_sha256']!=after['genesis_sha256']or before['genesis_sha256']==after['genesis_sha256']or
        after['genesis_sha256']in retired):
        raise ValueError('exact ROOT-authorized trainer run transition')
    return retired+[before['genesis_sha256']]

def migrate_legacy_marker(ack,authority,workspace):
    """Authenticate the exact old marker without retiring or promoting any bytes.

    Before an upgrade, supply its original ACK (not the incoming next job's).
    This permits ordinary same-genesis advancement after a two-field marker.
    """
    root=Path(workspace).absolute();_,_,_,lineage=original_ack(ack,authority,root)
    lifecycle=CacheLifecycle(root)
    with lifecycle.lease_checkpoint('trainer-state-retention',blocking=False):
        marker=lifecycle.meta/'trainer-current-state.json';previous=json.loads(marker.read_bytes())
        expected=dict(version=MARKER_VERSION,**lineage,ROOT_ack=ack,retired_geneses=[])
        if previous==expected:return dict(migrated=True,idempotent=True)
        if (set(previous)!={'optimizer_steps','descriptor_sha256'} or
            any(previous[k]!=lineage[k]for k in previous)):
            raise ValueError('legacy migration requires exact original marker ACK')
        lifecycle._save(marker,expected)
        return dict(migrated=True,idempotent=False)

def _promoted_guard(root,authority,lineage):
    """A successfully promoted cache also guards a crash before marker commit."""
    path=root/'.optimizer-state-cache'/'current.json'
    if not path.exists():return
    current=json.loads(path.read_bytes())
    value,_,_,bound=original_ack(current['ROOT_ack'],authority,root)
    if (current.get('descriptor_sha256')!=bound['descriptor_sha256']or current.get('job_id')!=value['job_id']or
        current.get('job_sha256')!=value['job_sha256']):
        raise ValueError('authenticated promoted optimizer head binding')
    if bound['genesis_sha256']!=lineage['genesis_sha256']:
        raise ValueError('trainer cleanup differs from promoted optimizer genesis')
    if bound['optimizer_steps']>lineage['optimizer_steps']or (
        bound['optimizer_steps']==lineage['optimizer_steps']and bound['descriptor_sha256']!=lineage['descriptor_sha256']):
        raise ValueError('trainer cleanup behind promoted optimizer lineage')

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
    root=Path(workspace).absolute();value,manifest,report,lineage=original_ack(ack,authority,root)
    jobid=value['job_id']
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
        retired=[]
        if previous:
            before,retired,previous_ack=_previous(previous,ack,authority,root,lineage)
            if lineage['genesis_sha256']in retired:
                return dict(status='superseded',reason='retired-genesis',removed_checkpoints=[],retired_downloads=[])
            if before['genesis_sha256']!=lineage['genesis_sha256']:
                retired=_transition(previous,previous_ack,ack,authority,root,before,lineage,retired)
            elif before['optimizer_steps']>lineage['optimizer_steps']:
                return dict(status='superseded',removed_checkpoints=[],retired_downloads=[])
            elif before['optimizer_steps']==lineage['optimizer_steps']and before['descriptor_sha256']!=lineage['descriptor_sha256']:
                raise ValueError('same counter different cleanup lineage')
        promotion=None
        if manifest.get('optimizer_state_local_cache')is not None:
            from .optimizer_state_cache import promote
            promotion=promote(ack,authority,workspace)
            if not isinstance(promotion,dict)or promotion.get('promoted')is not True:
                raise ValueError('optimizer promotion required before trainer cleanup')
        _promoted_guard(root,authority,lineage)
        result=_retire_owned(lifecycle,value,ack,jobid,root)
        lifecycle._save(marker,dict(version=MARKER_VERSION,**lineage,ROOT_ack=ack,retired_geneses=retired))
        if promotion is not None:result['optimizer_cache_promotion']=promotion
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
