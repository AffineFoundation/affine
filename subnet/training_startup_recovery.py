"""Explicit operator-authorized replacement of one proven pre-compute failure.

This is not original-job resume or proof that remote execution happened. The
operator witnesses terminal process absence and an empty output namespace.
"""
import copy,json,math,re,time
from pathlib import Path
from .storage import canonical
from .backend_jobs import signed
from .training_receipts import sha

VERSION='terminal-training-startup-recovery-v1'
FIELD='training_startup_recovery'
SCIENCE=('model','gpu_runtime','proofs','forced_sampling','sampling_contract','epoch_optimizer','covered_epoch_optimizer','task_normalized_training','persistent_cpu_adamw','persistent_training_state','persistent_training_worker')

def original_manifest(manifest,authority):
    value=signed(manifest[FIELD],authority)
    original=signed(value['original_signed_job'],authority)
    old=signed(original['manifest'],authority)
    normalized=copy.deepcopy(manifest);normalized.pop(FIELD,None);normalized['source_bundle']=old['source_bundle']
    def without_caps(m):
        m=copy.deepcopy(m);m['checkpoint'].pop('read_urls',None);return m
    if without_caps(normalized)!=without_caps(old):raise ValueError('startup recovery changed original computation/parent/coverage')
    return old

def input_inventory(rows):
    return [{k:copy.deepcopy(v)for k,v in row.items()if k!='url'}for row in rows]

def validate(job,manifest,authority):
    if FIELD not in manifest:return None
    value=signed(manifest[FIELD],authority)
    fields={'version','epoch','original_signed_job','original_job_sha256','original_terminal','startup_witness','replacement_source_bundle','replacement_job_label','created_at','expires_at'}
    if set(value)!=fields or value['version']!=VERSION or value['epoch']!=manifest['epoch']:raise ValueError('exact startup recovery declaration')
    original=signed(value['original_signed_job'],authority);old=original_manifest(manifest,authority)
    if (original.get('role')!='train' or original.get('training_policy')!='bf16-cpu-fp32-master-task-normalized-persistent-v4' or original.get('training_input_policy')!='authenticated-verifier-compact-inputs-v2'
        or FIELD in old or old.get('training_execution_amendment')is not None or sha(original)!=value['original_job_sha256']):raise ValueError('startup recovery original signed request')
    terminal=value['original_terminal'];witness=value['startup_witness']
    if (set(terminal)!={'phase','job_id','exit_code','runner_pid','runner_pid_ticks','child_pid','child_pid_ticks','started_at','finished_at'} or terminal['phase']!='failed' or terminal['job_id']!=original['job_id'] or type(terminal['exit_code'])is not int or terminal['exit_code']==0):raise ValueError('startup recovery original terminal failure')
    for p in ('runner','child'):
        if type(terminal[p+'_pid'])is not int or terminal[p+'_pid']<=0 or not isinstance(terminal[p+'_pid_ticks'],str)or not terminal[p+'_pid_ticks'].isdigit():raise ValueError('original startup PID/ticks')
    if (set(witness)!={'version','observed_at','exception','execution_started','cuda_allocated','model_loaded','original_processes_absent','output_namespace_empty','physical_gpu_idle','evidence_sha256'} or witness['version']!='operator-startup-failure-witness-v1'
        or witness['exception']!='fresh-source-bootstrap-admission' or any(witness[k]is not False for k in ('execution_started','cuda_allocated','model_loaded')) or any(witness[k]is not True for k in ('original_processes_absent','output_namespace_empty','physical_gpu_idle')) or re.fullmatch('[0-9a-f]{64}',witness['evidence_sha256'])is None):raise ValueError('startup recovery requires original pre-compute/no-output witness')
    times=[original['created_at'],terminal['started_at'],terminal['finished_at'],witness['observed_at'],value['created_at'],job['created_at'],value['expires_at']]
    if type(job.get('steps'))is not int or type(original.get('steps'))is not int or type(job.get('expires_at'))not in(int,float)or not math.isfinite(job['expires_at'])or not job['created_at']<job['expires_at']<=value['expires_at']:raise ValueError('startup recovery bounded replacement expiry/steps')
    if any(type(x)not in(int,float)or not math.isfinite(x)for x in times)or not times[0]<=times[1]<=times[2]<=times[3]<=times[4]<=times[5]<times[6]or not 0<times[6]-times[4]<=86400:raise ValueError('startup recovery original/new authorization times')
    label=value['replacement_job_label']
    if not isinstance(label,str)or re.fullmatch('[A-Za-z0-9_-]{1,100}',label)is None or not job['job_id'].startswith(label+'-')or job['job_id']==original['job_id']:raise ValueError('startup recovery distinct request identity')
    if (job.get('role')!='train' or job.get('training_policy')!=original['training_policy'] or job.get('training_input_policy')!=original['training_input_policy'] or job.get('steps')!=original['steps'] or input_inventory(job['submissions'])!=input_inventory(original['submissions']) or job['runtime_versions']!=original['runtime_versions']):raise ValueError('startup recovery exact original training inputs/objective/runtime')
    source=value['replacement_source_bundle']
    if source!=manifest['source_bundle']or source.get('sha256')==old['source_bundle'].get('sha256')or re.fullmatch('[0-9a-f]{64}',source.get('sha256',''))is None or 'subnet/training_startup_recovery.py'not in job['source_files']:raise ValueError('startup recovery replacement source pin')
    for module in SCIENCE:
        name='subnet/'+module+'.py'
        if name in original['source_files']and job['source_files'].get(name)!=original['source_files'][name]:raise ValueError('startup recovery scientific implementation changed: '+name)
    if job.get('persistent_training',{}).get('output_namespace')==original.get('persistent_training',{}).get('output_namespace'):raise ValueError('startup recovery requires separate output namespace')
    return value

def declaration(controller,epoch):
    files=getattr(controller,'training_startup_recovery_files',{})
    if epoch not in files:return None
    document=json.loads(Path(files[epoch]).read_bytes());value=signed(document,controller.authority.id)
    if value.get('epoch')!=epoch:raise ValueError('startup recovery configured epoch')
    return document

def label(controller,epoch):
    document=declaration(controller,epoch)
    return document['payload']['replacement_job_label']if document else epoch+'-train'

def apply(controller,manifest,steps):
    """Use original durable inputs; never manufacture fresh verifier receipts."""
    document=declaration(controller,manifest['epoch']);value=document['payload'];original=signed(value['original_signed_job'],controller.authority.id)
    old=signed(original['manifest'],controller.authority.id)
    record=json.loads((controller.state/'roles'/(manifest['epoch']+'-train.json')).read_bytes())
    if record['job_id']!=original['job_id']or record['job_sha256']!=sha(original):raise ValueError('startup recovery immutable original record')
    failure=json.loads((controller.state/'roles'/(original['job_id']+'-failure.json')).read_bytes())
    if any(failure.get(k)!=v for k,v in value['original_terminal'].items()):raise ValueError('startup recovery actual original failure record')
    if (controller.state/'roles'/(original['job_id']+'-report.json')).exists():raise ValueError('startup recovery forbidden after original completion')
    if manifest['trainer_state_binding']!=old['trainer_state_binding']or steps!=original['steps']:raise ValueError('startup recovery current parent/steps changed')
    reservation=controller.state/(manifest['epoch']+'-startup-recovery-reservation.json')
    binding=dict(declaration_sha256=sha(document),declaration=document,label=value['replacement_job_label'],original_job_sha256=value['original_job_sha256'])
    if reservation.exists()and json.loads(reservation.read_bytes())!=binding:raise ValueError('one immutable startup recovery declaration per epoch')
    replacement=controller.state/'roles'/(value['replacement_job_label']+'.json')
    if replacement.exists():
        record=json.loads(replacement.read_bytes());replacement_job=signed(json.loads((controller.state/'roles'/(record['job_id']+'-job.json')).read_bytes()),controller.authority.id)
        if record['job_sha256']!=sha(replacement_job)or replacement_job['manifest']['payload'].get(FIELD)!=document:raise ValueError('immutable replacement startup recovery request')
        validate(replacement_job,replacement_job['manifest']['payload'],controller.authority.id)
        return replacement_job['manifest']['payload'],replacement_job['submissions']
    if (controller.state/(manifest['epoch']+'-training-metrics.json')).exists():raise ValueError('startup recovery cannot replace completed epoch')
    parent_path=controller.state/'latest-trainer-state.json'
    if not parent_path.exists()or json.loads(parent_path.read_bytes())!=old['trainer_state_binding']['parent']:raise ValueError('startup recovery authoritative parent changed or absent')
    result=copy.deepcopy(old);result['source_bundle']=copy.deepcopy(value['replacement_source_bundle']);result[FIELD]=document
    result['checkpoint']=controller.checkpoint_with_reads(old['checkpoint'])
    rows=copy.deepcopy(original['submissions'])
    for obj in rows:obj['url']=controller.bucket.presign('private/compact-training-inputs/'+obj['sha256']+'.json')
    # Validate the declaration before reserving anything. This unsigned
    # preparation skeleton is never dispatched or presented as runtime evidence.
    candidate=copy.deepcopy(original);candidate.update(job_id=value['replacement_job_label']+'-preparation',created_at=time.time(),expires_at=value['expires_at'],manifest=controller.signed(result),submissions=rows)
    candidate['source_files']=dict(original['source_files'],**{'subnet/training_startup_recovery.py':'0'*64})
    candidate['persistent_training']=dict(original['persistent_training'],output_namespace='private/startup-recovery-preparation-only')
    validate(candidate,result,controller.authority.id)
    if not reservation.exists():
        from .remote_backend import save
        save(reservation,binding)
    return result,rows


def local_request(state,epoch,authority):
    """Select a replacement only through its immutable signed reservation."""
    state=Path(state);reservation=state/(epoch+'-startup-recovery-reservation.json')
    original_record=json.loads((state/'roles'/(epoch+'-train.json')).read_bytes())
    if not reservation.exists():
        job=signed(json.loads((state/'roles'/(original_record['job_id']+'-job.json')).read_bytes()),authority)
        return original_record,job,None
    value=json.loads(reservation.read_bytes())
    if set(value)!={'declaration_sha256','declaration','label','original_job_sha256'}or sha(value['declaration'])!=value['declaration_sha256']:raise ValueError('immutable local startup recovery reservation')
    declaration=signed(value['declaration'],authority)
    if value['label']!=declaration['replacement_job_label']or value['original_job_sha256']!=declaration['original_job_sha256']or original_record['job_sha256']!=value['original_job_sha256']:raise ValueError('local original startup recovery identity')
    original=signed(declaration['original_signed_job'],authority)
    if original_record['job_id']!=original['job_id']:raise ValueError('original recovery job record changed')
    failure=json.loads((state/'roles'/(original['job_id']+'-failure.json')).read_bytes())
    if any(failure.get(k)!=v for k,v in declaration['original_terminal'].items())or (state/'roles'/(original['job_id']+'-report.json')).exists():raise ValueError('original startup failure evidence changed')
    record=json.loads((state/'roles'/(value['label']+'.json')).read_bytes())
    job=signed(json.loads((state/'roles'/(record['job_id']+'-job.json')).read_bytes()),authority)
    if sha(job)!=record['job_sha256']or job['manifest']['payload'].get(FIELD)!=value['declaration']:raise ValueError('replacement recovery record changed')
    validate(job,job['manifest']['payload'],authority)
    evidence=dict(version=VERSION,declaration_sha256=value['declaration_sha256'],original_failed_job_id=original['job_id'],original_failed_job_sha256=sha(original),original_failure_sha256=sha(failure),startup_witness_sha256=sha(declaration['startup_witness']),replacement_source_sha256=job['manifest']['payload']['source_bundle']['sha256'],original_epoch_deadline=original['manifest']['payload'].get('deadline'),late_recovery=True)
    return record,job,evidence
