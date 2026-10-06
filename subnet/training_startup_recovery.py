"""Explicit operator-authorized replacement of a witnessed zero-update failure.

This is not original-job resume or proof that remote execution happened. The
operator separately witnesses pre-compute or pre-update restore failure.
"""
import copy,hashlib,json,math,re,time
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
    if value.get('version')in RESTORE_VERSIONS:
        if value.get('original_input_source_sha256')!=old['source_bundle']['sha256']or value.get('replacement_execution_source_sha256')!=manifest['source_bundle']['sha256']or value.get('authorized_input_inventory_sha256')!=sha(input_inventory(original['submissions'])):raise ValueError('explicit restore old-input/new-execution manifest scope')
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
    if value.get('version')in RESTORE_VERSIONS:return validate_restore(job,manifest,authority)
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
    if value.get('version')==BOOTSTRAP_VERSION:validate_predecessor_local(controller.state,value,controller.authority.id)
    reservation=reservation_path(controller.state,manifest['epoch'],value.get('version'))
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
    if value.get('version')in RESTORE_VERSIONS:
        for obj,location in zip(rows,value['authorized_input_objects']):obj['url']=controller.bucket.presign(location['key'],'get_object',int(value['expires_at']-time.time()))
    else:
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
    state=Path(state);reservation=reservation_path(state,epoch,BOOTSTRAP_VERSION)
    if not reservation.exists():reservation=reservation_path(state,epoch,RESTORE_VERSION)
    original_record=json.loads((state/'roles'/(epoch+'-train.json')).read_bytes())
    if not reservation.exists():
        job=signed(json.loads((state/'roles'/(original_record['job_id']+'-job.json')).read_bytes()),authority)
        return original_record,job,None
    value=json.loads(reservation.read_bytes())
    if set(value)!={'declaration_sha256','declaration','label','original_job_sha256'}or sha(value['declaration'])!=value['declaration_sha256']:raise ValueError('immutable local startup recovery reservation')
    declaration=signed(value['declaration'],authority)
    if reservation==reservation_path(state,epoch,BOOTSTRAP_VERSION)and declaration.get('version')!=BOOTSTRAP_VERSION:raise ValueError('only explicit v3 continuation may occupy v3 reservation')
    if declaration['version']==BOOTSTRAP_VERSION:validate_predecessor_local(state,declaration,authority)
    if value['label']!=declaration['replacement_job_label']or value['original_job_sha256']!=declaration['original_job_sha256']or original_record['job_sha256']!=value['original_job_sha256']:raise ValueError('local original startup recovery identity')
    original=signed(declaration['original_signed_job'],authority)
    if original_record['job_id']!=original['job_id']:raise ValueError('original recovery job record changed')
    failure=json.loads((state/'roles'/(original['job_id']+'-failure.json')).read_bytes())
    if any(failure.get(k)!=v for k,v in declaration['original_terminal'].items())or (state/'roles'/(original['job_id']+'-report.json')).exists():raise ValueError('original startup failure evidence changed')
    record=json.loads((state/'roles'/(value['label']+'.json')).read_bytes())
    job=signed(json.loads((state/'roles'/(record['job_id']+'-job.json')).read_bytes()),authority)
    if sha(job)!=record['job_sha256']or job['manifest']['payload'].get(FIELD)!=value['declaration']:raise ValueError('replacement recovery record changed')
    validate(job,job['manifest']['payload'],authority)
    evidence=dict(version=declaration['version'],declaration_sha256=value['declaration_sha256'],original_failed_job_id=original['job_id'],original_failed_job_sha256=sha(original),original_failure_sha256=sha(failure),startup_witness_sha256=sha(declaration['restore_witness']if declaration['version']in RESTORE_VERSIONS else declaration['startup_witness']),replacement_source_sha256=job['manifest']['payload']['source_bundle']['sha256'],original_epoch_deadline=original['manifest']['payload'].get('deadline'),late_recovery=True)
    if declaration['version']==BOOTSTRAP_VERSION:evidence['bootstrap_predecessor']=declaration['predecessor']
    if declaration['version']in RESTORE_VERSIONS:evidence.update(original_input_source_sha256=declaration['original_input_source_sha256'],replacement_execution_source_sha256=declaration['replacement_execution_source_sha256'],original_failed_stage='parent-state-fetch-before-train_epoch',original_optimizer_updates=0)
    return record,job,evidence

RESTORE_VERSION='terminal-parent-restore-pre-update-recovery-v2'
RESTORE_WITNESS='operator-parent-restore-pre-update-witness-v1'
BOOTSTRAP_VERSION='terminal-parent-restore-pre-update-bootstrap-recovery-v3'
RESTORE_VERSIONS=(RESTORE_VERSION,BOOTSTRAP_VERSION)

def validate_restore(job,manifest,authority):
    """A distinct pre-update restore failure, never a relaxed startup witness."""
    value=signed(manifest[FIELD],authority)
    fields={'version','epoch','original_signed_job','original_job_sha256','original_terminal','restore_witness','replacement_source_bundle','replacement_job_label','created_at','expires_at','original_input_source_sha256','replacement_execution_source_sha256','authorized_input_inventory_sha256','authorized_input_objects'}
    if value.get('version')==BOOTSTRAP_VERSION:fields=fields|{'predecessor'}
    if set(value)!=fields or value['version']not in RESTORE_VERSIONS or value['epoch']!=manifest['epoch']:raise ValueError('exact pre-update restore recovery declaration')
    if value['version']==BOOTSTRAP_VERSION:validate_bootstrap_predecessor(value,job)
    original=signed(value['original_signed_job'],authority);old=original_manifest(manifest,authority)
    if original.get('role')!='train' or original.get('training_policy')!='bf16-cpu-fp32-master-task-normalized-persistent-v4' or original.get('training_input_policy')!='committed-unaudited-training-v1' or FIELD in old or old.get('training_execution_amendment')is not None or sha(original)!=value['original_job_sha256']:raise ValueError('restore recovery original signed unaudited request')
    terminal=value['original_terminal'];w=value['restore_witness']
    if set(terminal)!={'phase','job_id','exit_code','runner_pid','runner_pid_ticks','child_pid','child_pid_ticks','started_at','finished_at'}or terminal['phase']!='failed'or terminal['job_id']!=original['job_id']or type(terminal['exit_code'])is not int or terminal['exit_code']!=1:raise ValueError('exact original failed restore terminal')
    for p in ('runner','child'):
        if type(terminal[p+'_pid'])is not int or terminal[p+'_pid']<=0 or not isinstance(terminal[p+'_pid_ticks'],str)or not terminal[p+'_pid_ticks'].isdigit():raise ValueError('original restore PID/start identity')
    fields={'version','observed_at','exception','cause','failed_stage','callchain','model_loaded','cuda_allocated','original_processes_absent','physical_gpu_idle','optimizer_step_reached','restore_state_returned','train_epoch_reached','output_checkpoint_absent','original_report_absent','optimizer_state_candidate_absent','update_ledger_absent','public_optimizer_steps','parent_publication_sha256','parent_descriptor_sha256','parent_shard_count','parent_total_bytes','selected_input_count','worker_log_sha256','evidence_sha256','science_source_files'}
    required_frames=['persistent_training_worker.train','persistent_training_state.restore_state','persistent_training_state.restore_one','persistent_training_worker.fetch','persistent_training_worker.cold','backend_jobs.get_object']
    if set(w)!=fields or w['version']!=RESTORE_WITNESS or w['exception']!='requests.exceptions.ConnectionError'or w['cause']!='urllib3.exceptions.ReadTimeoutError'or w['failed_stage']!='parent-state-fetch-before-train_epoch'or w['callchain']!=required_frames:raise ValueError('exact actual parent read-timeout callchain')
    if any(w[k]is not True for k in ('model_loaded','cuda_allocated','original_processes_absent','physical_gpu_idle','output_checkpoint_absent','original_report_absent','optimizer_state_candidate_absent','update_ledger_absent'))or any(w[k]is not False for k in ('optimizer_step_reached','restore_state_returned','train_epoch_reached')):raise ValueError('restore recovery requires definitive zero-update witness')
    binding=old['trainer_state_binding'];publication=signed(original['persistent_training']['parent_publication'],authority);parent=binding.get('parent')
    from .persistent_training_protocol import validate_parent
    descriptor=validate_parent(original['persistent_training']['parent_publication'],binding,authority)
    if parent is None or type(w['public_optimizer_steps'])is not int or w['public_optimizer_steps']!=binding['global_step_before']or descriptor['optimizer_steps']!=binding['global_step_before']or descriptor['inference_checkpoint']!=old['checkpoint']['id']or sha(descriptor)!=parent['descriptor_sha256']or publication['descriptor_sha256']!=sha(descriptor)or w['parent_publication_sha256']!=sha(original['persistent_training']['parent_publication'])or w['parent_descriptor_sha256']!=sha(descriptor):raise ValueError('restore recovery exact actual committed parent')
    if type(w['parent_shard_count'])is not int or w['parent_shard_count']!=len(descriptor['shards'])or type(w['parent_total_bytes'])is not int or w['parent_total_bytes']!=sum(s['size']for s in descriptor['shards'])or type(w['selected_input_count'])is not int or not 1<=w['selected_input_count']<=256 or w['selected_input_count']!=len(original['submissions']):raise ValueError('full original parent and frozen input inventory')
    expected={('subnet/'+m+'.py'):original['source_files'].get('subnet/'+m+'.py')for m in SCIENCE if m!='sampling_contract' or 'subnet/sampling_contract.py'in original['source_files']}
    if w['science_source_files']!=expected or any(re.fullmatch('[0-9a-f]{64}',h or '')is None for h in expected.values()):raise ValueError('original pre-update callgraph/math source pins')
    for k in ('worker_log_sha256','evidence_sha256'):
        if re.fullmatch('[0-9a-f]{64}',w[k]or '')is None:raise ValueError('exact preserved original restore evidence hashes')
    if value['original_input_source_sha256']!=old['source_bundle']['sha256']or value['replacement_execution_source_sha256']!=manifest['source_bundle']['sha256']or value['authorized_input_inventory_sha256']!=sha(input_inventory(original['submissions'])):raise ValueError('explicit old input/new execution source authorization')
    authorized=value['authorized_input_objects']
    if not isinstance(authorized,list)or len(authorized)!=len(original['submissions']):raise ValueError('explicit original frozen object locations')
    for a,o in zip(authorized,original['submissions']):
        admission=signed(o['learner_admission'],authority)
        if set(a)!={'key','sha256','size','learner_admission_sha256'}or a['sha256']!=o['sha256']or a['size']!=o['size']or a['learner_admission_sha256']!=sha(o['learner_admission'])or a['key']!=('public/'+old['epoch']+'/submissions/'+admission['miner_identity']+'/'+admission['commitment_sha256']+'/training/'+str(admission['slot'])+'.json'):raise ValueError('original committed input object key/hash/size')
    times=[original['created_at'],terminal['started_at'],terminal['finished_at'],w['observed_at'],value['created_at'],job['created_at'],value['expires_at']]
    if any(type(t)not in(int,float)or not math.isfinite(t)for t in times)or not times[0]<=times[1]<=times[2]<=times[3]<=times[4]<=times[5]<times[6]or not 0<times[6]-times[4]<=86400:raise ValueError('bounded original/new restore authorization chronology')
    if type(job.get('expires_at'))not in(int,float)or not math.isfinite(job['expires_at'])or not job['created_at']<job['expires_at']<=value['expires_at']:raise ValueError('bounded fresh restore attempt lifetime')
    label=value['replacement_job_label']
    if not isinstance(label,str)or re.fullmatch('[A-Za-z0-9_-]{1,100}',label)is None or not job['job_id'].startswith(label+'-')or job['job_id']==original['job_id']:raise ValueError('distinct one controlled restore attempt identity')
    if job.get('role')!='train'or job.get('training_policy')!=original['training_policy']or job.get('training_input_policy')!=original['training_input_policy']or type(job.get('steps'))is not int or job['steps']!=original['steps']or input_inventory(job['submissions'])!=input_inventory(original['submissions'])or job['runtime_versions']!=original['runtime_versions']:raise ValueError('same original256 inputs/objective/steps/runtime')
    source=value['replacement_source_bundle']
    if source!=manifest['source_bundle']or source.get('sha256')==old['source_bundle'].get('sha256')or re.fullmatch('[0-9a-f]{64}',source.get('sha256',''))is None or 'subnet/training_startup_recovery.py'not in job['source_files']:raise ValueError('fresh complete recovery execution source pin')
    for name,h in expected.items():
        if job['source_files'].get(name)!=h:raise ValueError('restore recovery scientific implementation changed: '+name)
    if job.get('persistent_training',{}).get('output_namespace')==original['persistent_training'].get('output_namespace'):raise ValueError('new restore attempt needs separate output namespace')
    return value

def reservation_path(state,epoch,version):
    suffix='-parent-restore-bootstrap-continuation-reservation.json'if version==BOOTSTRAP_VERSION else '-startup-recovery-reservation.json'
    return Path(state)/(epoch+suffix)

def validate_bootstrap_predecessor(value,job):
    """Small ROOT precompute attestation; never embed another 5MB request."""
    p=value['predecessor']
    if set(p)!={'version','job_id','job_sha256','execution_source_sha256','reservation_sha256','declaration_sha256','input_inventory_sha256','terminal','bootstrap_witness'}or p['version']!='terminal-recovery-envelope-guard-predecessor-v1':raise ValueError('exact one bootstrap predecessor')
    for k in ('job_sha256','execution_source_sha256','reservation_sha256','declaration_sha256','input_inventory_sha256'):
        if re.fullmatch('[0-9a-f]{64}',p[k]or '')is None:raise ValueError('exact failed predecessor digest')
    if p['input_inventory_sha256']!=value['authorized_input_inventory_sha256']or p['execution_source_sha256']==value['replacement_execution_source_sha256']or not isinstance(p['job_id'],str)or re.fullmatch('[A-Za-z0-9_-]{1,160}',p['job_id'])is None or p['job_id']==job['job_id']:raise ValueError('distinct bootstrap source/job and same frozen inputs')
    terminal=p['terminal'];w=p['bootstrap_witness']
    if set(terminal)!={'phase','job_id','exit_code','runner_pid','runner_pid_ticks','child_pid','child_pid_ticks','started_at','finished_at'}or terminal['phase']!='failed'or terminal['job_id']!=p['job_id']or type(terminal['exit_code'])is not int or terminal['exit_code']!=1:raise ValueError('actual failed precompute predecessor terminal')
    for prefix in ('runner','child'):
        if type(terminal[prefix+'_pid'])is not int or terminal[prefix+'_pid']<=0 or not isinstance(terminal[prefix+'_pid_ticks'],str)or not terminal[prefix+'_pid_ticks'].isdigit():raise ValueError('predecessor physical PID/start identity')
    fields={'version','observed_at','exception','failed_stage','cuda_allocated','model_loaded','execution_started','original_processes_absent','output_namespace_empty','physical_gpu_idle','worker_log_sha256','worker_log_bytes','evidence_sha256'}
    if set(w)!=fields or w['version']!='operator-recovery-envelope-guard-witness-v1'or w['exception']!='ValueError: job envelope size budget'or w['failed_stage']!='backend_jobs.main-before-execute'or any(w[k]is not False for k in ('cuda_allocated','model_loaded','execution_started'))or any(w[k]is not True for k in ('original_processes_absent','output_namespace_empty','physical_gpu_idle')):raise ValueError('definitive original envelope-guard precompute witness')
    if type(w['worker_log_bytes'])is not int or not 0<w['worker_log_bytes']<=65536 or any(re.fullmatch('[0-9a-f]{64}',w[k]or '')is None for k in ('worker_log_sha256','evidence_sha256')):raise ValueError('preserved bootstrap log/evidence hashes')
    times=[value['restore_witness']['observed_at'],terminal['started_at'],terminal['finished_at'],w['observed_at'],value['created_at']]
    if any(type(t)not in(int,float)or not math.isfinite(t)for t in times)or not times[0]<=times[1]<=times[2]<=times[3]<=times[4]:raise ValueError('actual prior restore then bootstrap then fresh authorization')

def validate_predecessor_local(state,value,authority):
    """Authenticate unchanged v2 journal before the ONE sibling v3 selection."""
    state=Path(state);p=value['predecessor'];path=reservation_path(state,value['epoch'],RESTORE_VERSION);raw=path.read_bytes();journal=json.loads(raw)
    if hashlib.sha256(raw).hexdigest()!=p['reservation_sha256']or set(journal)!={'declaration_sha256','declaration','label','original_job_sha256'}or sha(journal['declaration'])!=journal['declaration_sha256']or journal['declaration_sha256']!=p['declaration_sha256']:raise ValueError('immutable original v2 recovery journal')
    old=signed(journal['declaration'],authority)
    if old.get('version')!=RESTORE_VERSION or old['original_signed_job']!=value['original_signed_job']or old['original_job_sha256']!=value['original_job_sha256']or old['restore_witness']!=value['restore_witness']or old['original_terminal']!=value['original_terminal']or journal['label']!=old['replacement_job_label']or journal['original_job_sha256']!=value['original_job_sha256']:raise ValueError('one v3 sibling of exactly original v2 and original4db failure')
    record=json.loads((state/'roles'/(journal['label']+'.json')).read_bytes());job=signed(json.loads((state/'roles'/(record['job_id']+'-job.json')).read_bytes()),authority);manifest=signed(job['manifest'],authority)
    if record['job_id']!=p['job_id']or record['job_sha256']!=p['job_sha256']or sha(job)!=p['job_sha256']or manifest.get(FIELD)!=journal['declaration']or manifest['source_bundle']['sha256']!=p['execution_source_sha256']or sha(input_inventory(job['submissions']))!=p['input_inventory_sha256']:raise ValueError('actual immutable failed v2 signed job/source/inputs')
    validate(job,manifest,authority)
    failure=json.loads((state/'roles'/(job['job_id']+'-failure.json')).read_bytes())
    if any(failure.get(k)!=v for k,v in p['terminal'].items())or (state/'roles'/(job['job_id']+'-report.json')).exists():raise ValueError('actual failed v2 predecessor and no completed report')
    return job
