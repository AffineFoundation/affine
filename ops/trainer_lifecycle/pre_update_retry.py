"""One ROOT-authorized pre-update retry; original failure and public inputs persist."""
import json,time,hashlib,shlex
from pathlib import Path
VERSION='root-approved-pre-update-memory-retry-v1'
RETRY='fresh-genesis-pre-update-resource-retry-v1'
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def validate_scope(p,guards):
    row=p['pre_update_memory_retry']
    for k in ('authorization','original_job','original_status','worker_log'):
        if guards.file_hash(row[k]['path'])!=row[k]['sha256']:raise ValueError('immutable pre-update retry '+k)
    a=guards.signed(guards.read(row['authorization']['path']));original=guards.signed(guards.read(row['original_job']['path']));status=guards.read(row['original_status']['path']);manifest=guards.signed(original['manifest'])
    if(a['version']!=VERSION or a['epoch']!=manifest['epoch']or a['original_job_id']!=original['job_id']
       or a['original_job_sha256']!=digest(original)or a['original_status_sha256']!=digest(status)
       or a['worker_log_sha256']!=row['worker_log']['sha256']or status.get('job_id')!=original['job_id']
       or status.get('phase')!='failed'or status.get('exit_code')!=1 or status.get('actual_wait')is not True
       or a['replacement_label']!=a['epoch']+'-train-resource-r1'
       or manifest['trainer_state_binding']['global_step_before']!=0 or manifest['trainer_state_binding']['parent']is not None
       or a['genesis_sha256']!=manifest['trainer_state_binding']['genesis_sha256']
       or a['checkpoint']!=manifest['checkpoint']['id']):raise ValueError('exact original zero-update resource retry scope')
    text=Path(row['worker_log']['path']).read_text()
    if not text.rstrip().endswith('ValueError: alternating optimizer memory and model disk budget')or 'local_cache.admit(plan'not in text or 'train_epoch('in text:
        raise ValueError('exact original pre-optimizer resource-admission traceback')
    return row,a,original,status

def install(p,guards):
    row,authorization,original,status=validate_scope(p,guards)
    from subnet import training_startup_recovery as recovery,role_router
    from subnet.remote_backend import save
    previous_label=recovery.label
    def label(controller,epoch):
        if epoch!=authorization['epoch']:return previous_label(controller,epoch)
        if controller.authority.id!=p['authority']:raise ValueError('retry ROOT authority')
        return authorization['replacement_label']
    recovery.label=label
    previous_prepare=role_router.RoutedJobs.prepare_training_dispatch
    def prepare(self,label,manifest,submissions,steps):
        if manifest['epoch']!=authorization['epoch']:return previous_prepare(self,label,manifest,submissions,steps)
        if label!=authorization['replacement_label']:raise ValueError('exact recovery label')
        original_manifest=guards.signed(original['manifest'])
        if manifest!=original_manifest or submissions!=original['submissions']or steps!=original['steps']:
            raise ValueError('pre-update retry preserves all native-selected inputs and optimization')
        result=previous_prepare(self,label,manifest,submissions,steps)
        new_envelope=guards.read(result['job_path']);new=guards.signed(new_envelope)
        immutable=set(original)-{'job_id','created_at','expires_at','persistent_training','unaudited_training_execution','learner_selection_operator_admission'}
        if set(new)!=set(original)or any(new[k]!=original[k]for k in immutable)or new['job_id']==original['job_id']:
            raise ValueError('only separately signed attempt bindings may change')
        reset_path=Path(p['prepared_reset_dispatch']['local_root'])/'reset.ROOT-SIGNED.private.json'
        target=Path(row['retry_grant']);remote=row['remote_retry_grant']
        if target.exists():
            envelope=guards.read(target);grant=guards.signed(envelope)
            if grant['new_job']!=new_envelope or grant['original_job']!=guards.read(row['original_job']['path'])or grant['reset_envelope_sha256']!=digest(guards.read(reset_path)):
                raise ValueError('immutable original linked retry grant')
        else:
            now=time.time()
            if not authorization['created_at']<=now<authorization['expires_at']or result.get('already_issued'):
                raise ValueError('new authorized retry must be prepared and timely')
            grant=dict(version=RETRY,original_job=guards.read(row['original_job']['path']),original_status=status,
                original_worker_log_sha256=row['worker_log']['sha256'],new_job=new_envelope,
                reset_envelope_sha256=digest(guards.read(reset_path)),created_at=now,expires_at=min(new['expires_at'],authorization['expires_at']))
            envelope=self.controller.signed(grant);save(target,envelope)
        trainer=self.roles['train']._approved_training_execution_client
        trainer.copy_to(target,remote)
        save(Path(row['receipt']),dict(version=VERSION,original_job_id=original['job_id'],retry_job_id=new['job_id'],retry_grant_sha256=digest(envelope),same_input_manifest_sha256=digest(original['manifest']),same_native_inputs=True,new_optimizer_genesis=False,repeat_reset=False))
        return result
    role_router.RoutedJobs.prepare_training_dispatch=prepare
