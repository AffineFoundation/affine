"""Explicit CPU-selection peer admission; frozen scientific source stays separate.

Only ROOT-approved operator bytes may interpret enriched selection bindings.
No forward, sampler, grader, optimizer, job dispatch, storage mutation or signing
key reads. Signing is an injected existing-coordinator operation.
"""
import copy, hashlib, json, os, stat, math, re
from pathlib import Path
from .training_receipts import authenticate, sha
from .learner_blacklist_selection import FIELD, admit, partition

FIELD_ADMISSION='learner_selection_operator_admission'
VERSION='cpu-selection-peer-admission-v1'
AUTH_VERSION='cpu-selection-peer-authorization-v1'
K2L2_AUTH_VERSION='cpu-selection-peer-miner-bound-authorization-v2'
FP32_AUTH_VERSION='cpu-selection-peer-fp32-training-execution-authorization-v5'
LR_AUTH_VERSION='cpu-selection-peer-effective-lr-training-execution-authorization-v6'
GENESIS_LR_AUTH_VERSION='cpu-selection-peer-effective-lr-genesis-training-execution-authorization-v7'
OBJECTIVE_AUTH_VERSION='cpu-selection-peer-objective-horizon-training-execution-authorization-v9'
BASIC_AUTH_VERSION='cpu-selection-peer-basic-eligibility-objective-horizon-authorization-v10'
BASIC_THREAD_AUTH_VERSION='cpu-selection-peer-basic-eligibility-optimizer-cpu16-authorization-v11'
BASIC_VERSIONS=(BASIC_AUTH_VERSION,BASIC_THREAD_AUTH_VERSION)
BASIC_VERSION='cpu-basic-eligibility-peer-admission-v2'
BASIC_ASSERTIONS=dict(eligibility='basic-committed-inputs-v1',assurance='unaudited',reward_assessment_required=False)
FILES={'subnet/learner_selection_operator_bridge.py','subnet/learner_blacklist_selection.py',
       'subnet/committed_training_inputs.py','subnet/training_receipts.py'}

def core(job):
    return {k:copy.deepcopy(v)for k,v in job.items()if k!=FIELD_ADMISSION}

def approval(document,authority,manifest):
    p=authenticate(document,authority)
    fields={'version','source_sha256','scientific_source_files','operator_files','minimum_round','epoch_prefix','peer_entry_sha256','peer_runner_sha256','backend_execution_allowed'}
    miner_bound=manifest.get('sampling_contract',{}).get('version')=='forced-inverse-cdf-prefill-miner-bound-v5'
    expected_version=K2L2_AUTH_VERSION if miner_bound else AUTH_VERSION
    fp32=p.get('version') in (FP32_AUTH_VERSION, LR_AUTH_VERSION, GENESIS_LR_AUTH_VERSION, OBJECTIVE_AUTH_VERSION, BASIC_AUTH_VERSION, BASIC_THREAD_AUTH_VERSION)
    effective_lr=p.get('version') in (LR_AUTH_VERSION, GENESIS_LR_AUTH_VERSION, OBJECTIVE_AUTH_VERSION, BASIC_AUTH_VERSION, BASIC_THREAD_AUTH_VERSION)
    completed=p.get('version')=='cpu-selection-peer-completed-math-local-trainer-authorization-v4' or fp32
    local_trainer=p.get('version')=='cpu-selection-peer-local-trainer-authorization-v3' or completed
    expected_members=179 if miner_bound else 177
    if miner_bound:
        if 'subnet/batch_quotas.py' in p.get('scientific_source_files',{}):
            from .batch_quotas import configured_quotas
            configured_quotas(manifest)
            from .controller import class_quotas
            class_quotas(manifest.get('K'),manifest.get('L'),manifest.get('sampling_contract'))
            expected_members=180
            if type(manifest.get('max_batches'))is not int or not 1<=manifest['max_batches']<=256:raise ValueError('miner-bound peer signed batch cap')
        elif (type(manifest.get('K'))is not int or type(manifest.get('L'))is not int or manifest['K']!=2 or manifest['L']!=2 or type(manifest.get('max_batches'))is not int or manifest['max_batches']!=3):raise ValueError('historical miner-bound peer K2 L2 max3')
        if not {'subnet/sampling_uniqueness.py','subnet/trajectory_identity.py'}<=set(p.get('scientific_source_files',{})):raise ValueError('miner-bound peer runtime additions')
    if local_trainer:
        if (not miner_bound or manifest.get('optimizer_state_export_policy')!='trainer-local-only-v1' or
            'subnet/trainer_local_state.py'not in p.get('scientific_source_files',{})):
            raise ValueError('local trainer peer requires explicit local state and source')
        expected_version='cpu-selection-peer-local-trainer-authorization-v3';expected_members=181
        if completed:
            from .math_completion import FIELD, VERSION, enabled
            rows=manifest.get('environments',[])
            if len(rows)!=1 or rows[0]['spec']['config'].get(FIELD)!=VERSION:raise ValueError('explicit completed-math task contract')
            if not enabled(rows[0]['spec']):raise ValueError('completed-math marker required')
            if 'subnet/math_completion.py'not in p['scientific_source_files']:raise ValueError('completed-math helper closure')
            expected_version='cpu-selection-peer-completed-math-local-trainer-authorization-v4';expected_members=182
            if fp32:
                if not {'subnet/fp32_gradient_accumulation.py','subnet/unaudited_training_execution.py'}<=set(p['scientific_source_files']):raise ValueError('explicit FP32 execution dependency closure')
                expected_version=FP32_AUTH_VERSION;expected_members=184
                if effective_lr:
                    if 'subnet/learning_rate_transition.py'not in p['scientific_source_files']:
                        raise ValueError('explicit effective-LR state dependency closure')
                    expected_version=(p['version'] if p.get('version') in BASIC_VERSIONS else OBJECTIVE_AUTH_VERSION if p.get('version')==OBJECTIVE_AUTH_VERSION else GENESIS_LR_AUTH_VERSION if p.get('version')==GENESIS_LR_AUTH_VERSION else LR_AUTH_VERSION);expected_members=185
    threaded=p.get('version')==BASIC_THREAD_AUTH_VERSION
    if threaded:
        if 'subnet/optimizer_threading.py' not in p.get('scientific_source_files',{}):
            raise ValueError('explicit optimizer-only threading dependency closure')
        expected_members+=1
    elif 'subnet/optimizer_threading.py' in p.get('scientific_source_files',{}):
        raise ValueError('historical CPU peer cannot opt into optimizer threading')
    expected_operator_files=FILES
    if 'subnet/training_task_representatives.py' in p.get('scientific_source_files',{}):
        from .training_task_representatives import _policy
        if not effective_lr or _policy(manifest) is None:raise ValueError('explicit representative peer policy/source')
        expected_members+=1
        expected_operator_files=FILES|{'subnet/training_task_representatives.py'}
        if p.get('operator_files',{}).get('subnet/training_task_representatives.py')!=p['scientific_source_files']['subnet/training_task_representatives.py']:
            raise ValueError('same authenticated representative parser before scientific namespace')
    elif 'training_representative_policy' in manifest:
        raise ValueError('representative manifest requires qualified representative source')
    if (set(p)!=fields or p['version']!=expected_version or p['source_sha256']!=manifest['source_bundle']['sha256']or
        type(p['scientific_source_files'])is not dict or len(p['scientific_source_files'])!=expected_members or
        set(p['operator_files'])!=expected_operator_files or type(p['minimum_round'])is not int or p['minimum_round']<0 or
        type(p['epoch_prefix'])is not str or not p['epoch_prefix']or not manifest['epoch'].startswith(p['epoch_prefix'])):
        raise ValueError('exact ROOT CPU peer authorization/scientific source')
    if p['version'] in BASIC_VERSIONS:
        if p['minimum_round']<136 or expected_members!=(187 if threaded else 186):
            raise ValueError('independent basic eligibility requires round136 and exact representative/threading source')
        _basic_round(manifest,p)
    from .training_receipts import digest
    digest(p['peer_entry_sha256']);digest(p['peer_runner_sha256'])
    if type(p['backend_execution_allowed'])is not bool:raise ValueError('explicit backend execution boolean')
    for files in (p['scientific_source_files'],p['operator_files']):
        for name,h in files.items():
            if not name.startswith('subnet/')or '\\'in name or any(part in ('','..','.')for part in name.split('/')):raise ValueError('relative CPU/scientific file name')
            digest(h)
    return p

def _basic_round(manifest,p):
    if FIELD in manifest or 'learner_blacklist_selection_snapshot' in manifest:
        raise ValueError('independent basic eligibility requires no selection assessment policy/snapshot')
    epoch=manifest.get('epoch')
    match=re.fullmatch(re.escape(p['epoch_prefix'])+r'([1-9][0-9]*)-(0|[1-9][0-9]*)',epoch) if isinstance(epoch,str) else None
    if match is None:raise ValueError('exact independent epoch timestamp-round suffix')
    value=int(match.group(2))
    if value<p['minimum_round']:raise ValueError('independent basic eligibility minimum round')
    if 'learner_blacklist_selection_round' in manifest and (
            type(manifest['learner_blacklist_selection_round']) is not int or
            manifest['learner_blacklist_selection_round']!=value):
        raise ValueError('independent explicit round must match exact epoch suffix')
    return value


def _admission_fields(job,manifest,p):
    from .training_receipts import computation_binding
    from .committed_training_inputs import receipt_inventory
    expected=dict(job_core_sha256=sha(core(job)),manifest_sha256=sha(manifest),
        computation_binding_sha256=sha(computation_binding(manifest)),
        selected_inventory_sha256=sha(receipt_inventory(job['submissions'])),
        scientific_source_files_sha256=sha(job['source_files']),operator_files_sha256=sha(p['operator_files']))
    if p['version'] in BASIC_VERSIONS:
        expected.update(BASIC_ASSERTIONS,round=_basic_round(manifest,p))
    else:
        expected.update(selection_snapshot_sha256=sha(manifest['learner_blacklist_selection_snapshot']),
            round=manifest['learner_blacklist_selection_round'])
    return expected


def make_admission(job,manifest,authorization_document,authority,sign):
    """Pure prepare; caller persists SAME signed job before its first dispatch."""
    if FIELD not in manifest and authorization_document is None:return job
    if FIELD_ADMISSION in job:raise ValueError('original CPU admission must not be reissued')
    p=approval(authorization_document,authority,manifest)
    _context(job,manifest,p,authority)
    payload=dict(version=BASIC_VERSION if p['version'] in BASIC_VERSIONS else VERSION,
        authorization_document=authorization_document,**_admission_fields(job,manifest,p))
    return dict(job,**{FIELD_ADMISSION:sign(payload)})

def validate_execution_scope(job,manifest,p,authority):
    """Alias-namespace gate only; candidate backend validates full qualification.

    Avoid importing execution helpers before the peer installs the fresh subnet
    namespace. The independently signed declaration must nevertheless bind the
    exact original input manifest, selected records, parent, and execution map.
    """
    if p['version']not in (FP32_AUTH_VERSION, LR_AUTH_VERSION, GENESIS_LR_AUTH_VERSION, OBJECTIVE_AUTH_VERSION, BASIC_AUTH_VERSION, BASIC_THREAD_AUTH_VERSION):
        if 'unaudited_training_execution' in job:raise ValueError('execution amendment requires explicit FP32 peer')
        return None
    value=authenticate(job.get('unaudited_training_execution'),authority)
    from .committed_training_inputs import receipt_inventory
    binding=manifest.get('trainer_state_binding',{});parent=binding.get('parent')or{}
    effective_lr=p['version'] in (LR_AUTH_VERSION, GENESIS_LR_AUTH_VERSION, OBJECTIVE_AUTH_VERSION, BASIC_AUTH_VERSION, BASIC_THREAD_AUTH_VERSION)
    initial=p['version']==GENESIS_LR_AUTH_VERSION and binding.get('genesis') is not None
    objective=p['version'] in (OBJECTIVE_AUTH_VERSION,BASIC_AUTH_VERSION,BASIC_THREAD_AUTH_VERSION)
    threaded=p['version']==BASIC_THREAD_AUTH_VERSION
    version=('unaudited-training-execution-amendment-v6-optimizer-cpu16' if threaded else 'unaudited-training-execution-amendment-v5-objective-horizon' if objective else 'unaudited-training-execution-amendment-v3-effective-lr-genesis' if initial else
        'unaudited-training-execution-amendment-v2-effective-lr' if effective_lr else 'unaudited-training-execution-amendment-v1')
    method=('fp32-task-gradient-effective-lr-objective-horizon-v1' if objective else 'fp32-task-gradient-effective-lr-genesis-v1' if initial else
        'fp32-task-gradient-effective-lr-v1' if effective_lr else 'fp32-task-gradient-accumulation-v1')
    expected=dict(version=version,method=method,job_id=job.get('job_id'),epoch=manifest['epoch'],
        original_signed_manifest_sha256=sha(job['manifest']),
        original_source_bundle_sha256=manifest['source_bundle']['sha256'],
        execution_source_files=p['scientific_source_files'],runtime_versions=job.get('runtime_versions'),
        training_policy=job.get('training_policy'),training_input_policy=job.get('training_input_policy'),
        steps=job.get('steps'),input_inventory_sha256=sha(receipt_inventory(job['submissions'])),
        native_eligibility_receipt=manifest.get('native_training_eligibility_receipt'),
        trainer_binding_sha256=sha(binding),parent_descriptor_sha256=parent.get('descriptor_sha256'),
        genesis_sha256=binding.get('genesis_sha256'),optimizer_step_before=binding.get('global_step_before'))
    if any(value.get(k)!=v for k,v in expected.items()):
        raise ValueError('signed FP32 peer exact input/execution/parent scope')
    if threaded:
        policy=dict(version='optimizer-only-cpu-threads-v1',steady_intraop=2,optimizer_intraop=16,interop=2)
        actual=value.get('optimizer_threading')
        if (not isinstance(actual,dict) or set(actual)!=set(policy)
                or any(actual[k]!=v or type(actual[k])is not type(v)for k,v in policy.items())):
            raise ValueError('exact explicitly signed optimizer-only CPU16 policy')
    elif 'optimizer_threading' in value:
        raise ValueError('historical CPU peer cannot request optimizer threading')
    if objective:
        config=value.get('training_objective')
        # This peer namespace must not import unadmitted scientific code.
        if (not isinstance(config,dict) or set(config)!={'version','positive_nll_weight'}
                or config['version']!='sequence-mean-preference-positive-nll-v1'
                or type(config['positive_nll_weight'])not in(int,float)
                or not math.isfinite(config['positive_nll_weight']) or config['positive_nll_weight'] not in (0,1)):
            raise ValueError('explicit objective peer configuration')
        horizon=value.get('training_horizon')
        fields={'version','first_optimizer_step','updates','initial_checkpoint',
                'initial_parent_descriptor_sha256','coefficient_during','coefficient_after'}
        step=binding.get('global_step_before')
        if (not isinstance(horizon,dict) or set(horizon)!=fields
                or horizon['version']!='retained-adam-unit-positive-nll-16-v1'
                or type(horizon['first_optimizer_step']) is not int or horizon['first_optimizer_step']<1
                or type(horizon['updates']) is not int or horizon['updates']!=16
                or type(horizon['coefficient_during']) not in (int,float) or horizon['coefficient_during']!=1
                or type(horizon['coefficient_after']) not in (int,float) or horizon['coefficient_after']!=0
                or type(step) is not int or step<horizon['first_optimizer_step']
                or type(value.get('steps')) is not int or value['steps']!=1
                or value.get('effective_learning_rate')!=5e-7
                or config['positive_nll_weight']!=(1 if step<horizon['first_optimizer_step']+16 else 0)):
            raise ValueError('exact peer sixteen-update objective horizon')
        from .training_receipts import digest
        digest(horizon['initial_checkpoint']);digest(horizon['initial_parent_descriptor_sha256'])
        if step==horizon['first_optimizer_step'] and (
                parent.get('descriptor_sha256')!=horizon['initial_parent_descriptor_sha256']
                or binding.get('input_checkpoint')!=horizon['initial_checkpoint']):
            raise ValueError('objective peer exact original retained checkpoint and descriptor')
    if initial:
        if parent or type(binding.get('global_step_before'))is not int or binding['global_step_before']!=0:
            raise ValueError('explicit initial LR peer requires no parent and counter zero')
    elif not parent or binding.get('genesis')is not None:
        raise ValueError('signed FP32 peer exact existing parent scope')
    if effective_lr:
        grant=authenticate(value.get('learning_rate_authorization'),authority)
        fields={'version','epoch','job_id','input_checkpoint','parent_descriptor_sha256',
            'genesis_sha256','optimizer_step_before','steps','parameters_sha256',
            'base_hyperparameters_sha256','effective_learning_rate','created_at','expires_at',
            'execution_release_sha256'}
        if initial:fields.add('run_id')
        expected_lr=dict(version='persistent-adamw-effective-learning-rate-genesis-v1' if initial else 'persistent-adamw-effective-learning-rate-v1',
            epoch=manifest['epoch'],job_id=job.get('job_id'),input_checkpoint=manifest['checkpoint']['id'],
            parent_descriptor_sha256=parent.get('descriptor_sha256'),genesis_sha256=binding.get('genesis_sha256'),
            optimizer_step_before=binding.get('global_step_before'),steps=job.get('steps'),
            parameters_sha256=binding.get('parameters_sha256'),base_hyperparameters_sha256=sha(binding.get('hyperparameters')),
            execution_release_sha256=value.get('execution_release_sha256'))
        if set(grant)!=fields or any(grant.get(k)!=v for k,v in expected_lr.items()):
            raise ValueError('signed effective-LR peer job/parent/release scope')
        if type(grant['optimizer_step_before'])is not int or type(grant['steps'])is not int:
            raise ValueError('signed effective-LR peer counters must be integers')
        rate=grant['effective_learning_rate'];created=grant['created_at'];expires=grant['expires_at']
        if (type(rate)not in(int,float)or not math.isfinite(rate)or not 0<rate<=binding['hyperparameters']['lr']or
            any(type(x)not in(int,float)or not math.isfinite(x)for x in(created,expires))or not 0<=created<expires):
            raise ValueError('finite bounded effective-LR peer authorization')
        from .training_receipts import digest
        digest(grant['execution_release_sha256'])
        if initial:
            digest(grant['run_id'])
            document=dict(version='explicit-fp32-master-genesis-v2-effective-lr',policy=binding['policy'],
                hyperparameters=binding['hyperparameters'],parameters_sha256=binding['parameters_sha256'],
                input_checkpoint=manifest['checkpoint']['id'],explicit_optimizer_genesis=True,
                run_id=grant['run_id'],initial_effective_learning_rate=rate)
            if binding['genesis']!=document or sha(document)!=binding['genesis_sha256']:
                raise ValueError('explicit LR peer unique run/rate/genesis identity')
        elif p['version']==GENESIS_LR_AUTH_VERSION and grant['optimizer_step_before']<1:
            raise ValueError('LR peer continuation must preserve an existing state')
    return value


def _context(job,manifest,p,authority):
    round_number=(_basic_round(manifest,p) if p['version'] in BASIC_VERSIONS else manifest.get('learner_blacklist_selection_round'))
    if (job.get('role')!='train'or job.get('source_files')!=p['scientific_source_files']or
        authenticate(job['manifest'],authority)!=manifest or
        type(round_number)is not int or round_number<p['minimum_round']):
        raise ValueError('original signed manifest/round/remote177 job binding')
    validate_execution_scope(job,manifest,p,authority)
    if p['version'] in BASIC_VERSIONS:
        from .committed_training_inputs import validate_job
        validate_job(job,manifest,authority)
        return
    # Explicit CPU interpretation uses enriched fields, never the old projection.
    from .committed_training_inputs import receipt_inventory
    coverage=manifest.get('training_coverage',{})
    status=admit(manifest[FIELD],manifest,authority,at=coverage.get('captured_at'),
                 round_number=manifest['learner_blacklist_selection_round'])
    snapshot=manifest.get('learner_blacklist_selection_snapshot',{})
    if any(snapshot.get(k)!=v for k,v in status.items()):raise ValueError('original signed selection snapshot')
    kept,_=partition(job['submissions'],manifest,authority,at=coverage['captured_at'],
                      round_number=manifest['learner_blacklist_selection_round'])
    if kept!=job['submissions']or coverage['inventory_sha256']!=sha(receipt_inventory(kept)):
        raise ValueError('exact selected nonblacklisted receipt inventory')


def validate_metadata(job,manifest,authority):
    envelope=job.get(FIELD_ADMISSION)
    if envelope is None:raise ValueError('explicit CPU peer admission required')
    a=authenticate(envelope,authority)
    p=approval(a.get('authorization_document'),authority,manifest)
    _context(job,manifest,p,authority)
    expected=_admission_fields(job,manifest,p)
    version=BASIC_VERSION if p['version'] in BASIC_VERSIONS else VERSION
    if set(a)!=set(expected)|{'version','authorization_document'} or a['version']!=version:
        raise ValueError('exact CPU peer admission')
    if any(a[k]!=v or type(a[k]) is not type(v) for k,v in expected.items()):
        raise ValueError('enriched original CPU/job admission binding')
    return p,a


def _inventory(root,files,*,exact=False):
    root=Path(root)
    if not root.is_absolute()or root.resolve()!=root or root.is_symlink()or not root.is_dir()or root.stat().st_uid!=os.getuid():raise ValueError('ordinary owned CPU peer root')
    for name,h in files.items():
        path=root/name
        if path.is_symlink()or path.resolve()!=path:raise ValueError('no CPU peer symlink')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,'rb')as f:
            st=os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode)or st.st_uid!=os.getuid():raise ValueError('ordinary owned peer file')
            actual=hashlib.sha256(f.read()).hexdigest()
        if actual!=h:raise ValueError('full original CPU/scientific module SHA')
    if exact and {str(f.relative_to(root))for f in root.rglob('*')if f.is_file()}!=set(files):raise ValueError('exact operator peer membership')


def admit_peer(job_document,authority,*,operator_root,scientific_root):
    """CPU-only original-job gate, before invoking frozen science in another namespace."""
    job=authenticate(job_document,authority);manifest=authenticate(job['manifest'],authority)
    p,a=validate_metadata(job,manifest,authority)
    _inventory(operator_root,p['operator_files'],exact=True)
    _inventory(scientific_root,p['scientific_source_files'])
    # Ensure actual modules interpreting metadata are the separately admitted bytes.
    import sys
    package=__package__
    for name,h in p['operator_files'].items():
        module=sys.modules.get(package+'.'+Path(name).stem)
        if module is None or Path(module.__file__).resolve()!=Path(operator_root)/name:
            raise ValueError('executed CPU override namespace/path')
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=h:raise ValueError('executed CPU override SHA')
    from .committed_training_inputs import validate_job
    validate_job(job,manifest,authority)
    return dict(version='cpu-selection-peer-execution-v1',original_job_sha256=sha(job),
        original_manifest_sha256=sha(manifest),admission_sha256=sha(job[FIELD_ADMISSION]),
        enriched_computation_binding_sha256=a['computation_binding_sha256'],
        operator_files=p['operator_files'],scientific_source_files_sha256=sha(job['source_files']),
        scientific_source_unchanged=p['version']not in(FP32_AUTH_VERSION,LR_AUTH_VERSION,GENESIS_LR_AUTH_VERSION,OBJECTIVE_AUTH_VERSION,BASIC_AUTH_VERSION,BASIC_THREAD_AUTH_VERSION),model_loaded=False,proof_reverification=False,
        scientific_operation_started=False)
