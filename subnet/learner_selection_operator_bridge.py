"""Explicit CPU-selection peer admission; frozen scientific source stays separate.

Only ROOT-approved operator bytes may interpret enriched selection bindings.
No forward, sampler, grader, optimizer, job dispatch, storage mutation or signing
key reads. Signing is an injected existing-coordinator operation.
"""
import copy, hashlib, json, os, stat
from pathlib import Path
from .training_receipts import authenticate, sha
from .learner_blacklist_selection import FIELD, admit, partition

FIELD_ADMISSION='learner_selection_operator_admission'
VERSION='cpu-selection-peer-admission-v1'
AUTH_VERSION='cpu-selection-peer-authorization-v1'
K2L2_AUTH_VERSION='cpu-selection-peer-miner-bound-authorization-v2'
FILES={'subnet/learner_selection_operator_bridge.py','subnet/learner_blacklist_selection.py',
       'subnet/committed_training_inputs.py','subnet/training_receipts.py'}

def core(job):
    return {k:copy.deepcopy(v)for k,v in job.items()if k!=FIELD_ADMISSION}

def approval(document,authority,manifest):
    p=authenticate(document,authority)
    fields={'version','source_sha256','scientific_source_files','operator_files','minimum_round','epoch_prefix','peer_entry_sha256','peer_runner_sha256','backend_execution_allowed'}
    miner_bound=manifest.get('sampling_contract',{}).get('version')=='forced-inverse-cdf-prefill-miner-bound-v5'
    expected_version=K2L2_AUTH_VERSION if miner_bound else AUTH_VERSION
    local_trainer=p.get('version')=='cpu-selection-peer-local-trainer-authorization-v3'
    expected_members=179 if miner_bound else 177
    if miner_bound:
        if 'subnet/batch_quotas.py' in p.get('scientific_source_files',{}):
            from .batch_quotas import configured_quotas
            configured_quotas(manifest)
            from .controller import class_quotas
            class_quotas(manifest.get('K'),manifest.get('L'),manifest.get('sampling_contract'))
            expected_members=180
            if type(manifest.get('max_batches'))is not int or manifest['max_batches']!=3:raise ValueError('miner-bound peer max3')
        elif (type(manifest.get('K'))is not int or type(manifest.get('L'))is not int or manifest['K']!=2 or manifest['L']!=2 or type(manifest.get('max_batches'))is not int or manifest['max_batches']!=3):raise ValueError('historical miner-bound peer K2 L2 max3')
        if not {'subnet/sampling_uniqueness.py','subnet/trajectory_identity.py'}<=set(p.get('scientific_source_files',{})):raise ValueError('miner-bound peer runtime additions')
    if local_trainer:
        if (not miner_bound or manifest.get('optimizer_state_export_policy')!='trainer-local-only-v1' or
            'subnet/trainer_local_state.py'not in p.get('scientific_source_files',{})):
            raise ValueError('local trainer peer requires explicit local state and source')
        expected_version='cpu-selection-peer-local-trainer-authorization-v3';expected_members=181
    if (set(p)!=fields or p['version']!=expected_version or p['source_sha256']!=manifest['source_bundle']['sha256']or
        type(p['scientific_source_files'])is not dict or len(p['scientific_source_files'])!=expected_members or
        set(p['operator_files'])!=FILES or type(p['minimum_round'])is not int or p['minimum_round']<0 or
        type(p['epoch_prefix'])is not str or not p['epoch_prefix']or not manifest['epoch'].startswith(p['epoch_prefix'])):
        raise ValueError('exact ROOT CPU peer authorization/scientific source')
    from .training_receipts import digest
    digest(p['peer_entry_sha256']);digest(p['peer_runner_sha256'])
    if type(p['backend_execution_allowed'])is not bool:raise ValueError('explicit backend execution boolean')
    for files in (p['scientific_source_files'],p['operator_files']):
        for name,h in files.items():
            if not name.startswith('subnet/')or '\\'in name or any(part in ('','..','.')for part in name.split('/')):raise ValueError('relative CPU/scientific file name')
            digest(h)
    return p

def make_admission(job,manifest,authorization_document,authority,sign):
    """Pure prepare; caller persists SAME signed job before its first dispatch."""
    if FIELD not in manifest:
        if authorization_document is not None:raise ValueError('CPU admission without enabled selection')
        return job
    if FIELD_ADMISSION in job:raise ValueError('original CPU admission must not be reissued')
    p=approval(authorization_document,authority,manifest)
    _context(job,manifest,p,authority)
    from .training_receipts import computation_binding
    from .committed_training_inputs import receipt_inventory
    payload=dict(version=VERSION,authorization_document=authorization_document,
        job_core_sha256=sha(core(job)),manifest_sha256=sha(manifest),
        computation_binding_sha256=sha(computation_binding(manifest)),
        selection_snapshot_sha256=sha(manifest['learner_blacklist_selection_snapshot']),
        selected_inventory_sha256=sha(receipt_inventory(job['submissions'])),
        round=manifest['learner_blacklist_selection_round'],
        scientific_source_files_sha256=sha(job['source_files']),operator_files_sha256=sha(p['operator_files']))
    return dict(job,**{FIELD_ADMISSION:sign(payload)})

def _context(job,manifest,p,authority):
    if (job.get('role')!='train'or job.get('source_files')!=p['scientific_source_files']or
        authenticate(job['manifest'],authority)!=manifest or
        type(manifest.get('learner_blacklist_selection_round'))is not int or
        manifest['learner_blacklist_selection_round']<p['minimum_round']):
        raise ValueError('original signed manifest/round/remote177 job binding')
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
    fields={'version','authorization_document','job_core_sha256','manifest_sha256',
        'computation_binding_sha256','selection_snapshot_sha256','selected_inventory_sha256',
        'round','scientific_source_files_sha256','operator_files_sha256'}
    if set(a)!=fields or a['version']!=VERSION:raise ValueError('exact CPU peer admission')
    p=approval(a['authorization_document'],authority,manifest)
    _context(job,manifest,p,authority)
    from .training_receipts import computation_binding
    from .committed_training_inputs import receipt_inventory
    expected=dict(job_core_sha256=sha(core(job)),manifest_sha256=sha(manifest),
        computation_binding_sha256=sha(computation_binding(manifest)),
        selection_snapshot_sha256=sha(manifest['learner_blacklist_selection_snapshot']),
        selected_inventory_sha256=sha(receipt_inventory(job['submissions'])),
        round=manifest['learner_blacklist_selection_round'],scientific_source_files_sha256=sha(job['source_files']),
        operator_files_sha256=sha(p['operator_files']))
    if any(a[k]!=v for k,v in expected.items()):raise ValueError('enriched original CPU/job admission binding')
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
    if FIELD not in manifest:
        if FIELD_ADMISSION in job:raise ValueError('unexpected historical CPU admission')
        return None
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
        scientific_source_unchanged=True,model_loaded=False,proof_reverification=False,
        scientific_operation_started=False)
