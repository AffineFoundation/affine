"""Prospective job-level training routing contract; no deployment or dispatch.

Keep the original signed miner/native-input manifest and optimizer lineage.
A separately authenticated declaration selects a qualified training execution
bundle. This module does not regrade, reverify proofs, or alter inputs/Adam.
"""
import copy
import math
from subnet.training_receipts import authenticate, computation_binding, digest, sha
from subnet.committed_training_inputs import VERSION as INPUT_POLICY, receipt_inventory, training_document_cap
from subnet.persistent_cpu_adamw import POLICY
from subnet.persistent_training_protocol import validate_binding

FIELD = 'unaudited_training_execution'
VERSION = 'unaudited-training-execution-amendment-v2-effective-lr'
METHOD = 'fp32-task-gradient-effective-lr-v1'
QUALIFICATION = 'fp32-task-gradient-effective-lr-qualification-v1'
GENESIS_VERSION = 'unaudited-training-execution-amendment-v3-effective-lr-genesis'
GENESIS_METHOD = 'fp32-task-gradient-effective-lr-genesis-v1'
GENESIS_QUALIFICATION = 'fp32-task-gradient-effective-lr-genesis-qualification-v1'
GENESIS_PREPARATION = 'unaudited-training-execution-preparation-v3-effective-lr-genesis'
GENESIS_RELEASE = 'unaudited-training-execution-release-v3-effective-lr-genesis'
INTAKE_EQUIVALENCE = 'native-task-intake-source-equivalence-qualification-v1'
OBJECTIVE_INTAKE_EQUIVALENCE = 'retained-nll-native-intake-source-equivalence-qualification-v1'
OBJECTIVE_INTAKE_BASE_BUNDLE = '55206bde28d115cee9dabe1c21cfd0910b739296f0117e3b0e81f43e35e188a9'
OBJECTIVE_INTAKE_BASE_MAP = '8204f8526e6a653e0f697c60a36fa81eadb8de5fffde7c5d63b224ea3fbf6e45'
INTAKE_BASE_BUNDLE = '395bd10b007adc1332c524571505b57b72fb1eea4905c255e85d64dcd5841e1c'
INTAKE_FILES = frozenset(('subnet/backend_jobs.py', 'subnet/committed_training_inputs.py',
    'subnet/training_receipts.py', 'subnet/training_task_representatives.py',
    'subnet/unaudited_training_execution.py'))

OBJECTIVE_VERSION = 'unaudited-training-execution-amendment-v5-objective-horizon'
OBJECTIVE_METHOD = 'fp32-task-gradient-effective-lr-objective-horizon-v1'
OBJECTIVE_QUALIFICATION = 'configured-objective-horizon-retained-adam-qualification-v1'
OBJECTIVE_PREPARATION = 'unaudited-training-execution-preparation-v5-objective-horizon'
OBJECTIVE_RELEASE = 'unaudited-training-execution-release-v5-objective-horizon'
OBJECTIVE_VERSIONS = (OBJECTIVE_VERSION,OBJECTIVE_PREPARATION,OBJECTIVE_RELEASE)


def configured_objective(value):
    from .training_policy import objective_config, objective_horizon, objective_at_step
    if value.get('version') not in OBJECTIVE_VERSIONS:
        if 'training_objective' in value or 'training_horizon' in value:
            raise ValueError('historical execution cannot add an objective')
        return None
    if type(value.get('training_objective')) is not dict:
        raise ValueError('new objective execution requires explicit configuration')
    actual=objective_config(value['training_objective'])
    horizon=objective_horizon(value.get('training_horizon'))
    if type(value.get('steps')) is not int or value.get('steps')!=1 or value.get('effective_learning_rate')!=5e-7:
        raise ValueError('objective horizon keeps one update and exact qualified LR')
    if value['version']==OBJECTIVE_RELEASE:
        if value.get('minimum_optimizer_step')!=horizon['first_optimizer_step']:
            raise ValueError('release starts at the exact retained-Adam horizon parent')
        expected=objective_at_step(horizon,horizon['first_optimizer_step'])
    else:
        expected=objective_at_step(horizon,value.get('optimizer_step_before'))
        if (value['optimizer_step_before']==horizon['first_optimizer_step']
                and value.get('parent_descriptor_sha256')!=horizon['initial_parent_descriptor_sha256']):
            raise ValueError('first objective update requires the original retained descriptor')
    if actual!=expected:
        raise ValueError('coefficient must follow the fixed successful-update horizon')
    return actual


def _fields(value):
    return FIELDS | {'training_objective','training_horizon'} if value.get('version') in OBJECTIVE_VERSIONS else FIELDS


SCIENTIFIC_CHANGES = frozenset(('subnet/fp32_gradient_accumulation.py',
    'subnet/task_normalized_training.py', 'subnet/persistent_cpu_adamw.py',
    'subnet/learning_rate_transition.py','subnet/persistent_training_state.py',
    'subnet/persistent_training_protocol.py','subnet/persistent_training_worker.py'))
OPERATIONAL_CHANGES = frozenset(('subnet/unaudited_training_execution.py',
    'subnet/backend_jobs.py', 'subnet/remote_backend.py',
    'subnet/persistent_training_controller.py'))
REPRESENTATIVE_CHANGES = frozenset(('subnet/training_task_representatives.py',
    'subnet/committed_training_inputs.py','subnet/training_receipts.py'))
FIELDS = frozenset(('version','method','epoch','job_id','original_signed_manifest_sha256',
    'original_public_manifest_sha256','original_source_bundle_sha256',
    'training_source_bundle','original_source_files','execution_source_files',
    'changed_source_files','execution_qualification','runtime_versions',
    'training_runtime_sha256','training_policy','training_input_policy','steps',
    'input_inventory_sha256','native_eligibility_receipt','trainer_binding_sha256',
    'parent_descriptor_sha256','genesis_sha256','optimizer_step_before',
    'created_at','expires_at','execution_release_sha256','effective_learning_rate','learning_rate_authorization'))

def _base_hyperparameters_sha256():
    from .persistent_cpu_adamw import HYPERPARAMETERS
    return sha(HYPERPARAMETERS)

def _learning_rate(value):
    rate=value.get('effective_learning_rate')
    if type(rate)not in(int,float)or not math.isfinite(rate)or not 0<rate<1e-5:
        raise ValueError('explicit corrective positive LR below original1e-5')
    return rate

def _grant_payload(value,manifest):
    from .learning_rate_transition import VERSION as LR_VERSION, GENESIS_AUTH_VERSION
    binding=validate_binding(manifest['trainer_state_binding'],manifest)
    result = dict(version=LR_VERSION,epoch=value['epoch'],job_id=value['job_id'],
        input_checkpoint=binding['input_checkpoint'],parent_descriptor_sha256=value['parent_descriptor_sha256'],
        genesis_sha256=value['genesis_sha256'],optimizer_step_before=value['optimizer_step_before'],
        steps=value['steps'],parameters_sha256=binding['parameters_sha256'],
        base_hyperparameters_sha256=_base_hyperparameters_sha256(),
        effective_learning_rate=_learning_rate(value),created_at=value['created_at'],
        expires_at=value['expires_at'],execution_release_sha256=value['execution_release_sha256'])
    if value['method'] == GENESIS_METHOD:
        if binding['genesis'] is None:raise ValueError('explicit genesis binding required')
        result.update(version=GENESIS_AUTH_VERSION,run_id=binding['genesis']['run_id'])
    return result

def _unchanged_manifest(manifest):
    if any(k in manifest for k in ('training_startup_recovery', 'training_execution_amendment', FIELD)):
        raise ValueError('fresh ordinary unaudited execution only; no recovery nesting')
    if (manifest.get('training_policy') != POLICY
            or manifest.get('training_input_policy') != INPUT_POLICY):
        raise ValueError('original unaudited persistent training policy')

def _sources(value):
    before=value['original_source_files'];after=value['execution_source_files']
    if (not isinstance(before,dict) or not isinstance(after,dict) or not before
            or not set(before)<=set(after) or 'subnet/unaudited_training_execution.py'not in after):
        raise ValueError('complete unchanged source closure membership')
    for name,hashed in {**before,**after}.items():
        if (not isinstance(name,str) or not name.startswith('subnet/')
                or '..' in name or not name.endswith('.py')):
            raise ValueError('exact execution source member')
        digest(hashed)
    changed={name:hashed for name,hashed in after.items()if before.get(name)!=hashed}
    objective=configured_objective(value)
    extra={'subnet/persistent_training_evidence.py','subnet/training_policy.py'} if objective is not None else set()
    representative = 'subnet/training_task_representatives.py' in after
    additions = REPRESENTATIVE_CHANGES if representative else frozenset()
    if representative and not REPRESENTATIVE_CHANGES <= set(changed):
        raise ValueError('complete explicit representative collection/preparation source delta')
    if (changed!=value['changed_source_files'] or not SCIENTIFIC_CHANGES<=set(changed)
            or objective is not None and not extra<=set(changed)
            or set(changed)-SCIENTIFIC_CHANGES-OPERATIONAL_CHANGES-extra-additions):
        raise ValueError('only approved FP32 accumulation and explicit routing source delta')
    bundle=value['training_source_bundle']
    if (not isinstance(bundle,dict) or bundle.get('sha256')==value['original_source_bundle_sha256']):
        raise ValueError('distinct honest execution bundle')
    digest(bundle.get('sha256'));digest(value['original_source_bundle_sha256'])

def _qualification(value,authority):
    _learning_rate(value)
    q=authenticate(value['execution_qualification'],authority)
    if q.get('version') == OBJECTIVE_INTAKE_EQUIVALENCE:
        return _objective_intake_equivalence(value,q,authority)
    if q.get('version') == INTAKE_EQUIVALENCE:
        return _intake_equivalence(value,q,authority)
    fields={'version','method','execution_source_bundle_sha256','execution_source_files_sha256',
        'runtime_versions','training_runtime_sha256','actual_GPU_execution','passed',
        'report_sha256','optimizer_reset','objective_changed','hyperparameters_changed',
        'effective_learning_rate','base_hyperparameters_sha256','state_version'}
    objective=configured_objective(value)
    if objective is not None:
        from .training_policy import objective_config
        expected_fields=fields | {'training_objective','tested_positive_nll_weights','retained_parent_descriptor_sha256','retained_optimizer_step','retained_input_checkpoint','qualification_scope','live_optimizer_distribution_equivalence_claimed','initial_execution_source_bundle_sha256'}
        if (set(q)!=expected_fields or q['version']!=OBJECTIVE_QUALIFICATION
                or q['method']!=OBJECTIVE_METHOD or value['method']!=OBJECTIVE_METHOD
                or objective_config(q['training_objective'])!={'version':'sequence-mean-preference-positive-nll-v1','positive_nll_weight':1.0}
                or q['tested_positive_nll_weights']!=[0.0,1.0]
                or any(type(x) not in (int,float) for x in q['tested_positive_nll_weights'])
                or type(q['retained_optimizer_step']) is not int or q['retained_optimizer_step']<1
                or q['qualification_scope']!='private-nonzero-adam-mechanical-continuation-v1'
                or q['live_optimizer_distribution_equivalence_claimed'] is not False
                or q['execution_source_bundle_sha256']!=value['training_source_bundle']['sha256']
                or q['execution_source_files_sha256']!=sha(value['execution_source_files'])
                or q['runtime_versions']!=value['runtime_versions']
                or q['training_runtime_sha256']!=value['training_runtime_sha256']
                or q['actual_GPU_execution']is not True or q['passed']is not True
                or q['optimizer_reset']is not False
                or q['objective_changed']is not True
                or q['hyperparameters_changed']is not True
                or q['effective_learning_rate']!=value['effective_learning_rate']
                or q['base_hyperparameters_sha256']!=_base_hyperparameters_sha256()
                or q['state_version']!='persistent-fp32-trainer-state-v2-effective-lr'):
            raise ValueError('configured objective needs exact retained-Adam GPU qualification')
        for field in ('report_sha256','retained_parent_descriptor_sha256','retained_input_checkpoint','initial_execution_source_bundle_sha256'):
            digest(q[field])
        return
    genesis_qualified = q.get('version') == GENESIS_QUALIFICATION
    if genesis_qualified:
        fields = fields | {'tested_methods'}
        if q.get('tested_methods') != [GENESIS_METHOD, METHOD]:
            raise ValueError('genesis qualification must test creation and successor continuation')
    if (set(q)!=fields or q['version']!=(GENESIS_QUALIFICATION if genesis_qualified else QUALIFICATION)
            or q['method']!=(GENESIS_METHOD if genesis_qualified else METHOD)
            or value['method'] not in ((GENESIS_METHOD,METHOD) if genesis_qualified else (METHOD,))
            or q['execution_source_bundle_sha256']!=value['training_source_bundle']['sha256']
            or q['execution_source_files_sha256']!=sha(value['execution_source_files'])
            or q['runtime_versions']!=value['runtime_versions']
            or q['training_runtime_sha256']!=value['training_runtime_sha256']
            or q['actual_GPU_execution']is not True or q['passed']is not True
            or q['optimizer_reset']is not genesis_qualified or q['objective_changed']is not False
            or q['hyperparameters_changed']is not True
            or q['effective_learning_rate']!=value['effective_learning_rate']
            or q['base_hyperparameters_sha256']!=_base_hyperparameters_sha256()
            or q['state_version']!='persistent-fp32-trainer-state-v2-effective-lr'):
        raise ValueError('new execution requires its own authenticated GPU qualification')
    digest(q['report_sha256'])

def _intake_equivalence(value,q,authority):
    """Reuse real GPU mechanics evidence only for an explicit intake-only delta.

    This is not GPU qualification of the new bundle. A separate ROOT attestation
    binds exact source-equivalence and installed CPU controls; original actual GPU
    evidence remains unchanged and independently authenticated. New optimizer,
    model, gradient, state, objective or publication code cannot use this route.
    """
    fields={'version','method','execution_source_bundle_sha256','execution_source_files_sha256',
        'runtime_versions','training_runtime_sha256','actual_GPU_execution','passed',
        'optimizer_reset','objective_changed','hyperparameters_changed','effective_learning_rate',
        'base_hyperparameters_sha256','state_version','qualified_predecessor_release','equivalence_evidence'}
    if (set(q)!=fields or q['method']!=METHOD or value['method']!=METHOD
            or q['actual_GPU_execution'] is not False or q['passed'] is not True
            or any(q[k] is not False for k in ('optimizer_reset','objective_changed','hyperparameters_changed'))
            or q['execution_source_bundle_sha256']!=value['training_source_bundle']['sha256']
            or q['execution_source_files_sha256']!=sha(value['execution_source_files'])
            or q['runtime_versions']!=value['runtime_versions']
            or q['training_runtime_sha256']!=value['training_runtime_sha256']
            or q['effective_learning_rate']!=value['effective_learning_rate']
            or q['base_hyperparameters_sha256']!=_base_hyperparameters_sha256()
            or q['state_version']!='persistent-fp32-trainer-state-v2-effective-lr'):
        raise ValueError('explicit honest retained-state intake-equivalence qualification')
    previous=authenticate(q['qualified_predecessor_release'],authority)
    # Calling the existing release validator preserves all original actual GPU
    # checks. Nested equivalence or NLL qualifications are explicitly forbidden.
    oldq=authenticate(previous.get('execution_qualification'),authority)
    if (oldq.get('version') not in (QUALIFICATION,GENESIS_QUALIFICATION)
            or previous.get('training_source_bundle',{}).get('sha256')!=INTAKE_BASE_BUNDLE):
        raise ValueError('original actual GPU-qualified 395bd predecessor required')
    release(q['qualified_predecessor_release'],authority)
    for field in ('method','runtime_versions','training_runtime_sha256','training_policy',
                  'training_input_policy','effective_learning_rate','original_source_files',
                  'original_source_bundle_sha256'):
        if value.get(field)!=previous.get(field):
            raise ValueError('intake equivalence cannot change original model/runtime/training contract')
    before=previous['execution_source_files'];after=value['execution_source_files']
    changed={name:h for name,h in after.items() if before.get(name)!=h}
    if (len(before)!=185 or len(after)!=186
            or set(after)!=set(before)|{'subnet/training_task_representatives.py'}
            or set(changed)!=INTAKE_FILES):
        raise ValueError('only exact five-file metadata/native-intake delta; all math/state/publication files identical')
    evidence=authenticate(q['equivalence_evidence'],authority)
    expected={'version','previous_release_sha256','previous_source_files_sha256','source_files_sha256',
        'changed_files','all_other_scientific_files_byte_identical','actual_GPU_execution',
        'installed_native_child','identical_input_emission','original_integration_review'}
    if (set(evidence)!=expected or evidence['version']!='native-task-intake-source-equivalence-evidence-v1'
            or evidence['previous_release_sha256']!=sha(q['qualified_predecessor_release'])
            or evidence['previous_source_files_sha256']!=sha(before)
            or evidence['source_files_sha256']!=sha(after) or evidence['changed_files']!=changed
            or evidence['all_other_scientific_files_byte_identical'] is not True
            or evidence['actual_GPU_execution'] is not False):
        raise ValueError('exact separately authenticated intake-equivalence evidence')
    for name in ('installed_native_child','identical_input_emission','original_integration_review'):
        row=evidence[name]
        if type(row) is not dict or set(row)!={'sha256','passed'} or row['passed'] is not True:
            raise ValueError('passed exact installed/parity/selection evidence required')
        digest(row['sha256'])

def _objective_intake_equivalence(value,q,authority):
    """Keep private GPU mechanics and new intake-only CPU evidence distinct.

    The exact552 predecessor qualification is immutable. No synthetic production
    release or claim that the new186 bundle executed on a GPU is introduced.
    Relative to552, all model/gradient/Adam/state/publication modules are exact;
    only the five declared metadata/input modules may differ. The original
    objective validator still governs LR and the actual live16-update horizon.
    """
    fields={'version','method','execution_source_bundle_sha256','execution_source_files_sha256',
        'runtime_versions','training_runtime_sha256','actual_GPU_execution','passed',
        'optimizer_reset','objective_changed','hyperparameters_changed','effective_learning_rate',
        'base_hyperparameters_sha256','state_version','qualified_predecessor_qualification',
        'qualified_predecessor_source_files','equivalence_evidence'}
    if (set(q)!=fields or q['method']!=OBJECTIVE_METHOD or value['method']!=OBJECTIVE_METHOD
            or configured_objective(value) is None
            or q['actual_GPU_execution'] is not False or q['passed'] is not True
            or any(q[k] is not False for k in ('optimizer_reset','objective_changed','hyperparameters_changed'))
            or q['execution_source_bundle_sha256']!=value['training_source_bundle']['sha256']
            or q['execution_source_files_sha256']!=sha(value['execution_source_files'])
            or q['runtime_versions']!=value['runtime_versions']
            or q['training_runtime_sha256']!=value['training_runtime_sha256']
            or q['effective_learning_rate']!=value['effective_learning_rate']
            or q['base_hyperparameters_sha256']!=_base_hyperparameters_sha256()
            or q['state_version']!='persistent-fp32-trainer-state-v2-effective-lr'):
        raise ValueError('honest retained-NLL intake-equivalence qualification required')
    before=q['qualified_predecessor_source_files'];after=value['execution_source_files']
    oldq=authenticate(q['qualified_predecessor_qualification'],authority)
    if (oldq.get('version')!=OBJECTIVE_QUALIFICATION
            or oldq.get('execution_source_bundle_sha256')!=OBJECTIVE_INTAKE_BASE_BUNDLE
            or oldq.get('execution_source_files_sha256')!=OBJECTIVE_INTAKE_BASE_MAP
            or not isinstance(before,dict) or sha(before)!=OBJECTIVE_INTAKE_BASE_MAP):
        raise ValueError('exact552 private actual-GPU qualification and185 map required')
    predecessor=copy.deepcopy(value)
    predecessor['execution_qualification']=q['qualified_predecessor_qualification']
    predecessor['training_source_bundle']={'sha256':OBJECTIVE_INTAKE_BASE_BUNDLE}
    predecessor['execution_source_files']=before
    # This reuses every actual retained-Adam GPU qualification predicate. Its
    # private step1 is never substituted for the separately bound live horizon g.
    _qualification(predecessor,authority)
    changed={name:h for name,h in after.items() if before.get(name)!=h}
    if (len(before)!=185 or len(after)!=186
            or set(after)!=set(before)|{'subnet/training_task_representatives.py'}
            or set(changed)!=INTAKE_FILES):
        raise ValueError('only five explicit intake files may differ from qualified552')
    evidence=authenticate(q['equivalence_evidence'],authority)
    fields={'version','qualified_predecessor_qualification_sha256','previous_source_files_sha256',
        'source_files_sha256','changed_files','all_other_scientific_files_byte_identical',
        'actual_GPU_execution','installed_native_child','identical_input_emission',
        'representative_integration_review'}
    if (set(evidence)!=fields or evidence['version']!='retained-nll-native-intake-source-equivalence-evidence-v1'
            or evidence['qualified_predecessor_qualification_sha256']!=sha(q['qualified_predecessor_qualification'])
            or evidence['previous_source_files_sha256']!=sha(before)
            or evidence['source_files_sha256']!=sha(after) or evidence['changed_files']!=changed
            or evidence['all_other_scientific_files_byte_identical'] is not True
            or evidence['actual_GPU_execution'] is not False):
        raise ValueError('exact separately authenticated552-to186 intake evidence required')
    for name in ('installed_native_child','identical_input_emission','representative_integration_review'):
        row=evidence[name]
        if type(row) is not dict or set(row)!={'sha256','passed'} or row['passed'] is not True:
            raise ValueError('passed installed/parity/selection CPU evidence required')
        digest(row['sha256'])

def validate(job_envelope,authority,*,now=None):
    """Authenticate before using the execution bundle, even for source staging."""
    job=authenticate(job_envelope,authority)
    manifest=authenticate(job['manifest'],authority)
    _unchanged_manifest(manifest)
    value=authenticate(job.get(FIELD),authority)
    if set(value)!=_fields(value) or (value['version'],value['method'])not in ((VERSION,METHOD),(GENESIS_VERSION,GENESIS_METHOD),(OBJECTIVE_VERSION,OBJECTIVE_METHOD)):
        raise ValueError('exact prospective unaudited execution declaration')
    if (job.get('role')!='train' or value['epoch']!=manifest['epoch']
            or value['job_id']!=job.get('job_id')
            or value['original_signed_manifest_sha256']!=sha(job['manifest'])
            or value['original_source_bundle_sha256']!=manifest['source_bundle']['sha256']
            or job.get('training_policy')!=value['training_policy'] or value['training_policy']!=POLICY
            or job.get('training_input_policy')!=value['training_input_policy'] or value['training_input_policy']!=INPUT_POLICY
            or type(value['steps'])is not int or value['steps']<1 or value['steps']!=job.get('steps')
            or value['execution_source_files']!=job.get('source_files')
            or value['runtime_versions']!=job.get('runtime_versions')
            or value['training_runtime_sha256']!=sha(manifest.get('training_runtime'))):
        raise ValueError('exact original job manifest/runtime/source/objective scope')
    digest(value['original_public_manifest_sha256']);digest(value['execution_release_sha256'])
    if len(__import__('json').dumps(job[FIELD],sort_keys=True,separators=(',',':')).encode())>DECLARATION_MAX_BYTES:
        raise ValueError('compact execution declaration budget')
    _sources(value);_qualification(value,authority)
    if ('subnet/training_task_representatives.py' in value['execution_source_files']) != ('training_representative_policy' in manifest):
        raise ValueError('representative execution source requires explicit original signed policy')
    binding=validate_binding(manifest.get('trainer_state_binding'),manifest)
    parent=binding['parent']
    initial = value['version'] == GENESIS_VERSION
    if initial:
        from .learning_rate_transition import GENESIS_DOCUMENT_VERSION
        if (parent is not None or binding['global_step_before'] != 0
                or not isinstance(binding['genesis'],dict)
                or binding['genesis'].get('version') != GENESIS_DOCUMENT_VERSION
                or binding['genesis']['initial_effective_learning_rate'] != value['effective_learning_rate']):
            raise ValueError('exact explicit unique genesis and initial learning rate')
    elif parent is None or binding['genesis'] is not None or binding['global_step_before'] < 1:
        raise ValueError('exact retained parent descriptor/genesis/step; no optimizer reset')
    if (value['trainer_binding_sha256']!=sha(binding)
            or value['parent_descriptor_sha256']!=(None if initial else parent['descriptor_sha256'])
            or value['genesis_sha256']!=binding['genesis_sha256']
            or value['optimizer_step_before']!=binding['global_step_before']):
        raise ValueError('exact retained parent descriptor/genesis/step; no optimizer reset')
    if (value['version']==OBJECTIVE_VERSION
            and binding['global_step_before']==value['training_horizon']['first_optimizer_step']
            and binding['input_checkpoint']!=value['training_horizon']['initial_checkpoint']):
        raise ValueError('first objective update requires exact qualified checkpoint')
    from .learning_rate_transition import validate_authorization
    expected_grant=_grant_payload(value,manifest)
    actual_grant=authenticate(value['learning_rate_authorization'],authority)
    if actual_grant!=expected_grant:raise ValueError('exact LR grant must match execution declaration')
    validate_authorization(value['learning_rate_authorization'],authority,
        **{key:expected_grant[key]for key in ('epoch','job_id','input_checkpoint','parent_descriptor_sha256',
            'genesis_sha256','optimizer_step_before','steps','parameters_sha256')},
        now=now,historical=now is None)
    native=manifest.get('native_training_eligibility_receipt')
    if (native!=value['native_eligibility_receipt'] or not isinstance(native,dict)
            or native.get('sampling_assurance')!='unaudited'
            or any(native.get(k)is not False for k in ('proof_verification_performed','claims_rewritten','cheating_penalties'))):
        raise ValueError('original native subset remains unaudited and unmodified')
    if 'training_representative_policy' in manifest:
        from .training_task_representatives import validate_receipt
        validate_receipt(native)
    else:
        for key in ('context_sha256','grades_sha256','subset_sha256','authorization_sha256'):digest(native.get(key))
    submissions=job.get('submissions')
    if not isinstance(submissions,list) or not 1<=len(submissions)<=training_document_cap(manifest):
        raise ValueError('bounded original native-selected inputs')
    if value['input_inventory_sha256']!=sha(receipt_inventory(submissions)):
        raise ValueError('exact frozen native-selected input inventory')
    # These are the existing cheap signature/coverage checks; no proof or grader.
    from subnet.committed_training_inputs import validate_job
    validate_job(job,manifest,authority)
    times=[value['created_at'],job.get('created_at'),job.get('expires_at'),value['expires_at']]
    if (any(type(t)not in(int,float) or not math.isfinite(t)for t in times)
            or not times[0]<=times[1]<times[2]<=times[3] or not 0<times[3]-times[0]<=86400
            or now is not None and not times[1]<=now<times[2]):
        raise ValueError('original bounded execution authorization lifetime')
    return value

def preparation_scope(public_envelope,native_envelope,submissions,native_documents,authority):
    """Coordinator-only gate before signing a prospective execution declaration."""
    if 'training_representative_policy' in authenticate(native_envelope,authority):
        from .training_task_representatives import preparation_scope as representative_scope
        return representative_scope(public_envelope,native_envelope,submissions,native_documents,authority)
    public=authenticate(public_envelope,authority);native=authenticate(native_envelope,authority)
    _unchanged_manifest(native)
    if (computation_binding(public)!=computation_binding(native)
            or public.get('trainer_state_binding')!=native.get('trainer_state_binding')):
        raise ValueError('public miner contract and parent remain unchanged')
    receipt=native.get('native_training_eligibility_receipt',{})
    documents={name:authenticate(native_documents[name],authority)for name in ('context','grades','subset')}
    for name,envelope in native_documents.items():
        if name not in documents or receipt.get(name+'_sha256')!=sha(envelope):
            raise ValueError('exact authenticated native context/grades/subset')
    context,grades,subset=(documents[n]for n in ('context','grades','subset'))
    original=authenticate(context['original_signed_manifest'],authority)
    inventory=receipt_inventory(submissions)
    if (computation_binding(original)!=computation_binding(native)
            or context['parent_binding_sha256']!=sha(native['trainer_state_binding'])
            or grades['context_sha256']!=sha(native_documents['context'])
            or subset['context_sha256']!=sha(native_documents['context'])
            or subset['grade_receipt_sha256']!=sha(grades)
            or receipt_inventory(subset['accepted_submissions'])!=inventory
            or subset['accepted_inventory_sha256']!=sha(inventory)
            or subset['sampling_assurance']!='unaudited' or subset['claims_rewritten']is not False):
        raise ValueError('same authenticated frozen native-selected population')
    return dict(original_public_manifest_sha256=sha(public_envelope),
        original_signed_manifest_sha256=sha(native_envelope),input_inventory_sha256=sha(inventory),
        native_eligibility_receipt=copy.deepcopy(receipt))

def execution_bundle(job_envelope,authority,*,now=None):
    value=validate(job_envelope,authority,now=now)
    return copy.deepcopy(value['training_source_bundle'])

def provenance(job_envelope,authority):
    value=validate(job_envelope,authority)
    job=authenticate(job_envelope,authority)
    return dict(version=value['version'],method=value['method'],declaration_sha256=sha(job[FIELD]),execution_release_sha256=value['execution_release_sha256'],
        original_public_manifest_sha256=value['original_public_manifest_sha256'],
        original_input_source_sha256=value['original_source_bundle_sha256'],
        execution_source_bundle_sha256=value['training_source_bundle']['sha256'],
        execution_source_files_sha256=sha(value['execution_source_files']),
        qualification_sha256=sha(value['execution_qualification']),
        input_inventory_sha256=value['input_inventory_sha256'],
        native_eligibility_receipt_sha256=sha(value['native_eligibility_receipt']),
        trainer_binding_sha256=value['trainer_binding_sha256'],
        parent_descriptor_sha256=value['parent_descriptor_sha256'],
        genesis_sha256=value['genesis_sha256'],optimizer_step_before=value['optimizer_step_before'],
        optimizer_step_after=value['optimizer_step_before']+value['steps'],
        input_assurance='unaudited',proof_verification_performed=False,optimizer_reset=value['version']==GENESIS_VERSION,
        effective_learning_rate=value['effective_learning_rate'],
        learning_rate_authorization_sha256=sha(value['learning_rate_authorization']),
        base_hyperparameters_sha256=_base_hyperparameters_sha256(),
        state_version='persistent-fp32-trainer-state-v2-effective-lr',
        **({'training_objective':configured_objective(value),'training_horizon':copy.deepcopy(value['training_horizon'])} if value['version']==OBJECTIVE_VERSION else {}))

def validate_provenance(report,job_envelope,authority):
    job=authenticate(job_envelope,authority)
    if (report.get('job_sha256')!=sha(job) or report.get('source_files')!=job['source_files']
            or report.get(FIELD)!=provenance(job_envelope,authority)):
        raise ValueError('truthful original-input versus actual-execution report provenance')

PREPARATION_VERSION = 'unaudited-training-execution-preparation-v2-effective-lr'
DECLARATION_MAX_BYTES = 256_000
JOB_MAX_BYTES = 8_000_000
CAPACITY_JOB_MAX_BYTES = 32_000_000

def job_envelope_limit(manifest):
    """Only explicit signed9/512 ordinary execution gets the larger envelope."""
    from .committed_training_inputs import prospective_envelope_limit
    return prospective_envelope_limit(manifest,JOB_MAX_BYTES)

def preparation(authorization,authority,*,now=None):
    """Authenticate a compact, exact-epoch, exact-input ROOT preparation grant."""
    value=authenticate(authorization,authority)
    if (set(value)!=_fields(value)-{'job_id','learning_rate_authorization'} or
            (value.get('version'),value.get('method')) not in ((PREPARATION_VERSION,METHOD),(GENESIS_PREPARATION,GENESIS_METHOD),(OBJECTIVE_PREPARATION,OBJECTIVE_METHOD))):
        raise ValueError('exact training execution preparation authorization')
    if (value.get('training_policy')!=POLICY
            or value.get('training_input_policy')!=INPUT_POLICY):
        raise ValueError('approved unaudited preparation method')
    if len(__import__('json').dumps(authorization,sort_keys=True,separators=(',',':')).encode())>DECLARATION_MAX_BYTES:
        raise ValueError('compact training execution authorization budget')
    _sources(value);_qualification(value,authority)
    times=(value['created_at'],value['expires_at'])
    if (any(type(t)not in(int,float)or not math.isfinite(t)for t in times)
            or not 0<times[1]-times[0]<=86400
            or now is not None and not times[0]<=now<times[1]):
        raise ValueError('bounded preparation authorization lifetime')
    return value

def attach(payload,authorization,authority,sign):
    """Called once after the ordinary job ID/inputs/parent are fully prepared.

    No new source or runtime is selected from unsigned configuration. The remote
    source is already staged; its entire measured file map must equal this grant.
    """
    if FIELD in payload:raise ValueError('execution declaration already attached')
    value=copy.deepcopy(preparation(authorization,authority,now=payload.get('created_at')))
    value['version']=(OBJECTIVE_VERSION if value['method']==OBJECTIVE_METHOD else GENESIS_VERSION if value['method']==GENESIS_METHOD else VERSION);value['job_id']=payload['job_id']
    value['learning_rate_authorization']=sign(_grant_payload(value,authenticate(payload['manifest'],authority)))
    result=copy.deepcopy(payload);result[FIELD]=sign(value)
    envelope=sign(result);validate(envelope,authority,now=payload['created_at'])
    if len(__import__('json').dumps(envelope,sort_keys=True,separators=(',',':')).encode())>job_envelope_limit(authenticate(payload['manifest'],authority)):
        raise ValueError('ordinary amended job envelope exceeds bounded budget')
    return result

def admission(job_envelope,authority,*,now=None):
    """New FP32 execution cannot silently masquerade as the historical method."""
    job=authenticate(job_envelope,authority)
    fp32='subnet/fp32_gradient_accumulation.py'in job.get('source_files',{})
    if FIELD not in job:
        if fp32 and job.get('role')=='train' and job.get('training_policy')==POLICY:
            raise ValueError('FP32 persistent execution requires explicit qualified declaration')
        return None
    return validate(job_envelope,authority,now=now)

def required_report(report,job_envelope,authority):
    value=admission(job_envelope,authority)
    if value is None:
        if FIELD in report:raise ValueError('unrequested training execution provenance')
        return None
    validate_provenance(report,job_envelope,authority)
    return copy.deepcopy(report[FIELD])

RELEASE_VERSION = 'unaudited-training-execution-release-v2-effective-lr'
RELEASE_FIELDS = frozenset(('version','method','original_source_bundle_sha256',
    'training_source_bundle','original_source_files','execution_source_files','changed_source_files',
    'execution_qualification','runtime_versions','training_runtime_sha256','training_policy',
    'training_input_policy','genesis_sha256','hyperparameters_sha256','epoch_prefix',
    'first_round','minimum_optimizer_step','steps','created_at','expires_at','effective_learning_rate'))

def release(authorization,authority,*,now=None):
    value=authenticate(authorization,authority)
    initial_capable = value.get('version') == GENESIS_RELEASE
    objective_release = value.get('version') == OBJECTIVE_RELEASE
    expected_fields=(RELEASE_FIELDS | {'genesis_document'} if initial_capable else
                     RELEASE_FIELDS | {'training_objective','training_horizon'} if objective_release else RELEASE_FIELDS)
    if (set(value)!=expected_fields
            or value['version']!=(GENESIS_RELEASE if initial_capable else OBJECTIVE_RELEASE if objective_release else RELEASE_VERSION)
            or value['method']!=(GENESIS_METHOD if initial_capable else OBJECTIVE_METHOD if objective_release else METHOD) or value['training_policy']!=POLICY
            or value['training_input_policy']!=INPUT_POLICY):
        raise ValueError('exact approved training execution release')
    _sources(value);_qualification(value,authority)
    import re
    if (type(value['epoch_prefix'])is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,70}',value['epoch_prefix'])
            or any(type(value[k])is not int or value[k]<0 for k in ('first_round','minimum_optimizer_step'))
            or type(value['steps'])is not int or not 1<=value['steps']<=32):
        raise ValueError('bounded release activation and step policy')
    for key in ('genesis_sha256','hyperparameters_sha256'):digest(value[key])
    if initial_capable:
        from .learning_rate_transition import validate_genesis_document
        document=value['genesis_document']
        validate_genesis_document(document,document.get('parameters_sha256'),document.get('input_checkpoint'))
        if (sha(document)!=value['genesis_sha256'] or document['initial_effective_learning_rate']!=value['effective_learning_rate']
                or value['minimum_optimizer_step']!=0 or sha(document['hyperparameters'])!=value['hyperparameters_sha256']):
            raise ValueError('release exact unique genesis document/rate/zero-step scope')
    elif value['minimum_optimizer_step'] < 1:
        raise ValueError('existing-parent release cannot authorize genesis')
    times=(value['created_at'],value['expires_at'])
    if (any(type(t)not in(int,float)or not math.isfinite(t)for t in times)
            or not 0<times[1]-times[0]<=366*86400
            or now is not None and not times[0]<=now<times[1]):
        raise ValueError('bounded signed release lifetime')
    if len(__import__('json').dumps(authorization,sort_keys=True,separators=(',',':')).encode())>DECLARATION_MAX_BYTES:
        raise ValueError('compact execution release budget')
    return value

def derive_preparation(release_envelope,job,public_envelope,native_documents,authority,*,created_at):
    """Automatically derive one exact epoch grant from one approved release.

    The caller owns the existing ROOT signer. This function authenticates every
    input to signing; it is not a new worker bypass or an audit requirement.
    """
    r=release(release_envelope,authority,now=created_at)
    m=authenticate(job['manifest'],authority);_unchanged_manifest(m)
    import re
    suffix=m['epoch'][len(r['epoch_prefix']):]if m['epoch'].startswith(r['epoch_prefix'])else''
    if (not re.fullmatch(r'[0-9]+-[0-9]+',suffix) or int(suffix.rsplit('-',1)[1])<r['first_round']
            or job.get('role')!='train' or job.get('steps')!=r['steps']
            or job.get('training_policy')!=r['training_policy']
            or job.get('training_input_policy')!=r['training_input_policy']
            or m['source_bundle']['sha256']!=r['original_source_bundle_sha256']
            or job.get('source_files')!=r['execution_source_files']
            or job.get('runtime_versions')!=r['runtime_versions']
            or sha(m.get('training_runtime'))!=r['training_runtime_sha256']):
        raise ValueError('job outside approved execution release scope')
    binding=validate_binding(m.get('trainer_state_binding'),m)
    initial=binding['parent'] is None
    if initial:
        if (r['version']!=GENESIS_RELEASE or binding['genesis']!=r['genesis_document']
                or binding['global_step_before']!=0):
            raise ValueError('new optimizer requires explicitly released exact unique genesis')
    elif binding['genesis'] is not None or binding['global_step_before'] < 1:
        raise ValueError('exact authenticated successor parent required')
    if (binding['genesis_sha256']!=r['genesis_sha256']
            or sha(binding['hyperparameters'])!=r['hyperparameters_sha256']
            or binding['global_step_before']<r['minimum_optimizer_step']):
        raise ValueError('release requires current authenticated retained parent and unchanged genesis/hyperparameters')
    scope=preparation_scope(public_envelope,job['manifest'],job['submissions'],native_documents,authority)
    # All static source/qualification values remain those approved in the release.
    value={key:copy.deepcopy(r[key])for key in _fields(r) if key in r}
    objective_release=r['version']==OBJECTIVE_RELEASE
    value.update(version=GENESIS_PREPARATION if initial else OBJECTIVE_PREPARATION if objective_release else PREPARATION_VERSION,
        method=GENESIS_METHOD if initial else OBJECTIVE_METHOD if objective_release else METHOD,epoch=m['epoch'],**scope,
        execution_release_sha256=sha(release_envelope),trainer_binding_sha256=sha(binding),
        parent_descriptor_sha256=None if initial else binding['parent']['descriptor_sha256'],
        optimizer_step_before=binding['global_step_before'],created_at=created_at,
        expires_at=min(r['expires_at'],created_at+86400))
    if objective_release:
        from .training_policy import objective_at_step
        value['training_objective']=objective_at_step(value['training_horizon'],binding['global_step_before'])
        configured_objective(value)
        if (binding['global_step_before']==value['training_horizon']['first_optimizer_step']
                and binding['input_checkpoint']!=value['training_horizon']['initial_checkpoint']):
            raise ValueError('objective release requires the exact qualified initial checkpoint')
    if set(value)!=_fields(value)-{'job_id','learning_rate_authorization'}:raise ValueError('complete derived epoch execution scope')
    return value

def automatic_preparation(controller,job,release_envelope):
    """Authenticate native receipts and atomically journal/reuse an epoch grant."""
    import json,os,tempfile
    from pathlib import Path
    authority=controller.authority.id;manifest=authenticate(job['manifest'],authority)
    epoch=manifest['epoch']
    if not isinstance(epoch,str)or Path(epoch).name!=epoch or epoch in('.','..'):
        raise ValueError('canonical original epoch namespace')
    state=Path(controller.state).resolve();native=state/'native-outcome-eligibility'/epoch
    maximum=job_envelope_limit(manifest)
    def read(path):
        if path.is_symlink()or path.resolve()!=path or not path.is_file()or path.stat().st_size>maximum:
            raise ValueError('bounded original signed eligibility file')
        with path.open('rb')as stream:data=stream.read(maximum+1)
        if len(data)>maximum:raise ValueError('bounded original signed eligibility file')
        return json.loads(data)
    public=read(state/(epoch+'-first-signed-manifest.json'))
    if 'training_representative_policy' in manifest:
        from ops.native_task_representative_selection import read_documents
        documents=read_documents(native)
    else:
        documents={name:read(native/(name+'.ROOT-SIGNED.json'))for name in('context','grades','subset')}
    directory=state/'roles'/'execution-preparations';directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    if directory.is_symlink()or directory.resolve()!=directory:raise ValueError('canonical private preparation journal')
    path=directory/(epoch+'.ROOT-SIGNED.json')
    if path.exists():
        original=read(path);prior=preparation(original,authority,now=job['created_at'])
        expected=derive_preparation(release_envelope,job,public,documents,authority,created_at=prior['created_at'])
        if prior!=expected:raise ValueError('immutable original epoch execution preparation changed')
        return original
    value=derive_preparation(release_envelope,job,public,documents,authority,created_at=job['created_at'])
    envelope=controller.signed(value);preparation(envelope,authority,now=job['created_at'])
    encoded=__import__('json').dumps(envelope,sort_keys=True,separators=(',',':')).encode()
    fd,temp=tempfile.mkstemp(prefix='preparation-',suffix='.owned',dir=directory)
    try:
        with os.fdopen(fd,'wb')as stream:stream.write(encoded);stream.flush();os.fsync(stream.fileno())
        try:os.link(temp,path)
        except FileExistsError:
            original=read(path);prior=preparation(original,authority,now=job['created_at'])
            if prior!=value:raise ValueError('concurrent epoch preparation differs')
            return original
    finally:Path(temp).unlink(missing_ok=True)
    return envelope
