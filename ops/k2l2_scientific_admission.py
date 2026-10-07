"""Default-off scientific admission for a NEW miner-bound K2/L2 source.

Authenticates ROOT qualification attestations; does not run a GPU, sign policy,
change a controller, or reinterpret an ordinary orchestration-only approval.
"""
from pathlib import PurePosixPath
from ops import durable_audit_services as guards

POLICY = 'durable-pinned-k2l2-learner-service-v2'
SOURCE = 'k2l2-miner-bound-scientific-source-approval-v1'
QUALIFICATION = 'k2l2-miner-bound-training-qualification-approval-v1'
GPU_RESULT = 'k2l2-miner-bound-sm90-qualification-result-v1'
ACK = 'ROOT-k2l2-scientific-qualification-metadata-readback-ACK-v1'
SAMPLER = 'forced-inverse-cdf-prefill-miner-bound-v5'
REQUIRED_RUNTIME_ADDITIONS = {'subnet/sampling_uniqueness.py', 'subnet/trajectory_identity.py'}
CONTROLS = {'honest_four_rollout_batch', 'repacked_duplicate_rejected',
            'wrong_miner_rejected', 'out_of_range_nonce_rejected',
            'reused_attempt_rejected', 'changed_tokens_rejected',
            'legacy_draws_unchanged', 'all_four_native_outcomes',
            'two_disjoint_pairs_per_task', 'equal_total_task_weight',
            'finite_loss_full_gradient_coverage', 'parent_optimizer_restored',
            'disposable_update_step_advanced'}


def inventory(files):
    if type(files) is not dict or not files:
        raise ValueError('nonempty exact source inventory')
    for name, digest in files.items():
        path = PurePosixPath(name)
        if (type(name) is not str or path.is_absolute() or '..' in path.parts or
            str(path) != name or type(digest) is not str or len(digest) != 64 or
            any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('canonical relative source path and SHA256')
    return files


def contract(config):
    if (type(config.get('K')) is not int or config['K'] != 2 or
        type(config.get('L')) is not int or config['L'] != 2 or
        type(config.get('commitment_max_batches')) is not int or
        config['commitment_max_batches'] != 3):
        raise ValueError('explicit K2/L2 and three-task-batch contract')
    sampling = config.get('sampling_policy')
    if (type(sampling) is not dict or sampling.get('version') != SAMPLER or
        type(sampling.get('max_attempts')) is not int or sampling['max_attempts'] != 1000 or
        sampling.get('support_adjudication') != 'exact-cached-replay-v1' or
        type(sampling.get('calibration')) is not dict or not sampling['calibration']):
        raise ValueError('new miner-bound 1000-attempt calibrated-support sampler')
    if (config.get('token_artifact_policy') is not None or
        config.get('probability_artifact_policy') != {'version': 'selected-token-logprobs-v1'} or
        config.get('submission_transport_policy') != 'small-commitment-pairs-v2' or
        config.get('training_input_policy') != 'committed-unaudited-training-v1'):
        raise ValueError('preserve selected-token transport and unaudited training')
    return {key: config.get(key) for key in ('K', 'L', 'commitment_max_batches',
        'sampling_policy', 'probability_artifact_policy', 'token_artifact_policy',
        'submission_transport_policy', 'training_input_policy')}


def validate(source, qualification, config, source_sha256, authority, verify_row):
    """verify_row authenticates exact original row bytes/signature/payload hash.

    Called only by the NEW policy branch, alongside existing file-membership,
    execution identity, reward activation, translation and optimizer-state guards.
    """
    base = {'version', 'approved', 'source_sha256', 'optimizer_reset',
            'historical_relabel', 'full_source_files', 'runtime_source_files', 'evidence'}
    fields = base | {'predecessor_source_approval', 'runtime_execution_files',
                     'runtime_changes', 'contract_sha256', 'predecessor_learner_policy'}
    if (type(source) is not dict or set(source) != fields or source['version'] != SOURCE or
        source['approved'] is not True or source['source_sha256'] != source_sha256 or
        source['optimizer_reset'] is not False or source['historical_relabel'] is not False):
        raise ValueError('NEW scientific source approval; no ordinary approval reuse')
    full = inventory(source['full_source_files']); runtime = inventory(source['runtime_source_files'])
    declared = source['runtime_execution_files']
    if (type(declared) is not list or declared != sorted(runtime) or
        any(full.get(name) != value for name, value in runtime.items())):
        raise ValueError('derived declared exact runtime execution closure')
    old = verify_row(source['predecessor_source_approval'], authority)
    if (old.get('version') != 'ordinary-orchestration-only-source-approval-v1' or
        old.get('approved') is not True or old.get('optimizer_reset') is not False or
        old.get('historical_relabel') is not False or old.get('source_sha256') == source_sha256):
        raise ValueError('authentic distinct ordinary predecessor')
    old_runtime = inventory(old['runtime_source_files'])
    if len(old_runtime) != 177:
        raise ValueError('historical ordinary 177 closure remains strict')
    # Membership comes from the exact authentic predecessor, not a guessed count.
    if set(runtime) != set(old_runtime) | REQUIRED_RUNTIME_ADDITIONS:
        raise ValueError('exact two reviewed scientific runtime additions')
    changes = {name: {'before': old_runtime.get(name), 'after': value}
               for name, value in runtime.items() if old_runtime.get(name) != value}
    if source['runtime_changes'] != changes or not REQUIRED_RUNTIME_ADDITIONS <= set(changes):
        raise ValueError('exact declared scientific source delta')
    predecessor_policy = verify_row(source['predecessor_learner_policy'], authority)
    if (predecessor_policy.get('version') != 'durable-pinned-learner-service-v1' or
        predecessor_policy.get('source_sha256') != old['source_sha256'] or
        predecessor_policy.get('source_approval') != source['predecessor_source_approval']):
        raise ValueError('original signed learner policy and source approval')
    original_config = predecessor_policy['config']
    if guards.file_hash(original_config['path']) != original_config['file_sha256']:
        raise ValueError('exact original learner configuration bytes')
    original_admission = guards.read(original_config['path'])['persistent_training_admission']
    new_admission = config['persistent_training_admission']
    for key in ('parameters', 'parameters_sha256', 'genesis_round',
                'genesis_checkpoint', 'genesis_sha256'):
        if key not in original_admission or new_admission.get(key) != original_admission[key]:
            raise ValueError('original parameter inventory and optimizer genesis preserved')
    bound_contract = guards.digest(contract(config))
    if source['contract_sha256'] != bound_contract:
        raise ValueError('new source binds exact K2/L2 sampler contract')
    qfields = {'version', 'approved', 'candidate_source_sha256', 'translation_path',
               'translation_file_sha256', 'original_gpu_report', 'durable_readback_ack'}
    if (type(qualification) is not dict or set(qualification) != qfields or
        qualification['version'] != QUALIFICATION or qualification['approved'] is not True or
        qualification['candidate_source_sha256'] != source_sha256):
        raise ValueError('NEW scientific GPU qualification approval')
    report = verify_row(qualification['original_gpu_report'], authority)
    rfields = {'version', 'candidate_source_sha256', 'runtime_inventory_sha256',
               'contract_sha256', 'parent_checkpoint', 'parent_optimizer_sha256',
               'original_terminal_sha256', 'actual_original_wait0', 'controls',
               'optimizer_reset', 'historical_relabel', 'qualification_class', 'hardware',
               'runtime_profile', 'model_exported', 'optimizer_exported', 'model_export_destination',
               'model_uploaded', 'model_promoted', 'original_scope',
               'original_result', 'original_terminal'}
    if (type(report) is not dict or set(report) != rfields or report['version'] != GPU_RESULT or
        report['candidate_source_sha256'] != source_sha256 or
        report['runtime_inventory_sha256'] != guards.digest(runtime) or
        report['contract_sha256'] != bound_contract or report['actual_original_wait0'] is not True or
        report['optimizer_reset'] is not False or report['historical_relabel'] is not False or
        type(report['controls']) is not dict or set(report['controls']) != CONTROLS or
        any(value is not True for value in report['controls'].values())):
        raise ValueError('fresh authentic SM90 scientific qualification closure')
    scope = verify_row(report['original_scope'], authority)
    if (scope.get('version') != 'K2L2-miner-bound-v5-CP33-realGPU-smoke-v2' or
        scope.get('steps') != 1 or type(scope.get('steps')) is not int or
        scope.get('objective') != 'unchanged-task-normalized-pairwise' or
        scope.get('optimizer_disposition') != 'isolated-smoke-no-continuation-no-promotion' or
        scope.get('source_sha256') != source_sha256 or
        scope.get('candidate_source_bundle_sha256') != source_sha256 or
        scope.get('full_source_files') != full or
        scope.get('full_source_inventory_sha256') != guards.digest(full) or
        scope.get('checkpoint_id') != report['parent_checkpoint'] or
        scope.get('parent_descriptor_sha256') != report['parent_optimizer_sha256'] or
        scope.get('production_mutations') is not False or scope.get('network_operations') is not False):
        raise ValueError('authentic original scientific scope and source/parent inventory')
    originals = {}
    for key in ('original_result', 'original_terminal'):
        row = report[key]
        if (type(row) is not dict or set(row) != {'path','file_sha256'} or
            guards.file_hash(row['path']) != row['file_sha256']):
            raise ValueError('exact actual original result/terminal bytes')
        originals[key] = guards.read(row['path'])
    raw = originals['original_result']; terminal = originals['original_terminal']
    if (raw.get('version') != 'K2L2-miner-bound-v5-CP33-realGPU-smoke-v2' or
        raw.get('scope_sha256') != guards.digest(scope) or
        raw.get('new_source_sha256') != source_sha256 or
        raw.get('parent_checkpoint') != report['parent_checkpoint'] or
        raw.get('parent_descriptor_sha256') != report['parent_optimizer_sha256'] or
        raw.get('parent_step') != scope.get('parent_step') or raw.get('output_step') != scope.get('output_step') or
        type(raw.get('parent_step')) is not int or type(raw.get('output_step')) is not int or
        raw['output_step'] != raw['parent_step'] + 1 or
        raw.get('actual_native_labels') != ['positive','positive','negative','negative'] or
        raw.get('full_four_rollout_verification') is not True or
        raw.get('probability_artifact_policy') != {'version':'selected-token-logprobs-v1'} or
        raw.get('selected_token_probability_transport_bound') is not True or
        raw.get('model_state_exported') is not True or
        raw.get('model_export_destination') != 'local-isolated-smoke' or
        raw.get('model_state_uploaded') is not False or
        raw.get('optimizer_state_exported') is not False or raw.get('optimizer_state_durable') is not False or
        raw.get('production_pointer_writes') is not False or raw.get('network_operations') is not False or
        raw.get('heldout_gain_claimed') is not False or raw.get('complete') is not False or
        terminal.get('exit_code') != 0 or type(terminal.get('exit_code')) is not int or
        terminal.get('actual_child_wait_completed') is not True or terminal.get('timed_out') is not False or
        terminal.get('scope_sha256') != guards.digest(scope) or terminal.get('production_mutations') is not False or
        report['original_terminal_sha256'] != report['original_terminal']['file_sha256']):
        raise ValueError('actual disposable four-rollout one-step result and genuine original terminal')
    hardware = report['hardware']
    if (report['qualification_class'] != 'disposable-sm90-one-step-local-model-no-state-promotion-v1' or
        report['runtime_profile'] != 'cuda-fp32-eager-sm90-v1' or
        report['model_exported'] is not True or report['optimizer_exported'] is not False or
        report['model_export_destination'] != 'local-isolated-smoke' or
        report['model_uploaded'] is not False or report['model_promoted'] is not False or
        type(hardware) is not dict or set(hardware) != {'name', 'uuid', 'sm'} or
        type(hardware['name']) is not str or not any(name in hardware['name'] for name in ('H100','H200')) or
        type(hardware['uuid']) is not str or not hardware['uuid'].startswith('GPU-') or
        hardware['sm'] != [9,0] or
        type(scope.get('GPU_inventory')) is not list or len(scope['GPU_inventory']) != 1 or
        scope['GPU_inventory'][0].get('name') != hardware['name'] or
        scope['GPU_inventory'][0].get('uuid') != hardware['uuid']):
        raise ValueError('explicit actual SM90 hardware; disposable qualification is not state publication')
    for key in ('parent_checkpoint', 'parent_optimizer_sha256', 'original_terminal_sha256'):
        value = report[key]
        if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('exact original parent and terminal evidence hashes')
    ack = verify_row(qualification['durable_readback_ack'], authority)
    if (type(ack) is not dict or set(ack) != {'version', 'candidate_source_sha256',
        'report_payload_sha256', 'original_terminal_sha256', 'full_independent_metadata_readback',
        'actual_original_wait0', 'original_scope_payload_sha256', 'original_result_file_sha256'} or ack['version'] != ACK or
        ack['candidate_source_sha256'] != source_sha256 or
        ack['report_payload_sha256'] != guards.digest(report) or
        ack['original_scope_payload_sha256'] != guards.digest(scope) or
        ack['original_result_file_sha256'] != report['original_result']['file_sha256'] or
        ack['original_terminal_sha256'] != report['original_terminal_sha256'] or
        ack['full_independent_metadata_readback'] is not True or ack['actual_original_wait0'] is not True):
        raise ValueError('independent full qualification metadata readback ACK')
    return {'source': source_sha256, 'runtime_count': len(runtime),
            'runtime_inventory_sha256': guards.digest(runtime), 'contract_sha256': bound_contract}
