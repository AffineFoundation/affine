"""Read-only, allowlisted evidence for finalized training epochs.

The caller supplies original signed completion/manifest/training documents.
This module never signs, dispatches, mutates state, or fetches proof arrays.
"""
import base64
import hashlib
import json
import math
import re
import time
from pathlib import Path

from nacl.signing import VerifyKey

AUTHORITY = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
VERSION = 'finalized-training-evidence-v1'
EPOCH = re.compile(r'nonpayable-live-reward-math-v1--[0-9]+-([0-9]+)\Z')
HASH = re.compile(r'[0-9a-f]{64}\Z')
MAX_JSON_BYTES = 256 * 1024**2


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def authenticated(document, authority):
    if (type(document) is not dict or set(document) != {'payload', 'signature', 'signer'}
            or document['signer'] != authority):
        raise ValueError('evidence_signature_shape')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']),
        base64.b64decode(document['signature'], validate=True))
    return document['payload']


def read_regular_bytes(path):
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        raise FileNotFoundError(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('bounded_regular_evidence_file')
    return path.read_bytes()


def read_json(path):
    return json.loads(read_regular_bytes(path))


def available(payload, provenance):
    return {'status': 'available', 'payload': payload, 'provenance': provenance}


def unavailable(reason):
    return {'status': 'unavailable', 'reason': reason}


def fields(document, numeric=(), text=(), boolean=()):
    """Only named primitive fields cross the public boundary."""
    result = {}
    for name in numeric:
        value = document.get(name)
        if type(value) in (int, float) and math.isfinite(value):
            result[name] = value
    for name in text:
        value = document.get(name)
        if type(value) is str and len(value) <= 256 and not any(c in value for c in '\r\n'):
            result[name] = value
    for name in boolean:
        if type(document.get(name)) is bool:
            result[name] = document[name]
    return result


def finalized(epoch_id, record, authority):
    match = EPOCH.fullmatch(epoch_id)
    if not match or int(match[1]) < 14:
        raise ValueError('finalized_epoch14_onward_only')
    manifest = authenticated(record['manifest_envelope'], authority)
    completion = authenticated(record['completion_envelope'], authority)
    if (manifest['epoch'] != epoch_id or completion['epoch'] != epoch_id
            or completion['round'] != int(match[1])
            or completion['checkpoint'] != manifest['checkpoint']['id']
            or not HASH.fullmatch(completion['next_checkpoint'])
            or type(completion.get('completed_at')) not in (int, float)
            or not math.isfinite(completion['completed_at'])
            or not manifest['deadline'] <= completion['completed_at'] <= time.time()):
        raise ValueError('signed_finalized_epoch_binding')
    return manifest, completion


def hyperparameters(value):
    result = fields(value, numeric=('lr', 'eps', 'max_grad_norm', 'preference_beta', 'weight_decay'))
    betas = value.get('betas')
    if type(betas) is list and len(betas) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in betas):
        result['betas'] = betas
    return result


def update_projection(update):
    result = fields(update, numeric=(
        'epoch_optimizer_step', 'global_optimizer_step', 'steps', 'loss', 'preference_loss',
        'positive_nll', 'positive_nll_weight', 'positive_mean_logprob', 'negative_mean_logprob',
        'gradient_norm_before_clip', 'gradient_tasks', 'gradient_pairs', 'gradient_tensors',
        'unique_tasks', 'unique_verified_pairs', 'cumulative_unique_gradient_tasks',
        'gpu_peak_allocated_bytes', 'gpu_peak_reserved_bytes'),
        text=('gradient_accumulation', 'gradient_accumulation_dtype', 'optimizer_lifecycle',
              'task_weight_rule', 'reference_scope', 'training_policy', 'input_checkpoint'),
        boolean=('full_model_finetune',))
    result['hyperparameters'] = hyperparameters(update.get('hyperparameters', {}))
    precision = update.get('precision', {})
    result['precision'] = fields(precision,
        numeric=('bf16_changed_elements', 'master_changed_elements', 'optimizer_step'),
        text=('inference_dtype', 'master_dtype', 'optimizer_state_dtype'))
    result['precision']['parameters'] = [fields(row,
        numeric=('elements', 'bf16_changed_elements', 'master_changed_elements', 'master_delta_l2', 'master_delta_max_abs'),
        text=('name',)) for row in precision.get('parameters', [])]
    result['pairs'] = [fields(row,
        numeric=('pair_index', 'task_index', 'gradient_weight', 'beta', 'loss', 'margin_before',
                 'reference_margin', 'positive_nll', 'positive_nll_weight', 'preference_loss',
                 'positive_mean_logprob', 'negative_mean_logprob'),
        text=('pair_sha256', 'task_sha256')) for row in update.get('pairs', [])]
    result['metric_availability'] = {
        name: 'recorded' if name in result else 'not_retained'
        for name in ('loss', 'preference_loss', 'positive_nll', 'positive_nll_weight',
                     'gradient_norm_before_clip', 'gradient_accumulation_dtype')}
    result['metric_availability']['learning_rate'] = (
        'recorded' if 'lr' in result['hyperparameters'] else 'not_retained')
    result['metric_availability']['parameter_update_precision'] = (
        'recorded' if result['precision']['parameters'] else 'not_retained')
    norm = result.get('gradient_norm_before_clip')
    maximum = result['hyperparameters'].get('max_grad_norm')
    if norm is not None and maximum is not None and norm > 0:
        result['derived_clip_scale_upper_bound'] = min(1., maximum/norm)
    return result


def token_array(value):
    if (type(value) is not list or len(value) > 1_000_000
            or any(type(x) is not int or not 0 <= x < 2**31 for x in value)):
        raise ValueError('bounded_token_ids')
    return value


def rollout_projection(rollout):
    result = fields(rollout, numeric=('index', 'seed', 'env_seed', 'sample_index', 'reward'),
                    text=('classification', 'env_id', 'environment_version', 'task_hash'))
    result['sampling'] = fields(rollout.get('sampling', {}), numeric=('attempt',),
                                text=('binding_sha256', 'version'))
    result['turns'] = []
    for turn in rollout['turns']:
        row = fields(turn, numeric=('reward',), text=('classification',), boolean=('done',))
        row['prompt'] = token_array(turn['prompt'])
        row['output'] = token_array(turn['output'])
        if type(turn.get('text')) is str and len(turn['text']) <= 4*1024**2:
            row['text'] = turn['text']
        # Proofs, URLs, full-vocabulary logits, arbitrary observations and paths
        # are intentionally absent. Prompt/output tokens remain complete.
        result['turns'].append(row)
    return result


def job_report(state, metrics, epoch_id, manifest, completion, authority):
    job_id = metrics['remote_job_id']
    if (type(job_id) is not str or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,199}', job_id)):
        raise ValueError('exact_authoritative_training_job')
    job_document = read_json(state/'roles'/(job_id+'-job.json'))
    job = authenticated(job_document, authority)
    admitted_manifest = authenticated(job['manifest'], authority)
    report = read_json(state/'roles'/(job_id+'-report.json'))
    if (job['job_id'] != job_id or job['role'] != 'train'
            or admitted_manifest['epoch'] != epoch_id
            or admitted_manifest['checkpoint']['id'] != manifest['checkpoint']['id']
            or digest(job) != metrics['original_job_sha256']
            or report.get('success') is not True or report.get('role') != 'train'
            or report['job_id'] != job_id or report['job_sha256'] != digest(job)
            or report['epoch'] != epoch_id or report['checkpoint'] != manifest['checkpoint']['id']
            or report['new_checkpoint']['id'] != completion['next_checkpoint']
            or report['source_files'] != job['source_files']
            or report['runtime_versions'] != job['runtime_versions']
            or report['training']['updates'] != metrics['updates']
            or report['training']['persistent_diagnostics'] != metrics['persistent_diagnostics']
            or not job['created_at'] <= report['completed_at'] <= completion['completed_at']):
        raise ValueError('authenticated_training_report_binding')
    return job_document, job, admitted_manifest, report


def inputs_projection(job, report, authority):
    by_admission = {}
    for submission in job['submissions']:
        envelope = submission['learner_admission']
        body = authenticated(envelope, authority)
        key = digest(envelope)
        if key in by_admission or body['document_sha256'] != submission['sha256'] or body['document_size'] != submission['size']:
            raise ValueError('unique_admitted_training_inputs')
        by_admission[key] = body
    pairs = [pair for update in report['training']['updates'] for pair in update.get('pairs', [])]
    trained_tasks = {pair['task_sha256'] for pair in pairs}
    pair_tasks = {}
    for pair in pairs:
        previous = pair_tasks.setdefault(pair['pair_sha256'], pair['task_sha256'])
        if previous != pair['task_sha256']:
            raise ValueError('one_task_per_gradient_pair')
    matched_pairs = set()
    selected = []
    seen = set()
    for row in report['training_admissions']:
        key = row['learner_admission_sha256']
        admission = by_admission.get(key)
        batch = row['claimed_batch']
        if admission is None or key in seen:
            raise ValueError('exact_reported_admission_set')
        seen.add(key)
        for field in ('batch_sha256', 'document_sha256', 'document_size', 'miner_identity', 'slot', 'epoch'):
            if row[field] != admission[field]:
                raise ValueError('admitted_batch_identity')
        if digest(batch) != admission['batch_sha256'] or batch['checkpoint'] != admission['checkpoint'] or batch['epoch'] != admission['epoch']:
            raise ValueError('signed_batch_content_binding')
        hashes = {r['task_hash'] for r in batch['rollouts']}
        if len(hashes) != 1:
            raise ValueError('single_task_batch_identity')
        task_identity = {'env_id': batch['env_id'], 'index': batch['index'], 'task_hash': next(iter(hashes))}
        task_sha = digest(task_identity)
        if task_sha not in trained_tasks:
            continue
        used_rollouts = set()
        bound_pairs = []
        for positive_index, positive in enumerate(batch['rollouts']):
            if positive.get('classification') != 'positive':
                continue
            for negative_index, negative in enumerate(batch['rollouts']):
                if negative.get('classification') != 'negative':
                    continue
                pair_sha = digest(dict(env_id=batch['env_id'], index=batch['index'],
                                       positive=positive, negative=negative))
                if pair_sha not in pair_tasks:
                    continue
                if pair_tasks[pair_sha] != task_sha:
                    raise ValueError('pair_content_task_binding')
                if pair_sha in matched_pairs:
                    raise ValueError('ambiguous_duplicate_pair_attribution')
                matched_pairs.add(pair_sha)
                used_rollouts.update((positive_index, negative_index))
                bound_pairs.append({'pair_sha256': pair_sha, 'positive_rollout_index': positive_index,
                                    'negative_rollout_index': negative_index})
        if not bound_pairs:
            continue
        selected.append(dict(task_identity, task_sha256=task_sha,
            miner_identity=admission['miner_identity'], slot=admission['slot'],
            batch_sha256=admission['batch_sha256'], document_sha256=admission['document_sha256'],
            learner_admission_sha256=key, sampling_assurance=admission['assurance'],
            admitted_rollout_count=len(batch['rollouts']), actual_gradient_pairs=bound_pairs,
            rollouts=[dict(rollout_projection(rollout), admitted_rollout_index=index)
                      for index, rollout in enumerate(batch['rollouts']) if index in used_rollouts]))
    if (seen != set(by_admission) or {row['task_sha256'] for row in selected} != trained_tasks
            or matched_pairs != set(pair_tasks)):
        raise ValueError('all_actual_gradient_tasks_have_bound_rollouts')
    return {'admitted_batch_count': len(by_admission), 'trained_batch_count': len(selected),
            'actual_gradient_task_count': len(trained_tasks), 'actual_unique_gradient_pair_count': len(pair_tasks),
            'actual_gradient_pair_exposure_count': len(pairs),
            'rollout_count': sum(len(row['rollouts']) for row in selected),
            'full_vocabulary_probabilities_included': False, 'batches': selected}


def exclusion_projection(state, epoch_id, admitted_manifest, authority, population_envelope=None):
    receipt = admitted_manifest.get('native_training_eligibility_receipt', {})
    directory = state/'native-outcome-eligibility'/epoch_id
    grade_documents = []
    provenance = {}
    counts = {}
    population_hash = None
    if receipt.get('pool_sha256'):
        pool_document = read_json(directory/'pool.ROOT-SIGNED.json')
        pool = authenticated(pool_document, authority)
        result_document = read_json(directory/'result.ROOT-SIGNED.json')
        result = authenticated(result_document, authority)
        if digest(pool_document) != receipt['pool_sha256'] or digest(result_document) != receipt['result_sha256']:
            raise ValueError('native_pool_result_binding')
        population_hash = pool['original_population_file_sha256']
        provenance.update(pool_sha256=digest(pool_document), result_sha256=digest(result_document))
        counts.update(native_checked_count=result['checked_count'], native_accepted_count=result['accepted_count'])
        for index, wave in enumerate(result.get('native_waves', [])):
            document = read_json(directory/'waves'/f'{index:04d}'/'grades.ROOT-SIGNED.json')
            grade = authenticated(document, authority)
            if digest(document) != wave['grades_sha256']:
                raise ValueError('signed_native_wave_binding')
            grade_documents.append(grade)
    elif receipt.get('context_sha256'):
        context_document = read_json(directory/'context.ROOT-SIGNED.json')
        context = authenticated(context_document, authority)
        grade_document = read_json(directory/'grades.ROOT-SIGNED.json')
        grade = authenticated(grade_document, authority)
        subset_document = read_json(directory/'subset.ROOT-SIGNED.json')
        subset = authenticated(subset_document, authority)
        if (digest(context_document) != receipt['context_sha256']
                or digest(grade_document) != receipt['grades_sha256']
                or digest(subset_document) != receipt['subset_sha256']):
            raise ValueError('legacy_native_context_grade_subset_binding')
        population_hash = context['original_population_file_sha256']
        provenance.update(context_sha256=digest(context_document), grades_sha256=digest(grade_document),
                          subset_sha256=digest(subset_document))
        counts.update(native_checked_count=subset['accepted_count']+subset['excluded_count'],
                      native_accepted_count=subset['accepted_count'])
        grade_documents.append(grade)
    elif population_envelope is None:
        return unavailable('signed_population_receipt_unavailable')
    if population_hash is not None:
        path = state/(epoch_id+'-learner-population.json')
        raw_population = read_regular_bytes(path)
        population_document = json.loads(raw_population)
        if hashlib.sha256(raw_population).hexdigest() != population_hash:
            raise ValueError('authenticated_collection_exclusions')
        population = population_document['population']
        provenance['population_file_sha256'] = population_hash
    else:
        population = authenticated(population_envelope, authority)
        if (population['epoch'] != epoch_id
                or population['checkpoint'] != admitted_manifest['checkpoint']['id']):
            raise ValueError('signed_public_population_epoch_binding')
        provenance['population_envelope_sha256'] = digest(population_envelope)
    known = {'duplicate_task', 'structural_ineligible', 'temporary_exclusion', 'blacklisted', 'not_registered', 'over_limit'}
    exclusions = [{'document_sha256': row['document_sha256'],
                   'reason': row['reason'] if row.get('reason') in known else 'unrecognized_reason'}
                  for row in population.get('exclusions', []) if HASH.fullmatch(row.get('document_sha256', ''))]
    native = []
    rejected_pairs = []
    for grade in grade_documents:
        for row in grade.get('rows', []):
            if row.get('status') == 'accepted_native_labels':
                continue
            projected = fields(row, text=('pair_sha256',))
            status = row.get('status')
            projected['status'] = status if type(status) is str and re.fullmatch('[a-z_]{1,80}', status) else 'unrecognized_status'
            projected['grades'] = []
            for item in row.get('grades', []):
                value = fields(item, numeric=('native_score', 'output_tokens', 'approved_output_cap'),
                    text=('claim', 'output_sha256', 'decoded_reply_sha256', 'outcome_policy'),
                    boolean=('label_matches', 'complete_answer', 'non_eos_cap', 'terminal_framing_valid',
                             'pair_terminal_framing_valid', 'submitted_text_matches_decoded'))
                reason = item.get('reason')
                value['reason'] = reason if reason is None or (type(reason) is str and re.fullmatch('[a-z_]{1,100}', reason)) else 'unrecognized_reason'
                projected['grades'].append(value)
            rejected_pairs.append(projected)
        for row in grade.get('document_decisions', []):
            if row.get('accepted') is False:
                native.append(fields(row, text=('batch_sha256', 'document_sha256', 'learner_admission_sha256'), boolean=('accepted',)))
    payload = dict(legacy_reward_exclusions=exclusions,
        legacy_reward_exclusions_are_not_training_exclusions=True,
        population_counts=fields(population, numeric=('committed_count', 'eligible_count', 'training_count')),
        native_rejected_documents=native, native_rejected_pairs=rejected_pairs,
        native_grading_evidence_available=bool(grade_documents), sampling_assurance='unaudited',
        proof_verification_performed=False, **counts)
    return available(payload, provenance)


def finite_array(value, limit=65536):
    if (type(value) is not list or len(value) > limit or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError('bounded_finite_diagnostic_array')
    return value


def diagnostics_projection(source):
    result = fields(source, numeric=(
        'global_optimizer_step_before', 'global_optimizer_step_after', 'optimizer_steps', 'task_count', 'pair_count'),
        boolean=('complete', 'master_state_updated', 'inference_tensors_changed_during_updates'))
    for name in ('training_pair_margin_before', 'training_pair_margin_after', 'training_pair_margin_delta'):
        if name in source:
            result[name] = finite_array(source[name])
    phase = source.get('phase_seconds', {})
    result['phase_seconds'] = fields(phase, numeric=(
        'checkpoint_save', 'optimizer_initialization', 'post_update_forward', 'reference_forward'),
        boolean=('GPU_timings_synchronized',))
    for name in ('CPU_optimizer_by_step', 'gradient_and_clip_by_step'):
        if name in phase:
            result['phase_seconds'][name] = finite_array(phase[name], 32)
    result['transport_phase_seconds'] = fields(source.get('transport_phase_seconds', {}),
        numeric=('local_state_save', 'parent_cache_and_restore_total', 'parent_cache_validation_and_admission',
                 'parent_state_restore', 'state_transfer_concurrency', 'training_and_checkpoint'),
        boolean=('parent_restore_performed',))
    result['effective_hyperparameters'] = hyperparameters(source.get('effective_hyperparameters', {}))
    nll = source.get('positive_nll_components')
    if nll is not None:
        component_fields = ('pair_index', 'loss', 'margin', 'negative_mean_logprob',
                            'positive_mean_logprob', 'positive_nll', 'preference_loss')
        result['positive_nll_components'] = fields(nll,
            numeric=('beta', 'extra_model_forward_passes', 'learning_rate', 'positive_nll_weight'),
            text=('reference_scope', 'version'), boolean=('tail_mask_applied',))
        for name in ('before', 'after'):
            rows = nll.get(name, [])
            if type(rows) is not list or len(rows) > 65536:
                raise ValueError('bounded_pair_diagnostics')
            result['positive_nll_components'][name] = [fields(row, numeric=component_fields) for row in rows]
        for name in ('weighted_before', 'weighted_after'):
            result['positive_nll_components'][name] = fields(nll.get(name, {}), numeric=component_fields)
    return result



PROGRESS_LINE = re.compile(
    r'^(Loading weights|Writing model shards):\s*(\d{1,3})%\|[^|]*\|\s*'
    r'(\d+)/(\d+)\s*\[([0-9:]+)<([0-9?:]+),')
WARNING_MARKER = re.compile(r'\b(UserWarning|FutureWarning|DeprecationWarning|RuntimeWarning|WARNING)\b')


def progress_seconds(value):
    parts = value.split(':')
    if (len(parts) not in (2, 3) or any(not x.isdigit() for x in parts)
            or any(int(x) >= 60 for x in parts[1:]) or int(parts[0]) > 999):
        return None
    seconds = 0
    for part in parts:
        seconds = seconds*60+int(part)
    return seconds


def progress_projection(line):
    match = PROGRESS_LINE.match(line)
    if match is None:
        return None
    label, percent, completed, total, elapsed, remaining = match.groups()
    percent, completed, total = int(percent), int(completed), int(total)
    elapsed_seconds = progress_seconds(elapsed)
    if not (0 <= percent <= 100 and 0 <= completed <= total <= 1_000_000 and total > 0 and elapsed_seconds is not None):
        return None
    result = {'event_kind': 'runtime_progress_observation',
        'phase': 'loading_model_weights' if label == 'Loading weights' else 'writing_model_shards',
        'reported_percent': percent, 'completed_items': completed, 'total_items': total,
        'elapsed_seconds': elapsed_seconds,
        'progress_reports_completion': completed == total and percent == 100}
    remaining_seconds = progress_seconds(remaining)
    if remaining_seconds is not None:
        result['estimated_remaining_seconds'] = remaining_seconds
    return result

def trainer_log_projection(record, epoch_id, job, raw_job_sha256):
    if record is None:
        return unavailable('original_runtime_log_not_in_local_evidence_cache')
    raw = record['raw']
    receipt = record['receipt']
    if (type(raw) is not bytes or len(raw) > 1024**2
            or receipt.get('epoch') != epoch_id or receipt.get('job_id') != job['job_id']
            or receipt.get('raw_sha256') != hashlib.sha256(raw).hexdigest()
            or receipt.get('raw_size') != len(raw) or receipt.get('job_raw_sha256') != raw_job_sha256
            or receipt.get('original_job_bytes_match') is not True):
        raise ValueError('retained_runtime_log_identity')
    events = []
    line_counts = {name: 0 for name in ('structured_events', 'progress_observations', 'recognized_warning_markers',
                  'blank', 'unrecognized_text', 'unrecognized_json', 'oversized')}
    lines = raw.decode('utf-8', errors='replace').splitlines()
    for number, line in enumerate(lines, 1):
        if len(line) > 65536:
            line_counts['oversized'] += 1
            continue
        if not line.strip():
            line_counts['blank'] += 1
            continue
        progress = progress_projection(line)
        if progress is not None:
            events.append(dict(progress, source_line_number=number))
            line_counts['progress_observations'] += 1
            continue
        try:
            value = json.loads(line)
        except ValueError:
            warning = WARNING_MARKER.search(line)
            if warning is not None:
                events.append({'event_kind': 'recognized_warning_marker',
                    'marker': warning.group(1), 'source_line_number': number, 'message_text_included': False})
                line_counts['recognized_warning_markers'] += 1
            else:
                line_counts['unrecognized_text'] += 1
            continue
        if type(value) is not dict:
            line_counts['unrecognized_json'] += 1
            continue
        row = None
        if value.get('version') == 'cpu-selection-peer-execution-v1':
            row = fields(value, text=('version', 'admission_sha256', 'original_job_sha256',
                'original_manifest_sha256', 'peer_entry_sha256', 'scientific_source_files_sha256'),
                boolean=('backend_execution_allowed', 'proof_reverification', 'scientific_source_unchanged'))
            row['evidence_scope'] = 'peer_admission_metadata'
            result = value.get('backend_result', {})
            if result.get('job_id') == job['job_id']:
                row['backend_result'] = fields(result, boolean=('success',))
        elif value.get('version') == 'reference-boundary-CUDA-cache-admission-v1':
            row = fields(value, text=('version',), numeric=('allocated_bytes', 'allocated_peak_bytes',
                'backward_reserve_bytes', 'extra_buffer_bytes', 'free_bytes', 'inactive_split_bytes',
                'required_free_bytes', 'reserved_bytes', 'reserved_peak_bytes', 'total_bytes'))
        elif type(value.get('postload_input_page_advice')) is dict:
            source = value['postload_input_page_advice']
            if source.get('version') == 'owned-authenticated-input-postload-page-advice-v1':
                row = fields(source, text=('version', 'checkpoint'), numeric=('advised_bytes', 'bytes_deleted', 'files'),
                    boolean=('optimizer_files_touched', 'resource_guard_changed'))
                for name in ('before', 'after'):
                    row[name] = fields(source.get(name, {}),
                        numeric=('available_ram_bytes', 'cgroup_bytes', 'cgroup_current_bytes'))
        elif type(value.get('CPU_representative_bootstrap_reload')) is dict:
            source = value['CPU_representative_bootstrap_reload']
            if source.get('version') == 'authenticated-representative-bootstrap-reload-v1':
                row = fields(source, text=('version', 'sha256'), boolean=('evicted', 'original_loader_unchanged',
                    'other_modules_evicted', 'scientific_source_modified'))
        if row is not None:
            for name in list(row):
                if (name.endswith('sha256') or name == 'checkpoint') and not HASH.fullmatch(str(row[name])):
                    del row[name]
            events.append(dict(row, source_line_number=number))
            line_counts['structured_events'] += 1
        else:
            line_counts['unrecognized_json'] += 1
    return available({'structured_events': events, 'source_line_count': len(lines),
        'omitted_line_count': len(lines)-len(events), 'line_classification_counts': line_counts,
        'warning_detection_scope': 'recognized_warning_markers_only', 'raw_text_included': False,
        'raw_log_signed_by_training_authority': False},
        {'raw_log_sha256': receipt['raw_sha256'], 'raw_log_size': receipt['raw_size'],
         'original_job_raw_sha256': raw_job_sha256, 'job_id': job['job_id'],
         'evidence_kind': 'operator_retrieved_unsigned_runtime_log'})

def project_epoch(state, epoch_id, *, finalized_record, authority=AUTHORITY, input_loader=None):
    """Return four immutable JSON-ready categories; missing evidence is explicit."""
    state = Path(state)
    manifest, completion = finalized(epoch_id, finalized_record, authority)
    output = {'version': VERSION, 'epoch_id': epoch_id,
              **{key: unavailable('signed_training_evidence_unavailable') for key in
                 ('training', 'training_inputs', 'exclusions', 'log_metrics')}}
    training_document = finalized_record.get('training_envelope')
    if training_document is None and input_loader is not None:
        try:
            raw = input_loader('public/'+epoch_id+'/training.json')
            if type(raw) is bytes and len(raw) <= MAX_JSON_BYTES:
                training_document = json.loads(raw)
        except (FileNotFoundError, KeyError):
            pass
    if training_document is None:
        empty_path = state/(epoch_id+'-empty-closed.json')
        if empty_path.is_file() and not empty_path.is_symlink():
            empty = read_json(empty_path)
            if (empty.get('epoch') == epoch_id and empty.get('checkpoint') == completion['checkpoint']
                    and completion['next_checkpoint'] == completion['checkpoint']
                    and empty.get('status') in ('closed_no_eligible_batches', 'closed_no_accepted_batches', 'closed_without_submissions')):
                for key in ('training', 'training_inputs', 'exclusions', 'log_metrics'):
                    output[key] = dict(unavailable('empty_closure_without_signed_training_record'),
                        provenance={'completion_sha256': digest(finalized_record['completion_envelope']),
                            'empty_closure_file_sha256': hashlib.sha256(empty_path.read_bytes()).hexdigest(),
                            'empty_closure_authenticated': False, 'operator_observed_status': empty['status']})
        output['exclusions'] = exclusion_projection(state, epoch_id, manifest, authority, finalized_record.get('population_envelope'))
        return output
    metrics = authenticated(training_document, authority)
    if (set(metrics) == {'checkpoint', 'status'} and metrics['status'] == 'closed_no_eligible_batches'
            and metrics['checkpoint'] == completion['checkpoint'] == completion['next_checkpoint']):
        provenance = {'training_envelope_sha256': digest(training_document),
            'completion_sha256': digest(finalized_record['completion_envelope']),
            'original_manifest_sha256': digest(finalized_record['manifest_envelope']),
            'training_receipt_contains_epoch': False,
            'training_receipt_epoch_binding': 'public_object_key_and_signed_unchanged_checkpoint_completion'}
        output['training'] = available({'status': 'closed_no_eligible_batches',
            'input_checkpoint': completion['checkpoint'], 'output_checkpoint': completion['next_checkpoint'],
            'completed_at': completion['completed_at']}, provenance)
        for key in ('training_inputs', 'log_metrics'):
            output[key] = unavailable('no_training_job_for_signed_empty_closure')
        output['exclusions'] = exclusion_projection(state, epoch_id, manifest, authority,
            finalized_record.get('population_envelope'))
        return output
    if (metrics['source_epoch'] != epoch_id or metrics['input_checkpoint'] != completion['checkpoint']
            or metrics['checkpoint'] != completion['next_checkpoint']
            or metrics.get('state_authority_committed') is not True):
        raise ValueError('completed_signed_training_binding')
    provenance = {'training_envelope_sha256': digest(training_document),
        'completion_sha256': digest(finalized_record['completion_envelope']),
        'original_manifest_sha256': digest(finalized_record['manifest_envelope'])}
    updates = [update_projection(row) for row in metrics['updates']]
    training = fields(metrics, numeric=('steps',), text=('training_policy', 'input_assurance'),
                      boolean=('weights_changed', 'state_updated', 'trainer_verification_performed'))
    training.update(input_checkpoint=completion['checkpoint'], output_checkpoint=completion['next_checkpoint'],
                    completed_at=completion['completed_at'], updates=updates)
    output['training'] = available(training, provenance)
    diagnostics = diagnostics_projection(metrics['persistent_diagnostics'])
    output['log_metrics'] = available({'structured_training_diagnostics': diagnostics,
        'raw_trainer_log': unavailable('raw_logs_not_published_use_authenticated_structured_metrics')}, provenance)
    try:
        job_document, job, admitted_manifest, report = job_report(state, metrics, epoch_id, manifest, completion, authority)
    except FileNotFoundError:
        output['training_inputs'] = unavailable('authoritative_job_or_report_not_retained_locally')
        output['exclusions'] = exclusion_projection(state, epoch_id, manifest, authority, finalized_record.get('population_envelope'))
        return output
    provenance.update(job_envelope_sha256=digest(job_document), job_payload_sha256=digest(job),
        report_sha256=digest(report), job_id=job['job_id'],
        report_scalar_binding='signed-training-updates-and-diagnostics',
        rollout_binding='signed-job-admissions-and-batch-content-and-gradient-pair-hashes')
    output['training_inputs'] = available(inputs_projection(job, report, authority), provenance)
    output['log_metrics']['payload']['runtime_log'] = trainer_log_projection(
        finalized_record.get('trainer_log_record'), epoch_id, job,
        hashlib.sha256((state/'roles'/(job['job_id']+'-job.json')).read_bytes()).hexdigest())
    try:
        output['exclusions'] = exclusion_projection(state, epoch_id, admitted_manifest, authority,
            finalized_record.get('population_envelope'))
    except FileNotFoundError:
        output['exclusions'] = unavailable('signed_collection_artifacts_not_retained_locally')
    return output
