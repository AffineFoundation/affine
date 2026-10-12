"""Public aggregates of authenticated, durable fresh TRAIN generations only."""
import base64
import hashlib
import json
import math
import re
from pathlib import Path

from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError

AUTHORITY = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
VERSION = 'fresh-training-evaluation-v1'
LABEL = 'Training set — fresh generations'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def integer(value, maximum=None):
    return type(value) is int and value >= 0 and (maximum is None or value <= maximum)


def sha(value):
    return type(value) is str and re.fullmatch('[a-f0-9]{64}', value) is not None


def name(value):
    return type(value) is str and re.fullmatch('[A-Za-z0-9_.:-]{1,200}', value) is not None


def project(document, authority=AUTHORITY):
    if (set(document) != {'payload', 'signer', 'signature'} or document['signer'] != authority):
        raise ValueError('ROOT-signed fresh-training aggregate required')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']),
        base64.b64decode(document['signature'], validate=True))
    value = document['payload']
    if (value.get('version') != VERSION or value.get('series') != 'training-fresh' or
            value.get('label') != LABEL or value.get('generation_origin') != 'fresh-policy-generation' or
            value.get('env_id') != 'affine_math' or value.get('phase') not in ('baseline', 'latest')):
        raise ValueError('explicit fresh TRAIN generation scope')
    if any(not name(value.get(key)) for key in ('run_id', 'training_run_id', 'epoch')):
        raise ValueError('public run and epoch identities')
    if any(not sha(value.get(key)) for key in ('cohort_sha256', 'profile_sha256', 'seeds_sha256', 'checkpoint')):
        raise ValueError('exact cohort runtime seed checkpoint identities')
    if (not integer(value.get('optimizer_step')) or
            type(value.get('generated_at')) not in (int, float) or
            not math.isfinite(value['generated_at']) or value['generated_at'] <= 0):
        raise ValueError('actual checkpoint step and completion clock')
    profile = value['profile']
    if (digest(profile) != value['profile_sha256'] or
            not integer(profile.get('max_output_tokens'), 8192) or profile['max_output_tokens'] < 1 or
            not integer(profile.get('max_context_tokens'), 32768) or
            profile['max_context_tokens'] < profile['max_output_tokens'] or
            not sha(profile.get('source_files_sha256')) or
            not isinstance(profile.get('runtime_versions'), dict) or not profile['runtime_versions']):
        raise ValueError('pinned evaluation profile')
    requested = value.get('requested')
    if not integer(requested, 4096) or requested < 1:
        raise ValueError('positive finite cohort denominator')
    fields = ('completed', 'solved', 'wrong', 'unresolved', 'errors', 'cap_hits', 'terminal_eos')
    if any(not integer(value.get(key), requested) for key in fields):
        raise ValueError('complete original outcome accounting')
    if (value['completed'] + value['errors'] != requested or
            value['solved'] + value['wrong'] + value['unresolved'] != value['completed'] or
            max(value['cap_hits'], value['terminal_eos']) > value['completed']):
        raise ValueError('no missing or double-counted task outcomes')
    for key in ('mean_output_tokens', 'median_output_tokens'):
        item = value.get(key)
        if item is not None and (type(item) not in (int, float) or not math.isfinite(item) or
                                 not 0 <= item <= profile['max_output_tokens']):
            raise ValueError('bounded output-length summary')
    exposure = value['exposure']
    if exposure.get('scope') != 'current-training-run-through-checkpoint':
        raise ValueError('training exposure scope')
    unknown = exposure.get('unknown_tasks')
    if not integer(unknown, requested):
        raise ValueError('unknown exposure accounting')
    trained, untrained = exposure.get('trained_tasks'), exposure.get('untrained_tasks')
    if trained is None or untrained is None:
        if trained is not None or untrained is not None or unknown != requested or exposure.get('task_exposures') is not None:
            raise ValueError('missing exposure is unknown, never zero')
    elif (not integer(trained, requested) or not integer(untrained, requested) or
          trained + untrained + unknown != requested or
          not integer(exposure.get('task_exposures')) or exposure['task_exposures'] < trained):
        raise ValueError('actual trained task accounting')
    def outcome_group(group):
        fields = ('count', 'solved', 'wrong', 'unresolved', 'errors')
        if (not isinstance(group, dict) or any(not integer(group.get(k), requested) for k in fields) or
                sum(group[k] for k in fields[1:]) != group['count']):
            raise ValueError('complete descriptive exposure-group outcomes')
        return {k: group[k] for k in fields}
    groups = value.get('exposure_groups')
    if not isinstance(groups, dict) or set(groups) != {'trained', 'untrained', 'unknown'}:
        raise ValueError('all TRAIN exposure groups required')
    groups = {key: outcome_group(group) for key, group in groups.items()}
    if (sum(group['count'] for group in groups.values()) != requested or
            any(sum(group[key] for group in groups.values()) != value[key]
                for key in ('solved', 'wrong', 'unresolved', 'errors')) or
            groups['unknown']['count'] != unknown or
            groups['trained']['count'] != (trained or 0) or
            groups['untrained']['count'] != (untrained or 0) or
            value.get('exposure_subsets_descriptive') is not True):
        raise ValueError('exposure partition matches fixed-cohort totals')
    fixed = value.get('fixed_initial_trained')
    if fixed is not None:
        fixed_group = outcome_group(fixed)
        if (not sha(fixed.get('mask_sha256')) or not sha(fixed.get('checkpoint')) or
                not integer(fixed.get('optimizer_step')) or
                any(fixed[key] > value[key] for key in ('solved', 'wrong', 'unresolved', 'errors'))):
            raise ValueError('pinned fixed initial exposure subset')
        fixed = dict(fixed_group, **{key: fixed[key] for key in
                     ('mask_sha256', 'checkpoint', 'optimizer_step')})
    durable = value['durable_result']
    if (durable.get('full_get_verified') is not True or not sha(durable.get('sha256')) or
            type(durable.get('key')) is not str or not durable['key'] or
            '://' in durable['key'] or '?' in durable['key']):
        raise ValueError('full-GET durable original result required')
    identity = {key: value[key] for key in ('run_id', 'training_run_id', 'checkpoint',
                'cohort_sha256', 'profile_sha256', 'seeds_sha256')}
    if type(value.get('comparability_passed')) is not bool:
        raise ValueError('actual prompt task and seed comparability required')
    complete = value['errors'] == 0 and value['comparability_passed']
    reference_kind = value.get('reference_kind', 'unspecified')
    if reference_kind not in ('original-upstream-comparison', 'actual-run-genesis-comparison', 'unspecified'):
        raise ValueError('explicit supported TRAIN reference role')
    actual_genesis = value.get('actual_run_genesis_checkpoint')
    if actual_genesis is not None and not sha(actual_genesis):
        raise ValueError('actual run genesis identity')
    if reference_kind == 'actual-run-genesis-comparison':
        if (value['phase'] != 'baseline' or value['optimizer_step'] != 0 or
                actual_genesis != value['checkpoint'] or not sha(value.get('genesis_admission_sha256')) or
                value.get('reference_is_original_upstream') is not False):
            raise ValueError('authenticated actual run genesis, not upstream label')
        if (trained != 0 or untrained != requested or unknown != 0 or
                exposure.get('task_exposures') != 0):
            raise ValueError('actual run genesis has zero new training exposure in this run')
    reference_label = None
    if value['phase'] == 'baseline':
        reference_label = ('Actual run genesis — optimizer step 0 of this run; earlier training retained' if reference_kind == 'actual-run-genesis-comparison'
                           else 'Original 7B upstream reference; not actual run genesis' if reference_kind == 'original-upstream-comparison'
                           else 'Checkpoint reference — provenance unspecified')
    # Explicit allowlist: no raw text, object capabilities, paths or error traces.
    return dict(run_id='training-fresh-' + digest(identity), experiment_id=value['run_id'],
        training_run_id=value['training_run_id'], epoch_id=value['epoch'], checkpoint=value['checkpoint'],
        training_steps=value['optimizer_step'], env_id='affine_math', evaluation_split='train',
        label=LABEL, generation_origin='fresh-operator-generation', status='complete' if complete else 'partial',
        timestamp=value['generated_at'], dataset_id=value['cohort_sha256'], taskset_hash=value['cohort_sha256'],
        runtime_profile_sha256=value['profile_sha256'], seed=value['seeds_sha256'],
        count=requested, requested_count=requested, completed_count=value['completed'],
        successes=value['solved'], error_count=value['errors'], unresolved_count=value['unresolved'],
        cap_hits=value['cap_hits'], terminal_eos=value['terminal_eos'],
        mean_reward=value['solved'] / requested if complete else None,
        mean_output_tokens=value.get('mean_output_tokens'), median_output_tokens=value.get('median_output_tokens'),
        output_token_budget=profile['max_output_tokens'], context_token_budget=profile['max_context_tokens'],
        actual_trained_task_count=trained, untrained_task_count=untrained,
        unknown_exposure_task_count=unknown, actual_training_exposures=exposure.get('task_exposures'),
        exposure_groups=groups, fixed_initial_trained=fixed, exposure_subsets_descriptive=True,
        original_report_sha256=durable['sha256'], aggregate_sha256=digest(value),
        comparability_passed=value['comparability_passed'],
        phase=value['phase'], reference_kind=reference_kind, reference_label=reference_label,
        actual_run_genesis_checkpoint=actual_genesis, native_graded=True, proof_verification_performed=False)


def rows(directory, authority=AUTHORITY):
    result = {}
    conflicts = set()
    root = Path(directory)
    if not root.exists():
        return []
    if root.is_symlink():
        raise ValueError('canonical training-diagnostic directory')
    for path in sorted(root.glob('*.signed.json')):
        if path.is_symlink() or path.stat().st_size > 1024 * 1024:
            continue
        try:
            row = project(json.loads(path.read_bytes()), authority)
            key = row['run_id']
            if key in result and result[key] != row:
                conflicts.add(key)
            else:
                result[key] = row
        except (BadSignatureError, ValueError, KeyError, TypeError):
            continue  # Unavailable provenance is never a zero-score observation.
    return [row for key, row in result.items() if key not in conflicts]
