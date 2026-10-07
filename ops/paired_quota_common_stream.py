"""Default-off research common-attempt producer; no production call sites.

Use an independently admitted frozen scientific runtime. Every attempt is
persisted, including errors and unmet quotas. This helper never changes context,
seed schedule, sampler, native grader, checkpoints, optimizer or publication.
"""
import copy

from ops.paired_quota_qualification import digest, identities, select_pairs, _sha
from subnet.audit_policy import InvalidSample
from subnet.fast_prefill_audit import NumericalAmbiguity

VERSION = 'paired-quota-common-attempt-stream-v1'


def _artifact(value):
    if (not isinstance(value, dict) or set(value) != {'sha256', 'size', 'key'} or
            type(value['size']) is not int or value['size'] <= 0 or
            not isinstance(value['key'], str) or not value['key']):
        raise ValueError('durable full rollout/probability/proof artifact reference required')
    _sha(value['sha256'])
    return copy.deepcopy(value)


def collect(runtime, adapter, *, budget=16, admit_runtime=None,
            persist_artifact=None, persist_attempt=None, enabled=False):
    """Collect one task's fixed common stream; never stop after filling quota.

    admit_runtime must independently bind loaded checkpoint/source/runtime to
    the research authorization. Artifact persistence must preserve ORIGINAL
    rollout plus probability arrays/proofs; metadata sink must commit every row.
    Sink failure aborts rather than silently discarding an attempt or resampling.
    Numerical ambiguity and infrastructure errors remain unavailable, not fails.
    """
    if enabled is not True:
        raise ValueError('common stream requires explicit research opt-in')
    if (type(budget) is not int or not 1 <= budget <= 128 or
            tuple(range(budget)) != adapter.task.approved_attempts[:budget]):
        raise ValueError('fixed approved contiguous attempt budget')
    if not all(callable(f) for f in (admit_runtime, persist_artifact, persist_attempt)):
        raise ValueError('runtime admission and durable sinks required')
    if admit_runtime(runtime, adapter.task) is not True:
        raise ValueError('independently admitted research runtime required')
    if (runtime.sampling_context != adapter.sampling_context or
            digest(runtime.harness) != adapter.task.harness_sha256 or
            runtime.spec.id != adapter.task.env_id or
            runtime.spec.version != adapter.definition['spec']['version']):
        raise ValueError('common stream approved runtime binding')
    rows = []
    for attempt in range(budget):
        row = dict(version=VERSION, attempt=attempt, index=adapter.task.index,
                   task_binding=adapter.task.task_binding(),
                   sampling_context_sha256=adapter.task.sampling_context_sha256,
                   harness_sha256=adapter.task.harness_sha256, status='infrastructure_error',
                   artifact=None, normalized=None)
        try:
            rollout, arrays = runtime.rollout(adapter.task.index, attempt)
        except Exception as error:
            # A generator/native failure is not a legitimate negative rollout.
            row['error_type'] = type(error).__name__
        else:
            # Preserve original arrays/proofs BEFORE verification or metadata
            # projection; no proof synthesis or teacher-forced replacement.
            row['artifact'] = _artifact(persist_artifact(rollout, arrays))
            row['original_rollout_sha256'] = digest(rollout)
            try:
                verdict = runtime.verify(rollout, arrays)
                if verdict is not True:
                    raise RuntimeError('official verifier did not accept')
                if rollout.get('classification') not in ('positive', 'negative'):
                    row['status'] = 'verified_unusable_class'
                else:
                    batch = dict(schema=2, epoch=adapter.task.epoch, checkpoint=adapter.task.checkpoint,
                                 env_id=adapter.task.env_id, environment_version=adapter.definition['spec']['version'],
                                 index=adapter.task.index, sample_index=adapter.task.index, rollouts=[rollout])
                    normalized = adapter.normalize(batch)[0]
                    if normalized['attempt'] != attempt:
                        raise ValueError('generated attempt does not match common stream')
                    row.update(status='verified', normalized=normalized)
            except NumericalAmbiguity as error:
                row.update(status='numerical_unknown', error_type=type(error).__name__)
            except InvalidSample as error:
                row.update(status='confirmed_invalid', error_type=type(error).__name__)
            except Exception as error:
                row.update(status='infrastructure_error', error_type=type(error).__name__)
        # Callback must durably persist the complete row before another attempt.
        if persist_attempt(copy.deepcopy(row)) is not True:
            raise ValueError('common attempt row not durably persisted')
        rows.append(row)
    return rows


def select_nested(adapter, miner, rows, *, budget=16):
    """First distinct verified success/failure attempts under one fixed stream.

    The first K1 members are contained in K2. Tasks missing K2 remain in the
    supply report but enter neither matched-training arm. Never retry a missing
    task with different randomness to improve completion.
    """
    if (type(budget) is not int or not 1 <= budget <= 128 or
            not isinstance(rows, list) or len(rows) != budget or
            tuple(range(budget)) != adapter.task.approved_attempts[:budget] or
            any(type(r.get('attempt')) is not int for r in rows) or
            [r.get('attempt') for r in rows] != list(range(budget))):
        raise ValueError('complete ordered common attempt stream required')
    classes = {'positive': [], 'negative': []}; seen_content = {}; seen_tokens = {}
    unavailable = {}; completions = {1: None, 2: None}; duplicates = 0
    for row in rows:
        if (row.get('version') != VERSION or row.get('index') != adapter.task.index or
                row.get('task_binding') != adapter.task.task_binding() or
                row.get('sampling_context_sha256') != adapter.task.sampling_context_sha256 or
                row.get('harness_sha256') != adapter.task.harness_sha256):
            raise ValueError('common attempt context/task binding')
        status = row.get('status')
        if status not in ('verified', 'verified_unusable_class', 'numerical_unknown',
                          'confirmed_invalid', 'infrastructure_error'):
            raise ValueError('common attempt status')
        if status != 'verified':
            if row.get('normalized') is not None:
                raise ValueError('unavailable attempt has no usable sample')
            unavailable[status] = unavailable.get(status, 0) + 1
            continue
        _artifact(row.get('artifact')); _sha(row.get('original_rollout_sha256'))
        normalized = row.get('normalized')
        execution, content = identities(adapter.task, normalized)
        if normalized.get('attempt') != row['attempt']:
            raise ValueError('common attempt normalized draw binding')
        category = normalized.get('classification')
        if category not in classes:
            raise ValueError('verified explicit class required')
        tokens = digest([dict(prompt=t['prompt'], output=t['output']) for t in normalized['turns']])
        if tokens in seen_tokens and seen_tokens[tokens] != content:
            raise ValueError('same token trace with conflicting observations')
        seen_tokens[tokens] = content
        if content in seen_content:
            if seen_content[content] != category:
                raise ValueError('same content conflicting verified class')
            duplicates += 1
            continue
        seen_content[content] = category
        classes[category].append(copy.deepcopy(normalized))
        for quota in (1, 2):
            if completions[quota] is None and min(map(len, classes.values())) >= quota:
                completions[quota] = row['attempt'] + 1
    possible_k1 = min(map(len, classes.values())) >= 1
    matched = min(map(len, classes.values())) >= 2
    arms = {}
    if matched:
        for quota in (1, 2):
            selected = classes['positive'][:quota] + classes['negative'][:quota]
            arms[f'K{quota}L{quota}'] = select_pairs(adapter.task, miner, selected, quota=quota)
        one = {r['content_id'] for p in arms['K1L1']['pairs'] for r in (p['positive'], p['negative'])}
        two = {r['content_id'] for p in arms['K2L2']['pairs'] for r in (p['positive'], p['negative'])}
        if not one < two:
            raise ValueError('nested arm content population')
    return dict(arms=arms, supply=dict(index=adapter.task.index, attempted=budget,
                verified_unique_positive=len(classes['positive']), verified_unique_negative=len(classes['negative']),
                duplicate_verified_content=duplicates, unavailable=unavailable,
                K1L1_possible=possible_k1, matched_included=matched,
                first_K1L1_completion=completions[1], first_K2L2_completion=completions[2]),
                complete_stream_sha256=digest(rows))


def selected_artifact_refs(adapter, selected, rows):
    """Bind selected members back to original archived rollout/proof artifacts.

    This is a metadata inventory, not an authenticated readback or trainer input.
    Actual study admission must authenticate these rows and independently GET
    the referenced bytes, then recompute rollout/content/draw bindings.
    """
    from ops.paired_quota_research_ledger import validate_revision
    validate_revision(selected)
    by_execution = {}
    for row in rows:
        if row.get('status') != 'verified':
            continue
        normalized = row.get('normalized')
        execution, content = identities(adapter.task, normalized)
        if execution in by_execution:
            raise ValueError('duplicate original selected execution reference')
        by_execution[execution] = (content, normalized['classification'], row)
    result = []
    for pair in selected['pairs']:
        refs = {}
        for side in ('positive', 'negative'):
            member = pair[side]
            candidate = by_execution.get(member['execution_id'])
            if candidate is None or candidate[:2] != (member['content_id'], side):
                raise ValueError('selected member missing original artifact binding')
            row = candidate[2]
            refs[side] = dict(attempt=row['attempt'], execution_id=member['execution_id'],
                              content_id=member['content_id'], artifact=_artifact(row['artifact']),
                              original_rollout_sha256=_sha(row['original_rollout_sha256']))
        result.append(refs)
    return result
