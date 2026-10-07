"""Default-off first-prescribed-attempt research selection; no live caller.

Native record authenticator is a TRUSTED caller boundary: it independently checks
original admitted document/manifest/task/model and actual native grading evidence.
Returning this module's expected binding without those checks is not assurance.
No model, grader, proof verification, optimizer or public quota change occurs here.
"""
import copy
from ops.paired_quota_qualification import ApprovedTask, _sha, digest, identities
from subnet.trajectory_identity import token_trace_sha256

VERSION = 'first-prescribed-native-EOS-nested-quota-research-v1'
GRADE_VERSION = 'authenticated-admitted-native-grade-research-v1'
STATUSES = ('native-graded', 'indeterminate', 'admission-rejected', 'infrastructure')


def selection_scope(task, miner_public_key, eos_token_ids):
    if type(task) is not ApprovedTask:
        raise ValueError('original approved research task required')
    _sha(miner_public_key)
    if task.approved_attempts != tuple(sorted(task.approved_attempts)):
        raise ValueError('fixed ascending prescribed attempt order required')
    if (type(eos_token_ids) is not tuple or not eos_token_ids or
            len(set(eos_token_ids)) != len(eos_token_ids) or
            any(type(t) is not int or not 0 <= t < 200000 for t in eos_token_ids)):
        raise ValueError('pinned tokenizer EOS token IDs required')
    return dict(version=VERSION, epoch=task.epoch, miner_public_key=miner_public_key,
                task=task.task_binding(), harness_sha256=task.harness_sha256,
                sampling_context_sha256=task.sampling_context_sha256,
                ordered_attempts=list(task.approved_attempts), eos_token_ids=list(eos_token_ids))


def _revision(task, miner, positives, negatives, quota, duplicate_count):
    if len(positives) < quota or len(negatives) < quota:
        return None
    pairs = []
    for p, n in zip(positives[:quota], negatives[:quota]):
        pairs.append({side: {k: row[k] for k in ('execution_id', 'content_id', 'classification')}
                      for side, row in (('positive', p), ('negative', n))})
    value = dict(kind='selected-task-revision-v1', slot_id=task.slot_id(miner),
                 quota=quota, pairs=pairs, contribution_units=1, pair_weight_within_task=1/quota)
    return dict(value, revision_id=digest(value), duplicate_content_count=duplicate_count)


def select_nested(task, miner_public_key, records, *, eos_token_ids,
                  authenticate_admitted_native, enabled=False):
    """Select nested K1/L1 and K2/L2 from a COMPLETE frozen attempt observation.

    records contain exact {attempt,row,evidence}; row is normalized original trace
    or None for failed admission/infrastructure. No supplied member IDs are used.
    Callback(record, expected_binding) must authenticate original admission AND
    native grade and return the exact GRADE_VERSION receipt schema below. EOS IDs
    must come from the authenticated pinned tokenizer, not miner metadata.
    A task unable to fill remains explicit complete_K1/complete_K2=False.
    """
    if enabled is not True:
        raise ValueError('research nested selector requires explicit opt-in')
    scope = selection_scope(task, miner_public_key, eos_token_ids)
    if not callable(authenticate_admitted_native) or not isinstance(records, list) or len(records) > len(task.approved_attempts)*4:
        raise ValueError('bounded authenticated frozen observations required')
    by_attempt = {}; executions = {}; contents = {}; token_traces = {}
    duplicate_executions = 0
    for record in records:
        if type(record) is not dict or set(record) != {'attempt','row','evidence'}:
            raise ValueError('exact original observation fields')
        attempt = record['attempt']; row = record['row']
        if type(attempt) is not int or attempt not in task.approved_attempts:
            raise ValueError('unapproved prescribed attempt')
        execution = content = token_trace = None
        if row is not None:
            execution, content = identities(task, row)
            if row['attempt'] != attempt:
                raise ValueError('original attempt binding')
            token_trace = token_trace_sha256(row['turns'])
        binding = dict(attempt=attempt, selection_scope_sha256=digest(scope),
                       row_sha256=digest(row) if row is not None else None,
                       execution_id=execution, content_id=content)
        grade = authenticate_admitted_native(copy.deepcopy(record), copy.deepcopy(binding))
        if type(grade) is not dict or set(grade) != set(binding)|{'version','status','classification','reward','native_done'}:
            raise ValueError('independently authenticated admitted native receipt required')
        if (grade['version'] != GRADE_VERSION or grade['status'] not in STATUSES or
                any(type(grade[k]) is not type(v) or grade[k] != v for k,v in binding.items())):
            raise ValueError('exact authenticated grade scope')
        if grade['status'] == 'native-graded':
            if (row is None or grade['classification'] not in ('positive','negative') or
                    type(grade['reward']) not in (int,float) or grade['reward'] != int(grade['classification']=='positive') or
                    type(grade['native_done']) is not bool or row.get('classification') != grade['classification']):
                raise ValueError('binary native grade/classification consistency')
        elif any(grade[k] is not None for k in ('classification','reward','native_done')):
            raise ValueError('failed/indeterminate grade is not binary evidence')
        terminal_eos = row is not None and row['turns'][-1]['output'][-1] in eos_token_ids
        signature = (content, grade['status'], grade['classification'], grade['reward'], grade['native_done'], terminal_eos)
        if attempt in by_attempt:
            if by_attempt[attempt]['signature'] != signature:
                raise ValueError('conflicting original prescribed attempt')
            duplicate_executions += 1
            continue
        if execution is not None:
            if execution in executions and executions[execution] != signature:
                raise ValueError('conflicting canonical execution')
            executions[execution] = signature
        if grade['status'] == 'native-graded':
            label = grade['classification']
            if content in contents and contents[content] != label:
                raise ValueError('conflicting native content labels')
            contents[content] = label
            if token_trace in token_traces and token_traces[token_trace] != (content,label):
                raise ValueError('conflicting native token trace observations or labels')
            token_traces[token_trace] = (content,label)
        by_attempt[attempt] = dict(signature=signature, row=copy.deepcopy(row), grade=copy.deepcopy(grade),
                                   execution_id=execution, content_id=content, token_trace=token_trace,
                                   terminal_eos=terminal_eos)
    positives = []; negatives = []; seen_contents = set(); seen_tokens = set()
    supply = []; first_K1 = first_K2 = None; duplicate_contents = 0
    for position, attempt in enumerate(task.approved_attempts, 1):
        observed = by_attempt.get(attempt)
        reason = 'not-observed'; eligible = False
        if observed is not None:
            grade = observed['grade']; reason = grade['status']
            if grade['status'] == 'native-graded':
                if not grade['native_done']: reason = 'native-incomplete'
                elif not observed['terminal_eos']: reason = 'no-terminal-EOS'
                elif observed['content_id'] in seen_contents or observed['token_trace'] in seen_tokens:
                    reason = 'duplicate-content'; duplicate_contents += 1
                else:
                    eligible = True; reason = 'eligible'
                    seen_contents.add(observed['content_id']); seen_tokens.add(observed['token_trace'])
                    member = dict(execution_id=observed['execution_id'], content_id=observed['content_id'],
                                  classification=grade['classification'], attempt=attempt,
                                  row_sha256=digest(observed['row']))
                    (positives if grade['classification']=='positive' else negatives).append(member)
        supply.append(dict(attempt=attempt, observed=observed is not None, eligible=eligible, reason=reason,
                           classification=observed['grade']['classification'] if observed else None))
        if first_K1 is None and positives and negatives: first_K1 = position
        if first_K2 is None and len(positives)>=2 and len(negatives)>=2: first_K2 = position
    k1_gaps = [r['attempt'] for r in supply[:first_K1] if not r['observed']] if first_K1 is not None else []
    k2_gaps = [r['attempt'] for r in supply[:first_K2] if not r['observed']] if first_K2 is not None else []
    k1 = _revision(task,miner_public_key,positives,negatives,1,duplicate_contents) if first_K1 is not None and not k1_gaps else None
    k2 = _revision(task,miner_public_key,positives,negatives,2,duplicate_contents) if first_K2 is not None and not k2_gaps else None
    chosen = positives[:2]+negatives[:2] if k2 is not None else positives[:1]+negatives[:1] if k1 is not None else []
    return dict(version=VERSION, selection_scope_sha256=digest(scope), slot_id=task.slot_id(miner_public_key),
                complete_K1=k1 is not None, complete_K2=k2 is not None, K1L1=k1, K2L2=k2,
                first_K1_prefix_length=first_K1 if k1 is not None else None,
                first_K2_prefix_length=first_K2 if k2 is not None else None,
                completion_prefix_gaps=dict(K1L1=k1_gaps, K2L2=k2_gaps),
                supply=supply, observed_unique_attempts=len(by_attempt), duplicate_executions=duplicate_executions,
                duplicate_contents=duplicate_contents, selected_members=chosen,
                per_arm_task_contribution_units=dict(K1L1=int(k1 is not None), K2L2=int(k2 is not None)),
                grade_assurance='caller-authenticated-native-grading-not-inference-proof',
                optimizer_application_performed=False)
