"""Inactive training-only candidate: fixed task draw and native-valid fallback.

This is a planner and signed-evidence consumer, not a production installer or
an alternate structural/native validator. An original collector must authenticate
and structurally validate every document before signing the separate pool. The
existing reward/audit inventory and population files remain independent inputs.
No URL is opened, model loaded, grading dispatched, or authority key read here.
"""
import copy
import math
import secrets
from collections import Counter
from pathlib import Path

FIELD = 'training_representative_policy'
VERSION = 'first-native-valid-distinct-task-v2-bounded-waves'
POOL_VERSION = 'structurally-admitted-training-task-pool-v1'
DRAW_VERSION = 'immutable-training-task-draw-v1'
RESULT_VERSION = 'native-valid-task-representatives-v1'


def _deps():
    from subnet.training_receipts import authenticate, sha
    from subnet.committed_training_inputs import (validate_admission,
        receipt_inventory, training_document_cap, MAX_BYTES)
    from subnet.learner_blacklist_selection import partition
    return authenticate, sha, validate_admission, receipt_inventory, training_document_cap, MAX_BYTES, partition


def _policy(manifest):
    if FIELD not in manifest:
        return None  # Existing collection/selection must remain the caller's path.
    p = manifest[FIELD]
    fields = {'version', 'max_candidate_documents', 'max_total_input_bytes',
              'max_native_documents_per_wave', 'max_native_wall_seconds', 'exhausted_task_rule'}
    if (type(p) is not dict or set(p) != fields or p['version'] != VERSION
            or p['exhausted_task_rule'] != 'advance-fixed-task-order'
            or any(type(p[k]) is not int for k in ('max_candidate_documents',
                'max_total_input_bytes', 'max_native_documents_per_wave', 'max_native_wall_seconds'))
            or not 1 <= p['max_candidate_documents'] <= 2304
            or not 1 <= p['max_total_input_bytes'] <= p['max_candidate_documents'] * 2_000_000
            or not 1 <= p['max_native_documents_per_wave'] <= 256
            or not 1 <= p['max_native_wall_seconds'] <= 600):
        raise ValueError('explicit bounded training-only representative policy')
    return p


def admit_pool(pool_envelope, authority):
    """Recheck signatures/bindings; raw structural validation belongs to collector.

    A ROOT pool signature attests original admitted_submission completed, exactly
    as current learner population publication does. It is not sampler verification
    or grading evidence. The proposed collector has NOT been installed.
    """
    authenticate, sha, validate_admission, inventory, cap, maximum, _ = _deps()
    pool = authenticate(pool_envelope, authority)
    fields = {'version', 'original_signed_manifest', 'submissions',
              'capture_receipts_sha256', 'original_population_file_sha256',
              'structural_inventory_sha256', 'sampling_assurance',
              'proof_verification_performed'}
    if (set(pool) != fields or pool['version'] != POOL_VERSION
            or pool['sampling_assurance'] != 'unaudited'
            or pool['proof_verification_performed'] is not False):
        raise ValueError('exact unaudited structurally admitted candidate pool')
    manifest = authenticate(pool['original_signed_manifest'], authority)
    policy = _policy(manifest)
    if policy is None:
        raise ValueError('historical epoch cannot use representative candidate')
    from subnet.training_receipts import digest
    for k in ('capture_receipts_sha256', 'original_population_file_sha256',
              'structural_inventory_sha256'):
        digest(pool[k])
    objects = pool['submissions']
    if (not isinstance(objects, list) or len(objects) > policy['max_candidate_documents']
            or policy['max_native_documents_per_wave'] > cap(manifest)):
        raise ValueError('bounded original candidate population')
    identities = set(); documents = set(); tasks = []; rows = []
    for obj in objects:
        a, child = validate_admission(obj['learner_admission'], obj, manifest, authority)
        identity = (a['miner_identity'], a['slot'])
        if (identity in identities or obj['sha256'] in documents
                or a['miner_identity'] not in manifest['capabilities']):
            raise ValueError('unique original registered slots/documents')
        key = (child.get('env_id'), child.get('index'))
        if not isinstance(key[0], str) or type(key[1]) is not int:
            raise ValueError('original task identity')
        identities.add(identity); documents.add(obj['sha256']); tasks.append(key)
        rows.append(dict(task=list(key), miner=a['miner_identity'], slot=a['slot'],
                         document_sha256=obj['sha256'],
                         learner_admission_sha256=sha(obj['learner_admission'])))
    if (sum(o['size'] for o in objects) > policy['max_total_input_bytes']
            or len(objects) > len(manifest['capabilities']) * manifest['max_batches']
            or sha(inventory(objects)) != pool['structural_inventory_sha256']):
        raise ValueError('exact bounded original structural inventory')
    counts = Counter(tasks)
    # This order and rule are the EXACT existing pre-blacklist reward algorithm.
    reward = [obj for obj, task in zip(objects, tasks) if counts[task] == 1]
    return manifest, policy, objects, rows, reward


def freeze_draw(pool_envelope, authority, journal_path, *, now):
    """Persist one unpredictable seed after full inventory authentication/freeze.

    Operator randomness is an explicit assumption, not a public randomness beacon.
    Never renew/reroll this seed after restart or after seeing native outcomes.
    """
    _, sha, _, inventory, cap, _, partition = _deps()
    manifest, policy, objects, _, reward = admit_pool(pool_envelope, authority)
    if type(now) not in (int, float) or not math.isfinite(now) or now < manifest['deadline']:
        raise ValueError('postfreeze draw time')
    from ops.native_training_eligibility import _create, _load
    import json
    p = Path(journal_path)
    binding = dict(version=DRAW_VERSION, pool_envelope_sha256=sha(pool_envelope),
        policy_sha256=sha(policy), original_manifest_sha256=sha(pool_envelope['payload']['original_signed_manifest']),
        structural_inventory_sha256=sha(inventory(objects)),
        reward_eligible_inventory_sha256=sha(inventory(reward)), cap=cap(manifest))
    if not p.exists():
        _, blacklist = partition(objects, manifest, authority, at=now,
            round_number=manifest.get('learner_blacklist_selection_round'))
        value = dict(binding, seed=secrets.token_hex(32), captured_at=now,
                     blacklist_selection=blacklist)
        _create(p, value)
    draw = json.loads(_load(p))
    validate_draw(pool_envelope, authority, draw)
    return draw


def validate_draw(pool_envelope, authority, draw):
    _, sha, _, inventory, cap, _, partition = _deps()
    manifest, policy, objects, rows, reward = admit_pool(pool_envelope, authority)
    from subnet.training_receipts import digest
    fields = {'version', 'pool_envelope_sha256', 'policy_sha256',
        'original_manifest_sha256', 'structural_inventory_sha256',
        'reward_eligible_inventory_sha256', 'cap', 'seed', 'captured_at', 'blacklist_selection'}
    if (set(draw) != fields or draw['version'] != DRAW_VERSION
            or draw['pool_envelope_sha256'] != sha(pool_envelope)
            or draw['policy_sha256'] != sha(policy)
            or draw['original_manifest_sha256'] != sha(pool_envelope['payload']['original_signed_manifest'])
            or draw['structural_inventory_sha256'] != sha(inventory(objects))
            or draw['reward_eligible_inventory_sha256'] != sha(inventory(reward))
            or draw['cap'] != cap(manifest)
            or type(draw['captured_at']) not in (int, float)
            or not math.isfinite(draw['captured_at']) or draw['captured_at'] < manifest['deadline']):
        raise ValueError('immutable representative draw original bindings')
    digest(draw['seed'])
    kept, blacklist = partition(objects, manifest, authority, at=draw['captured_at'],
        round_number=manifest.get('learner_blacklist_selection_round'))
    if draw['blacklist_selection'] != blacklist:
        raise ValueError('immutable original training blacklist snapshot')
    kept_ids = {o['sha256'] for o in kept}
    groups = {}
    for row in rows:
        if row['document_sha256'] in kept_ids:
            groups.setdefault(tuple(row['task']), []).append(row)
    common = dict(seed=draw['seed'], epoch=manifest['epoch'],
                  checkpoint=manifest['checkpoint']['id'], source=manifest['source_bundle']['sha256'])
    # Pool multiplicity does not participate in either task rank or miner rank.
    order = sorted(groups, key=lambda t: (sha(dict(common, domain='distinct-training-task-v1', task=t)), t))
    for task in order:
        groups[task].sort(key=lambda r: (
            sha(dict(common, domain='task-miner-representative-v1', task=task, miner=r['miner'])),
            r['miner'], r['slot'], r['document_sha256']))
    return manifest, policy, objects, reward, order, groups


def _next(policy, cap, order, groups, accepted, tried):
    remaining = cap - len(accepted)
    if remaining <= 0:
        return []
    chosen = []
    for task in order:
        if task in accepted:
            continue
        available = [r for r in groups[task] if r['document_sha256'] not in tried]
        if available:
            chosen.append(available[0])
            if len(chosen) == min(remaining, policy['max_native_documents_per_wave']):
                break
    return chosen


def replay(pool_envelope, authority, draw, authorization_envelope, waves):
    """Reconstruct progress only from complete original signed native wave receipts.

    No exception/transport timeout is converted to an invalid candidate. Each
    wave has <=256 documents, one per task; rejected documents advance only that
    task's frozen candidate order. Exhausted tasks refill from fixed task order.
    """
    authenticate, sha, _, inventory, _, _, _ = _deps()
    manifest, policy, objects, reward, order, groups = validate_draw(pool_envelope, authority, draw)
    authorization = authenticate(authorization_envelope, authority)
    from ops.native_training_outcome_filter import AUTHORIZATION_VERSION, CONTEXT_VERSION
    from ops.native_training_eligibility import bind_subset
    if (authorization.get('version') != AUTHORIZATION_VERSION
            or authorization.get('source_sha256') != manifest['source_bundle']['sha256']
            or authorization.get('sampling_assurance') != 'unaudited'
            or authorization.get('no_credit') is not True or authorization.get('no_relabel') is not True
            or 'benchmark_scope' in authorization):
        raise ValueError('original native authorization and unaudited policy')
    by_id = {o['sha256']: o for o in objects}
    accepted = {}; tried = set(); evidence = []
    if not isinstance(waves, list) or len(waves) > len(objects):
        raise ValueError('bounded native wave journal')
    for wave in waves:
        if set(wave) != {'context', 'grades'}:
            raise ValueError('exact original native wave envelopes')
        selected = _next(policy, draw['cap'], order, groups, accepted, tried)
        if not selected:
            raise ValueError('no extra native waves after completion')
        expected_objects = [by_id[r['document_sha256']] for r in selected]
        context = authenticate(wave['context'], authority)
        expected = dict(version=CONTEXT_VERSION,
            original_signed_manifest=pool_envelope['payload']['original_signed_manifest'],
            submissions=expected_objects, source_files=authorization['source_files'],
            authorization_sha256=sha(authorization_envelope),
            original_population_file_sha256=sha(pool_envelope),
            original_selection_file_sha256=sha(draw),
            parent_binding_sha256=sha(manifest['trainer_state_binding']))
        if context != expected:
            raise ValueError('native wave exact next task representatives')
        grades = authenticate(wave['grades'], authority)
        if grades.get('terminal_rule') != authorization.get('limits', {}).get('terminal_rule'):
            raise ValueError('unchanged authorized native terminal rule')
        valid, subset = bind_subset(wave['context'], grades, expected_objects, authority)
        valid_ids = {o['sha256'] for o in valid}
        for row in selected:
            document = row['document_sha256']; tried.add(document)
            if document in valid_ids:
                accepted[tuple(row['task'])] = document
        evidence.append(dict(context_sha256=sha(wave['context']), grades_sha256=sha(wave['grades']),
                             derived_subset_sha256=sha(subset)))
    selected = _next(policy, draw['cap'], order, groups, accepted, tried)
    final_ids = set(accepted.values())
    # Original population order remains the final task/gradient accumulation order.
    final = [o for o in objects if o['sha256'] in final_ids]
    return dict(version=RESULT_VERSION, pool_envelope_sha256=sha(pool_envelope),
        draw_sha256=sha(draw), authorization_sha256=sha(authorization_envelope),
        native_waves=evidence, checked_count=len(tried), accepted_count=len(final),
        accepted_submissions=copy.deepcopy(final), accepted_inventory_sha256=sha(inventory(final)),
        reward_eligible_inventory=inventory(reward),
        reward_eligible_inventory_sha256=sha(inventory(reward)),
        next_submissions=[copy.deepcopy(by_id[r['document_sha256']]) for r in selected],
        complete=not selected, disposition=('pending_native_evidence' if selected else 'train' if final else 'no_update'),
        sampling_assurance='unaudited', proof_verification_performed=False,
        claims_rewritten=False, cheating_penalties=False)


RECEIPT_VERSION = 'native-task-representative-derivation-v1'


def collection_paths(state, epoch):
    import re
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{1,220}', epoch):
        raise ValueError('representative original epoch namespace')
    return (Path(state)/(epoch+'-training-structural-pool.ROOT-SIGNED.json'),
            Path(state)/(epoch+'-training-collection-transaction.json'))


def save_collection(controller, manifest, candidates, receipts, value):
    """Commit separate pool and original population together, recoverably.

    Called ONLY after each raw candidate passed original admitted_submission.
    The original singleton population and audit registration are not replaced.
    """
    from ops.native_training_eligibility import _create
    from .storage import canonical
    import hashlib
    _, sha, _, inventory, _, _, _ = _deps()
    pool_path, transaction_path = collection_paths(controller.state, manifest['epoch'])
    envelope = controller.signed(dict(version=POOL_VERSION,
        original_signed_manifest=controller.signed(manifest), submissions=candidates,
        capture_receipts_sha256=sha(receipts),
        original_population_file_sha256=hashlib.sha256(canonical(value)).hexdigest(),
        structural_inventory_sha256=sha(inventory(candidates)),
        sampling_assurance='unaudited', proof_verification_performed=False))
    admit_pool(envelope, controller.authority.id)
    transaction=dict(pool=envelope, original_population=value)
    if len(canonical(transaction))>64*1024**2:raise ValueError('bounded original representative collection transaction')
    _create(transaction_path, transaction)
    return resume_collection(controller, manifest)


def resume_collection(controller, manifest):
    import json, hashlib
    from .storage import canonical
    from .training_receipts import computation_binding
    from ops.native_training_eligibility import _create, _load
    pool_path, transaction_path = collection_paths(controller.state, manifest['epoch'])
    if not transaction_path.exists():
        return None
    transaction = json.loads(_load(transaction_path))
    if set(transaction) != {'pool','original_population'}:
        raise ValueError('exact original collection transaction')
    original, _, _, _, reward = admit_pool(transaction['pool'], controller.authority.id)
    value = transaction['original_population']
    if (computation_binding(original) != computation_binding(manifest)
            or original.get('trainer_state_binding') != manifest.get('trainer_state_binding')
            or computation_binding(value['manifest']) != computation_binding(manifest)
            or value['population']['eligible_inventory'] != _deps()[3](reward)
            or hashlib.sha256(canonical(value)).hexdigest() != transaction['pool']['payload']['original_population_file_sha256']):
        raise ValueError('original population and structurally admitted pool binding')
    _create(pool_path, transaction['pool'])
    _create(Path(controller.state)/(manifest['epoch']+'-learner-population.json'), value)
    controller.bucket.json('public/'+manifest['epoch']+'/learner-population.json', controller.signed(value['population']))
    return value['manifest'], value['submissions'], value['population']


def finalize(progress, draw, policy, finished_at):
    """Bounded completion never relabels an ungraded document as invalid."""
    if type(finished_at) not in (int,float) or not math.isfinite(finished_at) or finished_at < draw['captured_at']:
        raise ValueError('finite representative finalization time')
    exhausted = finished_at >= draw['captured_at'] + policy['max_native_wall_seconds']
    if not progress['complete'] and not exhausted:
        raise ValueError('unfinished native representatives before fixed deadline')
    result = copy.deepcopy(progress)
    result.update(finalized_at=finished_at,
        completion_reason='selection_complete' if progress['complete'] else 'native_budget_exhausted',
        disposition='train' if progress['accepted_submissions'] else 'no_update',
        ungraded_next_inventory_sha256=_deps()[1](_deps()[3](progress['next_submissions'])))
    result.pop('next_submissions')
    return result


def derivation_receipt(documents, authority):
    """Authenticate every real wave and reconstruct the whole final selection."""
    authenticate, sha, _, _, _, _, _ = _deps()
    if set(documents) != {'pool','draw','authorization','waves','result'}:
        raise ValueError('exact representative derivation documents')
    draw = authenticate(documents['draw'], authority)
    _, policy, *_ = validate_draw(documents['pool'], authority, draw)
    progress = replay(documents['pool'], authority, draw, documents['authorization'], documents['waves'])
    result = authenticate(documents['result'], authority)
    if finalize(progress, draw, policy, result.get('finalized_at')) != result:
        raise ValueError('original full representative replay differs')
    return dict(version=RECEIPT_VERSION, pool_sha256=sha(documents['pool']),
        draw_sha256=sha(documents['draw']), authorization_sha256=sha(documents['authorization']),
        waves_sha256=sha(documents['waves']), result_sha256=sha(documents['result']),
        sampling_assurance='unaudited', proof_verification_performed=False,
        claims_rewritten=False, cheating_penalties=False)


def validate_receipt(receipt):
    from .training_receipts import digest
    if (type(receipt) is not dict or set(receipt) != {'version','pool_sha256','draw_sha256',
            'authorization_sha256','waves_sha256','result_sha256','sampling_assurance',
            'proof_verification_performed','claims_rewritten','cheating_penalties'}
            or receipt['version'] != RECEIPT_VERSION or receipt['sampling_assurance'] != 'unaudited'
            or any(receipt[k] is not False for k in ('proof_verification_performed','claims_rewritten','cheating_penalties'))):
        raise ValueError('exact truthful task representative receipt')
    for name in ('pool_sha256','draw_sha256','authorization_sha256','waves_sha256','result_sha256'):
        digest(receipt[name])


def preparation_scope(public_envelope, native_envelope, submissions, documents, authority):
    from .training_receipts import authenticate, computation_binding, sha
    from .committed_training_inputs import receipt_inventory
    public, native = (authenticate(e, authority) for e in (public_envelope, native_envelope))
    original = authenticate(documents['pool']['payload']['original_signed_manifest'], authority)
    if (_policy(public) is None or computation_binding(public) != computation_binding(native)
            or computation_binding(original) != computation_binding(native)
            or public['trainer_state_binding'] != native['trainer_state_binding']
            or original['trainer_state_binding'] != native['trainer_state_binding']):
        raise ValueError('original representative public computation and retained parent')
    receipt = derivation_receipt(documents, authority)
    result = authenticate(documents['result'], authority)
    if (receipt != native.get('native_training_eligibility_receipt')
            or submissions != result['accepted_submissions'] or result['disposition'] != 'train'
            or native.get('training_coverage',{}).get('inventory_sha256') != result['accepted_inventory_sha256']):
        raise ValueError('exact native representative final selected inventory')
    draw=authenticate(documents['draw'],authority)
    if draw['blacklist_selection'] is not None and native.get('learner_blacklist_selection_snapshot')!=draw['blacklist_selection']:
        raise ValueError('exact representative full-pool blacklist snapshot')
    return dict(original_public_manifest_sha256=sha(public_envelope),
        original_signed_manifest_sha256=sha(native_envelope),
        input_inventory_sha256=sha(receipt_inventory(submissions)), native_eligibility_receipt=receipt)
