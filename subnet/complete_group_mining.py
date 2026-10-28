"""Prospective complete-group Miner path; never calls legacy pair transport.

The version-dispatched R2 transport adapter validates new wire semantics,
restores/prepares/persists complete groups and uploads the exact prepared bytes.
The local attempt journal preserves a task's original nonce prefix separately.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time

from verifiers.v1.errors import TaskError

from .storage import canonical
from .batch_contract import CompleteGroupContract, REQUIRED_CLIENT_FEATURES
from .complete_group_state import AttemptJournal
from .complete_group_contract import meets_minima

BATCH_VERSION = 'complete-canonical-group-batch-v1'
BEHAVIOR_VERSION = 'sampler-selected-behavior-logprobs-v1'


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


class GroupFiltered(RuntimeError):
    """All prescribed attempts finished but the complete group is not mixed."""


class GroupIndeterminate(RuntimeError):
    """A native task error leaves this entire group unavailable, never negative."""


def initialize(actor, identity, manifest, checkpoint, capability, state_path,
               progress_path, transport):
    from .miner import MinerProgress
    from .model import check_runtime_profile
    from .protocol import entries
    from .batches import compression_for_manifest
    from .forced_sampling import binding

    need(isinstance(actor.batch_contract, CompleteGroupContract), 'new group contract required')
    need(transport is not None, 'complete-group transport adapter required; legacy pair upload forbidden')
    need(REQUIRED_CLIENT_FEATURES <= set(getattr(transport, 'features', ())),
         'complete-group transport features unavailable')
    for method in ('validate', 'validate_capability', 'check_runtime', 'restore', 'prepare', 'persist', 'upload'):
        need(callable(getattr(transport, method, None)), 'complete-group transport method missing: '+method)
    # Before capability decryption, local state or any model/runtime creation.
    transport.validate(manifest, actor.batch_contract)
    compression_for_manifest(manifest)
    check_runtime_profile(manifest)
    entries(manifest)
    context = binding(manifest, identity.id)
    need(context is not None, 'complete-group sampling binding required')
    actor.identity, actor.manifest, actor.checkpoint = identity, deepcopy(manifest), checkpoint
    actor._group_context = context
    actor._group_transport = transport
    actor.state_path = Path(state_path) if state_path else None
    actor._group_journal = AttemptJournal(actor.state_path, actor.manifest, identity.id)
    try:
        actor._group_prepared = transport.restore(actor.manifest, identity.id, actor.state_path)
        need(type(actor._group_prepared) is list and len(actor._group_prepared) <= actor.batch_contract.max_batches,
             'restored complete-group batch cap')
        actor.batches = []
        tasks = set()
        for batch, prepared in actor._group_prepared:
            validate_group(batch, actor.manifest, actor.batch_contract, context, identity.id)
            key = task_key(actor, batch['env_id'], batch['index'])
            need(key not in tasks, 'duplicate restored canonical task group')
            tasks.add(key)
            actor.batches.append((batch, None))
            actor._group_journal.reconcile_accepted(key, batch)
        actor._group_journal.require_accepted(tasks)
        if capability is None:
            from .complete_group_transport import resolve_capability
            capability = resolve_capability(identity, manifest)
        actor.cap = capability
        transport.validate_capability(actor.cap, actor.manifest, identity.id)
    except BaseException:
        actor._group_journal.close()
        raise
    actor.runtime = None
    actor.runtimes = {}
    actor.progress = MinerProgress(progress_path, manifest) if progress_path else None
    actor._progress('epoch_start')


def task_key(actor, env_id, index):
    from .protocol import entry, harness_for
    definition = entry(actor.manifest, env_id)
    need(type(index) is int and index in definition['indices'], 'authorized canonical task index required')
    resolved = harness_for(definition, index)
    return digest(dict(miner=actor.identity.id, epoch=actor.manifest['epoch'],
                       sampling_context_sha256=digest(actor._group_context),
                       env_id=definition['env_id'], spec=definition['spec'],
                       harness=resolved, index=index))


def _deadline(actor):
    from .miner import EpochClosed
    if time.time() >= actor.manifest.get('deadline', float('inf')):
        actor._progress('epoch_deadline')
        raise EpochClosed('signed epoch window closed')


def _validate_witness(evidence, prompt, output, nonce, context, stop):
    need(type(evidence) is dict, 'actual sampler behavior witness missing')
    fields = {'version', 'attempt', 'turn', 'binding_sha256', 'prompt_sha256', 'output_sha256',
              'behavior_selected_logprobs', 'stop_reason'}
    need(set(evidence) == fields and evidence['version'] == BEHAVIOR_VERSION,
         'explicit behavior witness schema required')
    need(type(evidence['attempt']) is int and evidence['attempt'] == nonce
         and type(evidence['turn']) is int and evidence['turn'] == 0,
         'behavior witness prescribed attempt/turn binding')
    from .batch_lifetime import draw_context
    need(evidence['binding_sha256'] == digest(draw_context(context)), 'behavior witness checkpoint/runtime/miner binding')
    need(evidence['prompt_sha256'] == digest(prompt) and evidence['output_sha256'] == digest(output)
         and evidence['stop_reason'] == stop, 'behavior witness token/terminal binding')
    values = evidence['behavior_selected_logprobs']
    need(type(values) is list and len(values) == len(output)
         and all(type(x) in (int, float) and math.isfinite(x) and x <= 0 for x in values),
         'one finite sampler probability per generated token, including EOS')


def _rollout_from_runtime(runtime, rollout, nonce):
    """Retain actual sampled evidence; an unfinished answer is always neutral.

    The qualified runtime/session owns native grading and authoritative task
    reset. This projection never invents logprobs or interprets raw T=1 entries
    as generation probabilities. It supports single-turn math explicitly.
    """
    row = deepcopy(rollout)
    need(type(row) is dict and type(row.get('seed')) is int and row['seed'] == nonce,
         'runtime returned a different prescribed nonce')
    turns = row.get('turns')
    need(type(turns) is list and len(turns) == 1, 'complete-group initial single-turn scope')
    turn = turns[0]
    prompt, output = turn.get('prompt'), turn.get('output')
    need(type(prompt) is list and prompt and type(output) is list and output
         and all(type(x) is int and x >= 0 for x in prompt+output), 'actual prompt/output token arrays required')
    from .complete_group_distribution import eos_ids
    stops = eos_ids(runtime.sampling_context['token_geometry'])
    horizon = runtime.harness['max_output_tokens']
    need(type(horizon) is int and 0 < len(output) <= horizon,
         'qualified EOS and output horizon required')
    need(not any(t in stops for t in output[:-1]), 'tokens after first EOS')
    stop = 'eos' if output[-1] in stops else 'budget'
    need(stop == 'eos' or len(output) == horizon, 'incomplete nonterminal runtime output')
    evidence = deepcopy(getattr(runtime, 'complete_group_sample_evidence', None))
    _validate_witness(evidence, prompt, output, nonce, runtime.sampling_context, stop)
    kind = row.get('classification')
    need(kind in ('positive', 'negative', 'neutral'), 'explicit native outcome required')
    need(turn.get('classification') == kind and turn.get('done') is True,
         'complete native result required')
    need(type(row.get('reward')) in (int, float) and row['reward'] == float(kind == 'positive')
         and type(turn.get('reward')) in (int, float) and turn['reward'] == row['reward'],
         'binary native math reward/classification consistency')
    if stop == 'budget':
        # Keep original native claim visible; it cannot fill a resolved quota.
        row['native_classification'] = kind
        row['classification'], row['reward'] = 'neutral', 0.0
        turn['native_classification'] = kind
        turn['classification'], turn['reward'] = 'neutral', 0.0
    turn['stop_reason'] = stop
    turn['behavior_sampling'] = evidence
    return row


def validate_group(batch, manifest, contract, context, miner):
    """Cheap local shape checks only, not native grading/proof verification."""
    from .forced_sampling import receipt
    from .protocol import entry, harness_for
    need(type(batch) is dict and set(batch) == {
        'schema', 'epoch', 'checkpoint', 'miner', 'env_id', 'environment_version',
        'sample_index', 'index', 'task_hash', 'group_contract_sha256',
        'sampling_context_sha256', 'rollouts'}, 'exact prospective group batch fields')
    need(batch['schema'] == BATCH_VERSION and batch['epoch'] == manifest['epoch']
         and batch['checkpoint'] == manifest['checkpoint']['id'] and batch['miner'] == miner,
         'complete-group epoch/checkpoint/miner binding')
    need(batch['group_contract_sha256'] == digest(manifest['group_contract'])
         and batch['sampling_context_sha256'] == digest(context), 'complete-group contract binding')
    need(type(batch['index']) is int and type(batch['sample_index']) is int
         and batch['index'] == batch['sample_index'], 'canonical group index')
    rows = batch['rollouts']
    need(type(rows) is list and len(rows) == contract.group_size, 'exact N original attempts required')
    definition = entry(manifest, batch['env_id'])
    horizon = harness_for(definition, batch['index'])['max_output_tokens']
    for nonce, row in enumerate(rows):
        need(type(row) is dict and type(row.get('seed')) is int and row['seed'] == nonce,
             'canonical ordered nonces 0..N-1 required')
        need(all(row.get(k) == batch[k] for k in ('env_id', 'environment_version', 'index', 'sample_index', 'task_hash')),
             'original native task identity must match every rollout')
        need(row.get('sampling') == receipt(context, nonce), 'complete-group sampling receipt mismatch')
        need(row.get('classification') in ('positive', 'negative', 'neutral'), 'known native outcome required')
        need(type(row.get('reward')) in (int, float) and row['reward'] == float(row['classification'] == 'positive'),
             'binary local math reward/classification consistency')
        turns = row.get('turns')
        need(type(turns) is list and len(turns) == 1 and type(turns[0]) is dict, 'complete-group single turn')
        turn = turns[0]; prompt = turn.get('prompt'); output = turn.get('output')
        need(type(prompt) is list and prompt and type(output) is list and 0 < len(output) <= horizon
             and all(type(t) is int and t >= 0 for t in prompt+output), 'original token framing')
        stop = turn.get('stop_reason')
        need(stop in ('eos', 'budget') and (stop != 'budget' or len(output) == horizon and row['classification'] == 'neutral'),
             'budget cap remains unresolved')
        need(turn.get('classification') == row['classification'] and turn.get('reward') == row['reward']
             and turn.get('done') is True, 'same native result on rollout and turn')
        _validate_witness(turn.get('behavior_sampling'), prompt, output, nonce, context, stop)
    # No generated-content hash uniqueness requirement: distinct assigned
    # attempts may legitimately produce the same tokens.
    need(meets_minima(Counter(r['classification'] for r in rows), contract),
         'complete original group does not satisfy mixed-outcome minima')


def search(actor, index, *, seed=0, max_attempts=None, env_id=None):
    from .miner import make_runtime
    from .protocol import entry, harness_for
    from .forced_sampling import receipt
    contract = actor.batch_contract
    need(type(seed) is int and seed == 0, 'complete groups always use nonce 0..N-1; no offset')
    need(max_attempts is None or type(max_attempts) is int and max_attempts >= contract.group_size,
         'local search budget must be an integer at least signed N; exactly N attempts are generated')
    _deadline(actor)
    definition = entry(actor.manifest, env_id)
    env_id = definition['env_id']
    key = task_key(actor, env_id, index)
    for batch, _ in actor.batches:
        if batch['env_id'] == env_id and batch['index'] == index:
            return batch  # Same complete group, never another slot or model call.
    old = actor._group_journal.status(key)
    if old == 'filtered':
        raise GroupFiltered('canonical group already completed without mixed outcomes')
    if old == 'replaced':
        raise GroupFiltered('group was replaced out of the committed set; its draws are fixed, pick another task')
    if old == 'native_error':
        raise GroupIndeterminate('original native task error; no replacement rollout permitted')
    need(len(actor.batches) < contract.max_batches, 'signed complete-group batch cap reached')
    actor._progress('task_start', env_id=env_id, index=index)
    records = actor._group_journal.begin(key)
    resolved = harness_for(definition, index)
    if key not in actor.runtimes:
        if actor.runtime is None:
            actor.runtime = make_runtime(actor.checkpoint, actor.manifest, definition['spec'],
                                         resolved, miner=actor.identity.id)
            if getattr(actor, '_mimo_replicas', 1) != 1:
                from .mimo_parallel_mining import wrap_runtime
                actor.runtime = wrap_runtime(actor.runtime, actor=actor,
                                             replicas=actor._mimo_replicas)
            actor.runtimes[key] = actor.runtime
        else:
            actor.runtimes[key] = actor.runtime.for_environment(definition['spec'], resolved)
    runtime = actor.runtimes[key]
    actor._group_transport.check_runtime(runtime, actor.manifest)
    need(runtime.spec.max_turns == 1, 'complete-group initial single-turn math scope')
    for nonce in range(len(records), contract.group_size):
        _deadline(actor)
        actor._progress('attempt_start', env_id=env_id, index=index, attempt=nonce)
        started = time.monotonic()
        # A stale witness cannot be reused following any failed generation.
        runtime.complete_group_sample_evidence = None
        try:
            row, arrays = runtime.rollout(index, nonce)
            _deadline(actor)
            row = _rollout_from_runtime(runtime, row, nonce)
            need(row.get('sampling') == receipt(actor._group_context, nonce), 'runtime sampling receipt mismatch')
            need(row.get('index') == index and row.get('sample_index') == index
                 and row.get('env_id') == env_id and row.get('environment_version') == runtime.spec.version,
                 'native runtime changed authorized task')
            task_hash = row.get('task_hash')
            need(type(task_hash) is str and len(task_hash) == 64 and set(task_hash) <= set('0123456789abcdef'),
                 'native reset task digest required')
            if records:
                need(task_hash == records[0][0]['task_hash'], 'native reset changed task across nonces')
            actor._group_journal.append(key, row, arrays)
            records.append((row, arrays))
        except TaskError as error:
            actor._group_journal.finish(key, 'native_error', reason=type(error).__name__)
            actor._progress('attempt_end', env_id=env_id, index=index, attempt=nonce,
                            outcome='indeterminate', elapsed=time.monotonic()-started, error_type=type(error).__name__)
            raise GroupIndeterminate('native result missing; no negative or replacement attempt') from error
        except BaseException as error:
            actor._progress('attempt_end', env_id=env_id, index=index, attempt=nonce,
                            outcome='error', elapsed=time.monotonic()-started, error_type=type(error).__name__)
            raise  # Prefix remains original. Resume only this same missing nonce.
        actor._progress('attempt_end', env_id=env_id, index=index, attempt=nonce,
                        outcome=row['classification'], elapsed=time.monotonic()-started)
    rows, arrays = [x[0] for x in records], [x[1] for x in records]
    if not meets_minima(Counter(r['classification'] for r in rows), contract):
        actor._group_journal.finish(key, 'filtered')
        actor._progress('task_exhausted', env_id=env_id, index=index)
        raise GroupFiltered('complete original group lacks required mixed outcomes')
    batch = dict(schema=BATCH_VERSION, epoch=actor.manifest['epoch'], checkpoint=actor.manifest['checkpoint']['id'],
                 miner=actor.identity.id, env_id=env_id, environment_version=runtime.spec.version,
                 sample_index=index, index=index, task_hash=rows[0]['task_hash'],
                 group_contract_sha256=digest(actor.manifest['group_contract']),
                 sampling_context_sha256=digest(actor._group_context), rollouts=rows)
    validate_group(batch, actor.manifest, contract, actor._group_context, actor.identity.id)
    _deadline(actor)
    prepared = actor._group_transport.prepare(batch, arrays, actor.manifest)
    candidate = actor._group_prepared + [(batch, prepared)]
    _deadline(actor)
    # Durable accepted bytes come first. Restart reconciles a crash before the
    # terminal ledger marker; no second preparation or credit is generated.
    actor._group_transport.persist(actor.manifest, actor.identity.id, actor.state_path, candidate)
    actor._group_journal.finish(key, 'accepted', batch_sha256=digest(batch))
    actor._group_prepared = candidate
    actor.batches.append((batch, None))
    actor._progress('batch_complete', env_id=env_id, index=index)
    return batch


def replace(actor, keep):
    """Replace until seal: commit the kept subset (new groups can be added after).

    Requires the signed batch lifetime policy. The previous commitment stays in
    force until the new one is written, and nothing changes after the deadline."""
    from .batch_lifetime import policy
    need(policy(actor.manifest) is not None, 'this epoch does not allow replacing a committed batch')
    _deadline(actor)
    kept = [(b, p) for b, p in actor._group_prepared if keep(b)]
    dropped = [b for b, _ in actor._group_prepared if not keep(b)]
    actor._group_transport.persist(actor.manifest, actor.identity.id, actor.state_path, kept)
    for b in dropped:  # after persist: a crash leaves extra 'accepted' rows, tolerated under replace
        actor._group_journal.release(task_key(actor, b['env_id'], b['index']))
    actor._group_prepared = kept
    actor.batches = [(b, None) for b, _ in kept]
    return actor._group_transport.upload(actor.identity, actor.manifest, actor.cap, actor._group_prepared)


def carry(actor, old_manifest, old_state_path):
    """Head-bound carry-over: re-stamp the previous epoch's groups on an unchanged head.

    No resampling. Returns the number of groups carried (0 when the head changed)."""
    from .batch_lifetime import carries, restamp
    from .complete_group_codec import decode, limits
    if actor._group_prepared or not carries(old_manifest, actor.manifest):
        return 0
    from .forced_sampling import binding
    old_context = binding(old_manifest, actor.identity.id)
    old = type(actor._group_transport)().restore(old_manifest, actor.identity.id, old_state_path)
    out = []
    for batch, data in old[:actor.batch_contract.max_batches]:
        _, arrays = decode(data, limits(old_manifest))[0]
        moved = restamp(batch, old_context, actor._group_context)
        validate_group(moved, actor.manifest, actor.batch_contract, actor._group_context, actor.identity.id)
        out.append((moved, actor._group_transport.prepare(moved, arrays, actor.manifest)))
    if not out:
        return 0
    for b, _ in out:  # journal first: a crash before persist leaves tolerated extra rows
        actor._group_journal.adopt(task_key(actor, b['env_id'], b['index']), b)
    actor._group_transport.persist(actor.manifest, actor.identity.id, actor.state_path, out)
    actor._group_prepared = out
    actor.batches = [(b, None) for b, _ in out]
    return len(out)


def upload(actor):
    _deadline(actor)
    for batch, _ in actor._group_prepared:
        validate_group(batch, actor.manifest, actor.batch_contract, actor._group_context, actor.identity.id)
    return actor._group_transport.upload(actor.identity, actor.manifest, actor.cap, actor._group_prepared)
