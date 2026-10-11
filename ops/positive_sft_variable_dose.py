"""Isolated, prospective SFT routing; not installed in a live controller.

The caller authenticates native admission before creating the pair-snapshot
receipt. This helper checks that signed binding, but does not replay grading or
sampling. Train routes still require a separately qualified execution release,
actual-step LR grant, and the ordinary original-job/evidence machinery. An empty
route never enters that machinery or invents optimizer evidence.
"""
import copy
import hashlib
import json
import os
import re
from pathlib import Path

from subnet.covered_epoch_optimizer import pair_identity
from subnet.distributed_roles import authenticate
from subnet.persistent_cpu_adamw import checkpoint_id
from subnet.persistent_training_protocol import validate_pointer
from subnet.positive_only_objective import configuration, COMPLETION_POLICY
from subnet.positive_only_training import _positive_completion, fixed_task_minibatches
from subnet.storage import canonical
from subnet.task_normalized_training import task_groups
from subnet.trajectory_identity import token_trace_sha256

VERSION = 'positive-sft-variable-coverage-candidate-v1'
RELEASE_VERSION = 'positive-sft-fallback-scope-candidate-v1'
ADMISSION_VERSION = 'positive-sft-admitted-pair-snapshot-candidate-v1'
NO_UPDATE_VERSION = 'positive-sft-parent-preserving-no-update-candidate-v1'
MAX_METADATA_BYTES = 16*1024**2


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def policy(value):
    require(type(value) is dict and set(value) == {
        'target_tasks_per_step', 'max_tasks', 'max_steps'}, 'exact fallback capacity policy')
    require(all(type(v) is int for v in value.values()), 'integer fallback capacities')
    require(value['target_tasks_per_step'] == 64 and 1 <= value['max_tasks'] <= 512
            and 1 <= value['max_steps'] <= 8
            and (value['max_tasks']+63)//64 <= value['max_steps'],
            'release covers all eligible tasks within at most eight updates')
    return dict(value)


def selection(pairs, *, seed, eos_token_id, required_pairs_per_task, capacity):
    """Complete K/L admission followed by EOS filtering, including honest zero.

    This mirrors the pinned candidate's selection, reusing its completion and
    grouping helpers. No responses are rescored, appended, or drawn from history.
    Upstream selection must already respect the release's task cap: excess input
    is an error, not silently discarded work.
    """
    cap = policy(capacity)
    checkpoint_id(seed)
    require(type(pairs) is list, 'explicit admitted pair list')
    require(type(eos_token_id) is int and eos_token_id >= 0, 'authenticated integer EOS')
    require(type(required_pairs_per_task) is int and 2 <= required_pairs_per_task <= 64,
            'complete manifest K/L quota')
    admitted, input_tasks, identities = [], [], []
    if pairs:
        admitted, input_tasks, _, identities = task_groups(
            pairs, 1, seed, required_pairs_per_task=required_pairs_per_task)
    require(len(input_tasks) <= cap['max_tasks'], 'admitted task population exceeds release')
    by_pair = {i: task for task in input_tasks for i in task['pair_indices']}
    retained, excluded, compact = [], [], {}
    for i, (_, positive, _) in enumerate(admitted):
        task = by_pair[i]
        count, reasons = _positive_completion(positive, eos_token_id)
        row = dict(source_pair_index=i, pair_sha256=identities[i],
            trajectory_sha256=token_trace_sha256(positive['turns']),
            env_id=task['env_id'], index=task['index'], task_hash=task['task_hash'],
            task_sha256=task['task_sha256'], generated_tokens=count,
            turn_count=len(positive['turns']))
        if reasons:
            excluded.append(dict(row, reason='not-terminal-eos-complete', turn_reasons=reasons))
        else:
            compact[i] = len(retained)
            retained.append(dict(row, positive_index=len(retained)))
    tasks, excluded_tasks = [], []
    for task in input_tasks:
        identity = {k: task[k] for k in ('env_id', 'index', 'task_hash', 'task_sha256')}
        indices = [compact[i] for i in task['pair_indices'] if i in compact]
        if indices:
            tasks.append(dict(identity, positive_indices=indices))
        else:
            excluded_tasks.append(dict(identity, reason='no-terminal-eos-positives'))
    n = len(tasks)
    steps = (n+63)//64
    require(steps <= cap['max_steps'], 'actual updates exceed release')
    groups = fixed_task_minibatches(tasks, steps) if n else []
    details = dict(version='terminal-eos-positive-selection-v1',
        completion_policy=COMPLETION_POLICY, eos_token_id=eos_token_id,
        required_pairs_per_task=required_pairs_per_task,
        input_pair_count=len(admitted), input_task_count=len(input_tasks),
        retained=retained, excluded=excluded, excluded_tasks=excluded_tasks,
        generated_tokens=sum(r['generated_tokens'] for r in retained),
        excluded_generated_tokens=sum(r['generated_tokens'] for r in excluded))
    return dict(version=VERSION, disposition='train' if n else 'no_update',
        reason=None if n else ('no-admitted-pairs' if not admitted else 'no-terminal-eos-positives'),
        capacity=cap, task_count=n, positive_count=len(retained), steps=steps,
        tasks=tasks, groups=groups, selection=details,
        source_pair_sha256=identities, admitted_pairs_sha256=sha(pairs))


def route_proposal(pairs, signed_manifest, signed_admission, signed_scope, authority):
    """Check the authority's exact frozen inputs; never accept a worker's count."""
    manifest = authenticate(signed_manifest, authority)
    admission = authenticate(signed_admission, authority)
    scope = authenticate(signed_scope, authority)
    epoch = manifest['epoch']
    require(type(epoch) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,200}', epoch), 'safe epoch')
    require(scope.get('version') == RELEASE_VERSION, 'separate prospective fallback scope')
    require(scope.get('manifest_sha256') == sha(signed_manifest), 'scope original signed manifest')
    objective = configuration(scope['objective'])
    require(manifest.get('training_objective') == objective, 'original manifest SFT objective')
    require(manifest.get('K') == manifest.get('L') == scope['required_pairs_per_task']
            and type(manifest.get('K')) is int and type(manifest.get('L')) is int,
            'unchanged complete original K/L contract')
    require(manifest.get('training_input_policy') == 'committed-unaudited-training-v1',
            'explicit original unaudited training policy')
    checkpoint_id(scope['training_source_sha256'])
    require(manifest['trainer_state_binding']['source_sha256'] == scope['training_source_sha256'],
            'original training source binding')
    require(admission.get('version') == ADMISSION_VERSION
            and admission.get('manifest_sha256') == sha(signed_manifest)
            and admission.get('epoch') == epoch
            and admission.get('checkpoint') == manifest['checkpoint']['id'],
            'authenticated same-epoch same-checkpoint pair snapshot')
    require(admission.get('admitted_pairs_sha256') == sha(pairs)
            and admission.get('pair_sha256') == [pair_identity(p) for p in pairs],
            'complete frozen pair contents including environment definitions')
    require(admission.get('complete') is True and admission.get('pending_input_count') == 0
            and type(admission.get('pending_input_count')) is int
            and admission.get('infrastructure_error_count') == 0
            and type(admission.get('infrastructure_error_count')) is int,
            'incomplete or failed input collection is not an empty population')
    require(admission.get('sampling_assurance') == 'unaudited'
            and admission.get('proof_verification_performed') is False,
            'truthful input assurance')
    checkpoint_id(admission['native_admission_sha256'])
    selected = selection(pairs, seed=scope['seed'], eos_token_id=scope['eos_token_id'],
        required_pairs_per_task=scope['required_pairs_per_task'], capacity=scope['capacity'])
    return dict(version=VERSION, epoch=epoch, manifest_sha256=sha(signed_manifest),
        admission_sha256=sha(signed_admission), scope_sha256=sha(signed_scope),
        training_source_sha256=scope['training_source_sha256'], objective=objective,
        checkpoint=manifest['checkpoint']['id'],
        trainer_state_binding=copy.deepcopy(manifest['trainer_state_binding']),
        seed=scope['seed'], decision=selected, sampling_assurance='unaudited',
        proof_verification_performed=False,
        execution_release_required=selected['disposition'] == 'train')


def dispatch_arguments(route):
    """Arguments only, NOT execution authorization or a trainer invocation."""
    decision = route['decision']
    require(decision['disposition'] == 'train' and decision['steps'] > 0,
            'no optimizer dispatch for empty SFT selection')
    return dict(steps=decision['steps'], seed=route['seed'],
        objective=copy.deepcopy(route['objective']), fallback_route_sha256=sha(route),
        expected_global_step_before=route['trainer_state_binding']['global_step_before'],
        expected_global_step_after=route['trainer_state_binding']['global_step_before']+decision['steps'],
        requires_actual_step_execution_release=True,
        requires_actual_step_learning_rate_authorization=True)


def _read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'ordinary immutable journal')
    require(path.stat().st_size <= MAX_METADATA_BYTES, 'bounded journal')
    raw = path.read_bytes()
    require(len(raw) <= MAX_METADATA_BYTES, 'bounded journal')
    return raw


def _create(path, value):
    """Same atomic no-replace journal pattern as native eligibility lifecycle."""
    path = Path(path); data = canonical(value)
    require(len(data) <= MAX_METADATA_BYTES, 'bounded journal')
    temp = path.parent/('.'+path.name+'.'+os.urandom(12).hex())
    try:
        with temp.open('xb') as stream:
            os.chmod(temp, 0o600)
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            require(_read(path) == data, 'immutable fallback journal collision')
    finally:
        if temp.exists():
            temp.unlink()
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _no_issued_training(state, epoch):
    roles = Path(state)/'roles'
    require(not (roles/(epoch+'-train.json')).exists()
            and not list(roles.glob(epoch+'-train*-job.json')),
            'no-update cannot supersede any issued training job')


def _parent(controller, manifest, status):
    epoch = manifest['epoch']
    _no_issued_training(controller.state, epoch)
    require(status.get('active', {}).get('epoch') == epoch
            and status.get('active', {}).get('phase') == 'train', 'original active train phase')
    require(type(status.get('round')) is int and status['round'] >= 0
            and type(status.get('training_steps')) is int and status['training_steps'] >= 0,
            'original controller round and training counter')
    binding = manifest['trainer_state_binding']; pointer = status['trainer_state']
    validate_pointer(pointer)
    require(status.get('persistent_state_committed') is True
            and binding['parent'] == pointer and binding.get('genesis') is None
            and binding['global_step_before'] == pointer['optimizer_steps']
            and binding['epoch'] == epoch and binding['input_checkpoint'] == manifest['checkpoint']['id']
            and pointer['inference_checkpoint'] == manifest['checkpoint']['id']
            and status['checkpoint']['id'] == manifest['checkpoint']['id'], 'exact committed parent; no reset')
    before = _read(Path(controller.state)/'latest-trainer-state.json')
    require(json.loads(before) == pointer, 'current optimizer pointer matches parent')
    return before


def close_no_update(controller, pairs, signed_manifest, signed_admission, signed_scope, status):
    """Return the existing controller's zero-step after-phase ABI.

    Must run under the learner's existing singleton/boundary lock. This function
    does not advance controller state, alter the optimizer pointer, sign a job,
    touch GPU state, or set rewards. Publication errors propagate; the identical
    receipt is reused on retry before an epoch is considered closed.
    """
    authority = controller.authority.id
    route = route_proposal(pairs, signed_manifest, signed_admission, signed_scope, authority)
    require(route['decision']['disposition'] == 'no_update', 'authenticated zero eligible tasks')
    manifest = authenticate(signed_manifest, authority); epoch = manifest['epoch']
    before = _parent(controller, manifest, status)
    root = Path(controller.state)/(epoch+'-positive-sft-fallback')
    root.mkdir(mode=0o700, exist_ok=True)
    require(root.is_dir() and not root.is_symlink(), 'owned fallback journal directory')
    route_path = root/'route.ROOT-SIGNED.json'
    if route_path.exists():
        signed_route = json.loads(_read(route_path))
        require(authenticate(signed_route, authority) == route, 'immutable original route')
    else:
        signed_route = controller.signed(route); _create(route_path, signed_route)
    payload = dict(version=NO_UPDATE_VERSION, epoch=epoch, round=status['round'],
        status='closed_positive_sft_no_update', reason=route['decision']['reason'],
        route_sha256=sha(signed_route), manifest_sha256=sha(signed_manifest),
        admission_sha256=sha(signed_admission), scope_sha256=sha(signed_scope),
        checkpoint=manifest['checkpoint']['id'], next_checkpoint=manifest['checkpoint']['id'],
        checkpoint_descriptor_sha256=sha(status['checkpoint']),
        trainer_state=copy.deepcopy(status['trainer_state']),
        optimizer_pointer_sha256=hashlib.sha256(before).hexdigest(),
        optimizer_steps=status['trainer_state']['optimizer_steps'],
        training_steps_before=status['training_steps'], training_steps_after=status['training_steps'],
        optimizer_updates=0, training_dispatched=False, training_succeeded=False,
        training_report=None, losses=None, sampling_assurance='unaudited',
        proof_verification_performed=False, claims_rewritten=False,
        cheating_penalties=False, reward_policy_changed=False)
    receipt_path = root/'no-update.ROOT-SIGNED.json'
    if receipt_path.exists():
        receipt = json.loads(_read(receipt_path))
        require(authenticate(receipt, authority) == payload, 'immutable no-update receipt')
    else:
        receipt = controller.signed(payload); _create(receipt_path, receipt)
    key = 'public/'+epoch+'/positive-sft-no-update.json'
    data = canonical(receipt)
    # Repeated PUT is only this exact immutable signed receipt. A published
    # collision is refused; full readback binds closure to durable metadata.
    try:
        previous = controller.bucket.get_bounded(key, limit=len(data))
    except KeyError:
        previous = None
    except Exception as error:
        from botocore.exceptions import ClientError
        if not isinstance(error, ClientError) or str(error.response.get('Error', {}).get('Code')) not in ('NoSuchKey', '404', 'NotFound'):
            raise
        previous = None
    if previous is not None:
        require(previous == data, 'immutable durable no-update collision')
    else:
        controller.bucket.put(key, data)
    require(controller.bucket.get_bounded(key, limit=len(data)) == data,
            'full durable no-update receipt readback')
    require(_parent(controller, manifest, status) == before, 'parent changed during no-update closure')
    return copy.deepcopy(status['checkpoint']), dict(steps=0, checkpoint_path=status['checkpoint_path'],
        trainer_state=copy.deepcopy(status['trainer_state']),
        native_no_update=dict(payload, no_update_receipt_sha256=sha(receipt)))


def completion_fields(controller, epoch):
    """Prospective counterpart for the controller's native_completion_fields hook."""
    require(type(epoch) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,200}', epoch), 'safe epoch')
    root = Path(controller.state)/(epoch+'-positive-sft-fallback')
    receipt = json.loads(_read(root/'no-update.ROOT-SIGNED.json'))
    value = authenticate(receipt, controller.authority.id)
    require(value['version'] == NO_UPDATE_VERSION and value['epoch'] == epoch
            and value['optimizer_updates'] == 0 and value['training_succeeded'] is False
            and value['checkpoint'] == value['next_checkpoint'], 'truthful no-update completion')
    return dict(native_no_update=value, no_update_receipt_sha256=sha(receipt),
                optimizer_updates=0, status=value['status'])
