"""Prospective complete-group public draws; no historical execution claim.

Inputs come from an authenticated, source-pinned manifest and the native task
resolver. A miner may search tasks, but cannot add a nonce, prompt prefix, salt
or harness alias to create another group for a canonical task. This module is
not manifest authentication, native task resolution or a complete batch grader.
"""
from copy import deepcopy
import hashlib
import json
import math

VERSION = 'forced-inverse-cdf-complete-group-v1'
from .complete_group_distribution import MIMO_VERSION, geometry, eos_ids, distribution, pick
VERSIONS = (VERSION, MIMO_VERSION)
DOMAIN = 'affine-complete-group-token-draw-v1'
EVIDENCE_VERSION = 'sampler-selected-behavior-logprobs-v1'
SAMPLING_FIELDS = {'version', 'randomness', 'verification', 'generation', 'calibration', 'support_adjudication'}
CONTEXT_FIELDS = {'contract', 'group_contract', 'epoch', 'checkpoint', 'checkpoint_descriptor_sha256',
                  'model_runtime_revision', 'backend_profile_sha256', 'runtime_profile_sha256',
                  'harness_source_hash', 'environment_revision', 'environments', 'required_client_features', 'token_geometry'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value, name):
    need(type(value) is str and len(value) == 64 and set(value) <= set('0123456789abcdef'), name + ': lowercase SHA256 required')


def integer(value, low, high, name):
    need(type(value) is int and low <= value <= high, name + ': bounded integer required')
    return value


def validate_sampling(value):
    from .fast_prefill_audit import calibration
    need(type(value) is dict and set(value) == SAMPLING_FIELDS and value['version'] in VERSIONS,
         'exact complete-group sampler fields; no independent attempt limit or salt')
    sha(value['randomness'], 'public randomness')
    need(value['verification'] == 'prefill-cdf-calibrated'
         and value['generation'] == 'cached-eager-inverse-cdf'
         and value['support_adjudication'] == 'exact-cached-replay-v1', 'complete-group sampling profile')
    calibration(value['calibration'])
    return deepcopy(value)


def _group(value):
    from .complete_group_contract import GROUP_FIELDS, GROUP_VERSION, OUTCOME_POLICIES
    need(type(value) is dict and set(value) == GROUP_FIELDS and value['version'] == GROUP_VERSION,
         'exact complete group contract')
    n = integer(value['group_size'], 2, 128, 'group_size')
    good = integer(value['min_success'], 1, n, 'min_success')
    wrong = integer(value['min_completed_failure'], 1, n, 'min_completed_failure')
    need(good + wrong <= n and value['outcome_policy'] in OUTCOME_POLICIES, 'complete group outcome minima/policy')
    return n



def token_geometry(policy):
    return geometry(policy)


def geometry_policy(value):
    from .complete_group_distribution import MIMO_TRANSPORT
    return dict(version=MIMO_TRANSPORT if 'eos_token_ids' in value else 'complete-group-bounded-selected-v1', **value)


def validate_geometry(runtime, context):
    geometry=context['token_geometry']
    need(type(geometry) is dict, 'exact bound token geometry')
    token_geometry(geometry_policy(geometry))
    try:actual_vocab=len(runtime.tokenizer)
    except (TypeError,AttributeError) as exc:raise ValueError('actual tokenizer vocabulary geometry unavailable')from exc
    need(type(actual_vocab) is int and actual_vocab==geometry['vocab_size'], 'tokenizer support differs from signed support')
    stops = eos_ids(geometry)
    actual_stops = tuple(getattr(runtime, 'eos_token_ids', (runtime.tokenizer.eos_token_id,)))
    need(actual_stops == stops and runtime.tokenizer.eos_token_id in stops, 'actual generation EOS differs from signed EOS')
    if hasattr(runtime,'vocab'):
        need(type(runtime.vocab) is int and runtime.vocab==actual_vocab,'runtime support differs from signed tokenizer support')
    config=getattr(runtime.model,'config',None)
    config=getattr(config,'text_config',config)
    head=getattr(config,'vocab_size',None)
    need(type(head) is int and head>=actual_vocab,'pinned model head cannot cover signed tokenizer support')
    limit=getattr(config,'max_position_embeddings',None)
    need(type(limit) is int and limit>=geometry['max_context_tokens'],'model positional context below signed context')
    output_embeddings=getattr(runtime.model,'get_output_embeddings',None)
    if callable(output_embeddings):
        layer=output_embeddings()
        need(layer is not None and hasattr(layer,'weight') and layer.weight.shape[0]==head,'actual model head disagrees with pinned config')
    return geometry


def supported_logits(runtime, values):
    geometry=validate_geometry(runtime,runtime.sampling_context)
    config=runtime.model.config
    head=getattr(config,'text_config',config).vocab_size
    need(values.ndim>=1 and values.shape[-1]==head, 'actual model logits disagree with pinned head geometry')
    return values[...,:geometry['vocab_size']]

def binding(manifest, miner, *, expected_source_hash):
    from .batch_contract import for_manifest, REQUIRED_CLIENT_FEATURES, CompleteGroupContract
    from .protocol import _entries
    from .harness import source_hash
    parsed = for_manifest(manifest, supported_features=REQUIRED_CLIENT_FEATURES)
    need(isinstance(parsed, CompleteGroupContract), 'complete group manifest required')
    need(manifest.get('sampling_source_hash') == expected_source_hash, 'trusted sampling source mismatch')
    epoch = manifest.get('epoch')
    need(type(epoch) is str and bool(epoch), 'signed epoch identity required')
    cp = manifest.get('checkpoint')
    need(type(cp) is dict, 'checkpoint descriptor required')
    sha(cp.get('id'), 'checkpoint')
    for name in ('model_runtime_revision', 'environment_revision'):
        need(type(manifest.get(name)) is str and bool(manifest[name]), name + ' required')
    need(type(manifest.get('backend_profile')) is dict and bool(manifest['backend_profile']), 'backend profile required')
    need(type(manifest.get('runtime_profile', {})) is dict, 'runtime profile required')
    # Explicit environment registry only; unknown aliases never choose a draw.
    need(type(manifest.get('environments')) is list, 'complete groups require explicit environment registry')
    need(manifest.get('harness_source_hash') == source_hash(), 'trusted harness source mismatch')
    environments = deepcopy(_entries(manifest, source_hash()))
    for row in environments:
        need(type(row) is dict and not set(row) - {'env_id', 'spec', 'harness', 'indices', 'evaluation_only'},
             'canonical environment entry fields required')
        spec = row['spec']
        need(type(spec.get('version')) is str and bool(spec['version']), 'environment version required')
        integer(spec.get('max_turns'), 1, 1024, 'max_turns')
        integer(spec.get('num_samples'), 1, 2**31-1, 'num_samples')
        need(all(i < spec['num_samples'] for i in row['indices']), 'task index outside native registry')
        need(row['indices'] == sorted(row['indices']), 'canonical sorted mining indices required')
    need([r['env_id'] for r in environments] == sorted(r['env_id'] for r in environments), 'canonical sorted environment registry required')
    context = dict(contract=deepcopy(manifest['sampling_contract']), group_contract=deepcopy(manifest['group_contract']),
                   epoch=epoch, checkpoint=cp['id'], checkpoint_descriptor_sha256=digest(cp),
                   model_runtime_revision=manifest['model_runtime_revision'],
                   backend_profile_sha256=digest(manifest['backend_profile']),
                   runtime_profile_sha256=digest(manifest.get('runtime_profile', {})),
                   harness_source_hash=manifest['harness_source_hash'],
                   environment_revision=manifest['environment_revision'], environments=environments,
                   required_client_features=sorted(manifest['required_client_features']),
                   token_geometry=token_geometry(manifest.get('group_transport_policy')))
    if parsed.sampling_version == MIMO_VERSION:
        from .complete_group_distribution import MIMO_TRANSPORT
        # Read URLs are renewable transport capabilities, not model identity.
        # Historical complete-group draws keep their original full descriptor.
        need(type(cp.get('files')) is dict and bool(cp['files']), 'MiMo checkpoint file inventory required')
        context['checkpoint_descriptor_sha256'] = digest({k: cp[k] for k in ('id', 'files')})
        need(manifest['group_transport_policy']['version'] == MIMO_TRANSPORT, 'MiMo EOS geometry requires its version')
        need(type(manifest.get('run_id')) is str and bool(manifest['run_id']), 'explicit fresh run identity')
        sha(manifest.get('source_bundle', {}).get('sha256'), 'signed source bundle')
        need(type(manifest.get('mimo_runtime')) is dict and bool(manifest['mimo_runtime']), 'signed MiMo runtime spec required')
        context.update(run_id=manifest['run_id'], source_bundle_sha256=manifest['source_bundle']['sha256'],
                       mimo_runtime_sha256=digest(manifest['mimo_runtime']))
        from .mimo_answer_projection import policy as answer_policy, POLICY_KEY
        answer = answer_policy(manifest)
        if answer is not None:
            context[POLICY_KEY] = answer
        for row in environments:
            choices = row['harness']['by_index'].values() if row['harness'].get('version') == 'indexed-harness-v1' else [row['harness']]
            for h in choices:
                need(h.get('version') == 'text-tools-mimo-complete-group-v1', 'MiMo sampler requires its exact harness')
                need(h['max_context_tokens'] == context['token_geometry']['max_context_tokens'], 'harness/transport context must agree')
    from .complete_group_harness import bind_manifest
    long_context = bind_manifest(manifest, environments)
    if long_context is not None:
        context['long_context_contract'] = long_context
    if miner is not None:
        sha(miner, 'authenticated miner public identity')
        context['miner'] = miner
    return context


def validate_attempt(context, attempt):
    mimo_fields = {'run_id','source_bundle_sha256','mimo_runtime_sha256'} if context.get('contract', {}).get('version') == MIMO_VERSION else set()
    if mimo_fields and 'mimo_answer_projection' in context:
        from .mimo_answer_projection import policy as answer_policy, POLICY_KEY
        answer_policy({POLICY_KEY: context[POLICY_KEY], 'sampling_contract': context['contract']})
        mimo_fields.add(POLICY_KEY)
    need(type(context) is dict and set(context) == CONTEXT_FIELDS | {'miner'} | mimo_fields | ({'long_context_contract'} if 'long_context_contract' in context else set()),
         'exact miner-bound group sampling context; no alternate salt')
    sha(context['miner'], 'authenticated miner public identity')
    validate_sampling(context['contract'])
    geometry=context['token_geometry']
    need(type(geometry) is dict, 'exact bound token geometry')
    token_geometry(geometry_policy(geometry))
    n = _group(context['group_contract'])
    integer(attempt, 0, n-1, 'complete group nonce')


def _entry(context, env_id, index):
    need(type(env_id) is str, 'canonical environment id required')
    integer(index, 0, 2**31-1, 'task index')
    rows = [row for row in context['environments'] if row['env_id'] == env_id]
    need(len(rows) == 1 and index in rows[0]['indices'] and not rows[0].get('evaluation_only', False),
         'task not in signed canonical mining registry')
    return rows[0]


def validate_runtime(runtime, context, *, index=None):
    from .harness import normalize
    from .forced_sampling import validate_harness
    validate_attempt(context, 0)
    validate_geometry(runtime,context)
    rows = [r for r in context['environments'] if r['env_id'] == runtime.spec.id]
    need(len(rows) == 1 and getattr(runtime, 'env_config', None) == rows[0]['spec'], 'runtime native environment differs from signed registry')
    h = validate_harness(runtime.harness, context['contract'])
    from .complete_group_harness import validate_runtime as validate_long_runtime
    validate_long_runtime(runtime, context)
    declared = rows[0]['harness']
    if index is not None:
        from .protocol import harness_for
        selected = _entry(context, runtime.spec.id, index)
        need(normalize(harness_for(selected, index)) == h, 'runtime harness differs from exact signed task harness')
        return selected
    if declared.get('version') == 'indexed-harness-v1':
        need(any(normalize(v) == h for v in declared['by_index'].values()), 'runtime harness outside signed registry')
    else:
        need(normalize(declared) == h, 'runtime harness differs from signed registry')
    return rows[0]


def prepare_draws(context, env_id, task_hash, index, attempt, turn, prompt):
    from .protocol import harness_for
    from .harness import normalize
    validate_attempt(context, attempt)
    row = _entry(context, env_id, index)
    sha(task_hash, 'native canonical task hash')
    integer(turn, 0, row['spec']['max_turns']-1, 'environment turn')
    need(type(prompt) is list and 1 <= len(prompt) <= 1_000_000
         and all(type(t) is int and 0 <= t < 2**31 for t in prompt), 'regenerated canonical prompt tokens required')
    h = normalize(harness_for(row, index))
    from .complete_group_harness import validate_prompt
    validate_prompt(context, h, prompt)
    geometry=context['token_geometry']
    need(all(t<geometry['vocab_size'] for t in prompt),'canonical prompt token outside signed support')
    need(len(prompt)+h['max_output_tokens']<=geometry['max_context_tokens'],'full signed generation horizon exceeds context')
    prefix = bytes.fromhex(digest(dict(domain=DOMAIN, binding_sha256=digest(context),
                                       environment=env_id, environment_version=row['spec']['version'],
                                       task_hash=task_hash, index=index, attempt=attempt, turn=turn,
                                       prompt_sha256=digest(prompt), harness_sha256=digest(h))))
    horizon = h['max_output_tokens']

    def at(position):
        integer(position, 0, horizon-1, 'token position')
        raw = hashlib.sha256(prefix + position.to_bytes(8, 'big')).digest()
        return (int.from_bytes(raw[:8], 'big') >> 11) / 2**53
    return at


def cached_sample(runtime, prompt, attempt, turn, index, task_hash):
    """Capture the actual normalized draw probability, not raw T=1 logprobs.

    This compact miner witness still needs trusted old-policy preparation before
    GRPO training. Matching fingerprints do not authenticate claimed logprobs.
    """
    import torch
    from .forced_sampling import validate_harness
    runtime.complete_group_sample_evidence = None
    context = runtime.sampling_context
    validate_runtime(runtime, context, index=index)
    draw = prepare_draws(context, runtime.spec.id, task_hash, index, attempt, turn, prompt)
    config = validate_harness(runtime.harness, context['contract'])
    device = next(runtime.model.parameters()).device
    output, behavior = [], []
    cache = None
    inputs = torch.tensor([prompt], device=device)
    with torch.inference_mode():
        for position in range(config['max_output_tokens']):
            if callable(getattr(runtime, 'next_token_logits', None)):
                logits, next_cache = runtime.next_token_logits(inputs[0].tolist(), past_key_values=cache, use_cache=True)
            else:
                result = runtime.model(inputs, past_key_values=cache, use_cache=True)
                logits, next_cache = result.logits[0, -1], result.past_key_values
            values = supported_logits(runtime, logits)
            if context['contract']['version'] == VERSION:
                # Keep the originally reviewed v1 float32 normalization order.
                probs = torch.softmax(values.float() / config['temperature'], dim=-1)
                if config['top_p'] < 1:
                    sorted_probs, ids = torch.sort(probs, descending=True, stable=True)
                    sorted_probs = torch.where(sorted_probs.cumsum(0)-sorted_probs >= config['top_p'], 0., sorted_probs)
                    probs = torch.zeros_like(probs).scatter(0, ids, sorted_probs)
                    probs = probs / probs.sum()
                probs = probs.double(); probs /= probs.sum()
            else:
                probs = distribution(values, config['temperature'], config['top_p'], config['top_k'])
            token = pick(probs, draw(position))
            output.append(token)
            behavior.append(float(probs[token].log()))
            cache = next_cache
            inputs = torch.tensor([[token]], device=device)
            if token in eos_ids(context['token_geometry']):
                break
    runtime.complete_group_sample_evidence = dict(version=EVIDENCE_VERSION, attempt=attempt, turn=turn,
        binding_sha256=digest(context), prompt_sha256=digest(prompt), output_sha256=digest(output),
        behavior_selected_logprobs=behavior, stop_reason='eos' if output[-1] in eos_ids(context['token_geometry']) else 'budget')
    return output


def validate_report_batches(context, batches):
    """Sampling coverage only; native outcome grading remains the audit's job."""
    need(type(batches) is list, 'accepted batch list required')
    if not batches:
        return
    validate_attempt(context, 0)
    n = context['group_contract']['group_size']
    seen = set()
    for batch in batches:
        need(type(batch) is dict and batch.get('schema') == 'complete-canonical-group-batch-v1'
             and batch.get('epoch') == context['epoch'] and batch.get('miner') == context['miner']
             and batch.get('checkpoint') == context['checkpoint'], 'accepted group version/epoch/miner/checkpoint')
        need(batch.get('group_contract_sha256') == digest(context['group_contract'])
             and batch.get('sampling_context_sha256') == digest(context), 'accepted group contract binding')
        index = batch.get('index')
        need(batch.get('sample_index', index) == index, 'accepted canonical task index')
        env = batch.get('env_id')
        definition = _entry(context, env, index)
        need(batch.get('environment_version') == definition['spec']['version'], 'accepted environment version')
        sha(batch.get('task_hash'), 'accepted canonical task hash')
        key = (env, index)
        need(key not in seen, 'accepted canonical group cannot receive duplicate credit')
        seen.add(key)
        rows = batch.get('rollouts')
        need(type(rows) is list and len(rows) == n, 'accepted group requires every original nonce')
        task_hash = batch['task_hash']
        for nonce, row in enumerate(rows):
            need(type(row) is dict and type(row.get('seed')) is int and row['seed'] == nonce,
                 'accepted group nonce sequence must be exact 0..N-1')
            need(row.get('env_id') == env and row.get('index') == index
                 and row.get('sample_index', index) == index, 'accepted rollout canonical task mismatch')
            sha(row.get('task_hash'), 'accepted native task hash')
            need(row.get('environment_version') == batch['environment_version'], 'accepted rollout environment version')
            need(row['task_hash'] == task_hash, 'accepted group task hash mismatch')
