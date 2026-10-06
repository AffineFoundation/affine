"""Default-off research: checkpoint-consistent tokens, not execution evidence.

Callers must authenticate the original manifest/checkpoint and task allowlist.
No existing miner, verifier, reward policy or source inventory imports this module.
"""
import math

from .threeway_prefill_research import VERSION,POLICY,validate_policy

def document(rollout):
    """Export tokens/context/outcomes; probabilities and TOPLOC are absent."""
    import copy
    value = copy.deepcopy(rollout)
    for turn in value['turns']:
        turn.pop('proofs', None)
    return value


def verify(runtime, manifest, rollout, *, eligible_indices):
    from . import forced_sampling, harness
    from .environments import create_session
    from .audit_policy import InvalidSample
    validate_policy(manifest.get('token_only_verification_policy'))
    context = forced_sampling.binding(manifest)
    if context is None or context != getattr(runtime, 'sampling_context', None):
        raise InvalidSample('original checkpoint/public draw binding')
    forced_sampling.validate_harness(runtime.harness, context['contract'])
    from .fast_prefill_audit import bind, SUPPORT_VERSION
    from .threeway_prefill_research import verify_sampling
    if context['contract']['version'] != SUPPORT_VERSION:
        raise ValueError('explicit calibrated prefill support-adjudication contract required')
    calibration = bind(manifest, runtime.harness)
    if calibration != getattr(runtime, 'fast_sampling_calibration', None):
        raise InvalidSample('checkpoint-bound calibration')
    if rollout.get('schema') != 2 or rollout.get('env_id') != runtime.spec.id or rollout.get('environment_version') != runtime.spec.version:
        raise InvalidSample('environment binding')
    index = rollout.get('index')
    if type(index) is not int or index not in eligible_indices or rollout.get('sample_index') != index:
        raise InvalidSample('approved task eligibility')
    try:
        expected = forced_sampling.receipt(context, rollout.get('seed'))
    except ValueError as error:
        raise InvalidSample('attempt budget') from error
    if rollout.get('sampling') != expected:
        raise InvalidSample('original attempt/public draw receipt')
    turns = rollout.get('turns')
    if type(turns) is not list or not 1 <= len(turns) <= runtime.spec.max_turns:
        raise InvalidSample('trajectory length')
    session = create_session(runtime.spec)
    try:
        env_seed = int(runtime.spec.config.get('seed', 0))
        if rollout.get('env_seed') != env_seed:
            raise InvalidSample('environment seed')
        initial = session.reset(index, env_seed)
        if rollout.get('task_hash') != initial['task_hash']:
            raise InvalidSample('task hash')
        messages, tools = initial['messages'], initial.get('tools', [])
        for number, turn in enumerate(turns):
            if type(turn) is not dict or 'proofs' in turn or 'probabilities' in turn:
                raise InvalidSample('token-only transport framing')
            prompt = runtime.prompt(messages, tools)
            if turn.get('prompt') != prompt:
                raise InvalidSample('canonical prompt')
            output = turn.get('output')
            if (type(output) is not list or not 0 < len(output) <= min(runtime.spec.max_output_tokens, runtime.harness['max_output_tokens'])
                    or any(type(t) is not int or not 0 <= t < runtime.model.config.vocab_size for t in output)):
                raise InvalidSample('token framing')
            if len(prompt) + len(output) > min(getattr(runtime.model.config, 'max_position_embeddings', 8192), 8192):
                raise InvalidSample('model context')
            eos = runtime.tokenizer.eos_token_id
            if eos in output[:-1] or len(output) < runtime.harness['max_output_tokens'] and output[-1] != eos:
                raise InvalidSample('stop framing')
            # One causal teacher-forced model pass for every output position.
            # No activation fingerprints or uploaded probability arrays are read.
            import torch
            device = next(runtime.model.parameters()).device
            with torch.inference_mode():
                ids = torch.tensor([prompt + output], device=device)
                computed = runtime.model(ids, use_cache=False)
                logits = computed.logits[0, len(prompt)-1:len(prompt)+len(output)-1].float()
                logprobs = torch.log_softmax(logits, -1)
            # Preserve prescribed public draws and calibrated CDF bounds.
            # Uncertain boundaries stop as inconclusive; this research policy
            # never invokes cached autoregressive adjudication.
            verify_sampling(runtime, rollout, number, prompt, output, logprobs)
            text = runtime.tokenizer.decode(output, skip_special_tokens=True)
            if turn.get('text') != text:
                raise InvalidSample('decoded text')
            result = session.step(harness.action(text, runtime.harness))
            for scope in ((turn, rollout) if number == len(turns)-1 else (turn,)):
                reward = scope.get('reward')
                if type(reward) not in (int, float) or not math.isfinite(reward) or reward != result['reward'] or scope.get('classification') != result['classification']:
                    raise InvalidSample('environment outcome')
            if type(turn.get('done')) is not bool or turn['done'] != result['done'] or turn.get('observations') != result['observations']:
                raise InvalidSample('environment replay')
            if result['done'] != (number == len(turns)-1):
                raise InvalidSample('incomplete or extra trajectory')
            messages = messages + [dict(role='assistant', content=text)] + harness.observations(result['observations'], runtime.harness)
        return {'valid': True, 'assurance': 'checkpoint-consistent-prescribed-token-sequence',
                'historical_execution_proven': False, 'miner_probability_claims_verified': False,
                'TOPLOC_verified': False, 'sampler_check': 'calibrated-interior-prefill-threeway-no-cached-fallback'}
    finally:
        session.close()


def verify_pair(runtime, manifest, rollouts, *, eligible_indices):
    from .audit_policy import InvalidSample
    if type(rollouts) is not list or len(rollouts) != manifest['K'] + manifest['L']:
        raise InvalidSample('signed pair quota')
    indices = {r.get('index') for r in rollouts}
    sequences = {tuple(tuple(t['output']) for t in r['turns']) for r in rollouts}
    if len(indices) != 1 or len(sequences) != len(rollouts):
        raise InvalidSample('pair task/duplicate binding')
    results = [verify(runtime, manifest, r, eligible_indices=eligible_indices) for r in rollouts]
    if sum(r['classification'] == 'positive' for r in rollouts) != manifest['K'] or sum(r['classification'] == 'negative' for r in rollouts) != manifest['L']:
        raise InvalidSample('signed outcome quotas')
    return results
