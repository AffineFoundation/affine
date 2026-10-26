"""Version-dispatched group audits: native grading plus bound token replay.

Structural intake does not invoke this path. Audits may lag training and report
infrastructure failure as indeterminate, never as an invented wrong answer.
"""
import math
import time
import numpy as np

from .audit_policy import InvalidSample
from .complete_group_sampling import prepare_draws, validate_runtime, supported_logits
from .complete_group_distribution import distribution, pick, verify_intervals, eos_ids
from .complete_group_runtime import outcome
from .environments import create_session
from . import harness as policy


def exact_replay(runtime, prompt, output, nonce, index, task_hash):
    import torch
    context = runtime.sampling_context
    validate_runtime(runtime, context, index=index)
    draw = prepare_draws(context, runtime.spec.id, task_hash, index, nonce, 0, prompt)
    stops = eos_ids(context['token_geometry']); h = runtime.harness
    if (not output or len(output) > h['max_output_tokens']
            or any(t in stops for t in output[:-1])
            or len(output) < h['max_output_tokens'] and output[-1] not in stops):
        raise InvalidSample('exact cached terminal framing')
    cache = None; ids = list(prompt)
    device = next(runtime.model.parameters()).device
    with torch.inference_mode():
        for position, token in enumerate(output):
            if callable(getattr(runtime, 'next_token_logits', None)):
                logits, cache = runtime.next_token_logits(ids, past_key_values=cache, use_cache=True)
            else:
                result = runtime.model(torch.tensor([ids], device=device), past_key_values=cache, use_cache=True)
                logits, cache = result.logits[0, -1], result.past_key_values
            p = distribution(supported_logits(runtime, logits), h['temperature'], h['top_p'], h.get('top_k', 0))
            actual = pick(p, draw(position))
            if token != actual:
                raise InvalidSample('original cached group sampler mismatch')
            ids = [actual]
    return dict(positions=len(output), cached_reference_adjudication=True, all_intervals_verified=True)


def sampling_check(runtime, rollout, prompt, output, full_logprobs):
    from .fast_prefill_audit import NumericalAmbiguity, SupportMismatch
    draw = prepare_draws(runtime.sampling_context, runtime.spec.id, rollout['task_hash'],
                         rollout['index'], rollout['seed'], 0, prompt)
    start = time.monotonic()
    replay_seconds = 0.
    try:
        result = verify_intervals(full_logprobs[:, :runtime.sampling_context['token_geometry']['vocab_size']],
            output, [draw(i) for i in range(len(output))], runtime.harness,
            runtime.fast_sampling_calibration['cdf_abs_error'], exact_fallback_on_disagreement=True)
        result['cached_reference_adjudication'] = False
        result['prefill_checked_positions'] = len(output)
        interval_seconds = time.monotonic() - start
    except (NumericalAmbiguity, InvalidSample) as error:
        # MiMo's hybrid cached/prefill numerical agreement must be measured.
        # A failed fast CDF check is never by itself a confirmed forged path:
        # replay the original cached sampler before assigning an invalid result.
        interval_seconds = time.monotonic() - start
        replay_started = time.monotonic()
        result = exact_replay(runtime, prompt, output, rollout['seed'], rollout['index'], rollout['task_hash'])
        replay_seconds = time.monotonic() - replay_started
        result['fast_disagreement'] = type(error).__name__
        result['prefill_checked_positions'] = getattr(error, 'checked_positions', len(output))
    result['prefill_interval_seconds'] = interval_seconds
    result['cached_replay_seconds'] = replay_seconds
    result['interval_implementation'] = 'bounded-exact-support-cpu-v1'
    result['sampling_seconds'] = time.monotonic() - start
    return result


def native_projection(runtime, rollout):
    """Reconstruct prompt and native outcome; no submitted verdict is authority.

    This is usable by trusted training preparation without TOPLOC/sample replay.
    Source-pinned native task/session and checkpoint/tokenizer remain host inputs.
    """
    from .complete_group_sampling import need
    validate_runtime(runtime, runtime.sampling_context, index=rollout['index'])
    session = create_session(runtime.spec)
    try:
        initial = session.reset(rollout['index'], int(runtime.spec.config.get('seed', 0)))
        if initial['task_hash'] != rollout['task_hash']:
            raise InvalidSample('native task hash mismatch')
        prompt = runtime.prompt(initial['messages'], initial.get('tools', []))
        turns = rollout.get('turns')
        need(type(turns) is list and len(turns) == 1, 'exact native single turn')
        turn = turns[0]; output = turn['output']
        stops = eos_ids(runtime.sampling_context['token_geometry'])
        if prompt != turn['prompt'] or not output or any(type(t) is not int or t < 0
              or t >= runtime.sampling_context['token_geometry']['vocab_size'] for t in output):
            raise InvalidSample('native prompt or output tokens')
        if any(t in stops for t in output[:-1]) or len(output) > runtime.harness['max_output_tokens']:
            raise InvalidSample('native terminal framing')
        stop = 'eos' if output[-1] in stops else 'budget'
        if stop == 'budget' and len(output) != runtime.harness['max_output_tokens']:
            raise InvalidSample('incomplete nonterminal output')
        text = runtime.tokenizer.decode(output, skip_special_tokens=True)
        if text != turn['text']:
            raise InvalidSample('generated text differs from token decode')
        from .mimo_answer_projection import grade
        grading_manifest = {'sampling_contract': runtime.sampling_context['contract']}
        if 'mimo_answer_projection' in runtime.sampling_context:
            grading_manifest['mimo_answer_projection'] = runtime.sampling_context['mimo_answer_projection']
        result, kind, reward = grade(session, grading_manifest,
                                     runtime.harness, runtime.tokenizer, output, stop)
        return dict(prompt=prompt, output=list(output), task_hash=initial['task_hash'],
            classification=kind, reward=reward, stop_reason=stop, observations=result['observations'],
            native_classification=result['classification'], done=True)
    finally:
        session.close()


def verify_rollout(runtime, rollout, arrays):
    from .forced_sampling import receipt
    from .proofs import validate_framing
    start = time.monotonic()
    if rollout.get('sampling') != receipt(runtime.sampling_context, rollout.get('seed')):
        raise InvalidSample('group sampling attempt binding')
    projected = native_projection(runtime, rollout)
    native_seconds = time.monotonic() - start
    turn = rollout['turns'][0]; output = projected['output']; prompt = projected['prompt']
    if any(turn.get(k) != projected[k] for k in ('classification','reward','stop_reason','done','observations')):
        raise InvalidSample('native outcome or stopping claim')
    if rollout['classification'] != projected['classification'] or rollout['reward'] != projected['reward']:
        raise InvalidSample('native group outcome claim')
    if projected['stop_reason'] == 'budget' and any(row.get('native_classification') != projected['native_classification']
                                                  for row in (turn, rollout)):
        raise InvalidSample('original capped native classification claim')
    prefill_started = time.monotonic()
    acts, full = runtime.compute(prompt, output)
    prefill_seconds = time.monotonic() - prefill_started
    selected_started = time.monotonic()
    values = np.asarray(full)
    if values.ndim != 2 or values.shape[0] != len(output) or not np.isfinite(values).all():
        raise RuntimeError('invalid recomputed model distributions')
    selected = values[np.arange(len(output)), output].reshape(-1, 1)
    p = runtime.fast_sampling_calibration
    if (type(arrays) is not list or len(arrays) != 1 or arrays[0].shape != selected.shape
            or not np.isfinite(arrays[0]).all() or not np.allclose(arrays[0], selected, atol=p['logprob_atol'], rtol=0)):
        raise InvalidSample('selected T1 probabilities')
    count = 1+math.ceil(len(output)/16)
    selected_seconds = time.monotonic() - selected_started
    toploc_started = time.monotonic()
    try:
        validate_framing(turn['proofs'], count, 128)
    except ValueError as exc:
        raise InvalidSample('TOPLOC framing') from exc
    results = runtime.verify_proofs(acts, turn['proofs'], decode_batching_size=16, topk=128)
    if len(results) != count or any(r.exp_mismatches > p['toploc_exp_mismatches']
        or r.mant_err_mean > p['toploc_mant_err_mean'] or r.mant_err_median > p['toploc_mant_err_median'] for r in results):
        raise InvalidSample('TOPLOC mismatch')
    sampling_started = time.monotonic()
    result = sampling_check(runtime, rollout, prompt, output, values)
    result['native_seconds'] = native_seconds
    result['prefill_seconds'] = prefill_seconds
    result['selected_probability_seconds'] = selected_seconds
    result['toploc_seconds'] = sampling_started - toploc_started
    result['full_verify_seconds'] = time.monotonic() - start
    return result


def verify(data, manifest, checkpoint, *, miner, runtime_factory=None):
    """Audit one committed compact group with an operator-authenticated miner."""
    from .batch_contract import for_manifest, REQUIRED_CLIENT_FEATURES
    from .complete_group_codec import decode, limits
    from .complete_group_admission import TaskAuthority, structural_batch
    from .protocol import entry, harness_for
    from .forced_sampling import assurance
    from .storage import sha
    from .runtime_factory import runtime as default_runtime
    from .complete_group_sampling import need
    need(type(miner) is str and len(miner) == 64 and set(miner) <= set('0123456789abcdef'),
         'authenticated audit miner identity required')
    for_manifest(manifest, supported_features=REQUIRED_CLIENT_FEATURES)
    batch, arrays = decode(data, limits(manifest))[0]
    definition = entry(manifest, batch['env_id']); index = batch['index']
    factory = runtime_factory or default_runtime
    if callable(getattr(factory, 'runtime_for_job', None)):
        lifetime = factory.runtime_for_job(checkpoint, manifest, definition['spec'],
                                           harness_for(definition, index), miner=miner)
    else:
        from contextlib import nullcontext
        lifetime = nullcontext(factory(checkpoint, manifest, definition['spec'],
                                       harness_for(definition, index), miner=miner))
    with lifetime as runtime:
        return _verify_loaded(data, manifest, miner, batch, arrays, definition, index, runtime)


def _verify_loaded(data, manifest, miner, batch, arrays, definition, index, runtime):
    """The caller holds its persistent checkpoint lease through every check."""
    from .complete_group_admission import TaskAuthority, structural_batch
    from .forced_sampling import assurance
    from .storage import sha
    session = create_session(runtime.spec)
    try:
        initial = session.reset(index, int(runtime.spec.config.get('seed', 0)))
        prompt = runtime.prompt(initial['messages'], initial.get('tools', []))
    finally:
        session.close()
    stops = eos_ids(runtime.sampling_context['token_geometry'])
    task = TaskAuthority(manifest['epoch'], manifest['checkpoint']['id'], definition['env_id'],
        runtime.spec.version, index, initial['task_hash'], tuple(prompt), runtime.tokenizer.eos_token_id,
        len(runtime.tokenizer), stops if 'eos_token_ids' in manifest['group_transport_policy'] else ())
    try:
        structural_batch(batch, manifest, miner, task, arrays)
    except (ValueError, KeyError, TypeError) as error:
        raise InvalidSample('complete-group structural admission failed: '+str(error)) from error
    details = [verify_rollout(runtime, row, tensors) for row, tensors in zip(batch['rollouts'], arrays)]
    return dict(epoch=manifest['epoch'], miner=miner, submission_sha256=sha(data),
        selected_batches=[0], sampling_assurance=assurance(manifest, miner), accepted=[batch],
        outcomes=[dict(batch=0, env_id=batch['env_id'], index=index, valid=True, structural_valid=True,
                       fully_audited=True, rollouts=details)],
        training_eligibility='structurally-eligible-unaudited-permitted',
        assurance='all-N-attempts-audited')
