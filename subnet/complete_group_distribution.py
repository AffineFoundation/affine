"""Versioned MiMo draw distribution shared by generation and verification.

The training engine's unfiltered, temperature-scaled OLD probabilities are a
different quantity. This module defines the actual public sampling distribution.
Ties use ascending token ID, top-k precedes top-p, and CDFs use token-ID order.
"""
import math

MIMO_VERSION = 'forced-inverse-cdf-complete-group-mimo-v1'
MIMO_HARNESS = 'text-tools-mimo-complete-group-v1'
MIMO_TRANSPORT = 'complete-group-bounded-selected-mimo-v1'
MIMO_EOS = (248046, 248044)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def geometry(policy):
    need(type(policy) is dict, 'signed tokenizer geometry required')
    mimo = policy.get('version') == MIMO_TRANSPORT
    eos_key = 'eos_token_ids' if mimo else 'eos_token_id'
    need(set(policy) == {'version', 'max_context_tokens', 'vocab_size', eos_key}
         and (mimo or policy['version'] == 'complete-group-bounded-selected-v1'),
         'exact versioned tokenizer geometry fields')
    support = policy['vocab_size']; context = policy['max_context_tokens']
    need(type(support) is int and 2 <= support <= (1_000_000 if mimo else 200_000),
         'bounded vocabulary support')
    need(type(context) is int and 2 <= context <= 65536, 'bounded context')
    if mimo:
        eos = policy[eos_key]
        need(type(eos) is list and all(type(t) is int for t in eos)
             and tuple(eos) == MIMO_EOS, 'exact MiMo generation EOS tuple')
    else:
        eos = [policy[eos_key]]
    need(all(type(t) is int and 0 <= t < support for t in eos), 'EOS inside tokenizer support')
    return {key: policy[key] for key in ('vocab_size', eos_key, 'max_context_tokens')}


def eos_ids(geometry_or_policy):
    value = geometry_or_policy
    if 'eos_token_ids' in value:
        result = value['eos_token_ids']
        need(type(result) is list and tuple(result) == MIMO_EOS
             and all(type(t) is int for t in result), 'exact MiMo EOS tuple')
        return tuple(result)
    token = value.get('eos_token_id')
    need(type(token) is int, 'explicit scalar historical EOS')
    return (token,)


def distribution(logits, temperature, top_p, top_k=0):
    """FP32 softmax and stable filters, normalized in FP64 for public draws."""
    import torch
    need(type(temperature) in (int, float) and math.isfinite(temperature) and temperature > 0,
         'finite positive sampling temperature')
    need(type(top_p) in (int, float) and math.isfinite(top_p) and 0 < top_p <= 1,
         'bounded nucleus probability')
    need(type(top_k) is int and top_k >= 0, 'integer top-k')
    values = torch.as_tensor(logits).float()
    need(values.ndim in (1, 2) and values.shape[-1] >= 2, 'one or more full supported distributions')
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError('nonfinite model distribution; infrastructure failure')
    # Filtering logits first matches the conventional top-k -> top-p recipe.
    if top_k and top_k < values.shape[-1]:
        ordered, indices = torch.sort(values, dim=-1, descending=True, stable=True)
        ordered[..., top_k:] = -torch.inf
        values = torch.full_like(values, -torch.inf).scatter(-1, indices, ordered)
    probabilities = torch.softmax(values / temperature, dim=-1)
    if top_p < 1:
        ordered, indices = torch.sort(probabilities, dim=-1, descending=True, stable=True)
        ordered = torch.where(ordered.cumsum(-1) - ordered >= top_p, 0., ordered)
        probabilities = torch.zeros_like(probabilities).scatter(-1, indices, ordered)
    probabilities = probabilities.double()
    probabilities /= probabilities.sum(-1, keepdim=True)
    if not bool(torch.isfinite(probabilities).all()):
        raise RuntimeError('invalid normalized sampling distribution; infrastructure failure')
    return probabilities


def pick(probabilities, draw):
    import torch
    need(probabilities.ndim == 1 and type(draw) in (int, float) and math.isfinite(draw)
         and 0 <= draw < 1, 'canonical draw framing')
    cdf = probabilities.cumsum(-1); cdf[-1] = 1.
    token = int(torch.searchsorted(cdf, torch.tensor(draw, dtype=torch.float64,
                                                   device=cdf.device), right=True))
    if token >= len(probabilities) or not probabilities[token] > 0:
        raise RuntimeError('invalid zero-mass draw')
    return token


def verify_intervals_reference(logits, tokens, draws, config, error):
    """Ambiguous or excluded support triggers exact cached adjudication upstream."""
    import torch
    from .audit_policy import InvalidSample
    from .fast_prefill_audit import NumericalAmbiguity, SupportMismatch
    need(type(error) in (int, float) and math.isfinite(error) and 0 <= error <= 1e-3,
         'bounded calibrated CDF error')
    p = distribution(logits, config['temperature'], config['top_p'], config.get('top_k', 0))
    need(p.ndim == 2 and len(tokens) == len(draws) == len(p), 'complete position coverage')
    if any(type(t) is not int or not 0 <= t < p.shape[-1] for t in tokens):
        raise InvalidSample('unsupported sampled token')
    need(all(type(u) in (int, float) and math.isfinite(u) and 0 <= u < 1 for u in draws),
         'canonical public draws')
    positions = torch.arange(len(p), device=p.device)
    ids = torch.tensor(tokens, device=p.device)
    cdf = p.cumsum(-1); cdf[:, -1] = 1.
    high = cdf[positions, ids]; mass = p[positions, ids]; low = high - mass
    u = torch.tensor(draws, dtype=torch.float64, device=p.device)
    if bool((mass <= 0).any()):
        raise SupportMismatch('token excluded by MiMo top-k/top-p support')
    outside = (u < low) | (u >= high)
    if bool(((u < low-error) | (u >= high+error)).any()):
        raise InvalidSample('forced CDF outside calibrated interval')
    if bool(outside.any()):
        raise NumericalAmbiguity('CDF boundary needs exact cached adjudication')
    return {'positions': len(tokens), 'all_intervals_verified': True,
            'cdf_abs_error': error, 'sampler_distribution': 'top-k-then-top-p-stable-v1'}


def audit_distribution(logits, temperature, top_p, top_k=0):
    """Same dense arithmetic; avoid full-vocabulary sorts for finite top-k.

    The public generation function above remains the reference. We retain its
    full-width softmax, FP64 normalization, and token-ID CDF arithmetic. Only
    support selection is optimized: ties at the k-th logit use ascending ID,
    then nucleus sorting operates on the k retained probabilities. CPU/CUDA
    numerical qualification is still required for a new execution profile.
    """
    import torch
    if not (type(top_k) is int and top_k > 0):
        return distribution(logits, temperature, top_p, top_k)
    need(type(temperature) in (int, float) and math.isfinite(temperature) and temperature > 0,
         'finite positive sampling temperature')
    need(type(top_p) in (int, float) and math.isfinite(top_p) and 0 < top_p <= 1,
         'bounded nucleus probability')
    values = torch.as_tensor(logits).float()
    need(values.ndim in (1, 2) and values.shape[-1] >= 2, 'one or more full supported distributions')
    if top_k >= values.shape[-1]:
        return distribution(values, temperature, top_p, top_k)
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError('nonfinite model distribution; infrastructure failure')
    squeeze = values.ndim == 1
    rows = values.unsqueeze(0) if squeeze else values
    cutoff = torch.topk(rows, top_k, dim=-1, sorted=False).values.amin(-1, keepdim=True)
    greater = rows > cutoff
    ties = rows == cutoff
    remaining = top_k - greater.sum(-1, keepdim=True)
    keep = greater | (ties & (ties.cumsum(-1) <= remaining))
    # nonzero is row-major, thus retained indices start in ascending token ID.
    indices = keep.nonzero(as_tuple=False)[:, 1].reshape(len(rows), top_k)
    filtered = rows.masked_fill(~keep, -torch.inf)
    probabilities = torch.softmax(filtered / temperature, dim=-1)
    if top_p < 1:
        retained = probabilities.gather(-1, indices)
        ordered, order = torch.sort(retained, dim=-1, descending=True, stable=True)
        ordered = torch.where(ordered.cumsum(-1) - ordered >= top_p, 0., ordered)
        retained_ids = indices.gather(-1, order)
        probabilities = torch.zeros_like(probabilities).scatter(-1, retained_ids, ordered)
    probabilities = probabilities.double()
    probabilities /= probabilities.sum(-1, keepdim=True)
    if not bool(torch.isfinite(probabilities).all()):
        raise RuntimeError('invalid normalized sampling distribution; infrastructure failure')
    return probabilities[0] if squeeze else probabilities


def verify_intervals(logits, tokens, draws, config, error, *, row_chunk_size=32, device=None,
                     exact_fallback_on_disagreement=False):
    """Check every position in bounded chunks with unchanged verdict priority.

    The default keeps the input device, so CPU-prefill data stay on CPU. A
    caller may explicitly select CUDA only as a separately qualified execution
    profile; no implicit hardware migration or tolerance change happens here.
    """
    import torch
    from .audit_policy import InvalidSample
    from .fast_prefill_audit import NumericalAmbiguity, SupportMismatch
    need(type(error) in (int, float) and math.isfinite(error) and 0 <= error <= 1e-3,
         'bounded calibrated CDF error')
    need(type(row_chunk_size) is int and 1 <= row_chunk_size <= 128,
         'bounded interval row chunk size')
    need(type(exact_fallback_on_disagreement) is bool, 'explicit exact-fallback mode')
    values = torch.as_tensor(logits)
    need(values.ndim == 2 and len(values) > 0 and values.shape[-1] >= 2
         and len(tokens) == len(draws) == len(values), 'complete position coverage')
    if any(type(t) is not int or not 0 <= t < values.shape[-1] for t in tokens):
        raise InvalidSample('unsupported sampled token')
    need(all(type(u) in (int, float) and math.isfinite(u) and 0 <= u < 1 for u in draws),
         'canonical public draws')
    if exact_fallback_on_disagreement:
        # Preserve infrastructure errors even if an earlier row would disagree.
        # Bounded linear scans are cheap compared with distribution construction.
        for start in range(0, len(values), row_chunk_size):
            if not bool(torch.isfinite(values[start:start+row_chunk_size]).all()):
                raise RuntimeError('nonfinite model distribution; infrastructure failure')
    excluded = outside_bound = ambiguous = False
    for start in range(0, len(values), row_chunk_size):
        stop = min(start+row_chunk_size, len(values))
        chunk = values[start:stop]
        if device is not None:
            chunk = chunk.to(device)
        p = audit_distribution(chunk, config['temperature'], config['top_p'], config.get('top_k', 0))
        positions = torch.arange(len(p), device=p.device)
        ids = torch.tensor(tokens[start:stop], device=p.device)
        cdf = p.cumsum(-1); cdf[:, -1] = 1.
        high = cdf[positions, ids]; mass = p[positions, ids]; low = high-mass
        u = torch.tensor(draws[start:stop], dtype=torch.float64, device=p.device)
        excluded |= bool((mass <= 0).any())
        outside_bound |= bool(((u < low-error) | (u >= high+error)).any())
        ambiguous |= bool(((u < low) | (u >= high)).any())
        if exact_fallback_on_disagreement and (excluded or outside_bound or ambiguous):
            # This exception is not a verdict. The only production opt-in caller
            # runs the original cached sampler over every token before deciding.
            kind = SupportMismatch if excluded else InvalidSample if outside_bound else NumericalAmbiguity
            failure = kind('fast interval disagreement requires full cached adjudication')
            failure.checked_positions = stop
            raise failure
        # Release dense temporaries before processing the next chunk.
        del p, cdf, chunk
    # Match the full reference's precedence even when failures span chunks.
    if excluded:
        raise SupportMismatch('token excluded by MiMo top-k/top-p support')
    if outside_bound:
        raise InvalidSample('forced CDF outside calibrated interval')
    if ambiguous:
        raise NumericalAmbiguity('CDF boundary needs exact cached adjudication')
    return {'positions': len(tokens), 'all_intervals_verified': True,
            'cdf_abs_error': error, 'sampler_distribution': 'top-k-then-top-p-stable-v1'}
