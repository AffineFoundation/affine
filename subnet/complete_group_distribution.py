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


def verify_intervals(logits, tokens, draws, config, error):
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
