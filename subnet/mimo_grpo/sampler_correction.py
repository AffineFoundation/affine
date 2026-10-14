"""Sampler-mismatch correction helpers (OFF unless config.sampler_correction).

The miner samples from the top-k -> top-p filtered distribution q, while the
v1 trainer differentiates raw full-vocabulary log-probs. With the flag set to
SC_MIS the trainer replays q's support from trusted OLD logits (no miner claim
is used) and applies masked importance sampling with score centering
(arXiv 2609.20807 Eq. 9/12), identical to the private E42 objective.py.
"""
import torch


def replay_support(logits, top_k, top_p):
    """Candidate ids and normalized q for every position, mirroring
    complete_group_distribution.distribution (stable top-k, then top-p).

    Returns (indices [n,k] long, q [n,k] float64) padded with q == 0.
    """
    values = torch.as_tensor(logits).float()
    if values.ndim != 2 or not torch.isfinite(values).all():
        raise ValueError('finite [positions, vocabulary] logits')
    k = min(int(top_k), values.shape[-1])
    ordered, indices = torch.sort(values, dim=-1, descending=True, stable=True)
    ordered, indices = ordered[:, :k], indices[:, :k]
    probabilities = torch.softmax(ordered, dim=-1)
    if top_p < 1:
        probabilities = torch.where(probabilities.cumsum(-1) - probabilities >= top_p, 0., probabilities)
    q = probabilities.double()
    q = q / q.sum(-1, keepdim=True)
    return indices, q


def selected_candidates(indices, q, tokens):
    """Index of each sampled token in its replayed support, -1 if outside."""
    tokens = torch.as_tensor(tokens, dtype=torch.long, device=indices.device)
    hit = (indices == tokens[:, None]) & (q > 0)
    found = hit.any(-1)
    return torch.where(found, hit.float().argmax(-1), torch.full_like(tokens, -1))


def corrected_terms(candidate_logits, q, selected, *, lower=.2, upper=5., center=True):
    """Per-token score terms WITHOUT advantage or reduction.

    Multiply by the advantage and aggregate upstream. Ratios and the centering
    expectation are stop-gradient. Rows with selected == -1 must be masked.
    """
    if candidate_logits.ndim != 2 or q.shape != candidate_logits.shape:
        raise ValueError('matching candidate tensors')
    if selected.shape != candidate_logits.shape[:1] or selected.dtype != torch.long:
        raise ValueError('selected candidate per position')
    if not 0 < lower < upper or not torch.isfinite(candidate_logits).all():
        raise ValueError('finite logits and ordered ratio bounds')
    q = q.detach().to(candidate_logits)
    support = q > 0
    inside = selected >= 0
    safe_selected = selected.clamp_min(0)
    logp = torch.log_softmax(candidate_logits.masked_fill(~support, -torch.inf), -1)
    safe_logp = logp.masked_fill(~support, 0.)
    with torch.no_grad():
        ratio = torch.where(support, logp.exp() / q.clamp_min(torch.finfo(q.dtype).tiny), 0.)
        weights = torch.where(support & (ratio >= lower) & (ratio <= upper), ratio, 0.)
    chosen = safe_logp.gather(-1, safe_selected[:, None]).squeeze(-1)
    chosen_weight = weights.gather(-1, safe_selected[:, None]).squeeze(-1)
    correction = (q * weights * safe_logp).sum(-1) if center else torch.zeros_like(chosen)
    term = torch.where(inside, chosen_weight * chosen - correction, torch.zeros_like(chosen))
    return term, dict(selected_weight=chosen_weight.detach(), inside=inside)
