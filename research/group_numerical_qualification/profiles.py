"""Condition on immutable original tokens; no draw, grading or optimizer code."""
import math
import time
import hashlib

import torch

from research.fixed_group_grpo.core import (
    Group, Policy, Rollout, Stamp, policy_probabilities, require, validate_group,
)
from research.fixed_group_grpo.scoring import score


def prepare_dtype(model, dtype, *, device):
    """BF16 parameters must NOT downcast original FP32 non-parameter buffers."""
    require(dtype in (torch.bfloat16, torch.float32), 'declared profile dtype')
    require(all(p.dtype == torch.bfloat16 for p in model.parameters()),
            'upstream initially loaded in original BF16 parameter dtype')
    if dtype == torch.float32:
        model.float()
    # Original collector uses .to('cuda'), preserving its FP32 RoPE buffers.
    return model.to(device).eval()


def buffer_inventory(model):
    return [dict(name=name, dtype=str(value.dtype), shape=list(value.shape),
                 sha256=hashlib.sha256(value.detach().cpu().contiguous().reshape(-1)
                                       .view(torch.uint8).numpy().tobytes()).hexdigest())
            for name, value in sorted(model.named_buffers())]


def decode_group(document):
    value = document['group']
    stamp = dict(value['stamp'])
    stamp['policy'] = Policy(**stamp['policy'])
    group = Group(Stamp(**stamp), value['task_id'], tuple(value['prompt']), tuple(
        Rollout(r['attempt'], tuple(r['output']), tuple(r['behavior_logprobs']),
                r['stop'], r['outcome']) for r in value['rollouts']))
    return validate_group(group, 8)


def no_dropout(model):
    require(getattr(model.config, 'attention_dropout', 0.) == 0.
            and not any(isinstance(m, torch.nn.Dropout) and m.p > 0
                        for m in model.modules()), 'dropout-free pinned profile')


def cached_replay(model, group, *, guard=lambda: None):
    """Replay exact batch-eight context/stop masks, forcing submitted tokens.

    The returned probabilities describe these forced prefixes. In FP32 they are
    NOT evidence that the original BF16 tokens were sampled from an FP32 policy.
    Ended rows receive masked EOS padding, matching the original collector; no
    post-EOS token is recorded or counted. No sampler/randomness is called.
    """
    validate_group(group, 8)
    no_dropout(model)
    model.eval()
    model.gradient_checkpointing_disable()
    require(all(not m.training for m in model.modules()), 'all modules in eval')
    policy = group.stamp.policy
    device = next(model.parameters()).device
    ids = torch.tensor([group.prompt] * 8, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(ids)
    values = [[] for _ in group.rollouts]
    cache = None
    started = time.monotonic()
    calls = 0
    with torch.inference_mode():
        for position in range(max(len(r.output) for r in group.rollouts)):
            guard()
            position_ids = (attention_mask.cumsum(-1)-1).clamp_min(0)[:, -ids.shape[1]:]
            result = model(input_ids=ids, attention_mask=attention_mask,
                           position_ids=position_ids, past_key_values=cache,
                           use_cache=True, logits_to_keep=1)
            cache = result.past_key_values
            calls += 1
            require(cache is not None and int(cache.get_seq_length()) == len(group.prompt)+position,
                    'same original cached length')
            require(result.logits.shape[:2] == (8, 1), 'original batch-eight logits')
            next_ids, active_next = [], []
            for index, row in enumerate(group.rollouts):
                if position < len(row.output):
                    token = row.output[position]
                    # Match original generation's one-row transform exactly.
                    probabilities = policy_probabilities(result.logits[index, 0], policy)
                    value = float(probabilities[token].log())
                    require(math.isfinite(value), 'finite selected forced-token likelihood')
                    values[index].append(value)
                else:
                    token = policy.eos_token_id
                next_ids.append(token)
                active_next.append(position+1 < len(row.output))
            del result
            if not any(active_next):
                break
            ids = torch.tensor(next_ids, dtype=torch.long, device=device)[:, None]
            active = torch.tensor(active_next, dtype=attention_mask.dtype, device=device)[:, None]
            attention_mask = torch.cat((attention_mask, active), -1)
    del cache
    require(all(len(v) == len(r.output) for v, r in zip(values, group.rollouts)),
            'exact original token accounting')
    return values, dict(seconds=time.monotonic()-started, forward_calls=calls)


def prefill(model, group, *, gradients, guard=lambda: None):
    """Original scorer's full-sequence shape and mode; no backward or step."""
    validate_group(group, 8)
    no_dropout(model)
    if not gradients:
        model.gradient_checkpointing_disable()
    values, started = [], time.monotonic()
    for row in group.rollouts:
        guard()
        selected = score(model, group.prompt, row.output, group.stamp.policy,
                         gradients=gradients)
        require(selected.requires_grad == gradients, 'declared autograd mode')
        values.append(selected.detach().cpu().tolist())
        del selected
        require(all(p.grad is None for p in model.parameters()), 'no parameter gradients or backward')
    return values, dict(seconds=time.monotonic()-started, forward_calls=8)


def drift(reference, compared, *, epsilon=.2):
    """Record signed distributions before any decision; no acceptance threshold."""
    require(len(reference) == len(compared) > 0, 'same group shape')
    require(all(len(a) == len(b) > 0 for a, b in zip(reference, compared)), 'same token shape')
    signed = torch.tensor([b-a for aa, bb in zip(reference, compared)
                           for a, b in zip(aa, bb)], dtype=torch.float64)
    require(bool(torch.isfinite(signed).all()), 'finite difference')
    absolute = signed.abs()
    below = signed < math.log(1-epsilon)
    above = signed > math.log(1+epsilon)
    return dict(tokens=signed.numel(), mean_signed_logratio=float(signed.mean()),
                mean_abs_logratio=float(absolute.mean()),
                p01_signed_logratio=float(torch.quantile(signed, .01)),
                p50_signed_logratio=float(torch.quantile(signed, .50)),
                p99_signed_logratio=float(torch.quantile(signed, .99)),
                p99_abs_logratio=float(torch.quantile(absolute, .99)),
                min_signed_logratio=float(signed.min()), max_signed_logratio=float(signed.max()),
                max_abs_logratio=float(absolute.max()),
                ppo_clip_lower=1-epsilon, ppo_clip_upper=1+epsilon,
                clipped_lower_tokens=int(below.sum()), clipped_upper_tokens=int(above.sum()),
                clipped_tokens=int((below | above).sum()),
                clipped_fraction=float((below | above).double().mean()),
                diagnostic_only=True, qualification_pass_claimed=False)


def compare_rows(group, reference, compared):
    """All-token summary plus each original nonce, with original stopping strata."""
    return dict(all_tokens=drift(reference, compared), rows=[dict(
        nonce=row.attempt, original_stop=row.stop, original_outcome=row.outcome,
        original_tokens=len(row.output), signed_logratios=[y-x for x, y in zip(a, b)],
        **drift([a], [b]))
        for row, a, b in zip(group.rollouts, reference, compared)])
