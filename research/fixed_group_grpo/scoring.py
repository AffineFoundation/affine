"""Matched support/temperature selected-token scoring with bounded head memory."""
import torch
from .core import require, policy_probabilities


def selected_logprobs_from_logits(logits, prompt, output, policy, chunk_tokens=64):
    require(logits.ndim == 2 and logits.shape[0] >= len(prompt)+len(output)-1,
            'complete unpadded logits rows')
    require(bool(prompt) and bool(output) and len(prompt)+len(output) <= policy.max_context_tokens,
            'nonempty untruncated tokens')
    require(all(type(t) is int and 0 <= t < policy.support_size for t in output), 'output support')
    scores = []
    for begin in range(0, len(output), chunk_tokens):
        chosen = torch.tensor(output[begin:begin+chunk_tokens], device=logits.device, dtype=torch.long)
        start = len(prompt)-1+begin
        probabilities = policy_probabilities(logits[start:start+len(chosen)], policy)
        scores.append(probabilities.gather(1, chosen[:, None])[:, 0].log())
    result = torch.cat(scores)
    require(bool(torch.isfinite(result).all()), 'finite all-token scores')
    return result


def score(model, prompt, output, policy, *, gradients):
    require(len(prompt)+policy.max_output_tokens <= policy.max_context_tokens, 'declared full horizon')
    require(getattr(model.config, 'attention_dropout', 0.) == 0.
            and not any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()),
            'dropout-free generation and scoring')
    if gradients:
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt)+list(output)], device=device)
    with torch.set_grad_enabled(gradients):
        result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        return selected_logprobs_from_logits(result.logits[0], prompt, output, policy)
