"""Private eight-row Qwen2 collector, not a live miner or GPU-qualified runtime.

Every task has exactly eight assigned attempts. No outcome quota, replacement,
content deduplication, or fallback sampler. A row ends at its first actual EOS or
the frozen horizon. Remaining batch rows keep their own cache and random draws.
Selected LP is captured from the actual generation logits with the objective's
exact transform; no full-vocabulary arrays are returned or persisted here.

The proposed first H200 canary is PRIVATE BF16 eager batch8 generation versus
actual BF16 gradient-checkpointed teacher forcing, using the established FP32
master/accumulator optimizer. This differs from public FP32 miner generation.
CPU FP32/BF16 fixtures measure drift but set no H200 admission tolerance. The
signed eventual runtime must bind dtype, attention implementation, train/eval
mode, dropout, accumulation, support and temperature; equal weight bytes alone
do not establish equal numerical policies. No new FP32 optimizer is introduced.

The caller supplies an authenticated checkpoint/runtime stamp, unchanged model,
native prompt/grader, and an append-only attempt recorder. This module cannot
prove historical execution or replace source/model authentication. An exception
invalidates the original collection; previously recorded attempts remain evidence
and cannot be silently regenerated. The narrow private _sample_rows helper is
also used for CPU batched-versus-single-row qualification, not selected-slot mining.
"""
from dataclasses import dataclass
import inspect
import time

import torch

from .core import (
    Group, Rollout, Stamp, attempt_uniform, draw_and_record, require,
)


@dataclass(frozen=True)
class SampledAttempt:
    stamp: Stamp
    task_id: str
    prompt: tuple[int, ...]
    attempt: int
    output: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    stop: str


@dataclass(frozen=True)
class Collection:
    group: Group
    forward_calls: int
    generated_tokens: int
    collection_seconds: float


def _sample_rows(model, prompt, *, stamp, task_id, attempts, record_attempt, group_size=8):
    stamp.validate()
    policy = stamp.policy
    require(isinstance(task_id, str) and bool(task_id), "fixed task identity")
    require(type(prompt) is tuple and bool(prompt)
            and all(type(t) is int and 0 <= t < policy.support_size for t in prompt),
            "complete unpadded supported prompt")
    require(len(prompt) + policy.max_output_tokens <= policy.max_context_tokens,
            "full horizon fits context; no prompt/output truncation")
    require(type(attempts) is tuple and bool(attempts) and len(set(attempts)) == len(attempts)
            and all(type(i) is int and 0 <= i < group_size for i in attempts), "assigned distinct attempt slots")
    require(callable(record_attempt), "original attempt recorder required")
    require(not model.training and all(not m.training for m in model.modules()),
            "generation model and every submodule must be in eval mode")
    require(model.config.model_type == "qwen2", "explicit Qwen2 model API")
    require(policy.support_size <= model.config.vocab_size
            and policy.max_context_tokens <= model.config.max_position_embeddings,
            "pinned policy within actual model vocabulary/context")
    required = {"input_ids", "attention_mask", "position_ids", "past_key_values", "use_cache", "logits_to_keep"}
    require(required <= set(inspect.signature(model.forward).parameters), "qualified explicit cache/position API")
    device = next(model.parameters()).device
    batch_size, prompt_size = len(attempts), len(prompt)
    ids = torch.tensor([prompt] * batch_size, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(ids)
    outputs, behavior = [[] for _ in attempts], [[] for _ in attempts]
    finished = [False] * batch_size
    rows = [None] * batch_size
    cache = None
    calls = 0
    start = time.monotonic()
    with torch.inference_mode():
        for position in range(policy.max_output_tokens):
            # Active rows have no padding, so position_ids are exactly the native
            # absolute positions. Finished-row padding never enters a rollout.
            position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1]:]
            result = model(input_ids=ids, attention_mask=attention_mask,
                           position_ids=position_ids, past_key_values=cache,
                           use_cache=True, logits_to_keep=1)
            calls += 1
            cache = result.past_key_values
            require(cache is not None and hasattr(cache, "get_seq_length")
                    and int(cache.get_seq_length()) == prompt_size + position,
                    "actual KV-cache sequence length")
            require(result.logits.ndim == 3 and tuple(result.logits.shape[:2]) == (batch_size, 1)
                    and result.logits.shape[-1] == model.config.vocab_size,
                    "one next-token logits row per assigned attempt")
            next_ids = []
            for row, attempt in enumerate(attempts):
                if finished[row]:
                    next_ids.append(policy.eos_token_id)
                    continue
                token, logprob = draw_and_record(result.logits[row, 0],
                                                attempt_uniform(stamp, task_id, attempt, position), policy)
                outputs[row].append(token)
                behavior[row].append(logprob)
                next_ids.append(token)
                stop = "eos" if token == policy.eos_token_id else (
                    "budget" if position + 1 == policy.max_output_tokens else None)
                if stop is not None:
                    finished[row] = True
                    rows[row] = SampledAttempt(stamp, task_id, prompt, attempt,
                                               tuple(outputs[row]), tuple(behavior[row]), stop)
                    # Capture before any outcome-based decision or grader call.
                    record_attempt(rows[row])
            del result
            if all(finished):
                break
            ids = torch.tensor(next_ids, dtype=torch.long, device=device)[:, None]
            active = torch.tensor([not done for done in finished], dtype=attention_mask.dtype,
                                  device=device)[:, None]
            attention_mask = torch.cat((attention_mask, active), dim=-1)
    del cache
    require(all(row is not None for row in rows), "every original attempt reached a declared stop")
    require(tuple(row.attempt for row in rows) == attempts, "unchanged attempt accounting")
    return tuple(rows), calls, time.monotonic() - start


def collect_group(model, prompt, *, stamp, task_id, grade_completed, record_attempt, group_size=8):
    """Collect all eight first, then grade every EOS response independently.

    grade_completed(output_tokens) must return correct/wrong/unresolved from the
    native environment. Budget-capped attempts stay unresolved and are retained.
    An exception/invalid grader status propagates; no Group is returned and no
    attempt is replaced. The recorder has already retained original token paths.
    """
    require(callable(grade_completed), "native completed-output grader required")
    rows, calls, seconds = _sample_rows(model, prompt, stamp=stamp, task_id=task_id,
                                       attempts=tuple(range(group_size)), record_attempt=record_attempt, group_size=group_size)
    rollouts = []
    for row in rows:
        outcome = "unresolved" if row.stop == "budget" else grade_completed(row.output)
        require(outcome in ("correct", "wrong", "unresolved"),
                "native grading failed; original collection is invalid, not reward zero")
        rollouts.append(Rollout(row.attempt, row.output, row.behavior_logprobs, row.stop, outcome))
    group = Group(stamp, task_id, prompt, tuple(rollouts))
    return Collection(group, calls, sum(len(row.output) for row in rows), seconds)
