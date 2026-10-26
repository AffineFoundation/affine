"""Fixed-group contract and declared policy distribution for private research."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math

import torch


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class Policy:
    temperature: float = .7
    support_size: int = 151665
    eos_token_id: int = 151645
    max_output_tokens: int = 4096
    max_context_tokens: int = 8192

    def validate(self):
        require(type(self.temperature) in (float, int) and math.isfinite(self.temperature)
                and self.temperature > 0, 'positive temperature')
        require(type(self.support_size) is int and self.support_size > 1, 'support size')
        require(type(self.eos_token_id) is int and 0 <= self.eos_token_id < self.support_size, 'EOS support')
        require(type(self.max_output_tokens) is int and self.max_output_tokens > 0
                and type(self.max_context_tokens) is int and self.max_context_tokens > self.max_output_tokens,
                'untruncated context and output bounds')


@dataclass(frozen=True)
class Stamp:
    run_id: str
    checkpoint_sha256: str
    optimizer_step: int
    runtime_sha256: str
    policy: Policy

    def validate(self):
        require(bool(self.run_id) and type(self.optimizer_step) is int and self.optimizer_step >= 0, 'run and step')
        require(all(type(x) is str and len(x) == 64 and set(x) <= set('0123456789abcdef')
                    for x in (self.checkpoint_sha256, self.runtime_sha256)), 'checkpoint/runtime digests')
        self.policy.validate()


@dataclass(frozen=True)
class Rollout:
    attempt: int
    output: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    stop: str
    outcome: str


@dataclass(frozen=True)
class Group:
    stamp: Stamp
    task_id: str
    prompt: tuple[int, ...]
    rollouts: tuple[Rollout, ...]


def validate_group(group, group_size):
    group.stamp.validate()
    p = group.stamp.policy
    require(type(group_size) is int and group_size >= 2, 'group size')
    require(bool(group.task_id) and bool(group.prompt), 'task and prompt')
    require(len(group.prompt) + p.max_output_tokens <= p.max_context_tokens, 'full output horizon fits')
    require(all(type(t) is int and 0 <= t < p.support_size for t in group.prompt), 'prompt support')
    require(len(group.rollouts) == group_size and
            [r.attempt for r in group.rollouts] == list(range(group_size)), 'complete ordered canonical nonce set')
    for r in group.rollouts:
        require(type(r.attempt) is int and 0 < len(r.output) <= p.max_output_tokens, 'attempt and output length')
        require(all(type(t) is int and 0 <= t < p.support_size for t in r.output), 'output support')
        require(p.eos_token_id not in r.output[:-1], 'no tokens after EOS')
        require(r.stop in ('eos', 'budget') and r.outcome in ('correct', 'wrong', 'unresolved'), 'declared termination and grade')
        if r.stop == 'eos':
            require(r.output[-1] == p.eos_token_id, 'terminal EOS token')
        else:
            require(len(r.output) == p.max_output_tokens and r.output[-1] != p.eos_token_id
                    and r.outcome == 'unresolved', 'budget caps are unresolved, never negative quota')
        require(len(r.behavior_logprobs) == len(r.output) and
                all(type(v) in (int, float) and math.isfinite(v) and v <= 0 for v in r.behavior_logprobs),
                'actual finite selected behavior probabilities')
    # Intentionally no text deduplication: independent assigned draws can agree.
    return group


def group_eligibility(group, group_size):
    validate_group(group, group_size)
    counts = {key: sum(r.outcome == key for r in group.rollouts) for key in ('correct', 'wrong', 'unresolved')}
    return dict(eligible=counts['correct'] > 0 and counts['wrong'] > 0,
                counts=counts, reason='mixed_completed_outcomes' if counts['correct'] and counts['wrong'] else 'no_completed_reward_contrast')


def attempt_uniform(stamp, task_id, attempt, position):
    stamp.validate()
    require(type(attempt) is int and attempt >= 0 and type(position) is int and position >= 0, 'nonce and token position')
    data = canonical([stamp.run_id, stamp.checkpoint_sha256, stamp.optimizer_step,
                      stamp.runtime_sha256, asdict(stamp.policy), task_id, attempt, position])
    return (int.from_bytes(hashlib.sha256(data).digest()[:8], 'big') >> 11) / 2**53


def policy_probabilities(logits, policy):
    policy.validate()
    require(logits.shape[-1] >= policy.support_size, 'actual output-head support')
    selected = logits[..., :policy.support_size].float()
    require(bool(torch.isfinite(selected).all()), 'finite supported logits')
    probabilities = torch.softmax(selected / policy.temperature, -1).double()
    return probabilities / probabilities.sum(-1, keepdim=True)


def draw_and_record(logits, uniform, policy):
    require(logits.ndim == 1 and math.isfinite(uniform) and 0 <= uniform < 1, 'one canonical draw')
    with torch.no_grad():
        p = policy_probabilities(logits, policy)
        cdf = p.cumsum(-1)
        cdf[-1] = 1.
        token = int(torch.searchsorted(cdf, torch.tensor(uniform, dtype=torch.float64, device=p.device), right=True))
        require(token < policy.support_size and float(p[token]) > 0, 'positive-mass sampled token')
        return token, float(p[token].log())
