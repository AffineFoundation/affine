"""Trusted training projection of an admitted, natively graded task group.

The transport/admission layer, not these dataclasses, verifies signatures,
checkpoint bytes, taskset membership, prompt construction and native grading.
"""
from dataclasses import dataclass
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True)
class InputBinding:
    run: str
    epoch: str
    checkpoint: str
    contract_sha256: str
    optimizer_step: int

    def validate(self):
        if not self.run or not self.epoch or not sha(self.checkpoint) or not sha(self.contract_sha256):
            raise ValueError('immutable run/epoch/checkpoint/contract binding required')
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError('nonnegative optimizer step')


@dataclass(frozen=True)
class Trajectory:
    nonce: int
    tokens: tuple[int, ...]
    outcome: str
    stop: str


@dataclass(frozen=True)
class TaskGroup:
    task_id: str
    prompt: tuple[int, ...]
    trajectories: tuple[Trajectory, ...]
    # Native admission content receipt, not a miner signature on outcome claims.
    admission_sha256: str

    def validate(self, size, *, vocab_size, eos_token_ids):
        if type(self.task_id) is not str or not self.task_id or not sha(self.admission_sha256) or not self.prompt:
            raise ValueError('trusted task, prompt and admission receipt required')
        if (len(self.trajectories) != size or any(type(r.nonce) is not int for r in self.trajectories)
                or [r.nonce for r in self.trajectories] != list(range(size))):
            raise ValueError('exact ordered nonce set 0 through N-1')
        outcomes = [r.outcome for r in self.trajectories]
        if 'correct' not in outcomes or 'wrong' not in outcomes:
            raise ValueError('one completed correct and one completed wrong required')
        for token in self.prompt:
            if type(token) is not int or not 0 <= token < vocab_size:
                raise ValueError('valid trusted prompt tokens')
        for row in self.trajectories:
            if not row.tokens or any(type(t) is not int or not 0 <= t < vocab_size for t in row.tokens):
                raise ValueError('valid nonempty output tokens')
            if row.outcome not in ('correct', 'wrong', 'unresolved'):
                raise ValueError('grader/infra error cannot enter a training group')
            if row.outcome == 'unresolved':
                if row.stop == 'max_tokens':
                    if any(t in eos_token_ids for t in row.tokens):
                        raise ValueError('budget-capped trajectory cannot contain EOS')
                elif row.stop == 'eos':
                    if row.tokens[-1] not in eos_token_ids or any(t in eos_token_ids for t in row.tokens[:-1]):
                        raise ValueError('unresolved EOS must terminate at its first approved EOS')
                else:
                    raise ValueError('unresolved output requires budget cap or EOS without a gradeable final answer')
            elif row.stop != 'eos' or row.tokens[-1] not in eos_token_ids or any(t in eos_token_ids for t in row.tokens[:-1]):
                raise ValueError('completed outcome must terminate at its first approved EOS')
        # Intentionally no token-text uniqueness requirement: distinct assigned
        # random draws can legitimately produce identical outputs.
