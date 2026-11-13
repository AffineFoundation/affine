"""From a frozen batch to training rows. Pure Python: no GPU and no torch.

A group is one prompt with its rollouts. A rollout is one sampled answer
with its reward. The advantage of a rollout is its reward minus the mean
reward of its group, with no division by the standard deviation.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..errors import BatchRejected

END_OF_TURN = "<|im_end|>"  # the token that ends an assistant turn
CONTROL_TOKENS = ("<|im_start|>", END_OF_TURN, "<|endoftext|>")

Groups = Sequence[tuple[str, Sequence[tuple[str, float]]]]  # ((prompt, ((output, reward), ...)), ...)


def advantages(rewards: Sequence[float]) -> list[float]:
    """Advantages of one group, in file order. A group of one has no baseline: its advantage is its reward."""
    mean = math.fsum(rewards) / len(rewards) if len(rewards) > 1 else 0.0
    return [reward - mean for reward in rewards]


def training_rows(groups: Groups) -> list[dict]:
    """One row per rollout: prompt, output, reward and advantage. Bad text or a bad reward rejects the file."""
    rows = []
    for prompt, rollouts in groups:
        if not prompt or not rollouts:
            raise BatchRejected("every group needs a prompt and at least one rollout")
        rewards = [reward for _, reward in rollouts]
        if not all(type(reward) in (int, float) and math.isfinite(reward) for reward in rewards):
            raise BatchRejected("every reward must be a finite number")
        for text in (prompt, *(output for output, _ in rollouts)):
            if not isinstance(text, str) or any(token in text for token in CONTROL_TOKENS):
                raise BatchRejected("a prompt or an output holds a chat control token")
        for (output, reward), advantage in zip(rollouts, advantages(rewards)):
            rows.append({"prompt": prompt, "output": output, "reward": float(reward), "advantage": advantage})
    if not rows:
        raise BatchRejected("the frozen batch holds no rollout")
    return rows


def fit_tokens(n_prompt: int, output: Sequence[int], eos_id: int, max_output: int, max_context: int) -> tuple[list[int], bool, bool]:
    """The output tokens to train on, whether they end without an end token, and whether tokens were removed.

    The room is ``max_output`` tokens, or less when the prompt is long. An
    output shorter than the room ended by itself: the end-of-turn token is
    appended. An output that fills the room or is longer is cut to the room:
    nothing is appended, so it never counts as ended. A long row is cut, not
    rejected: the training cap is shorter than the cap a miner samples with.
    """
    room = min(max_output, max_context - n_prompt)
    if not output or room < 1:
        raise BatchRejected("a row needs at least one output token and a prompt that leaves room for it")
    if len(output) >= room:
        return list(output[:room]), True, len(output) > room
    return [*output, eos_id], False, False


def derail_keep(logps: Sequence[float], window: int, threshold: float, tail: int) -> int:
    """How many leading tokens of a cut, below-average rollout still train.

    The derail point is the first token whose trailing full window of
    sampled-token log-probabilities has a mean below ``threshold``. Tokens up
    to ``tail`` after that point stay; the rest is noise and is masked.
    """
    if window < 1 or len(logps) < window:
        return len(logps)
    total = math.fsum(logps[:window])
    for last in range(window - 1, len(logps)):
        if last >= window:
            total += logps[last] - logps[last - window]
        if total / window < threshold:
            return min(len(logps), last + tail + 1)
    return len(logps)
