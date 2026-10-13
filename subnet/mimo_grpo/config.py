"""Explicit resolved reference settings and Affine adaptations."""
from dataclasses import asdict, dataclass
import math

UPSTREAM_REVISION = 'a2ad9f6160b03ff2d47e59832bfb6b289f37c917'
MASK_UNRESOLVED = 'affine-unresolved-zero-reward-mask-all-v1'
TRAIN_UNRESOLVED = 'affine-unresolved-zero-reward-active-v1'
MCORE_DECAY = 'mcore-bias-and-1d-excluded-v1'
ALL_DECAY = 'affine-all-trainable-decay-v1'


@dataclass(frozen=True)
class TrainerConfig:
    group_size: int = 8
    max_tasks: int = 512
    learning_rate: float = 1e-6
    weight_decay: float = .01
    weight_decay_policy: str = MCORE_DECAY
    beta1: float = .9
    beta2: float = .999
    adam_epsilon: float = 1e-8
    max_gradient_norm: float = 1.
    clip_ratio: float = .2
    dual_clip: float = 3.
    ppo_epochs: int = 1
    temperature: float = 1.
    top_p: float = .95
    top_k: int = 20
    kl_coefficient: float = 0.
    entropy_coefficient: float = 0.
    normalize_advantage_std: bool = False
    cap_policy: str = MASK_UNRESOLVED
    old_actor_logprob_tolerance: float = 1e-5

    def validate(self):
        if type(self.group_size) is not int or self.group_size < 2:
            raise ValueError('at least two fixed nonce attempts')
        if type(self.max_tasks) is not int or not 1 <= self.max_tasks <= 512:
            raise ValueError('one through 512 distinct tasks per optimizer update')
        if self.ppo_epochs != 1 or self.normalize_advantage_std:
            raise ValueError('reference one PPO epoch and unnormalized group advantages')
        if self.kl_coefficient != 0 or self.entropy_coefficient != 0:
            raise ValueError('reference explicit KL and entropy objectives disabled')
        if self.cap_policy not in (MASK_UNRESOLVED, TRAIN_UNRESOLVED):
            raise ValueError('explicit versioned unresolved policy required')
        if self.weight_decay_policy not in (MCORE_DECAY, ALL_DECAY):
            raise ValueError('explicit versioned optimizer parameter decay policy required')
        for name in ('learning_rate', 'adam_epsilon', 'max_gradient_norm', 'temperature'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError('positive finite ' + name)
        if not (math.isfinite(self.dual_clip) and 0 <= self.beta1 < 1 and 0 <= self.beta2 < 1 and
                math.isfinite(self.weight_decay) and self.weight_decay >= 0 and
                0 < self.clip_ratio < 1 and self.dual_clip > 1 and
                0 < self.top_p <= 1 and type(self.top_k) is int and self.top_k > 0 and
                math.isfinite(self.old_actor_logprob_tolerance) and self.old_actor_logprob_tolerance >= 0):
            raise ValueError('invalid resolved trainer hyperparameters')
        return self

    def document(self):
        self.validate()
        return dict(version='affine-mimo-reference-grpo-v1', upstream_revision=UPSTREAM_REVISION,
            objective='vanilla-dual-clip-ppo-prompt-mean', old_source='trusted-training-engine-prefill',
            rollout_correction_bypass_mode=False, optimizer='torch.optim.AdamW-cpu-fp32-masters',
            settings=asdict(self))
