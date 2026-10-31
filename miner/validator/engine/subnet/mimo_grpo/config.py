"""Explicit resolved reference settings and Affine adaptations."""
from dataclasses import asdict, dataclass, fields
import math

UPSTREAM_REVISION = 'a2ad9f6160b03ff2d47e59832bfb6b289f37c917'
MASK_UNRESOLVED = 'affine-unresolved-zero-reward-mask-all-v1'
TRAIN_UNRESOLVED = 'affine-unresolved-zero-reward-active-v1'
MCORE_DECAY = 'mcore-bias-and-1d-excluded-v1'
ALL_DECAY = 'affine-all-trainable-decay-v1'
PROMPT_MEAN = 'prompt-mean'
CONSTANT_NORM = 'seq-mean-token-sum-norm'
NO_SAMPLER_CORRECTION = 'none'
# Score centering + masked importance sampling on the replayed top-k/top-p
# support (arXiv 2609.20807 Eq. 9/12, as in the private E42 objective.py).
SC_MIS = 'sc-mis-replayed-support-v1'
K3 = 'low_var_kl'

# Fields added after the v1 objective. When all of them hold these legacy
# values the published config document is byte-identical to v1, so historical
# TRAINING-JOB/transaction journals keep validating.
EXTENSION_DEFAULTS = dict(
    loss_agg_mode=PROMPT_MEAN, loss_scale_factor=0,
    derail_mask=False, derail_window=32, derail_threshold=-3., derail_tail=64,
    kl_reference_checkpoint='', kl_type=K3,
    sampler_correction=NO_SAMPLER_CORRECTION, sampler_ratio_lower=.2, sampler_ratio_upper=5.,
    overlong_soft_start=0, overlong_max_tokens=0, overlong_penalty=0.)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


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
    # --- extensions (see EXTENSION_DEFAULTS) ---
    # Loss aggregation. prompt-mean is the v1 objective. seq-mean-token-sum-norm
    # divides every rollout's token SUM by (active rollouts x constant), so an
    # 8192-token runaway weighs per token exactly like a short answer.
    loss_agg_mode: str = PROMPT_MEAN
    loss_scale_factor: int = 0
    # Derail mask: for truncated (max_tokens) rollouts with negative advantage,
    # find the first token whose trailing derail_window mean sampled-token
    # log-prob (trusted OLD) is < derail_threshold; train only through
    # derail point + derail_tail. Without a derail point the whole rollout trains.
    derail_mask: bool = False
    derail_window: int = 32
    derail_threshold: float = -3.
    derail_tail: int = 64
    # k3 ("low_var_kl") KL to a fixed reference checkpoint (descriptor id).
    kl_reference_checkpoint: str = ''
    kl_type: str = K3
    # Sampler-mismatch (top-k/top-p) correction. OFF by default.
    sampler_correction: str = NO_SAMPLER_CORRECTION
    sampler_ratio_lower: float = .2
    sampler_ratio_upper: float = 5.
    # Soft overlong penalty (DAPO-style length shaping, OFF by default). The
    # training reward of every rollout becomes
    #   float(correct) - overlong_penalty * clip((len - soft_start) / (max_tokens - soft_start), 0, 1)
    # so lengths past soft_start are discouraged linearly up to the sampling
    # hard cap (max_tokens). Group advantages are computed from these rewards.
    overlong_soft_start: int = 0
    overlong_max_tokens: int = 0
    overlong_penalty: float = 0.

    @property
    def extended(self):
        return any(getattr(self, k) != v for k, v in EXTENSION_DEFAULTS.items())

    def validate(self):
        if type(self.group_size) is not int or self.group_size < 2:
            raise ValueError('at least two fixed nonce attempts')
        if type(self.max_tasks) is not int or not 1 <= self.max_tasks <= 512:
            raise ValueError('one through 512 distinct tasks per optimizer update')
        if self.ppo_epochs != 1 or self.normalize_advantage_std:
            raise ValueError('reference one PPO epoch and unnormalized group advantages')
        if self.entropy_coefficient != 0:
            raise ValueError('reference explicit entropy objective disabled')
        if (type(self.kl_coefficient) not in (int, float) or not math.isfinite(self.kl_coefficient)
                or not 0 <= self.kl_coefficient <= 1):
            raise ValueError('bounded nonnegative KL coefficient')
        if self.kl_type != K3:
            raise ValueError('only k3/low_var_kl reference KL is supported')
        if self.kl_coefficient > 0 and not _sha(self.kl_reference_checkpoint):
            raise ValueError('KL anchoring requires an explicit reference checkpoint descriptor id')
        if self.kl_coefficient == 0 and self.kl_reference_checkpoint:
            raise ValueError('KL reference checkpoint without a KL coefficient')
        if self.cap_policy not in (MASK_UNRESOLVED, TRAIN_UNRESOLVED):
            raise ValueError('explicit versioned unresolved policy required')
        if self.weight_decay_policy not in (MCORE_DECAY, ALL_DECAY):
            raise ValueError('explicit versioned optimizer parameter decay policy required')
        if self.loss_agg_mode == PROMPT_MEAN:
            if self.loss_scale_factor != 0:
                raise ValueError('prompt-mean has no constant loss scale')
        elif self.loss_agg_mode == CONSTANT_NORM:
            if type(self.loss_scale_factor) is not int or not 1 <= self.loss_scale_factor <= 1 << 20:
                raise ValueError('constant normalization needs an explicit positive integer loss_scale_factor')
        else:
            raise ValueError('unsupported loss aggregation mode')
        if type(self.derail_mask) is not bool:
            raise ValueError('explicit boolean derail mask')
        if (type(self.derail_window) is not int or not 1 <= self.derail_window <= 4096 or
                type(self.derail_tail) is not int or not 0 <= self.derail_tail <= 1 << 20 or
                type(self.derail_threshold) not in (int, float) or not math.isfinite(self.derail_threshold)
                or self.derail_threshold >= 0):
            raise ValueError('bounded derail window/tail and negative finite threshold')
        if self.derail_mask and self.cap_policy != TRAIN_UNRESOLVED:
            raise ValueError('derail mask only applies when truncated rollouts are trained')
        if self.sampler_correction not in (NO_SAMPLER_CORRECTION, SC_MIS):
            raise ValueError('unknown sampler correction')
        if not (math.isfinite(self.sampler_ratio_lower) and math.isfinite(self.sampler_ratio_upper)
                and 0 < self.sampler_ratio_lower < 1 < self.sampler_ratio_upper):
            raise ValueError('ordered sampler importance ratio bounds')
        if type(self.overlong_penalty) not in (int, float) or not math.isfinite(self.overlong_penalty):
            raise ValueError('finite overlong penalty')
        if self.overlong_penalty == 0:
            if self.overlong_soft_start != 0 or self.overlong_max_tokens != 0:
                raise ValueError('overlong window without an overlong penalty')
        elif (not 0 < self.overlong_penalty <= 1 or type(self.overlong_soft_start) is not int
                or type(self.overlong_max_tokens) is not int
                or not 1 <= self.overlong_soft_start < self.overlong_max_tokens <= 1 << 20):
            raise ValueError('bounded overlong penalty in (0, 1] with 1 <= soft_start < max_tokens')
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

    def reward(self, outcome, length):
        """Training reward: 1 for correct else 0, minus the soft overlong penalty."""
        value = float(outcome == 'correct')
        if self.overlong_penalty:
            span = self.overlong_max_tokens - self.overlong_soft_start
            value -= self.overlong_penalty * min(1., max(0., (length - self.overlong_soft_start) / span))
        return value

    def document(self):
        self.validate()
        settings = asdict(self)
        if not self.extended:
            for name in EXTENSION_DEFAULTS:
                settings.pop(name)
            return dict(version='affine-mimo-reference-grpo-v1', upstream_revision=UPSTREAM_REVISION,
                objective='vanilla-dual-clip-ppo-prompt-mean', old_source='trusted-training-engine-prefill',
                rollout_correction_bypass_mode=False, optimizer='torch.optim.AdamW-cpu-fp32-masters',
                settings=settings)
        objective = ('sc-mis-replayed-support' if self.sampler_correction == SC_MIS else 'vanilla-dual-clip-ppo')
        objective += '-' + self.loss_agg_mode
        if self.derail_mask:
            objective += '-derail-masked'
        if self.kl_coefficient:
            objective += '-k3-kl-reference'
        if self.overlong_penalty:
            objective += '-overlong-soft-penalty'
        return dict(version='affine-mimo-reference-grpo-v2', upstream_revision=UPSTREAM_REVISION,
            objective=objective, old_source='trusted-training-engine-prefill',
            rollout_correction_bypass_mode=False, optimizer='torch.optim.AdamW-cpu-fp32-masters',
            settings=settings)


assert set(EXTENSION_DEFAULTS) == {f.name for f in fields(TrainerConfig)} - {
    'group_size', 'max_tasks', 'learning_rate', 'weight_decay', 'weight_decay_policy', 'beta1', 'beta2',
    'adam_epsilon', 'max_gradient_norm', 'clip_ratio', 'dual_clip', 'ppo_epochs', 'temperature', 'top_p',
    'top_k', 'kl_coefficient', 'entropy_coefficient', 'normalize_advantage_std', 'cap_policy',
    'old_actor_logprob_tolerance'}
