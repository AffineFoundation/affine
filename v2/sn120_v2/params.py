"""Every settable number in v2, in one place.

``Params`` is the single source for caps, the recipe, the judging rule and
the payout rule. The controller pins one ``Params`` snapshot into each round
ticket, so the trainer, the evaluator, the website and the miner contract can
never disagree about a cap. Defaults marked "placeholder" wait for B300
measurements.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .canonical import Record, _hints, from_jsonable, is_sha256, to_jsonable
from .errors import BatchRejected, ConfigError

SEED_SCOPES = ("round", "batch")
MINER_UPGRADE_COMMAND = "git fetch origin affine/king4-kltok && git checkout affine/king4-kltok && git pull && python -m pip install -e './v2[miner,envs,links,chain]'"

# Identity of the launch model. The base hash is the manifest digest of the
# published checkpoint files (see canonical.manifest_digest).
GENESIS_MODEL_REPO = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
GENESIS_MODEL_REVISION = "2367e865d009c13ac81713a2878291d33ab28177"
GENESIS_BASE_HASH = "87de8304e9d33dde39b81a41fdbbf306426ea6fd993b7d85332e6a48e8a4bbae"


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise ConfigError(message)


def _unit(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


ATTENTION_AND_MLP_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# Projections of the linear-attention layers. A miner may name them; the default recipe does not hold them.
# ``in_proj_a`` and ``in_proj_b`` of those layers are not choices.
OPTIONAL_TARGET_MODULES = ("in_proj_qkv", "in_proj_z", "out_proj")
TARGET_MODULE_CHOICES = ATTENTION_AND_MLP_PROJECTIONS + OPTIONAL_TARGET_MODULES
# The server that scores a LoRA (vLLM 0.32.0) holds ``in_proj_qkv`` and ``in_proj_z`` as one layer. It loads an adapter
# for both, or for ``in_proj_qkv`` alone. It refuses an adapter for ``in_proj_z`` alone, and that failure is one of
# the box, not of the file. So a list with the first name must hold the second name too.
TARGET_MODULE_NEEDS = (("in_proj_z", "in_proj_qkv"),)


@dataclass(frozen=True)
class MinerBound(Record):
    """One recipe field that a miner may set under ``train`` in the batch file.

    A number must be inside [``low``, ``high``], both ends included. A list
    must be a non-empty subset of ``choices``. ``decided`` is False while the
    bound is a placeholder that waits for a decision.
    """

    key: str  # the recipe field, and the key under "train"
    low: float = 0.0
    high: float = 0.0
    integer: bool = False  # the number must be a whole number
    choices: tuple[str, ...] = ()
    decided: bool = False

    def check(self, value: Any) -> Any:
        """Return the value to store, or raise ``BatchRejected``. A value is never clamped."""
        if self.choices:
            names = list(value) if isinstance(value, (list, tuple)) else []
            known = bool(names) and all(isinstance(name, str) and name in self.choices for name in names)
            if not known or len(set(names)) != len(names):
                raise BatchRejected(f"train.{self.key} must be a non-empty subset of {list(self.choices)}")
            for name, needed in TARGET_MODULE_NEEDS:
                if name in names and needed not in names:
                    raise BatchRejected(f"train.{self.key} with {name} must hold {needed} too")
            return tuple(name for name in self.choices if name in names)
        wanted = (int,) if self.integer else (int, float)
        # The range test comes first: it is safe for a whole number too large to be a float.
        if type(value) not in wanted or not self.low <= value <= self.high:
            kind = "a whole number" if self.integer else "a number"
            raise BatchRejected(f"train.{self.key} must be {kind} from {self.low:g} to {self.high:g}")
        return value if self.integer else float(value)


# What a miner may set: learning rate and Adam epsilon; steps and minibatches; GRPO group size
# and sampling temperature; LoRA rank and alpha; target modules; the cap on
# the norm of the whole update.
# PLACEHOLDER marks a bound the spec does not give. It is not a decision.
SFT_MINER_BOUND = MinerBound("sft_weight", 0.0, 1.0, decided=True)

DEFAULT_MINER_BOUNDS = (
    MinerBound("lora_lr", 1e-6, 1e-4),  # PLACEHOLDER
    MinerBound("adam_eps", 1e-10, 1e-4, decided=True),
    MinerBound("steps", 1, 4, integer=True),  # PLACEHOLDER
    MinerBound("minibatches", 1, 8000, integer=True, decided=True),  # at the top one minibatch is one rollout
    MinerBound("group_size", 2, 64, integer=True),  # PLACEHOLDER
    MinerBound("sample_temperature", 0.5, 1.5),  # PLACEHOLDER
    MinerBound("rank", 8, 64, integer=True, decided=True),  # spec: "suggested 8-64"
    MinerBound("alpha", 8.0, 128.0),  # PLACEHOLDER
    # The 7 projections of the default, and the 3 of the linear-attention layers that a miner may add.
    MinerBound("target_modules", choices=TARGET_MODULE_CHOICES, decided=True),
    MinerBound("global_norm_cap", 0.01, 6.0, decided=True),  # the high end is the ceiling for every miner
)


@dataclass(frozen=True)
class Recipe(Record):
    """The published training recipe. Its digest is the ``recipe_id``.

    The validator owns every field. A miner may set only the fields named in
    ``miner_bounds``, each inside its bound. ``miner_bounds`` is the one
    place that the batch checker, the trainer and the publisher read.
    """

    # ``minibatches`` and ``adam_eps`` came after the first rounds. While at their defaults they are not written
    # (``canonical.to_jsonable``): a stored recipe keeps its JSON form and ``recipe_id``, and code from before the
    # fields can read it.
    LEFT_OUT_AT_DEFAULT = ("minibatches", "adam_eps", "sft_weight")

    method: str = "grpo-lora"
    steps: int = 1  # passes over the whole file
    minibatches: int = 1  # optimizer updates in one pass; 1 = one update on the whole file
    lora_lr: float = 1e-5
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    global_norm_cap: float | None = 6.0  # cap on the norm of the whole adapter update; None = no cap
    group_size: int = 0  # PLACEHOLDER: rollouts per GRPO group; 0 = use each prompt's rollouts as the file groups them
    sample_temperature: float = 1.0  # temperature the batch's rollouts were sampled at
    sft_weight: float = 1.0  # multiplier of each supervised trace's token-mean cross-entropy
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ATTENTION_AND_MLP_PROJECTIONS
    ppo_clip: float = 0.2
    dual_clip: float = 3.0
    kl: float = 0.0
    support_top_k: int = 20  # the policy-gradient term uses only tokens the head could sample
    support_top_p: float = 0.95
    derail_window: int = 32
    derail_threshold: float = -3.0
    derail_tail: int = 64
    train_max_output_tokens: int = 32768  # the training cap is independent of the scoring cap
    train_max_context_tokens: int = 36864
    miner_bounds: tuple[MinerBound, ...] = DEFAULT_MINER_BOUNDS
    code_digest: str = ""  # sha256 of the training code, set by the train/eval workstream
    image_digests: tuple[str, ...] = ()  # worker images allowed to produce LoRAs and scores

    @property
    def settable_fields(self) -> frozenset[str]:
        """Fields a bound may name; also the public compatibility filter's allowlist."""
        return frozenset(f.name for f in dataclasses.fields(self)) - {"miner_bounds", "code_digest", "image_digests", "method"}

    def __post_init__(self) -> None:
        _need(self.steps >= 1, "recipe.steps must be at least 1")
        _need(self.minibatches >= 1, "recipe.minibatches must be at least 1")
        _need(self.rank >= 1, "recipe.rank must be at least 1")
        _need(math.isfinite(self.lora_lr) and self.lora_lr > 0, "recipe.lora_lr must be a positive number")
        _need(math.isfinite(self.adam_eps) and self.adam_eps > 0, "recipe.adam_eps must be a finite positive number")
        _need(_unit(self.sft_weight), "recipe.sft_weight must be in [0, 1]")
        _need(math.isfinite(self.max_grad_norm) and self.max_grad_norm > 0, "recipe.max_grad_norm must be positive")
        cap = self.global_norm_cap
        _need(cap is None or (math.isfinite(cap) and cap > 0), "recipe.global_norm_cap must be positive, or null for no cap")
        _need(bool(self.target_modules), "recipe.target_modules must not be empty")
        for name, needed in TARGET_MODULE_NEEDS:
            _need(name not in self.target_modules or needed in self.target_modules, f"recipe.target_modules with {name} must hold {needed} too")
        settable = self.settable_fields
        keys = [bound.key for bound in self.miner_bounds]
        _need(len(set(keys)) == len(keys) and set(keys) <= settable,
              f"miner_bounds must name distinct recipe fields; an unknown or duplicate key needs newer miner tools. Upgrade with: {MINER_UPGRADE_COMMAND}")
        for bound in self.miner_bounds:
            _need(bool(bound.choices) or bound.low <= bound.high, f"bad miner bound for {bound.key!r}")
        # The high end of the miner bound is the ceiling for every file: the value of a file that sets none may not be above it.
        ceiling = self.bound("global_norm_cap")
        _need(ceiling is None or (cap is not None and cap <= ceiling.high), "recipe.global_norm_cap must not be above the high end of its miner bound")

    @property
    def recipe_id(self) -> str:
        return self.digest()

    def bound(self, key: str) -> MinerBound | None:
        """The bound of a miner-settable field, or None when a miner may not set ``key``."""
        return next((bound for bound in self.miner_bounds if bound.key == key), None)

    def for_miner(self, train: Mapping[str, Any]) -> Recipe:
        """This recipe with one miner's ``train`` values applied.

        Every key must be miner-settable and every value inside its bound.
        Anything else raises ``BatchRejected``: the whole file is invalid.
        """
        changes = {}
        for key, value in train.items():
            bound = self.bound(key)
            if bound is None:
                raise BatchRejected("train has a key that a miner may not set")  # never repeat the miner's own text
            changes[key] = bound.check(value)
        return dataclasses.replace(self, **changes) if changes else self


@dataclass(frozen=True)
class Params(Record):
    """One immutable snapshot of every parameter. Its digest is the ``params_hash`` pinned at round open."""

    # A stored Params without this field loads with the Fisher default. Keep its JSON and params_hash unchanged;
    # selecting the old band rule writes the field and changes the hash, just like any other nondefault setting.
    LEFT_OUT_AT_DEFAULT = (
        "null_samples_per_task", "irt_draw_rule", "kv_cache_dtype", "max_num_seqs",
        "rotation_enabled", "rotation_k", "rotation_retire_p", "rotation_n_min",
        "rotation_retire_low_p", "rotation_n_min_low", "rotation_active_size",
        "rotation_rule", "rotation_churn_frac", "rotation_prior_p", "rotation_queue_publish", "rotation_churn_max",
        "eval_merged", "null_prune_from_round", "min_kl_seq", "min_kl_tok", "kl_rollouts",
        "kl_max_tokens", "kl_time_limit_s", "confirm_next_round", "max_stages", "train_tokens_per_s_floor",
        "z_crown", "recite_max_share", "max_kl_tok", "anchor_n", "anchor_min_p", "anchor_min_rollouts", "anchor_hi_p",
        "anchor_hi_share", "anchor_z_veto", "anchor_min_scored", "merge_after_confirm",
    )

    netuid: int = 120
    env: str = "math"
    env_pin: str = ""  # sha256 that pins the public task pool and grader of ``env``; set by the train/eval workstream

    # Round size.
    batch_cap: int = 32  # launch rule: at most 32 batches (LoRAs) in one round
    max_loras_per_box: int = 32  # serving limit; batch_cap may never exceed it
    n_min_start: int = 1  # a short queue runs right away; a round never starts with 0 batches
    n_gpus: int = 8

    # Evaluation. Sample 0 of null M and every LoRA uses the same tasks, seed
    # and caps. Null M may add independent, king-only samples after sample 0.
    n_tasks: int = 2000  # placeholder: pinned so EVAL fits eval_time_budget_s on the reference box
    eval_time_budget_s: int = 3600  # placeholder
    samples_per_task: int = 1
    null_samples_per_task: int | None = None  # None = samples_per_task; omitted from JSON for old rounds
    looks: tuple[float, ...] = (0.25, 0.5, 1.0)  # shares of n_tasks; the win test runs at the last look only
    drop_alpha: float = 0.05  # PLACEHOLDER: total chance, over all early looks, to drop a LoRA that is truly on the win bar
    max_output_tokens: int = 65536
    max_context_tokens: int = 69632
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "auto"
    max_num_seqs: int = 256

    # Serve a candidate's merged base, made by the install fold, instead of a runtime LoRA.
    # Omitted while false so the parameter sets of every earlier round keep their hash.
    eval_merged: bool = False

    # Task draw.
    irt_draw_rule: str = "fisher"
    irt_middle_band: tuple[float, float] = (0.2, 0.8)  # placeholder: favoured band of the head's pass rate under "band"
    # Under "fisher"/"sqrt_fisher", share of the draw spread uniformly over every task; under "band", share for barely seen tasks.
    irt_explore_frac: float = 0.1

    # Public reserve -> active pool. Disabled by default; no reserve read or
    # write happens while disabled. These defaults are omitted from JSON.
    rotation_enabled: bool = False
    rotation_k: int = 32
    rotation_retire_p: float = 0.95
    rotation_n_min: int = 3
    rotation_retire_low_p: float = 0.0  # zero: no empirical low-pass retirement
    rotation_n_min_low: int = 8
    rotation_active_size: int = 0  # zero: size of the initial base-pool active set
    rotation_rule: str = "v1"  # "info": score every candidate by p * (1 - p). "info-guard": the same ranking, plus the fail-closed admission gate
    rotation_churn_frac: float = 0.10  # maximum steady-state swaps: floor(frac * target)
    rotation_prior_p: float = 0.5  # when neither an IRT row nor a reserve prior is known
    rotation_queue_publish: int = 2000  # ranked upcoming candidates in the public rotation decision
    rotation_churn_max: int = 0  # optional operational cap on swaps; 0 = no cap beyond churn_frac and better candidates

    # Judging. The field-median gate adds a second, final-look z test against
    # the per-task median of the post-dedup candidates plus the king.
    z_bar: float = 1.0  # the paired king bar; a winner needs z strictly above this
    # None = the crown bar is z_bar. Set, it is the crown/accept bar only (argmax_symmetric); the confirmation
    # test always uses z_bar. Omitted from JSON while None, so existing params_hash values do not change.
    z_crown: float | None = None
    crown_gate: str = "field_median"
    z_field_bar: float = 1.0  # finite; calibrated values may be negative
    cap_hit_floor: float = 0.33  # reject only when cap rate > max(cap_hit_floor, null cap rate + cap_hit_margin)
    cap_hit_margin: float = 0.02
    min_scored_tasks: int = 100  # fewer scored tasks than this can never produce a winner
    min_cover: float = 0.95  # share of rollouts that must resolve; below it the entry is no_decision
    calibration_tol: float = 0.003  # multi-box guard on the null-M rate
    confirm_next_round: bool = False  # apply a crown now; pay it only after a fresh paired win in the next round
    # With confirm_next_round: a crown is not merged into the head until its LoRA passes the next round's fresh paired
    # re-test on the unchanged head (z >= z_bar). A crown that fails is discarded. Omitted from JSON while false.
    merge_after_confirm: bool = False
    min_kl_seq: float = 0.0  # minimum mean KL(new||head) per sampled rollout; 0 = off
    min_kl_tok: float = 0.0  # minimum summed KL divided by measured completion tokens; 0 = off
    kl_rollouts: int = 128  # deterministic sample-0 subsample; capped completions included
    kl_max_tokens: int = 0  # first N completion tokens per rollout; 0 = the full completion
    kl_time_limit_s: int = 1200  # independent hard cap, also bounded by the promotion deadline
    max_kl_tok: float = 0.0  # behaviour KL ceiling: kl_tok above it cannot crown or confirm; 0 = off
    recite_max_share: float = 0.0  # anti-recitation: share of target 10-grams found in active-pool task text; 0 = off

    # Anchors: an easy-task regression check. ``anchor_n`` retired tasks the head solved (measured record pass rate
    # >= anchor_min_p over >= anchor_min_rollouts head rollouts while active) are scored next to the gate draw.
    # They never count toward delta, IRT, rotation or confirmation. A crown is vetoed when the paired anchor z is
    # below -anchor_z_veto over at least anchor_min_scored scored anchors. 0 = off; every default is omitted from JSON.
    anchor_n: int = 0
    anchor_min_p: float = 0.8
    anchor_min_rollouts: int = 4
    anchor_hi_p: float = 0.95  # stratum boundary: round(anchor_n * anchor_hi_share) anchors come from records >= this
    anchor_hi_share: float = 0.25
    anchor_z_veto: float = 1.645  # one-sided: veto when anchor z < -anchor_z_veto
    anchor_min_scored: int = 100  # fewer scored anchors: the check is not evaluated and does not veto

    recipe: Recipe = field(default_factory=Recipe)

    # One training seed for every LoRA in a new round. Legacy records without this field decode as ``batch``.
    # Legacy JSON omits these fields. ``None`` inherits their historical values; assigning a new value
    # through dataclasses.replace writes it into params_hash and cannot silently preserve the old hash.
    train_seed_scope: str = "round"
    dedup_enabled: bool = True
    dedup_rollout_overlap: float = 0.5
    dedup_cos_shared_seed: float = 0.135
    dedup_cos_own_seed: float = 0.002
    min_signal_rollouts: int = 64
    min_step_norm: float = 0.02
    _legacy_fields: frozenset[str] | None = field(default=None, compare=False, repr=False)

    # Head.
    fold_every_k: int = 1  # fold stacked LoRAs into a new base after this many applies; 1 = every green

    # Admission. The batch checker and the miner contract read these same values.
    max_batch_bytes: int = 128 * 1024 * 1024
    max_prompts: int = 2000
    max_rollouts: int = 8000
    min_reasoning_tokens: int = 32
    reasoning_token_share: float = 0.5
    max_answer_only_frac: float = 0.05
    max_answer_repeat_frac: float = 0.10
    max_answer_repeat_count: int = 32
    reward_tolerance: float = 0.0
    min_pool_prompt_frac: float = 0.0  # 0 = prompts need not come from the public pool
    # PLACEHOLDER: cap on steps x the tokens the trainer runs through the model for one file; 0 = no limit.
    # The cap does not count minibatches: each optimizer update has a fixed cost that no admission rule counts.
    max_train_token_steps: int = 16_000_000
    max_stages: int = 5  # affine-batch-v4: one to five ordered GRPO/SFT stages
    # When positive, reserve enough wall time for the configured token-step
    # floor plus the ten-minute post-training reserve. 0 leaves old params off.
    train_tokens_per_s_floor: int = 0

    # Queue and leases.
    round_time_limit_s: int = 100 * 60  # launch rule: at most this long from the lease to the promotion of the winner
    # PLACEHOLDER: training time of a round, from the lease. At its end no training starts and a job that still runs
    # is stopped; that batch is not charged and is first in the next round. Then the scoring starts.
    train_time_budget_s: int = 40 * 60
    promote_reserve_s: int = 600  # PLACEHOLDER: the box stops scoring this long before the limit, so judging and promotion fit
    # PLACEHOLDER: the common prefix scored at the stop time is truncated to a whole block of this many tasks.
    # Streaming feeds each server continuously; this is no longer a generation or grading barrier. A block that
    # the limit cuts is never used, and fewer tasks than the first look decide nothing. So 500 = the first look.
    eval_chunk_tasks: int = 500
    # A box result that arrives later than this is never used. The box stops scoring at the round's time limit minus
    # promote_reserve_s, whatever this value is. The time after the limit is only a reserve for the steps between the
    # stop and the arrival of the result (grading of the last chunk, stop of the servers), so that a round that
    # stopped in time never loses its result to the lease. No task is scored in it.
    lease_timeout_s: int = 2 * 3600
    max_infra_attempts: int = 3  # no_decision outcomes before an entry is held for the operator
    squat_min_wait_s: int = 86400
    squat_age_factor: float = 2.0
    apply_retry_s: int = 300
    apply_max_attempts: int = 12

    # Payout.
    kings_window: int = 10
    # Empty: the original last-window rule. Otherwise judged nulls from this round prune the oldest paid king.
    # Omitted while empty so every earlier params record keeps its canonical form and hash.
    null_prune_from_round: str = ""
    burn_uid: int = 0  # the owner's UID: it has no queue seat. No burn: the weights writer refuses a table with weight on it and sets nothing
    weights_cadence_s: int = 3600
    weights_rate_limit_blocks: int = 100

    # Records from before these fields use these values; new Params use the production defaults above.
    LEGACY_FIELDS = ("train_seed_scope", "dedup_enabled", "dedup_rollout_overlap", "dedup_cos_shared_seed",
                     "dedup_cos_own_seed", "min_signal_rollouts", "min_step_norm", "crown_gate", "z_field_bar")
    LEGACY_VALUES = dict(train_seed_scope="batch", dedup_enabled=False, dedup_rollout_overlap=0.5,
                         dedup_cos_shared_seed=0.135, dedup_cos_own_seed=0.002, min_signal_rollouts=1, min_step_norm=0.0,
                         crown_gate="king", z_field_bar=1.0)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Params:
        legacy = set(cls.LEGACY_FIELDS) - set(data)
        params = from_jsonable(cls, data, cls.__name__)
        if legacy:
            for name in legacy:
                object.__setattr__(params, name, cls.LEGACY_VALUES[name])
            object.__setattr__(params, "_legacy_fields", frozenset(legacy))
        return params

    @classmethod
    def from_public_dict(cls, data: Mapping[str, Any]) -> Params:
        """Read a public mining.json, ignoring fields a newer validator adds.

        Journal and ticket decoding must use ``from_dict``: it stays strict.
        Unknown record fields are removed recursively for this read-only path;
        every known field is still checked by the strict constructor.
        """
        if not isinstance(data, Mapping):
            return cls.from_dict(data)

        def known(tp: Any, value: Any) -> Any:
            if typing.get_origin(tp) in (typing.Union, types.UnionType):
                args = [arg for arg in typing.get_args(tp) if arg is not type(None)]
                tp = args[0] if len(args) == 1 else Any
            if dataclasses.is_dataclass(tp) and isinstance(value, Mapping):
                if tp is Recipe and isinstance(value.get("miner_bounds"), list):
                    # A future bound is not a local training permission. Drop it only on this public-read path;
                    # canonical journals, tickets and stored Params keep the strict Recipe constructor.
                    known_keys = Recipe().settable_fields
                    value = value | {"miner_bounds": [bound for bound in value["miner_bounds"]
                                                       if not isinstance(bound, Mapping) or bound.get("key") in known_keys]}
                return {key: known(hint, value[key]) for key, hint in _hints(tp).items()
                        if not key.startswith("_") and key in value}
            origin, args = typing.get_origin(tp), typing.get_args(tp)
            if origin in (dict, list, tuple) and isinstance(value, (list, tuple, Mapping)):
                if origin is dict:
                    return {key: known(args[1] if len(args) == 2 else Any, item) for key, item in value.items()}
                if origin is list or (len(args) == 2 and args[1] is Ellipsis):
                    return [known(args[0] if args else Any, item) for item in value]
                return [known(hint, item) for hint, item in zip(args, value)]
            return value

        return cls.from_dict(known(cls, data))

    def to_dict(self) -> dict[str, Any]:
        legacy = self._legacy_fields or ()
        # Omission is valid only while the field retains the exact effective legacy value.
        data = {f.name: to_jsonable(getattr(self, f.name), _hints(Params)[f.name])
                for f in dataclasses.fields(self)
                if not f.name.startswith("_") and (f.name not in legacy or getattr(self, f.name) != self.LEGACY_VALUES.get(f.name))
                and not (f.name in self.LEFT_OUT_AT_DEFAULT and getattr(self, f.name) == f.default)}
        return data

    budget_usd_per_hour: float = 200.0

    def __post_init__(self) -> None:
        _need(1 <= self.batch_cap <= self.max_loras_per_box, "batch_cap must be between 1 and max_loras_per_box")
        _need(1 <= self.n_min_start <= self.batch_cap, "n_min_start must be between 1 and batch_cap")
        _need(self.n_gpus >= 1 and self.n_tasks >= 1 and self.samples_per_task >= 1, "n_gpus, n_tasks, samples_per_task >= 1")
        _need(self.null_samples_per_task is None or (type(self.null_samples_per_task) is int and self.null_samples_per_task >= 1), "null_samples_per_task must be null or an integer of at least 1")
        _need(self.max_num_seqs >= 1, "max_num_seqs must be positive")
        _need(bool(self.looks) and self.looks[-1] == 1.0, "looks must end at 1.0")
        _need(all(0 < a < b for a, b in zip(self.looks, self.looks[1:])) and self.looks[0] > 0, "looks must rise from above 0")
        _need(math.isfinite(self.z_bar) and self.z_bar >= 0, "z_bar must be zero or more")
        _need(self.z_crown is None or (math.isfinite(self.z_crown) and self.z_crown >= 0), "z_crown must be None or zero or more")
        _need(self.z_crown is None or self.crown_gate == "argmax_symmetric", "z_crown applies only to crown_gate argmax_symmetric")
        _need(self.crown_gate in ("king", "field_median", "argmax_symmetric"), "crown_gate must be 'king', 'field_median' or 'argmax_symmetric'")
        _need(math.isfinite(self.z_field_bar), "z_field_bar must be finite")
        _need(self.train_seed_scope in SEED_SCOPES, "train_seed_scope must be 'round' or 'batch'")
        _need(_unit(self.dedup_rollout_overlap) and self.dedup_rollout_overlap > 0, "dedup_rollout_overlap must be in (0, 1]")
        _need(_unit(self.dedup_cos_shared_seed) and self.dedup_cos_shared_seed > 0, "dedup_cos_shared_seed must be in (0, 1]")
        _need(_unit(self.dedup_cos_own_seed) and self.dedup_cos_own_seed > 0, "dedup_cos_own_seed must be in (0, 1]")
        _need(self.min_signal_rollouts >= 1, "min_signal_rollouts must be at least 1")
        _need(math.isfinite(self.min_step_norm) and self.min_step_norm >= 0, "min_step_norm must be finite and nonnegative")
        _need(math.isfinite(self.min_kl_seq) and self.min_kl_seq >= 0, "min_kl_seq must be finite and nonnegative")
        _need(math.isfinite(self.min_kl_tok) and self.min_kl_tok >= 0, "min_kl_tok must be finite and nonnegative")
        _need(math.isfinite(self.max_kl_tok) and self.max_kl_tok >= 0, "max_kl_tok must be finite and nonnegative")
        _need(self.max_kl_tok <= 0 or self.max_kl_tok > self.min_kl_tok, "max_kl_tok must be above min_kl_tok")
        _need(self.max_kl_tok <= 0 or self.kl_floor_on, "max_kl_tok needs the KL measurement")
        _need(isinstance(self.recite_max_share, (int, float)) and not isinstance(self.recite_max_share, bool)
              and 0 <= self.recite_max_share < 1, "recite_max_share must be in [0, 1)")
        _need(type(self.kl_rollouts) is int and self.kl_rollouts >= 8, "kl_rollouts must be an integer of at least 8")
        _need(type(self.kl_max_tokens) is int and self.kl_max_tokens >= 0, "kl_max_tokens must be a nonnegative integer")
        _need(type(self.kl_time_limit_s) is int and self.kl_time_limit_s > 0, "kl_time_limit_s must be a positive integer")
        _need(type(self.anchor_n) is int and self.anchor_n >= 0, "anchor_n must be an integer of at least 0")
        _need(type(self.anchor_min_rollouts) is int and self.anchor_min_rollouts >= 0, "anchor_min_rollouts must be an integer of at least 0")
        _need(type(self.anchor_min_scored) is int and self.anchor_min_scored >= 0, "anchor_min_scored must be an integer of at least 0")
        _need(_unit(self.anchor_min_p) and _unit(self.anchor_hi_p) and _unit(self.anchor_hi_share), "anchor_min_p, anchor_hi_p and anchor_hi_share must be in [0, 1]")
        _need(math.isfinite(self.anchor_z_veto) and self.anchor_z_veto >= 0, "anchor_z_veto must be zero or more")
        # Anchors live only on the argmax path: it has no early drops, so no anchor can enter a drop test.
        _need(self.anchor_n == 0 or self.crown_gate == "argmax_symmetric", "anchor_n applies only to crown_gate argmax_symmetric")
        _need(0.0 < self.drop_alpha < 1.0, "drop_alpha must be between 0 and 1")
        _need(_unit(self.cap_hit_floor) and _unit(self.cap_hit_margin), "cap-hit settings must be in [0, 1]")
        _need(_unit(self.min_cover) and self.min_cover > 0, "min_cover must be in (0, 1]")
        _need(self.min_scored_tasks >= 2, "min_scored_tasks must be at least 2: one task has no standard error")
        _need(self.irt_draw_rule in ("band", "fisher", "sqrt_fisher"), "irt_draw_rule must be 'band', 'fisher' or 'sqrt_fisher'")
        low, high = self.irt_middle_band
        _need(_unit(low) and _unit(high) and low < high, "irt_middle_band must be two rising shares")
        _need(_unit(self.irt_explore_frac), "irt_explore_frac must be in [0, 1]")
        _need(self.rotation_k >= 0 and self.rotation_n_min >= 0 and self.rotation_n_min_low >= 0, "rotation counts must be >= 0")
        _need(self.rotation_active_size >= 0, "rotation_active_size must be >= 0")
        _need(self.rotation_rule in ("v1", "info", "info-guard"), "rotation_rule must be 'v1', 'info' or 'info-guard'")
        _need(_unit(self.rotation_churn_frac) and _unit(self.rotation_prior_p), "rotation churn fraction and prior must be in [0, 1]")
        _need(self.rotation_queue_publish >= 0 and self.rotation_churn_max >= 0, "rotation queue and churn cap must be >= 0")
        _need(_unit(self.rotation_retire_p) and _unit(self.rotation_retire_low_p), "rotation pass rates must be in [0, 1]")
        _need(self.fold_every_k >= 1, "fold_every_k must be at least 1")
        _need(type(self.confirm_next_round) is bool, "confirm_next_round must be true or false")
        _need(type(self.eval_merged) is bool, "eval_merged must be true or false")
        _need(type(self.merge_after_confirm) is bool, "merge_after_confirm must be true or false")
        _need(not self.merge_after_confirm or self.confirm_next_round, "merge_after_confirm needs confirm_next_round")
        _need(not self.merge_after_confirm or self.fold_every_k == 1, "merge_after_confirm needs fold_every_k = 1")
        _need(not self.eval_merged or self.fold_every_k == 1, "merged-candidate evaluation needs fold_every_k = 1")
        _need(self.max_output_tokens < self.max_context_tokens, "max_output_tokens must be below max_context_tokens")
        _need(self.lease_timeout_s > 0 and self.max_infra_attempts >= 1, "lease settings must be positive")
        _need(0 <= self.promote_reserve_s < self.round_time_limit_s, "promote_reserve_s must be below round_time_limit_s")
        _need(self.lease_timeout_s >= self.round_time_limit_s, "lease_timeout_s must not be below round_time_limit_s")
        _need(
            0 < self.train_time_budget_s < self.round_time_limit_s - self.promote_reserve_s,
            "train_time_budget_s must leave time for the scoring inside round_time_limit_s",
        )
        _need(type(self.max_stages) is int and 1 <= self.max_stages <= 5, "max_stages must be an integer from 1 to 5")
        _need(type(self.train_tokens_per_s_floor) is int and self.train_tokens_per_s_floor >= 0, "train_tokens_per_s_floor must be a nonnegative integer")
        _need(self.train_tokens_per_s_floor <= 0 or self.train_time_budget_s >= math.ceil(self.max_train_token_steps / self.train_tokens_per_s_floor) + 600,
              "train_time_budget_s must cover max_train_token_steps / train_tokens_per_s_floor, rounded up, plus 600 seconds")
        _need(self.eval_chunk_tasks >= 1, "eval_chunk_tasks must be at least 1")
        _need(self.apply_retry_s > 0 and self.apply_max_attempts >= 1, "apply retry settings must be positive")
        _need(self.kings_window >= 1 and self.burn_uid >= 0, "kings_window >= 1 and burn_uid >= 0")
        _need(isinstance(self.null_prune_from_round, str) and (not self.null_prune_from_round or re.fullmatch(r"r\d{6}", self.null_prune_from_round)), "null_prune_from_round must be empty or rNNNNNN")
        _need(self.weights_cadence_s > 0, "weights_cadence_s must be positive")
        _need(math.isfinite(self.budget_usd_per_hour) and self.budget_usd_per_hour > 0, "budget must be positive")

    @property
    def crown_bar(self) -> float:
        """The bar a crown needs (strict z above it). Confirmation never uses this; it uses ``z_bar``."""
        return self.z_bar if self.z_crown is None else self.z_crown

    @property
    def null_samples(self) -> int:
        return self.samples_per_task if self.null_samples_per_task is None else self.null_samples_per_task

    @property
    def kl_floor_on(self) -> bool:
        """KL is measured: a floor or the ``max_kl_tok`` ceiling is on."""
        return self.min_kl_seq > 0 or self.min_kl_tok > 0 or self.max_kl_tok > 0

    @property
    def params_hash(self) -> str:
        return self.digest()

    @property
    def recipe_id(self) -> str:
        return self.recipe.recipe_id

    def gpus_per_lora(self, n_batches: int) -> int:
        """Elastic training on one box: floor(n_gpus / batches), and never less than one GPU."""
        _need(n_batches >= 1, "a round needs at least one batch")
        return max(1, self.n_gpus // n_batches)

    def look_sizes(self, n_tasks: int | None = None) -> tuple[int, ...]:
        """How many tasks are scored by the end of each look. The last look is always every task."""
        total = self.n_tasks if n_tasks is None else n_tasks
        sizes: list[int] = []
        for share in self.looks:
            size = total if share == 1.0 else max(1, math.ceil(share * total))
            if not sizes or size > sizes[-1]:
                sizes.append(size)
        return tuple(sizes)


@dataclass(frozen=True)
class GenesisSpec(Record):
    """The first head of the log. A real genesis needs both digests from the real files."""

    model_repo: str
    model_revision: str
    base_hash: str
    base_fp32_master_digest: str
    validator_hotkey: str  # ss58 address of the one hotkey allowed to set weights

    def __post_init__(self) -> None:
        _need(bool(self.model_repo) and bool(self.model_revision), "genesis needs a model repo and revision")
        _need(is_sha256(self.base_hash), "genesis base_hash must be a 64-hex sha256")
        _need(is_sha256(self.base_fp32_master_digest), "genesis base_fp32_master_digest must be a 64-hex sha256")
        _need(bool(self.validator_hotkey), "genesis needs the validator hotkey address")


def load_params(path: str | Path) -> Params:
    """Read a file of *new overrides*, not a historical serialized Params record.

    Missing new fields mean the current production defaults here. Only journal/store decoding via
    ``Params.from_dict`` gives old records their legacy behavior and preserves their digest.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read params file {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError("a params override file must hold a JSON object")
    return from_jsonable(Params, data, "Params")


def require_env(name: str, environ: Mapping[str, str] | None = None) -> str:
    """Read a required environment variable. A missing value is an error, never a default path."""
    value = (os.environ if environ is None else environ).get(name, "")
    if not value.strip():
        raise ConfigError(f"environment variable {name} is not set; refusing to guess a default")
    return value
