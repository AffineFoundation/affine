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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .canonical import Record, digest, is_sha256
from .errors import BatchRejected, ConfigError

# Replay-only compatibility for the one reviewed AffineII combined release.
# Seven dedup fields and eval_merged are absent from old Params on purpose;
# the historical crown-threshold override must also be removed to restore old
# z_bar=1.0. Old training and old params_set records retain their pre-dedup
# behavior and bba7a409... hash. The loader validates the complete historical
# record before dropping these fields.
DEDUP_FIELDS = {
    "train_seed_scope": "round",
    "dedup_enabled": True,
    "dedup_rollout_overlap": 0.5,
    "dedup_cos_shared_seed": 0.135,
    "dedup_cos_own_seed": 0.002,
    "min_signal_rollouts": 1,
    "min_step_norm": 0.0,
    "eval_merged": True,
}
FIELD_GATE_FIELDS = {"crown_gate": ("field_median", "argmax_symmetric")}
RELEASE_FIELDS = DEDUP_FIELDS | {"crown_gate": "field_median"} | {"z_field_bar": 1.0, "z_bar": 1.5}
HISTORICAL_NEW_PARAMS_HASH = "d5979c301c7acf8a56cb6a9989e5b212e5887ec1bf0117f9250202af6a95f35c"
REVIEWED_NEW_PARAMS_HASH = "8caddd9c6fc276a9669dff9e01e1bb7e437d98bed5b12a605141e6025b9f9ec4"
FIELD_GATE_PARAMS_HASH = "642cff7a80a51ca30606ff775e96693fa2de6d546703d9c2c30703519a72b8ed"
K1F1_PARAMS_HASH = "471b74f4a306efbb1e9444069ea832b5e02d68ec410f5ffebb06353f2deb9c5a"
KING_LED_PARAMS_HASH = "848707723628c85dc13f8c93e39d132432fcb5851076165490ac01693176236e"
K025F025_PARAMS_HASH = "ab2c7084bc94a7ce1ac6138498447aeaecadc117ae66e43131dfc1c0d5aa268d"
ARGMAX_PARAMS_HASH = "ab4e228c961058a29ed062bc67f6da6881cf3c13396eb03730d54badb4f261c0"
NULL_PRUNE_PARAMS_HASH = "1449210f89f30a65dc3f6e8d4ac956f5f2c1c8d37364a46b20e5cfb01c24eace"
TRAIN64K_PARAMS_HASH = "a8d0f5fe50a1d47de9cb5bac618453321ce58d460f75618290a0db3001d695ac"
TRAIN64K_T3600_PARAMS_HASH = "05b688fcc0ceddb4bc78b56cef471110a9b9b107cc82bf2b6318c7d1ad355f40"
TRAIN64K_T3600_PRUNE_PARAMS_HASH = "c5d4405bcfb01b68766f97409f7ce1bfc98f875cf77af661421a8a3a28df2057"
TRAIN64K_PARAMS_HASHES = (TRAIN64K_PARAMS_HASH, TRAIN64K_T3600_PARAMS_HASH, TRAIN64K_T3600_PRUNE_PARAMS_HASH)
TRAIN64K_LEGACY_VALUES = {
    "recipe.train_max_output_tokens": (65536, 32768),
    "recipe.train_max_context_tokens": (69632, 36864),
    "max_train_token_steps": (22130000, 16000000),
    "train_time_budget_s": ((2400, 3600), 2400),
}
OLD_FIELD_GATE_PARAMS_HASHES = (
    "17c20f822fb9c97b397d24272e18eaebf8de9574a9d0056361c8d529856696d4",
    "346b99a875be5a9d5e2467bf846f20d124229397b67935c6563f1e7c2409b8b1",
    "05a1b55556a14cd001f926f774a4231023851e43c2393ec41bcd23646fe6ca4b",
    "0eb4867f6c08aa179015c9e927bb38d6ade14a022c406401a026280845a8e098",
)
FIELD_GATE_PARAMS_HASHES = (
    FIELD_GATE_PARAMS_HASH, K1F1_PARAMS_HASH, KING_LED_PARAMS_HASH, K025F025_PARAMS_HASH,
    ARGMAX_PARAMS_HASH, *TRAIN64K_PARAMS_HASHES,
)
NULL_PRUNE_FIELD = "null_prune_from_round"
NULL_PRUNE_CUTOFF = "r000098"
REVIEWED_NEW_PARAMS_HASHES = frozenset((
    HISTORICAL_NEW_PARAMS_HASH, REVIEWED_NEW_PARAMS_HASH, *FIELD_GATE_PARAMS_HASHES,
    NULL_PRUNE_PARAMS_HASH, *OLD_FIELD_GATE_PARAMS_HASHES,
))
REVIEWED_OLD_PARAMS_HASH = "bba7a409fc12d9ceb7b85f4e935fadfbd3f4b1efccedb7a6fde90c451f6e534a"
HISTORICAL_NEW_CODE_DIGEST = "28f1fb8bc7bd4e415a4f7a62c5876d99a4404bee7b4afd156a0290b81eb6510a"
REVIEWED_NEW_CODE_DIGEST = "22b2d191b32e394d4621ce477e702b50410a6ef7706cce6cfd20c60ff64e42e1"
OLD_FIELD_GATE_CODE_DIGEST = "2f7fd4ecaa8fa26800d8a3f54063d6286d614e309fcdffd9af59bfffeb7824fa"
REVIEWED_NEW_CODE_DIGESTS = dict.fromkeys(REVIEWED_NEW_PARAMS_HASHES, REVIEWED_NEW_CODE_DIGEST) | {
    HISTORICAL_NEW_PARAMS_HASH: HISTORICAL_NEW_CODE_DIGEST,
    REVIEWED_NEW_PARAMS_HASH: OLD_FIELD_GATE_CODE_DIGEST,
    **dict.fromkeys(OLD_FIELD_GATE_PARAMS_HASHES, OLD_FIELD_GATE_CODE_DIGEST),
}
# Train-render code was already live for the k0/f1 field-led pin.
REVIEWED_NEW_CODE_DIGESTS["17c20f822fb9c97b397d24272e18eaebf8de9574a9d0056361c8d529856696d4"] = REVIEWED_NEW_CODE_DIGEST
REVIEWED_OLD_CODE_DIGEST = "5027620615cbd7bc707668cc9f93e98e2133180613404b558e6f05a2b648eb47"

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
    LEFT_OUT_AT_DEFAULT = ("minibatches", "adam_eps")

    method: str = "grpo-lora"
    steps: int = 1  # passes over the whole file
    minibatches: int = 1  # optimizer updates in one pass; 1 = one update on the whole file
    lora_lr: float = 1e-5
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    global_norm_cap: float | None = 6.0  # cap on the norm of the whole adapter update; None = no cap
    group_size: int = 0  # PLACEHOLDER: rollouts per GRPO group; 0 = use each prompt's rollouts as the file groups them
    sample_temperature: float = 1.0  # temperature the batch's rollouts were sampled at
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

    def __post_init__(self) -> None:
        _need(self.steps >= 1, "recipe.steps must be at least 1")
        _need(self.minibatches >= 1, "recipe.minibatches must be at least 1")
        _need(self.rank >= 1, "recipe.rank must be at least 1")
        _need(math.isfinite(self.lora_lr) and self.lora_lr > 0, "recipe.lora_lr must be a positive number")
        _need(math.isfinite(self.adam_eps) and self.adam_eps > 0, "recipe.adam_eps must be a finite positive number")
        _need(math.isfinite(self.max_grad_norm) and self.max_grad_norm > 0, "recipe.max_grad_norm must be positive")
        cap = self.global_norm_cap
        _need(cap is None or (math.isfinite(cap) and cap > 0), "recipe.global_norm_cap must be positive, or null for no cap")
        _need(bool(self.target_modules), "recipe.target_modules must not be empty")
        for name, needed in TARGET_MODULE_NEEDS:
            _need(name not in self.target_modules or needed in self.target_modules, f"recipe.target_modules with {name} must hold {needed} too")
        settable = {f.name for f in dataclasses.fields(self)} - {"miner_bounds", "code_digest", "image_digests", "method"}
        keys = [bound.key for bound in self.miner_bounds]
        _need(len(set(keys)) == len(keys) and set(keys) <= settable, "miner_bounds must name distinct recipe fields")
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
    rotation_rule: str = "v1"  # "info": score every candidate by p * (1 - p)
    rotation_churn_frac: float = 0.10  # maximum steady-state swaps: floor(frac * target)
    rotation_prior_p: float = 0.5  # when neither an IRT row nor a reserve prior is known
    rotation_queue_publish: int = 2000  # ranked upcoming candidates in the public rotation decision
    rotation_churn_max: int = 0  # optional operational cap on swaps; 0 = no cap beyond churn_frac and better candidates

    # Judging.
    z_bar: float = 1.0  # a winner needs z strictly above this
    cap_hit_floor: float = 0.33  # reject only when cap rate > max(cap_hit_floor, null cap rate + cap_hit_margin)
    cap_hit_margin: float = 0.02
    min_scored_tasks: int = 100  # fewer scored tasks than this can never produce a winner
    min_cover: float = 0.95  # share of rollouts that must resolve; below it the entry is no_decision
    calibration_tol: float = 0.003  # multi-box guard on the null-M rate

    recipe: Recipe = field(default_factory=Recipe)

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
    burn_uid: int = 0  # the owner's UID: it has no queue seat. No burn: the weights writer refuses a table with weight on it and sets nothing
    weights_cadence_s: int = 3600
    weights_rate_limit_blocks: int = 100

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
        _need(self.rotation_rule in ("v1", "info"), "rotation_rule must be 'v1' or 'info'")
        _need(_unit(self.rotation_churn_frac) and _unit(self.rotation_prior_p), "rotation churn fraction and prior must be in [0, 1]")
        _need(self.rotation_queue_publish >= 0 and self.rotation_churn_max >= 0, "rotation queue and churn cap must be >= 0")
        _need(_unit(self.rotation_retire_p) and _unit(self.rotation_retire_low_p), "rotation pass rates must be in [0, 1]")
        _need(self.fold_every_k >= 1, "fold_every_k must be at least 1")
        _need(self.max_output_tokens < self.max_context_tokens, "max_output_tokens must be below max_context_tokens")
        _need(self.lease_timeout_s > 0 and self.max_infra_attempts >= 1, "lease settings must be positive")
        _need(0 <= self.promote_reserve_s < self.round_time_limit_s, "promote_reserve_s must be below round_time_limit_s")
        _need(self.lease_timeout_s >= self.round_time_limit_s, "lease_timeout_s must not be below round_time_limit_s")
        _need(
            0 < self.train_time_budget_s < self.round_time_limit_s - self.promote_reserve_s,
            "train_time_budget_s must leave time for the scoring inside round_time_limit_s",
        )
        _need(self.eval_chunk_tasks >= 1, "eval_chunk_tasks must be at least 1")
        _need(self.apply_retry_s > 0 and self.apply_max_attempts >= 1, "apply retry settings must be positive")
        _need(self.kings_window >= 1 and self.burn_uid >= 0, "kings_window >= 1 and burn_uid >= 0")
        _need(self.weights_cadence_s > 0, "weights_cadence_s must be positive")
        _need(math.isfinite(self.budget_usd_per_hour) and self.budget_usd_per_hour > 0, "budget must be positive")

    @property
    def null_samples(self) -> int:
        return self.samples_per_task if self.null_samples_per_task is None else self.null_samples_per_task

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Params":
        """Decode exactly the reviewed dedup pin, without adopting its behavior.

        Unknown/newer new-code fields fail in Record.from_dict. A dedup pin is
        checked as a complete canonical record, so rollback cannot silently
        accept a different threshold, recipe or code digest under the same name.
        The exact reviewed null-prune record may additionally carry its one new
        field; removing it restores the already-reviewed argmax record, whose
        ordinary translation below is then required to succeed.
        """
        if not isinstance(data, Mapping):
            raise ConfigError("Params must be a JSON object")
        if NULL_PRUNE_FIELD in data:
            if digest(data) not in (NULL_PRUNE_PARAMS_HASH, TRAIN64K_T3600_PRUNE_PARAMS_HASH) or data[NULL_PRUNE_FIELD] != NULL_PRUNE_CUTOFF:
                raise ConfigError("rollback compatibility accepts only the reviewed null-prune params record")
            data = {key: value for key, value in data.items() if key != NULL_PRUNE_FIELD}
            if digest(data) not in (ARGMAX_PARAMS_HASH, TRAIN64K_T3600_PARAMS_HASH):
                raise ConfigError("null-prune params do not reduce to the reviewed argmax record")
        fields = set(data) & set(RELEASE_FIELDS)
        if fields:
            unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
            if unknown not in (set(DEDUP_FIELDS), set(RELEASE_FIELDS) - {"z_bar"}):
                # Let the ordinary strict decoder report any other new field.
                return super().from_dict(data)
            params_hash = digest(data)
            if params_hash not in REVIEWED_NEW_PARAMS_HASHES:
                raise ConfigError("historical params are not a reviewed dedup or field-gate pin")
            is_field_gate = params_hash in FIELD_GATE_PARAMS_HASHES + OLD_FIELD_GATE_PARAMS_HASHES
            expected = RELEASE_FIELDS | ({"crown_gate": data["crown_gate"], "z_field_bar": data["z_field_bar"], "z_bar": data["z_bar"]} if is_field_gate else {})
            present = set(RELEASE_FIELDS) if is_field_gate else set(DEDUP_FIELDS) | {"z_bar"}
            if {name: data[name] for name in present} != {name: expected[name] for name in present}:
                raise ConfigError("rollback compatibility accepts only the reviewed dedup or field-gate parameter values")
            if is_field_gate and data["crown_gate"] not in FIELD_GATE_FIELDS["crown_gate"]:
                raise ConfigError("rollback compatibility accepts only reviewed field-gate modes")
            if not is_field_gate and any(name in data for name in FIELD_GATE_FIELDS):
                raise ConfigError("rollback compatibility accepts only the reviewed dedup parameter values")
            if data.get("recipe", {}).get("code_digest") != REVIEWED_NEW_CODE_DIGESTS[params_hash]:
                # Each reviewed release pins its own merged-evaluation worker code.
                raise ConfigError("rollback compatibility accepts only the reviewed combined worker recipe")
            if params_hash in TRAIN64K_PARAMS_HASHES:
                # These are ordinary old Params/Recipe values, not unknown
                # fields. Restore the exact old record after authenticating the
                # complete new pin, or old code would retain the new caps and
                # the translated record would not have the legacy params hash.
                recipe = data.get("recipe", {})
                for name, (new_value, _old_value) in TRAIN64K_LEGACY_VALUES.items():
                    if name.startswith("recipe."):
                        actual = recipe.get(name.removeprefix("recipe."))
                    else:
                        actual = data.get(name)
                    if actual not in (new_value if isinstance(new_value, tuple) else (new_value,)):
                        raise ConfigError("rollback compatibility accepts only the reviewed train64k values")
                data = dict(data)
                data["recipe"] = dict(recipe, train_max_output_tokens=32768, train_max_context_tokens=36864)
                data.update(max_train_token_steps=16000000, train_time_budget_s=2400)
            # Strip every release-only judging override, including z_bar. Old
            # rollback behavior stays at the legacy z_bar=1.0 for every pin.
            data = {key: value for key, value in data.items() if key not in RELEASE_FIELDS}
            # Historical replay exposes the old worker identity too. The full
            # combined pin above authorizes this translation; the journal bytes
            # themselves remain untouched and native release readers still see
            # the original new recipe digest.
            data["recipe"] = dict(data.get("recipe", {})) | {"code_digest": REVIEWED_OLD_CODE_DIGEST}
        return super().from_dict(data)

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
    """Read a JSON file of overrides. Unknown keys and wrong types fail closed."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read params file {path}: {exc}") from exc
    return Params.from_dict(data)


def require_env(name: str, environ: Mapping[str, str] | None = None) -> str:
    """Read a required environment variable. A missing value is an error, never a default path."""
    value = (os.environ if environ is None else environ).get(name, "")
    if not value.strip():
        raise ConfigError(f"environment variable {name} is not set; refusing to guess a default")
    return value
