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

from .canonical import Record, is_sha256
from .errors import BatchRejected, ConfigError

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
            names = list(value) if isinstance(value, (list, tuple)) else None
            if not names or len(set(names)) != len(names) or not set(names) <= set(self.choices):
                raise BatchRejected(f"train.{self.key} must be a non-empty subset of {list(self.choices)}")
            return tuple(name for name in self.choices if name in names)
        wanted = (int,) if self.integer else (int, float)
        if type(value) not in wanted or not math.isfinite(value) or not self.low <= value <= self.high:
            kind = "a whole number" if self.integer else "a number"
            raise BatchRejected(f"train.{self.key} must be {kind} from {self.low:g} to {self.high:g}")
        return value if self.integer else float(value)


# The five things a miner may set: learning rate; steps; GRPO group size and
# sampling temperature; LoRA rank and alpha; target modules.
# PLACEHOLDER marks a bound the spec does not give. It is not a decision.
DEFAULT_MINER_BOUNDS = (
    MinerBound("lora_lr", 1e-6, 1e-4),  # PLACEHOLDER
    MinerBound("steps", 1, 4, integer=True),  # PLACEHOLDER
    MinerBound("group_size", 2, 64, integer=True),  # PLACEHOLDER
    MinerBound("sample_temperature", 0.5, 1.5),  # PLACEHOLDER
    MinerBound("rank", 8, 64, integer=True, decided=True),  # spec: "suggested 8-64"
    MinerBound("alpha", 8.0, 128.0),  # PLACEHOLDER
    MinerBound("target_modules", choices=ATTENTION_AND_MLP_PROJECTIONS, decided=True),  # spec: these 7 projections
)


@dataclass(frozen=True)
class Recipe(Record):
    """The published training recipe. Its digest is the ``recipe_id``.

    The validator owns every field. A miner may set only the fields named in
    ``miner_bounds``, each inside its bound. ``miner_bounds`` is the one
    place that the batch checker, the trainer and the publisher read.
    """

    method: str = "grpo-lora"
    steps: int = 1
    lora_lr: float = 1e-5
    max_grad_norm: float = 1.0
    global_norm_cap: float | None = None  # optional cap on the norm of the whole adapter update
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
        _need(self.rank >= 1, "recipe.rank must be at least 1")
        _need(math.isfinite(self.lora_lr) and self.lora_lr > 0, "recipe.lora_lr must be a positive number")
        _need(math.isfinite(self.max_grad_norm) and self.max_grad_norm > 0, "recipe.max_grad_norm must be positive")
        _need(bool(self.target_modules), "recipe.target_modules must not be empty")
        settable = {f.name for f in dataclasses.fields(self)} - {"miner_bounds", "code_digest", "image_digests", "method"}
        keys = [bound.key for bound in self.miner_bounds]
        _need(len(set(keys)) == len(keys) and set(keys) <= settable, "miner_bounds must name distinct recipe fields")
        for bound in self.miner_bounds:
            _need(bool(bound.choices) or bound.low <= bound.high, f"bad miner bound for {bound.key!r}")

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
                raise BatchRejected(f"train.{key} is not a miner-settable field")
            changes[key] = bound.check(value)
        return dataclasses.replace(self, **changes) if changes else self


@dataclass(frozen=True)
class Params(Record):
    """One immutable snapshot of every parameter. Its digest is the ``params_hash`` pinned at round open."""

    netuid: int = 120
    env: str = "math"
    env_pin: str = ""  # sha256 that pins the public task pool and grader of ``env``; set by the train/eval workstream

    # Round size.
    batch_cap: int = 32  # placeholder: most batches (LoRAs) in one round
    max_loras_per_box: int = 32  # serving limit; batch_cap may never exceed it
    n_min_start: int = 1  # a short queue runs right away; a round never starts with 0 batches
    n_gpus: int = 8

    # Evaluation. Null M and every LoRA use the same tasks, seeds and caps.
    n_tasks: int = 2000  # placeholder: pinned so EVAL fits eval_time_budget_s on the reference box
    eval_time_budget_s: int = 3600  # placeholder
    samples_per_task: int = 1
    looks: tuple[float, ...] = (0.25, 0.5, 1.0)  # shares of n_tasks; the win test runs at the last look only
    drop_alpha: float = 0.05  # PLACEHOLDER: total chance, over all early looks, to drop a LoRA that is truly on the win bar
    max_output_tokens: int = 65536
    max_context_tokens: int = 69632
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    dtype: str = "bfloat16"

    # Task draw.
    irt_middle_band: tuple[float, float] = (0.2, 0.8)  # placeholder: favoured band of the head's pass rate
    irt_explore_frac: float = 0.1  # placeholder: share of the draw for new or barely seen tasks

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
    max_batch_bytes: int = 64 * 1024 * 1024
    max_prompts: int = 250
    max_rollouts: int = 2000
    min_reasoning_tokens: int = 32
    reasoning_token_share: float = 0.5
    max_answer_only_frac: float = 0.05
    max_answer_repeat_frac: float = 0.10
    max_answer_repeat_count: int = 32
    reward_tolerance: float = 0.0
    min_pool_prompt_frac: float = 0.0  # 0 = prompts need not come from the public pool

    # Queue and leases.
    lease_timeout_s: int = 4 * 3600  # placeholder for the measured round wall clock
    max_infra_attempts: int = 3  # no_decision outcomes before an entry is held for the operator
    squat_min_wait_s: int = 86400
    squat_age_factor: float = 2.0
    apply_retry_s: int = 300
    apply_max_attempts: int = 12

    # Payout.
    kings_window: int = 10
    burn_uid: int = 0
    weights_cadence_s: int = 3600
    weights_rate_limit_blocks: int = 100

    budget_usd_per_hour: float = 200.0

    def __post_init__(self) -> None:
        _need(1 <= self.batch_cap <= self.max_loras_per_box, "batch_cap must be between 1 and max_loras_per_box")
        _need(1 <= self.n_min_start <= self.batch_cap, "n_min_start must be between 1 and batch_cap")
        _need(self.n_gpus >= 1 and self.n_tasks >= 1 and self.samples_per_task >= 1, "n_gpus, n_tasks, samples_per_task >= 1")
        _need(bool(self.looks) and self.looks[-1] == 1.0, "looks must end at 1.0")
        _need(all(0 < a < b for a, b in zip(self.looks, self.looks[1:])) and self.looks[0] > 0, "looks must rise from above 0")
        _need(math.isfinite(self.z_bar) and self.z_bar >= 0, "z_bar must be zero or more")
        _need(0.0 < self.drop_alpha < 1.0, "drop_alpha must be between 0 and 1")
        _need(_unit(self.cap_hit_floor) and _unit(self.cap_hit_margin), "cap-hit settings must be in [0, 1]")
        _need(_unit(self.min_cover) and self.min_cover > 0, "min_cover must be in (0, 1]")
        _need(self.min_scored_tasks >= 2, "min_scored_tasks must be at least 2: one task has no standard error")
        low, high = self.irt_middle_band
        _need(_unit(low) and _unit(high) and low < high, "irt_middle_band must be two rising shares")
        _need(_unit(self.irt_explore_frac), "irt_explore_frac must be in [0, 1]")
        _need(self.fold_every_k >= 1, "fold_every_k must be at least 1")
        _need(self.max_output_tokens < self.max_context_tokens, "max_output_tokens must be below max_context_tokens")
        _need(self.lease_timeout_s > 0 and self.max_infra_attempts >= 1, "lease settings must be positive")
        _need(self.apply_retry_s > 0 and self.apply_max_attempts >= 1, "apply retry settings must be positive")
        _need(self.kings_window >= 1 and self.burn_uid >= 0, "kings_window >= 1 and burn_uid >= 0")
        _need(self.weights_cadence_s > 0, "weights_cadence_s must be positive")
        _need(math.isfinite(self.budget_usd_per_hour) and self.budget_usd_per_hour > 0, "budget must be positive")

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
