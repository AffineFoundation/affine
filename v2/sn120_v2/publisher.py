"""The publisher: llms.txt and mining.json, written together from one public snapshot.

llms.txt is the only text a miner or a miner's agent needs. mining.json is
its one machine-readable companion. Both are pure functions of the same
snapshot, and both carry the same ``snapshot_digest``, so a reader can tell
that the two files belong together.

Every number in the text comes from the ``Params`` in the snapshot. The
miner-settable bounds come from ``Recipe.miner_bounds`` and the object keys
from ``layout.py``. No cap, bound or key is typed here. The one exception is
``READ_LINKS``: the site pull job owns that file, and no module names it yet.
``REBUILD_COST`` is measured, and it shows only when the box keeps the folded
weight files (``head_chain`` in the snapshot).

``python -m sn120_v2.publisher check`` reads a published pair and fails when
llms.txt is not true to the params of mining.json (``check_pair``; the
operator's steps are in LLMS_CHECK.md).
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import inspect
import json
import math
import re
import shlex
import sys
import textwrap
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import layout
from .admission import CONTROL_TOKENS, FAULT_HOLD_S, FAULT_TRIES, FORMATS, MIN_ROLLOUTS, NO_SIGNAL, RULES, TIME_LIMIT, TRAIN_COST, TRANSIENT
from .canonical import digest, sha256_hex, to_jsonable
from .envs.math_pool import ENV_ID, MANDATE, POOL_FILE, POOL_SHA256, POOL_V2_0_SHA256, Task, load_pool
from .errors import ConfigError, ContractError, InfraError, V2Error
from .gpu.batch import END_OF_TURN
from .gpu.evaluator import ServeSettings, chat_body, server_argv
from .gpu.trainer import LORA_FILES
from .headchain import FIXED_ORDER, MATMUL
from .irt import DRAW_RULE, MODEL, MASS_RULES, TASK_COLUMNS, draw_from_snapshot
from .journal import JOURNAL_FILE
from .judging import TIE_TOLERANCE, drop_margin
from .links import upload_entry
from .dedup import (
    STAGE_EXACT,
    STAGE_KING_EXACT,
    STAGE_KING_ROLLOUT,
    STAGE_LORA,
    STAGE_ROLLOUT,
)
from .params import (
    GENESIS_BASE_HASH,
    GENESIS_MODEL_REPO,
    GENESIS_MODEL_REVISION,
    OPTIONAL_TARGET_MODULES,
    TARGET_MODULE_NEEDS,
    MinerBound,
    Params,
    Recipe,
)
from .public import CUT_ADDRESS, CUT_LOGIN, CUT_QUERY, CUT_SECRET, NOT_JUDGED, NOT_VALID, TEXT_LIMIT, VALID
from .queue import SECONDS_PER_DAY
from .sampling import SEED_BITS
from .seams import ObjectStore
from .state import ABANDONED, load
from .types import CAP, DUPLICATE, ERROR, FAIL, NO_DECISION, NOT_RUN, NOT_SCORED, NOT_TRAINED, NULL_MODEL, NULL_UPDATE, PASS, SKIP, HeadManifest, HeadVersion
from .worker import AT_THE_LEASE_S, LocalRunner

REPO = "https://github.com/AffineFoundation/affine"
# The site pull job (``v2/site/pull.py``, ``LINKS``) writes this file on the web host, under the public address. It
# is not a store key, and the publisher never writes it: it holds signed download links for the weight files of
# the current head. The pull is a command outside this package, so the path is typed here; a test compares the two.
READ_LINKS = "public/read_links.json"
# The local train-and-score tool. The publisher names it and does not import it: the module needs the GPU install.
REPLICATE_MODULE, REPLICATE_BACKEND, REPLICATE_DOC = "sn120_v2.replicate", "backend", "v2/REPLICATE.md"
# The rebuild tool needs the GPU install too, so the publisher names it and does not import it. A test compares the names.
REBUILD_MODULE, FOLD_IMPLEMENTATION_ENV = "sn120_v2.headfold", "SN120_FOLD_IMPLEMENTATION"
# What a rebuild of the launch model costs a reader. These numbers are measured on that model, not derived from the params.
REBUILD_COST = (
    "the first run downloads 37.6 GB, and a fold needs about 115 GB of free disk and about 2 GB of memory. One fold of a "
    "rank 64 LoRA with the kernel `fixed-order-v1` takes about 21 minutes on 4 CPU cores (calculated from the measured "
    "time of one weight shape). `--device cuda:0` runs the product on a GPU; that time is not measured."
)
LORA_TABLE = "<batches of the round>"  # what llms.txt prints for --max-loras
LOCAL_OPTIONS = ("--host", "--port")  # options of the vLLM server that llms.txt does not print: the local address of the box
# Two texts of modules that the publisher may not import: the training job needs torch and the grader needs sympy,
# and the publisher runs with the standard library alone. Each is typed here; a test compares it with its source.
OPTIMIZER = "adam, fresh per LoRA, betas 0.9/0.999, eps from recipe (default 1e-8), weight decay 0"  # gpu/train_job.py
GRADER_VERSION = "math-verify-0.9.0+lenient-v1"  # envs/math_grader.py
RETIRED_KING_Z_BAR = 1.5
MERGED_EXPLANATION = (
    "Each candidate is tested as the merged model round_bf16(fp32 king master + your LoRA update), "
    "built with the same arithmetic as the install fold; the model that is tested is bit-for-bit the model that is installed. "
    "Before this change the LoRA was applied at serving time on top of the rounded bf16 king, which rounded most of a small update away."
)
# How often the box tries a training or a chunk of the scoring that fails for a reason of its own.
# The fixed cost of one update of a training (the Adam step, the sum of the gradients of the model copies, the copy of
# the weights back to them), in seconds. An estimate for a LoRA of rank 64. No run has measured it.
UPDATE_COST_S = 0.1
BOX_TRIES = inspect.signature(LocalRunner.__init__).parameters["attempts"].default
TRAIN_FORMATS = FORMATS[1:]  # the first format may not carry "train"
# Batch content and non-public store zones. A snapshot that holds one of these is refused.
PRIVATE_KEYS = frozenset({"groups", "rollouts", "put_url", "put_headers"})
PRIVATE_ZONES = ("private/", "pending/")
OWN_KEYS = ("snapshot_digest", "contract")
# A presigned upload link signs these headers (``Presigner.presign_put``). An upload without them is refused.
UPLOAD_HEADERS = {"Content-Type": "application/octet-stream"}
ENTRY_COLUMNS = ("queue_pos", "uid", "hotkey", "batch_sha256", "status", "delta", "se", "z", "cap_rate")
# The order of ``public.snapshot``. mining.json is written with sorted keys, and llms.txt must be the same text from
# the snapshot and from the mining.json that was read back (``check_pair``). A key that is not named here comes
# after these, in sorted order.
ROUND_FIELDS = ("round_id", "head_version_id", "params_hash", "recipe_id", "round_seed", "irt_snapshot_id", "n_tasks", "opened_at", "closed_at")
ROUND_FIELDS += ("outcome", "seats", "skipped", "winner_uid", "reason", "null_pass_rate", "null_cap_rate")
QUEUE_COLUMNS = ("position", "uid", "hotkey", "registration_block", "file_valid", "front", "demoted", "held")
KING_COLUMNS = ("uid", "hotkey", "round_id", "head_version_id", "applied_at", "z")
DUPLICATE_MEMBER_COLUMNS = ("queue_pos", "uid", "hotkey", "batch_sha256", "stage", "similarity", "first_seen_unix")
DUPLICATE_STAGES = {
    STAGE_EXACT: "same batch sha256 as the keeper in this round",
    STAGE_ROLLOUT: "normalized rollout overlap with a keeper in this round",
    STAGE_KING_EXACT: "same batch sha256 as a past king in the payout window",
    STAGE_KING_ROLLOUT: "normalized rollout overlap with a past king in the payout window",
    STAGE_LORA: "LoRA update cosine with a keeper in this round",
}
PARAMS_HEADING = "Every parameter"  # the section whose table states each parameter; ``stated_params`` reads it back

RECIPE_NOTES = {
    "method": "GRPO on the LoRA weights only; the base weights are frozen",
    "steps": "passes over your batch; each pass makes `minibatches` updates",
    "minibatches": "updates in each pass; 1 = one update from all rows of your batch",
    "lora_lr": "learning rate of each update",
    "adam_eps": "epsilon of each Adam update; larger values damp small-gradient weights",
    "max_grad_norm": "gradient clip of each update; it does not cap the size of an update (the training procedure, below)",
    "global_norm_cap": "cap on the norm of the whole adapter update, for a file that sets none; none = no cap",
    "group_size": "rollouts per GRPO group; 0 = the groups of your file",
    "sample_temperature": "temperature your rollouts were sampled at",
    "rank": "LoRA rank",
    "alpha": "LoRA scale",
    "dropout": "LoRA dropout",
    "target_modules": "layers that get an adapter",
    "ppo_clip": "the probability ratio (trained policy / head) of a token is clipped to 1 +- this value",
    "dual_clip": "a rollout with a negative advantage: its loss per token is at most this many times its advantage",
    "kl": "KL term; 0 = off",
    "support_top_k": "the policy-gradient term uses only tokens the head could sample (top-k, then top-p)",
    "support_top_p": "top-p of that support",
    "derail_window": "a rollout that hit the cap, with a negative advantage: window of the derail test, in tokens",
    "derail_threshold": "the derail point is the first token where the mean log-probability of the window is below this",
    "derail_tail": "tokens later than this many after the derail point do not train",
    "train_max_output_tokens": "training reads at most this many output tokens of a rollout; it is not the scoring cap",
    "train_max_context_tokens": "training reads at most this many prompt + output tokens of a rollout",
    "code_digest": "sha256 of the training code",
    "image_digests": "worker images that may train and score",
}


def show(value: Any) -> str:
    """One value as text for a miner."""
    if value is None or value == "" or value == [] or value == ():
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(show(item) for item in value)
    return str(value)


def duration(seconds: float) -> str:
    """A time for a miner: in days when it is a whole number of days, in seconds when it is not."""
    days, rest = divmod(seconds, SECONDS_PER_DAY)
    if days >= 1 and rest == 0:
        return f"{int(days)} day" if days == 1 else f"{int(days)} days"
    return f"{int(seconds) if seconds == int(seconds) else seconds} s"


def utc(seconds: Any) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(seconds)) if isinstance(seconds, int) and seconds > 0 else "unknown"


def table(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    return lines + ["| " + " | ".join(show(cell) for cell in row) + " |" for row in rows]


def in_order(keys: Iterable[str], first: Sequence[str]) -> list[str]:
    """The keys of a row in the order of ``first``, then every other key in sorted order."""
    found = set(keys)
    return [key for key in first if key in found] + sorted(found - set(first))


def rows_table(rows: Sequence[Mapping[str, Any]], first: Sequence[str]) -> list[str]:
    columns = in_order(rows[0], first)
    return table(columns, ([row.get(column) for column in columns] for row in rows))


def reflow(lines: Iterable[str], width: int = 112) -> list[str]:
    """Join each paragraph and each list item, then wrap it. Headings, tables and code blocks stay as they are."""
    out: list[str] = []
    block: list[str] = []
    fenced = False
    for line in [*lines, ""]:
        fixed = fenced or not line or line.startswith(("|", "#", "```"))
        fenced ^= line.startswith("```")
        if block and (fixed or line.startswith("- ")):
            indent = "  " if block[0].startswith("- ") else ""
            # A lone operator stays on the line of the word before it: a line that starts with "- " reads as a new item.
            text = re.sub(r" (?=[-+*/=<>x] )", "\xa0", " ".join(part.strip() for part in block))
            wrapped = textwrap.wrap(text, width, subsequent_indent=indent, break_long_words=False, break_on_hyphens=False)
            out += [part.replace("\xa0", " ") for part in wrapped]
            block = []
        (out if fixed else block).append(line)
    return out[:-1]


def bound_text(bound: MinerBound) -> str:
    """The allowed values of one miner-settable field, in words."""
    if bound.choices:
        needs = [f"; a list with {name} must hold {needed} too" for name, needed in TARGET_MODULE_NEEDS if name in bound.choices]
        return "a list: a non-empty subset of " + ", ".join(bound.choices) + "".join(needs)
    kind = "a whole number" if bound.integer else "a number"
    return f"{kind} from {bound.low:g} to {bound.high:g}, both ends included"


def miner_settable(recipe: Recipe) -> list[dict[str, Any]]:
    """One row per field a miner may set under ``train``. ``Recipe.miner_bounds`` is the only source."""
    return [
        bound.to_dict() | {"default": to_jsonable(getattr(recipe, bound.key)), "allowed": bound_text(bound)}
        for bound in recipe.miner_bounds
    ]


def object_keys() -> dict[str, str]:
    """Where each public object is. UPPER-CASE parts are ids to fill in."""
    return {
        "llms_txt": layout.LLMS_TXT,
        "mining_json": layout.MINING_JSON,
        "current_head": layout.CURRENT,
        "head_manifest": layout.head_manifest("VERSION_ID"),
        "base_file": layout.public_base_prefix("BASE_HASH") + "/FILE",
        "master_file": layout.public_master_prefix("MASTER_DIGEST") + "/FILE",
        "lora_file": layout.public_lora_prefix("LORA_HASH") + "/FILE",
        "round_file": layout.round_file("ROUND_ID"),
        "winner_batch": layout.winner_batch("ROUND_ID"),
        "irt_snapshot": layout.irt_snapshot("SNAPSHOT_ID"),
    }


def contract(params: Params, public_url: str, links_url: str = "") -> dict[str, Any]:
    """The fixed rules in machine-readable form. Everything is derived from ``params`` and the two public addresses."""
    recipe = params.recipe
    doc = {
        "params_hash": params.params_hash,
        "recipe_id": params.recipe_id,
        "miner_settable": miner_settable(recipe),
        "batch_file": {
            "versions": list(FORMATS),
            "train_allowed_in": list(TRAIN_FORMATS),
            "forbidden_text": list(CONTROL_TOKENS),
            "private_object": layout.miner_object("HOTKEY"),
            "upload_headers": UPLOAD_HEADERS,
        },
        "duplicate_policy": {
            "enabled": params.dedup_enabled,
            "stage_a": {
                "exact_batch_sha256": True,
                "rollout_overlap_min": params.dedup_rollout_overlap,
                "comparison": "intersection / min(size of each distinct normalized (prompt, output) set)",
                "normalize": "strip and collapse whitespace; ignore rewards",
                "past_kings": "current payout window",
            },
            "stage_b": {
                "metric": "cosine of scaled LoRA delta-W, computed from A and B without materializing delta-W",
                "shared_verified_train_seed_min": params.dedup_cos_shared_seed,
                "own_or_mixed_train_seed_min": params.dedup_cos_own_seed,
            },
            "priority": ["first_seen_unix", "queue_pos"],
            "dropped_status": DUPLICATE,
            "null_update_status": NULL_UPDATE,
            "min_signal_rollouts": "min_signal_rollouts; no implicit effective recipe.minibatches floor",
            "min_step_norm": params.min_step_norm,
            "min_step_norm_rule": (
                "static global_norm_cap and stored step_norm must be at least min_step_norm"
                if params.min_step_norm > 0 else "no minimum norm; only a non-finite stored step_norm is null_update"
            ),
            "public_cluster_member_fields": list(DUPLICATE_MEMBER_COLUMNS),
            "stages": DUPLICATE_STAGES,
        },
        "outcome_codes": {"pass": PASS, "fail": FAIL, "token_cap": CAP, "infra_error": ERROR, "not_run": NOT_RUN},
        "null_model": NULL_MODEL,
        "public_url": public_url,
        "links_url": links_url,
        "upload": upload_entry(links_url) if links_url else None,
        "objects": object_keys() | ({"active_pool": layout.active_pool("ACTIVE_SHA"), "pool_rotation": layout.pool_rotation("ROUND_ID"),
            "retired_pool": layout.RETIRED_POOL, "reserve_manifest": layout.RESERVE_MANIFEST} if params.rotation_enabled else {}) | ({"pool_ranking": layout.pool_ranking("ROUND_ID"), "pool_summary": layout.POOL_SUMMARY} if params.rotation_rule == "info" else {}),
        "read_links": READ_LINKS,
        "code": REPO + "/tree/v2/v2",
        **({"pool_rotation_doc": REPO + "/blob/v2/v2/POOL_ROTATION.md"} if params.rotation_enabled else {}),
    }
    if params.eval_merged:
        doc["candidate_evaluation"] = {
            "merged": True,
            "model": "round_bf16(fp32 king master + your LoRA update)",
            "tested_equals_installed": True,
            "arithmetic": "the install fold kernel named in the head chain; the final base manifest hash is checked before install",
            "explanation": MERGED_EXPLANATION,
        }
    return doc


def refuse_private(value: Any, where: str = "snapshot") -> None:
    """Fail closed on batch content, upload links and keys outside the public zone."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in PRIVATE_KEYS:
                raise ContractError(f"{where}.{key}: batch content and upload links are never published")
            refuse_private(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            refuse_private(item, f"{where}[{position}]")
    elif isinstance(value, str) and value.startswith(PRIVATE_ZONES):
        raise ContractError(f"{where}: {value.split('/')[0]}/ objects are never published")


def mining_document(snapshot: Mapping[str, Any], public_url: str = "", links_url: str = "") -> dict[str, Any]:
    """mining.json: the snapshot of ``public.snapshot`` itself, plus its digest and the contract block."""
    clash = [key for key in OWN_KEYS if key in snapshot]
    if clash:
        raise ContractError(f"the snapshot may not use the publisher's keys {clash}")
    refuse_private(snapshot)
    plain = to_jsonable(snapshot)
    params = Params.from_dict(plain.get("params") or {})
    if not plain.get("head"):
        raise ContractError("the snapshot has no head")
    HeadManifest.from_dict(plain["head"])  # a head whose ids do not match its content is refused
    return {"snapshot_digest": digest(plain), **plain, "contract": contract(params, public_url, links_url)}


def _recipe_lines(params: Params) -> list[str]:
    recipe, recipe_id = params.recipe, params.recipe_id
    cap, cap_bound = recipe.global_norm_cap, recipe.bound("global_norm_cap")
    cap_now = "This recipe has none: there is no cap." if cap is None else f"It is {show(cap)} now."
    cap_yours = (
        f"`train.global_norm_cap` sets the cap of your LoRA, from {show(cap_bound.low)} to {show(cap_bound.high)}."
        if cap_bound is not None
        else "A file may not set a cap of its own now."
    )
    if cap_bound is not None:
        if params.min_step_norm > 0:
            cap_yours += (
                f" Values below {show(params.min_step_norm)} are invalid (`{NULL_UPDATE}`, section 5) even when the generic lower"
                " bound is smaller."
            )
        else:
            cap_yours += " There is no minimum norm, so every valid nonnegative cap passes this static floor check."
    fields = [f.name for f in dataclasses.fields(recipe) if f.name != "miner_bounds"]
    rows = [(f"`{name}`", getattr(recipe, name), RECIPE_NOTES.get(name, "see the training code")) for name in fields]
    settable = [(f"`train.{row['key']}`", row["default"], row["allowed"]) for row in miner_settable(recipe)]
    targets = recipe.bound("target_modules")
    offered = [name for name in OPTIONAL_TARGET_MODULES if targets is not None and name in targets.choices]
    optional_targets = [
        f"- `target_modules`: the default is {show(recipe.target_modules)}. You may also name {show([f'`{name}`' for name in offered])}:",
        "  projections of the linear-attention layers of the head. The trainer handles each of them as it handles the",
        "  other names: every linear layer with that name gets an adapter (the training procedure, below). `in_proj_a`",
        "  and `in_proj_b` are not choices.",
    ]
    for name, needed in TARGET_MODULE_NEEDS:
        if name in offered:
            optional_targets += [
                f"- A list with `{name}` must hold `{needed}` too. The server that scores your LoRA holds these two layers as",
                f"  one layer, and it does not load an adapter for `{name}` alone. A list with `{name}` and no `{needed}`",
                f"  makes the whole file invalid. `{needed}` without `{name}` is allowed.",
            ]
    pinned = [
        "- `code_digest` is the id of the file set `sn120_v2/gpu/*.py` (section 9). The box refuses to train or score",
        "  when its own files have another id. `python -m sn120_v2.gpu.plan` prints the `code_digest` of your copy of",
        "  the code.",
    ] if recipe.code_digest else []
    return [
        f"## 2. The training recipe (recipe_id `{recipe_id}`)",
        "",
        "The validator trains one LoRA from head M for each admitted batch. It runs the same code and the same recipe",
        "for every batch. Your batch is the only training data. A batch made against an older head is trained on the",
        "current head. `recipe_id` is the sha256 of the canonical JSON (sorted keys, no spaces) of the recipe object in",
        "mining.json (`params.recipe`). A new recipe gets a new id, and it is shown here before the next round starts.",
        "The object does not hold `minibatches` while the value is 1, or `adam_eps` while it is the default 1e-8.",
        "",
        *table(("field", "value", "meaning"), rows),
        "",
        f"- Training code: `sn120_v2/gpu/` in {REPO}/tree/v2/v2 (sha256 `{show(recipe.code_digest)}`).",
        *pinned,
        "- Each prompt is one user message, rendered with the chat template of the head and the generation prompt.",
        "",
        "### What you may set",
        "",
        "You may set only these fields, in the `train` object of your file. Each one replaces the recipe value above",
        "for your LoRA only.",
        "",
        *table(("key", "default", "allowed"), settable),
        "",
        "- These are the bounds for launch. The bounds are part of the recipe: a change of a bound gives a new",
        "  `recipe_id`.",
        "- Every key is optional. A key that you do not set keeps its default.",
        "- Values are never clamped. A value outside its bound, a value of the wrong type, or any other key under",
        "  `train` makes the whole file invalid.",
        f"- `train` is allowed in {show(TRAIN_FORMATS)}. It is not allowed in {FORMATS[0]}.",
        "- Your `train` values are published only if your batch wins.",
        "- A file with `train` keys that only v1 had (`lr`, `kl`, `step_mult`, the knobs) is invalid. The seat is",
        "  skipped and not charged, as with any invalid file (section 4), until the miner uploads a valid file.",
        "",
        "### What the trainer does with them",
        "",
        "- `group_size`: 0 means one group per prompt, with all rollouts of the prompt as the file has them. A value G",
        "  above 0 cuts the rollouts of each prompt, in file order, into groups of G; a last group with fewer than G",
        "  rollouts stays a group. The advantage of a rollout is its reward minus the mean reward of its group. The",
        f"  default is {recipe.group_size}.",
        *(optional_targets if offered else []),
        "- `sample_temperature` T: you state the temperature at which you sampled your rollouts. The trainer divides the",
        "  logits of the head by T before it scores the tokens of a rollout. Evaluation always samples at temperature",
        f"  {show(params.temperature)}. The default T is {show(recipe.sample_temperature)}.",
        "- `global_norm_cap`: the largest allowed Frobenius norm of the whole adapter update,",
        "  sqrt(sum over all target modules of ||(alpha / rank) * B @ A||_F^2), in the units of the model weights. If the",
        "  trained update is larger, every B matrix is multiplied by cap / norm. The train report states the norm before",
        "  the cap and the norm after it (`step_norm` is the norm after the cap). The recipe value is the cap of a file",
        f"  that sets none. {cap_now} {cap_yours} The cap holds for the whole",
        "  update, however many steps and minibatches made it.",
        "- `adam_eps`: the epsilon of each Adam update. With one Adam step, a larger epsilon damps weights with small",
        "  gradients, so the step concentrates on high-signal weights. Use this instead of the old zero-reward filler",
        "  trick: zero-advantage rows do not train or enter the loss denominator, so adding them has no effect.",
        "- `minibatches`: the count of updates in each pass over your rows. The training procedure, below, has the",
        "  rule. Section 5 has what it costs.",
        f"- Train caps: a rollout above `train_max_output_tokens` ({recipe.train_max_output_tokens}) or",
        f"  `train_max_context_tokens` ({recipe.train_max_context_tokens}, prompt + output) is truncated, not rejected. The",
        "  train report counts the truncated rollouts. A rollout with no output token left is dropped and counted.",
        "",
        *_training_lines(params),
    ]


def _training_lines(params: Params) -> list[str]:
    """The training of one LoRA, step by step.

    Every word here follows ``gpu/trainer.py`` (the checks around the job), ``gpu/train_job.py`` (the job) and
    ``gpu/batch.py`` (rows, advantages, token caps). When one of them changes, change this function.
    """
    recipe = params.recipe
    seed_input = "<round_seed>|train" if params.train_seed_scope == "round" else "<round_seed>|train|<batch_sha256>"
    seed_repeat = (
        "every LoRA in the round starts from the same A. The shuffle stream has the same seed, but different rows give different minibatch orders"
        if params.train_seed_scope == "round"
        else "the same file with the same round seed gets the same A and the same minibatch orders"
    )
    return [
        "### The training procedure",
        "",
        "This is everything that the trainer does with a file, in the order in which it does it. The code is in three",
        "files: `sn120_v2/gpu/trainer.py` (the checks around the job), `sn120_v2/gpu/train_job.py` (the job on the GPUs)",
        "and `sn120_v2/gpu/batch.py` (rows, advantages and token caps). Section 10 has the command that runs this code",
        "on your machine.",
        "",
        f"- Model: the base files of head M (`base_hash`, section 1), loaded as `{params.dtype}` in the model class that",
        "  their `config.json` names. The model is in evaluation mode and all its weights are frozen. The box trains",
        "  and scores only a head with no stacked LoRA: it refuses a head whose LoRAs are not folded.",
        "- Recipe: the values of the table above, with your `train` values in their place. The trainer runs only the",
        "  method `grpo-lora`, with no dropout and with the KL term off.",
        f'- Seed (`train_seed_scope` is `{params.train_seed_scope}`): the first {SEED_BITS} bits of sha256("{seed_input}"), as a whole number.',
        "  The train report states it. With `round` scope every batch in the round shares one training seed; legacy `batch` scope adds",
        "  `|<batch_sha256>` after `|train`, so different files have different training seeds.",
        f"- LoRA layers: each linear layer whose name is one of `target_modules` ({show(recipe.target_modules)}) gets two",
        "  matrices: A (rank x inputs) and B (outputs x rank). The layer then gives layer(x) + (alpha / rank) * B(A(x)),",
        f"  with rank {recipe.rank} and alpha {show(recipe.alpha)}. A starts with values drawn uniformly from -1/sqrt(inputs) to",
        "  +1/sqrt(inputs), from a `torch.Generator` on the CPU with the seed above, layer by layer in the order of",
        "  `named_modules()`. B starts at zero, so the model is head M before the first update. A and B are float32,",
        "  and only A and B train.",
        "- Rows: one row for each rollout of the groups that the file check keeps (section 5), in file order. The",
        "  advantage of a row is its reward minus the mean reward of its group (`group_size`, above). The trainer does",
        "  nothing else to it: no division by a standard deviation and no clip.",
        '- Tokens of a row. Prompt: `tokenizer.apply_chat_template([{"role": "user", "content": prompt}],',
        "  add_generation_prompt=True, tokenize=False)`, then `tokenizer.encode(text, add_special_tokens=False)`. The",
        "  template gets no other argument. Output: `tokenizer.encode(output, add_special_tokens=False)`. The tokenizer",
        "  is the one of the base files.",
        "- Room: room = min(`train_max_output_tokens`, `train_max_context_tokens` - prompt tokens). An output shorter",
        f"  than the room gets the end-of-turn token `{END_OF_TURN}` after its last token. An output as long as the room or",
        "  longer is cut to the room and gets no end token: the trainer takes it as a rollout that did not end (a capped",
        "  row). A row with no output token or with no room is dropped.",
        f"- Rows that train: the rows whose advantage is not zero. With no such row the batch is not trained: `{SKIP}`",
        "  (section 4).",
        "- The score of a token. The trainer takes the logits of the model at each output token, as float32. The",
        f"  support at a token: take the {recipe.support_top_k} tokens with the highest logits (`support_top_k`), divide their logits",
        "  by T (`sample_temperature`) and take the softmax over them. From the most likely token down, a token is in",
        f"  the support while the summed probability of the tokens before it is below {show(recipe.support_top_p)} (`support_top_p`). log q",
        "  of the token of your rollout = its logit / T - log(sum of exp(logit / T) over the support).",
        "- First pass, before any update: the trainer stores log q of every output token under head M. This is the old",
        "  policy. The mask of a row holds its tokens that are in the support of head M. A token outside the mask",
        "  never trains. With more than one minibatch this is a pass of its own over all rows that train, before the",
        "  first update, with no loss and no gradient: the old policy of every row is head M, not a model that has",
        "  moved.",
        "- Derail rule, only for a capped row with a negative advantage. Take the plain log-probability of each output",
        "  token under head M (all logits, no temperature). The derail point is the first token at which the mean of",
        f"  the last {recipe.derail_window} such values (`derail_window`), that token included, is below {show(recipe.derail_threshold)} (`derail_threshold`).",
        f"  The derail point and the {recipe.derail_tail} tokens after it (`derail_tail`) stay in the mask. Each later token leaves",
        "  the mask. A row with fewer tokens than the window has no derail point.",
        "- Loss, at each step, for a token of a row with advantage A: ratio = exp(log q now - log q of the first",
        "  pass). \"Now\" is the model with its LoRA at that step, with the support of that model. Token loss =",
        f"  max(-A * ratio, -A * clip(ratio, 1 - `ppo_clip`, 1 + `ppo_clip`)), with `ppo_clip` {show(recipe.ppo_clip)}. With A below 0",
        f"  the token loss is at most -A * `dual_clip`, with `dual_clip` {show(recipe.dual_clip)}. A token counts when it is in the mask of",
        "  its row and in the support of the model now. Row loss = the sum of its token losses / the number of tokens",
        "  in its mask. A row with an empty mask gives no loss. With one minibatch, the loss of an update = the sum of",
        "  the row losses / the number of rows that train. Rows with advantage 0 do not count. With more than one",
        "  minibatch, the loss of an update = the sum of the row losses of its minibatch /",
        "  the number of rows in that minibatch.",
        f"- Steps: {recipe.steps} (`steps`). Minibatches: {recipe.minibatches} (`minibatches`). One step is one pass over all rows that train.",
        "  With one minibatch a pass ends with one update, from all rows. The gradient of the loss of an update, over all A",
        f"  and B together, is clipped to the norm {show(recipe.max_grad_norm)} (`max_grad_norm`). Then comes the update, with the learning",
        f"  rate {show(recipe.lora_lr)} (`lora_lr`) and epsilon {show(recipe.adam_eps)} (`adam_eps`). The optimizer: {OPTIMIZER}. The learning rate has no",
        f"  schedule. A gradient that is not finite stops the training: `{SKIP}`.",
        "- The clip does not cap the size of an update. Adam divides each gradient value by its own running size. So",
        "  with the default epsilon the first update moves each weight that has a gradient by about `lora_lr`, whatever",
        "  the norm of the gradient",
        "  is. The size of the whole update grows with `lora_lr`, with the count of updates and with the count of",
        "  weights. The one cap on it is `global_norm_cap`.",
        "- Minibatches, for a value N above 1. Each pass cuts the rows that train into N minibatches and makes one",
        "  update for each minibatch: steps x N updates in all. If fewer than N rows train, each row is a minibatch",
        "  of its own, and a pass makes one update for each row. The order comes from one `random.Random(seed)` of",
        "  Python for the whole training, with the seed above. Each pass takes the rows that train, the longest row",
        "  first and rows of the same length in file order, and calls `shuffle` on a copy of that list. Row k of the",
        "  shuffled list (k from 0) goes to minibatch k modulo N. The minibatches train in the order 0 to N - 1. So each",
        f"  pass has another order, and {seed_repeat}. The optimizer is made one time for your file: its state goes from each",
        "  update to the next, through all passes.",
        "  The LoRA of another file starts with a new optimizer.",
        "- GPUs: the job has the GPUs of its slot (section 3). It puts one copy of the model on each GPU, but not more",
        "  copies than rows that train. The rows that train go to the copies in turn, the longest row first. The",
        "  gradients of the copies are added before the clip. With more than one minibatch, the rows of each minibatch",
        "  go to the copies in turn in the same way. The job never trains on a CPU: with no GPU it stops.",
        "- After the last step: the cap `global_norm_cap` (above), then a check that each weight is finite. If one is",
        f"  not: `{SKIP}`.",
        f"- The LoRA is these files: {show([f'`{name}`' for name in LORA_FILES])}. The weight file holds A and B of each layer as",
        "  float32, with the names `base_model.model.<layer path>.lora_A.weight` and",
        "  `base_model.model.<layer path>.lora_B.weight`. The config file has the PEFT layout; its",
        "  `base_model_name_or_path` is the `base_hash`. The train report holds the optimizer, the seed, the recipe",
        "  values that were used, the loss and the gradient norm of each update, the count of the mask tokens of the",
        "  rows that trained, the count of the rows, of the truncated rows and of the dropped rows, `step_norm` and",
        "  `step_norm_before_cap`. With more than one minibatch the recipe values hold `minibatches` too. `lora_hash` is the",
        "  id of this file set (section 9).",
        f"- A failure of the box in a training (not of the batch) is tried again: at most {BOX_TRIES} tries in all, inside the",
        f"  training time. Then the entry is `{NO_DECISION}` (section 4).",
        f"- What a training of your own repeats: with the same round seed, {seed_repeat}. The trained B can differ",
        "  between two machines, and between two runs on one machine; no run has measured by how much. The tests of",
        "  the code run this training on a CPU, on a very small model. The box runs it on GPUs.",
    ]


def _round_time_lines(params: Params) -> list[str]:
    """What the clock does to a round. Every word here follows ``worker.LocalRunner.run``.

    When the rule of the time limit changes, change this function. Three sentences elsewhere depend on it: the
    ``not_trained`` and ``not_scored`` bullets of section 4 and the "time limit" bullet of section 6. They refer to
    this section.
    """
    # The two times of worker.LocalRunner.run, in seconds from the lease.
    stop = min(params.lease_timeout_s, params.round_time_limit_s) - params.promote_reserve_s
    train_end = min(stop, params.train_time_budget_s)
    limit, reserve, first_look = show(params.round_time_limit_s / 60), show(params.promote_reserve_s / 60), params.look_sizes()[0]
    return [
        "### The time of a round",
        "",
        "All times count from the lease: the moment a round takes its entries. A round ends early, as soon as every",
        "LoRA is dropped or decided.",
        "",
        f"- Time limit: {limit} minutes from the lease to the promotion of the winner. The box stops scoring {reserve} minutes",
        f"  before it, at minute {show(stop / 60)} (the stop time). The stop is hard. The {reserve} minutes are for the grading of the last",
        "  chunk, the judging and the promotion.",
        f"- Training time: the first {show(train_end / 60)} minutes. Batches train longest-expected-first, with queue order breaking ties.",
        "  Each batch trains on one GPU, up to one training job per GPU at a time; any remaining batch waits for a free GPU.",
        "- At the end of the training time no new training starts, and each training that still runs is stopped. Such",
        f"  a batch is `{NOT_TRAINED}`: you are not charged, your seat is first in the next round, and no failure is counted",
        "  for the seat. Each GPU starts scoring when its own training is done or stopped; other GPUs may still train.",
        "- One exception: a batch that started at the lease had the whole training time. If the first try of its",
        f"  training did not finish in that time, that is a failure of ours: `{NO_DECISION}`. Section 4 says when such a",
        "  failure is counted for the seat. If a try of its training failed before (a failure of the box) and a later try",
        f"  is stopped at the end of the training time, the batch is `{NOT_TRAINED}` as above: the box took a part of its time.",
        f"- The box can start its work on a round later than {AT_THE_LEASE_S} s after the start of the lease, for example when the",
        "  validator's program started again in the round. Then no batch had the whole training time: a batch that",
        f"  does not finish is `{NOT_TRAINED}` as above, and no failure is counted for the seat.",
        f"- Scoring streams over the draw with no chunk barrier. Null M and every open LoRA answer every task. An open LoRA is a",
        "  LoRA that was not dropped at a look (section 6).",
        "- The stop time cuts every look, the first one too. No rollout may run past the stop time. The common prefix scored",
        f"  by then is truncated to a whole block of {params.eval_chunk_tasks} tasks; an unfinished block is not used. There are no further looks.",
        f"- At the stop time, with the first look ({first_look} tasks) complete: each open LoRA is judged by the win rule of",
        "  section 6 on the complete cutoff blocks that null M and every open LoRA finished. Those tasks are then the whole",
        f"  test: z strictly above {show(params.z_bar)} can win. Such an entry is charged, win or lose, as in a full round.",
        f"- At the stop time, with fewer than {first_look} tasks scored: each trained LoRA is `{NOT_SCORED}`. You are not charged,",
        "  your seat is first in the next round, and no failure is counted for the seat.",
        f"- The lease ends {show(params.lease_timeout_s / 60)} minutes after it starts. A box result that arrives after the end of the lease is",
        "  never used: the round then has no result (section 4). No task is scored after the stop time: the time between",
        "  the stop time and the end of the lease is only a reserve, so that a round that stopped in time does not lose",
        "  its result.",
        f"- `{NO_DECISION}` as the status of an entry means: not charged, and the seat goes to the front of the queue",
        "  (section 4).",
        "",
    ]


def _cluster_lines(record: Any, label: str) -> list[str]:
    clusters = record.get("duplicate_clusters") or [] if isinstance(record, Mapping) else []
    if not clusters:
        return []
    lines = [f"{label} duplicate clusters (public evidence only; timestamps are UTC):", ""]
    for cluster in clusters:
        keeper, dropped = cluster["keeper"], cluster.get("dropped") or []
        lines += [f"Keeper for {len(dropped)} dropped cop{'y' if len(dropped) == 1 else 'ies'}:", ""]
        lines += rows_table([keeper, *dropped], DUPLICATE_MEMBER_COLUMNS)
        lines += ["", "Dropped-copy links (a past-king keeper has the sentinel queue position documented in section 5):", ""]
        lines += table(("dropped queue_pos", "keeper_queue_pos"), ((row["queue_pos"], row["keeper_queue_pos"]) for row in dropped))
        lines.append("")
    return lines


def _duplicate_lines(params: Params) -> list[str]:
    enabled = "on" if params.dedup_enabled else "off"
    settable_keys = {row["key"] for row in miner_settable(params.recipe)}
    signal_recipe = "`minibatches`" if "minibatches" in settable_keys else "minibatches"
    static_cap = "`train.global_norm_cap`" if "global_norm_cap" in settable_keys else "global_norm_cap"
    signal = [
        f"- Minimum signal: after `{NO_SIGNAL}` rejects a file in which nothing trains, `{MIN_ROLLOUTS}` rejects a file with fewer",
        f"  than {params.min_signal_rollouts} rollouts with a nonzero advantage and at least one usable output token inside the training caps.",
        f"  The configured minimum signal is {params.min_signal_rollouts}; {signal_recipe} adds no admission floor. With fewer active rows than",
        "  requested minibatches, training makes fewer nonempty minibatches rather than rejecting the file. Without tokenizer checks no rollout",
        "  is known to be dropped by the training caps. This admission rejection is invalid and uncharged: the seat keeps its place, and no",
        "  training GPU is used. Upload a new file with enough signal rather than equal-reward filler, whose zero-advantage rows do not count.",
    ]
    static_norm = [
        f"- Static null-update admission: `{NULL_UPDATE}` rejects {static_cap} below {show(params.min_step_norm)} (`min_step_norm`)",
        "  before any training GPU work. The file is invalid, uncharged and keeps its seat. A cap equal to the floor passes",
        "  this static test, but does not guarantee that training reaches it.",
    ] if params.min_step_norm > 0 else []
    stored_norm = [
        f"- Stored null update: after training, a stored `step_norm` below {show(params.min_step_norm)}, or one that is not finite, is terminal",
        f"  `{NULL_UPDATE}`. It is charged, cannot win or be paid, and is not a candidate or keeper in Stage B. The floor applies to",
        "  the stored norm after any global cap, not to the uncapped gradient or to `step_norm_before_cap`.",
    ] if params.min_step_norm > 0 else [
        f"- Stored null update: after training, a stored `step_norm` that is not finite is terminal `{NULL_UPDATE}`. It is charged,",
        "  cannot win or be paid, and is not a candidate or keeper in Stage B. There is no minimum norm.",
    ]
    subfloor = [
        "- Why a sub-floor update cannot beat M except by luck: the trainer's first-order step changes the head by at most",
        "  its small stored norm. The scoring noise and a favorable task sample can then produce a positive result even",
        "  when the update carries no reproducible improvement. Without the floor, many tiny steps could win by chance",
        "  and turn the payout into a lottery. The floor makes a charged entry show a material model change; it is not a",
        "  guarantee that a larger update improves M.",
    ] if params.min_step_norm > 0 else []
    return [
        "### Duplicate and null batches",
        "",
        "This section covers both the file-check rules and the terminal outcomes `duplicate` and `null_update`. A",
        "frozen file can also fail another admission rule first; the rule table above gives that order.",
        "",
        *signal,
        *static_norm,
        *stored_norm,
        f"- Duplicate filtering is {enabled} (`dedup_enabled`). Stage A runs immediately after the files are frozen and before the",
        "  entropy read, training tickets or GPU work. Exact equal `batch_sha256` is a duplicate even when rollout evidence",
        "  is unavailable. Otherwise each rollout becomes a (prompt, output) pair: strip its ends, collapse each whitespace",
        "  run to one space, and ignore rewards. Compare the distinct normalized sets by intersection size / the size of",
        f"  the smaller set. Overlap >= {show(params.dedup_rollout_overlap)} (`dedup_rollout_overlap`) is a duplicate; exact equality is not required.",
        "- Stage A compares with earlier batches in this round and with the exact hashes and available normalized sets of",
        f"  the past winners in the current payout window (the last {params.kings_window} distinct king UIDs, section 7). A past winner always",
        "  keeps its position against a later copy. Reusing a past winner's rollouts counts as a duplicate even from the same",
        "  hotkey: a king resubmitting >= dedup_rollout_overlap of its own already-applied rollouts is dropped as",
        "  `past_king_rollout`, or `past_king_exact` for the same batch bytes. Within the round, greedy priority is (first_seen,",
        "  queue_pos): the first `first_seen_unix` recorded for valid batch bytes in UTC, then `queue_pos` to break ties. Files",
        "  already waiting",
        "  when this ledger starts all receive its first post-deployment check time, so queue position breaks their tie.",
        "  Once kept, that entry remains the keeper for later comparisons; dropped entries never become comparison anchors.",
        f"  A half-and-half mixture ({show(params.dedup_rollout_overlap)} of each) of two kept honest batches is itself the only copy;",
        "  it cannot drop either part through a dropped mixture. Priority is not a proof of who generated rollouts first.",
        "- Stage B runs on trained, verified LoRA artifacts after Stage A and the null-update exclusion. It compares the",
        "  cosine of the whole scaled LoRA delta-W across target modules, computed from A and B without materializing",
        f"  delta-W. With the same verified training seed, cosine >= {show(params.dedup_cos_shared_seed)} (`dedup_cos_shared_seed`) is a duplicate.",
        f"  With own or mixed seeds, the threshold is {show(params.dedup_cos_own_seed)} (`dedup_cos_own_seed`). The same greedy (first_seen,",
        "  queue_pos) priority applies. Missing or unusable evidence does not create a duplicate.",
        f"- A dropped copy has status `{DUPLICATE}`: it is charged, cannot win and receives no payout. This applies to a copy",
        "  found before training as well as one found by Stage B. A duplicate never displaces its keeper merely by scoring",
        "  higher. Regenerated or rewritten rollouts are new evidence, not an exemption: judge them by the same hashes,",
        "  normalized sets and update cosine. Changing whitespace or rewards alone does not avoid normalized overlap.",
        *subfloor,
        "- Public duplicate evidence never includes batch content, prompts, outputs, rewards or the normalized rollout",
        "  hashes. `duplicate_clusters` in the open/closed round objects of mining.json and in the round file has one",
        "  `keeper` and a `dropped` list for each cluster. Each member publishes `queue_pos`, `uid`, `hotkey`,",
        "  `batch_sha256`, `stage`, `similarity` and `first_seen_unix`; a dropped member also names `keeper_queue_pos`.",
        "  Past-king keepers use the past-king sentinel queue position. Stage names are " + ", ".join(f"`{name}` ({meaning})" for name, meaning in DUPLICATE_STAGES.items()) + ".",
        "  All published times are UTC; `first_seen_unix` counts seconds from the Unix epoch.",
        "",
    ]


def _state_lines(doc: Mapping[str, Any], params: Params) -> list[str]:
    round_now, queue = doc.get("round") or {}, doc.get("queue") or {}
    entries, seats = round_now.get("entries") or [], queue.get("next") or []
    checked = queue.get("files_checked_at")
    return [
        "## 3. The round now",
        "",
        f"- phase: {show(doc.get('phase'))} (IDLE: no round is open; RUNNING: training and scoring; APPLY: decided, the",
        "  new head is being published). The last closed rounds are in mining.json under `history`.",
        *(f"- {key}: {show(round_now[key])}" for key in in_order(round_now, ROUND_FIELDS) if not isinstance(round_now[key], (dict, list))),
        *(["", *table(ENTRY_COLUMNS, ([row.get(column) for column in ENTRY_COLUMNS] for row in entries)), ""] if entries else []),
        f"- A round takes at most {params.batch_cap} batches and needs at least {params.n_min_start}. It does not wait",
        "  for a full round. Registrations for the next round are always open.",
        "- Round loop: (1) pin head M, the recipe and the task sampler snapshot; (2) copy the next valid files in queue",
        "  order, freeze their bytes and run Stage-A duplicate checks, then read entropy only after those files are frozen",
        "  to derive and pin the public round seed and task draw; (3) train one LoRA per remaining file; (4) score every",
        "  LoRA and a null copy of M (no adapter), and run null-update and Stage-B duplicate checks;",
        *([
            "  null M sample 0 and every LoRA use the same tasks and seed; later king samples are king-only;",
        ] if params.null_samples_per_task is not None else [
            "  null M and every LoRA use the same tasks and seeds;",
        ]),
        "  (5) apply at most one winner; (6) publish; (7) start again. Batches cannot use advance knowledge of",
        "  the entropy, round seed or training seeds to target a particular draw or initialization.",
        "",
        *_round_time_lines(params),
        "## 4. The queue",
        "",
        f"Seats that wait: {show(queue.get('length'))}. The next seats, in order:",
        "",
        *(rows_table(seats, QUEUE_COLUMNS) if seats else ["The queue is empty."]),
        "",
        "- `file_valid`: the validator's verdict on the file that was stored for the hotkey at its last check of the",
        f"  files: {utc(checked) if checked else 'no check yet'} (`queue.files_checked_at` in mining.json). The validator reads the waiting files when it picks the entries",
        "  of a round, and again and again while no round runs. It does not read them while a round runs: a file that",
        "  you store then shows the old verdict until the round ends. The validator never publishes batch content or",
        "  `train` values. To see the rule that a file breaks, run the `check` command of section 5: it prints the",
        "  rule code and the reason.",
        "- A new process of the validator has no verdict until its first check of the files. A publication before that",
        "  check has `queue.files_checked_at` null and `file_valid` no for every waiting seat, also for a good file.",
        "  This can be so when a process starts while the chain gives no answer. It can also be so when a process",
        f"  starts, or closes a round, in the {duration(params.apply_retry_s)} after a round with no result: in that time the validator does",
        "  not read the files. No seat and no place in the queue changes by it: the validator picks the entries of a",
        "  round from a new check of the files, never from the published value. So do not replace a good file because",
        "  of it: wait for the next check.",
        "- The verdicts of the whole line are in mining.json under `queue.files`. Its list `verdicts` has one row for",
        "  each hotkey with a waiting seat, in UID order: `uid`, `hotkey`, `file_valid` and `file_verdict`.",
        f"  `file_verdict` is `{VALID}`, `{NOT_VALID}` or `{NOT_JUDGED}`. `{NOT_JUDGED}`: the validator has no verdict for the hotkey. No file",
        "  was stored at the last check, or the seat came after that check, or a new process made no check yet.",
        f"  `file_valid` is yes only with `{VALID}`. `hotkeys` counts the rows; `{VALID}`, `{NOT_VALID}` and `{NOT_JUDGED}` count the rows",
        "  with that verdict; `seats` counts the waiting seats. `checked_at` is `queue.files_checked_at`: the time at",
        "  which that check started. The validator does not publish the sha256 of a waiting file.",
        "- Identity: a registered UID on this subnet (`btcli subnets register --netuid "
        f"{params.netuid}`). One registration (UID + registration block) buys one seat in one round, win or lose.",
        "  A hotkey gets one turn. When its entry was charged, the hotkey never enters again, also with a new",
        "  registration. A turn in v1 that the cutover imported counts too. To enter again, register a new hotkey.",
        f"- Order: registration block, then UID. UID {params.burn_uid} has no seat.",
        "- A seat exists only while its registration is on chain. The validator reads the chain before each round.",
        "  When the registration leaves the chain (the UID is deregistered, or the UID goes to a newer registration),",
        "  the seat is removed and its turn is lost. A seat that is in an open round stays until that round closes.",
        "- When the hotkey of the UID changes and the registration stays, the seat keeps its place. The validator then",
        "  reads the file of the new hotkey: a file that was stored for the old hotkey does not count. When the new",
        "  hotkey already had its turn, the seat is removed.",
        "- One entry for each miner in a round. A second seat of the same hotkey or coldkey waits for a later round.",
        f"- `{NO_DECISION}`: our infrastructure failed (section 3). You are not charged and you go to the front of the",
        "  queue.",
        "- A failure is counted for your seat in one case only: your batch had the whole training time of the round",
        f"  (it started at the lease, and the box started its work on the round at most {AT_THE_LEASE_S} s after the start of the",
        "  lease), the first try of its training did not finish in that time (section 3), and at least one other batch",
        "  of the same round finished its training.",
        f"  Your entry then has the status `{NO_DECISION}` and no scores. When {params.max_infra_attempts} failures",
        "  are counted for a seat, the seat is held: it takes no turn until the operator releases it. A release also",
        f"  sets the count to zero. A `{NO_DECISION}` that comes from the scoring (section 6) has scores and is not counted.",
        "- A training job that failed is never counted for your seat: out of memory on the box, a crash of the job,",
        "  any other error of the training. The box tries the training again (section 3). If each try fails, your",
        f"  entry has the status `{NO_DECISION}` and no scores, and its `reason` starts with `training failed`. You are not",
        "  charged and you go to the front of the queue.",
        "- A round with no result: the box failed, not a seat. This is the case when the training failed for at least",
        "  one batch and finished for no batch, when the box stopped with an error, when its result did not arrive",
        "  before the end of the lease, or when the operator ended the round. Ordinarily nobody is charged, no failure",
        "  is counted for a seat, and each seat of the round goes to the front of the queue. The next round starts",
        f"  {duration(params.apply_retry_s)} after the",
        "  failure, or later. In the round file (section 9) you see it thus: `outcome` is `no_decision`, no entry has a",
        "  `status`, and `note` gives the cause. In `history` of mining.json the round has the same `outcome` and its",
        "  entries have no `status`; the `note` is only in the round file. If Stage A already found duplicates, those",
        f"  copies remain charged `{DUPLICATE}` and lose their seats even when every lease fails. Their terminal entries",
        "  are shown; every other seat stays uncharged and goes to the front, and the same retry delay applies.",
        f"- A round that has a result also gets the outcome `{NO_DECISION}` when none of its entries was charged. Each entry",
        "  then has a `status`, and the status of your entry says what applies to your seat.",
        f"- `{NOT_TRAINED}`: the training time of the round ended before your batch was trained (section 3). You are not",
        "  charged and you go to the front of the queue. It is not counted as a failure, so it never holds a seat.",
        f"- `{NOT_SCORED}`: the time limit of the round came before the first look of the scoring (section 3). You are",
        "  not charged and you go to the front of the queue. It is not counted as a failure, so it never holds a seat.",
        f"- `{SKIP}`: the trainer could not train the copy of your file, for example because a gradient or trained weight",
        "  was not finite. This training failure is uncharged and you keep your place. A non-finite stored `step_norm`",
        f"  that reaches judging is instead the charged `{NULL_UPDATE}` outcome described in section 5.",
        f"- `{DUPLICATE}` and `{NULL_UPDATE}`: your frozen batch was a duplicate, or its stored update fell below the floor."
        if params.min_step_norm > 0 else
        f"- `{DUPLICATE}` and `{NULL_UPDATE}`: your frozen batch was a duplicate, or its stored `step_norm` was not finite.",
        "  These are charged terminal outcomes, not `skip`: the hotkey's turn is used, the entry cannot win or be paid,",
        "  and it does not return to the queue. The rules and evidence are in `Duplicate and null batches` (section 5).",
        f"- A round with the outcome `{ABANDONED}`: a winner was named, but its files could not be checked or applied.",
        "  The head does not change, no entry of the round is charged, and a seat that was scored goes to the front of",
        "  the queue.",
        "",
        "### Your place while you wait",
        "",
        "- A seat that has a valid file stored keeps its place and is never moved back. While your entry is not",
        "  evaluated, you may replace your file as often as you like (section 5): a new valid file does not change",
        "  your place.",
        "- If your file is missing or invalid when the validator picks the entries of a round, your seat is skipped. You are not charged, and you",
        "  keep your place, with one exception: squat-trim.",
        "- Squat-trim moves a seat to the back of the queue. It applies only to a seat that has no valid file stored",
        f"  when the validator picks the entries of a round, and only when the registration of that seat is older than {show(params.squat_age_factor)} x its expected",
        f"  wait. Expected wait = max({duration(params.squat_min_wait_s)}, seats ahead x seconds per turn). Seconds per turn = {SECONDS_PER_DAY} / the number",
        f"  of entries that were charged in the last {SECONDS_PER_DAY // 3600} hours.",
        f"- So a seat with no file or with an invalid file keeps its place for at least {duration(params.squat_age_factor * params.squat_min_wait_s)} after its",
        f"  registration, and for as long as no entry was charged in the last {SECONDS_PER_DAY // 3600} hours. The two values",
        "  (`squat_min_wait_s`, `squat_age_factor`) are parameters (section 11). The validator can change them. This",
        "  text always shows the values that it uses.",
        "- A seat at the back still gets its turn: a round takes it when it has room after the front seats and the",
        "  normal seats that have a valid file. The seat stays at the back when a valid file arrives later.",
        "- A seat that v1 had moved to the back of its queue starts at the back here.",
        "- The operator can send a waiting seat to the front, between rounds only. The action is for a seat that had",
        "  a turn outside v2 that was not charged. Such a seat is then no longer at the back. Nothing is charged or",
        "  counted for it.",
        "- In the table: `front` = yes: the seat goes before the normal seats (it had a turn that was not charged).",
        "  `demoted` = yes: the seat is at the back, unless `front` is also yes. `held` = yes: the seat takes no turn",
        "  until the operator releases it. `position` counts every seat ahead of you, also a seat with no valid file,",
        "  which a round skips.",
    ]


def _file_lines(doc: Mapping[str, Any], params: Params) -> list[str]:
    recipe = params.recipe
    example = {
        "version": TRAIN_FORMATS[0],
        "hotkey": "<your hotkey ss58 address>",
        "train": {recipe.miner_bounds[0].key: getattr(recipe, recipe.miner_bounds[0].key)} if recipe.miner_bounds else {},
        "groups": [
            {
                "prompt": "<problem text>",
                "rollouts": [
                    {"output": "<reasoning> \\boxed{<right answer>}", "reward": 1.0},
                    {"output": "<reasoning> \\boxed{<wrong answer>}", "reward": 0.0},
                ],
            }
        ],
    }
    cost_cap = [
        f"  A file with a cost above {show(params.max_train_token_steps)} token-steps is invalid. To get below the cap: fewer steps, fewer",
        "  rollouts, or shorter outputs. The rule needs token counts, so `check` runs it only with `--tokenizer`.",
    ] if params.max_train_token_steps > 0 else ["  There is no cap on the cost now (`max_train_token_steps` is 0)."]
    return [
        "## 5. Your file",
        "",
        "You have one private object for each hotkey. A hotkey whose queue entry is not evaluated yet keeps its queue",
        "position, and may replace its file as often as it likes until the entry is evaluated. There is no fixed",
        "window. The validator copies the latest file when a round takes the entry: that copy is",
        "what is trained. The admission checks run again on each new file. The only rule that moves a waiting seat",
        "back is squat-trim (section 4): it applies to a seat with no valid file stored, never to a seat with a valid",
        "file. The file stays private unless it wins. Then the exact file is published (section 9).",
        "",
        "```json",
        json.dumps(example, indent=1),
        "```",
        "",
        f"- UTF-8 JSON, at most {params.max_batch_bytes} bytes. Top keys: `version`, `hotkey`, `groups`, and the optional",
        f"  `train`. `version` is one of {show(FORMATS)}. `hotkey` must be the owner of the object.",
        "- `groups`: a non-empty list of {prompt, rollouts}, with no other key. Each prompt is text with more than white",
        "  space in it, and differs from every other prompt. Each prompt has at least 1 rollout: {output, reward}, with",
        "  no other key. `output` is text.",
        f"- {show(CONTROL_TOKENS)} may not appear in a prompt or an output.",
        f"- Caps: {params.max_prompts} prompts and {params.max_rollouts} rollouts. Whole groups are kept in file order while",
        "  both caps hold. The other groups are dropped. All rules below apply to the groups that are kept.",
        f"- Tokens: an output has at most {params.max_output_tokens} tokens. Prompt + output has at most",
        f"  {params.max_context_tokens} tokens: an output below the output cap counts 1 token more there, its end-of-sequence",
        "  token. The prompt is counted as one user message in the chat template of the",
        f"  head. Training reads at most {recipe.train_max_output_tokens} output tokens and",
        f"  {recipe.train_max_context_tokens} prompt + output tokens of a rollout. A longer rollout is valid: the trainer",
        "  truncates it (section 2).",
        "- Reasoning: the reasoning of an output is the text before its first `\\boxed`, or the whole output when it",
        "  has none. A rollout with fewer than",
        f"  {params.min_reasoning_tokens} reasoning tokens is answer-only. At most {show(params.max_answer_only_frac)} of the",
        f"  rollouts may be answer-only. At least {show(params.reasoning_token_share)} of all output tokens must be reasoning.",
        "- Repeats: the final answer is the content of the last `\\boxed{}`, with all white space removed. One final",
        "  answer may be in at most",
        f"  max({params.max_answer_repeat_count}, {show(params.max_answer_repeat_frac)} x rollouts) rollouts.",
        "- `reward`: a finite number. The rewards of one prompt must be small enough to take their mean. A reward must",
        "  equal the grade that the official grader gives to (prompt, output). The",
        f"  allowed difference is {show(params.reward_tolerance)}. One wrong reward makes the whole file invalid. An output of exactly",
        f"  {params.max_output_tokens} tokens stopped at the cap: its official grade is 0. At least",
        f"  {show(params.min_pool_prompt_frac)} of the prompts must be tasks of the public pool (0 = no such rule). A prompt that",
        "  is not a task of the pool has no official grade: its rewards are not compared with anything.",
        "- A rollout that the official grader cannot grade inside its time limit is not allowed in a file, whatever its",
        f"  reward: rule code `{TIME_LIMIT}`.",
        f"- `{NO_SIGNAL}`: your file must train something. A rollout trains when its advantage is not 0 and it has at",
        "  least 1 output token inside the training caps. The advantage is the reward minus the mean reward of its",
        "  group (`group_size`, section 2). A file in which no rollout trains is invalid. One rollout per prompt never",
        "  trains. So the example above has two rollouts with different rewards.",
        f"- `{TRAIN_COST}`: cost = steps x the tokens of the rollouts that train. The tokens of one rollout are its",
        f"  prompt tokens plus its output tokens. room = min({recipe.train_max_output_tokens},",
        f"  {recipe.train_max_context_tokens} - prompt tokens). An output shorter than the room counts its tokens + 1",
        "  (the end-of-turn token). An output as long as the room or longer counts the room. A rollout with advantage",
        "  0 counts 0.",
        *cost_cap,
        f"- Two limits keep the training of a round short: the training time of the round ({show(params.train_time_budget_s / 60)} minutes for",
        f"  all batches, section 3) and this cost cap for one file ({f'{show(params.max_train_token_steps)} token-steps' if params.max_train_token_steps > 0 else 'no cap now'}).",
        "- The cost does not count `minibatches`. Two things come on top of the cost with more than one minibatch. One:",
        "  the first pass is one more run of every row that trains through the model. Two: each update has a fixed cost",
        f"  of its own, about {show(UPDATE_COST_S)} s. That number is an estimate and not a measured value. So steps x minibatches",
        f"  updates add about {show(UPDATE_COST_S)} s x steps x minibatches to the training of your file, and no rule of the file check",
        "  counts that time. A training that still runs at the end of the training time is stopped, and the batch is not",
        "  judged in that round (section 3). Keep a margin of your own when you set `minibatches` high.",
        "- Nothing is repaired or clamped. The first rule that fails makes the whole file invalid.",
        "- Check a file before you upload it, with the same code as the validator:",
        "  `python v2/miner/affine_miner.py check --mining-json mining.json --hotkey <ss58> batch.json`. Add",
        "  `--tokenizer <head directory>` for the token rules and `--grade` for the reward rule. `--grade` needs",
        "  `--tokenizer`: only a token count can tell that an output stopped at the cap. Install, from a copy of the",
        '  code (section 2): `pip install -e "v2[miner,envs]"`.',
        "",
        "### Rule codes",
        "",
        "A check stops at the first rule that a file breaks. The `check` command names that rule. The validator",
        "publishes only `file_valid` (section 4), not the rule. The rules run in this order:",
        "",
        *table(("rule code", "the file is invalid when"), ((f"`{code}`", meaning) for code, meaning in RULES.items())),
        "",
        f"Not a final verdict: {', '.join(f'`{code}`' for code in TRANSIENT)}. Such a code comes from one check on our machine at one",
        f"moment, so the validator checks the file again when it tries to open a round, at most once each {FAULT_HOLD_S // 60} minutes.",
        "A failed check is counted only when our checker passes its own known-good case at that moment. After",
        f"{FAULT_TRIES} counted checks in a row that fail at the same place of the same",
        "file, the verdict stays until you upload a new file. A new file is checked at the next try, with no wait.",
        "With such a code, before and after that, a round does not take the file: you are not charged, and the seat",
        "is skipped as with any invalid file (section 4). The validator does not publish the code or the count: you",
        "see only `file_valid`. Every other code is final for those bytes: upload a new file.",
        "",
        *_rollout_lines(params),
        "",
        *_duplicate_lines(params),
        "### How to upload",
        "",
        *_upload_lines(doc["contract"]),
    ]


def _rollout_lines(params: Params) -> list[str]:
    """A guide: how a miner makes rollouts that the trainer can use.

    It states no rule of its own. Each line follows the file check (``admission.py``), the trainer (``gpu/``),
    the grader call (``envs/math_grader.grade``) or the scoring request (``gpu/evaluator.chat_body``).
    """
    recipe = params.recipe
    pool = [
        "  For a task of the pool, `prompt` is the problem text, or the instruction of section 6, one blank line and the",
        "  problem text. Both are the same task for the grader, and two different prompts for the model.",
    ] if params.env == ENV_ID else []
    return [
        "### How to make rollouts",
        "",
        "A rollout is one sampled answer of a model to a prompt, with its reward. This is one way to make a file that",
        "is valid and that trains. Each step says what the code of the validator does with your choice. To see what",
        "the file does before you upload it, train and score it on your machine (section 10).",
        "",
        "- Model: sample from head M. Get its files as section 1 says. The trainer scores your tokens under head M,",
        "  and a token outside the support of head M does not train (section 2).",
        "- Prompt: the trainer reads `prompt` as one user message (section 2).",
        *pool,
        "  For your sampler, render the prompt as the trainer does: one user message, the chat template of the head,",
        "  the generation prompt, and no other argument to the template.",
        "- Scoring renders a task in another way than training (section 6): the instruction is the system message, the",
        "  problem is the user message, and the template gets `enable_thinking` true. So the model does not read the",
        "  same tokens before its answer in training and in scoring.",
        f"- Sampling: use the temperature that you state in `train.sample_temperature` (default {show(recipe.sample_temperature)}), top_k {recipe.support_top_k}",
        f"  and top_p {show(recipe.support_top_p)}. These are the settings of the support (section 2). The sampler must work as the",
        "  support does: top_k first, then top_p over the tokens that are left. Then each token that you",
        "  sample is in the support of head M on your machine. On the box the logits come from other hardware, so a",
        "  token at the edge of the support can be outside it there; no run has measured how often. No rule checks",
        "  where an output comes from: text of another model or of other settings is valid, and only its tokens in the",
        "  support of head M train.",
        "- Output: `output` is the text that the model wrote after the generation prompt: its tokens decoded with the",
        "  special tokens kept (`skip_special_tokens` false), so that `<think>` and `</think>` are in the text, and",
        f"  without the final `{END_OF_TURN}`. The trainer encodes this text again and adds the end-of-turn token itself",
        "  (section 2). It trains on the tokens of that encoding, which are the tokens that you sampled only if the text",
        "  encodes back to them.",
        f"- Stop: let the model end by itself, or stop it at {params.max_output_tokens} tokens (`max_output_tokens`). An output of",
        "  exactly that many tokens counts as stopped at the cap, and a longer output is invalid.",
        "- Reward of a pool task: the official grade, 1.0 for a pass and 0.0 for every other result. Take it from the",
        "  grader itself, `sn120_v2.envs.math_grader.grade(task, output, capped).reward`, with `capped` true for an",
        "  output that stopped at the cap. Or let the `check` command compare your rewards (`--tokenizer` and `--grade`,",
        "  above). The grader reads the same `output` text.",
        "- Signal: give a prompt two rollouts or more, with different rewards. The advantage is the reward minus the",
        "  mean reward of the group, so a group in which all rewards are equal trains nothing.",
        "- Length: the trainer reads the room of section 2 and no more. It cuts a longer output there and takes it as a",
        "  rollout that did not end; with a negative advantage the derail rule then applies to it. The cost rule above",
        "  counts the same tokens.",
        "- Before each upload, run the `check` command with `--tokenizer` and `--grade`. It runs the check of the",
        "  validator. It does not train.",
    ]


def _upload_lines(contract_block: Mapping[str, Any]) -> list[str]:
    """The upload steps. The addresses and the two times come from ``links.upload_entry``."""
    upload, links_url = contract_block.get("upload"), contract_block.get("links_url")
    header = show([f"`{name}: {value}`" for name, value in UPLOAD_HEADERS.items()])
    tool = "`python v2/miner/affine_miner.py upload --mining-json mining.json --hotkey <ss58> --tokenizer <head directory> --grade batch.json`"
    if not upload:
        return [
            "- Each registered hotkey gets a sealed link: a public file that only the secret key of the hotkey can",
            "  open. It holds a time-limited upload URL for your object.",
            "- Where to get the link: not published yet. This file gives the address and the steps when the link files",
            "  exist.",
            f"- Upload: an HTTP PUT of the file bytes to the opened URL with the header {header}. {tool} checks the file,",
            "  reads the URL from the environment variable `AFFINE_PUT_URL` and sends the file with that header.",
        ]
    return [
        "You upload with a sealed link: a public file that only the secret key of your hotkey can open. It holds a",
        "time-limited upload URL for your object.",
        "",
        "- Activation (one time): your hotkey must be an Ed25519 key. Run",
        "  `python -m sn120_v2.links activate --wallet <wallet name> --hotkey <hotkey name>` to see the activation. Add",
        "  `--execute` to write it to the chain. It costs the fee of one commitment. The chain shows it after about 3",
        "  minutes. An activation made for v1 stays valid: do not make it again. Install:",
        '  `pip install -e "v2[links,chain]"`.',
        f"- Where to get the link: {upload['sealed_links']}. The file is public, but only the secret key of your hotkey",
        f"  opens it. A new file is written every {show(upload['reissue_every_seconds'] / 60)} minutes. Each link works for {show(upload['link_ttl_seconds'] / 3600)} hours.",
        "- You have a link while your hotkey is registered, has the activation, and its queue entry is not evaluated",
        "  yet. You may upload again as often as you like. Each upload replaces the file before. The file that is",
        "  stored when a round takes your entry is the file that is trained.",
        "- Open the link:",
        f'  `export AFFINE_PUT_URL="$(python -m sn120_v2.links open --hotkey <ss58> --key <seed file> --links-url {links_url})"`.',
        "  The seed file holds the 32-byte Ed25519 seed of your hotkey as 64 hex characters. A bittensor hotkey file",
        '  that is not encrypted also works. Install: `pip install -e "v2[links]"`.',
        f"- Upload: {tool} checks the file, reads the URL from the environment variable `AFFINE_PUT_URL` and sends the",
        f"  file. Or use any HTTP tool: an HTTP {upload['method']} of the file bytes to that URL with the header {header}. The link",
        "  is signed for that header: without it the store refuses the upload.",
        "- The opened URL is a secret. Anyone who has it can replace your file until the link expires.",
        "- HTTP 404 on your link file means one of three things: the hotkey is not registered, it has no activation, or",
        "  its entry was evaluated. After a new registration or activation, wait for the next link file. A hotkey",
        "  whose entry was evaluated gets no link again (section 4).",
        "- After the upload, read `file_valid` in section 4.",
    ]


def _pool_lines(params: Params) -> list[str]:
    if params.env != ENV_ID:
        return []
    pool = load_pool()
    dropped_file = POOL_FILE.with_suffix(".dropped.json")
    dropped = json.loads(dropped_file.read_bytes())
    if dropped["from_sha256"] != POOL_V2_0_SHA256 or dropped["to_sha256"] != POOL_SHA256:
        raise ConfigError("the dropped-task list does not belong to the current pool")
    count, removed = f"{len(pool):,}", f"{len(dropped['dropped']):,}"
    rotation = []
    rotation_changed = any(getattr(params, f.name) != f.default for f in dataclasses.fields(params) if f.name.startswith("rotation_"))
    if params.rotation_rule == "info":
        rotation = [
            "- Task rotation uses `affine-pool-rotation-v2`. Every active, reserve and non-withdrawn retired task is a public candidate.",
            "  With enough known history, p is the IRT predicted head pass rate. A reserve row may publish `prior_p`; otherwise the configured prior is used.",
            "  Score = p(1-p), Bernoulli Fisher information. Dead tasks (at least the configured head rollouts and zero passes) score zero.",
            "  Saturated tasks keep their score; their retirement reason is `high`. Withdrawn active tasks leave outside the churn cap.",
            f"  The target is {params.rotation_active_size or 'the initial active size'}: shrink and fill have no churn cap. Steady-state swaps need a strictly higher score and are capped at",
            f"  floor(rotation_churn_frac * target) = floor({show(params.rotation_churn_frac)} * target). Ties use sha256(round_id|rank|task_id).",
            "  Retired tasks may return by score and restore their archived IRT row. The complete ranking, next-up queue and summary are public.",
            "  `python -m sn120_v2.replicate rotation --mining-json <mining.json> --round rNNNNNN` recomputes a decision from public inputs.",
        ]
    elif params.rotation_enabled or rotation_changed:
        low = "off" if params.rotation_retire_low_p == 0 else f"on at empirical head p < {show(params.rotation_retire_low_p)} with {params.rotation_n_min_low} rollouts"
        rotation = [
            "- Task rotation is " + ("on" if params.rotation_enabled else "off") + ". Each round may retire at most the configured k high/low pass tasks and admits from the",
            "  public FIFO reserve before the draw. Active ids, the full decision, the next-up queue and mirrored reserve",
            "  bytes are linked from `pool` in the current round and mining.json. New tasks start at head difficulty (p=.5).",
            f"- Retire high: rounds_seen >= {params.rotation_n_min} and IRT head p > {show(params.rotation_retire_p)}. Low retirement is {low}.",
            "  Withdrawn active tasks leave outside k. `python -m sn120_v2.replicate rotation --mining-json <mining.json>",
            "  --round rNNNNNN` recomputes a decision from its public inputs.",
        ]
    return [
        f"- Task pool: {count} tasks in `sn120_v2/envs/{POOL_FILE.name}` in the code (sha256 `{POOL_SHA256}`). Scoring sends this",
        f'  instruction as the system message and the problem as the user message: "{MANDATE}"',
        f"- {removed} tasks of the earlier pool (sha256 `{POOL_V2_0_SHA256[:8]}…`) were removed because their key is wrong, the problem",
        "  is ambiguous or broken. The list is",
        f"  `sn120_v2/envs/{dropped_file.name}` in the code: {REPO}/blob/v2/v2/sn120_v2/envs/{dropped_file.name}.",
        "  A prompt that is one of the removed tasks is no longer a pool task. It has no official grade, and its reward is not compared,",
        "  same as any non-pool prompt (see the `min_pool_prompt_frac` line).",
        "- A prompt of your file is a pool task when it is the problem text, with or without that instruction and one",
        "  blank line before it. The official grader is `sn120_v2.envs.math_grader.grade(task, output, capped)`. It",
        "  grades model outputs in scoring and the rewards in your file.",
        *rotation,
    ]


def _server_command(params: Params) -> list[str]:
    """The command of one vLLM server (``gpu.evaluator.server_argv``), one option on a line.

    The address of the server is left out: it is local to the box and changes no score.
    """
    argv = server_argv("<directory of the base files>", 0, params, params.max_num_seqs)
    if not params.eval_merged:
        argv[argv.index("--max-loras") + 1] = LORA_TABLE  # a value of the round, not of the parameters: gpu.evaluator.lora_table
    start = argv.index("-m")
    options: list[list[str]] = []
    for word in argv[start + 2 :]:
        if word.startswith("--"):
            options.append([word])
        else:
            options[-1].append(shlex.quote(word))
    lines = [f"python -m {argv[start + 1]}", *("    " + " ".join(option) for option in options if option[0] not in LOCAL_OPTIONS)]
    return [line + " \\" for line in lines[:-1]] + lines[-1:]


def _scoring_lines(params: Params) -> list[str]:
    """The scoring of a round, step by step.

    Every word here follows ``worker.LocalRunner._evaluate`` (chunks, looks, the clock), ``gpu/evaluator.py``
    (servers, requests, outcome codes), the head of ``envs/math_grader.py`` (the steps of the grader) and
    ``judging.py`` (the estimator and the rules). When one of them changes, change this function.
    """
    # The messages of a task as the pool makes them. The instruction stands in section 6 as it is sent; JSON would double its backslashes.
    named = {MANDATE: "<the instruction above>"}
    messages = [message | {"content": named.get(message["content"], message["content"])} for message in Task("", "", "<the problem text>", "").messages()]
    body = chat_body(f"<{NULL_MODEL} or the lora_hash>", messages, 0, params) | {"seed": "<the seed of the rollout>"}
    grader = [
        f"- Grader: `sn120_v2.envs.math_grader.grade(task, output, capped)`, version `{GRADER_VERSION}`. `output` is the text of",
        "  the reply, with the special tokens kept. The strict format is: one `<think>...</think>` block first, then",
        "  exactly one balanced, non-empty `\\boxed{}` with nothing after it. math-verify compares the boxed answer with",
        "  the reference answer of the task. When the answer is not yet right, the lenient rules try again: P1 and P2",
        "  find a box that the strict format refused, and E1 to E7 accept other common ways to write the same answer.",
        "  That file states each rule and the version of each package that the grader needs; with another version",
        f"  the grader refuses to run. The grade is `{PASS}` or `{FAIL}`, or `{ERROR}` when the grader itself cannot decide: it cannot",
        f"  parse the reference, math-verify raises an error, or a time limit of the grader passes. Such an `{ERROR}` is",
        "  final: the text is not graded again. The grader does not know which model wrote a text.",
    ] if params.env == ENV_ID else []
    return [
        "### The scoring procedure",
        "",
        *([
            f"- Merged-candidate evaluation: {MERGED_EXPLANATION}",
            "  Each candidate has its own vLLM server for its merged base; null keeps the bf16 king base. A GPU serves one model at a time",
            "  and may switch after its in-flight work finishes. No runtime LoRA is loaded. The candidate's result records the merged base",
            "  manifest hash and install kernel; promotion refuses a fold whose base manifest differs.",
        ] if params.eval_merged else []),
        "This is everything that the box does to score a round, in the order in which it does it. The code is in four",
        "files: `sn120_v2/worker.py` (chunks, looks and the clock), `sn120_v2/gpu/evaluator.py` (servers, requests and",
        "outcome codes), `sn120_v2/envs/math_grader.py` (the grader) and `sn120_v2/judging.py` (the estimator and the",
        "rules). Section 10 has the command that runs this code on your machine.",
        "",
        f"- Servers: an idle GPU may prewarm its normal vLLM server for null. A GPU never starts vLLM while its training job",
        *([
            "  is active; after training exits it serves the candidate it trained, or null if it trained no candidate. Each merged",
            f"  model is served under its candidate's `lora_hash`; null is `{NULL_MODEL}`. The servers run with `HF_HUB_OFFLINE` set.",
            "  The vLLM version is the one that the `gpu` extra of `v2/pyproject.toml` names. The command of one merged server,",
        ] if params.eval_merged else [
            "  is active; after training exits it starts or reuses that same default-argv server. Each server",
            f"  holds the base of head M under the model name `{NULL_MODEL}`. The box loads each LoRA of the round into every",
            "  server with `/v1/load_lora_adapter`, with its `lora_hash` as the model name. The servers run with the",
            "  variable `VLLM_ALLOW_RUNTIME_LORA_UPDATING` set (vLLM needs it for that route) and with `HF_HUB_OFFLINE` set.",
            "  The vLLM version is the one that the `gpu` extra of `v2/pyproject.toml` names. `--max-loras` is the number of",
            "  batches of the round, trained or not: it is the same for every chunk of the round. The command of one server,",
        ]),
        "  without its local address:",
        "",
        "```",
        *_server_command(params),
        "```",
        "",
        "- Request: one POST to `/v1/chat/completions` for each model, each sample and each task of a chunk. The body",
        "  is the same for null M and for a LoRA, but for `model`:",
        "",
        "```json",
        json.dumps(body, indent=1, ensure_ascii=False),
        "```",
        "",
        f"- `model` is `{NULL_MODEL}` for null M and the `lora_hash` for a LoRA. The seed is the seed of the pairing rule above.",
        f"  Task i of the full draw (i counts from 0) goes to a free GPU according to the solo scheduler. A GPU may steal",
        *([
            "  another candidate's remaining block. Null M sample 0 and every LoRA use the same task, sample and seed.",
            f"- Null M (the king) runs {params.null_samples} sample(s) per task; every LoRA runs {params.samples_per_task}. Sample 0 has the same seed",
            "  for null M and every LoRA. Later king samples use their own seed and are not run for any LoRA. Only the adapter",
            "  and the serving GPU can differ between paired rollouts.",
        ] if params.null_samples_per_task is not None else [
            "  another candidate's remaining block. Null M and every LoRA use the same task, sample and seed; only the adapter",
            "  and the serving GPU can differ.",
        ]),
        f"- The box feeds {params.max_num_seqs} concurrent requests to each server (`max_num_seqs`), subject to the local file",
        "  descriptor limit. A server runs at most `--max-num-seqs` of them at a time; the",
        f"  others wait in the server. The time limit of one try of a request is {params.eval_time_budget_s} s (`eval_time_budget_s`),",
        "  or the time that is left to the stop time of the round, when that is less.",
        f"- Reply: with `finish_reason` `stop` the model ended by itself. A final `{END_OF_TURN}` is removed from the text,",
        f"  and the grader grades the text. With `finish_reason` `length` the model hit the token cap: the outcome is `{CAP}`",
        "  and the text is never graded. A reply of any other form is not usable.",
        "- A request is sent again after a failure of the connection, a time limit, an HTTP status of 500 or above, or",
        f"  a reply that is not usable: at most {ServeSettings().retries + 1} tries in all. After an HTTP status below 500 it is not sent again. A",
        f"  request with no usable reply has the outcome `{ERROR}`. One exception: when the stop time passed during the chunk",
        "  and a request has no reply, the whole chunk is not used (section 3).",
        *grader,
        "- A grade that is lost because a worker process of the grader died is not an outcome. The reply is graded again:",
        f"  at most {ServeSettings().grade_tries} grades in all. When a worker still dies, the chunk failed.",
        f"- Streaming: tasks are claimed in draw order by the solo scheduler and may be served or stolen by any free GPU.",
        "  Each model is graded in draw order;",
        "  a look is judged only after null and that model have all samples through the exact look size. Speculation",
        "  past a look is allowed, but results after a drop are discarded and never appear in the outcome strings.",
        f"- Cutoff blocks: for the stop-time prefix, tasks are grouped in blocks of {params.eval_chunk_tasks} tasks",
        "  (`eval_chunk_tasks`). Only a complete block shared by null and every open LoRA is used. This is not a",
        "  scheduling barrier (section 3).",
        "- A failure of the box in a chunk (a server died, a LoRA did not load, the grader failed) is not an outcome.",
        f"  The chunk runs again: at most {BOX_TRIES} runs in all. When each of them fails before the stop time, the box stops with an",
        "  error and the round has no result (section 4). A run that fails after the stop time ends the scoring, and",
        "  the chunk is not used.",
        "- Outcome strings: each model has one string for each sample" + (": null M has `null_samples_per_task` strings (or" if params.null_samples_per_task is not None else ". A string has one character for each task, in"),
        *([
            "  `samples_per_task` when it is null), and every LoRA has `samples_per_task`. A string has one character for each task, in",
        ] if params.null_samples_per_task is not None else []),
        f"  the order of the draw: `{PASS}`, `{FAIL}`, `{CAP}`, `{ERROR}`, or `{NOT_RUN}` for a task that was not run.",
        "- Estimator, in full: n is the number of tasks that the LoRA answered. They are the first n tasks of the draw.",
        f"  The pass rate of a model on a task is the share of its samples with `{PASS}`. The sums of the estimator above run",
        "  over these n tasks, with the weights w_t of the draw. When se is 0, z is 0.",
        "- Drop rule, in full: after each look but the last, the box computes z of each open LoRA on the tasks so far.",
        "  Look k has scored the share t_k of the draw. a_k = spent(t_k) - spent(t_(k-1)), with spent(t) = `drop_alpha` *",
        "  ln(1 + (e - 1) * t), and spent = 0 before the first look. c_k is the normal quantile of 1 - a_k. The LoRA is",
        "  dropped when (z + c_k) / sqrt(t_k) <= `z_bar`. No LoRA is dropped at a look with fewer than",
        f"  {params.min_scored_tasks} tasks (`min_scored_tasks`), or when its se is 0. A dropped LoRA answers no later chunk. When no LoRA is",
        "  open, the scoring ends.",
        *([
            "- Field-median win rule, in full: duplicates are removed first by the existing dedup, the round uses one round-wide",
            "  training seed, and there are no eligibility floors. For each task t, m_t is the median of the per-task pass rates",
            "  of the deduped scoring batches plus the king; an even 0/1 split gives 0.5. A batch dropped early contributes only",
            "  on tasks it ran. For your scored tasks, d_t = your pass rate - m_t; `field_delta` is the weighted mean of d_t",
            "  and `field_se` its standard error, with the same weights as the king estimate. `field_z` = `field_delta` / `field_se`,",
            "  or 0 when `field_se` is 0. Crown only when `field_z` > `z_field_bar` AND paired z",
            f"  against the king > `z_bar`. The highest delta among eligible LoRAs wins; otherwise the king stays. The old z {show(RETIRED_KING_Z_BAR)}",
            "  king-only bar is retired. The thresholds are provisional and will be recalibrated.",
        ] if params.crown_gate == "field_median" else []),
        "- Win rule, in full. A LoRA is eligible when all of these are true. It has an outcome for every task of the",
        "  draw, or, when the stop time ended the scoring, for every task that null M has an outcome for; a LoRA that",
        "  was dropped at a look does not. n is at least `min_scored_tasks`.",
        *(["  z > `z_bar` AND `field_z` > `z_field_bar`."] if params.crown_gate == "field_median" else ["  z > `z_bar`."]),
        "  (its count of",
        *([
            f"  `{CAP}`) / (n x LoRA sample count) <= max(`cap_hit_floor`, (the count of `{CAP}` of null M on the same n tasks) / (n x king sample count) +",
            "  `cap_hit_margin`), where the LoRA denominator is n x `samples_per_task` and the null M denominator is n x its number of",
            "  outcome strings. Each side is divided by its own denominator. This test uses exact fractions of whole counts, and each parameter as the",
        ] if params.null_samples_per_task is not None else [
            f"  `{CAP}`) / rollouts <= max(`cap_hit_floor`, (the count of `{CAP}` of null M on the same n tasks) / rollouts +",
            "  `cap_hit_margin`), where rollouts = n x `samples_per_task`. This last test uses exact fractions of whole counts, and each parameter as the",
        ]),
        "  decimal number that is written here: a share exactly on the limit passes.",
        f"- Cover, in full: (rollouts - the count of `{ERROR}`) / rollouts >= `min_cover`, as exact fractions, for null M on the",
        "  tasks that it answered and for a LoRA on its n tasks.",
        "- Winner: the eligible LoRAs are read in queue order. A later LoRA takes the place of the best one so far",
        f"  only when its delta is higher by more than {show(TIE_TOLERANCE)}.",
        "- The box does not pick the winner. The controller computes every score again from the outcome strings",
        "  (`sn120_v2.judging.decide`). The round file (section 9) holds the draw with its weights (`draw`), the outcome",
        "  strings of null M (`null_outcomes`) and those of the winner (`winner_outcomes`). With",
        "  `sn120_v2.judging.score_entry` you can compute the scores of the winner again. The outcome strings of the",
        "  other entries are not published.",
        "- What a scoring of your own repeats: the draw and its weights, the seed and the body of each request, the",
        "  grade of a given text (but for a time limit of the grader), and each formula above. It does not repeat the",
        "  text of a rollout: the same request with the same seed can give another text on another machine or at",
        "  another time, and no run has measured how often. So your outcome strings and your delta can differ from",
        "  those of the box. The tests of the code start no vLLM server: they put a stand-in in its place. The box",
        "  runs vLLM on GPUs.",
    ]


def _win_rule_lines(params: Params) -> list[str]:
    if params.crown_gate != "field_median":
        return [
            f"- Win rule: z is strictly above {show(params.z_bar)} (beat M by more than that many standard errors), on at least",
            f"  {params.min_scored_tasks} tasks, with no correction for the number of LoRAs. The cap-hit rate of the LoRA may",
            f"  not be above max({show(params.cap_hit_floor)}, null M cap-hit rate + {show(params.cap_hit_margin)}).",
        ]
    return [
        "- Win rule (`crown_gate` = `field_median`): duplicates are removed first (the existing dedup), and the round has one",
        "  round-wide training seed. There are no floors on how many candidates may be eligible.",
        "- Per task t, the field median m_t is the median of the per-task pass rates of the deduped scoring batches plus the king.",
        "  With an even number of members, the two middle values are averaged (0.5 for a 0/1 split). A batch dropped at an",
        "  early look contributes only on the tasks it ran. On your scored tasks, d_t = your pass rate - m_t; `field_delta` is",
        "  the weighted mean of d_t and `field_se` its standard error, using the same sampling weights as the king estimate.",
        "  `field_z` = `field_delta` / `field_se`, or 0 when `field_se` is 0. There is no task re-weighting.",
        f"- To crown, you need `field_z` > `z_field_bar` ({show(params.z_field_bar)}) AND paired z against the king > `z_bar`",
        f"  ({show(params.z_bar)}), on at least {params.min_scored_tasks} tasks. The cap-hit limit is",
        f"  max({show(params.cap_hit_floor)}, null M cap-hit rate + {show(params.cap_hit_margin)}), as before. The old z {show(RETIRED_KING_Z_BAR)}",
        "  king-only bar is retired. These thresholds are provisional and will be recalibrated.",
    ]


def _judging_lines(doc: Mapping[str, Any], params: Params) -> list[str]:
    kings, keys, url = doc.get("kings") or [], object_keys(), doc["contract"]["public_url"]
    snapshots = {"the open round": (doc.get("round") or {}).get("irt_snapshot_id"), "the last closed round": doc.get("irt_snapshot_id")}
    where = [
        f"- Snapshot of {name}: id `{found}`, object `{layout.irt_snapshot(found)}`" + (f", {url}/{layout.irt_snapshot(found)}" if url else "")
        for name, found in snapshots.items() if found
    ]
    looks, n = params.look_sizes(), params.n_tasks
    mass = ([
        f"- The IRT sampler gives each task a mass in proportion to sqrt(p(1 - p)), where p is the head's predicted pass rate on it, mixed with a uniform share of {show(params.irt_explore_frac)} over all tasks. Every task keeps a mass above zero; the published weights 1/pi keep the estimate unbiased.",
    ] if params.irt_draw_rule == "sqrt_fisher" else [
        f"- The IRT sampler gives each task a mass in proportion to p(1 - p), where p is the head's predicted pass rate on it (the Fisher information of the task under the Rasch model), mixed with a uniform share of {show(params.irt_explore_frac)} over all tasks. Every task keeps a mass above zero; the published weights 1/pi keep the estimate unbiased.",
    ]) if params.irt_draw_rule in ("fisher", "sqrt_fisher") else [
        f"- The IRT sampler favours tasks where the predicted pass rate of the head is from {show(params.irt_middle_band[0])}",
        f"  to {show(params.irt_middle_band[1])}, and keeps {show(params.irt_explore_frac)} of the draw for new tasks.",
    ]
    drops = [
        f"{params.z_bar * math.sqrt(size / n) - drop_margin(look, n, params):.3f} after {size} tasks"
        for look, size in enumerate(looks[:-1])
    ] or "no early look"
    separate_king = params.null_samples_per_task is not None
    return [
        "## 6. Judging",
        "",
        "- All tasks are public. There is no hidden task set.",
        *_pool_lines(params),
        *([
            f"- Draw: {params.n_tasks} tasks from the public pool of `{params.env}`. The king (null M) has {params.null_samples} sample(s)",
            f"  per task; every LoRA has {params.samples_per_task}.",
        ] if separate_king else [
            f"- Draw: {params.n_tasks} tasks from the public pool of `{params.env}`, {params.samples_per_task} sample(s) per task.",
        ]),
        "  The draw is a fixed function of the public sampler snapshot and the public round seed. Both are frozen at",
        f"  round start. Rebuild a draw with `{draw_from_snapshot.__module__}.{draw_from_snapshot.__name__}(snapshot,",
        "  round_seed, n_tasks)`.",
        *mass,
        *([
            "- Pairing: null M sample 0 and every LoRA run the same tasks with the same seeds. Later null M samples are king-only.",
        ] if separate_king else [
            "- Pairing: the null copy of M and every LoRA run the same tasks with the same seeds. The seed of rollout j",
        ]),
        f"  The seed of rollout j on a task is the first {SEED_BITS} bits of sha256(\"<round_seed>|<task_id>|<j>\"), as a whole number; j counts from 0." if separate_king else f"  on a task is the first {SEED_BITS} bits of sha256(\"<round_seed>|<task_id>|<j>\"), as a whole number; j counts from 0.",
        f"- Decode: temperature {show(params.temperature)}, top_p {show(params.top_p)}, top_k {params.top_k}, {params.dtype},",
        f"  at most {params.max_output_tokens} output tokens and {params.max_context_tokens} prompt + output tokens.",
        f"- Outcome of a rollout: `{PASS}` pass, `{FAIL}` fail, `{CAP}` hit the token cap (not passed), `{ERROR}` our",
        "  infrastructure failed (not passed, stays in the count).",
        *([
            "- Estimator: d_t = lora_pass_t - mean(king_samples_t), where lora_pass_t and each king sample are 1 for PASS and 0 otherwise. delta = sum(w_t * d_t) / sum(w_t), with the",
            "  published sampling weights w_t. se = sqrt(n / (n - 1) * sum((w_t * (d_t - delta))^2)) / sum(w_t). z = delta / se.",
            "  Conditional on task and sample-0 pairing, Var(d_t) = Var(lora_t) + Var(king_t)/k - 2Cov(lora_t, king_t),",
            "  for k king samples. The realized-d spread retains the same design-based standard error with HT weights.",
        ] if separate_king else [
            "- Estimator: d_t = LoRA pass rate - null M pass rate on task t. delta = sum(w_t * d_t) / sum(w_t), with the",
            "  published sampling weights w_t. se = sqrt(n / (n - 1) * sum((w_t * (d_t - delta))^2)) / sum(w_t). z = delta / se.",
        ]),
        *_win_rule_lines(params),
        "- The eligible LoRA with the highest delta wins. A tie goes to the miner that was earlier in the queue.",
        "  With no eligible LoRA, M stays and there is no king. There is no second confirmation round.",
        f"- Looks: scores are checked after {show(looks)} tasks. The win rule runs only at the last look. An early",
        "  look can only drop a LoRA that can no longer reach the win bar: after a share t of the tasks, it is",
        f"  dropped when (z + c) / sqrt(t) <= {show(params.z_bar)}. c comes from a Pocock-type alpha-spending function with",
        f"  total level {show(params.drop_alpha)}. For this draw, a LoRA is dropped when z is at or below: {show(drops)}.",
        "- When the time limit of the round stops the scoring, the win rule runs on the tasks scored by then, with no",
        f"  further look. With fewer tasks than the first look, there is no decision: `{NOT_SCORED}` (section 3).",
        f"- If fewer than {show(params.min_cover)} of the rollouts of a LoRA resolve without an infrastructure error, its entry is",
        f"  `{NO_DECISION}`. If that is so for null M, every scored entry of the round is `{NO_DECISION}`. Neither is counted as a",
        "  failure for the seat (section 4).",
        f"- After {params.fold_every_k} applied winner(s), the stacked LoRAs are folded into a new base."
        + (
            " The validator publishes the LoRAs and the sha256 of every file of the new base and of the new fp32 master. It"
            " publishes the files themselves only for a checkpoint head (section 1)."
            if doc.get("head_chain")
            else ""
        ),
        "",
        *_scoring_lines(params),
        "",
        "## 7. Payout",
        "",
        f"- A king is a round winner that was applied. The window is the last {params.kings_window} distinct king UIDs. A UID that wins",
        "  again has one place in the window, with its newest win.",
        "- Each king of the window gets an equal share of all the weight, while the hotkey that won is still registered",
        "  on that UID: 1 / (the number of such kings).",
        "- The share of a king that is no longer registered is split among the other registered kings of the window.",
        f"- A round with no winner changes nothing in the payout: the window of the last {params.kings_window} kings keeps being",
        "  paid.",
        f"- No burn, in any case: no weight goes to UID {params.burn_uid} or to the owner as a fallback. If no king of the window is",
        "  registered, the writer sets nothing and the last valid weights stay on chain.",
        f"- One writer sets the weights, with one attempt each {params.weights_cadence_s} s. If the chain read fails or is not complete, or",
        "  the state does not load, that attempt sets nothing and the last valid weights stay on chain.",
        "",
        *(rows_table(kings, KING_COLUMNS) if kings else ["There is no king yet."]),
        "",
        "## 8. The IRT snapshot",
        "",
        "The task table of the sampler is public for every round. It is one JSON object for each snapshot id:",
        f"`{keys['irt_snapshot']}`. mining.json names the id (`irt_snapshot_id`); it does not hold the table.",
        "",
        *(where or ["- No round has used a snapshot yet. The first round names its snapshot in section 3."]),
        f"- Model: {MODEL}. The object holds the ability of each head and LoRA, and for each task: {show(TASK_COLUMNS)}.",
        f"- Mass rule: {MASS_RULES[params.irt_draw_rule]}.",
        f"- Draw rule: {DRAW_RULE}.",
        "- A round names its snapshot (`irt_snapshot_id`) and its seed (`round_seed`) in section 3, in `history` and in",
        "  its round file. With the snapshot object and the round seed you can rebuild the draw of the round.",
        "",
        "## 9. Objects and how to verify them",
        "",
        *table(("object", "key"), keys.items()),
        "",
        "- A file set (base, fp32 master, LoRA) has the id sha256(canonical JSON of {file name: sha256}). Check each",
        "  file hash, then the id. The fp32 master has its own prefix: do not put it in the directory of the base.",
        "- `version_id` of a head is the sha256 of the canonical JSON of the head object without `version_id`.",
        "- Only a winner has a public batch, and with it public `train` values. For every other entry the round file",
        "  shows the decision, batch sha256 and available scores. `duplicate` and `null_update` are charged and cannot",
        "  win or be paid. Duplicate clusters publish only the identity and comparison fields listed in section 5; they",
        "  never publish batch content.",
        "- In a round file each entry has a `reason`. The round has a `note`, and a `reason` when it has a result. The",
        f"  `note` is empty unless the round had no result or has the outcome `{ABANDONED}` (section 4). These texts give a",
        "  cause in words, for a person to read: the class of an error and its message. Before the validator",
        f"  publishes such a text, a marker replaces each part that it takes for private (`{CUT_SECRET}`,",
        f"  `{CUT_LOGIN}`, `{CUT_QUERY}` for the query string of a link, `{CUT_ADDRESS}` for an IP address), and a",
        f"  text of more than {TEXT_LIMIT} characters is cut there and ends with a note of how many characters were cut. The",
        "  same rule applies to the `reason` of a round in mining.json and in section 3. To decide what applies to your",
        "  seat, read `outcome` and `status`, not the words of a text.",
        *(
            [
                "- The LoRA of a winner has a download link while it is stacked on the head, and after its fold for as long",
                "  as a rebuild of the head needs it: until a newer head is a checkpoint (section 1).",
            ]
            if doc.get("head_chain")
            else [
                "- The LoRA of a winner has a download link only while it is stacked on the head (section 1). After the fold",
                "  (section 6) it is part of the new base, and there is no link for the LoRA alone.",
            ]
        ),
        "",
        *_replicate_lines(doc, params),
    ]


def _replicate_lines(doc: Mapping[str, Any], params: Params) -> list[str]:
    """Section 10: the local train-and-score tool, ``python -m sn120_v2.replicate``.

    The publisher does not import that module (it needs the GPU install). A test parses each command of this
    text with the parser of the tool, and checks that the document and the backend function exist.
    """
    mining = f"{doc['contract']['public_url'] or '<the public address>'}/{layout.MINING_JSON.rsplit('/', 1)[-1]}"
    tool = f"python -m {REPLICATE_MODULE}"
    return [
        "## 10. Train and score your batch at home",
        "",
        f"`{tool}` takes your file, checks it as the validator does, trains the LoRA from head M, scores the",
        "LoRA and null M on the tasks of a round, and judges the result with the win rule. It has no training code and",
        "no scoring code of its own: it runs the code of the box (the procedures of section 2 and of section 6) on your GPUs. So",
        "the GPU backend is published: it is the code under `sn120_v2/gpu/`, and this tool builds it for a machine that",
        f"is not the box. The document of the tool is `{REPLICATE_DOC}` in the code: {REPO}/blob/v2/{REPLICATE_DOC}. It has",
        "every argument, the fields of the report, and the complete list of what differs from the box.",
        "",
        "- Install, in `v2/` of a copy of the code (section 2): `uv pip install -e \".[gpu,envs,miner]\"`. The",
        "  document has the complete steps. Run each command below in `v2/`. N is the number of your GPUs.",
        f"- `{tool} compare --mining-json {mining} --gpus <N>` prints what differs between your machine and the box.",
        "  It starts nothing and needs no GPU.",
        f"- `{tool} fetch --mining-json {mining} --work <directory>` gets the base files of head M through the signed",
        "  links of section 1 and checks each file against the head manifest.",
        f"- `{tool} run --mining-json {mining} --hotkey <ss58> --work <directory> --gpus <N> batch.json` checks the",
        "  file, trains, scores and judges. It prints one report. With no other argument it takes the seed and the",
        "  draw of the open round, or of the last closed round when none is open.",
        "- What differs from the box. The report of `run` names each difference under `differences`. Your install:",
        "  the list of the packages of the box, with their versions, is not published, so in practice your install has",
        "  another digest than the pinned one, and its versions of torch and vLLM can differ. The number of GPUs, when it is not",
        f"  {params.n_gpus}. The time limits of a round: they do not apply unless you ask for them. The round has one batch, yours:",
        "  `winner` in the report says that your LoRA beats null M by the win rule, not that it wins a real round,",
        "  where the eligible LoRA with the highest delta wins. For such a difference the tool changes that one value",
        "  in a local copy of the params. Each training value, each scoring value and each rule of the decision stays",
        "  that of mining.json.",
        "- What a run cannot repeat exactly: the text of a rollout, and so its outcome code, and so `delta` and `z`;",
        "  and the trained values of the LoRA, and so `lora_hash`. These depend on the GPU type, on the package",
        "  versions and on the requests that a server computes together. Expect the same pass rate within the noise",
        "  of sampling, not the same text for the same seed. What a run repeats exactly: the draw with its order and",
        "  its weights, each seed, the prompt and the settings of each request, the rows that train with their",
        "  advantages and tokens, the grade of a given text, and the decision for given outcome strings.",
        "- What is verified. The box trains and scores its rounds on GPUs with the code that `run` calls. The tool",
        "  itself has not run on a real GPU yet. Tests run it on a CPU, with a stand-in for the training process, for",
        "  the vLLM servers and for the grader. So it is not known which GPU is large enough, how long a run takes, or",
        "  how close null M on your machine is to null M on the box. With `--round-file` and the round file of a closed",
        "  round (section 9), the report compares the two, task by task.",
        "- The miner tool can call the same code: `python v2/miner/affine_miner.py reproduce --mining-json",
        "  mining.json --hotkey <ss58> --irt-snapshot irt.json --round-seed <seed> --work <directory> --backend",
        f"  {REPLICATE_MODULE}:{REPLICATE_BACKEND} batch.json`. This is the code of the box with the params as they are: the same",
        "  trainer, evaluator, round runner and judging. It runs only on a machine with the pinned install, the pinned",
        "  code and the GPU count of the params, so in practice only on the box. It needs the base files: run `fetch`",
        f"  with the same `--work` directory first. On any other machine use `{tool} run`.",
        "- `python v2/miner/affine_miner.py bounds --mining-json mining.json` prints the bounds of section 2.",
    ]


def _weight_lines(doc: Mapping[str, Any], where: str) -> list[str]:
    """Section 1: where the weight files of head M are. The second form is for a box that keeps the folded weight files."""
    chain = doc.get("head_chain")
    if not chain:
        return [
            "- Weight files are not served under the public address. Signed download links for the files of the current",
            f"  head (the base, the stacked LoRAs and the fp32 master) are in {where}/{READ_LINKS}. Each link works until",
            "  `expires_at`; fetch the file again for new links. Check each file against its `sha256` and `size`.",
            "  `head_version_id` in that file names the head of the links. To mine you need the `base` and the `loras`.",
            "  You do not need the `master`: it is only for a check of a fold.",
        ]
    start, steps = chain["start"], chain["steps"]
    kernels = ", ".join(f"`{kernel}`" for kernel in sorted({str(step["kernel"]) for step in steps}))
    behind = f"The base of head M is {len(steps)} fold(s) after it (kernel {kernels})." if steps else "The base of head M is that checkpoint."
    mining = f"{where}/{layout.MINING_JSON.rsplit('/', 1)[-1]}"
    return [
        "- Weight files are not served under the public address, and the store does not hold the weight files of every",
        "  head. It holds the full files (the base and the fp32 master) of each **checkpoint** head, and the LoRA of every",
        "  winner. The launch model is a checkpoint. You make the base of a head that is not a checkpoint on your",
        "  machine: start from a checkpoint and fold each LoRA after it, in order. The result is exact: each file must",
        "  have the sha256 that the head manifest names.",
        "- `head_chain` in mining.json has the newest checkpoint (`start`) and one entry in `steps` for each fold after",
        "  it: the LoRAs of the fold, its `kernel`, and the ids of the base and the fp32 master after it. The whole chain",
        f"  from the launch model is `{chain['file']}`.",
        f"- The newest checkpoint is head `{start['version_id']}`. {behind}",
        f"- Signed download links are in {where}/{READ_LINKS}. Each link works until `expires_at`; fetch the file again",
        "  for new links. Check each file against its `sha256` and `size`. `head_version_id` in that file names the head",
        "  of the links. `loras` has the LoRAs that are stacked on the head. `checkpoints` has the base and the fp32",
        "  master of the newest checkpoint, and `chain_loras` has the LoRAs of each fold after it. `base` and `master`",
        "  are the files of the base of head M when it is a checkpoint, and null when it is not.",
        f"- To get the base of head M: `python -m {REPLICATE_MODULE} fetch` (section 10), or `python -m {REBUILD_MODULE}",
        f"  rebuild --chain {mining} --work <directory>`. It downloads what the directory does not hold (the first",
        "  time: the fp32 master of the checkpoint, which has two times the bytes of the base), does each fold again,",
        "  checks every file, and leaves the base in `<directory>/cache/<base_hash>/`. After that, a new king costs one",
        "  LoRA download and one fold. Keep the directory: the next fold starts from the fp32 master in it.",
        f"- `kernel` names the rule for the arithmetic of the LoRA product. `{FIXED_ORDER}` gives the same bytes on each",
        f"  machine. `{MATMUL}` is the matrix product of torch on the CPU of the box, and another machine can give other",
        "  bytes: start from a checkpoint after such a fold. Stop when you do not know a kernel name.",
        f"- For the launch model (`{GENESIS_MODEL_REPO}`): {REBUILD_COST}",
        f"- `{FOLD_IMPLEMENTATION_ENV}=blas` in the environment of the rebuild uses the matrix product of torch, which",
        "  takes less time on a CPU. Its bytes are those of the rule only on some machines. The sha256 values decide:",
        "  a fold with other values is deleted, and the tool prints a line that starts with `refused: ` and names the",
        "  default (`reference`) as the way out.",
    ]


def _genesis_lines(head: HeadVersion) -> list[str]:
    """Where the base files are while the base is the launch model. Nothing when the base is a folded one."""
    if head.base_hash != GENESIS_BASE_HASH:
        return []
    return [
        f"- The base of this head is the genesis model: the files of `{GENESIS_MODEL_REPO}` on Hugging Face at revision",
        f"  `{GENESIS_MODEL_REVISION}`. Get the base files there. `base_hash` is the id of that file set (section 9)."
        + ("" if head.parent_version_id else " No winner is applied yet: head M is this model as it is."),
    ]


def llms_text(doc: Mapping[str, Any]) -> str:
    """llms.txt from a mining.json document. Pure: the same document always gives the same text."""
    params, head = Params.from_dict(doc["params"]), HeadManifest.from_dict(doc["head"]).head
    loras = [(ref.lora_hash, ref.round_id, ref.miner_uid, ref.batch_sha256) for ref in head.loras]
    numbers = [(f"`{f.name}`", getattr(params, f.name)) for f in dataclasses.fields(params)
               if not f.name.startswith("_") and f.name != "recipe" and not (f.name in Params.LEFT_OUT_AT_DEFAULT and (f.name == "null_samples_per_task" or f.name.startswith("rotation_")) and getattr(params, f.name) == f.default)]
    url = doc["contract"]["public_url"]
    where, pair = url or "<the public address>", [key.rsplit("/", 1)[-1] for key in (layout.LLMS_TXT, layout.MINING_JSON)]
    lines = [
        "# Affine (Bittensor SN120) v2: the miner contract",
        "",
        f"Generated {utc(doc.get('generated_at'))} from snapshot `{doc['snapshot_digest']}`.",
        "The machine-readable companion is mining.json. It has the same `snapshot_digest`. If the two digests differ,",
        "an update is in progress: fetch both files again. These two files are the whole miner contract. The document",
        "of the local tool (section 10) explains a tool; it sets no rule.",
        "",
        "You upload one JSON file per hotkey: prompts, rollouts and rewards. At round start the validator copies it,",
        "checks it, and trains one bounded LoRA from the current head M with the published recipe. Every LoRA of the",
        *([
            "round and a null copy of M are scored on the same public tasks. Sample 0 of each uses the same seed; later king",
            "samples are king-only. At most one LoRA wins,",
        ] if params.null_samples_per_task is not None else [
            "round and a null copy of M are scored on the same public tasks with the same seeds. At most one LoRA wins,",
        ]),
        f"and only if it beats M by more than {show(params.z_bar)} standard error. The winner is applied to the head and",
        "its miner becomes a king. Only kings earn.",
        "",
        "## 1. Network and head",
        "",
        f"- Network: Bittensor Finney, netuid {params.netuid}. Environment: `{params.env}` (pin `{show(params.env_pin)}`).",
        f"- Head M `version_id`: `{head.version_id}` (parent `{show(head.parent_version_id)}`)",
        f"- `base_hash`: `{head.base_hash}`; fp32 master digest: `{head.base_fp32_master_digest}`",
        f"- Made by round `{head.round_id}` at {utc(head.applied_at)}.",
        *_genesis_lines(head),
        f"- Head manifest (every file with its sha256): `{layout.head_manifest(head.version_id)}`",
        f"- `params_hash`: `{params.params_hash}`; `recipe_id`: `{params.recipe_id}`",
        f"- Public objects are served under: {url or 'not set in this file'}. Each key below is a path under it, except",
        "  the weight files.",
        *_weight_lines(doc, where),
        f"- This file and mining.json are also at {where}/{pair[0]} and {where}/{pair[1]}. There the two files are",
        "  always a matching pair.",
        "- Send a client name of your own (the HTTP header `User-Agent`) with each read of a file under",
        f"  `{READ_LINKS.split('/')[0]}/` and of a link file (section 5). The site can answer HTTP 403 to a read with the default client",
        "  name of Python's `urllib` (`Python-urllib/<version>`), also for a file that exists. curl, wget and the Python",
        "  package `requests` send a name that the site accepts. HTTP 403 does not say that a file is missing: with an",
        "  accepted name a missing file gives HTTP 404. Each command that this file names sends a name of its own.",
        "",
        *(table(("stacked LoRA", "round", "miner uid", "batch sha256"), loras) if loras else ["No LoRA is stacked on the base."]),
        "",
        *_recipe_lines(params),
        "",
        *_state_lines(doc, params),
        "",
        *_file_lines(doc, params),
        "",
        *_cluster_lines(doc.get("round"), "Open round"),
        *_judging_lines(doc, params),
        "",
        f"## 11. {PARAMS_HEADING}",
        "",
        "Each value is pinned into a round when the round starts. The `train_seed_scope` formula is in section 2.",
        "The duplicate and null-update parameters are explained in `Duplicate and null batches` (section 5), and the",
        "machine-readable `contract.duplicate_policy` block repeats their thresholds and public evidence schema.",
        "",
        *table(("parameter", "value"), numbers),
        "",
        "- `eval_time_budget_s`: the longest time that the box waits for one answer of a model in scoring. It is not the",
        "  time of the scoring: the times of a round are in section 3.",
        "- `burn_uid`: the UID of the owner. It has no seat (section 4) and gets no weight: nothing is burned (section 7).",
        "",
    ]
    return "\n".join(reflow(lines)) + "\n"


def render(snapshot: Mapping[str, Any], public_url: str = "", links_url: str = "") -> tuple[bytes, bytes]:
    """(mining.json bytes, llms.txt bytes) of one snapshot."""
    doc = mining_document(snapshot, public_url, links_url)
    mining = json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    return mining.encode("utf-8"), llms_text(doc).encode("utf-8")


SNAPSHOT_NAMED = re.compile(r"from snapshot\s+`([0-9a-f]{64})`")
PARAM_ROW = re.compile(r"^\| `(\w+)` \| (.*) \|$", re.MULTILINE)


def stated_params(text: str) -> dict[str, str]:
    """Each parameter with the value that a llms.txt text states for it, from the table of the section "Every parameter"."""
    section = re.split(rf"^## \d+\. {PARAMS_HEADING}$", text, flags=re.MULTILINE)
    return dict(PARAM_ROW.findall(section[1])) if len(section) == 2 else {}


def check_pair(mining: bytes, llms: bytes, live: Params | None = None, whole_text: bool = True) -> list[str]:
    """What is wrong with a published pair of mining.json and llms.txt. An empty list: llms.txt is true to the params.

    Four checks, each on the bytes as a reader gets them:

    1. mining.json is whole: its content has the ``snapshot_digest`` that it names, and its params have the
       ``params_hash`` and the ``recipe_id`` that it names.
    2. llms.txt is the text of that mining.json: it names the same digest, the same ``params_hash`` and the same
       ``recipe_id``, and its parameter table states each parameter of mining.json with the same value.
    3. With ``live`` (the params of the validator's state): they are the published params.
    4. With ``whole_text``: llms.txt is, byte for byte, the text that this code renders from that mining.json. A
       difference means that other code published the files, or that a file was changed after it was written.
    """
    try:
        doc, text = json.loads(mining), llms.decode("utf-8")
        params, named = Params.from_dict(doc["params"]), doc["contract"]
        whole = digest({key: value for key, value in doc.items() if key not in OWN_KEYS}) == doc["snapshot_digest"]
    except (ValueError, KeyError, TypeError, V2Error) as exc:
        return [f"mining.json or llms.txt cannot be read: {type(exc).__name__}: {exc}"]
    problems = [] if whole else ["mining.json: the content does not have the `snapshot_digest` that the file names"]
    for name, real in (("params_hash", params.params_hash), ("recipe_id", params.recipe_id)):
        if not real == doc.get(name) == named.get(name):
            problems.append(f"mining.json: `{name}` is not the id of its own params ({real})")
        if f"`{real}`" not in text:
            problems.append(f"llms.txt does not name the `{name}` of the params in mining.json ({real})")
    found = SNAPSHOT_NAMED.search(text)
    if found is None or found.group(1) != doc["snapshot_digest"]:
        problems.append(f"llms.txt and mining.json are not a pair: llms.txt names snapshot {found and found.group(1)}, mining.json {doc['snapshot_digest']}")
    stated = stated_params(text)
    values = {f.name: show(getattr(params, f.name)) for f in dataclasses.fields(params)
              if not f.name.startswith("_") and f.name != "recipe" and not (f.name in Params.LEFT_OUT_AT_DEFAULT and (f.name == "null_samples_per_task" or f.name.startswith("rotation_")) and getattr(params, f.name) == f.default)}
    for name in sorted(values.keys() | stated.keys()):
        if stated.get(name) != values.get(name):
            problems.append(f"`{name}`: llms.txt states {stated.get(name, 'no value')}, mining.json has {values.get(name, 'no such parameter')}")
    if live is not None and live.params_hash != params.params_hash:
        ours, theirs = to_jsonable(live.to_dict()), to_jsonable(params.to_dict())
        names = sorted(key for key in ours.keys() | theirs.keys() if ours.get(key) != theirs.get(key))
        problems.append(f"the published params are not the params of the state: {', '.join(names)} differ")
    if whole_text:
        try:
            ours = llms_text(doc)
        except Exception as exc:  # a document of other code can lack anything; the cause is named, never raised
            return problems + [f"this code cannot render the published mining.json: {type(exc).__name__}: {exc}"]
        if ours != text:
            theirs, mine = text.splitlines(), ours.splitlines()
            changes = [op for op in difflib.SequenceMatcher(None, theirs, mine, autojunk=False).get_opcodes() if op[0] != "equal"]
            _, i1, i2, j1, j2 = changes[0]
            first = f'published "{" ".join(theirs[i1:i2])[:200]}", this code "{" ".join(mine[j1:j2])[:200]}"'
            count = sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in changes)
            problems.append(f"llms.txt is not the text that this code renders from mining.json: {count} lines differ. The first, at line {i1 + 1}: {first}")
    return problems


def fetch(location: str) -> bytes:
    """The bytes of a file or of an http(s) address, as any reader gets them."""
    if not location.startswith(("http://", "https://")):
        return Path(location).read_bytes()
    request = urllib.request.Request(location, headers={"User-Agent": "sn120-v2-publisher-check", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=30) as reply:
        return reply.read()


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m sn120_v2.publisher check``: exit 0 when the published llms.txt is true to the params, 1 when not."""
    parser = argparse.ArgumentParser(prog="python -m sn120_v2.publisher", description="Check a published llms.txt against the params.")
    check = parser.add_subparsers(dest="command", required=True).add_parser("check", help="compare llms.txt with mining.json and the params")
    check.add_argument("--url", help="the public address: the two files are read from <url>/llms.txt and <url>/mining.json")
    check.add_argument("--mining-json", help="a mining.json file or address, in place of --url")
    check.add_argument("--llms-txt", help="a llms.txt file or address, in place of --url")
    check.add_argument("--state-dir", help="the state directory of the validator: its params must be the published params")
    check.add_argument("--values-only", action="store_true", help="do not compare the whole text: for a host whose code is not the code that publishes")
    check.add_argument("--tries", type=int, default=3, help="fetch again this many times when the two files are not a pair (an update is in progress)")
    check.add_argument("--wait-s", type=float, default=5.0, help="seconds between two tries")
    args = parser.parse_args(argv)
    names = [key.rsplit("/", 1)[-1] for key in (layout.MINING_JSON, layout.LLMS_TXT)]
    places = [args.mining_json, args.llms_txt] if args.mining_json and args.llms_txt else [f"{(args.url or '').rstrip('/')}/{name}" for name in names]
    if not (args.url or (args.mining_json and args.llms_txt)):
        parser.error("give --url, or both --mining-json and --llms-txt")
    try:
        live = None
        if args.state_dir:
            live = load(Path(args.state_dir) / JOURNAL_FILE).params
            if live is None:
                raise ConfigError(f"the state in {args.state_dir} holds no params")
        for attempt in range(max(1, args.tries)):
            mining, llms = (fetch(place) for place in places)
            problems = check_pair(mining, llms, live, whole_text=not args.values_only)
            if not any("are not a pair" in problem for problem in problems):
                break
            time.sleep(args.wait_s if attempt + 1 < args.tries else 0)
    except (OSError, V2Error) as exc:
        print(f"the check did not run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if problems:
        print(f"DIVERGED: {places[1]} against {places[0]}:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    doc = json.loads(mining)
    scope = "every stated parameter" if args.values_only else "every line"
    print(f"ok: {places[1]} is true to the params of {places[0]} ({scope}; params_hash {doc['params_hash']}; generated {utc(doc.get('generated_at'))})")
    return 0


class StorePublisher:
    """The ``Publisher`` seam: writes both files to the public zone of an object store.

    ``public_url`` is the address under which the public zone is served. ``links_url`` is the public address of
    the upload link files (``python -m sn120_v2.links issue --links-url``). Each is "" until it exists.
    """

    def __init__(self, store: ObjectStore, public_url: str = "", links_url: str = "") -> None:
        self.store = store
        self.public_url = public_url.rstrip("/")
        self.links_url = links_url.rstrip("/")
        if self.links_url:
            upload_entry(self.links_url)  # a bad address stops the start, not each publish

    def publish(self, snapshot: Mapping[str, Any]) -> None:
        """Render both files first, so a bad snapshot changes nothing. Then write and read back each one.

        A store that is behind or fails raises ``InfraError``: the controller keeps running and publishes again.
        """
        mining, llms = render(snapshot, self.public_url, self.links_url)
        head_key = layout.head_manifest(snapshot["head"]["head"]["version_id"])
        try:
            if not self.store.exists(head_key):
                raise InfraError(f"head manifest {head_key} is not public yet; the contract may not name this head")
            for key, data in ((layout.MINING_JSON, mining), (layout.LLMS_TXT, llms)):
                self.store.put(key, data)
                if self.store.sha256(key) != sha256_hex(data):
                    raise InfraError(f"{key} does not read back as written")
        except V2Error:
            raise
        except Exception as exc:  # any store failure is ours: the controller retries the same snapshot
            raise InfraError(f"cannot publish the miner contract: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
