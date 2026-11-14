"""The publisher: llms.txt and mining.json, written together from one public snapshot.

llms.txt is the only text a miner or a miner's agent needs. mining.json is
its one machine-readable companion. Both are pure functions of the same
snapshot, and both carry the same ``snapshot_digest``, so a reader can tell
that the two files belong together.

Every number in the text comes from the ``Params`` in the snapshot. The
miner-settable bounds come from ``Recipe.miner_bounds`` and the object keys
from ``layout.py``. No cap, bound or key is typed here. The one exception is
``READ_LINKS``: the site pull job owns that file, and no module names it yet.

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
from .admission import CONTROL_TOKENS, FAULT_HOLD_S, FAULT_TRIES, FORMATS, NO_SIGNAL, RULES, TIME_LIMIT, TRAIN_COST, TRANSIENT
from .canonical import digest, sha256_hex, to_jsonable
from .envs.math_pool import ENV_ID, MANDATE, POOL_FILE, POOL_SHA256, Task
from .errors import ConfigError, ContractError, InfraError, V2Error
from .gpu.batch import END_OF_TURN
from .gpu.evaluator import ServeSettings, chat_body, server_argv
from .gpu.trainer import LORA_FILES
from .irt import DRAW_RULE, MODEL, TASK_COLUMNS, draw_from_snapshot
from .journal import JOURNAL_FILE
from .judging import TIE_TOLERANCE, drop_margin
from .links import upload_entry
from .params import GENESIS_BASE_HASH, GENESIS_MODEL_REPO, GENESIS_MODEL_REVISION, MinerBound, Params, Recipe
from .public import CUT_ADDRESS, CUT_LOGIN, CUT_QUERY, CUT_SECRET, TEXT_LIMIT
from .queue import SECONDS_PER_DAY
from .sampling import SEED_BITS
from .seams import ObjectStore
from .state import ABANDONED, load
from .types import CAP, ERROR, FAIL, NO_DECISION, NOT_RUN, NOT_SCORED, NOT_TRAINED, NULL_MODEL, PASS, SKIP, HeadManifest, HeadVersion
from .worker import AT_THE_LEASE_S, LocalRunner

REPO = "https://github.com/AffineFoundation/affine"
# The site pull job (``v2/site/pull.py``, ``LINKS``) writes this file on the web host, under the public address. It
# is not a store key, and the publisher never writes it: it holds signed download links for the weight files of
# the current head. The pull is a command outside this package, so the path is typed here; a test compares the two.
READ_LINKS = "public/read_links.json"
# The local train-and-score tool. The publisher names it and does not import it: the module needs the GPU install.
REPLICATE_MODULE, REPLICATE_BACKEND, REPLICATE_DOC = "sn120_v2.replicate", "backend", "v2/REPLICATE.md"
LORA_TABLE = "<batches of the round>"  # what llms.txt prints for --max-loras
LOCAL_OPTIONS = ("--host", "--port")  # options of the vLLM server that llms.txt does not print: the local address of the box
# Two texts of modules that the publisher may not import: the training job needs torch and the grader needs sympy,
# and the publisher runs with the standard library alone. Each is typed here; a test compares it with its source.
OPTIMIZER = "adam, fresh per LoRA, betas 0.9/0.999, eps 1e-8, weight decay 0"  # gpu/train_job.py
GRADER_VERSION = "math-verify-0.9.0+lenient-v1"  # envs/math_grader.py
# How often the box tries a training or a chunk of the scoring that fails for a reason of its own.
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
PARAMS_HEADING = "Every parameter"  # the section whose table states each parameter; ``stated_params`` reads it back

RECIPE_NOTES = {
    "method": "GRPO on the LoRA weights only; the base weights are frozen",
    "steps": "optimizer steps over your batch",
    "lora_lr": "learning rate of each step",
    "max_grad_norm": "gradient clip: with steps and lora_lr, this is the step-size cap",
    "global_norm_cap": "optional cap on the norm of the whole adapter update",
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
        return "a list: a non-empty subset of " + ", ".join(bound.choices)
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
    return {
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
        "outcome_codes": {"pass": PASS, "fail": FAIL, "token_cap": CAP, "infra_error": ERROR, "not_run": NOT_RUN},
        "null_model": NULL_MODEL,
        "public_url": public_url,
        "links_url": links_url,
        "upload": upload_entry(links_url) if links_url else None,
        "objects": object_keys(),
        "read_links": READ_LINKS,
        "code": REPO + "/tree/v2/v2",
    }


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
    cap = recipe.global_norm_cap
    fields = [f.name for f in dataclasses.fields(recipe) if f.name != "miner_bounds"]
    rows = [(f"`{name}`", getattr(recipe, name), RECIPE_NOTES.get(name, "see the training code")) for name in fields]
    settable = [(f"`train.{row['key']}`", row["default"], row["allowed"]) for row in miner_settable(recipe)]
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
        "- `sample_temperature` T: you state the temperature at which you sampled your rollouts. The trainer divides the",
        "  logits of the head by T before it scores the tokens of a rollout. Evaluation always samples at temperature",
        f"  {show(params.temperature)}. The default T is {show(recipe.sample_temperature)}.",
        "- `global_norm_cap`: the largest allowed Frobenius norm of the whole adapter update,",
        "  sqrt(sum over all target modules of ||(alpha / rank) * B @ A||_F^2), in the units of the model weights. If the",
        "  trained update is larger, every B matrix is multiplied by cap / norm. The train report states the norm before",
        "  the cap and the norm after it (`step_norm` is the norm after the cap). The number is set in the params at",
        f"  the mainnet genesis; unset means no cap. {'It is unset now: there is no cap.' if cap is None else f'The cap is now {show(cap)}.'}",
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
        f'- Seed: the first {SEED_BITS} bits of sha256("<round_seed>|train|<batch_sha256>"), as a whole number. The train',
        "  report states it.",
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
        "  never trains.",
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
        "  in its mask. A row with an empty mask gives no loss. Step loss = the sum of the row losses / the number of",
        "  rows. That number counts each row that was not dropped, also a row with advantage 0.",
        f"- Steps: {recipe.steps} (`steps`). One step is one pass over all rows that train and then one update. The gradient of the",
        f"  step loss, over all A and B together, is clipped to the norm {show(recipe.max_grad_norm)} (`max_grad_norm`). Then comes one update",
        f"  with the learning rate {show(recipe.lora_lr)} (`lora_lr`). The optimizer: {OPTIMIZER}. The learning rate has no",
        f"  schedule. A gradient that is not finite stops the training: `{SKIP}`.",
        "- GPUs: the job has the GPUs of its slot (section 3). It puts one copy of the model on each GPU, but not more",
        "  copies than rows that train. The rows that train go to the copies in turn, the longest row first. The",
        "  gradients of the copies are added before the clip. The job never trains on a CPU: with no GPU it stops.",
        "- After the last step: the cap `global_norm_cap` (above), then a check that each weight is finite. If one is",
        f"  not: `{SKIP}`.",
        f"- The LoRA is these files: {show([f'`{name}`' for name in LORA_FILES])}. The weight file holds A and B of each layer as",
        "  float32, with the names `base_model.model.<layer path>.lora_A.weight` and",
        "  `base_model.model.<layer path>.lora_B.weight`. The config file has the PEFT layout; its",
        "  `base_model_name_or_path` is the `base_hash`. The train report holds the optimizer, the seed, the recipe",
        "  values that were used, the loss and the gradient norm of each step, the count of the mask tokens of the rows",
        "  that trained, the count of the rows, of the truncated rows and of the dropped rows, `step_norm` and",
        "  `step_norm_before_cap`. `lora_hash` is the id of this file set (section 9).",
        f"- A failure of the box in a training (not of the batch) is tried again: at most {BOX_TRIES} tries in all, inside the",
        f"  training time. Then the entry is `{NO_DECISION}` (section 4).",
        "- What a training of your own repeats: the seed gives the same A for the same file and round seed. The trained",
        "  B can differ between two machines, and between two runs on one machine; no run has measured by how much.",
        "  The tests of the code run this training on a CPU, on a very small model. The box runs it on GPUs.",
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
        f"- Training time: the first {show(train_end / 60)} minutes. Batches train in queue order. Each batch trains on its own slot of",
        f"  GPUs, and the slots train at the same time. A slot is floor({params.n_gpus} / batches) GPUs, and at least 1. So with",
        f"  {params.n_gpus} batches or more, each batch trains on one GPU, {params.n_gpus} batches at a time, and each other batch waits for a",
        "  free GPU.",
        "- At the end of the training time no new training starts, and each training that still runs is stopped. Such",
        f"  a batch is `{NOT_TRAINED}`: you are not charged, your seat is first in the next round, and no failure is counted",
        "  for the seat. Scoring starts when all training is done or stopped.",
        "- One exception: a batch that started at the lease had the whole training time. If the first try of its",
        f"  training did not finish in that time, that is a failure of ours: `{NO_DECISION}`. Section 4 says when such a",
        "  failure is counted for the seat. If a try of its training failed before (a failure of the box) and a later try",
        f"  is stopped at the end of the training time, the batch is `{NOT_TRAINED}` as above: the box took a part of its time.",
        f"- The box can start its work on a round later than {AT_THE_LEASE_S} s after the start of the lease, for example when the",
        "  validator's program started again in the round. Then no batch had the whole training time: a batch that",
        f"  does not finish is `{NOT_TRAINED}` as above, and no failure is counted for the seat.",
        f"- Scoring runs in chunks of {params.eval_chunk_tasks} tasks. Null M and every open LoRA answer each chunk. An open LoRA is a",
        "  LoRA that was not dropped at a look (section 6).",
        "- The stop time cuts every look, the first one too. The box starts no new chunk after the stop time, and a",
        "  chunk that is not finished at the stop time is not used. There are no further looks.",
        f"- At the stop time, with the first look ({first_look} tasks) complete: each open LoRA is judged by the win rule of",
        "  section 6 on the whole chunks that null M and every open LoRA finished. Those tasks are then the whole",
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
        "- Round loop: (1) pin head M, the recipe, the task draw and the public round seed; (2) copy the next valid",
        "  files in queue order; (3) train one LoRA per file; (4) score every LoRA and a null copy of M (no adapter)",
        "  on the same tasks and seeds; (5) apply at most one winner; (6) publish; (7) start again.",
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
        "  before the end of the lease, or when the operator ended the round. Such a round costs you nothing: nobody is charged, no failure is counted for a",
        f"  seat, and each seat of the round goes to the front of the queue. The next round starts {duration(params.apply_retry_s)} after the",
        f"  failure, or later. In the round file (section 9) you see it thus: `outcome` is `{NO_DECISION}`, no entry has a",
        "  `status`, and `note` gives the cause. In `history` of mining.json the round has the same `outcome` and its",
        "  entries have no `status`; the `note` is only in the round file.",
        f"- A round that has a result also gets the outcome `{NO_DECISION}` when none of its entries was charged. Each entry",
        "  then has a `status`, and the status of your entry says what applies to your seat.",
        f"- `{NOT_TRAINED}`: the training time of the round ended before your batch was trained (section 3). You are not",
        "  charged and you go to the front of the queue. It is not counted as a failure, so it never holds a seat.",
        f"- `{NOT_SCORED}`: the time limit of the round came before the first look of the scoring (section 3). You are",
        "  not charged and you go to the front of the queue. It is not counted as a failure, so it never holds a seat.",
        f"- `{SKIP}`: the trainer could not train the copy of your file, for example because the training gave a number",
        "  that is not finite. You are not charged and you keep your place.",
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
    return [
        f"- Task pool: `sn120_v2/envs/{POOL_FILE.name}` in the code (sha256 `{POOL_SHA256}`). Scoring sends this",
        f'  instruction as the system message and the problem as the user message: "{MANDATE}"',
        "- A prompt of your file is a pool task when it is the problem text, with or without that instruction and one",
        "  blank line before it. The official grader is `sn120_v2.envs.math_grader.grade(task, output, capped)`. It",
        "  grades model outputs in scoring and the rewards in your file.",
    ]


def _server_command(params: Params) -> list[str]:
    """The command of one vLLM server (``gpu.evaluator.server_argv``), one option on a line.

    The address of the server is left out: it is local to the box and changes no score.
    """
    argv = server_argv("<directory of the base files>", 0, params, ServeSettings().clients_per_server)
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
        "This is everything that the box does to score a round, in the order in which it does it. The code is in four",
        "files: `sn120_v2/worker.py` (chunks, looks and the clock), `sn120_v2/gpu/evaluator.py` (servers, requests and",
        "outcome codes), `sn120_v2/envs/math_grader.py` (the grader) and `sn120_v2/judging.py` (the estimator and the",
        "rules). Section 10 has the command that runs this code on your machine.",
        "",
        f"- Servers: when all training is done, the box starts one vLLM server on each of its {params.n_gpus} GPUs. Each server",
        f"  holds the base of head M under the model name `{NULL_MODEL}`. The box loads each LoRA of the round into every",
        "  server with `/v1/load_lora_adapter`, with its `lora_hash` as the model name. The servers run with the",
        "  variable `VLLM_ALLOW_RUNTIME_LORA_UPDATING` set (vLLM needs it for that route) and with `HF_HUB_OFFLINE` set.",
        "  The vLLM version is the one that the `gpu` extra of `v2/pyproject.toml` names. `--max-loras` is the number of",
        "  batches of the round, trained or not: it is the same for every chunk of the round. The command of one server,",
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
        f"  Task i of a chunk (i counts from 0) goes to server i mod {params.n_gpus}, for every model. So a LoRA and null M differ",
        "  by the adapter alone.",
        "- The box sends all requests of a chunk at once. A server runs at most `--max-num-seqs` of them at a time; the",
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
        f"- Chunks: the tasks run in the order of the draw, in chunks of {params.eval_chunk_tasks} tasks (`eval_chunk_tasks`). A chunk also",
        "  ends where a look ends. Null M and each open LoRA answer every task of a chunk. The box reads the clock",
        "  before each chunk (section 3).",
        "- A failure of the box in a chunk (a server died, a LoRA did not load, the grader failed) is not an outcome.",
        f"  The chunk runs again: at most {BOX_TRIES} runs in all. When each of them fails before the stop time, the box stops with an",
        "  error and the round has no result (section 4). A run that fails after the stop time ends the scoring, and",
        "  the chunk is not used.",
        "- Outcome strings: each model has one string for each sample. A string has one character for each task, in",
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
        "- Win rule, in full. A LoRA is eligible when all of these are true. It has an outcome for every task of the",
        "  draw, or, when the stop time ended the scoring, for every task that null M has an outcome for; a LoRA that",
        "  was dropped at a look does not. n is at least `min_scored_tasks`. z > `z_bar`. (its count of",
        f"  `{CAP}`) / rollouts <= max(`cap_hit_floor`, (the count of `{CAP}` of null M on the same n tasks) / rollouts +",
        "  `cap_hit_margin`), where rollouts = n x `samples_per_task`. This last test uses exact fractions of whole counts, and each parameter as the",
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


def _judging_lines(doc: Mapping[str, Any], params: Params) -> list[str]:
    kings, keys, url = doc.get("kings") or [], object_keys(), doc["contract"]["public_url"]
    snapshots = {"the open round": (doc.get("round") or {}).get("irt_snapshot_id"), "the last closed round": doc.get("irt_snapshot_id")}
    where = [
        f"- Snapshot of {name}: id `{found}`, object `{layout.irt_snapshot(found)}`" + (f", {url}/{layout.irt_snapshot(found)}" if url else "")
        for name, found in snapshots.items() if found
    ]
    looks, n = params.look_sizes(), params.n_tasks
    drops = [
        f"{params.z_bar * math.sqrt(size / n) - drop_margin(look, n, params):.3f} after {size} tasks"
        for look, size in enumerate(looks[:-1])
    ] or "no early look"
    return [
        "## 6. Judging",
        "",
        "- All tasks are public. There is no hidden task set.",
        *_pool_lines(params),
        f"- Draw: {params.n_tasks} tasks from the public pool of `{params.env}`, {params.samples_per_task} sample(s) per task.",
        "  The draw is a fixed function of the public sampler snapshot and the public round seed. Both are frozen at",
        f"  round start. Rebuild a draw with `{draw_from_snapshot.__module__}.{draw_from_snapshot.__name__}(snapshot,",
        "  round_seed, n_tasks)`.",
        f"- The IRT sampler favours tasks where the predicted pass rate of the head is from {show(params.irt_middle_band[0])}",
        f"  to {show(params.irt_middle_band[1])}, and keeps {show(params.irt_explore_frac)} of the draw for new tasks.",
        "- Pairing: the null copy of M and every LoRA run the same tasks with the same seeds. The seed of rollout j",
        f"  on a task is the first {SEED_BITS} bits of sha256(\"<round_seed>|<task_id>|<j>\"), as a whole number; j counts from 0.",
        f"- Decode: temperature {show(params.temperature)}, top_p {show(params.top_p)}, top_k {params.top_k}, {params.dtype},",
        f"  at most {params.max_output_tokens} output tokens and {params.max_context_tokens} prompt + output tokens.",
        f"- Outcome of a rollout: `{PASS}` pass, `{FAIL}` fail, `{CAP}` hit the token cap (not passed), `{ERROR}` our",
        "  infrastructure failed (not passed, stays in the count).",
        "- Estimator: d_t = LoRA pass rate - null M pass rate on task t. delta = sum(w_t * d_t) / sum(w_t), with the",
        "  published sampling weights w_t. se = sqrt(n / (n - 1) * sum((w_t * (d_t - delta))^2)) / sum(w_t). z = delta / se.",
        f"- Win rule: z is strictly above {show(params.z_bar)} (beat M by more than that many standard errors), on at least",
        f"  {params.min_scored_tasks} tasks, with no correction for the number of LoRAs. The cap-hit rate of the LoRA may",
        f"  not be above max({show(params.cap_hit_floor)}, null M cap-hit rate + {show(params.cap_hit_margin)}).",
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
        f"- After {params.fold_every_k} applied winner(s), the stacked LoRAs are folded into a new base.",
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
        "  shows the decision, the batch sha256 and the scores only.",
        "- In a round file each entry has a `reason`. The round has a `note`, and a `reason` when it has a result. The",
        f"  `note` is empty unless the round had no result or has the outcome `{ABANDONED}` (section 4). These texts give a",
        "  cause in words, for a person to read: the class of an error and its message. Before the validator",
        f"  publishes such a text, a marker replaces each part that it takes for private (`{CUT_SECRET}`,",
        f"  `{CUT_LOGIN}`, `{CUT_QUERY}` for the query string of a link, `{CUT_ADDRESS}` for an IP address), and a",
        f"  text of more than {TEXT_LIMIT} characters is cut there and ends with a note of how many characters were cut. The",
        "  same rule applies to the `reason` of a round in mining.json and in section 3. To decide what applies to your",
        "  seat, read `outcome` and `status`, not the words of a text.",
        "- The LoRA of a winner has a download link only while it is stacked on the head (section 1). After the fold",
        "  (section 6) it is part of the new base, and there is no link for the LoRA alone.",
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
    numbers = [(f"`{f.name}`", getattr(params, f.name)) for f in dataclasses.fields(params) if f.name != "recipe"]
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
        "round and a null copy of M are scored on the same public tasks with the same seeds. At most one LoRA wins,",
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
        "- Weight files are not served under the public address. Signed download links for the files of the current",
        f"  head (the base, the stacked LoRAs and the fp32 master) are in {where}/{READ_LINKS}. Each link works until",
        "  `expires_at`; fetch the file again for new links. Check each file against its `sha256` and `size`.",
        "  `head_version_id` in that file names the head of the links. To mine you need the `base` and the `loras`.",
        "  You do not need the `master`: it is only for a check of a fold.",
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
        *_judging_lines(doc, params),
        "",
        f"## 11. {PARAMS_HEADING}",
        "",
        "Each value is pinned into a round when the round starts.",
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
    values = {f.name: show(getattr(params, f.name)) for f in dataclasses.fields(params) if f.name != "recipe"}
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
