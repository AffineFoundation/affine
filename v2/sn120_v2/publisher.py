"""The publisher: llms.txt and mining.json, written together from one public snapshot.

llms.txt is the only text a miner or a miner's agent needs. mining.json is
its one machine-readable companion. Both are pure functions of the same
snapshot, and both carry the same ``snapshot_digest``, so a reader can tell
that the two files belong together.

Every number in the text comes from the ``Params`` in the snapshot. The
miner-settable bounds come from ``Recipe.miner_bounds`` and the object keys
from ``layout.py``. No cap, bound or key is typed here. The one exception is
``READ_LINKS``: the site pull job owns that file, and no module names it yet.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import textwrap
import time
from typing import Any, Iterable, Mapping, Sequence

from . import layout
from .admission import CONTROL_TOKENS, FORMATS, NO_SIGNAL, RULES, TIME_LIMIT, TRAIN_COST, TRANSIENT
from .canonical import digest, sha256_hex, to_jsonable
from .envs.math_pool import ENV_ID, MANDATE, POOL_FILE, POOL_SHA256
from .errors import ContractError, InfraError, V2Error
from .irt import DRAW_RULE, MODEL, TASK_COLUMNS, draw_from_snapshot
from .judging import drop_margin
from .links import upload_entry
from .params import GENESIS_BASE_HASH, GENESIS_MODEL_REPO, GENESIS_MODEL_REVISION, MinerBound, Params, Recipe
from .queue import SECONDS_PER_DAY
from .seams import ObjectStore
from .types import CAP, ERROR, FAIL, NO_DECISION, NOT_RUN, NOT_TRAINED, NULL_MODEL, PASS, HeadManifest, HeadVersion

REPO = "https://github.com/AffineFoundation/affine"
# The site pull job (``v2/site/pull.py``, ``LINKS``) writes this file on the web host, under the public address. It
# is not a store key, and the publisher never writes it: it holds signed download links for the weight files of
# the current head. The pull is a command outside this package, so the path is typed here; a test compares the two.
READ_LINKS = "public/read_links.json"
TRAIN_FORMATS = FORMATS[1:]  # the first format may not carry "train"
# Batch content and non-public store zones. A snapshot that holds one of these is refused.
PRIVATE_KEYS = frozenset({"groups", "rollouts", "put_url", "put_headers"})
PRIVATE_ZONES = ("private/", "pending/")
OWN_KEYS = ("snapshot_digest", "contract")
# A presigned upload link signs these headers (``Presigner.presign_put``). An upload without them is refused.
UPLOAD_HEADERS = {"Content-Type": "application/octet-stream"}
ENTRY_COLUMNS = ("queue_pos", "uid", "hotkey", "batch_sha256", "status", "delta", "se", "z", "cap_rate")

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


def utc(seconds: Any) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(seconds)) if isinstance(seconds, int) and seconds > 0 else "unknown"


def table(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    return lines + ["| " + " | ".join(show(cell) for cell in row) + " |" for row in rows]


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
        "round_scores": layout.round_scores("ROUND_ID"),
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
        "- A file with `train` keys that only v1 had (`lr`, `kl`, `step_mult`, the knobs) is invalid. The hotkey keeps",
        "  its place and is not charged until the miner uploads a new file.",
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
    ]


def _round_time_lines(params: Params) -> list[str]:
    """What the clock does to a round. Every word here follows ``worker.LocalRunner.run``.

    When the rule of the time limit changes, change this function. Two sentences elsewhere depend on it: the
    ``not_trained`` bullet of section 4 and the "time limit" bullet of section 6. Both refer to this section.
    """
    # The three times of worker.LocalRunner.run, in seconds from the lease.
    last = params.lease_timeout_s - params.promote_reserve_s
    stop = min(last, params.round_time_limit_s - params.promote_reserve_s)
    train_end = min(stop, params.train_time_budget_s)
    limit, reserve, first_look = show(params.round_time_limit_s / 60), show(params.promote_reserve_s / 60), params.look_sizes()[0]
    return [
        "### The time of a round",
        "",
        "All times count from the lease: the moment a round takes its entries. A round ends early, as soon as every",
        "LoRA is dropped or decided.",
        "",
        f"- Time limit: {limit} minutes from the lease to the promotion of the winner. The box stops {reserve} minutes",
        f"  before it, at minute {show(stop / 60)} (the stop time), so that judging and promotion fit.",
        f"- Training time: the first {show(train_end / 60)} minutes. Batches train in queue order. Each batch trains on its own slot of",
        f"  GPUs, and the slots train at the same time. A slot is floor({params.n_gpus} / batches) GPUs, and at least 1. So with",
        f"  {params.n_gpus} batches or more, each batch trains on one GPU, {params.n_gpus} batches at a time, and each other batch waits for a",
        "  free GPU.",
        "- At the end of the training time no new training starts, and each training that still runs is stopped. Such",
        f"  a batch is `{NOT_TRAINED}`: you are not charged, your seat is first in the next round, and no failure is counted",
        "  for the seat. Scoring starts when all training is done or stopped.",
        "- One exception: a batch that started at the lease had the whole training time. If it did not finish, that",
        f"  is a failure of ours: `{NO_DECISION}`.",
        f"- Scoring runs in chunks of {params.eval_chunk_tasks} tasks. Null M and every open LoRA answer each chunk. An open LoRA is a",
        "  LoRA that was not dropped at a look (section 6).",
        f"- The first look ({first_look} tasks) is always scored in full for null M and every trained LoRA. The time limit does",
        "  not cut it. So each trained LoRA gets a decision.",
        "- After the first look, the stop time applies. The box starts no new chunk after it, and a chunk that is not",
        "  finished at the stop time is not used. Each open LoRA is then judged by the win rule of section 6 on the",
        "  whole chunks that null M and every open LoRA finished. Those tasks are then the whole test: z strictly",
        f"  above {show(params.z_bar)} can win. Such an entry is charged, win or lose, as in a full round.",
        f"- So a round is longer than {limit} minutes only when the training time and the first look together are longer",
        "  than that.",
        f"- The lease ends {show(params.lease_timeout_s / 60)} minutes after it starts. If the first look is not complete {reserve} minutes before",
        f"  that, each trained LoRA is `{NO_DECISION}`. A box result that arrives after the end of the lease is never used:",
        f"  each entry of that box is `{NO_DECISION}`.",
        f"- `{NO_DECISION}` means: not charged, and the seat goes to the front of the queue (section 4).",
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
        *(f"- {key}: {show(value)}" for key, value in round_now.items() if not isinstance(value, (dict, list))),
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
        *(table(tuple(seats[0]), (row.values() for row in seats)) if seats else ["The queue is empty."]),
        "",
        "- `file_valid`: the validator's verdict on the file that was stored for the hotkey at its last check of the",
        f"  files: {utc(checked) if checked else 'no check yet'} (`queue.files_checked_at` in mining.json). The validator reads the waiting files when it picks the entries",
        "  of a round, and again and again while no round runs. It does not read them while a round runs: a file that",
        "  you store then shows the old verdict until the round ends. The validator never publishes batch content or",
        "  `train` values. To see the rule that a file breaks, run the `check` command of section 5: it prints the",
        "  rule code and the reason.",
        "- Identity: a registered UID on this subnet (`btcli subnets register --netuid "
        f"{params.netuid}`). One registration (UID + registration block) buys one seat in one round, win or lose.",
        "  To enter again, register again.",
        f"- Order: registration block, then UID. UID {params.burn_uid} has no seat.",
        "- One entry for each miner in a round. A second seat of the same hotkey or coldkey waits for a later round.",
        f"- `{NO_DECISION}`: our infrastructure failed (section 3). You are not charged and you go to the front of the",
        f"  queue. When the box has reported `{NO_DECISION}` for your entry {params.max_infra_attempts} times, the seat is held: it takes no",
        "  turn until the operator releases it.",
        f"- `{NOT_TRAINED}`: the training time of the round ended before your batch was trained (section 3). You are not",
        "  charged and you go to the front of the queue. It is not counted as a failure, so it never holds a seat.",
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
        f"  wait. Expected wait = max({params.squat_min_wait_s} s, seats ahead x seconds per turn). Seconds per turn = {SECONDS_PER_DAY} / the number",
        "  of entries that were charged in the last 24 hours.",
        f"- So no seat moves back in the first {show(params.squat_age_factor * params.squat_min_wait_s)} s after its registration, and no seat moves back when no entry",
        "  was charged in the last 24 hours.",
        "- A seat at the back still gets its turn, after every other seat. It stays at the back when a valid file",
        "  arrives later. To keep your place: store a valid file early, and replace it only with a file that passes",
        "  `check` (section 5).",
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
        "file. The file stays private unless it wins. Then the exact file and the LoRA are published.",
        "",
        "```json",
        json.dumps(example, indent=1),
        "```",
        "",
        f"- UTF-8 JSON, at most {params.max_batch_bytes} bytes. Top keys: `version`, `hotkey`, `groups`, and the optional",
        f"  `train`. `version` is one of {show(FORMATS)}. `hotkey` must be the owner of the object.",
        "- `groups`: a non-empty list of {prompt, rollouts}, with no other key. Each prompt is non-empty text and",
        "  differs from every other prompt. Each prompt has at least 1 rollout: {output, reward}, with no other key.",
        f"- {show(CONTROL_TOKENS)} may not appear in a prompt or an output.",
        f"- Caps: {params.max_prompts} prompts and {params.max_rollouts} rollouts. Whole groups are kept in file order while",
        "  both caps hold. The other groups are dropped. All rules below apply to the groups that are kept.",
        f"- Tokens: an output has at most {params.max_output_tokens} tokens. Prompt + output has at most",
        f"  {params.max_context_tokens} tokens. The prompt is counted as one user message in the chat template of the",
        f"  head. Training reads at most {recipe.train_max_output_tokens} output tokens and",
        f"  {recipe.train_max_context_tokens} prompt + output tokens of a rollout. A longer rollout is valid: the trainer",
        "  truncates it (section 2).",
        "- Reasoning: the reasoning of an output is the text before its first `\\boxed`. A rollout with fewer than",
        f"  {params.min_reasoning_tokens} reasoning tokens is answer-only. At most {show(params.max_answer_only_frac)} of the",
        f"  rollouts may be answer-only. At least {show(params.reasoning_token_share)} of all output tokens must be reasoning.",
        "- Repeats: the final answer is the content of the last `\\boxed{}`. One final answer may be in at most",
        f"  max({params.max_answer_repeat_count}, {show(params.max_answer_repeat_frac)} x rollouts) rollouts.",
        "- `reward`: a finite number. The rewards of one prompt must be small enough to take their mean. A reward must",
        "  equal the grade that the official grader gives to (prompt, output). The",
        f"  allowed difference is {show(params.reward_tolerance)}. One wrong reward makes the whole file invalid. An output of exactly",
        f"  {params.max_output_tokens} tokens stopped at the cap: its official grade is 0. At least",
        f"  {show(params.min_pool_prompt_frac)} of the prompts must be tasks of the public pool (0 = no such rule).",
        "- A rollout that the official grader cannot grade inside its time limit is not allowed in a file, whatever its",
        f"  reward: rule code `{TIME_LIMIT}`.",
        f"- `{NO_SIGNAL}`: your file must train something. A rollout trains when its advantage is not 0 and it has at",
        "  least 1 output token inside the training caps. The advantage is the reward minus the mean reward of its",
        "  group (`group_size`, section 2). A file in which no rollout trains is invalid. One rollout per prompt never",
        "  trains. So the example above has two rollouts with different rewards.",
        f"- `{TRAIN_COST}`: cost = steps x the tokens of the rollouts that train. The tokens of one rollout are its",
        f"  prompt tokens plus its output tokens. room = min({recipe.train_max_output_tokens},",
        f"  {recipe.train_max_context_tokens} - prompt tokens). An output shorter than the room counts its tokens + 1",
        "  (the end-of-turn token). A longer output counts the room. A rollout with advantage 0 counts 0.",
        *cost_cap,
        f"- Two limits keep the training of a round short: the training time of the round ({show(params.train_time_budget_s / 60)} minutes for",
        f"  all batches, section 3) and this cost cap for one file ({f'{show(params.max_train_token_steps)} token-steps' if params.max_train_token_steps > 0 else 'no cap now'}).",
        "- Nothing is repaired or clamped. The first rule that fails makes the whole file invalid.",
        "- Check a file before you upload it, with the same code as the validator:",
        "  `python v2/miner/affine_miner.py check --mining-json mining.json --hotkey <ss58> batch.json`. Add",
        "  `--tokenizer <head directory>` for the token rules and `--grade` for the reward rule. `--grade` needs",
        "  `--tokenizer`: only a token count can tell that an output stopped at the cap.",
        "",
        "### Rule codes",
        "",
        "The validator and the `check` command name the first rule that a file breaks. The rules run in this order:",
        "",
        *table(("rule code", "the file is invalid when"), ((f"`{code}`", meaning) for code, meaning in RULES.items())),
        "",
        f"Not a final verdict: {', '.join(f'`{code}`' for code in TRANSIENT)}. Such a code comes from one check on our machine at one",
        "moment. The validator does not keep it: it checks the file again at its next check. While a file has such a",
        "code, a round does not take it: you are not charged and the seat keeps its place (section 4). Every other",
        "code is final for those bytes: upload a new file.",
        "",
        "### How to upload",
        "",
        *_upload_lines(doc["contract"]),
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
        "  its entry was evaluated. After a new registration or activation, wait for the next link file.",
        "- After the upload, read `file_valid` in section 4.",
    ]


def _pool_lines(params: Params) -> list[str]:
    if params.env != ENV_ID:
        return []
    return [
        f"- Task pool: `sn120_v2/envs/{POOL_FILE.name}` in the code (sha256 `{POOL_SHA256}`). Scoring sends this",
        f"  instruction as the system message and the problem as the user message: {json.dumps(MANDATE)}",
        "- A prompt of your file is a pool task when it is the problem text, with or without that instruction and one",
        "  blank line before it. The official grader is `sn120_v2.envs.math_grader.grade(task, output, capped)`. It",
        "  grades model outputs in scoring and the rewards in your file.",
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
        "  on a task is the first 63 bits of sha256(\"<round_seed>|<task_id>|<j>\").",
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
        "- When the time limit of the round stops the scoring after the first look, the win rule runs on the tasks",
        "  scored by then (section 3).",
        f"- If fewer than {show(params.min_cover)} of the rollouts resolve without an infrastructure error, the entry is",
        "  `no_decision`.",
        f"- After {params.fold_every_k} applied winner(s), the stacked LoRAs are folded into a new base.",
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
        *(table(tuple(kings[0]), (row.values() for row in kings)) if kings else ["There is no king yet."]),
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
        "- Only a winner has a public batch, LoRA and `train` values. For every other entry the round file shows the",
        "  decision, the batch sha256 and the scores only.",
        "",
        "## 10. Reproduce a round at home",
        "",
        "`python v2/miner/affine_miner.py reproduce --mining-json mining.json --hotkey <ss58> --irt-snapshot irt.json",
        "--round-seed <seed> --backend <module>:<function> batch.json` checks your file, trains the LoRA and scores it",
        "against null M. It uses the same check, round runner and judging code as the validator, and your `train` values.",
        "`--backend` names a function that returns (store, trainer, evaluator, image digest). It has no default: a",
        "backend for real GPUs is not published yet.",
        "`python v2/miner/affine_miner.py bounds --mining-json mining.json` prints the bounds of section 2.",
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
        "an update is in progress: fetch both files again. These two files are the only miner documents.",
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
        "## 11. Every parameter",
        "",
        "Each value is pinned into a round when the round starts.",
        "",
        *table(("parameter", "value"), numbers),
        "",
    ]
    return "\n".join(reflow(lines)) + "\n"


def render(snapshot: Mapping[str, Any], public_url: str = "", links_url: str = "") -> tuple[bytes, bytes]:
    """(mining.json bytes, llms.txt bytes) of one snapshot."""
    doc = mining_document(snapshot, public_url, links_url)
    mining = json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    return mining.encode("utf-8"), llms_text(doc).encode("utf-8")


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
