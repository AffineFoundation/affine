"""The publisher: llms.txt and mining.json, written together from one public snapshot.

llms.txt is the only text a miner or a miner's agent needs. mining.json is
its one machine-readable companion. Both are pure functions of the same
snapshot, and both carry the same ``snapshot_digest``, so a reader can tell
that the two files belong together.

Every number in the text comes from the ``Params`` in the snapshot. The
miner-settable bounds come from ``Recipe.miner_bounds`` and the object keys
from ``layout.py``. No cap, bound or key is typed here.
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
from .admission import CONTROL_TOKENS, FORMATS, TIME_LIMIT
from .canonical import digest, sha256_hex, to_jsonable
from .envs.math_pool import ENV_ID, MANDATE, POOL_FILE, POOL_SHA256
from .errors import ContractError, InfraError, V2Error
from .irt import DRAW_RULE, MODEL, TASK_COLUMNS, draw_from_snapshot
from .judging import drop_margin
from .params import MinerBound, Params, Recipe
from .seams import ObjectStore
from .types import CAP, ERROR, FAIL, NOT_RUN, NULL_MODEL, PASS, HeadManifest

REPO = "https://github.com/AffineFoundation/affine"
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


def contract(params: Params, public_url: str) -> dict[str, Any]:
    """The fixed rules in machine-readable form. Everything is derived from ``params``."""
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
        "objects": object_keys(),
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


def mining_document(snapshot: Mapping[str, Any], public_url: str = "") -> dict[str, Any]:
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
    return {"snapshot_digest": digest(plain), **plain, "contract": contract(params, public_url)}


def _recipe_lines(params: Params) -> list[str]:
    recipe, recipe_id = params.recipe, params.recipe_id
    cap = recipe.global_norm_cap
    fields = [f.name for f in dataclasses.fields(recipe) if f.name != "miner_bounds"]
    rows = [(f"`{name}`", getattr(recipe, name), RECIPE_NOTES.get(name, "see the training code")) for name in fields]
    settable = [
        (f"`train.{row['key']}`", row["default"], row["allowed"], "decided" if row["decided"] else "placeholder: it can change")
        for row in miner_settable(recipe)
    ]
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
        *table(("key", "default", "allowed", "status"), settable),
        "",
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


def _state_lines(doc: Mapping[str, Any], params: Params) -> list[str]:
    round_now, queue = doc.get("round") or {}, doc.get("queue") or {}
    entries, seats = round_now.get("entries") or [], queue.get("next") or []
    return [
        "## 3. The round now",
        "",
        f"- phase: {show(doc.get('phase'))} (IDLE: no round is open; RUNNING: training and scoring; APPLY: decided, the",
        "  new head is being published). The last closed rounds are in mining.json under `history`.",
        *(f"- {key}: {show(value)}" for key, value in round_now.items() if not isinstance(value, (dict, list))),
        *(["", *table(ENTRY_COLUMNS, ([row.get(column) for column in ENTRY_COLUMNS] for row in entries)), ""] if entries else []),
        f"- A round takes at most {params.batch_cap} batches and needs at least {params.n_min_start}. It does not wait",
        "  for a full round. Registrations for the next round are always open.",
        f"- Round time: at most {show(params.lease_timeout_s / 60)} minutes from the lease to the promotion. A round ends",
        "  early, as soon as every LoRA is dropped or decided.",
        "- Round loop: (1) pin head M, the recipe, the task draw and the public round seed; (2) copy the next valid",
        "  files in queue order; (3) train one LoRA per file; (4) score every LoRA and a null copy of M (no adapter)",
        "  on the same tasks and seeds; (5) apply at most one winner; (6) publish; (7) start again.",
        "",
        "## 4. The queue",
        "",
        f"Seats that wait: {show(queue.get('length'))}. The next seats, in order:",
        "",
        *(table(tuple(seats[0]), (row.values() for row in seats)) if seats else ["The queue is empty."]),
        "",
        "- `file_valid`: the validator's verdict on the file that is stored for the hotkey now. It checks a stored file",
        "  again about every minute. It never publishes batch content or `train` values. To see the rule that a file",
        "  breaks, run the `check` command of section 5: it prints the rule code and the reason.",
        "- Identity: a registered UID on this subnet (`btcli subnets register --netuid "
        f"{params.netuid}`). One registration (UID + registration block) buys one seat in one round, win or lose.",
        "  To enter again, register again.",
        f"- Order: registration block, then UID. UID {params.burn_uid} has no seat.",
        "- One entry for each miner in a round. A second seat of the same hotkey or coldkey waits for a later round.",
        "- If your file is missing or invalid when the round copies files, your seat is skipped. You are not charged",
        "  and you keep your place.",
        "- If our infrastructure fails, your entry is `no_decision`. You are not charged and you go to the front of",
        f"  the queue. After {params.max_infra_attempts} such failures the seat is held for the operator.",
        f"- Squat-trim: expected wait = max({params.squat_min_wait_s} s, seats ahead / turn rate of the last 24 h). A seat",
        f"  with no valid file that is older than {show(params.squat_age_factor)} x its expected wait moves to the back.",
        "  A seat with a valid file is never moved.",
    ]


def _file_lines(doc: Mapping[str, Any], params: Params) -> list[str]:
    recipe, upload = params.recipe, (doc.get("upload") or {}).get("sealed_links")
    example = {
        "version": TRAIN_FORMATS[0],
        "hotkey": "<your hotkey ss58 address>",
        "train": {recipe.miner_bounds[0].key: getattr(recipe, recipe.miner_bounds[0].key)} if recipe.miner_bounds else {},
        "groups": [{"prompt": "<problem text>", "rollouts": [{"output": "<reasoning> \\boxed{<answer>}", "reward": 1.0}]}],
    }
    return [
        "## 5. Your file",
        "",
        "You have one private object for each hotkey. A hotkey whose queue entry is not evaluated yet keeps its queue",
        "position, and may replace its file as often as it likes until the entry is evaluated. There is no fixed",
        "window. The validator reads the latest file at lease time (the moment a round takes the entry): that copy is",
        "what is trained. The admission checks run again on each new file. The file stays private unless it wins.",
        "Then the exact file and the LoRA are published.",
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
        "- `reward`: a finite number. It must equal the grade that the official grader gives to (prompt, output). The",
        f"  allowed difference is {show(params.reward_tolerance)}. One wrong reward makes the whole file invalid. An output of exactly",
        f"  {params.max_output_tokens} tokens stopped at the cap: its official grade is 0. At least",
        f"  {show(params.min_pool_prompt_frac)} of the prompts must be tasks of the public pool (0 = no such rule).",
        "- A rollout that the official grader cannot grade inside its time limit is not allowed in a file, whatever its",
        f"  reward: rule code `{TIME_LIMIT}`.",
        "- Nothing is repaired or clamped. The first rule that fails makes the whole file invalid.",
        "- Check a file before you upload it, with the same code as the validator:",
        "  `python v2/miner/affine_miner.py check --mining-json mining.json --hotkey <ss58> batch.json`. Add",
        "  `--tokenizer <head directory>` for the token rules and `--grade` for the reward rule. `--grade` needs",
        "  `--tokenizer`: only a token count can tell that an output stopped at the cap.",
        "- Upload: each registered hotkey has a sealed link. It is a SealedBox that only the private key of the hotkey",
        "  can open. It holds a time-limited upload URL for your object. PUT the file bytes to that URL with the header",
        f"  {show([f'`{name}: {value}`' for name, value in UPLOAD_HEADERS.items()])}. The link is signed for that header: an upload",
        "  without it is refused. `python v2/miner/affine_miner.py upload --mining-json mining.json --hotkey <ss58>",
        "  --tokenizer <head directory> --grade batch.json` checks the file, reads the URL from the environment variable",
        "  `AFFINE_PUT_URL` and sends the file with that header. Then read `file_valid` in section 4.",
        f"  Where to get the link: {upload or 'not published yet'}.",
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
        f"- If fewer than {show(params.min_cover)} of the rollouts resolve without an infrastructure error, the entry is",
        "  `no_decision`.",
        f"- After {params.fold_every_k} applied winner(s), the stacked LoRAs are folded into a new base.",
        "",
        "## 7. Payout",
        "",
        f"- A king is a round winner that was applied. The last {params.kings_window} distinct king UIDs each get an",
        "  equal share of the miner weight, while the hotkey that won is still registered on that UID.",
        f"- A round with no winner changes nothing in the payout: the window of the last {params.kings_window} kings keeps being",
        f"  paid. No burn, in any case: no weight goes to UID {params.burn_uid} or to the owner as a fallback.",
        f"- One writer sets the weights every {params.weights_cadence_s} s.",
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


def llms_text(doc: Mapping[str, Any]) -> str:
    """llms.txt from a mining.json document. Pure: the same document always gives the same text."""
    params, head = Params.from_dict(doc["params"]), HeadManifest.from_dict(doc["head"]).head
    loras = [(ref.lora_hash, ref.round_id, ref.miner_uid, ref.batch_sha256) for ref in head.loras]
    numbers = [(f"`{f.name}`", getattr(params, f.name)) for f in dataclasses.fields(params) if f.name != "recipe"]
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
        f"- Head manifest (every file with its sha256): `{layout.head_manifest(head.version_id)}`",
        f"- `params_hash`: `{params.params_hash}`; `recipe_id`: `{params.recipe_id}`",
        f"- Public objects are served under: {doc['contract']['public_url'] or 'not set in this file'}. Each key below is a path under it.",
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


def render(snapshot: Mapping[str, Any], public_url: str = "") -> tuple[bytes, bytes]:
    """(mining.json bytes, llms.txt bytes) of one snapshot."""
    doc = mining_document(snapshot, public_url)
    mining = json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    return mining.encode("utf-8"), llms_text(doc).encode("utf-8")


class StorePublisher:
    """The ``Publisher`` seam: writes both files to the public zone of an object store."""

    def __init__(self, store: ObjectStore, public_url: str = "") -> None:
        self.store = store
        self.public_url = public_url.rstrip("/")

    def publish(self, snapshot: Mapping[str, Any]) -> None:
        """Render both files first, so a bad snapshot changes nothing. Then write and read back each one.

        A store that is behind or fails raises ``InfraError``: the controller keeps running and publishes again.
        """
        mining, llms = render(snapshot, self.public_url)
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
