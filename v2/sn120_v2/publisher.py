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
import time
from typing import Any, Iterable, Mapping, Sequence

from . import layout
from .canonical import digest, sha256_hex, to_jsonable
from .errors import ContractError, InfraError, V2Error, VerifyError
from .params import MinerBound, Params, Recipe
from .seams import ObjectStore
from .types import CAP, ERROR, FAIL, NOT_RUN, NULL_MODEL, PASS, HeadVersion

SCHEMA = "affine-mining-v2"
REPO = "https://github.com/AffineFoundation/affine"
BATCH_VERSIONS = ("affine-batch-v1", "affine-batch-v2", "affine-batch-v3")
TRAIN_VERSIONS = BATCH_VERSIONS[1:]  # a v1 file may not carry "train"
CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
# Batch content and non-public store zones. A snapshot that holds one of these is refused.
PRIVATE_KEYS = frozenset({"groups", "rollouts", "put_url", "put_headers"})
PRIVATE_ZONES = ("private/", "pending/")
OWN_KEYS = ("schema", "snapshot_digest", "contract")

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
    "target_modules": "layers that get an adapter",
    "kl": "KL term; 0 = off",
    "support_top_k": "the policy-gradient term uses only tokens the head could sample (top-k, then top-p)",
    "train_max_output_tokens": "training cap on output tokens; it is not the scoring cap",
    "train_max_context_tokens": "training cap on prompt + output tokens",
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
            "versions": list(BATCH_VERSIONS),
            "train_allowed_in": list(TRAIN_VERSIONS),
            "forbidden_text": list(CONTROL_TOKENS),
            "private_object": layout.miner_object("HOTKEY"),
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
    """mining.json: the snapshot itself, plus its digest and the contract block."""
    clash = [key for key in OWN_KEYS if key in snapshot]
    if clash:
        raise ContractError(f"the snapshot may not use the publisher's keys {clash}")
    refuse_private(snapshot)
    plain = to_jsonable(snapshot)
    params = Params.from_dict(plain.get("params") or {})
    if "head" not in plain:
        raise ContractError("the snapshot has no head")
    HeadVersion.from_dict(plain["head"])  # a head whose id does not match its content is refused
    return {"schema": SCHEMA, "snapshot_digest": digest(plain), **plain, "contract": contract(params, public_url)}


def _recipe_lines(recipe: Recipe, recipe_id: str) -> list[str]:
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
        "  The advantage of a rollout comes from its reward. The official grader has already checked the reward.",
        f"- An output with fewer than {recipe.train_max_output_tokens} tokens ends with the end-of-sequence token.",
        "  A longer output is cut at that cap and gets no end-of-sequence token.",
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
        f"- `train` is allowed in {show(TRAIN_VERSIONS)}. It is not allowed in {BATCH_VERSIONS[0]}.",
        "- Your `train` values are published only if your batch wins.",
    ]


def _state_lines(doc: Mapping[str, Any], params: Params) -> list[str]:
    round_now, queue = doc.get("round") or {}, doc.get("queue") or []
    verdicts = doc.get("verdicts") or {}
    return [
        "## 3. The round now",
        "",
        *(f"- {key}: {show(value)}" for key, value in round_now.items() if not isinstance(value, (dict, list))),
        f"- A round takes at most {params.batch_cap} batches and needs at least {params.n_min_start}. It does not wait",
        "  for a full round. Registrations for the next round are always open.",
        "- Round loop: (1) pin head M, the recipe, the task draw and the public round seed; (2) copy the next valid",
        "  files in queue order; (3) train one LoRA per file; (4) score every LoRA and a null copy of M (no adapter)",
        "  on the same tasks and seeds; (5) apply at most one winner; (6) publish; (7) start again.",
        "",
        "## 4. The queue",
        "",
        *(table(tuple(queue[0]), (row.values() for row in queue)) if queue else ["The queue is empty."]),
        "",
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
        "",
        "### Live file verdicts",
        "",
        "The validator checks each stored file again within about a minute of an upload. A verdict never shows batch",
        "content or `train` values.",
        "",
        *(
            table(("hotkey", "valid", "reason", "format", "prompts", "rollouts", "checked"), (
                (hotkey, v.get("valid"), v.get("reason"), v.get("format"), v.get("n_prompts"), v.get("n_rollouts"), utc(v.get("checked_at")))
                for hotkey, v in sorted(verdicts.items())
            ))
            if verdicts
            else ["No file is stored."]
        ),
    ]


def _file_lines(doc: Mapping[str, Any], params: Params) -> list[str]:
    recipe, upload = params.recipe, (doc.get("upload") or {}).get("sealed_links")
    example = {
        "version": TRAIN_VERSIONS[0],
        "hotkey": "<your hotkey ss58 address>",
        "train": {recipe.miner_bounds[0].key: getattr(recipe, recipe.miner_bounds[0].key)} if recipe.miner_bounds else {},
        "groups": [{"prompt": "<problem text>", "rollouts": [{"output": "<reasoning> \\boxed{<answer>}", "reward": 1.0}]}],
    }
    return [
        "## 5. Your file",
        "",
        "You have one private object for each hotkey. You can overwrite it until a round copies it. The copy is what",
        "is trained. The file stays private unless it wins. Then the exact file and the LoRA are published.",
        "",
        "```json",
        json.dumps(example, indent=1),
        "```",
        "",
        f"- UTF-8 JSON, at most {params.max_batch_bytes} bytes. `version` is one of {show(BATCH_VERSIONS)}.",
        "  `hotkey` must be the owner of the object.",
        "- `groups`: a list of {prompt, rollouts}. Each prompt is non-empty text and differs from every other prompt.",
        "  Each prompt has at least 1 rollout. A rollout is {output, reward}.",
        f"- `output`: the assistant text. {show(CONTROL_TOKENS)} may not appear in a prompt or an output.",
        f"- Training reads at most {recipe.train_max_output_tokens} output tokens and {recipe.train_max_context_tokens}",
        "  prompt + output tokens of a rollout.",
        f"- Caps: {params.max_prompts} prompts and {params.max_rollouts} rollouts. Prompts over a cap are dropped whole,",
        "  in file order. The part that is kept must be valid.",
        "- `reward`: a finite number. It must equal the reward that the official grader gives to (prompt, output),",
        f"  within {show(params.reward_tolerance)}. One wrong reward makes the whole file invalid.",
        f"- Reasoning: each output needs at least {params.min_reasoning_tokens} tokens of reasoning before the first",
        f"  `\\boxed{{}}`, and at least {show(params.reasoning_token_share)} of the tokens of the file must be reasoning.",
        f"- At most {show(params.max_answer_only_frac)} of the rollouts may be an answer with no reasoning.",
        f"- No boxed answer may repeat in more than {show(params.max_answer_repeat_frac)} of the rollouts or in more than",
        f"  {params.max_answer_repeat_count} rollouts, whichever limit is higher.",
        f"- At least {show(params.min_pool_prompt_frac)} of the prompts must come from the public task pool (0 = no rule).",
        "- Check a file before you upload it, with the same code as the validator:",
        "  `python v2/miner/check_batch.py --mining-json mining.json --hotkey <ss58> batch.json`",
        "- Upload: each registered hotkey has a sealed link. It is a SealedBox that only the private key of the hotkey",
        "  can open. It holds a time-limited upload URL for your object and the headers to send. PUT the file bytes",
        f"  to that URL. Then read your verdict in section 4. Where to get the link: {upload or 'not published yet'}.",
    ]


def _judging_lines(doc: Mapping[str, Any], params: Params) -> list[str]:
    irt, kings, keys = doc.get("irt") or {}, doc.get("kings") or [], object_keys()
    looks = params.look_sizes()
    return [
        "## 6. Judging",
        "",
        "- All tasks are public. There is no hidden task set.",
        f"- Draw: {params.n_tasks} tasks from the public pool of `{params.env}`, {params.samples_per_task} sample(s) per task.",
        "  The draw is a fixed function of the public sampler snapshot and the public round seed. Both are frozen at",
        f"  round start. Sampling method of the current snapshot: {show(irt.get('method'))}.",
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
        f"  {params.min_scored_tasks} tasks, with no correction for the number of LoRAs. The cap-hit rate of the LoRA must",
        f"  be below max({show(params.cap_hit_floor)}, null M cap-hit rate + {show(params.cap_hit_margin)}).",
        "- The eligible LoRA with the highest delta wins. A tie goes to the miner that was earlier in the queue.",
        "  With no eligible LoRA, M stays and there is no king. There is no second confirmation round.",
        f"- Looks: scores are checked after {show(looks)} tasks. At an early look with at least {params.min_scored_tasks}",
        f"  tasks, a LoRA is dropped if z is at or below {show(params.futility_z)}. The win rule runs only at the last look.",
        f"- If fewer than {show(params.min_cover)} of the rollouts resolve without an infrastructure error, the entry is",
        "  `no_decision`.",
        f"- After {params.fold_every_k} applied winner(s), the stacked LoRAs are folded into a new base.",
        "",
        "## 7. Payout",
        "",
        f"- A king is a round winner that was applied. The last {params.kings_window} distinct king UIDs each get an",
        "  equal share of the miner weight, while the hotkey that won is still registered on that UID.",
        f"- A null round adds no king. Weight that no king earned goes to UID {params.burn_uid}. Weights are never all zero.",
        f"- One writer sets the weights every {params.weights_cadence_s} s.",
        "",
        *(table(tuple(kings[0]), (row.values() for row in kings)) if kings else ["There is no king yet."]),
        "",
        "## 8. The IRT snapshot",
        "",
        "It is public every round: task difficulties, model abilities and sampling weights, tagged with the head.",
        f"Object: `{keys['irt_snapshot']}`. With it and the round seed you can rebuild the draw.",
        "",
        *(f"- {key}: {show(value)}" for key, value in irt.items() if not isinstance(value, (dict, list))),
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
        "`python v2/miner/reproduce.py --mining-json mining.json --hotkey <ss58> batch.json` checks your file, trains",
        "the LoRA and scores it against null M, with the same trainer and evaluator code as the validator.",
        "`python v2/miner/reproduce.py --mining-json mining.json --bounds` prints the bounds of section 2.",
    ]


def llms_text(doc: Mapping[str, Any]) -> str:
    """llms.txt from a mining.json document. Pure: the same document always gives the same text."""
    params, head = Params.from_dict(doc["params"]), HeadVersion.from_dict(doc["head"])
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
        f"- Public objects are served under `{show(doc['contract']['public_url'])}`.",
        "",
        *(table(("stacked LoRA", "round", "miner uid", "batch sha256"), loras) if loras else ["No LoRA is stacked on the base."]),
        "",
        *_recipe_lines(params.recipe, params.recipe_id),
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
    return "\n".join(lines)


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
        """Render both files first, so a bad snapshot changes nothing. Then write and read back each one."""
        mining, llms = render(snapshot, self.public_url)
        head_key = layout.head_manifest(snapshot["head"]["version_id"])
        try:
            if not self.store.exists(head_key):
                raise VerifyError(f"head manifest {head_key} is not public yet; the contract may not name this head")
            for key, data in ((layout.MINING_JSON, mining), (layout.LLMS_TXT, llms)):
                self.store.put(key, data)
                if self.store.sha256(key) != sha256_hex(data):
                    raise VerifyError(f"{key} does not read back as written")
        except V2Error:
            raise
        except Exception as exc:  # any store failure is ours: the controller retries the same snapshot
            raise InfraError(f"cannot publish the miner contract: {exc}") from exc
