"""Miner tools for Affine SN120 v2: see your bounds, check a file, reproduce a round at home.

    python v2/miner/affine_miner.py bounds --mining-json mining.json
    python v2/miner/affine_miner.py check --mining-json mining.json --hotkey <ss58> batch.json
    python v2/miner/affine_miner.py reproduce --mining-json mining.json --hotkey <ss58> \
        --draw draw.json --backend <module>:<function> batch.json

Every rule and every number comes from mining.json and from the validator's
own code in ``sn120_v2``: the same batch check, the same ``Trainer`` and
``Evaluator`` seams, the same judging. This file holds no rule of its own.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from sn120_v2 import judging, layout, sampling
from sn120_v2.admission import Batch, HFTokens, Invalid, Tokens, check_batch
from sn120_v2.canonical import to_jsonable
from sn120_v2.errors import ContractError, V2Error
from sn120_v2.params import Params
from sn120_v2.publisher import miner_settable
from sn120_v2.seams import Evaluator, ObjectStore, Trainer
from sn120_v2.types import (
    LOST,
    NULL_MODEL,
    SCORED,
    BoxResult,
    Draw,
    Entry,
    EntryResult,
    EvalRequest,
    FrozenBatch,
    HeadVersion,
    Registration,
    RoundTicket,
    TrainRequest,
)

ROUND_ID, LEASE_ID, BOX_ID = "local", "local-l0", "local-box"


def load_contract(source: str) -> tuple[Params, HeadVersion]:
    """Params and head from mining.json: a file or an http(s) URL. There is no built-in default."""
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as reply:
            raw = reply.read()
    else:
        raw = Path(source).read_bytes()
    doc = json.loads(raw)
    return Params.from_dict(doc["params"]), HeadVersion.from_dict(doc["head"])


def bounds_text(params: Params) -> str:
    """Every field a miner may set under ``train``, read from ``Recipe.miner_bounds``."""
    lines = [f'Fields you may set under "train" (recipe_id {params.recipe_id}):']
    for row in miner_settable(params.recipe):
        status = "" if row["decided"] else " [placeholder: it can change]"
        lines.append(f"  {row['key']}: default {json.dumps(row['default'])}; allowed: {row['allowed']}{status}")
    return "\n".join(lines)


def applied_train(params: Params, train: Mapping[str, Any]) -> dict[str, Any]:
    """The values the validator trains with for this ``train`` object. Raises ``BatchRejected``; never clamps."""
    recipe = params.recipe.for_miner(train)
    return {bound.key: to_jsonable(getattr(recipe, bound.key)) for bound in params.recipe.miner_bounds}


def check(raw: bytes, hotkey: str, params: Params, tokens: Tokens | None = None) -> dict[str, Any]:
    """The validator's check of one file, as a report. Raises ``Invalid`` when the file breaks a rule."""
    return check_report(check_batch(raw, hotkey, params, tokens=tokens), params)


def check_report(batch: Batch, params: Params) -> dict[str, Any]:
    return {
        "valid": True,
        "format": batch.format,
        "batch_sha256": batch.sha256,
        "n_prompts": batch.n_prompts,
        "n_rollouts": batch.n_rollouts,
        "dropped_groups": batch.dropped_groups,
        "rules_run": list(batch.checked),
        "rules_not_run": [family for family in ("tokens", "grader") if family not in batch.checked],
        "train_applied": applied_train(params, batch.train),
    }


def reproduce(
    raw: bytes,
    hotkey: str,
    params: Params,
    head: HeadVersion,
    draw: Draw,
    store: ObjectStore,
    trainer: Trainer,
    evaluator: Evaluator,
    *,
    train: Mapping[str, Any] | None = None,
    tokens: Tokens | None = None,
) -> dict[str, Any]:
    """Run one batch as a one-entry round: check, train, score against null M, decide.

    ``train`` replaces the ``train`` object of the file. It has the same
    shape and passes the same gate (``Recipe.for_miner``).
    """
    batch = check_batch(raw, hotkey, params, tokens=tokens)
    train = dict(batch.train if train is None else train)
    report = check_report(batch, params) | {"train_applied": applied_train(params, train)}
    key = layout.frozen_batch(ROUND_ID, hotkey)
    store.put(key, raw)
    frozen = FrozenBatch(hotkey, batch.sha256, key, batch.format, batch.n_prompts, batch.n_rollouts, train)
    entry, started = Entry(0, Registration(0, hotkey, 0), frozen), int(time.time())
    ticket = RoundTicket(ROUND_ID, LEASE_ID, BOX_ID, head, params, draw, (entry,), started, started + params.lease_timeout_s)
    seed = sampling.train_seed(draw.round_seed, frozen.batch_sha256)
    gpu_ids = tuple(range(params.gpus_per_lora(1)))
    out_prefix = layout.pending_lora_prefix(ROUND_ID, LEASE_ID, 0)
    try:
        lora = trainer.train(TrainRequest(ROUND_ID, head, entry, params, seed, gpu_ids, out_prefix))
    finally:
        trainer.close()  # every GPU is free before scoring starts
    report |= {"lora_hash": lora.lora_hash, "train_report": lora.report.to_dict()}
    n, done, image = len(draw.task_ids), 0, ""
    null, mine = [""] * params.samples_per_task, [""] * params.samples_per_task
    try:
        for look, size in enumerate(params.look_sizes(n)):
            chunk = evaluator.evaluate(EvalRequest(ROUND_ID, head, (lora,), draw.task_ids[done:size], draw.round_seed, look, params))
            if lora.lora_hash not in chunk.outcomes:
                raise ContractError("the evaluator did not score the LoRA")
            null = [a + b for a, b in zip(null, chunk.outcomes[NULL_MODEL])]
            mine = [a + b for a, b in zip(mine, chunk.outcomes[lora.lora_hash])]
            done, image = size, chunk.image_digest
            partial = judging.score_entry(draw.weights, null, mine, params)
            if judging.drop_at_look(partial, look, n, params):
                reason = f"dropped after {size} of {n} tasks: it can no longer reach the win bar"
                return report | {"status": LOST, "reason": reason, "score": partial.to_dict()}
    finally:
        evaluator.close()
    scored = EntryResult(0, SCORED, lora=lora, outcomes=tuple(mine))
    result = BoxResult(ROUND_ID, LEASE_ID, BOX_ID, ticket.digest(), head.version_id, image, tuple(null), (scored,), started, int(time.time()))
    try:
        decision = judging.decide([(ticket, result)])
    except ContractError as exc:  # for example a worker image that the contract does not pin
        score = judging.score_entry(draw.weights, null, mine, params, n_total=n)
        return report | {"status": "unofficial", "reason": f"the validator would refuse this run: {exc}", "score": score.to_dict()}
    row = decision.entries[0]
    return report | {"status": row.status, "reason": row.reason or decision.reason, "score": row.score.to_dict()}


def load_tokens(path: str | None) -> Tokens | None:
    """The head's tokenizer, or None. ``transformers`` is needed only when a tokenizer is given."""
    return HFTokens(importlib.import_module("transformers").AutoTokenizer.from_pretrained(path)) if path else None


def load_draw(path: str, n_tasks: int | None) -> Draw:
    """A ``Draw`` record, or a round file that holds one under ``draw``. ``n_tasks`` keeps the first tasks only."""
    data = json.loads(Path(path).read_bytes())
    draw = Draw.from_dict(data.get("draw", data))
    return draw if n_tasks is None else Draw(draw.snapshot_id, draw.round_seed, draw.task_ids[:n_tasks], draw.weights[:n_tasks])


def load_backend(spec: str, workdir: str) -> tuple[ObjectStore, Trainer, Evaluator]:
    """``module:function``. The function takes a work directory and returns (store, trainer, evaluator)."""
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)(workdir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("bounds", "check", "reproduce"):
        command = commands.add_parser(name)
        command.add_argument("--mining-json", required=True, help="mining.json: a file or an http(s) URL")
        if name == "bounds":
            continue
        command.add_argument("batch", help="your batch file")
        command.add_argument("--hotkey", required=True, help="the ss58 address of your hotkey")
        command.add_argument("--tokenizer", help="tokenizer of the head (directory or model id); without it the token rules do not run")
        if name == "reproduce":
            command.add_argument("--draw", required=True, help="a task draw: a Draw record or a public round file")
            command.add_argument("--tasks", type=int, help="score only the first N tasks of the draw")
            command.add_argument("--train", help='a JSON object that replaces "train" of the file')
            command.add_argument("--backend", required=True, help="module:function that returns (store, trainer, evaluator)")
            command.add_argument("--work", help="work directory (default: a temporary directory)")
    args = parser.parse_args(argv)
    try:
        params, head = load_contract(args.mining_json)
        if args.command == "bounds":
            print(bounds_text(params))
            return 0
        raw, tokens = Path(args.batch).read_bytes(), load_tokens(args.tokenizer)
        if args.command == "check":
            report = check(raw, args.hotkey, params, tokens)
        else:
            print(bounds_text(params), file=sys.stderr)
            store, trainer, evaluator = load_backend(args.backend, args.work or tempfile.mkdtemp(prefix="affine-reproduce-"))
            train = json.loads(args.train) if args.train else None
            draw = load_draw(args.draw, args.tasks)
            report = reproduce(raw, args.hotkey, params, head, draw, store, trainer, evaluator, train=train, tokens=tokens)
    except Invalid as exc:
        print(json.dumps({"valid": False, "rule": exc.rule, "reason": str(exc)}, indent=1))
        return 1
    except V2Error as exc:
        print(json.dumps({"error": type(exc).__name__, "reason": str(exc)}, indent=1))
        return 2
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
