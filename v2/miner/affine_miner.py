"""Miner tools for Affine SN120 v2: see your bounds, check a file, upload it, reproduce a round at home.

    python v2/miner/affine_miner.py bounds --mining-json mining.json
    python v2/miner/affine_miner.py check --mining-json mining.json --hotkey <ss58> batch.json
    AFFINE_PUT_URL=<link> python v2/miner/affine_miner.py upload --mining-json mining.json --hotkey <ss58> batch.json
    python v2/miner/affine_miner.py reproduce --mining-json mining.json --hotkey <ss58> \
        --irt-snapshot irt.json --round-seed <seed> --backend <module>:<function> batch.json

``--backend sn120_v2.replicate:backend`` is the code of the validator's box. It runs as it is only on a machine
with the pinned install and the GPU count of the params. On any other machine use
``python -m sn120_v2.replicate run``: see ``v2/REPLICATE.md``.

Every rule and every number comes from mining.json and from the validator's
own code in ``sn120_v2``: the same batch check, the same box runner on the
``Trainer`` and ``Evaluator`` seams, the same judging. This file holds no
rule of its own.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from sn120_v2 import judging, layout
from sn120_v2.admission import Batch, Grader, HFTokens, Invalid, Tokens, check_batch, env_grader
from sn120_v2.canonical import to_jsonable
from sn120_v2.envs.math_pool import ENV_ID, load_pool
from sn120_v2.errors import ConfigError, ContractError, InfraError, V2Error
from sn120_v2.irt import draw_from_snapshot
from sn120_v2.params import Params
from sn120_v2.publisher import UPLOAD_HEADERS, miner_settable
from sn120_v2.seams import Evaluator, ObjectStore, RoundRunner, Trainer
from sn120_v2.types import Draw, Entry, FrozenBatch, HeadManifest, HeadVersion, Registration, RoundTicket
from sn120_v2.worker import LocalRunner

ROUND_ID, LEASE_ID, BOX_ID = "local", "local-l0", "local-box"
PUT_URL_ENV = "AFFINE_PUT_URL"  # the opened upload link; it is a secret, so it is not a command-line argument
# The public site answers 403 to the default client name of urllib on its files under public/. So every request of this tool names itself.
CLIENT = {"User-Agent": "sn120-v2-miner/1"}


def load_contract(source: str) -> tuple[Params, HeadVersion]:
    """Params and head from mining.json: a file or an http(s) URL. There is no built-in default."""
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(urllib.request.Request(source, headers=CLIENT), timeout=30) as reply:
            raw = reply.read()
    else:
        raw = Path(source).read_bytes()
    doc = json.loads(raw)
    return Params.from_dict(doc["params"]), HeadManifest.from_dict(doc["head"]).head


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


def check(raw: bytes, hotkey: str, params: Params, tokens: Tokens | None = None, grader: Grader | None = None) -> dict[str, Any]:
    """The validator's check of one file, as a report. Raises ``Invalid`` when the file breaks a rule."""
    return check_report(check_batch(raw, hotkey, params, tokens=tokens, grader=grader), params)


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
    runner: RoundRunner,
    *,
    train: Mapping[str, Any] | None = None,
    tokens: Tokens | None = None,
    grader: Grader | None = None,
) -> dict[str, Any]:
    """Run one batch as a one-entry round: check, then the validator's box code and judging.

    ``train`` replaces the ``train`` object of the file. It has the same
    shape and passes the same gate (``Recipe.for_miner``).
    """
    batch = check_batch(raw, hotkey, params, tokens=tokens, grader=grader)
    train = dict(batch.train if train is None else train)
    report = check_report(batch, params) | {"train_applied": applied_train(params, train)}
    key, now = layout.frozen_batch(ROUND_ID, hotkey), int(time.time())
    store.put(key, raw)
    frozen = FrozenBatch(hotkey, batch.sha256, key, batch.format, batch.n_prompts, batch.n_rollouts, train)
    entry = Entry(0, Registration(0, hotkey, 0), frozen)
    ticket = RoundTicket(ROUND_ID, LEASE_ID, BOX_ID, head, params, draw, (entry,), now, now + params.lease_timeout_s)
    result = runner.run(ticket)
    mine = result.entries[0]
    if mine.lora is not None:
        report |= {"lora_hash": mine.lora.lora_hash, "train_report": mine.lora.report.to_dict()}
    try:
        decision = judging.decide([(ticket, result)])
    except ContractError as exc:  # for example a worker image that the contract does not pin
        score = judging.score_entry(draw.weights, result.null_outcomes, mine.outcomes, params, n_total=len(draw.task_ids)) if mine.outcomes else None
        return report | {"status": "unofficial", "reason": f"the validator would refuse this run: {exc}", "score": score and score.to_dict()}
    row = decision.entries[0]
    return report | {"status": row.status, "reason": row.reason or decision.reason, "eliminated_at_look": row.eliminated_at_look, "score": row.score and row.score.to_dict()}


def upload(raw: bytes, url: str) -> None:
    """PUT the file to the presigned link, with the headers the link is signed for. The URL is a secret: it is never printed."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=raw, method="PUT", headers=UPLOAD_HEADERS | CLIENT), timeout=300):
            pass
    except OSError as exc:
        raise InfraError(f"the upload failed: {type(exc).__name__} {getattr(exc, 'code', '')}".strip()) from None


def load_tokens(path: str | None) -> Tokens | None:
    """The head's tokenizer, or None. ``transformers`` is needed only when a tokenizer is given."""
    return HFTokens(importlib.import_module("transformers").AutoTokenizer.from_pretrained(path)) if path else None


def official_grader(params: Params) -> Grader:
    """The validator's grader of the environment, built with the validator's own call. Needs the ``envs`` extra."""
    if params.env != ENV_ID:
        raise ConfigError(f"this tool has no grader for environment {params.env!r}")
    return env_grader(load_pool(), importlib.import_module("sn120_v2.envs.math_grader").grade)


def load_backend(spec: str, workdir: str) -> tuple[ObjectStore, Trainer, Evaluator, str]:
    """``module:function``. The function takes a work directory and returns (store, trainer, evaluator, image digest)."""
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)(workdir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("bounds", "check", "upload", "reproduce"):
        command = commands.add_parser(name)
        command.add_argument("--mining-json", required=True, help="mining.json: a file or an http(s) URL")
        if name == "bounds":
            continue
        command.add_argument("batch", help="your batch file")
        command.add_argument("--hotkey", required=True, help="the ss58 address of your hotkey")
        command.add_argument("--tokenizer", help="tokenizer of the head (directory or model id); without it the token rules do not run")
        command.add_argument("--grade", action="store_true", help="match your rewards with the official grader (needs --tokenizer and the envs extra)")
        if name == "reproduce":
            command.add_argument("--irt-snapshot", required=True, help="the published IRT snapshot of the round (a JSON file)")
            command.add_argument("--round-seed", required=True, help="the public seed of the round")
            command.add_argument("--tasks", type=int, help="draw N tasks (default: n_tasks of the contract)")
            command.add_argument("--train", help='a JSON object that replaces "train" of the file')
            command.add_argument("--backend", required=True, help="module:function that returns (store, trainer, evaluator, image digest)")
            command.add_argument("--work", help="work directory (default: a temporary directory)")
    args = parser.parse_args(argv)
    try:
        params, head = load_contract(args.mining_json)
        if args.command == "bounds":
            print(bounds_text(params))
            return 0
        if args.grade and not args.tokenizer:  # with no token count no row counts as capped: not the validator's verdict
            raise ConfigError("--grade needs --tokenizer")
        if args.command == "upload" and not os.environ.get(PUT_URL_ENV):
            raise ConfigError(f"set {PUT_URL_ENV} to the upload URL of your sealed link")
        raw, tokens = Path(args.batch).read_bytes(), load_tokens(args.tokenizer)
        grader = official_grader(params) if args.grade else None
        if args.command in ("check", "upload"):
            report = check(raw, args.hotkey, params, tokens, grader)
            if args.command == "upload":  # an invalid file is never sent
                upload(raw, os.environ[PUT_URL_ENV])
                report["uploaded"] = True
        else:
            print(bounds_text(params), file=sys.stderr)
            store, trainer, evaluator, image = load_backend(args.backend, args.work or tempfile.mkdtemp(prefix="affine-reproduce-"))
            runner = LocalRunner(trainer, evaluator, lambda: int(time.time()), image)
            train = json.loads(args.train) if args.train else None
            snapshot = json.loads(Path(args.irt_snapshot).read_bytes())
            draw = draw_from_snapshot(snapshot, args.round_seed, args.tasks or params.n_tasks)
            report = reproduce(raw, args.hotkey, params, head, draw, store, runner, train=train, tokens=tokens, grader=grader)
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
