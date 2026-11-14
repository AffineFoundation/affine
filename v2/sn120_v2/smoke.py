"""Smoke-test helpers: the parameter files of the ramp, and small copies of real miner files.

The ramp is five rounds on one bootstrap, small to full. Each round has its
own parameter file, and ``set-params`` pins it before the round:

  1-tiny   1 batch, 1 GPU, 1 task: the whole path in minutes; it can have no winner
  2-two    2 batches, 2 GPUs, 8 tasks
  3-eight  8 batches, 8 GPUs, 32 tasks with the real reply cap: the round that should promote and fold
  4-multi  2 batches on 8 GPUs, on a folded head: each training job has all 8 GPUs, like every job on the box
  5-full   the launch values

Two more rounds, between 4-multi and 5-full, hit the round's time limit on
purpose. Each is 2-two with a test of 2000 tasks and a short time limit:

  limit-judged      the limit comes after the first look: each LoRA is judged on the chunks finished
  limit-not-scored  the limit comes inside the first look: each LoRA is not_scored and is not charged

  python -m sn120_v2.smoke params --plan plan.json --out DIR     write the seven files
  python -m sn120_v2.smoke copy --state-dir DIR                  small copies for the next round
  python -m sn120_v2.smoke copy --state-dir DIR --whole          whole files for the next round

Nothing here can change mainnet. ``copy`` refuses a store prefix that does
not start with ``smoke``, and it only reads v1's bucket. A parameter file has
an effect only when an operator pins it with ``set-params`` on a state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import layout
from .admission import Invalid, check_batch
from .canonical import sha256_hex
from .cli import read_state
from .errors import ConfigError, V2Error
from .params import Params, require_env
from .queue import waiting
from .r2 import R2Store
from .seams import ObjectStore

# For the four small rounds only. A winner needs z > 0 on at least 2 tasks, so a small round can promote and fold.
# No seat is put on hold while faults are fixed, and a new round starts 5 seconds after a failed one, not 5 minutes.
_SMALL = {"z_bar": 0.0, "min_scored_tasks": 2, "max_infra_attempts": 1000, "apply_retry_s": 5}
# A reply stops after this many tokens in the rounds that only prove the path. One reply at the real cap can take 15 minutes.
SHORT_REPLY = 2048
# For the two rounds at the time limit: 2-two with a test that can never be scored inside the limit.
_AT_LIMIT = _SMALL | {"batch_cap": 2, "n_gpus": 2, "n_tasks": 2000, "max_output_tokens": SHORT_REPLY}
MAX_FAILED_CHECKS = 8  # a small copy gives up on a file after this many groups that fail the admission check
TOKEN_FAULT = "the token count fails for each"  # the start of the reason when the tokenizer itself fails, not the cap
FAULT_CHARS = 200  # the most characters of a tokenizer error in a line (the error can quote a miner's text)
# Overrides of the launch values, one set per round of the ramp, in order. The two digests are added to each.
RAMP: dict[str, dict[str, Any]] = {
    "1-tiny": _SMALL | {"batch_cap": 1, "n_gpus": 1, "n_tasks": 1, "eval_chunk_tasks": 1, "max_output_tokens": SHORT_REPLY},
    # Looks after 2, 4 and 8 tasks: three blocks of tasks, and a LoRA can be dropped at a look.
    "2-two": _SMALL | {"batch_cap": 2, "n_gpus": 2, "n_tasks": 8, "eval_chunk_tasks": 4, "max_output_tokens": SHORT_REPLY},
    # The real reply cap, so null M passes tasks and a LoRA can win. One look and one block: the block lasts as long as its longest reply.
    "3-eight": _SMALL | {"batch_cap": 8, "n_gpus": 8, "n_tasks": 32, "eval_chunk_tasks": 32, "looks": [1.0]},
    "4-multi": _SMALL | {"batch_cap": 2, "n_gpus": 8, "n_tasks": 8, "eval_chunk_tasks": 4, "max_output_tokens": SHORT_REPLY},
    # The round's time limit on purpose (worker.LocalRunner.run). The training time is far more than two small copies
    # need, so that no batch is not_trained.
    # Stop at minute 23. First look = 4 tasks = one chunk, so it is complete long before the stop, even when the first
    # replies of a new server take minutes. min_scored_tasks 8 is above the first look, so no LoRA is dropped there, and
    # each LoRA is judged at the stop by the launch rule (z above 1) on the chunks finished.
    "limit-judged": _AT_LIMIT
    | {"eval_chunk_tasks": 4, "looks": [0.002, 1.0], "z_bar": 1.0, "min_scored_tasks": 8}
    | {"round_time_limit_s": 1500, "lease_timeout_s": 3000, "promote_reserve_s": 120, "train_time_budget_s": 600},
    # Stop at minute 6. First look = all 2000 tasks in one chunk: it cannot be complete at the stop.
    "limit-not-scored": _AT_LIMIT
    | {"eval_chunk_tasks": 2000, "looks": [1.0]}
    | {"round_time_limit_s": 480, "lease_timeout_s": 960, "promote_reserve_s": 120, "train_time_budget_s": 300},
    "5-full": {},
}


def ramp_files(plan: Mapping[str, str], short_reply: int = SHORT_REPLY) -> dict[str, str]:
    """File name -> JSON text of each parameter file. ``plan`` is the output of ``python -m sn120_v2.gpu.plan``.

    ``short_reply`` is the reply cap of the rounds that have a low cap.
    """
    files = {}
    for name, overrides in RAMP.items():
        data = overrides | {"recipe": {"image_digests": [plan["image_digest"]], "code_digest": plan["code_digest"]}}
        if "max_output_tokens" in data:
            data["max_output_tokens"] = short_reply
        Params.from_dict(data)  # a wrong value fails here, not at set-params
        files[f"params-{name}.json"] = json.dumps(data, indent=1) + "\n"
    return files


def _signal(group: Any) -> bool:
    """True when the group has two rollouts with different rewards: without that it trains nothing."""
    try:
        return len({row["reward"] for row in group["rollouts"]}) > 1 and all(isinstance(row["output"], str) for row in group["rollouts"])
    except (KeyError, TypeError):
        return False


def cut_batch(raw: bytes, groups: int, cap: int, count: Callable[[str], int], check: Callable[[bytes], str]) -> tuple[bytes | None, str]:
    """A small copy of a batch file: up to ``groups`` whole groups of it, each one unchanged.

    A group is taken when it has two different rewards, every rollout of it
    has at most ``cap`` tokens by ``count``, and the copy with it still
    passes ``check`` (``check`` returns "" or the code of the failed rule).
    The ``train`` object is left out, so the copy trains with the default
    recipe. Returns the bytes and a note, or None and the reason. When
    ``count`` fails for every group, the reason starts with ``TOKEN_FAULT``
    and holds the error: the cause is then the tokenizer, not the cap.
    """
    try:
        value = json.loads(raw)
        head, source = {"version": value["version"], "hotkey": value["hotkey"]}, [group for group in value["groups"] if _signal(group)]
    except (ValueError, KeyError, TypeError):
        return None, "the file is not a batch file"
    if not source:
        return None, "no group has two different rewards"

    faults: list[str] = []

    def longest(group: Any) -> int | None:
        try:
            return max(count(row["output"]) for row in group["rollouts"])
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # a text that makes the tokenizer fail (a Rust panic is a BaseException): never use that group
            faults.append(" ".join(f"{type(exc).__name__}: {exc}".split())[:FAULT_CHARS])
            return None

    # Tokens are counted for the groups with the shortest text first; a group far above the cap is never counted.
    source.sort(key=lambda group: max(len(row["output"]) for row in group["rollouts"]))
    kept: list[Any] = []
    best, size, rule, failed = None, 0, "", 0
    for group in source:
        tokens = longest(group)
        if tokens is None:
            continue
        if tokens > cap:
            if best is None and not rule:
                return None, f"no group with two different rewards has every rollout within {cap} tokens; the nearest group needs {tokens}"
            break
        data = json.dumps(head | {"groups": kept + [group]}).encode()
        rule = check(data)
        if rule:
            failed += 1
            if failed == MAX_FAILED_CHECKS:
                break
            continue
        kept, best, size = kept + [group], data, max(size, tokens)
        if len(kept) == groups:
            break
    if best is None and not rule:  # no group came as far as the cap or the check: each one failed in the tokenizer
        return None, f"{TOKEN_FAULT} of its {len(faults)} groups with two different rewards: {faults[0]}"
    if best is None:
        return None, f"a copy of its short groups fails the check: {rule}"
    rollouts = sum(len(group["rollouts"]) for group in kept)
    return best, f"{len(kept)} of {len(value['groups'])} groups, {rollouts} rollouts, longest {size} tokens"


def copy_files(
    src: ObjectStore, dst: ObjectStore, hotkeys: Sequence[str], count: int, cut: Callable[[str, bytes], tuple[bytes | None, str]] | None = None
) -> tuple[list[str], list[str]]:
    """Give the first ``count`` of ``hotkeys`` that have a usable file in ``src`` a file in ``dst``. ``src`` is only read.

    ``cut(hotkey, raw)`` returns the bytes to store and a note, or None and
    the reason. Without ``cut`` the whole file is copied. A file that ``dst``
    already has is replaced. Returns the lines of the copied files and the
    lines of the files that were passed over.
    """
    copied: list[str] = []
    passed: list[str] = []
    for hotkey in hotkeys:
        if len(copied) >= count:
            break
        key = layout.miner_object(hotkey)
        try:
            raw = src.get(key)
        except KeyError:
            continue
        data, note = (raw, "whole file") if cut is None else cut(hotkey, raw)
        if data is None:
            passed.append(f"passed over {hotkey}: {note}")
            continue
        dst.put(key, data)
        if dst.sha256(key) != sha256_hex(data):
            raise V2Error(f"the copy of {key} does not read back as written")
        try:
            train = json.loads(data).get("train")
        except (ValueError, AttributeError):
            train = "not a JSON object"
        recipe = "default recipe" if train is None else f"train={json.dumps(train, sort_keys=True)}"
        copied.append(f"copied {hotkey} {note}, {len(data)} of {len(raw)} bytes sha256={sha256_hex(data)} {recipe}")
    return copied, passed


def admission_cut(admission: Any, groups: int) -> Callable[[str, bytes], tuple[bytes | None, str]]:
    """``cut`` for ``copy_files``: a small copy that passes the full admission check under the pinned parameters."""
    params, tokens, grader = admission.params, admission.tokens, admission.grader

    def cut(hotkey: str, raw: bytes) -> tuple[bytes | None, str]:
        def check(data: bytes) -> str:
            try:
                check_batch(data, hotkey, params, tokens=tokens, grader=grader)
            except Invalid as exc:
                return exc.rule
            return ""

        return cut_batch(raw, groups, params.max_output_tokens, tokens.output_tokens, check)

    return cut


def cmd_params(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    for name, text in ramp_files(plan, args.short_reply).items():
        (Path(args.out) / name).write_text(text, encoding="utf-8")
        print(Path(args.out) / name)
    return 0


def cmd_copy(args: argparse.Namespace) -> int:
    if not os.environ.get("SN120_R2_PREFIX", "").startswith("smoke"):
        raise ConfigError("SN120_R2_PREFIX must start with 'smoke': this copy is only for a smoke test")
    state, params, stack = read_state(args, "status")
    hotkeys = list(dict.fromkeys(seat.reg.hotkey for seat in waiting(state.queue) if not seat.held))
    src = R2Store(
        endpoint=require_env("V1_R2_ENDPOINT"), bucket=require_env("V1_R2_BUCKET"),
        access_key_id=require_env("V1_R2_ACCESS_KEY_ID"), secret_access_key=require_env("V1_R2_SECRET_ACCESS_KEY"),
    )  # fmt: skip
    count = args.count or params.batch_cap
    copied, passed = copy_files(src, stack.store, hotkeys, count, None if args.whole else admission_cut(stack.admission, args.groups))
    print("\n".join(copied), flush=True)
    print("\n".join(passed), file=sys.stderr)
    print(f"{len(copied)} of {count} files copied, for the first waiting seats in queue order", file=sys.stderr)
    if any(TOKEN_FAULT in line for line in passed):
        print("the tokenizer failed, so this is not a matter of the reply cap: a higher --short-reply does not help. Send the lines above.", file=sys.stderr)
    elif len(copied) < count and not args.whole and params.max_output_tokens < Params().max_output_tokens:
        print(f"too few files have a short group within {params.max_output_tokens} tokens: write the parameter files with a higher --short-reply", file=sys.stderr)
    return 0 if len(copied) == count else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sn120_v2.smoke", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("params", help="write the parameter files of the ramp and of the two rounds at the time limit")
    sub.add_argument("--plan", required=True, help="the output file of python -m sn120_v2.gpu.plan")
    sub.add_argument("--out", required=True, help="the directory for the files")
    sub.add_argument("--short-reply", type=int, default=SHORT_REPLY, help="max_output_tokens of the rounds with a low reply cap")
    sub.set_defaults(handler=cmd_params)
    sub = commands.add_parser("copy", help="copy real queued miner files from v1's bucket into the smoke prefix")
    sub.add_argument("--state-dir", required=True)
    sub.add_argument("--stack", default="sn120_v2.assembly:stack")
    sub.add_argument("--count", type=int, help="how many files (default: the pinned batch_cap), for the first waiting seats in queue order")
    sub.add_argument("--whole", action="store_true", help="copy whole files; the default is a small copy")
    sub.add_argument("--groups", type=int, default=2, help="most groups (prompts) in a small copy")
    sub.set_defaults(handler=cmd_copy)
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except V2Error as exc:
        print(f"refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
