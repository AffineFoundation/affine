"""Ingest, verify, inspect and withdraw public reserve tasks.

    python -m sn120_v2.envs.reserve ingest --source NAME@REV --license CC-BY-4.0 \
      --prefix omni --audit-method audited --audit-ref REPORT.json input.jsonl

Input rows have ``problem``, ``answer`` and optional ``source_id`` and
``prior_p`` (a finite public prior pass rate in [0, 1]). No active
pool is written. A rejected row goes to ``<shard>.rejected.json``, never to
the reserve shard.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from ...canonical import canonical_json, sha256_hex
from ...errors import V2Error
from ..math_pool import MANDATE, POOL_V2_0_FILE, POOL_V2_0_SHA256, Task, load_pool
from . import (
    ID, MANIFEST, MIN_ANSWER_CHARS, MAX_ANSWER_CHARS, MAX_PROBLEM_CHARS, WITHDRAWN, atomic_write, empty_manifest, ingest_lock, load_reserve, normalise, read_raw,
    reserve_dir, task_row,
)


def utc_now() -> str:
    """The specified UTC ISO date, not a wall-clock timestamp."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _write_new(path: Path, data: bytes) -> None:
    """Reserve shards are immutable: never replace one that exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def retired(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "retired.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_bytes())
    if not isinstance(rows, list):
        raise V2Error("retired archive must be a list")
    return {row["id"]: row for row in rows}


def _retired_catalog(rows: dict[str, dict[str, Any]], root: Path) -> dict[str, Task]:
    path = root / "retired_tasks.json"
    if not path.exists():
        return {}
    return {row["id"]: Task(row["id"], row["env"], row["problem"], row["answer"]) for row in json.loads(path.read_bytes())}


def ingest(args: argparse.Namespace) -> dict[str, Any]:
    root = reserve_dir()
    with ingest_lock(root, exclusive=True):
        return _ingest_locked(args, root)


def _ingest_locked(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    if not ID.fullmatch(args.prefix + ":0") or args.prefix in ("math", "dapo"):
        raise V2Error("prefix must be lowercase alphanumeric/underscore, and not reserved math or dapo")
    reserve, catalog = load_reserve(root, with_catalog=True)
    retired_rows = retired(root)
    retired_tasks = {task_id: catalog.task(task_id) for task_id in retired_rows if task_id in catalog.task_ids}
    if retired_rows.keys() - retired_tasks.keys():
        # The controller mirrors complete Task rows separately from the small
        # public retired index, solely for exact readmission and prompt grading.
        retired_tasks.update(_retired_catalog(retired_rows, root))
    readmit = set(args.readmit or ())
    if readmit - set(retired_rows) or readmit - set(retired_tasks):
        raise V2Error("readmit needs retired ids with their archived original rows")
    used = {normalise(catalog.task(task_id).problem): task_id for task_id in catalog.task_ids}
    readmitted = {row["id"] for shard in reserve.shards for row in (json.loads(line) for line in (root / "shards" / shard.shard).read_bytes().splitlines())
                  if "readmit_of" in row["audit"]}
    # Withdrawn rows remain dedupe evidence, even though they leave the
    # admission catalog. Only an explicitly readmitted retired row is exempt.
    for shard in reserve.shards:
        for raw in (root / "shards" / shard.shard).read_text(encoding="utf-8").splitlines():
            row = json.loads(raw)
            used.setdefault(normalise(row["problem"]), row["id"])
    for task in retired_tasks.values():
        used.setdefault(normalise(task.problem), task.task_id)
    ids = set(catalog.task_ids) | set(retired_rows) | set(reserve.withdrawn_ids)
    drops: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    prefix_max: dict[str, int] = {}
    for task in ids:
        prefix, number = task.rsplit(":", 1)
        prefix_max[prefix] = max(prefix_max.get(prefix, -1), int(number))
    next_n = prefix_max.get(args.prefix, -1) + 1
    if args.reject_mc:
        from ..math_grader import _options
    try:
        source_rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()]
    except (OSError, ValueError) as exc:
        raise V2Error(f"cannot read ingest input: {exc}") from exc
    for number, row in enumerate(source_rows):
        reason = ""
        if not isinstance(row, dict) or not {"problem", "answer"} <= set(row) or set(row) - {"problem", "answer", "source_id", "prior_p"}:
            reason = "schema"
        problem, answer = str(row.get("problem", "")), str(row.get("answer", "")) if isinstance(row, dict) else ("", "")
        old_id = str(row.get("source_id", "")) if isinstance(row, dict) else ""
        task_id = old_id if old_id in readmit else f"{args.prefix}:{next_n}"
        if old_id in readmit:
            old = retired_tasks[old_id]
            reason = "" if (problem, answer) == (old.problem, old.answer) else "readmit_changed"
        norm = normalise(problem)
        if not reason and (task_id in ids or (norm in used and used[norm] != task_id)):
            if task_id in ids and old_id not in readmit:
                reason = "duplicate_id"
            elif old_id in readmit and task_id in readmitted:
                reason = "already_readmitted"
            elif norm in used and used[norm] != task_id:
                reason = "duplicate_input" if norm in {normalise(item["problem"]) for item in kept} else "duplicate_problem"
        if not reason and norm in {normalise(item["problem"]) for item in kept}:
            reason = "duplicate_input"
        if not reason and ("prior_p" in row and (type(row["prior_p"]) not in (int, float) or not 0.0 <= row["prior_p"] <= 1.0)):
            reason = "prior_p"
        if not reason and (not problem.strip() or len(problem) > MAX_PROBLEM_CHARS):
            reason = "problem_length"
        if not reason and (not MIN_ANSWER_CHARS <= len(answer.strip()) <= MAX_ANSWER_CHARS or any(c in answer for c in "\r\n\x00")):
            reason = "answer_format"
        if not reason and MANDATE in problem:
            reason = "mandate"
        if not reason and args.reject_mc and _options(problem):
            reason = "multiple_choice"
        if not reason:
            from ..math_grader import grade_answer
            grade = grade_answer(answer, problem, "<think>x</think> \\boxed{" + answer + "}", False)
            if grade.code != "P":
                reason = "grader_self_check"
        if reason:
            drops.append({"row": number, "source_id": old_id or None, "reason": reason})
        else:
            if old_id not in readmit:
                next_n += 1
            audit = {"verdict": "keep", "method": args.audit_method, "ref": args.audit_ref}
            if old_id in readmit:
                audit["readmit_of"] = retired_rows[old_id]["round_id"]
            kept.append(task_row(Task(task_id, args.env, problem, answer), source=f"{args.source}:{old_id or number}",
                                 license=args.license, added_at=utc_now(), audit=audit,
                                 **({"prior_p": row["prior_p"]} if "prior_p" in row else {})))
            ids.add(task_id)
            used[norm] = task_id
    if not kept:
        return {"ingested": 0, "rejected": len(drops), "counts": {reason: sum(row["reason"] == reason for row in drops) for reason in sorted({row["reason"] for row in drops})}, "message": "nothing kept; reserve unchanged"}
    slug = re.sub(r"[^a-z0-9]+", "-", args.source.lower()).strip("-")[:50] or "tasks"
    listed = list(reserve.shards)
    # One source+input is one shard. Replaying that exact command at a later
    # date, or through the CLI with the same provenance, reproduces its bytes
    # instead of creating an empty duplicate. An altered same-source input is
    # rejected rather than silently taking a different timestamped identity.
    source_name = f"{args.source}:{Path(args.input).name}"
    for shard in listed:
        rows = [json.loads(line) for line in (root / "shards" / shard.shard).read_bytes().splitlines()]
        if rows and rows[0].get("source", "").split(":", 1)[0] == source_name:
            same = [(row["id"], row["problem"], row["answer"], row["audit"], row.get("prior_p")) for row in rows]
            proposed = [(row["id"], row["problem"], row["answer"], row["audit"], row.get("prior_p")) for row in kept]
            if same != proposed:
                raise V2Error(f"source {source_name!r} already has shard {shard.shard} with different content")
            report = {"shard": shard.shard, "sha256": shard.sha256, "ingested": 0, "rejected": 0,
                      "counts": {}, "already_ingested": len(rows), "next_ids": [row["id"] for row in rows[:32]]}
            sidecar = root / "shards" / (shard.shard + ".rejected.json")
            if sidecar.exists():
                report["rejected"] = len(json.loads(sidecar.read_bytes())["rejected"])
                report["counts"] = {reason: sum(row["reason"] == reason for row in json.loads(sidecar.read_bytes())["rejected"])
                                    for reason in sorted({row["reason"] for row in json.loads(sidecar.read_bytes())["rejected"]})}
            return report
    number = max((int(row.shard[:4]) for row in listed), default=0) + 1
    name = f"{number:04d}-{slug}.jsonl"
    # One UTC date per shard makes its bytes independent of execution speed.
    stamp = utc_now()
    for row in kept:
        row["added_at"] = stamp
    raw = b"".join(canonical_json(row) + b"\n" for row in kept)
    shard_line = {"shard": name, "sha256": sha256_hex(raw), "count": len(kept), "added_at": stamp}
    _write_new(root / "shards" / name, raw)
    _write_new(root / "shards" / (name + ".rejected.json"), canonical_json({"input": Path(args.input).name, "rejected": drops}) + b"\n")
    # The withdrawn pointer must stay last. Append the shard before it, not
    # after it; append-only verification covers the complete earlier prefix.
    history = (root / MANIFEST).read_bytes().splitlines()
    history[-1:-1] = [canonical_json(shard_line)]
    atomic_write(root / MANIFEST, b"\n".join(history) + b"\n")
    # The shard filename is source-scoped, so input collisions never replace
    # bytes of another source.
    load_reserve(root)  # verify the committed bytes, fail loudly on any problem
    return {"shard": name, "sha256": shard_line["sha256"], "ingested": len(kept), "rejected": len(drops),
            "counts": {reason: sum(row["reason"] == reason for row in drops) for reason in sorted({row["reason"] for row in drops})},
            "next_ids": [row["id"] for row in kept[:32]]}


def withdraw(args: argparse.Namespace) -> dict[str, Any]:
    root = reserve_dir()
    with ingest_lock(root, exclusive=True):
        return _withdraw_locked(args, root)


def _withdraw_locked(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    reserve, _ = load_reserve(root)
    if args.id in reserve.withdrawn_ids:
        raise V2Error(f"{args.id} is already withdrawn")
    row = {"id": args.id, "reason": args.reason.strip(), "added_at": utc_now()}
    if not row["reason"]:
        raise V2Error("withdraw needs a nonempty reason")
    # One rewrite of the pointer is intentional: the previous pointer and
    # withdrawn bytes are sealed in the preceding round's Rotation history.
    with (root / WITHDRAWN).open("ab") as stream:
        stream.write(canonical_json(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    raw = (root / WITHDRAWN).read_bytes()
    history = (root / MANIFEST).read_bytes().splitlines()
    if "withdrawn" not in json.loads(history[-1]):
        raise V2Error("reserve manifest has no withdrawn pointer")
    history[-1] = canonical_json({"withdrawn": {"sha256": sha256_hex(raw), "count": len(raw.splitlines())}})
    atomic_write(root / MANIFEST, b"\n".join(history) + b"\n")
    load_reserve(root)
    return row


def withdraw_many_locked(root: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    """Withdraw a batch under one pointer rewrite, with the same invariants as one withdrawal.

    The caller already holds the exclusive ``ingest_lock``. A batch is one
    rewrite of the manifest pointer rather than one per id: the previous
    pointer is sealed in the preceding round, so a round sees either the whole
    batch or none of it. An id already withdrawn, a repeat inside the batch, or
    an empty reason fails the whole batch before anything is written.
    """
    reserve, _ = load_reserve(root)
    stamp = utc_now()
    prepared: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        task_id, reason = row["id"], str(row.get("reason", "")).strip()
        if not reason:
            raise V2Error(f"withdraw needs a nonempty reason for {task_id}")
        if task_id in reserve.withdrawn_ids:
            raise V2Error(f"{task_id} is already withdrawn")
        if task_id in seen:
            raise V2Error(f"{task_id} is withdrawn twice in one batch")
        seen.add(task_id)
        prepared.append({"id": task_id, "reason": reason, "added_at": stamp})
    if not prepared:
        return {"withdrawn": 0, "ids": []}
    with (root / WITHDRAWN).open("ab") as stream:
        stream.write(b"".join(canonical_json(row) + b"\n" for row in prepared))
        stream.flush()
        os.fsync(stream.fileno())
    raw = (root / WITHDRAWN).read_bytes()
    history = (root / MANIFEST).read_bytes().splitlines()
    if "withdrawn" not in json.loads(history[-1]):
        raise V2Error("reserve manifest has no withdrawn pointer")
    history[-1] = canonical_json({"withdrawn": {"sha256": sha256_hex(raw), "count": len(raw.splitlines())}})
    atomic_write(root / MANIFEST, b"\n".join(history) + b"\n")
    load_reserve(root)  # verify the committed bytes, fail loudly on any problem
    return {"withdrawn": len(prepared), "ids": [row["id"] for row in prepared], "added_at": stamp}


def status() -> dict[str, Any]:
    reserve, _ = load_reserve()
    return {"shards": len(reserve.shards), "tasks": sum(row.count for row in reserve.shards),
            "withdrawn": reserve.withdrawn_count, "queued": len(reserve.queue),
            "reserve_manifest_sha": reserve.manifest_sha, "next_ids": list(reserve.queue[:64])}


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="python -m sn120_v2.envs.reserve", description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)
    add = commands.add_parser("ingest")
    add.add_argument("input")
    for flag in ("source", "license", "prefix", "audit-method", "audit-ref"):
        add.add_argument("--" + flag, required=True)
    add.add_argument("--reject-mc", action="store_true")
    add.add_argument("--env", default="affine_math", help="task slice (default: affine_math, scored with the base pool)")
    add.add_argument("--readmit", action="append", help="retired id to readmit explicitly; repeatable")
    generator = commands.add_parser("ingest-generator")
    generator.add_argument("input")
    generator.add_argument("--epoch", required=True)
    generator.add_argument("--results")
    generator.add_argument("--done", help="accepted-EPOCH.done; defaults to the JSONL path with .done suffix")
    generator.add_argument("--prior-index")
    generator.add_argument("--active", help="public active id list; defaults to the current public summary's active ids")
    generator.add_argument("--public-url", default="https://affine.io")
    generator.set_defaults(reject_mc=True)
    commands.add_parser("verify")
    commands.add_parser("status")
    remove = commands.add_parser("withdraw")
    remove.add_argument("id")
    remove.add_argument("--reason", required=True)
    return top


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "ingest-generator":
            from .generator import ingest_generator
            report = ingest_generator(args)
        elif args.command == "ingest":
            report = ingest(args)
        elif args.command == "withdraw":
            report = withdraw(args)
        elif args.command == "status":
            report = status()
        else:
            report = status() | {"verified": True}
    except (V2Error, OSError, ValueError) as exc:
        if args.command == "ingest-generator":
            from .generator import Incomplete, InputChanged
            if isinstance(exc, (Incomplete, InputChanged)):
                print(json.dumps({"error": "incomplete" if isinstance(exc, Incomplete) else "input_changed", "reason": str(exc)}), file=sys.stderr)
                return 3 if isinstance(exc, Incomplete) else 4
        print(json.dumps({"error": type(exc).__name__, "reason": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
