"""Ingest accepted generator tasks keyed by their prior-v1 problem id.

Imported only by the CLI. No accepted-file submitter or rollout metadata enters
public reserve rows or rotation replay inputs. Source and license come from
the verified prior-v1 release, not from accepted-file or third-party text.
"""
from __future__ import annotations

import datetime
import json
import math
import re
from pathlib import Path
from typing import Any

from ...canonical import canonical_json, sha256_hex
from ...errors import V2Error
from . import MANIFEST, normalise, reserve_dir
from .prior import PREFIXES, _existing, _validate, read_rows, verify_release

GENERATOR_METHOD = "generator accepted, measured king pass rate"
FIELDS = {"problem", "answer", "source_id", "prior_p", "submitter", "n", "k"}
EPOCH = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")


def prior_v1_id(problem: str) -> str:
    """The published id: first 20 hex of the normalized problem's SHA-256."""
    return sha256_hex(normalise(problem).encode("utf-8"))[:20]


def _prior_ids(root: Path) -> set[str]:
    """Hash public problem text to recover prior ids after ID allocation.

    The prior batch uses source-prefixed numeric IDs. It retains original row
    provenance, while this hash also covers earlier reserve sources and
    withdrawn/retired rows without needing a live service.
    """
    from . import load_reserve
    reserve, _ = load_reserve(root)
    return {
        prior_v1_id(row["problem"])
        for shard in reserve.shards
        for row in map(json.loads, (root / "shards" / shard.shard).read_bytes().splitlines())
    }


def _epoch(path: Path) -> str:
    name = path.name
    if not name.startswith("accepted-") or not name.endswith(".jsonl"):
        raise V2Error(f"generator input must be named accepted-<epoch>.jsonl: {path.name}")
    epoch = name[len("accepted-"):-len(".jsonl")]
    if not EPOCH.fullmatch(epoch):
        raise V2Error("generator epoch must contain only letters, digits, '.', '_', ':' or '-'")
    return epoch


def _read(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, ValueError) as exc:
        raise V2Error(f"cannot read generator input {path}: {exc}") from exc
    if not rows:
        raise V2Error(f"generator input {path} is empty")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise V2Error(f"generator row {index} must contain exactly {sorted(FIELDS)}")
        if not isinstance(row["problem"], str) or not isinstance(row["answer"], str):
            raise V2Error(f"generator row {index} problem and answer must be strings")
        if not isinstance(row["source_id"], str) or not re.fullmatch(r"[0-9a-f]{20}", row["source_id"]):
            raise V2Error(f"generator row {index} has no prior v1 source_id")
        if prior_v1_id(row["problem"]) != row["source_id"]:
            raise V2Error(f"generator row {index} source_id is not the prior v1 problem hash")
        if type(row["prior_p"]) not in (int, float) or not math.isfinite(row["prior_p"]) or not 0.0 <= row["prior_p"] <= 1.0:
            raise V2Error(f"generator row {index} prior_p must be a finite rate in [0,1]")
        if not isinstance(row["submitter"], str) or not row["submitter"]:
            raise V2Error(f"generator row {index} has no submitter")
        if any(type(row[key]) is not int or row[key] < 0 for key in ("n", "k")):
            raise V2Error(f"generator row {index} n/k must be nonnegative integers")
        if row["k"] > row["n"]:
            raise V2Error(f"generator row {index} k must not exceed n")
    return rows


def _write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        import os
        os.fsync(stream.fileno())


def _metadata(prior_dir: Path) -> dict[str, dict[str, Any]]:
    """Read and bind only rows from the checksum-verified prior release."""
    release = verify_release(prior_dir)
    result = {}
    for row in read_rows(prior_dir, release):
        prior_id = row["id"]
        if not isinstance(prior_id, str) or not re.fullmatch(r"[0-9a-f]{20}", prior_id):
            raise V2Error(f"prior row {prior_id!r} is not a prior-v1 problem hash")
        if prior_v1_id(row["problem"]) != prior_id:
            raise V2Error(f"prior row {prior_id} does not match its normalized problem hash")
        source = str(row["source"]).split("@", 1)[0]
        if source not in PREFIXES:
            raise V2Error(f"prior row {prior_id} has no supported source-prefix mapping")
        if prior_id in result:
            raise V2Error(f"prior release repeats id {prior_id}")
        result[prior_id] = {"source": f"{row['source']}:{row['source_id']}",
                            "license": row["license"], "prefix": PREFIXES[source]}
    return result


def ingest_generator(args: Any) -> dict[str, Any]:
    root = reserve_dir()
    ids, problems, shards, next_numbers = _existing(root)
    known_prior = _prior_ids(root)
    inputs = [(Path(name), _epoch(Path(name))) for name in args.files]
    epochs = [epoch for _, epoch in inputs]
    if len(set(epochs)) != len(epochs):
        raise V2Error("each accepted generator epoch may be supplied only once")
    pending: list[tuple[str, dict[str, Any]]] = []
    counts: dict[str, int] = {}
    for path, epoch in inputs:
        for row in sorted(_read(path), key=lambda row: row["source_id"]):
            source_id = row["source_id"]
            if source_id in known_prior:
                counts["already_ingested"] = counts.get("already_ingested", 0) + 1
                continue
            if normalise(row["problem"]) in problems:
                counts["duplicate_problem"] = counts.get("duplicate_problem", 0) + 1
                continue
            if source_id in ids:
                counts["duplicate_id"] = counts.get("duplicate_id", 0) + 1
                continue
            pending.append((epoch, row))
            known_prior.add(source_id)
    if not pending:
        return {"ingested": 0, "counts": dict(sorted(counts.items())), "message": "nothing new; reserve unchanged"}

    # Validate every accepted task and its original license before writing any
    # shard. A malformed or ungradeable accepted row is never silently admitted.
    metadata = _metadata(Path(args.prior_dir))
    added_at = args.added_at or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    by_epoch: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for epoch, row in pending:
        source_id = row["source_id"]
        original = metadata.get(source_id)
        if original is None:
            raise V2Error(f"generator source_id {source_id} is not in the verified prior-v1 release")
        prefix = original["prefix"]
        task_id = f"{prefix}:{next_numbers.get(prefix, -1) + 1}"
        next_numbers[prefix] = next_numbers.get(prefix, -1) + 1
        task = {
            "id": task_id, "env": "affine_math", "problem": row["problem"], "answer": row["answer"],
            "source": original["source"], "license": original["license"],
            "added_at": added_at, "prior_p": row["prior_p"],
            "audit": {"method": GENERATOR_METHOD, "verdict": "screened"},
        }
        _validate(task)
        proof = {"id": task_id, "source_id": source_id, "submitter": row["submitter"],
                 "n": row["n"], "k": row["k"], "epoch": epoch}
        by_epoch.setdefault(epoch, []).append((task, proof))

    manifest_lines = (root / MANIFEST).read_bytes().splitlines()
    appended: list[bytes] = []
    number = max((int(shard.shard[:4]) for shard in shards), default=0)
    written: list[dict[str, Any]] = []
    for epoch, pairs in sorted(by_epoch.items()):
        number += 1
        safe_epoch = re.sub(r"[^a-z0-9]+", "-", epoch.lower()).strip("-")
        name = f"{number:04d}-generator-accepted-{safe_epoch}.jsonl"
        raw = b"".join(canonical_json(task) + b"\n" for task, _ in pairs)
        line = {"shard": name, "sha256": sha256_hex(raw), "count": len(pairs), "added_at": added_at}
        _write_new(root / "shards" / name, raw)
        _write_new(root / "shards" / (name + ".rejected.json"),
                   canonical_json({"input": f"accepted-{epoch}.jsonl", "rejected": []}) + b"\n")
        _write_new(root / "provenance" / f"generator-{epoch}.jsonl",
                   b"".join(canonical_json(proof) + b"\n" for _, proof in pairs))
        appended.append(canonical_json(line))
        written.append(line | {"epoch": epoch})
    manifest_lines[-1:-1] = appended
    (root / MANIFEST).write_bytes(b"\n".join(manifest_lines) + b"\n")
    from . import load_reserve
    reserve, _ = load_reserve(root)
    return {"ingested": sum(len(pairs) for pairs in by_epoch.values()), "counts": dict(sorted(counts.items())),
            "shards": written, "reserve_manifest_sha256": reserve.manifest_sha,
            "tasks": sum(shard.count for shard in reserve.shards),
            "provenance": [f"provenance/generator-{epoch}.jsonl" for epoch in sorted(by_epoch)]}
