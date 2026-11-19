"""Deadline snapshot: verify all uploads, select owners, and write a private job."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..canonical import sha256_hex
from ..errors import ConfigError, ContractError, VerifyError, V2Error
from .common import PublicSink, atomic_json, json_bytes, object_metadata, open_store, read_json, timestamp, utc_text
from .config import Config, add_config
from .constants import MAX_SUBMISSION_BYTES, POOL_ID, POOL_MANIFEST_URL
from .epoch import epoch_deadline, epoch_id, epoch_start, submission_object
from .payout import winner_key
from .protocol import Submission, valid_task_id, verify_submission


def seed_for(secret: bytes, task_id: str, sample: int) -> int:
    """The committed sampling seed; the secret is revealed only in results."""
    if type(sample) is not int or sample < 0:
        raise ValueError("sample index must be nonnegative")
    return int(hashlib.sha256(secret + b":" + task_id.encode() + b":" + str(sample).encode()).hexdigest()[:12], 16)


def round_robin(owners: Mapping[str, Mapping[str, Any]], hotkeys: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """One task per hotkey per round, preserving each owner's upload order."""
    order = sorted(hotkeys if hotkeys is not None else {row["hotkey"] for row in owners.values()})
    queues: dict[str, deque[str]] = {hotkey: deque() for hotkey in order}
    for task_id, row in sorted(owners.items(), key=lambda item: (winner_key_from_row(item[1]), item[0])):
        if row["hotkey"] in queues:
            queues[row["hotkey"]].append(task_id)
    result = []
    while any(queues.values()):
        for hotkey in order:
            if queues[hotkey]:
                task_id = queues[hotkey].popleft()
                result.append({"task_id": task_id, **owners[task_id]})
    return result


def winner_key_from_row(row: Mapping[str, Any]) -> tuple[float, str]:
    return float(row["uploaded_at"]), str(row["file_sha256"])


def accepted_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    """Read the permanent once-ever intake ledger, rejecting ambiguous history."""
    result = set()
    for row in rows:
        task_id = row.get("task_id", row.get("id"))
        if not valid_task_id(task_id):
            raise VerifyError("accepted-all.jsonl contains an invalid task ID")
        if task_id in result:
            raise VerifyError(f"accepted-all.jsonl contains duplicate task ID {task_id}")
        result.add(task_id)
    return result


def load_history(sink: PublicSink) -> tuple[bytes, set[str]]:
    from .common import read_jsonl
    try:
        raw = sink.read("accepted-all.jsonl")
    except FileNotFoundError:
        return b"", set()
    return raw, accepted_ids(read_jsonl(raw, "accepted-all.jsonl"))


def _safe_submission_key(key: str, epoch: str) -> str | None:
    prefix = f"private/generator/sub/{epoch}/"
    if not key.startswith(prefix) or not key.endswith(".json"):
        return None
    hotkey = key[len(prefix):-5]
    if "/" in hotkey:
        return None
    try:
        if submission_object(epoch, hotkey) != key:
            return None
    except (ContractError, ValueError):
        return None
    return hotkey


def _eligibility(registered: Mapping[int, str], activations: Iterable[str]) -> dict[str, int]:
    active = set(activations)
    return {hotkey: uid for uid, hotkey in registered.items() if type(uid) is int and uid != 0 and hotkey in active}


def build_snapshot(
    store: Any,
    *,
    epoch: str,
    registered: Mapping[int, str],
    activations: Iterable[str],
    known_task_ids: Any,
    previously_accepted: Iterable[str],
    king: str,
    seed_commitment: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Inspect every object at its deadline head, including malformed submissions."""
    epoch = epoch_id(epoch)
    if not isinstance(king, str) or not king:
        raise ConfigError("the frozen king version must be nonempty")
    if len(seed_commitment) != 64 or set(seed_commitment) - set("0123456789abcdef"):
        raise ConfigError("the seed commitment must be a SHA-256 hex digest")
    eligible = _eligibility(registered, activations)
    deadline, start = epoch_deadline(epoch).timestamp(), epoch_start(epoch).timestamp()
    known = known_task_ids if callable(known_task_ids) else lambda task_id: task_id in known_task_ids
    previous = set(previously_accepted)
    files = []
    winners: dict[str, dict[str, Any]] = {}
    counts: dict[str, dict[str, int]] = {}
    for key in store.list(f"private/generator/sub/{epoch}/"):
        hotkey = _safe_submission_key(key, epoch)
        if hotkey is None:
            continue
        metadata = object_metadata(store, key)
        digest = store.sha256(key)
        record: dict[str, Any] = {
            "hotkey": hotkey,
            "sha256": digest,
            "size": metadata["size"],
            "uploaded_at": metadata["uploaded_at"],
            "uploaded_at_utc": utc_text(metadata["uploaded_at"]),
            "valid": False,
            "reason": "",
            "task_count": 0,
            "eligible_count": 0,
        }
        submission = None
        reason = ""
        try:
            if not start <= metadata["uploaded_at"] < deadline:
                reason = "outside_submission_window"
            elif metadata["size"] > MAX_SUBMISSION_BYTES:
                reason = "size_over_16_kib"
            elif hotkey not in eligible:
                reason = "hotkey_not_registered_and_activated"
            else:
                # Only bounded submissions are parsed. Oversized uploads are still listed and hashed.
                raw = store.get(key)
                if len(raw) != metadata["size"] or sha256_hex(raw) != digest:
                    raise VerifyError("object changed while its deadline snapshot was read")
                submission = Submission.from_json(raw)
                if submission.epoch != epoch:
                    reason = "epoch_mismatch"
                elif submission.hotkey != hotkey:
                    reason = "object_hotkey_mismatch"
                elif submission.pool != POOL_ID:
                    reason = "pool_pin_mismatch"
                elif not verify_submission(submission, registered=True):
                    reason = "invalid_schema_or_signature"
        except (ContractError, ValueError, TypeError, VerifyError):
            reason = "invalid_schema_or_signature" if not reason else reason
        record["reason"] = reason
        if submission is not None and not reason:
            record["valid"] = True
            record["task_count"] = len(submission.task_ids)
            count = counts.setdefault(hotkey, {"submitted": 0, "eligible": 0})
            count["submitted"] += len(submission.task_ids)
            for task_id in submission.task_ids:
                if task_id in previous or not known(task_id):
                    continue
                candidate = {
                    "hotkey": hotkey, "uid": eligible[hotkey], "uploaded_at": metadata["uploaded_at"],
                    "file_sha256": digest,
                }
                if task_id not in winners or winner_key_from_row(candidate) < winner_key_from_row(winners[task_id]):
                    winners[task_id] = candidate
        files.append(record)
    for task_id, row in winners.items():
        counts[row["hotkey"]]["eligible"] += 1
    for record in files:
        if record["valid"]:
            record["eligible_count"] = counts.get(record["hotkey"], {}).get("eligible", 0)
            if not record["eligible_count"]:
                record["reason"] = "no_new_tasks_in_pinned_pool"
    manifest = {
        "version": "affine-generator-manifest-v1",
        "epoch": epoch,
        "deadline": utc_text(deadline),
        "snapshot_at": utc_text(deadline + 60),
        "pool": POOL_ID,
        "files": sorted(files, key=lambda row: (row["hotkey"], row["uploaded_at"], row["sha256"])),
        "unique_tasks": len(winners),
        "note": "Object hashes and counts only. Task IDs are private until results publication.",
    }
    # Private head-version evidence is bound to each unique task, not just stated in a sidecar.
    for row in winners.values():
        row["head_version"] = king
        row["head_version_at_deadline"] = king
    job = {
        "version": "affine-generator-job-v1",
        "epoch": epoch,
        "created_at": utc_text(deadline + 60),
        "deadline": utc_text(deadline),
        "results_deadline": utc_text(deadline + 3 * 3600 + 45 * 60),
        "pool": POOL_ID,
        "pool_manifest_url": POOL_MANIFEST_URL,
        "king": king,
        "seed_commitment": seed_commitment,
        "tasks": round_robin(winners),
        "counts": counts,
    }
    return manifest, job


def scorer_job(job: Mapping[str, Any], secret: str) -> dict[str, Any]:
    """Exact input contract of the sibling scorer; evidence remains private."""
    if not isinstance(secret, str) or not secret or sha256_hex(secret.encode()) != job["seed_commitment"]:
        raise VerifyError("private seed secret does not match the job commitment")
    if not job["tasks"]:
        raise ConfigError("an empty epoch is published locally without a scorer job")
    return {
        "version": "affine-generator-scoring-job-v1", "epoch": job["epoch"],
        "tasks": [{"id": row["task_id"], "hotkey": row["hotkey"]} for row in job["tasks"]],
        "head_version": job["king"], "seed_secret": secret,
        "budget_deadline": epoch_deadline(job["epoch"]).timestamp() + 3 * 3600 + 45 * 60,
        "pool_pin": POOL_ID,
    }


def fetch_head(url: str, path: Path, *, now: float) -> dict[str, Any]:
    """Capture public head.json at T, without accepting an unbounded response."""
    request = urllib.request.Request(url, headers={"User-Agent": "sn120-generator-control/1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as reply:  # noqa: S310 - explicit public config URL
            raw = reply.read(1024 * 1024 + 1)
    except OSError as exc:
        raise ConfigError(f"cannot read frozen king document: {type(exc).__name__}") from exc
    if len(raw) > 1024 * 1024:
        raise VerifyError("frozen king document exceeds 1 MiB")
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise VerifyError("frozen king document is not JSON") from exc
    version = head_version(document)
    result = {"version": version, "captured_at": utc_text(now), "source_url": url, "document": document}
    atomic_json(path, result)
    return result


def head_version(document: Mapping[str, Any]) -> str:
    """Read an existing public head pointer without inventing a model identifier."""
    for key in ("version", "head_version", "head_version_id", "id"):
        if isinstance(document.get(key), str) and document[key]:
            return document[key]
    head = document.get("head")
    if isinstance(head, Mapping):
        return head_version(head)
    raise VerifyError("public head.json contains no explicit head version")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sn120_v2.generator.snapshot")
    add_config(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--epoch")
    group.add_argument("--due", action="store_true")
    parser.add_argument("--now", type=float, help="testing clock, Unix seconds")
    args = parser.parse_args(argv)
    try:
        import time
        now = time.time() if args.now is None else args.now
        config = Config.load(args.config)
        epoch = epoch_id(now - 61) if args.due else epoch_id(args.epoch)
        if now < epoch_deadline(epoch).timestamp() + 60:
            raise ConfigError("an epoch snapshot is due at its deadline plus one minute")
        king_path = config.work_dir / "king" / f"{epoch}.json"
        if not king_path.is_file():
            raise ConfigError(f"missing frozen king {king_path}; run generator.tick at the deadline")
        king = read_json(king_path)
        if king.get("epoch", epoch) != epoch:
            raise ConfigError("the frozen king belongs to another epoch")
        commitment = read_json(config.work_dir / "seeds" / f"{epoch}.json")["seed_commitment"]
        store = open_store(config.credentials_file)
        sink = PublicSink(config.site_dir, store, config.public_url)
        relative = f"epochs/{epoch}/manifest.json"
        job_path = config.work_dir / "jobs" / epoch / "evidence.json"
        if sink.exists(relative) and job_path.is_file():
            print(f"snapshot {epoch}: already complete")
            print(f"snapshot {epoch}: already complete")
            return 0
        from ..chain import Reader, SubtensorClient
        from ..links import read_activated
        from .pool import PriorPool
        client = SubtensorClient.connect(120, config.network)
        try:
            snapshot = Reader(client, 120).snapshot()
            registered = {reg.uid: reg.hotkey for reg in snapshot.registrations}
            activated = read_activated(client, registered.values(), 120)
        finally:
            client.chain.close()
        history, previous = load_history(sink)
        pool = PriorPool(config.pool_dir)
        manifest, job = build_snapshot(
            store, epoch=epoch, registered=registered, activations=activated,
            known_task_ids=pool, previously_accepted=previous,
            king=king["version"], seed_commitment=commitment,
        )
        # Publish hashes immediately; private evidence and seed stay off public R2.
        sink.put(relative, json_bytes(manifest))
        atomic_json(job_path, job)
        if job["tasks"]:
            atomic_json(job_path.with_name("job.json"), scorer_job(job, read_json(config.work_dir / "seeds" / f"{epoch}.json")["secret"]))
        print(f"snapshot {epoch}: {len(manifest['files'])} files, {len(job['tasks'])} unique eligible tasks")
        return 0
    except V2Error as exc:
        print(f"snapshot failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
