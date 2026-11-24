"""Offline re-check of a crown confirmation whose parent evaluation failed.

    python -m sn120_v2.recheck export --state-dir DIR --round rNNNNNN --out bundle.json
    python -m sn120_v2.recheck links --state-dir DIR --round rNNNNNN --out links.json [--ttl 21600]
    python -m sn120_v2.recheck score --bundle bundle.json --work-dir W [--n-tasks N] [--gpus K] [--links links.json] --out verdict.json

Export copies the journal to a temporary directory before replaying it. It never
opens the journal writer, takes its lock, or writes below the state directory.
Signed GET URLs are secrets: ``links`` writes them only to a mode-0600 file.

Scoring uses the original sealed draw and new-head outcomes, the live
``VllmEvaluator.score_parent`` and ``judging.score_confirmation``. It does not
change the journal, the store, chain weights, or any production service.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import layout, state
from .canonical import canonical_json, sha256_hex
from .errors import ConfigError, InfraError, V2Error, VerifyError
from .headchain import file_set
from .journal import JOURNAL_FILE, read_journal
from .judging import legacy_parent_one_sample, score_confirmation
from .params import GenesisSpec, Params
from .state import CONFIRMATION_RESOLVED, GENESIS, PARAMS_SET, ROUND_OPENED, ROUND_SETTLED, RoundRecord
from .store import LocalStore
from .types import BoxResult, Confirmation, RoundTicket

BUNDLE_SCHEMA = "sn120-v2-offline-recheck-2"
TASK_SCHEMA = "sn120-v2-task-record-1"
LINKS_SCHEMA = "sn120-v2-read-links-1"
PUBLIC_ROOT = "https://affine.io"
LEGACY_NOTE = "crown round used 1-sample design; scored under live 2-sample contract"
SANITY_NOTE = "sanity prefix only; not a full-draw confirmation"
DEFAULT_TTL = 21600
_URL = re.compile(r"https?://\S+")


class _LocalMirror(LocalStore):
    """A local store reader whose constructor cannot create a directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()


class SigningStore(LocalStore):
    """A local evidence mirror can sign its own public objects, without R2 credentials."""

    def presign_get(self, key: str, ttl_s: int) -> str:
        return self.root.joinpath(*key.split("/")).resolve().as_uri()


def _json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read JSON file {path}: {type(exc).__name__}") from None
    if not isinstance(value, dict):
        raise ConfigError(f"JSON file {path} is not an object")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any], *, private: bool = False) -> None:
    """Atomically write canonical JSON; private output is 0600 before any bytes are written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        if private:
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as out:
            out.write(canonical_json(value) + b"\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, target)
        if private:
            target.chmod(0o600)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise


def _journal(state_dir: str | Path) -> tuple[bytes, list[Any], Mapping[str, RoundRecord], Mapping[str, Confirmation], Mapping[str, Mapping[str, Any]]]:
    """Replay only a temporary copy; retain records that bounded State.history would drop."""
    source = Path(state_dir) / JOURNAL_FILE
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read the journal copy source: {type(exc).__name__}") from None
    with tempfile.TemporaryDirectory(prefix="sn120-recheck-journal-") as temporary:
        copied = Path(temporary) / JOURNAL_FILE
        copied.write_bytes(raw)
        loaded = state.load(copied)  # read_journal only: no Journal, pull, flock or store mirror
        records = read_journal(copied)
    rounds: dict[str, RoundRecord] = {}
    confirmations: dict[str, Confirmation] = {}
    params_at_open: dict[str, Mapping[str, Any]] = {}
    current_params: Mapping[str, Any] | None = None
    for record in records:
        if record.kind == PARAMS_SET:
            current_params = record.data["params"]
        elif record.kind == ROUND_OPENED:
            opened = RoundRecord.from_dict(record.data["round"])
            rounds[opened.round_id] = opened
            if current_params is not None:
                params_at_open[opened.round_id] = current_params
        elif record.kind == ROUND_SETTLED:
            opened = rounds[record.data["round_id"]]
            digests = record.data["results"]
            leases = tuple(dataclasses.replace(lease, status="done", result_digest=digests[lease.lease_id])
                           if lease.lease_id in digests else lease for lease in opened.leases)
            rounds[opened.round_id] = dataclasses.replace(opened, leases=leases)
        elif record.kind == CONFIRMATION_RESOLVED:
            confirmations[record.data["round_id"]] = Confirmation.from_dict(record.data["confirmation"])
    if loaded.last_seq != (records[-1].seq if records else -1):
        raise VerifyError("the temporary journal copy did not replay completely")
    return raw, records, rounds, confirmations, params_at_open


def _evidence_store(state_dir: str | Path, *, signing: bool = False) -> Any:
    if not signing:
        mirror = Path(state_dir) / "store"
        if mirror.is_dir():
            return _LocalMirror(mirror)
    # Link issuance must use the authoritative object store, not a possibly
    # incomplete production disk cache. Tests explicitly inject SigningStore.
    from .r2 import R2Store  # boto3 is needed only for production evidence/links
    return R2Store.from_env()


def _training_evidence(store: Any, pending: Any) -> dict[str, Any]:
    """The winner's own training file, exactly as the controller used at round open."""
    key = layout.winner_batch(pending.crown_round_id)
    try:
        raw = store.get(key)
        if sha256_hex(raw) != pending.batch_sha256:
            return {"available": False, "reason": "stored bytes differ from the pending crown batch hash"}
        body = json.loads(raw)
        from .admission import FORMATS
        if body.get("version") not in FORMATS or not isinstance(body.get("groups"), list):
            raise ValueError("not a groups file")
        groups = []
        for group in body["groups"]:
            if not isinstance(group, dict) or not isinstance(group.get("prompt"), str):
                continue
            rollouts = group.get("rollouts")
            groups.append({"prompt": group["prompt"], "rollouts": rollouts if isinstance(rollouts, list) else []})
        return {
            "available": True, "sha256": sha256_hex(raw), "version": body["version"],
            "hotkey": body.get("hotkey", ""), "groups": groups, "file": body,
            "file_json_b64": base64.b64encode(raw).decode("ascii"),
            "exclusion_status": "read_and_hash_verified",
        }
    except KeyError:
        return {"available": False, "reason": "winner batch object is unavailable; live exclusion was best-effort and may have been empty"}
    except (ValueError, TypeError, UnicodeError):
        return {"available": False, "reason": "winner batch is not a valid affine-batch groups file; live exclusion was best-effort and may have been empty"}


def _read_ticket(store: Any, record: RoundRecord, lease: Any) -> RoundTicket:
    try:
        raw = store.get(layout.ticket(record.round_id, lease.lease_id))
    except KeyError:
        raise VerifyError(f"sealed ticket is missing for lease {lease.lease_id}") from None
    ticket = RoundTicket.from_json(raw)
    if ticket.digest() != lease.ticket_digest:
        raise VerifyError(f"sealed ticket does not match lease {lease.lease_id}")
    return ticket


def _read_result(store: Any, record: RoundRecord, lease: Any, ticket: RoundTicket) -> BoxResult | None:
    try:
        raw = store.get(layout.sealed_result(record.round_id, lease.lease_id))
    except KeyError:
        return None
    result = BoxResult.from_json(raw)
    if result.digest() != lease.result_digest or result.ticket_digest != ticket.digest():
        raise VerifyError(f"sealed result does not match lease {lease.lease_id}")
    if (result.round_id, result.lease_id, result.box_id) != (record.round_id, lease.lease_id, lease.box_id):
        raise VerifyError(f"sealed result identity does not match lease {lease.lease_id}")
    return result


def _pending(record: RoundRecord) -> Any:
    if record.parent is None or record.parent.manifest is None:
        raise ConfigError("the round has no pinned parent HeadManifest to re-check")
    pending = record.parent.pending
    if record.confirmation is None or record.confirmation.confirmed or not record.confirmation.reason.startswith("no_decision"):
        raise ConfigError("the selected round is not an unresolved, unpaid crown confirmation")
    confirmation = record.confirmation
    identity = (confirmation.pending_round_id, confirmation.pending_uid, confirmation.parent_version_id, confirmation.new_version_id)
    expected = (pending.crown_round_id, pending.uid, pending.parent_version_id, pending.new_version_id)
    if identity != expected or record.parent.manifest.head.version_id != pending.parent_version_id:
        raise VerifyError("the pending crown, parent manifest and resolved confirmation do not match")
    return pending


def _task_record(task: Any) -> dict[str, Any]:
    """The exact live Task fields used for both chat messages and reference grading."""
    body = {"id": task.task_id, "env": task.slice, "problem": task.problem, "answer": task.answer}
    if any(not isinstance(value, str) for value in body.values()):
        raise VerifyError("a live pool task has a non-string task field")
    return {"schema": TASK_SCHEMA, **body, "sha256": sha256_hex(canonical_json(body))}


def _task_evidence(store: Any, record: RoundRecord, draw: Any) -> dict[str, Any]:
    """Resolve the draw from immutable public inputs, without PoolRuntime I/O.

    The rotation carries the pre-decision reserve history. Its full shards still
    contain earlier admissions, so no current reserve/retired pointer is read.
    """
    from .envs.math_pool import POOL_V2_0_FILE, POOL_V2_0_SHA256, Task, load_pool
    from .envs.reserve import load_reserve
    from .irt import SCHEMA, verify_snapshot
    from .pool_runtime import load_active
    snapshot_raw = store.get(layout.irt_snapshot(draw.snapshot_id))
    snapshot = json.loads(snapshot_raw)
    if snapshot.get("schema") == SCHEMA:
        verify_snapshot(snapshot)
        if set(draw.task_ids) - set(snapshot["tasks"]):
            raise VerifyError("the sealed draw has a task outside its IRT snapshot")
    elif snapshot.get("method") != "uniform placeholder: every pool task has the same chance":
        raise VerifyError("the draw snapshot is not a supported live sampler snapshot")
    # Reserve tasks must not switch the packaged pool from current to archive.
    # Live Catalog lookup prefers current, then archive, then reserve records.
    current = load_pool()
    archive = load_pool(POOL_V2_0_FILE, POOL_V2_0_SHA256)
    provenance: dict[str, Any] = {
        "snapshot_object": layout.irt_snapshot(draw.snapshot_id),
        "snapshot_sha256": sha256_hex(snapshot_raw),
        "base_pool_sha256": {"current": current.sha256, "archive": archive.sha256},
        "record_schema": TASK_SCHEMA,
        "record_hash": "sha256 of canonical JSON {id,env,problem,answer}",
        "lookup": "VllmEvaluator uses this bundle pool for both task.messages() rollout prompts and task.answer grading",
    }
    reserve_tasks: dict[str, Any] = {}
    if record.rotation is None:
        provenance["rotation"] = None
    else:
        block = record.rotation
        active_ids = load_active(store, block["active_sha"])
        if len(active_ids) != block["size"] or set(draw.task_ids) - set(active_ids):
            raise VerifyError("the sealed draw has a task outside its rotated active set")
        rotation_raw = store.get(layout.pool_rotation(record.round_id))
        from .rotation import Rotation
        rotation = Rotation.from_json(rotation_raw)
        if (rotation.round_id != record.round_id or rotation.active_sha != block["active_sha"]
                or rotation.digest() != block["state_sha"] or rotation.reserve_manifest_sha != block["reserve_manifest_sha"]
                or rotation.journal_block() != {key: value for key, value in block.items() if key not in ("state_sha", "size")}):
            raise VerifyError("the round rotation does not match its journal seals")
        history = rotation.reserve_history
        manifest = b"".join(canonical_json(row) + b"\n" for row in [*history["shards"], {
            "withdrawn": {"sha256": history["withdrawn_sha"], "count": history["withdrawn_count"]},
        }])
        withdrawn = b"".join(canonical_json(row) + b"\n" for row in history["withdrawn"])
        if sha256_hex(manifest) != history["manifest_sha"] or sha256_hex(withdrawn) != history["withdrawn_sha"]:
            raise VerifyError("the rotation reserve history does not match its content hashes")
        shard_objects = {row["shard"]: layout.reserve_shard(row["shard"], row["sha256"]) for row in history["shards"]}
        withdrawn_ids = {row["id"] for row in history["withdrawn"]}
        with tempfile.TemporaryDirectory(prefix="sn120-recheck-pool-") as temporary:
            root = Path(temporary)
            (root / "shards").mkdir()
            (root / "MANIFEST.jsonl").write_bytes(manifest)
            (root / "withdrawn.jsonl").write_bytes(withdrawn)
            for row in history["shards"]:
                raw = store.get(shard_objects[row["shard"]])
                if sha256_hex(raw) != row["sha256"]:
                    raise VerifyError(f"reserve shard {row['shard']} differs from its rotation hash")
                (root / "shards" / row["shard"]).write_bytes(raw)
            # Verify the exact reserve bytes and schemas using the live verifier.
            # Read task text from those bytes directly: unlike today's Catalog,
            # this retains tasks consumed or retired by earlier sealed rounds.
            load_reserve(root)
            for shard in history["shards"]:
                for line in (root / "shards" / shard["shard"]).read_bytes().splitlines():
                    row = json.loads(line)
                    if row["id"] not in withdrawn_ids:
                        reserve_tasks[row["id"]] = Task(row["id"], row["env"], row["problem"], row["answer"])
        provenance["rotation"] = {
            "rotation_object": layout.pool_rotation(record.round_id),
            "rotation_sha256": sha256_hex(rotation_raw),
            "active_object": layout.active_pool(block["active_sha"]),
            "active_sha256": block["active_sha"],
            "reserve_manifest_sha256": history["manifest_sha"],
            "reserve_history_sha256": sha256_hex(canonical_json(history)),
            "reserve_shard_objects": shard_objects,
        }
    records = []
    for task_id in draw.task_ids:
        try:
            task = current.task(task_id)
        except Exception:
            try:
                task = archive.task(task_id)
            except Exception:
                if task_id not in reserve_tasks:
                    raise VerifyError(f"draw task {task_id!r} is not in the verified historical pool") from None
                task = reserve_tasks[task_id]
        records.append(_task_record(task))
    provenance["task_count"] = len(records)
    return {"provenance": provenance, "records": records}


def _bundle_pool(body: Mapping[str, Any], task_ids: Sequence[str]) -> Any:
    """Build a draw-only live-compatible pool, with no local task fallback."""
    from .envs.math_pool import MathPool, Task
    evidence = body["tasks"]
    records = evidence["records"]
    if not isinstance(records, list) or len(records) != len(body["draw"]["task_ids"]):
        raise VerifyError("bundle task evidence does not cover the exact draw")
    tasks = []
    for task_id, row in zip(body["draw"]["task_ids"], records):
        if not isinstance(row, dict) or row.get("schema") != TASK_SCHEMA or set(row) != {"schema", "id", "env", "problem", "answer", "sha256"}:
            raise VerifyError("a bundle task record has the wrong schema")
        fields = {key: row[key] for key in ("id", "env", "problem", "answer")}
        if any(not isinstance(value, str) for value in fields.values()) or row["id"] != task_id:
            raise VerifyError(f"bundle task {task_id!r} is missing or has the wrong identity")
        if row["sha256"] != sha256_hex(canonical_json(fields)):
            raise VerifyError(f"bundle task {task_id!r} hash differs from its prompt or grading reference")
        tasks.append(Task(task_id, row["env"], row["problem"], row["answer"]))
    pool = MathPool(tuple(tasks), "bundle-" + sha256_hex(canonical_json(records)))
    if set(task_ids) - set(pool.task_ids):
        missing = next(task_id for task_id in task_ids if task_id not in pool.task_ids)
        raise VerifyError(f"bundle task {missing!r} is missing; refusing local pool substitution")
    return pool


def _export_body(state_dir: str | Path, round_id: str, store: Any, *, task_loader: Callable[[Any, RoundRecord, Any], dict[str, Any]] | None = None) -> dict[str, Any]:
    journal_raw, records, rounds, confirmations, params_at_open = _journal(state_dir)
    if round_id not in rounds:
        raise ConfigError(f"round {round_id} is not in the journal")
    record = rounds[round_id]
    if round_id in confirmations and record.confirmation is None:
        record = dataclasses.replace(record, confirmation=confirmations[round_id])
    pending = _pending(record)
    pairs = []
    for lease in record.leases:
        ticket = _read_ticket(store, record, lease)
        result = _read_result(store, record, lease, ticket)
        if result is not None and ticket.parent == record.parent:
            pairs.append((lease, ticket, result))
    # Exactly the controller's preference: a completed parent result wins over parent errors.
    usable = [pair for pair in pairs if pair[2].parent_outcomes]
    failed = [pair for pair in pairs if pair[2].parent_error]
    if not usable and not failed:
        raise ConfigError("no matching sealed result has parent outcomes or a parent infrastructure error")
    lease, ticket, result = (usable or failed)[0]
    if len(result.null_outcomes) != ticket.params.null_samples:
        raise VerifyError("the sealed result does not contain every new-head null sample")
    if any(len(text) != len(ticket.draw.task_ids) for text in result.null_outcomes):
        raise VerifyError("a sealed null outcome string does not cover the exact draw")
    if ticket.params.params_hash != record.params_hash or ticket.draw.round_seed != record.round_seed:
        raise VerifyError("the selected ticket does not match the confirmation round record")
    if ticket.parent.manifest != record.parent.manifest:
        raise VerifyError("the selected ticket's parent manifest differs from the journal")
    if ticket.draw.snapshot_id != record.snapshot_id:
        raise VerifyError("the selected draw does not match the confirmation round record")
    crown_raw = params_at_open.get(pending.crown_round_id)
    if crown_raw is None:
        raise VerifyError("the crown round's pinned params are not in the journal")
    crown_params = Params.from_dict(crown_raw)
    legacy = legacy_parent_one_sample(crown_params, 1)
    ticket_raw = store.get(layout.ticket(record.round_id, lease.lease_id))
    result_raw = store.get(layout.sealed_result(record.round_id, lease.lease_id))
    manifest_raw = store.get(layout.head_manifest(record.parent.manifest.head.version_id))
    from .types import HeadManifest
    if HeadManifest.from_json(manifest_raw) != record.parent.manifest:
        raise VerifyError("the stored parent manifest differs from the journal")
    body = {
        "round_id": round_id,
        "pending_crown": pending.to_dict(),
        "params": ticket.params.to_dict(),
        "params_hash": ticket.params.params_hash,
        "genesis": next(item.data["genesis"] for item in records if item.kind == GENESIS),
        "draw": ticket.draw.to_dict(),
        "tasks": (task_loader or _task_evidence)(store, record, ticket.draw),
        "parent_manifest": record.parent.manifest.to_dict(),
        "null_samples": list(result.null_outcomes),
        "null_sample_representation": "one outcome string per new-head sample; one outcome code per draw task",
        "confirmation_resolved": record.confirmation.to_dict(),
        "exclusions": {
            "excluded_training_prompts": record.excluded_training_prompts,
            "winner_batch_sha256": pending.batch_sha256,
            "winner_batch": _training_evidence(store, pending),
            "draw_is_post_exclusion": True,
            "training_prompt_scope": "every prompt group in the pending crown winner's own file; admission caps groups and rollouts separately, and rollout text is not a prompt",
            "mapping": "map each winner-file prompt through task_for_prompt(task.batch_prompt()); skip unmapped prompts and exclude mapped task ids before the shared draw",
            "scoring": "ticket.draw.weights are already post-exclusion; score_confirmation receives those weights, all sealed new-head null strings and fresh parent strings; never redraw or exclude again",
        },
        "crown_design": {
            "params": crown_params.to_dict(),
            "params_hash": crown_params.params_hash,
            "code_digest": crown_params.recipe.code_digest,
            "parent_samples": 1 if legacy else None,
            "note": LEGACY_NOTE if legacy else "",
        },
        "sealed_evidence": {
            "lease_id": lease.lease_id,
            "box_id": lease.box_id,
            "ticket_object": layout.ticket(record.round_id, lease.lease_id),
            "result_object": layout.sealed_result(record.round_id, lease.lease_id),
            "parent_manifest_object": layout.head_manifest(record.parent.manifest.head.version_id),
            "ticket_digest": lease.ticket_digest,
            "result_digest": lease.result_digest,
            "ticket_json_b64": base64.b64encode(ticket_raw).decode("ascii"),
            "result_json_b64": base64.b64encode(result_raw).decode("ascii"),
            "parent_error": result.parent_error,
            "selection": "first lease with parent outcomes, else first lease with parent_error, in controller lease order",
        },
        "journal": {
            "file_sha256": sha256_hex(journal_raw),
            "record_count": len(records),
            "round_opened_seq": next(item.seq for item in records if item.kind == ROUND_OPENED and item.data["round"]["round_id"] == round_id),
            "confirmation_resolved_seq": next(item.seq for item in records if item.kind == CONFIRMATION_RESOLVED and item.data["round_id"] == round_id),
        },
    }
    return body


def export_bundle(state_dir: str | Path, round_id: str, out: str | Path, *, store: Any = None, task_loader: Callable[[Any, RoundRecord, Any], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Write a deterministic, self-contained bundle without writing to ``state_dir``."""
    body = _export_body(state_dir, round_id, _evidence_store(state_dir) if store is None else store, task_loader=task_loader)
    envelope = {"schema": BUNDLE_SCHEMA, "body": body, "sha256": sha256_hex(canonical_json(body))}
    _write_json(out, envelope)
    return envelope


def issue_links(state_dir: str | Path, round_id: str, out: str | Path, ttl: int = DEFAULT_TTL, *, store: Any = None, now: Callable[[], int] | None = None) -> dict[str, Any]:
    """Issue signed GETs for only this round's parent bf16 base, with no URL on stdout/stderr."""
    if type(ttl) is not int or not 0 < ttl <= 7 * 24 * 3600:
        raise ConfigError("ttl must be a whole number of seconds from 1 through 604800")
    _, _, rounds, confirmations, _ = _journal(state_dir)
    if round_id not in rounds:
        raise ConfigError(f"round {round_id} is not in the journal")
    record = rounds[round_id]
    if record.confirmation is None and round_id in confirmations:
        record = dataclasses.replace(record, confirmation=confirmations[round_id])
    _pending(record)
    manifest = record.parent.manifest
    head = manifest.head
    store = _evidence_store(state_dir, signing=True) if store is None else store
    missing = []
    for name in sorted(manifest.base_files):
        try:
            store.size(layout.public_base_file(head.base_hash, name))
        except KeyError:
            missing.append(name)
    if missing:
        raise ConfigError("parent base objects are missing: " + ", ".join(missing))
    # The same bounded file-set builder used by the site's read_links producer.
    base = file_set(store, head.base_hash, manifest.base_files, layout.public_base_file, ttl)
    at = int(time.time() if now is None else now())
    document = {
        "schema": LINKS_SCHEMA,
        "generated_at": at,
        "expires_at": at + ttl,
        "ttl_s": ttl,
        "round_id": round_id,
        "head_version_id": head.version_id,
        "how": "GET each url before expires_at, then check the size and sha256 against the parent HeadManifest. This file authorizes only this round's parent bf16 base.",
        "base": base,
        "master": None,
        "loras": [],
    }
    _write_json(out, document, private=True)
    return document


def load_bundle(path: str | Path) -> tuple[dict[str, Any], RoundTicket, tuple[str, ...]]:
    """Verify the bundle seal and every duplicated ticket identity before using it."""
    envelope = _json(path)
    if envelope.get("schema") != BUNDLE_SCHEMA or set(envelope) != {"schema", "body", "sha256"}:
        raise VerifyError("not an offline re-check bundle")
    body = envelope["body"]
    if not isinstance(body, dict) or envelope["sha256"] != sha256_hex(canonical_json(body)):
        raise VerifyError("offline re-check bundle sha256 does not match its body")
    try:
        evidence = body["sealed_evidence"]
        ticket_key = layout.ticket(body["round_id"], evidence["lease_id"])
        if evidence["ticket_object"] != ticket_key or evidence["result_object"] != layout.sealed_result(body["round_id"], evidence["lease_id"]):
            raise VerifyError("bundle sealed-evidence object keys are wrong")
        raw = base64.b64decode(evidence["ticket_json_b64"], validate=True)
        ticket = RoundTicket.from_json(raw)
        if ticket.digest() != evidence["ticket_digest"] or ticket.params.to_dict() != body["params"]:
            raise VerifyError("bundle ticket and params disagree")
        if ticket.params.params_hash != body["params_hash"] or ticket.draw.to_dict() != body["draw"]:
            raise VerifyError("bundle ticket, params hash and draw disagree")
        _bundle_pool(body, ticket.draw.task_ids)
        if ticket.parent is None or ticket.parent.manifest is None or ticket.parent.manifest.to_dict() != body["parent_manifest"]:
            raise VerifyError("bundle ticket and parent manifest disagree")
        if evidence["parent_manifest_object"] != layout.head_manifest(ticket.parent.manifest.head.version_id):
            raise VerifyError("bundle parent-manifest object key is wrong")
        if evidence["box_id"] != ticket.box_id:
            raise VerifyError("bundle sealed-evidence box is wrong")
        if ticket.parent.pending.to_dict() != body["pending_crown"] or ticket.round_id != body["round_id"]:
            raise VerifyError("bundle ticket and pending crown disagree")
        result = BoxResult.from_json(base64.b64decode(evidence["result_json_b64"], validate=True))
        if (result.digest() != evidence["result_digest"] or result.ticket_digest != ticket.digest()
                or list(result.null_outcomes) != body["null_samples"]
                or (result.round_id, result.lease_id, result.box_id) != (ticket.round_id, ticket.lease_id, ticket.box_id)):
            raise VerifyError("bundle result bytes do not match its journal seals")
        nulls = tuple(result.null_outcomes)
        if len(nulls) != ticket.params.null_samples or any(len(text) != len(ticket.draw.task_ids) for text in nulls):
            raise VerifyError("bundle null outcomes do not match the live sample contract")
        if ticket.params.null_samples < 2:
            raise VerifyError("the re-check requires the live two-sample null contract")
        if result.parent_outcomes or not result.parent_error or evidence["parent_error"] != result.parent_error:
            raise VerifyError("the bundle is not an infrastructure-failed parent confirmation")
        from .public import public_text
        if public_text(result.parent_error) not in body["confirmation_resolved"]["reason"]:
            raise VerifyError("the resolved confirmation does not identify the sealed parent error")
        confirmation = Confirmation.from_dict(body["confirmation_resolved"])
        pending = ticket.parent.pending
        if (confirmation.pending_round_id, confirmation.pending_uid, confirmation.parent_version_id, confirmation.new_version_id) != (
                pending.crown_round_id, pending.uid, pending.parent_version_id, pending.new_version_id):
            raise VerifyError("bundle confirmation does not answer this pending crown")
        if confirmation.confirmed or not confirmation.reason.startswith("no_decision"):
            raise VerifyError("bundle does not hold an unresolved confirmation")
        GenesisSpec.from_dict(body["genesis"])
        crown = Params.from_dict(body["crown_design"]["params"])
        if crown.params_hash != body["crown_design"]["params_hash"]:
            raise VerifyError("the bundle's crown params hash is wrong")
        legacy = legacy_parent_one_sample(crown, 1)
        if (body["crown_design"].get("note") or "") != (LEGACY_NOTE if legacy else ""):
            raise VerifyError("the bundle's crown sampling note does not match its pinned code identity")
        exclusions = body["exclusions"]
        if (exclusions["winner_batch_sha256"] != pending.batch_sha256
                or exclusions["excluded_training_prompts"] != confirmation.excluded_training_prompts):
            raise VerifyError("bundle exclusion provenance does not match its pending crown and confirmation")
        training = exclusions["winner_batch"]
        if training.get("available"):
            training_raw = base64.b64decode(training["file_json_b64"], validate=True)
            if sha256_hex(training_raw) != pending.batch_sha256:
                raise VerifyError("the bundle's winner batch does not match its pending crown")
            training_body = json.loads(training_raw)
            from .admission import FORMATS
            if training_body.get("version") not in FORMATS or training_body != training["file"]:
                raise VerifyError("the bundle's winner batch is not the original groups file")
            groups = [
                {"prompt": group["prompt"], "rollouts": group.get("rollouts") if isinstance(group.get("rollouts"), list) else []}
                for group in training_body.get("groups", [])
                if isinstance(group, dict) and isinstance(group.get("prompt"), str)
            ]
            if groups != training["groups"]:
                raise VerifyError("the bundle's training prompts and rollouts differ from its original winner batch")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, V2Error):
            raise
        raise VerifyError("offline re-check bundle is incomplete") from None
    return body, ticket, nulls


class _PublicSource:
    """Anonymous public-object GETs; the private scoring store is never used."""

    def __init__(self, base_hash: str, public_root: str) -> None:
        self.base_hash, self.public_root, self.fetched = base_hash, public_root.rstrip("/"), 0

    def get(self, kind: str, set_id: str, name: str, sha256: str, path: Path) -> None:
        if kind != "base" or set_id != self.base_hash:
            raise VerifyError("the public source was asked for a different file set")
        from .headfold import AGENT, download
        url = f"{self.public_root}/{layout.public_base_file(set_id, name)}"
        try:
            request = urllib.request.Request(url, headers=AGENT, method="HEAD")
            with urllib.request.urlopen(request, timeout=60) as reply:  # noqa: S310 - fixed public root and manifest key
                size = int(reply.headers["Content-Length"])
        except (OSError, KeyError, ValueError):
            raise InfraError(f"public parent base file {name} is unavailable") from None
        download(url, path, sha256, size)
        self.fetched += size


def _fetch_parent(ticket: RoundTicket, work: Path, links_path: str | Path | None) -> Path:
    from .headfold import BASE, LinkSource, fetch_set  # torch is lazy, for scoring only
    manifest = ticket.parent.manifest
    head = manifest.head
    if links_path is None:
        source = _PublicSource(head.base_hash, os.environ.get("SN120_PUBLIC_URL", PUBLIC_ROOT))
        cache = fetch_set(source, BASE, head.base_hash, manifest.base_files, work)
    else:
        links = _json(links_path)
        base = links.get("base") or {}
        if links.get("schema") != LINKS_SCHEMA or links.get("head_version_id") != head.version_id or base.get("id") != head.base_hash:
            raise VerifyError("the links file is not for this bundle's parent base")
        if links.get("round_id", ticket.round_id) != ticket.round_id:
            raise VerifyError("the links file is for another confirmation round")
        cache = work / "cache" / head.base_hash
        # A complete valid cache never needs the signed query string. It is
        # nevertheless verified against the sealed manifest on every run.
        from .canonical import file_sha256
        if not (cache.is_dir() and all((cache / name).is_file() and file_sha256(cache / name) == sha
                                      for name, sha in manifest.base_files.items())):
            cache = fetch_set(LinkSource(links), BASE, head.base_hash, manifest.base_files, work)
    sizes = {row["name"]: int(row["size"]) for row in (base.get("files") or []) if isinstance(row, dict)} if links_path else {}
    if links_path and (set(sizes) != set(manifest.base_files) or any(sizes[name] != (cache / name).stat().st_size for name in sizes)):
        raise VerifyError("a signed parent file size differs from the downloaded base")
    if any(not (cache / name).is_file() or file_sha256(cache / name) != sha for name, sha in manifest.base_files.items()):
        raise VerifyError("the parent base cache does not match the sealed manifest")
    return cache


class _ParentResident:
    """Serve an old parent's separately prepared input without changing its sealed cache.

    Live ``score_parent`` passes ``resident_parent_base`` straight to
    ``_start_one``, whose ordinary ``fold_input`` fallback is then bypassed.
    This offline-only adapter applies that same preparation at the resident
    boundary. Every other file-set lookup and the original Resident stay intact.
    """

    def __init__(self, resident: Any, base_hash: str, prepared: Path) -> None:
        self.resident, self.base_hash, self.prepared = resident, base_hash, prepared

    def tree(self, prefix: str, identity: str) -> Path:
        if identity == self.base_hash:
            return self.prepared
        return self.resident.tree(prefix, identity)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.resident, name)


def _default_evaluator(body: Mapping[str, Any], ticket: RoundTicket, work: Path) -> Any:
    from .admission import LazyPool
    from .envs.math_grader import GradePool
    from .gpu.evaluator import VllmEvaluator
    from .gpu.fold_input import FoldInputs
    from .gpu.plan import BoxConfig, install_digest
    store = LocalStore(work / "store")  # local only: no store credentials on the scoring host
    pool = _bundle_pool(body, ticket.draw.task_ids)
    fold_input = FoldInputs(work, GenesisSpec.from_dict(body["genesis"]))
    # FoldInputs adds missing metadata only in a separate directory. Supply it
    # at score_parent's actual resident boundary, not through _start_one's
    # bypassed fallback. score_parent itself remains the live implementation.
    base = work / "cache" / ticket.parent.manifest.head.base_hash
    prepared = fold_input(base, ticket.parent.manifest.head)
    evaluator = VllmEvaluator(store, BoxConfig(work, install_digest()), pool, LazyPool(GradePool), fold_input=fold_input)
    evaluator.resident = _ParentResident(evaluator.resident, ticket.parent.manifest.head.base_hash, prepared)
    # SoloRunner does this before live confirmation. score_parent's own
    # clients_for fallback does not raise NOFILE or share the socket budget.
    evaluator._solo_clients = evaluator.solo_client_budget(ticket.params.n_gpus, evaluator.clients_for(ticket.params))
    return evaluator


def _git_head() -> str:
    try:
        reply = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, check=True, capture_output=True, text=True, timeout=10)
        return reply.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _parent_stats(work: Path, round_id: str, previous: int) -> dict[str, Any]:
    path = work / "logs" / "phases.jsonl"
    if not path.is_file():
        return {}
    found = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines()[previous:]:
            row = json.loads(line)
            if row.get("phase") == "parent_eval" and row.get("round_id") == round_id:
                found.append(row)
    except (OSError, ValueError, AttributeError):
        return {}
    return found[-1] if found else {}


def score_bundle(
    bundle_path: str | Path,
    work_dir: str | Path,
    out: str | Path,
    *,
    n_tasks: int | None = None,
    gpus: int | None = None,
    links_path: str | Path | None = None,
    evaluator_factory: Callable[[Mapping[str, Any], RoundTicket, Path], Any] | None = None,
    downloader: Callable[[RoundTicket, Path, str | Path | None], Any] | None = None,
) -> dict[str, Any]:
    """Re-score the parent and apply the live paired confirmation rule unchanged."""
    started = time.monotonic()
    body, original, sealed_nulls = load_bundle(bundle_path)
    count = len(original.draw.task_ids)
    if n_tasks is not None and (type(n_tasks) is not int or not 1 <= n_tasks <= count):
        raise ConfigError(f"n-tasks must be from 1 through the draw size {count}")
    if gpus is not None and (type(gpus) is not int or gpus < 1):
        raise ConfigError("gpus must be a positive integer")
    sanity = n_tasks is not None
    used = n_tasks if sanity else count
    draw = dataclasses.replace(original.draw, task_ids=original.draw.task_ids[:used], weights=original.draw.weights[:used])
    params = dataclasses.replace(original.params, n_gpus=gpus or original.params.n_gpus)
    ticket = dataclasses.replace(original, draw=draw, params=params)
    nulls = tuple(text[:used] for text in sealed_nulls)
    from .gpu.plan import code_digest
    digest = code_digest()
    pinned_digest = original.params.recipe.code_digest
    code_identity_matches = not pinned_digest or pinned_digest == digest
    # Historical crowns were sealed before live two-sample parent scoring.
    # The bundle records that pin; this tool deliberately uses the current
    # score_parent rather than reproducing an obsolete one-sample design.
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    phase_log = work / "logs" / "phases.jsonl"
    previous = len(phase_log.read_text(encoding="utf-8").splitlines()) if phase_log.is_file() else 0
    (downloader or _fetch_parent)(ticket, work, links_path)
    evaluator = (evaluator_factory or _default_evaluator)(body, ticket, work)
    try:
        evaluator.begin_round(ticket)
        parent_outcomes = tuple(evaluator.score_parent(ticket, deadline=0))
    finally:
        evaluator.close()
    if len(parent_outcomes) != len(nulls) or any(len(text) != used for text in parent_outcomes):
        raise VerifyError("parent evaluator did not return the live number of full-draw sample strings")
    fields = score_confirmation(draw.weights, nulls, parent_outcomes)
    notes = [body["crown_design"].get("note") or "", SANITY_NOTE if sanity else ""]
    verdict = {
        "schema": "sn120-v2-offline-recheck-verdict-1",
        "round_id": original.round_id,
        "uid": original.parent.pending.uid,
        "hotkey": original.parent.pending.hotkey,
        "crown_round_id": original.parent.pending.crown_round_id,
        "new_version_id": original.parent.pending.new_version_id,
        "parent_version_id": original.parent.pending.parent_version_id,
        **fields,
        "null_samples": len(nulls),
        "parent_samples": len(parent_outcomes),
        "null_outcomes": list(nulls),
        "parent_outcomes": list(parent_outcomes),
        "outcome_representation": {
            "null_samples": "count of new-head outcome strings, not a task count",
            "parent_samples": "count of fresh parent outcome strings; parent sample k uses the seed of new-head null sample k",
            "outcome_strings": "one outcome code per task, in the sealed draw order",
            "tasks_per_string": used,
        },
        "bundle_sha256": _json(bundle_path)["sha256"],
        "code_digest": digest,
        "confirmation_code_digest": pinned_digest,
        "code_identity_matches": code_identity_matches,
        "git_head": _git_head(),
        "vllm_version": _version("vllm"),
        "torch_version": _version("torch"),
        "wall_s": time.monotonic() - started,
        "parent_eval": _parent_stats(work, original.round_id, previous),
        "parent_seed_rule": "rollout_seed(round_seed, task_id, parent_sample_k) == rollout_seed(round_seed, task_id, null_sample_k)",
        "sanity": sanity,
        "note": "; ".join(note for note in notes if note),
        "scoring_contract": "live VllmEvaluator.score_parent with all sealed new-head null strings; no z-score payout threshold; the pinned historical crown code identity is not the scorer version",
        "local_overrides": {"n_gpus": params.n_gpus, "round_n_gpus": original.params.n_gpus, "n_tasks": used, "round_n_tasks": count},
        "parent_base": {"base_hash": original.parent.manifest.head.base_hash, "source": "signed-links" if links_path else "anonymous-public-store", "files_verified": len(original.parent.manifest.base_files)},
        "excluded_training_prompts": body["exclusions"]["excluded_training_prompts"],
        "existing_confirmation": body["confirmation_resolved"],
    }
    _write_json(out, verdict)
    return verdict


def _safe_error(exc: BaseException, *, secret: bool) -> str:
    # Only manifest names are safe to repeat. An arbitrary store exception may contain a signed URL or a credential.
    missing = "parent base objects are missing: "
    if secret and isinstance(exc, ConfigError) and str(exc).startswith(missing):
        return "recheck: ConfigError: " + str(exc)
    if secret:
        return f"recheck: {type(exc).__name__}: link issuance failed; no links file was written"
    return f"recheck: {_URL.sub('<redacted-url>', str(exc))}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sn120_v2.recheck", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "links"):
        command = commands.add_parser(name)
        command.add_argument("--state-dir", required=True)
        command.add_argument("--round", required=True, dest="round_id")
        command.add_argument("--out", required=True)
        if name == "links":
            command.add_argument("--ttl", type=int, default=DEFAULT_TTL)
    command = commands.add_parser("score")
    command.add_argument("--bundle", required=True)
    command.add_argument("--work-dir", required=True)
    command.add_argument("--n-tasks", type=int)
    command.add_argument("--gpus", type=int)
    command.add_argument("--links")
    command.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            result = export_bundle(args.state_dir, args.round_id, args.out)
            print(json.dumps({"round_id": args.round_id, "bundle_sha256": result["sha256"], "out": args.out}))
        elif args.command == "links":
            result = issue_links(args.state_dir, args.round_id, args.out, args.ttl)
            # Do not serialize the result: it contains signed URLs.
            print(json.dumps({"round_id": args.round_id, "files": len(result["base"]["files"]), "expires_at": result["expires_at"], "out": args.out}))
        else:
            result = score_bundle(args.bundle, args.work_dir, args.out, n_tasks=args.n_tasks, gpus=args.gpus, links_path=args.links)
            print(json.dumps({key: result[key] for key in ("round_id", "uid", "delta", "confirmed", "n_tasks", "sanity")} | {"out": args.out}))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary; library callers retain the original failure
        print(_safe_error(exc, secret=args.command == "links"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
