#!/usr/bin/env python3
"""Plan and fetch explicit SN120 dedup-dryrun inputs, using Store GET only.

This file is deliberately independent of deployment/rollback tooling. Run the
fetch in an operator-selected production worktree, with SN120_R2_* credentials
in the environment. Do not put credentials in the plan, command line or shell
history. Private keys and paths are written only to the selected local output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import sys
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "sn120-dedup-frozen-inputs-v1"
DEFAULT_ROUNDS = ("r000085", "r000086", "r000087", "r000088")
SUPPLIED_JOURNAL_SHA256 = "1a7475525b8c176b445927fe864b1df6e7abf31b8083e011e9a3e4c2c3acf1ca"
LORA_FILES = ("adapter_model.safetensors", "adapter_config.json", "train_report.json")
FIRST_SEEN_KEY = "private/dedup/first_seen.json"
DRYRUN_LAUNCHER = str(Path(__file__).resolve().with_name("frozen_input_dryrun.py"))
V2_ROOT = Path(__file__).resolve().parents[2]
SHA256_HEX = frozenset("0123456789abcdef")


class InputError(RuntimeError):
    """A journal, plan, object or operator input failed its contract."""


def canonical_json(value: Any) -> bytes:
    def check(item: Any) -> None:
        if item is None or isinstance(item, (str, bool, int, float)):
            if isinstance(item, float) and not (-float("inf") < item < float("inf")):
                raise InputError("non-finite JSON number")
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                check(child)
            return
        if isinstance(item, dict):
            if not all(isinstance(k, str) for k in item):
                raise InputError("non-string JSON object key")
            for child in item.values():
                check(child)
            return
        raise InputError(f"unsupported JSON value {type(item).__name__}")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and SHA256_HEX.issuperset(value)


def manifest_digest(files: Mapping[str, str]) -> str:
    if not files or not all(isinstance(k, str) and k and k.isascii() and is_sha256(v)
                            for k, v in files.items()):
        raise InputError("invalid LoRA file manifest")
    return sha256(canonical_json(dict(files)))


def safe_key(value: Any) -> str:
    if not isinstance(value, str) or not value or value.startswith("/") or value.endswith("/"):
        raise InputError("unsafe object key")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts) or any("\x00" in part or "\\" in part for part in parts):
        raise InputError("unsafe object key")
    return value


def safe_relative(value: Any) -> str:
    path = PurePosixPath(value) if isinstance(value, str) else PurePosixPath("")
    if (not isinstance(value, str) or path.is_absolute() or not path.parts
            or any(part in ("", ".", "..") for part in path.parts) or "\\" in value):
        raise InputError("unsafe local path")
    return path.as_posix()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeError, ValueError) as exc:
        raise InputError(f"cannot read JSON input {path}: {type(exc).__name__}") from exc


def read_journal(path: Path) -> list[dict[str, Any]]:
    """Verify the complete chain and canonical bytes without importing shared code."""
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError as exc:
        raise InputError(f"cannot read journal: {type(exc).__name__}") from exc
    records = []
    previous = ""
    for seq, raw in enumerate(lines):
        if not raw.endswith(b"\n") or raw.endswith(b"\r\n"):
            raise InputError("journal has a torn or non-canonical last line")
        line = raw[:-1]
        try:
            fields = json.loads(line)
            if not isinstance(fields, dict):
                raise InputError("journal row is not an object")
            claimed = fields.pop("sum")
            record = {"seq": fields.pop("seq"), "fence": fields.pop("fence"), "at": fields.pop("at"),
                      "kind": fields.pop("kind"), "data": fields.pop("data"), "prev": fields.pop("prev")}
            if fields:
                raise InputError("journal row has unknown envelope fields")
            expected = sha256(canonical_json(record))
            if (record["seq"] != seq or record["prev"] != previous or claimed != expected
                    or record["kind"] not in ("leader",) and seq > 0 and record["fence"] < records[-1]["fence"]):
                raise InputError("journal chain verification failed")
            if canonical_json(record | {"sum": claimed}) != line:
                raise InputError("journal row is not canonical JSON")
            records.append(record | {"sum": claimed})
            previous = expected
        except (KeyError, TypeError, ValueError, UnicodeError) as exc:
            raise InputError(f"journal row {seq} is invalid: {type(exc).__name__}") from exc
    if not records:
        raise InputError("journal is empty")
    return records


def round_open(records: Iterable[Mapping[str, Any]], round_id: str) -> Mapping[str, Any]:
    found = [row for row in records if row["kind"] == "round_opened"
             and isinstance(row["data"].get("round"), dict)
             and row["data"]["round"].get("round_id") == round_id]
    if len(found) != 1:
        raise InputError(f"{round_id}: expected one round_opened record, found {len(found)}")
    return found[0]["data"]["round"]


def result_reference(records: Iterable[Mapping[str, Any]], round_id: str, lease_id: str) -> str | None:
    found = []
    for row in records:
        if row["kind"] != "round_settled" or row["data"].get("round_id") != round_id:
            continue
        results = row["data"].get("results")
        if not isinstance(results, dict):
            raise InputError(f"{round_id}: malformed settlement result map")
        if lease_id in results:
            found.append((results[lease_id], row["seq"]))
    if len(found) > 1:
        raise InputError(f"{round_id}: duplicate result references")
    if not found:
        return None
    if not is_sha256(found[0][0]):
        raise InputError(f"{round_id}: malformed result digest")
    opened_seq = next(row["seq"] for row in records if row["kind"] == "round_opened"
                      and row["data"].get("round", {}).get("round_id") == round_id)
    if found[0][1] < opened_seq:
        raise InputError(f"{round_id}: result reference predates its open record")
    return found[0][0]


def result_digest(raw: bytes) -> str:
    """The shared BoxResult digest is its canonical JSON, not its byte SHA."""
    try:
        return sha256(canonical_json(json.loads(raw)))
    except (ValueError, UnicodeError, TypeError) as exc:
        raise InputError(f"sealed result is not canonical JSON: {type(exc).__name__}") from exc


def validate_result(raw: bytes, rnd: Mapping[str, Any], lease: Mapping[str, Any],
                    expected_digest: str | None) -> dict[str, Any]:
    try:
        obj = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise InputError(f"{rnd['round_id']}: sealed result is not JSON") from exc
    if not isinstance(obj, dict):
        raise InputError(f"{rnd['round_id']}: sealed result is not an object")
    if expected_digest is not None and result_digest(raw) != expected_digest:
        raise InputError(f"{rnd['round_id']}: sealed result digest differs from the journal")
    if obj.get("round_id") != rnd["round_id"] or obj.get("lease_id") != lease["lease_id"]:
        raise InputError(f"{rnd['round_id']}: sealed result is for another round or lease")
    if obj.get("ticket_digest") != lease["ticket_digest"]:
        raise InputError(f"{rnd['round_id']}: sealed result does not answer the opened ticket")
    seats = {entry["queue_pos"]: entry for entry in lease["entries"]}
    rows = obj.get("entries")
    if not isinstance(rows, list) or len(rows) != len(seats):
        raise InputError(f"{rnd['round_id']}: sealed result seat count mismatch")
    by_position = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("queue_pos")) is not int or row["queue_pos"] in by_position:
            raise InputError(f"{rnd['round_id']}: invalid sealed result queue positions")
        position = row["queue_pos"]
        if position not in seats:
            raise InputError(f"{rnd['round_id']}: sealed result names an unknown seat")
        if row.get("status") == "scored":
            artifact = row.get("lora")
            if not isinstance(artifact, dict):
                raise InputError(f"{rnd['round_id']}: scored result has no LoRA")
            if artifact.get("batch_sha256") != seats[position]["batch"]["batch_sha256"]:
                raise InputError(f"{rnd['round_id']}: LoRA belongs to another frozen batch")
            if not is_sha256(artifact.get("lora_hash")):
                raise InputError(f"{rnd['round_id']}: invalid LoRA hash")
            files, sizes = artifact.get("files"), artifact.get("sizes")
            if (not isinstance(files, dict) or not isinstance(sizes, dict)
                    or set(files) != set(LORA_FILES) or set(sizes) != set(LORA_FILES)):
                raise InputError(f"{rnd['round_id']}: LoRA needs exactly the three dry-run files")
            if manifest_digest(files) != artifact["lora_hash"]:
                raise InputError(f"{rnd['round_id']}: LoRA manifest hash mismatch")
            if any(type(sizes[name]) is not int or sizes[name] < 0 for name in files):
                raise InputError(f"{rnd['round_id']}: invalid LoRA file size")
            safe_key(artifact.get("prefix"))
        elif row.get("lora") is not None:
            raise InputError(f"{rnd['round_id']}: unscored result unexpectedly carries a LoRA")
        by_position[position] = row
    return obj


def seat_key(rnd: Mapping[str, Any], seat: Mapping[str, Any]) -> str:
    batch = seat.get("batch", {})
    expected = f"private/rounds/{rnd['round_id']}/batches/{batch.get('hotkey')}.json"
    if safe_key(batch.get("uri")) != expected or not is_sha256(batch.get("batch_sha256")):
        raise InputError(f"{rnd['round_id']}: unsafe or malformed frozen batch reference")
    if batch.get("hotkey") != seat.get("reg", {}).get("hotkey"):
        raise InputError(f"{rnd['round_id']}: seat registration/batch identity mismatch")
    return expected


def planned_seats(records: Sequence[Mapping[str, Any]], round_id: str,
                  results: Mapping[str, bytes] | None = None) -> list[dict[str, Any]]:
    rnd = round_open(records, round_id)
    rows = []
    for lease in rnd.get("leases", []):
        if not isinstance(lease.get("entries"), list) or not isinstance(lease.get("lease_id"), str):
            raise InputError(f"{round_id}: malformed lease")
        result_key = f"private/rounds/{round_id}/{safe_key(lease['lease_id'])}/result.json"
        reference = result_reference(records, round_id, lease["lease_id"])
        raw = (results or {}).get(result_key)
        result = validate_result(raw, rnd, lease, reference) if raw is not None else None
        entries = {entry["queue_pos"]: entry for entry in result["entries"]} if result else {}
        for seat in sorted(lease["entries"], key=lambda entry: entry.get("queue_pos", -1)):
            position = seat.get("queue_pos")
            if type(position) is not int or position < 0:
                raise InputError(f"{round_id}: invalid seat queue position")
            key = seat_key(rnd, seat)
            row = {"round_id": round_id, "lease_id": lease["lease_id"], "queue_pos": position,
                   "batch_sha256": seat["batch"]["batch_sha256"], "batch_key": key,
                   "batch_path": f"{round_id}/q{position:06d}/batch.json",
                   "result_key": result_key, "result_path": f"results/{round_id}/{lease['lease_id']}.json",
                   "result_expected": reference is not None, "result_digest": reference,
                   "lora": None, "lora_status": "not_referenced"}
            if result:
                entry = entries[position]
                artifact = entry.get("lora")
                if artifact is not None:
                    prefix = safe_key(artifact["prefix"])
                    # Never infer that pending files were retained. Complete
                    # pending evidence is a candidate, but not a promise: fetch
                    # must fail rather than silently produce a partial LoRA set.
                    row["lora_status"] = "complete_candidate" if prefix.startswith("pending/rounds/") else "retained_public_candidate"
                    files = {name: f"{prefix}/{name}" for name in LORA_FILES}
                    row["lora"] = {"prefix": prefix, "files": files, "hashes": artifact["files"],
                                   "sizes": artifact["sizes"], "path": f"{round_id}/q{position:06d}/lora",
                                   "lora_hash": artifact["lora_hash"]}
            rows.append(row)
    positions = [row["queue_pos"] for row in rows]
    if not rows or positions != sorted(set(positions)):
        raise InputError(f"{round_id}: seats are not globally ordered and unique")
    if not all(row["result_expected"] for row in rows) and any(row["result_expected"] for row in rows):
        raise InputError(f"{round_id}: inconsistent settlement references")
    return rows


def first_seen_values(path: Path | None, seats: Sequence[Mapping[str, Any]]) -> tuple[list[int] | None, list[str]]:
    if path is None:
        return None, ["not requested"]
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise InputError("first-seen ledger must be a JSON object")
    values, missing = [], []
    for seat in seats:
        row = raw.get(seat["batch_sha256"])
        if not isinstance(row, dict) or type(row.get("first_seen_unix")) is not int:
            missing.append(f"{seat['round_id']}/q{seat['queue_pos']:06d}")
        else:
            values.append(row["first_seen_unix"])
    if missing:
        return None, missing
    return values, []


def dryrun_command(seats: Sequence[Mapping[str, Any]], input_root: str, python: str,
                   params: str | None = None, first_seen: Sequence[int] | None = None,
                   first_seen_missing: Sequence[str] = (), v2_root: str = str(V2_ROOT)) -> dict[str, Any]:
    argv = [python, "-I", DRYRUN_LAUNCHER, "--v2-root", v2_root, "dedup-dryrun"]
    for seat in seats:
        argv += ["--batch", str(PurePosixPath(input_root) / seat["batch_path"])]
    complete_loras = [seat["lora"] for seat in seats]
    with_loras = all(lora is not None for lora in complete_loras)
    if with_loras:
        for lora in complete_loras:
            argv += ["--lora", str(PurePosixPath(input_root) / lora["path"])]
    if params is not None:
        argv += ["--params", params]
    if first_seen is not None:
        if len(first_seen) != len(seats):
            raise InputError("first-seen does not cover every batch")
        for value in first_seen:
            argv += ["--first-seen", str(value)]
    return {"argv": argv, "shell": shlex.join(argv), "with_loras": with_loras,
            "lora_unavailable_seats": [f"{s['round_id']}/q{s['queue_pos']:06d}" for s in seats if s["lora"] is None],
            "with_first_seen": first_seen is not None,
            "first_seen_missing_seats": list(first_seen_missing),
            "note": "LoRAs are all-or-none for the CLI; absent first-seen timestamps are never invented."}


def make_plan(journal: Path, root: str, rounds: Sequence[str], params: str | None,
              python: str, results: Mapping[str, bytes] | None = None,
              first_seen: Path | None = None, journal_sha256: str | None = None,
              fetch_first_seen: bool = False, v2_root: str = str(V2_ROOT),
              results_only: bool = False) -> dict[str, Any]:
    try:
        journal_bytes = journal.read_bytes()
    except OSError as exc:
        raise InputError(f"cannot read journal: {type(exc).__name__}") from exc
    actual = sha256(journal_bytes)
    if journal_sha256 is not None and actual != journal_sha256:
        raise InputError("journal sha256 does not match the requested frozen input")
    if len(rounds) != len(set(rounds)):
        raise InputError("round ids must not repeat")
    records = read_journal(journal)
    seats = [seat for round_id in rounds for seat in planned_seats(records, round_id, results)]
    timestamps, missing = first_seen_values(first_seen, seats)
    if timestamps is not None and len(timestamps) != len(seats):
        raise InputError("incomplete first-seen coverage")
    if fetch_first_seen and first_seen is not None:
        raise InputError("choose a local first-seen file or a first-seen GET, not both")
    commands = []
    for round_id in rounds:
        selected = [seat for seat in seats if seat["round_id"] == round_id]
        indices = [i for i, seat in enumerate(seats) if seat["round_id"] == round_id]
        seen = [timestamps[i] for i in indices] if timestamps is not None else None
        commands.append(dryrun_command(selected, root, python, params, seen,
                                       missing if seen is None and first_seen else (), v2_root))
    objects = [{"kind": "batch", "key": s["batch_key"], "path": s["batch_path"],
                "sha256": s["batch_sha256"], "size": None, "required": True} for s in seats]
    seen_results = set()
    for seat in seats:
        if seat["result_expected"] and seat["result_key"] not in seen_results:
            seen_results.add(seat["result_key"])
            objects.append({"kind": "result", "key": seat["result_key"], "path": seat["result_path"],
                            "sha256": None, "size": None, "required": False})
    for seat in seats:
        if seat["lora"] is not None:
            lora = seat["lora"]
            for name in LORA_FILES:
                objects.append({"kind": "lora", "key": lora["files"][name],
                                "path": f"{lora['path']}/{name}", "sha256": lora["hashes"][name],
                                "size": lora["sizes"][name], "required": True})
    if fetch_first_seen:
        objects.append({"kind": "first_seen", "key": FIRST_SEEN_KEY, "path": "first_seen.json",
                        "sha256": None, "size": None, "required": False})
    if results_only:
        objects = [obj for obj in objects if obj["kind"] in ("result", "first_seen")]
    keys = [obj["key"] for obj in objects]
    paths = [obj["path"] for obj in objects]
    if len(keys) != len(set(keys)) or len(paths) != len(set(paths)):
        raise InputError("plan has duplicate keys or paths")
    for obj in objects:
        safe_key(obj["key"])
        safe_relative(obj["path"])
    return {"schema": SCHEMA, "journal_sha256": actual, "input_root": root, "rounds": list(rounds),
            "params": params, "python": python, "v2_root": v2_root, "seats": seats, "objects": objects,
            "derive_from_journal": results is None, "results_only": results_only,
            "fetch_first_seen": fetch_first_seen, "dryrun_commands": commands}


def public_summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Counts and local paths only: no hotkeys, private keys or batch content."""
    return {"schema": SCHEMA, "journal_sha256": plan["journal_sha256"], "rounds": plan["rounds"],
            "input_root": plan["input_root"], "batch_count": sum(o["kind"] == "batch" for o in plan["objects"]),
            "lora_candidate_count": sum(s["lora"] is not None for s in plan["seats"]),
            "commands": [{"round_id": r, "shell": c["shell"], "with_loras": c["with_loras"],
                          "with_first_seen": c["with_first_seen"], "lora_unavailable_seats": c["lora_unavailable_seats"],
                          "first_seen_missing_seats": c["first_seen_missing_seats"]}
                         for r, c in zip(plan["rounds"], plan["dryrun_commands"])]}


def write_private(path: Path, value: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), mode)
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def emit_commands(plan: Mapping[str, Any], output: Path | None) -> None:
    text = "\n\n".join(command["shell"] for command in plan["dryrun_commands"]) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        handle = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(text)


class GetOnlyStore:
    """Expose exactly store.get; every other store operation fails closed."""

    def __init__(self, delegate: Any):
        self._delegate = delegate
        self.calls: list[str] = []

    def get(self, key: str) -> bytes:
        self.calls.append("get")
        return self._delegate.get(key)

    def __getattr__(self, name: str) -> Any:
        raise InputError(f"GET-only guard refused store.{name}")


class GetOnlyS3:
    """Low-level boto3 guard: GetObject is the only possible S3 operation."""

    def __init__(self, client: Any):
        self._client = client
        self._make_api_call = self._guarded_call
        self.meta = client.meta

    def _guarded_call(self, operation_name: str, api_params: Mapping[str, Any]) -> Any:
        if operation_name != "GetObject" or set(api_params) != {"Bucket", "Key"}:
            raise InputError(f"GET-only transport refused {operation_name}")
        return self._client._make_api_call(operation_name, api_params)

    def get_object(self, *, Bucket: str, Key: str) -> Any:
        return self._guarded_call("GetObject", {"Bucket": Bucket, "Key": Key})

    def __getattr__(self, name: str) -> Any:
        raise InputError(f"GET-only transport refused S3 member {name}")


def require_env(name: str, environ: Mapping[str, str]) -> str:
    value = environ.get(name, "")
    if not value.strip():
        raise InputError(f"required environment variable {name} is not set")
    return value


def production_store(v2_root: Path = V2_ROOT) -> GetOnlyStore:
    """Build R2Store without its constructor's ListObjectsV2 preflight.

    The credential values are consumed only by boto3 and are never returned,
    logged, repr'd, added to a plan or written to the target directory.
    """
    root = v2_root.resolve()
    if not root.is_dir() or not (root / "sn120_v2" / "r2.py").is_file():
        raise InputError("invalid production v2 directory")
    sys.path.insert(0, str(root))
    try:
        r2 = importlib.import_module("sn120_v2.r2")
        boto3 = importlib.import_module("boto3")
        config_module = importlib.import_module("botocore.config")
        environ = os.environ
        env = {name: require_env(name, environ) for name in
               ("SN120_R2_ENDPOINT", "SN120_R2_BUCKET", "SN120_R2_ACCESS_KEY_ID", "SN120_R2_SECRET_ACCESS_KEY")}
        client = boto3.client(
            "s3", endpoint_url=env["SN120_R2_ENDPOINT"], region_name="auto",
            aws_access_key_id=env["SN120_R2_ACCESS_KEY_ID"],
            aws_secret_access_key=env["SN120_R2_SECRET_ACCESS_KEY"],
            config=config_module.Config(
                signature_version="s3v4",
                s3={"addressing_style": "path", "payload_signing_enabled": True},
                retries={"max_attempts": 5, "mode": "standard"},
                connect_timeout=10, read_timeout=120, max_pool_connections=4,
                request_checksum_calculation="when_required", response_checksum_validation="when_required"))
        raw = object.__new__(r2.R2Store)
        raw._bucket = env["SN120_R2_BUCKET"]
        prefix = environ.get("SN120_R2_PREFIX", "")
        prefix = prefix.strip("/")
        raw._prefix = f"{safe_key(prefix)}/" if prefix else ""
        raw._s3 = GetOnlyS3(client)
        # Assigning a delegate's methods cannot bypass the method/transport guards.
        return GetOnlyStore(raw)
    finally:
        sys.path.remove(str(root))


def atomic_write(root: Path, relative: str, stream: Any, expected_sha: str | None,
                 expected_size: int | None) -> tuple[str, int]:
    target = root.joinpath(*safe_relative(relative).split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or target.is_symlink():
        raise InputError("target path traverses a symlink")
    partial = target.with_name(target.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise InputError("target already has a partial download")
    digest = hashlib.sha256()
    size = 0
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    handle = os.open(partial, flags, 0o600)
    try:
        with os.fdopen(handle, "wb") as out:
            while True:
                chunk = stream.read(4 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        actual = digest.hexdigest()
        if expected_sha is not None and actual != expected_sha:
            raise InputError("downloaded object SHA-256 mismatch")
        if expected_size is not None and size != expected_size:
            raise InputError("downloaded object size mismatch")
        os.replace(partial, target)
        return actual, size
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def store_get_stream(store: Any, key: str) -> Any:
    """A direct R2 get_object stream; Local/fake stores may return bytes."""
    raw = store._delegate if isinstance(store, GetOnlyStore) else store
    s3 = getattr(raw, "_s3", None)
    if s3 is not None:
        bucket = getattr(raw, "_bucket", None)
        prefix = getattr(raw, "_prefix", None)
        if not isinstance(bucket, str) or not isinstance(prefix, str):
            raise InputError("invalid R2 store configuration")
        response = s3.get_object(Bucket=bucket, Key=prefix + safe_key(key))
        body = response.get("Body") if isinstance(response, dict) else None
        if body is None or not hasattr(body, "read"):
            raise InputError("GET response has no readable body")
        return body
    return io.BytesIO(store.get(safe_key(key)))


def fetch_plan(plan: Mapping[str, Any], target: Path, store: Any, save_receipt: bool = True,
               results_only: bool = False) -> dict[str, Any]:
    if plan.get("schema") != SCHEMA:
        raise InputError("unrecognized frozen-input plan schema")
    if target.exists():
        raise InputError("fetch target must be a new, empty path that does not exist")
    target = target.resolve()
    target.mkdir(mode=0o700)
    if results_only and (not plan.get("derive_from_journal") or not plan.get("results_only")
                         or any(obj["kind"] not in ("result", "first_seen") for obj in plan.get("objects", []))):
        raise InputError("results-only fetch requires a fresh plan without resolved results or batches")
    receipt = {"schema": SCHEMA + "-receipt", "journal_sha256": plan.get("journal_sha256"),
               "target": str(target), "fetched": [], "optional_missing": [], "guards": {
                   "store_methods": ["get"], "s3_operations": ["GetObject"],
                   "r2_list_preflight": "bypassed; never called"}}
    try:
        # Validate the whole plan before the first GET or local object write.
        seen_keys, seen_paths = set(), set()
        for obj in plan["objects"]:
            if (not isinstance(obj, dict) or set(obj) != {"kind", "key", "path", "sha256", "size", "required"}
                    or obj["kind"] not in ("batch", "result", "lora", "first_seen")
                    or type(obj["required"]) is not bool):
                raise InputError("malformed plan object")
            safe_key(obj["key"])
            safe_relative(obj["path"])
            if obj["sha256"] is not None and not is_sha256(obj["sha256"]):
                raise InputError("plan contains an invalid SHA-256")
            if obj["size"] is not None and (type(obj["size"]) is not int or obj["size"] < 0):
                raise InputError("plan contains an invalid size")
            if obj["kind"] == "batch" and (not obj["required"] or obj["sha256"] is None):
                raise InputError("batch objects must be required and hash-pinned")
            if obj["kind"] == "lora" and (not obj["required"] or obj["sha256"] is None or obj["size"] is None):
                raise InputError("LoRA objects must be required and hash/size-pinned")
            if obj["kind"] in ("result", "first_seen") and (obj["required"] or obj["sha256"] is not None or obj["size"] is not None):
                raise InputError("optional result/ledger objects cannot have fabricated pins")
            if obj["key"] in seen_keys or obj["path"] in seen_paths:
                raise InputError("plan has duplicate keys or paths")
            seen_keys.add(obj["key"])
            seen_paths.add(obj["path"])
        for obj in plan["objects"]:
            body = None
            try:
                body = store_get_stream(store, obj["key"])
                digest, size = atomic_write(target, obj["path"], body, obj["sha256"], obj["size"])
            except KeyError:
                if obj["required"]:
                    raise InputError("required object is missing") from None
                receipt["optional_missing"].append({"key": obj["key"], "path": obj["path"]})
                continue
            finally:
                if body is not None:
                    body.close()
            receipt["fetched"].append({"key": obj["key"], "path": obj["path"], "sha256": digest, "size": size})
        if results_only:
            write_private(target / "source-journal.json", {
                "schema": SCHEMA + "-source", "journal_sha256": plan["journal_sha256"],
                "rounds": plan["rounds"], "sealed_results_fetched": len(receipt["fetched"]),
                "optional_missing": len(receipt["optional_missing"])})
        if save_receipt:
            write_private(target / "fetch-receipt.json", receipt)
        return receipt
    except BaseException:
        # No store mutation is used for cleanup. Partial local files are removed;
        # successfully fetched evidence is retained for diagnosis in the new dir.
        for partial in target.rglob("*.part"):
            partial.unlink(missing_ok=True)
        raise


def select_sealed_results(target: Path, records: Sequence[Mapping[str, Any]], rounds: Sequence[str],
                          expected_journal_sha256: str | None = None) -> dict[str, bytes]:
    """Read a results-only fetch and prove it came from this exact journal.

    Results-only fetch does not store private object keys by default. Its local
    provenance contains only the source journal SHA and counts. A caller may
    provide an independently retained receipt with the same source SHA.
    """
    provenance = target / "source-journal.json"
    try:
        source = read_json(provenance)
    except InputError:
        source = None
    if source is not None:
        if not isinstance(source, dict) or source.get("schema") != SCHEMA + "-source":
            raise InputError("invalid results-only journal provenance")
        actual = source.get("journal_sha256")
    elif expected_journal_sha256 is not None:
        # A results-only plan carries this expected source hash and is retained
        # by the operator; never guess it from a directory name or later journal.
        actual = expected_journal_sha256
    else:
        receipt = read_json(target / "fetch-receipt.json")
        if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA + "-receipt":
            raise InputError("results-only directory needs source-journal.json or an explicit journal hash")
        actual = receipt.get("journal_sha256")
    if not is_sha256(actual) or actual != expected_journal_sha256:
        raise InputError("sealed-result source journal does not match this frozen input")
    selected = {}
    for round_id in rounds:
        rnd = round_open(records, round_id)
        for lease in rnd["leases"]:
            reference = result_reference(records, round_id, lease["lease_id"])
            if reference is None:
                continue
            key = f"private/rounds/{round_id}/{lease['lease_id']}/result.json"
            path = target / "results" / round_id / f"{lease['lease_id']}.json"
            try:
                raw = path.read_bytes()
            except FileNotFoundError as exc:
                raise InputError(f"{round_id}: journal-referenced sealed result was not fetched") from exc
            if result_digest(raw) != reference:
                raise InputError(f"{round_id}: fetched sealed result differs from the journal")
            validate_result(raw, rnd, lease, reference)
            selected[key] = raw
    return selected


def plan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="frozen-input-plan", description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--journal-sha256", help="optional expected SHA; the supplied frozen journal hash is not implicit")
    parser.add_argument("--round", action="append", dest="rounds", help="repeat; default r000085 through r000088")
    parser.add_argument("--root", required=True, help="local path prefix used in emitted dry-run commands")
    parser.add_argument("--output", type=Path, help="new private JSON plan; required unless --stdout/--summary")
    parser.add_argument("--params", help="optional literal path passed through to dedup-dryrun")
    parser.add_argument("--v2-root", default=str(V2_ROOT), help="operator-selected production worktree's v2 directory")
    parser.add_argument("--python", default="/home/bb/scratch/kingx2-genfeed-rebase/testenv/bin/python")
    parser.add_argument("--results-dir", type=Path, help="directory from a prior fetch of sealed results")
    parser.add_argument("--results-only", action="store_true", help="private plan with only referenced results; use fetch --results-only")
    parser.add_argument("--first-seen", type=Path, help="optional local private/dedup/first_seen.json; no timestamps are invented")
    parser.add_argument("--fetch-first-seen", action="store_true", help="request one GET of private/dedup/first_seen.json in this plan")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stdout", action="store_true", help="print exact commands only, not private keys")
    mode.add_argument("--summary", action="store_true", help="print counts and exact commands, not private keys")
    return parser


def fetch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="frozen-input-fetch", description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True, help="new path; must not exist")
    parser.add_argument("--journal", type=Path, help="with a commands-only plan: verified journal for key derivation")
    parser.add_argument("--results-dir", type=Path, help="with a commands-only plan: previously fetched sealed results")
    parser.add_argument("--save-receipt", action="store_true", help="save private keys in fetch-receipt.json; off by default")
    parser.add_argument("--results-only", action="store_true", help="fetch only journal-referenced sealed results, not batches/LoRAs")
    parser.add_argument("--v2-root", type=Path, default=V2_ROOT,
                        help="operator-selected production worktree's v2 directory; no service is started")
    parser.add_argument("--fake-store-module", help=argparse.SUPPRESS)  # offline qualification only
    parser.add_argument("--allow-test-store", action="store_true", help=argparse.SUPPRESS)
    return parser


def cmd_plan(args: argparse.Namespace) -> int:
    if args.output is None and not (args.stdout or args.summary):
        raise InputError("choose --output for a private plan or --stdout/--summary for commands only")
    if args.output is not None and args.output.exists():
        raise InputError("plan output must not exist")
    if args.output is not None and (args.stdout or args.summary):
        raise InputError("--output cannot be combined with --stdout/--summary")
    if args.fetch_first_seen and (args.stdout or args.summary):
        raise InputError("--fetch-first-seen needs a private --output plan; fetch it before regenerating commands")
    if args.fetch_first_seen and args.results_dir is not None:
        raise InputError("fetch results before generating a plan that includes a first-seen GET")
    if args.results_only and (args.results_dir is not None or args.stdout or args.summary):
        raise InputError("--results-only requires --output without --results-dir/--stdout/--summary")
    results = None
    if args.results_dir is not None:
        records = read_journal(args.journal)
        results = select_sealed_results(args.results_dir, records, args.rounds or DEFAULT_ROUNDS,
                                        expected_journal_sha256=args.journal_sha256)
    plan = make_plan(args.journal, args.root, args.rounds or DEFAULT_ROUNDS, args.params, args.python,
                     results, args.first_seen, args.journal_sha256, args.fetch_first_seen, args.v2_root,
                     results_only=args.results_only)
    if args.output is None and not args.summary:
        args.stdout = True
    if args.stdout or args.summary:
        if args.stdout:
            emit_commands(plan, None)
        else:
            print(json.dumps(public_summary(plan), indent=2, sort_keys=True))
    else:
        write_private(args.output, plan)
        print(json.dumps(public_summary(plan), indent=2, sort_keys=True))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    plan = read_json(args.plan)
    if not isinstance(plan, dict):
        raise InputError("plan must be a JSON object")
    if args.fake_store_module:
        if not args.allow_test_store:
            raise InputError("the fake-store qualification hook requires --allow-test-store")
        # Qualification hook only. It is deliberately hidden and must never be
        # used for production fetching; the operator command omits it.
        module = importlib.import_module(args.fake_store_module)
        store = GetOnlyStore(module.store())
    else:
        store = production_store(args.v2_root)
    if plan.get("derive_from_journal"):
        if args.journal is None or args.results_dir is None:
            raise InputError("commands-only fetch needs --journal and --results-dir")
        records = read_journal(args.journal)
        results = select_sealed_results(args.results_dir, records, plan["rounds"],
                                        expected_journal_sha256=plan["journal_sha256"])
        root = str(args.target)
        plan = make_plan(args.journal, root, plan["rounds"], plan.get("params"),
                         plan.get("python", sys.executable), results,
                         journal_sha256=plan["journal_sha256"], fetch_first_seen=plan["fetch_first_seen"],
                         v2_root=plan.get("v2_root", str(V2_ROOT)))
    elif args.journal is not None or args.results_dir is not None:
        raise InputError("--journal/--results-dir belong only to a commands-only plan")
    if args.results_only and (not plan.get("derive_from_journal") or not plan.get("results_only")
                              or args.journal is not None or args.results_dir is not None):
        raise InputError("results-only needs a plan generated with --results-only before resolving sealed results")
    receipt = fetch_plan(plan, args.target, store, save_receipt=args.save_receipt,
                         results_only=args.results_only)
    print(json.dumps({"target": str(args.target), "fetched": len(receipt["fetched"]),
                      "optional_missing": len(receipt["optional_missing"]),
                      "guards": receipt["guards"]}, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "fetch":
        args = fetch_parser().parse_args(argv[1:])
        handler = cmd_fetch
    else:
        args = plan_parser().parse_args(argv)
        handler = cmd_plan
    try:
        return handler(args)
    except Exception as exc:
        # Normalize all messages: library/network exceptions may otherwise echo
        # signed URLs, environment values or private object names.
        print(f"refused: {type(exc).__name__}: frozen-input operation failed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
