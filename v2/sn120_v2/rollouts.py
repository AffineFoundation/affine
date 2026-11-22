"""Public model/task rollouts. Capture and publication never decide a score.

The evaluator appends plain JSONL as grades arrive. At round close null/head,
an applied winner and every scored entry are queued for public merged-model
eval publication. Candidate training batches stay private. A disk-backed
publication queue survives a controller restart. The background worker retries
store failures, parking a job after 20 attempts.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import layout
from .errors import V2Error
from .types import BoxResult, CAP, NOT_RUN, NULL_MODEL, PASS, RoundTicket

SCHEMA = "sn120-v2-rollouts-gate-2"
MIN_CAPTURE_FREE_BYTES = 50 << 30
MAX_PUBLISH_ATTEMPTS = 20


def utc(timestamp: int | float | None) -> str | None:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z") if timestamp is not None else None


def deterministic_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, int, str]:
    """Write one complete deterministic gzip member; return rows, bytes, sha256."""
    tmp = path.with_name(path.name + ".tmp")
    digest = hashlib.sha256()
    rows_n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
            for row in rows:
                data = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode() + b"\n"
                out.write(data)
                rows_n += 1
    with tmp.open("rb") as raw:
        for chunk in iter(lambda: raw.read(8 << 20), b""):
            digest.update(chunk)
    os.replace(tmp, path)
    return rows_n, path.stat().st_size, digest.hexdigest()


@dataclass(frozen=True)
class CapturedReply:
    """The same text/capped decision as before, plus unused-by-scoring metadata."""

    text: str
    capped: bool
    output_tokens: int
    finish_reason: str

    def __iter__(self):
        yield self.text
        yield self.capped

    def __getitem__(self, index: int):
        # Preserve the old tuple API for every caller, including unpacking.
        return (self.text, self.capped)[index]


class RolloutCapture:
    """Append-only local JSONL. No gzip member remains open across settlement."""

    def __init__(self, root: Path, pool):
        self.root, self.pool = Path(root), pool
        self.lock = threading.Lock()
        self.round_id = ""
        self.enabled = True
        self.warned = False

    def start(self, request) -> None:
        with self.lock:
            if self.round_id == request.round_id:
                return
            self.round_id = request.round_id
            self._clean_old()
            free = shutil.disk_usage(self.root.parent).free
            self.enabled = free >= MIN_CAPTURE_FREE_BYTES
            if not self.enabled and not self.warned:
                print(f"public rollout capture disabled: {free} free bytes, below {MIN_CAPTURE_FREE_BYTES}", flush=True)
                self.warned = True

    def _clean_old(self) -> None:
        current = int(self.round_id[-6:]) if self.round_id.startswith("r") and self.round_id[-6:].isdigit() else None
        publication = self.root / "publication"
        for path in self.root.glob("r[0-9][0-9][0-9][0-9][0-9][0-9]"):
            number = int(path.name[1:])
            if current is None or number >= current - 1:
                continue
            if not (publication / f"{path.name}.json").exists():
                shutil.rmtree(path, ignore_errors=True)

    def reply(self, request, job, text: str, capped: bool, output_tokens: int, finish_reason: str) -> CapturedReply:
        return CapturedReply(text, capped, output_tokens, finish_reason)

    def graded(self, request, job, reply, code: str, grade=None) -> None:
        self.start(request)
        if not self.enabled:
            return
        model, sample, index = job
        task = self.pool.task(request.task_ids[index])
        if isinstance(reply, CapturedReply):
            text = reply.text
            tokens, finish = reply.output_tokens, reply.finish_reason
        else:
            text = reply[0] if reply is not None else None
            tokens, finish = 0, None
        row = dict(round_id=request.round_id, role="head" if model == NULL_MODEL else "candidate",
                   model_version_id=request.head.version_id if model == NULL_MODEL else model,
                   winner_uid=None, sample_index=sample, task_pos=index, task_id=request.task_ids[index],
                   prompt=task.messages(), gold_answer=getattr(task, "answer", None),
                   completion_text=text, extracted_answer=getattr(grade, "answer", None), outcome=code,
                   reward=1.0 if code == PASS else 0.0, capped=code == CAP,
                   output_tokens=tokens, finish_reason=finish)
        data = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        with self.lock:
            path = capture_file(self.root, request.round_id, model)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as handle:  # flush+close per row: readers never see a gzip trailer problem
                handle.write(data)
                handle.flush()

    def close(self) -> None:
        pass  # JSONL writes own their file handle; there is no compressor to finalize.


def capture_file(root: Path, round_id: str, model: str) -> Path:
    return Path(root) / round_id / "captures" / f"{model}.jsonl"


def _entry_pair(pairs, queue_pos: int):
    for ticket, result in pairs:
        seat = next((entry for entry in ticket.entries if entry.queue_pos == queue_pos), None)
        result_entry = next((entry for entry in result.entries if entry.queue_pos == queue_pos), None)
        if seat is not None and result_entry is not None and result_entry.lora is not None:
            return ticket, result, seat, result_entry
    raise KeyError(f"no sealed result for queue position {queue_pos}")


def _publication_entry(ticket, result, seat, entry) -> dict[str, Any]:
    return dict(queue_pos=seat.queue_pos, uid=seat.reg.uid, hotkey=seat.reg.hotkey, lora_hash=entry.lora.lora_hash)


def published_rows(capture: Path, ticket, result, role: str, model: str, winner_uid: int | None,
                   identity: dict[str, Any] | None = None):
    """One scan, then one seek per scored row. ``identity`` labels entry files without re-scoring."""
    positions: dict[tuple[int, int], int] = {}
    with capture.open("rb") as raw:
        while True:
            offset = raw.tell()
            line = raw.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if role != "entry":
                    raise  # retain the legacy head/winner failure behavior
                break  # a truncated last entry capture line is not a scored rollout
            positions[row["sample_index"], row["task_pos"]] = offset
    outcomes = result.null_outcomes if role == "head" else next(
        entry.outcomes for entry in result.entries if entry.lora is not None and entry.lora.lora_hash == model
    )
    with capture.open("rb") as raw:
        for sample, text in enumerate(outcomes):
            for task_pos, outcome in enumerate(text):
                if outcome == NOT_RUN:
                    continue
                offset = positions.get((sample, task_pos))
                if offset is None:
                    continue  # a missing/partial capture never invents a completion
                raw.seek(offset)
                found = json.loads(raw.readline())
                if found["outcome"] != outcome:
                    raise ValueError("a captured outcome differs from the sealed result")
                yield found | dict(role=role, winner_uid=winner_uid,
                                   model_version_id=ticket.head.version_id if role == "head" else model,
                                   **(identity or {}))


class GateRolloutPublisher:
    """Durable local publication jobs, retried away from the settling thread."""

    def __init__(self, store, work_dir: Path, retry_s: float = 30.0):
        self.store, self.root, self.retry_s = store, Path(work_dir) / "rollouts", retry_s
        self.jobs: queue.Queue[Path | None] = queue.Queue()
        self.errors: dict[str, str] = {}
        self.parked: dict[str, str] = {}
        self.attempts: dict[str, int] = {}
        for job_path in (self.root / "publication").glob("*.json") if (self.root / "publication").exists() else ():
            if job_path.name.endswith(".tickets.json"):
                continue
            try:
                job = json.loads(job_path.read_text())
                self.attempts[job_path.stem] = int(job.get("attempts", 0))
                if job.get("parked"):
                    self.parked[job_path.stem] = str(job.get("error") or "previous failure")
            except (OSError, ValueError, TypeError):
                self.attempts[job_path.stem] = 0
        self.worker = threading.Thread(target=self._run, name="sn120-v2-public-rollouts", daemon=True)
        self.worker.start()
        for path in sorted((self.root / "publication").glob("*.json")):
            if not path.name.endswith(".tickets.json") and path.stem not in self.parked:
                self.jobs.put(path)

    def enqueue(self, pairs, winner_uid: int | None, params_hash: str, round_seed: str, closed_at: int) -> None:
        """Persist the complete publication plan; gzip and upload only in the worker."""
        if not pairs or not pairs[0][1].null_outcomes:
            if pairs:
                self.discard(pairs[0][0].round_id)
            return
        ticket, result = pairs[0]
        round_id = ticket.round_id
        winner = next(((ticket, entry) for ticket, result in pairs for seat in ticket.entries for entry in result.entries
                       if seat.queue_pos == entry.queue_pos and seat.reg.uid == winner_uid and entry.lora is not None), None) if winner_uid is not None else None
        winner_model = winner[1].lora.lora_hash if winner else None
        entries = sorted([
            _publication_entry(*_entry_pair(pairs, entry.queue_pos))
            for _, result in pairs for entry in result.entries
            if entry.outcomes and entry.lora is not None
        ], key=lambda entry: entry["queue_pos"])
        # The exact small sealed records are enough after a restart; no private
        # batch, path, box id, host or credential is written into the job file.
        ticket_path = self.root / "publication" / f"{round_id}.tickets.json"
        ticket_path.parent.mkdir(parents=True, exist_ok=True)
        ticket_tmp = ticket_path.with_suffix(".tmp")
        from .canonical import canonical_json, to_jsonable
        ticket_tmp.write_bytes(canonical_json([dict(ticket=to_jsonable(ticket), result=to_jsonable(result)) for ticket, result in pairs]))
        os.replace(ticket_tmp, ticket_path)
        job = dict(round_id=round_id, winner_uid=winner_uid, params_hash=params_hash, round_seed=round_seed, closed_at=closed_at,
                   head_model=NULL_MODEL, winner_model=winner_model, entries=entries, attempts=0, parked=False)
        publication = self.root / "publication"
        publication.mkdir(parents=True, exist_ok=True)
        path = publication / f"{round_id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, sort_keys=True))
        os.replace(tmp, path)
        models = {NULL_MODEL, winner_model, *(entry["lora_hash"] for entry in entries)}
        for capture in (self.root / round_id / "captures").glob("*.jsonl"):
            if capture.name.removesuffix(".jsonl") not in models:
                capture.unlink(missing_ok=True)  # no outcomes: never published and not needed by a restart
        self.jobs.put(path)

    def discard(self, round_id: str) -> None:
        shutil.rmtree(self.root / round_id, ignore_errors=True)
        (self.root / "publication" / f"{round_id}.json").unlink(missing_ok=True)

    def _upload(self, path: Path) -> None:
        job = json.loads(path.read_text())
        saved = path.with_name(path.name.removesuffix(".json") + ".tickets.json")
        pairs = [(RoundTicket.from_dict(item["ticket"]), BoxResult.from_dict(item["result"])) for item in json.loads(saved.read_text())]
        pairs = [(ticket, result) for ticket, result in pairs if ticket.round_id == job["round_id"]]
        if not pairs:
            raise KeyError(f"no sealed result for {job['round_id']}")
        round_id = job["round_id"]
        files = []
        publication_failures: list[str] = []
        for role, model in (("head", NULL_MODEL), ("winner", job.get("winner_model"))):
            if model is None:
                continue
            if role == "winner":
                ticket, result = next((ticket, result) for ticket, result in pairs
                                      if any(entry.lora is not None and entry.lora.lora_hash == model for entry in result.entries))
            else:
                ticket, result = pairs[0]
            capture = capture_file(self.root, round_id, model)
            local = self.root / round_id / f"{role}.jsonl.gz"
            rows_n, size, sha = deterministic_gzip(local, published_rows(capture, ticket, result, role, model, job["winner_uid"]))
            key = layout.public_rollout_gate(round_id, role)
            if self.store.exists(key) and self.store.sha256(key) == sha:
                files.append(dict(role=role, key=key, rows=rows_n, bytes=size, sha256=sha))
                local.unlink(missing_ok=True)
                continue
            self.store.put_file(key, local)
            if self.store.sha256(key) != sha:
                raise IOError("stored gate rollout sha256 mismatch")
            files.append(dict(role=role, key=key, rows=rows_n, bytes=size, sha256=sha))
            local.unlink(missing_ok=True)

        # Gate-1 jobs have no entries key and retain their head/winner-only
        # behavior. For gate-2, a missing or partial entry is metadata, not a
        # round failure. The existing sealed-outcome equality check still
        # applies to every found scored row.
        for entry in job.get("entries") or []:
            queue_pos = int(entry["queue_pos"])
            ticket, result, seat, result_entry = _entry_pair(pairs, queue_pos)
            if (entry["lora_hash"], entry["uid"], entry["hotkey"]) != (result_entry.lora.lora_hash, seat.reg.uid, seat.reg.hotkey):
                raise ValueError(f"publication entry {queue_pos} differs from its sealed result")
            key = layout.public_rollout_gate_entry(round_id, queue_pos)
            identity = _publication_entry(ticket, result, seat, result_entry)
            capture = capture_file(self.root, round_id, entry["lora_hash"])
            meta = dict(role="entry", key=key, **identity, rows=0, bytes=0, missing=not capture.is_file())
            if meta["missing"]:
                files.append(meta)
                continue
            local = self.root / round_id / "entries" / f"{queue_pos}.jsonl.gz"
            rows_n, size, sha = deterministic_gzip(local, published_rows(
                capture, ticket, result, "entry", entry["lora_hash"], job["winner_uid"], identity,
            ))
            meta |= dict(rows=rows_n, missing=False, bytes=size, sha256=sha)
            if self.store.exists(key) and self.store.sha256(key) == sha:
                files.append(meta)
                local.unlink(missing_ok=True)
                continue
            try:
                self.store.put_file(key, local)
                if self.store.sha256(key) != sha:
                    raise IOError("stored gate entry rollout sha256 mismatch")
            except (V2Error, OSError) as exc:
                # A store/network failure of one entry must not block the others.
                # The uploaded bytes are not listed as verified in this meta.
                publication_failures.append(f"{queue_pos}: {type(exc).__name__}: {exc}")
                meta |= dict(rows=0, bytes=0, missing=True)
                meta.pop("sha256", None)  # the local gzip was not verified in the store
                files.append(meta)
            else:
                files.append(meta)
            local.unlink(missing_ok=True)
        hashes = ({row["role"]: row["sha256"] for row in files} if "entries" not in job else
                  {row["key"]: row["sha256"] for row in files if not row.get("missing")})
        combined = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        meta = dict(schema=SCHEMA if "entries" in job else "sn120-v2-rollouts-gate-1", round_id=round_id,
                    rows=sum(row["rows"] for row in files), bytes=sum(row.get("bytes", 0) for row in files),
                    sha256=combined, params_hash=job["params_hash"], round_seed=job["round_seed"], closed_at=utc(job["closed_at"]), files=files)
        self.store.put(layout.public_rollout_gate_meta(round_id), json.dumps(meta, sort_keys=True, indent=1).encode())
        path.unlink(missing_ok=True)
        saved.unlink(missing_ok=True)
        shutil.rmtree(self.root / round_id, ignore_errors=True)
        for failure in publication_failures:
            print(f"public rollouts: {round_id}: entry {failure} (recorded as missing in meta)", flush=True)

    def _run(self) -> None:
        while True:
            path = self.jobs.get()
            if path is None:
                return
            if path.stem in self.parked:
                self.jobs.task_done()
                continue
            try:
                self._upload(path)
                self.errors.pop(path.stem, None)
                self.attempts.pop(path.stem, None)
            except Exception as exc:  # publication is never a round failure
                attempts = self.attempts.get(path.stem, 0) + 1
                self.attempts[path.stem] = attempts
                self.errors[path.stem] = f"{type(exc).__name__}: {exc}"
                job = json.loads(path.read_text())
                job.update(attempts=attempts, error=self.errors[path.stem], parked=attempts >= MAX_PUBLISH_ATTEMPTS)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(job, sort_keys=True))
                os.replace(tmp, path)
                if job["parked"]:
                    self.parked[path.stem] = self.errors[path.stem]
                    print(f"public rollouts: {path.stem}: parked after {attempts} failed attempts: {self.errors[path.stem]}", flush=True)
                else:
                    print(f"public rollouts: {path.stem}: {self.errors[path.stem]}; retrying ({attempts}/{MAX_PUBLISH_ATTEMPTS})", flush=True)
                    time.sleep(self.retry_s)
                    self.jobs.put(path)
            finally:
                self.jobs.task_done()

    def close_round(self, pairs, winner_uid, params_hash, round_seed, closed_at) -> None:
        self.enqueue(pairs, winner_uid, params_hash, round_seed, closed_at)

    def wait_idle(self) -> None:
        """Test/operator drain seam; does not form part of round settlement."""
        self.jobs.join()
