"""Public model/task rollouts. Capture and publication never decide a score.

The evaluator streams local gzip files as grades arrive. At round close only
null/head and an applied winner are queued; every other candidate capture is
unlinked locally and was never uploaded. A disk-backed publication queue
survives a controller restart. The background worker retries store failures.
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
from .types import BoxResult, CAP, NOT_RUN, NULL_MODEL, PASS, RoundTicket

SCHEMA = "sn120-v2-rollouts-gate-1"


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


def read_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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
    """Thread-safe append-only local spool. No model text is held in memory."""

    def __init__(self, root: Path, pool):
        self.root, self.pool = Path(root), pool
        self.lock = threading.Lock()
        self.round_id = ""
        self.files: dict[str, Any] = {}
        self.paths: dict[str, Path] = {}

    def start(self, request) -> None:
        with self.lock:
            if self.round_id == request.round_id:
                return
            self._close_unlocked()
            self.round_id = request.round_id

    def path(self, round_id: str, model: str) -> Path:
        # The model id is a content hash, not an arbitrary local file name.
        return self.root / round_id / "captures" / f"{model}.jsonl.gz"

    def reply(self, request, job, text: str, capped: bool, output_tokens: int, finish_reason: str) -> CapturedReply:
        """Preserve the reply outside scoring; _grade adds its outcome and answer."""
        return CapturedReply(text, capped, output_tokens, finish_reason)

    def graded(self, request, job, reply, code: str, grade=None) -> None:
        model, sample, index = job
        self.start(request)
        task = self.pool.task(request.task_ids[index])
        if isinstance(reply, CapturedReply):
            text, capped = reply.text, reply.capped
            tokens, finish = reply.output_tokens, reply.finish_reason
        else:
            text, capped = reply if reply is not None else (None, False)
            tokens, finish = 0, None
        extracted = getattr(grade, "answer", None)
        row = dict(round_id=request.round_id, role="head" if model == NULL_MODEL else "candidate",
                   model_version_id=request.head.version_id if model == NULL_MODEL else model,
                   winner_uid=None, sample_index=sample, task_pos=index, task_id=request.task_ids[index],
                   prompt=task.messages(), gold_answer=getattr(task, "answer", None),
                   completion_text=text, extracted_answer=extracted, outcome=code,
                   reward=1.0 if code == PASS else 0.0, capped=code == CAP,
                   output_tokens=tokens, finish_reason=finish)
        with self.lock:
            if model not in self.files:
                path = self.path(request.round_id, model)
                path.parent.mkdir(parents=True, exist_ok=True)
                self.paths[model] = path
                self.files[model] = gzip.GzipFile(filename="", mode="wb", fileobj=path.open("ab"), mtime=0)
            data = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode() + b"\n"
            self.files[model].write(data)
            self.files[model].flush()

    def close(self) -> None:
        with self.lock:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        for handle in self.files.values():
            try:
                handle.close()
            except OSError:
                pass
        self.files.clear()
        self.round_id = ""


def capture_file(root: Path, round_id: str, model: str) -> Path:
    return Path(root) / round_id / "captures" / f"{model}.jsonl.gz"


def published_rows(capture: Path, ticket, result, role: str, model: str, winner_uid: int | None):
    """Latest capture per scored key, with bounded-memory disk deduplication.

    A generation retry appends a fresh gzip member. Offset-index replacement
    keeps the latest row. The SQLite index is disposable and never contains text.
    """
    import sqlite3
    db_path = capture.with_name("positions.sqlite")
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("create table if not exists positions (sample integer, task_pos integer, offset integer, primary key(sample, task_pos))")
        offset = 0
        with capture.open("rb") as raw:
            while True:
                start = offset
                with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
                    for line in gz:
                        row = json.loads(line)
                        connection.execute("insert or replace into positions values (?, ?, ?)",
                                           (row["sample_index"], row["task_pos"], start))
                offset = raw.tell()
                if not raw.read(1):
                    break
                raw.seek(offset)
        connection.commit()
        outcomes = result.null_outcomes if role == "head" else next(
            entry.outcomes for entry in result.entries if entry.lora is not None and entry.lora.lora_hash == model
        )
        for sample, text in enumerate(outcomes):
            for task_pos, outcome in enumerate(text):
                if outcome == NOT_RUN:
                    continue
                position = connection.execute("select offset from positions where sample=? and task_pos=?", (sample, task_pos)).fetchone()
                if position is None:
                    continue  # a round from before capture never invents a completion
                with capture.open("rb") as raw:
                    raw.seek(position[0])
                    with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
                        found = next((json.loads(line) for line in gz
                                      if (item := json.loads(line))["sample_index"] == sample and item["task_pos"] == task_pos), None)
                if found is None:
                    raise ValueError("a captured rollout position is missing")
                if found["outcome"] != outcome:
                    raise ValueError("a captured outcome differs from the sealed result")
                yield found | dict(role=role, winner_uid=winner_uid,
                                   model_version_id=ticket.head.version_id if role == "head" else model)
    finally:
        connection.close()
        db_path.unlink(missing_ok=True)


class GateRolloutPublisher:
    """Durable local publication jobs, retried away from the settling thread."""

    def __init__(self, store, work_dir: Path, retry_s: float = 30.0):
        self.store, self.root, self.retry_s = store, Path(work_dir) / "rollouts", retry_s
        self.jobs: queue.Queue[Path | None] = queue.Queue()
        self.errors: dict[str, str] = {}
        self.worker = threading.Thread(target=self._run, name="sn120-v2-public-rollouts", daemon=True)
        self.worker.start()
        for path in sorted((self.root / "publication").glob("*.json")):
            if not path.name.endswith(".tickets.json"):
                self.jobs.put(path)

    def enqueue(self, pairs, winner_uid: int | None, params_hash: str, round_seed: str, closed_at: int) -> None:
        """Persist the publication plan before deleting any losing capture."""
        if not pairs or not pairs[0][1].null_outcomes:
            if pairs:
                self.discard(pairs[0][0].round_id)
            return
        ticket, result = pairs[0]
        round_id = ticket.round_id
        winner = next(((ticket, entry) for ticket, result in pairs for seat, entry in zip(ticket.entries, result.entries)
                       if seat.reg.uid == winner_uid and entry.lora is not None), None) if winner_uid is not None else None
        winner_model = winner[1].lora.lora_hash if winner else None
        models = {NULL_MODEL}
        if winner_model is not None:
            models.add(winner_model)
        # The exact small sealed records are enough after a restart; no private
        # batch, path, box id, host or credential is written into the job file.
        ticket_path = self.root / "publication" / f"{round_id}.tickets.json"
        ticket_path.parent.mkdir(parents=True, exist_ok=True)
        ticket_tmp = ticket_path.with_suffix(".tmp")
        from .canonical import canonical_json, to_jsonable
        ticket_tmp.write_bytes(canonical_json([dict(ticket=to_jsonable(ticket), result=to_jsonable(result)) for ticket, result in pairs]))
        os.replace(ticket_tmp, ticket_path)
        capture_dir = self.root / round_id / "captures"
        job = dict(round_id=round_id, winner_uid=winner_uid, params_hash=params_hash, round_seed=round_seed, closed_at=closed_at,
                   head_model=NULL_MODEL, winner_model=winner[1].lora.lora_hash if winner else None)
        publication = self.root / "publication"
        publication.mkdir(parents=True, exist_ok=True)
        path = publication / f"{round_id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, sort_keys=True))
        os.replace(tmp, path)
        for capture in capture_dir.glob("*.jsonl.gz"):
            if capture.name.removesuffix(".jsonl.gz") not in models:
                capture.unlink(missing_ok=True)
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
        combined = hashlib.sha256(json.dumps({row["role"]: row["sha256"] for row in files}, sort_keys=True).encode()).hexdigest()
        meta = dict(schema=SCHEMA, round_id=round_id, rows=sum(row["rows"] for row in files), bytes=sum(row["bytes"] for row in files),
                    sha256=combined, params_hash=job["params_hash"], round_seed=job["round_seed"], closed_at=utc(job["closed_at"]), files=files)
        self.store.put(layout.public_rollout_gate_meta(round_id), json.dumps(meta, sort_keys=True, indent=1).encode())
        path.unlink(missing_ok=True)
        saved.unlink(missing_ok=True)
        shutil.rmtree(self.root / round_id, ignore_errors=True)

    def _run(self) -> None:
        while True:
            path = self.jobs.get()
            if path is None:
                return
            try:
                self._upload(path)
                self.errors.pop(path.stem, None)
            except Exception as exc:  # publication is never a round failure
                self.errors[path.stem] = f"{type(exc).__name__}: {exc}"
                print(f"public rollouts: {path.stem}: {self.errors[path.stem]}; retrying", flush=True)
                time.sleep(self.retry_s)
                self.jobs.put(path)
            finally:
                self.jobs.task_done()

    def close_round(self, pairs, winner_uid, params_hash, round_seed, closed_at) -> None:
        self.enqueue(pairs, winner_uid, params_hash, round_seed, closed_at)

    def wait_idle(self) -> None:
        """Test/operator drain seam; does not form part of round settlement."""
        self.jobs.join()
