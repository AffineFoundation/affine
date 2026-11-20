"""Publish verified scorer output, once-ever claims, intake and trailing payout.

The scorer returns evidence; this module owns the public site, permanent
accepted ledger and payout. No reserve feed is read or written.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from ..canonical import canonical_json, is_sha256, sha256_hex
from ..errors import ContractError, VerifyError, V2Error
from .common import PublicSink, append_once, atomic_bytes, atomic_json, json_bytes, jsonl_bytes, read_json, read_jsonl, utc_text
from .config import Config, add_config
from .constants import ACCEPT, BURN_UID, MAX_SAMPLES, MIN_SAMPLES, POOL_ID
from .decisions import decision as protocol_decision
from .epoch import epoch_deadline, epoch_id, rollout_object
from .payout import Claim, first_accepted_claims, payout_document, payout_weights, trailing_claims
from .protocol import valid_task_id
from .snapshot import accepted_ids, load_history


class PoolRows(Protocol):
    def row(self, task_id: str) -> Mapping[str, Any]: ...


def _count(value: Any, where: str, maximum: int = MAX_SAMPLES) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise VerifyError(f"{where} must be an integer in 0..{maximum}")
    return value


def _task_row(raw: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    if raw.get("task_id", raw.get("id")) != expected["task_id"]:
        raise VerifyError("scorer result contains an unexpected or duplicate task ID")
    # Accept the scorer's compact k/n names, but publish the protocol's s/n names.
    passes = raw.get("s", raw.get("k"))
    total = raw.get("n")
    passes, total = _count(passes, "scorer s"), _count(total, "scorer n")
    if passes > total:
        raise VerifyError("scorer passes exceed total samples")
    action = raw.get("decision")
    q_value = raw.get("q")
    if action == "unscored":
        if total >= MIN_SAMPLES or q_value is not None:
            raise VerifyError("unscored scorer result is inconsistent")
        return {"task_id": expected["task_id"], "hotkey": expected["hotkey"], "uid": expected["uid"], "s": passes, "n": total, "q": None, "decision": action}
    if total < MIN_SAMPLES:
        raise VerifyError("a task below the minimum samples cannot be admitted")
    final = action in (ACCEPT, "reject") and total < MAX_SAMPLES
    expected_decision = protocol_decision(passes, total, final=final)
    if expected_decision.action != action:
        raise VerifyError("scorer decision does not match the protocol")
    if type(q_value) not in (int, float) or not math.isfinite(q_value) or abs(float(q_value) - expected_decision.q) > 1e-12:
        raise VerifyError("scorer q does not match the protocol")
    return {
        "task_id": expected["task_id"], "hotkey": expected["hotkey"], "uid": expected["uid"],
        "s": passes, "n": total, "q": float(q_value), "decision": action,
    }


def normalize_output(output: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    """Require exact task coverage and bind results to the frozen private job."""
    epoch = epoch_id(job["epoch"])
    if output.get("epoch", epoch) != epoch:
        raise VerifyError("scorer output belongs to another epoch")
    if output.get("pool", POOL_ID) != POOL_ID or job.get("pool") != POOL_ID:
        raise VerifyError("scorer output is not bound to the pinned pool")
    if output.get("king", output.get("head_version", job["king"])) != job["king"]:
        raise VerifyError("scorer did not use the frozen king")
    commitment = output.get("seed_commitment", job["seed_commitment"])
    if commitment != job["seed_commitment"] or not is_sha256(commitment):
        raise VerifyError("scorer seed commitment does not match the job")
    expected = {row["task_id"]: row for row in job["tasks"]}
    tasks = output.get("tasks", output.get("results"))
    if not isinstance(tasks, list):
        raise VerifyError("scorer output needs a tasks array")
    if len(tasks) != len(expected):
        raise VerifyError("scorer output does not cover exactly the snapshot's unique tasks")
    normalized, seen = [], set()
    for raw in tasks:
        if not isinstance(raw, Mapping):
            raise VerifyError("scorer task result is not an object")
        row = _task_row(raw, expected.get(raw.get("task_id", raw.get("id")), {"task_id": None}))
        if row["task_id"] in seen:
            raise VerifyError("scorer output repeats a task ID")
        seen.add(row["task_id"])
        normalized.append(row)
    if seen != set(expected):
        raise VerifyError("scorer output omits tasks from the snapshot")
    return {"epoch": epoch, "king": job["king"], "seed_commitment": commitment, "tasks": normalized}


def _rollout_rows(raw: Any, task_id: str) -> list[dict[str, Any]]:
    if isinstance(raw, (str, bytes)):
        rows = read_jsonl(raw, f"rollouts for {task_id}")
    elif isinstance(raw, list):
        if not all(isinstance(row, dict) for row in raw):
            raise VerifyError("scorer rollouts must be JSON objects")
        rows = list(raw)
    else:
        raise VerifyError("scorer rollouts must be a list or JSONL bytes")
    return rows


def load_rollouts(out_dir: Path, output: Mapping[str, Any], accepted: Iterable[str]) -> dict[str, bytes]:
    """Only accepted tasks may have public full-text rollout files."""
    accepted = set(accepted)
    found: dict[str, bytes] = {}
    embedded = output.get("rollouts", {})
    if isinstance(embedded, list):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in embedded:
            if not isinstance(row, dict) or not valid_task_id(row.get("task_id")):
                raise VerifyError("invalid task ID in scorer rollouts")
            grouped.setdefault(row["task_id"], []).append({k: v for k, v in row.items() if k != "task_id"})
        embedded = grouped
    if not isinstance(embedded, Mapping):
        raise VerifyError("scorer rollouts must be keyed by task ID")
    for task_id, rows in embedded.items():
        if task_id in accepted:
            found[task_id] = jsonl_bytes(_rollout_rows(rows, task_id))
    # Optional out/rollouts/<task_id>.jsonl keeps huge texts out of results.json.
    rollout_dir = out_dir / "rollouts"
    if rollout_dir.is_dir():
        for path in rollout_dir.glob("*.jsonl"):
            task_id = path.stem
            if not valid_task_id(task_id):
                raise VerifyError("invalid rollout filename")
            if task_id in accepted:
                found[task_id] = _check_rollout_bytes(path.read_bytes(), task_id)
    if set(found) != accepted:
        raise VerifyError("full-text rollouts must be present for every accepted task and no rejected task")
    for task_id, raw in found.items():
        _check_rollout_bytes(raw, task_id)
    return found


def _check_rollout_bytes(raw: bytes, task_id: str) -> bytes:
    rows = _rollout_rows(raw, task_id)
    for row in rows:
        if "text" not in row and "output" not in row and "response" not in row:
            raise VerifyError("an accepted rollout must contain full generated text")
        if "pass" in row and type(row["pass"]) is not bool:
            raise VerifyError("rollout pass must be boolean")
        if "grade" in row and (not isinstance(row["grade"], Mapping) or type(row["grade"].get("passed")) is not bool):
            raise VerifyError("accepted rollout needs a complete boolean grade")
        if "seed" in row and (type(row["seed"]) is not int or row["seed"] < 0):
            raise VerifyError("rollout seed must be a nonnegative integer")
    return jsonl_bytes(rows)


def publication(
    job: Mapping[str, Any],
    output: Mapping[str, Any],
    *,
    out_dir: Path,
    secret: str,
    history: Iterable[Mapping[str, Any]],
    registered: Mapping[int, str],
    now: float,
    pool_rows: PoolRows,
    frozen_at: float,
) -> dict[str, Any]:
    """Pure publication plan; no file or remote store is modified here."""
    epoch = epoch_id(job["epoch"])
    normalized = normalize_output(output, job)
    if not isinstance(secret, str) or not secret or sha256_hex(secret.encode()) != normalized["seed_commitment"]:
        raise VerifyError("the revealed seed secret does not match the window commitment")
    if not epoch_deadline(epoch).timestamp() <= frozen_at <= epoch_deadline(epoch).timestamp() + 120:
        raise VerifyError("the king was not captured within the epoch deadline grace period")
    rows = normalized["tasks"]
    accepted = [row for row in rows if row["decision"] == ACCEPT]
    by_id = {row["task_id"]: row for row in rows}
    jobs = {row["task_id"]: row for row in job["tasks"]}
    claims = _history_claims(history)
    uploads = []
    for row in accepted:
        task = jobs[row["task_id"]]
        uploads.append({
            "task_id": row["task_id"], "uid": task["uid"], "hotkey": task["hotkey"],
            "uploaded_at": task["uploaded_at"], "file_sha256": task["file_sha256"], "valid": True,
        })
    claims = first_accepted_claims(uploads, claims)
    counts: dict[str, dict[str, int]] = {}
    for row in job["tasks"]:
        counts.setdefault(row["hotkey"], {"submitted": 0, "eligible": 0, "scored": 0, "accepted": 0})
    for hotkey, count in job.get("counts", {}).items():
        if hotkey in counts and type(count.get("submitted")) is int:
            counts[hotkey]["submitted"] = count["submitted"]
    for row in job["tasks"]:
        counts[row["hotkey"]]["eligible"] += 1
        counts[row["hotkey"]]["scored"] += 1
    for row in accepted:
        counts[row["hotkey"]]["accepted"] += 1
    results = {
        "version": "affine-generator-results-v1", "epoch": epoch,
        "published_at": utc_text(now), "pool": POOL_ID, "king": job["king"],
        "king_frozen_at": utc_text(frozen_at), "seed_commitment": normalized["seed_commitment"],
        "seed_secret": secret, "hotkeys": counts, "tasks": [
            {
                "task_id": row["task_id"], "decision": row["decision"], "s": row["s"], "n": row["n"],
                "q": row["q"], "rollouts": rollout_object(epoch, f"{row['task_id']}.jsonl"),
                "hotkey": row["hotkey"], "uid": row["uid"],
            } if row["decision"] == ACCEPT else {
                "task_sha256": sha256_hex(row["task_id"].encode()),
                "decision": row["decision"], "n": row["n"], "s": row["s"],
            }
            for row in rows
        ],
        "counts": {"tasks": len(rows), "accepted": len(accepted), "rejected": len(rows) - len(accepted)},
    }
    accepted_rows = []
    intake_rows = []
    for row in accepted:
        task = jobs[row["task_id"]]
        accepted_rows.append({
            "task_id": row["task_id"], "id": row["task_id"], "epoch": epoch,
            "submitter": task["hotkey"], "hotkey": task["hotkey"], "uid": task["uid"],
            "uploaded_at": task["uploaded_at"], "uploaded_at_utc": utc_text(task["uploaded_at"]),
            "file_sha256": task["file_sha256"], "s": row["s"], "n": row["n"], "q": row["q"],
            "k": row["s"], "king": job["king"],
        })
        source = pool_rows.row(row["task_id"])
        problem, answer, source_id = source.get("problem"), source.get("answer"), source.get("source_id")
        if not all(isinstance(value, str) for value in (problem, answer, source_id)):
            raise VerifyError("the pinned pool row lacks problem, answer or source_id")
        # The mixer receives only the exact published pool identity and measured rate.
        intake_rows.append({"problem": problem, "answer": answer, "source_id": source_id, "prior_p": row["s"] / row["n"] if row["n"] else 0.0})
    rollouts = load_rollouts(out_dir, output, {row["task_id"] for row in accepted})
    history_rows = list(history)
    old_claims = _history_claims(history_rows)
    recent = trailing_claims(claims.values(), now)
    payout = payout_document(claims.values(), now=now, updated_at=now)
    payout["weights"] = {str(uid): weight for uid, weight in payout_weights(claims.values(), registered, now=now, updated_at=now).items()}
    payout["epochs"] = sorted({str(row["epoch"]) for row in history_rows if isinstance(row.get("epoch"), str)} | {epoch})[-6:]
    # Claims outside the trailing window remain in accepted-all but never in payout.
    payout["window"] = {"start": utc_text(now - 86400), "end": utc_text(now), "exclusive_start": True, "inclusive_end": True}
    return {
        "results": results, "accepted": accepted_rows, "intake": intake_rows, "rollouts": rollouts,
        "claims": claims, "old_claims": old_claims, "payout": payout,
        "state_counts": {"eligible": len(rows), "scored": len(rows), "accepted": len(accepted)},
    }


def _history_claims(rows: Iterable[Mapping[str, Any] | Claim]) -> dict[str, Claim]:
    result = {}
    for row in rows:
        if isinstance(row, Claim):
            claim = row
        else:
            claim = Claim(
                task_id=str(row.get("task_id", row.get("id"))), uid=row.get("uid"),
                hotkey=str(row.get("hotkey", row.get("submitter"))), uploaded_at=row.get("uploaded_at"),
                file_sha256=row.get("file_sha256"),
            )
        if claim.task_id in result:
            raise VerifyError("the permanent accepted ledger contains a duplicate ID")
        result[claim.task_id] = claim
    return result


def _validate_history(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result = list(rows)
    ids = accepted_ids(result)
    for row in result:
        task_id, hotkey = row.get("task_id", row.get("id")), row.get("hotkey", row.get("submitter"))
        if not isinstance(hotkey, str) or not hotkey or type(row.get("uid")) is not int:
            raise VerifyError("accepted-all ledger lacks its earning hotkey or UID")
        if type(row.get("uploaded_at")) not in (int, float) or not math.isfinite(row["uploaded_at"]):
            raise VerifyError("accepted-all ledger lacks its store upload timestamp")
        if not is_sha256(row.get("file_sha256")):
            raise VerifyError("accepted-all ledger lacks its winning submission hash")
        if row.get("task_id") not in ids:
            raise VerifyError("accepted-all ledger task identity is inconsistent")
    return result


def publish(
    *, sink: PublicSink, job: Mapping[str, Any], output: Mapping[str, Any], out_dir: Path,
    secret: str, registered: Mapping[int, str], now: float, pool_rows: PoolRows, frozen_at: float,
    state_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Idempotently publish an epoch; intake is last, then its digest and state."""
    epoch = epoch_id(job["epoch"])
    raw_history, _ = load_history(sink)
    prefix = f"epochs/{epoch}"
    complete = sink.exists(f"{prefix}/intake.sha256") and sink.exists(f"{prefix}/intake.jsonl")
    if complete:
        if sink.read(f"{prefix}/intake.sha256") != f"{sha256_hex(sink.read(f'{prefix}/intake.jsonl'))}  intake.jsonl\n".encode():
            raise VerifyError(f"published {epoch} intake checksum is invalid")
        for row in read_json(sink.path("state.json")).get("published_epochs", []):
            if row.get("epoch") == epoch:
                return {"status": "already_complete", "epoch": epoch, "accepted": []}
        raise VerifyError(f"published {epoch} is complete but state.json lacks its published_epochs entry")
    history = _validate_history(read_jsonl(raw_history, "accepted-all.jsonl"))
    plan = publication(job, output, out_dir=out_dir, secret=secret, history=history, registered=registered, now=now, pool_rows=pool_rows, frozen_at=frozen_at)
    if any(row["task_id"] in plan["old_claims"] for row in plan["accepted"]):
        raise VerifyError("snapshot admitted a task that was already accepted; refusing duplicate payment")
    results_raw = json_bytes(plan["results"])
    accepted_raw = jsonl_bytes(plan["accepted"])
    intake_raw = jsonl_bytes(plan["intake"])
    sink.put(f"{prefix}/results.json", results_raw)
    sink.put(f"{prefix}/accepted.jsonl", accepted_raw)
    for task_id in sorted(plan["rollouts"]):
        sink.put(f"{prefix}/rollouts/{task_id}.jsonl", plan["rollouts"][task_id])
    append_once(sink.path("accepted-all.jsonl"), raw_history, plan["accepted"])
    if sink.store is not None:
        sink.store.put("public/generator/accepted-all.jsonl", sink.read("accepted-all.jsonl"))
    sink.json("payout.json", plan["payout"])
    sink.put(f"{prefix}/intake.jsonl", intake_raw)
    digest = sha256_hex(intake_raw)
    sink.put(f"{prefix}/intake.sha256", f"{digest}  intake.jsonl\n".encode())
    old_state = dict(state_document or (read_json(sink.path("state.json")) if sink.exists("state.json") else {}))
    published = [row for row in old_state.get("published_epochs", []) if row.get("epoch") != epoch]
    published.append({"epoch": epoch, "results": sink.url(f"{prefix}/results.json"), "intake": sink.url(f"{prefix}/intake.jsonl"), "intake_sha256": digest})
    published.sort(key=lambda row: row["epoch"])
    old_state["published_epochs"] = published
    old_state["current_epoch"] = epoch_id(now)
    old_state["last_results"] = sink.url(f"{prefix}/results.json")
    old_state["last_epoch"] = epoch
    old_state["pool"] = POOL_ID
    old_state["pool_url"] = "https://github.com/AffineFoundation/affine/releases/tag/sn120-math-prior-v1"
    old_state["counts"] = plan["state_counts"]
    sink.json("state.json", old_state)
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sn120_v2.generator.publish")
    add_config(parser)
    parser.add_argument("--epoch", required=True)
    parser.add_argument("--now", type=float)
    args = parser.parse_args(argv)
    try:
        import time
        from .pool import PriorPool
        config = Config.load(args.config)
        epoch = epoch_id(args.epoch)
        job = read_json(config.work_dir / "jobs" / epoch / "evidence.json")
        output = read_json(config.work_dir / "epochs" / epoch / "out" / "results.json")
        secret = read_json(config.work_dir / "seeds" / f"{epoch}.json")["secret"]
        frozen = read_json(config.work_dir / "king" / f"{epoch}.json")
        if "captured_at_epoch" not in frozen:
            raise VerifyError("frozen king record lacks its epoch deadline timestamp")
        from ..chain import Reader, SubtensorClient
        from .common import open_store
        client = SubtensorClient.connect(120, config.network)
        try:
            snapshot = Reader(client, 120).snapshot()
        finally:
            client.chain.close()
        plan = publish(
            sink=PublicSink(config.site_dir, open_store(config.credentials_file), config.public_url),
            job=job, output=output, out_dir=config.work_dir / "epochs" / epoch / "out",
            secret=secret, registered={reg.uid: reg.hotkey for reg in snapshot.registrations},
            now=time.time() if args.now is None else args.now, pool_rows=PriorPool(config.pool_dir),
            frozen_at=float(frozen["captured_at_epoch"]),
        )
        print(f"published {epoch}: {len(plan['accepted'])} accepted tasks; intake complete")
        return 0
    except V2Error as exc:
        print(f"publish failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
