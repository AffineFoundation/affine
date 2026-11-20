"""Clock-driven control-plane operations shared by the minute timer and tests."""
from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping

from ..canonical import sha256_hex
from ..errors import ConfigError, VerifyError
from ..links import public_key
from .common import PublicSink, atomic_json, json_bytes, read_json, utc_text
from .config import Config
from .epoch import epoch_deadline, epoch_id, epoch_start
from .links import GeneratorPresigner, link_files
from .constants import POOL_ID
from .cap import cap_inputs, freeze, frozen as frozen_cap, public
from .remote import ScorerRemote
from .snapshot import build_snapshot, fetch_head


def write_seed(work_dir: Path, epoch: str) -> dict[str, Any]:
    """Commit to a per-window secret at its opening; never publish the secret yet."""
    path = work_dir / "seeds" / f"{epoch}.json"
    if path.is_file():
        row = read_json(path)
        if row.get("epoch") != epoch or sha256_hex(str(row.get("secret", "")).encode()) != row.get("seed_commitment"):
            raise VerifyError(f"the existing seed record for {epoch} is invalid")
        return row
    row = {"epoch": epoch, "secret": secrets.token_hex(32), "window_open": utc_text(epoch_start(epoch))}
    row["seed_commitment"] = sha256_hex(row["secret"].encode())
    atomic_json(path, row, private=True)
    return row


KING_CAPTURE_GRACE_S = 120
RESULTS_GRACE_S = 45 * 60


def results_deadline(config: Config, epoch: str) -> float:
    """The epoch's persisted scorer/collection budget, shared by dispatch and merge."""
    job_path = config.work_dir / "jobs" / epoch / "job.json"
    if job_path.is_file():
        job = read_json(job_path)
        state_path = config.work_dir / "jobs" / epoch / "remote-state.json"
        state = read_json(state_path) if state_path.is_file() else {}
        grace = float(state.get("results_grace_s", RESULTS_GRACE_S))
        if not math.isfinite(grace) or grace < 0:
            raise ConfigError("Generator results grace must be a nonnegative finite number")
        return float(job["budget_deadline"]) + grace
    return epoch_deadline(epoch).timestamp() + 3 * 3600 + 45 * 60 + RESULTS_GRACE_S


def _within_results_window(config: Config, epoch: str, now: float) -> bool:
    return now <= results_deadline(config, epoch)


def frozen_at(king_path: Path, epoch: str) -> float:
    row = read_json(king_path)
    if row.get("epoch") != epoch:
        raise VerifyError("frozen king epoch does not match the job")
    captured = float(row["captured_at_epoch"])
    deadline = epoch_deadline(epoch).timestamp()
    if not deadline <= captured <= deadline + KING_CAPTURE_GRACE_S:
        raise VerifyError("frozen king capture is outside the deadline grace period")
    return captured


def read_registrations(client: Any, clock: Callable[[], float]) -> tuple[dict[int, str], set[str]]:
    from ..chain import Reader
    from ..links import read_activated
    snapshot = Reader(client, 120, clock=clock).snapshot()
    registered = {reg.uid: reg.hotkey for reg in snapshot.registrations}
    return registered, read_activated(client, registered.values(), 120)


def _write_links(sink: PublicSink, files: Mapping[str, bytes]) -> None:
    """Publish every per-hotkey grant before the index, mirroring in parallel."""
    grants = {
        name.removeprefix("generator/"): raw for name, raw in files.items()
        if name.removeprefix("generator/").startswith("links/")
    }
    # A slow store cannot hold up local atomic publication or other mirrors.
    with ThreadPoolExecutor(max_workers=16, thread_name_prefix="generator-links") as executor:
        futures = [executor.submit(sink.put_mirror, relative, grants[relative]) for relative in sorted(grants)]
        for future in futures:
            future.result()
    if "generator/links.json" in files:
        sink.put_mirror("links.json", files["generator/links.json"])


def issue_links(config: Config, *, now: float, client: Any, store: Any, force: bool = False) -> dict[str, Any]:
    """Refresh only grants that need it; preserve grants through epoch expiry."""
    from .links import GeneratorPresigner, PREFETCH_S, link_files
    state_path = config.work_dir / "links-state.json"
    state = read_json(state_path) if state_path.is_file() else {}
    if not force and now - float(state.get("chain_read_at", state.get("issued_at", 0))) < 600:
        return {"status": "not_due"}
    registered, activated = read_registrations(client, lambda: now)
    hotkeys = sorted({hotkey for uid, hotkey in registered.items() if uid != 0 and hotkey in activated})
    epoch = epoch_id(now)
    if epoch_deadline(epoch) < epoch_deadline(config.first_epoch):
        return {"status": "before_first_epoch", "epoch": epoch, "count": 0}
    following = epoch_id(epoch_deadline(epoch))
    inputs = cap_inputs(config.submission_cap, ssh=config.ssh)
    remaining = epoch_deadline(epoch).timestamp() - now
    prefetch = 0 < remaining <= PREFETCH_S
    sink = PublicSink(config.site_dir, store, config.public_url)
    previous = state.get("hotkeys", {})
    if isinstance(previous, dict):
        known = set(previous)
    elif isinstance(previous, list):  # Upgrade the original aggregate state safely.
        known = set(previous)
    else:
        known = set()
    sink_hotkeys = []
    for path in (config.site_dir / "generator" / "links").glob("*.json"):
        try:
            document = json.loads(path.read_bytes())
            if document.get("hotkey") == path.stem:
                sink_hotkeys.append(path.stem)
        except (OSError, ValueError):
            continue
    all_hotkeys = sorted(known | set(sink_hotkeys) | set(hotkeys))
    issued = state.get("issued", {})
    if not isinstance(issued, dict):
        issued = {}
    due = []
    for hotkey in hotkeys:
        old = issued.get(hotkey, {})
        path = sink.path(f"links/{hotkey}.json")
        current = {}
        if path.is_file():
            try:
                current = json.loads(path.read_bytes())
            except (OSError, ValueError):
                current = {}
        expected_deadline = int(epoch_deadline(epoch).timestamp())
        valid_file = (
            current.get("version") == "affine-generator-link-v1"
            and current.get("hotkey") == hotkey
            and current.get("sealed_to") == public_key(hotkey).hex()
            and current.get("epoch") == epoch
            and current.get("deadline") == expected_deadline
        )
        if (old.get("epoch") != epoch or not valid_file
                or (prefetch and "next" not in current)):
            due.append(hotkey)
    caps: dict[str, int] = {}
    if due:
        current_record = freeze(config.work_dir, epoch, inputs, len(hotkeys), now=now)
        if prefetch:
            next_record = freeze(config.work_dir, following, inputs, len(hotkeys), now=now)
        else:
            next_record = frozen_cap(config.work_dir, following)
        caps = {record["epoch"]: record["cap"] for record in (current_record, next_record) if record is not None}
    else:
        records = [frozen_cap(config.work_dir, target) for target in (epoch, following)]
        caps = {record["epoch"]: record["cap"] for record in records if record is not None}
    files = link_files(due, GeneratorPresigner(store), now, caps=caps) if due else {}
    index = {
        "version": "affine-generator-link-v1", "issued_at": int(now),
        "epoch": epoch, "deadline": int(epoch_deadline(epoch).timestamp()),
        "hotkeys": all_hotkeys, "link_file": "generator/links/<hotkey>.json",
        "prefetch_before_deadline_seconds": PREFETCH_S,
        "upload_mode": "overwrite_until_deadline",
        **({"max_tasks": caps[epoch]} if epoch in caps else {}),
        **({"next_epoch": following, "next_max_tasks": caps[following]} if prefetch and following in caps else {}),
    }
    # Keep the index synchronized even if no grants need replacement.
    files["generator/links.json"] = json_bytes(index)
    _write_links(sink, files)
    for hotkey in due:
        issued[hotkey] = {"epoch": epoch, "issued_at": now, "prefetch": prefetch}
    document = {
        "epoch": epoch, "chain_read_at": now, "hotkeys": all_hotkeys,
        "issued": issued,
    }
    atomic_json(state_path, document, private=True)
    return {"status": "issued" if due else "unchanged", "epoch": epoch, "count": len(due), "active": len(hotkeys)}


def void_missed_king(config: Config, *, epoch: str, now: float, store: Any) -> dict[str, Any]:
    """Finalize an unscoreable epoch with no claims, payout or retry loop."""
    sink = PublicSink(config.site_dir, store, config.public_url)
    missed_path = config.work_dir / "king" / f"{epoch}.missed.json"
    missed = {
        "epoch": epoch,
        "deadline": utc_text(epoch_deadline(epoch)),
        "observed_at": utc_text(now),
    }
    atomic_json(missed_path, missed, private=True)
    prefix = f"epochs/{epoch}"
    digest = sha256_hex(b"")
    sink.json(f"{prefix}/manifest.json", {
        "version": "affine-generator-manifest-v1", "epoch": epoch,
        "deadline": utc_text(epoch_deadline(epoch)),
        "status": "void_missed_king_capture", "pool": POOL_ID,
        "files": [], "unique_tasks": 0,
    })
    sink.put(f"{prefix}/intake.jsonl", b"")
    sink.put(f"{prefix}/intake.sha256", f"{digest}  intake.jsonl\n".encode())
    state = read_json(sink.path("state.json")) if sink.exists("state.json") else {}
    published = [row for row in state.get("published_epochs", []) if row.get("epoch") != epoch]
    published.append({
        "epoch": epoch, "status": "void",
        "manifest": sink.url(f"{prefix}/manifest.json"),
        "intake": sink.url(f"{prefix}/intake.jsonl"), "intake_sha256": digest,
    })
    published.sort(key=lambda row: row["epoch"])
    state["published_epochs"] = published
    sink.json("state.json", state)
    return {"status": "void_missed_king_capture", "epoch": epoch}


def snapshot_epoch(config: Config, *, epoch: str, now: float, client: Any, store: Any, pool: Any) -> dict[str, Any]:
    sink = PublicSink(config.site_dir, store, config.public_url)
    if epoch_deadline(epoch) < epoch_deadline(config.first_epoch):
        return {"status": "before_first_epoch", "epoch": epoch}
    manifest_rel = f"epochs/{epoch}/manifest.json"
    job_path = config.work_dir / "jobs" / epoch / "evidence.json"
    if sink.exists(manifest_rel) and job_path.is_file():
        return {"status": "already_complete", "epoch": epoch}
    if (config.work_dir / "king" / f"{epoch}.missed.json").is_file():
        return {"status": "void_missed_king_capture", "epoch": epoch}
    if now < epoch_deadline(epoch).timestamp() + 60:
        return {"status": "not_due", "epoch": epoch}
    if pool is None:
        try:
            from .pool import PriorPool
            pool = PriorPool(config.pool_dir)
        except ConfigError as exc:
            return {"status": "error", "epoch": epoch, "error": f"{type(exc).__name__}: {exc}"}
    king = read_json(config.work_dir / "king" / f"{epoch}.json")
    frozen = frozen_at(config.work_dir / "king" / f"{epoch}.json", epoch)
    seed = read_json(config.work_dir / "seeds" / f"{epoch}.json")
    registered, activated = read_registrations(client, lambda: now)
    from .snapshot import load_history
    _, previous = load_history(sink)
    manifest, job = build_snapshot(
        store, epoch=epoch, registered=registered, activations=activated,
        known_task_ids=lambda task_id: task_id in pool, previously_accepted=previous,
        king=king["version"], seed_commitment=seed["seed_commitment"],
        submission_cap=frozen_cap(config.work_dir, epoch),
    )
    atomic_json(job_path, job)
    manifest["cap_rejections"] = [{"hotkey": row["hotkey"], "reason": row["reason"]} for row in manifest["files"] if row.get("cap_rejected")]
    atomic_json(config.work_dir / "jobs" / epoch / "manifest.json", manifest, private=True)
    sink.put(manifest_rel, json_bytes(manifest))
    update_window_state(config, epoch=epoch_id(now), now=now, store=store)
    if job["tasks"]:
        from .snapshot import scorer_job
        atomic_json(job_path.with_name("job.json"), scorer_job(job, seed["secret"]), private=True)
    else:
        out_dir = config.work_dir / "epochs" / epoch / "out"
        atomic_json(out_dir / "results.json", {
            "version": "affine-generator-results-v1", "epoch": epoch, "head_version": job["king"],
            "pool": POOL_ID,
            "pool_pin": POOL_ID,
            "seed_secret": seed["secret"], "seed_commitment": seed["seed_commitment"], "tasks": [], "hotkeys": {},
            **({"submission_cap": public(job["submission_cap"])} if "submission_cap" in job else {}),
        }, private=True)
    return {"status": "snapshot", "epoch": epoch, "tasks": len(job["tasks"])}


def update_window_state(config: Config, *, epoch: str, now: float, store: Any) -> None:
    """Update only the open-window fields, preserving published_epochs."""
    sink = PublicSink(config.site_dir, store, config.public_url)
    state = read_json(sink.path("state.json")) if sink.exists("state.json") else {}
    seed = read_json(config.work_dir / "seeds" / f"{epoch}.json")
    closed = epoch_id(epoch_start(epoch).timestamp() - 1)
    closed_king = config.work_dir / "king" / f"{closed}.json"
    closed_job = config.work_dir / "jobs" / closed / "evidence.json"
    manifest_rel = f"epochs/{closed}/manifest.json"
    manifest_exists = sink.exists(manifest_rel)
    published = next((row for row in state.get("published_epochs", []) if row.get("epoch") == closed), None)
    if (config.work_dir / "king" / f"{closed}.missed.json").is_file():
        scoring_status = "void"
    elif published:
        scoring_status = "published"
    elif closed_job.is_file() or manifest_exists:
        scoring_status = "scoring"
    else:
        scoring_status = "snapshot"
    scoring = {
        "epoch": closed,
        "deadline": utc_text(epoch_deadline(closed)),
        "status": scoring_status,
        "manifest": sink.url(manifest_rel) if manifest_exists else None,
    }
    if closed_king.is_file():
        king = read_json(closed_king)
        scoring["frozen_king_version"] = king["version"]
        scoring["frozen_king_at"] = king["captured_at"]
    state["scoring"] = scoring
    king_path = config.work_dir / "king" / f"{epoch}.json"
    state.update({
        "version": "affine-generator-state-v1", "current_epoch": epoch,
        "window_open": utc_text(epoch_start(epoch)), "window_close": utc_text(epoch_deadline(epoch)),
        "deadline": utc_text(epoch_deadline(epoch)), "seed_commitment": seed["seed_commitment"],
        "links": {"index": sink.url("links.json"), "epoch": epoch},
        "pool": "sn120-math-prior-v1:" + "45b793b4d00d1f7225d34fd35e58018612a0ded00d96a9c23b750cfb18994115",
        "pool_url": "https://github.com/AffineFoundation/affine/releases/tag/sn120-math-prior-v1",
    })
    if king_path.is_file():
        king = read_json(king_path)
        state["frozen_king_version"] = king["version"]
        state["frozen_king_at"] = king["captured_at"]
    if isinstance(state.get("links"), dict):
        state["links"]["epochs"] = [
            {"epoch": path.parent.name, "manifest": sink.url(f"epochs/{path.parent.name}/manifest.json")}
            for path in sorted((config.site_dir / "generator" / "epochs").glob("*/manifest.json"))
        ]
    state.pop("submission_cap", None)
    state.pop("next_submission_cap", None)
    current_cap = frozen_cap(config.work_dir, epoch)
    if current_cap is not None:
        state["submission_cap"] = public(current_cap)
    next_cap = frozen_cap(config.work_dir, epoch_id(epoch_deadline(epoch)))
    if next_cap is not None:
        state["next_submission_cap"] = public(next_cap)
    sink.json("state.json", state)


def publish_epoch(config: Config, *, epoch: str, now: float, client: Any, store: Any, pool: Any) -> dict[str, Any]:
    if epoch_deadline(epoch) < epoch_deadline(config.first_epoch):
        return {"status": "before_first_epoch", "epoch": epoch, "accepted": 0}
    from .publish import publish
    seed = read_json(config.work_dir / "seeds" / f"{epoch}.json")
    frozen = frozen_at(config.work_dir / "king" / f"{epoch}.json", epoch)
    job = read_json(config.work_dir / "jobs" / epoch / "evidence.json")
    out_dir = config.work_dir / "epochs" / epoch / "out"
    output = read_json(out_dir / "results.json")
    snapshot = client if isinstance(client, Mapping) else {reg.uid: reg.hotkey for reg in client}
    cap_rejections = []
    if frozen_cap(config.work_dir, epoch) is not None:
        cap_rejections = [row for row in read_json(config.work_dir / "jobs" / epoch / "manifest.json").get("files", []) if row.get("cap_rejected")]
    plan = publish(
        sink=PublicSink(config.site_dir, store, config.public_url), job=job, output=output, out_dir=out_dir,
        secret=seed["secret"], registered=snapshot, now=now, pool_rows=pool, frozen_at=frozen,
        cap_rejections=cap_rejections,
    )
    # Rebuilding state here retains the open window and the previous ledger.
    update_window_state(config, epoch=epoch_id(now), now=now, store=store)
    return {"status": "published", "epoch": epoch, "accepted": len(plan["accepted"])}


def tick(
    config: Config, *, now: float, client: Any, store: Any, pool: Any, remote: ScorerRemote | None = None,
    steps: Mapping[str, Callable[..., Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run every due phase once. A failed phase does not publish later evidence."""
    actions = steps or {}
    failed_epochs = set()
    def run(name, default, **kwargs):
        phase = kwargs.get("epoch")
        try:
            return actions.get(name, default)(**kwargs)
        except Exception as exc:
            # A phase fault is observable but must not strand later epochs.
            if phase is not None:
                failed_epochs.add(phase)
            return {"status": "error", "phase": name, "epoch": phase,
                    "error": f"{type(exc).__name__}: {exc}" if isinstance(exc, ConfigError) else type(exc).__name__}
    current = epoch_id(now)
    events: list[dict[str, Any]] = []
    planned = max(current, config.first_epoch)
    write_seed(config.work_dir, planned)
    # Freeze only the just-closed epoch, and only during the explicit grace.
    closed = epoch_id(int(now // 14400) * 14400 - 1)
    king_path = config.work_dir / "king" / f"{closed}.json"
    if (closed >= config.first_epoch and not king_path.is_file()
            and not (config.work_dir / "king" / f"{closed}.missed.json").is_file()):
        elapsed = now - epoch_deadline(closed).timestamp()
        if 0 <= elapsed <= KING_CAPTURE_GRACE_S and (config.work_dir / "seeds" / f"{closed}.json").is_file():
            fetch_head(config.head_url, king_path, now=now)
            row = read_json(king_path)
            row.update({"epoch": closed, "captured_at_epoch": now})
            atomic_json(king_path, row)
        elif elapsed > KING_CAPTURE_GRACE_S:
            events.append({"phase": "king_capture", **run("void", void_missed_king, config=config, epoch=closed, now=now, store=store)})
    # The deadline snapshot and scorer phases must not wait for link issuance.
    for number in range(4):
        epoch = epoch_id(epoch_deadline(current).timestamp() - number * 4 * 3600 - 1)
        deadline = epoch_deadline(epoch).timestamp()
        job_path = config.work_dir / "jobs" / epoch / "evidence.json"
        results_path = config.work_dir / "epochs" / epoch / "out" / "results.json"
        intake_path = config.site_dir / "generator" / "epochs" / epoch / "intake.sha256"
        if now > results_deadline(config, epoch) and not results_path.is_file():
            continue
        if (now < deadline + 60 or intake_path.is_file()
                or epoch < config.first_epoch
                or not (config.work_dir / "seeds" / f"{epoch}.json").is_file()):
            continue
        if (config.work_dir / "king" / f"{epoch}.missed.json").is_file():
            events.append({"phase": "void", **run("void", void_missed_king, config=config, epoch=epoch, now=now, store=store)})
            continue
        if not job_path.is_file() and now <= deadline + 3 * 3600 + 45 * 60:
            result = run("snapshot", snapshot_epoch, config=config, epoch=epoch, now=now, client=client, store=store, pool=pool)
            events.append({"phase": "snapshot", **result})
            if result.get("status") == "error":
                continue
        if epoch in failed_epochs:
            continue
        if job_path.is_file() and not results_path.is_file() and _within_results_window(config, epoch, now):
            if remote is None:
                failed_epochs.add(epoch)
                events.append({"phase": "dispatch", "epoch": epoch, "status": "error", "error": "ConfigError: Generator ssh config is required for dispatch and collect"})
                continue
            dispatched = run("dispatch", remote.dispatch, epoch=epoch)
            events.append({"phase": "dispatch", **(dispatched.to_dict() if not isinstance(dispatched, Mapping) else dispatched)})
            if epoch in failed_epochs:
                continue
            collected = run("collect", remote.collect, epoch=epoch)
            events.append({"phase": "collect", **(collected.to_dict() if not isinstance(collected, Mapping) else collected)})
            if epoch in failed_epochs:
                continue
        if results_path.is_file() and epoch not in failed_epochs:
            if not _within_results_window(config, epoch, now):
                error = f"scoring results for {epoch} missed the results deadline"
                failed_epochs.add(epoch)
                events.append({"phase": "publish", "epoch": epoch, "status": "error", "error": f"ConfigError: {error}"})
                continue
            events.append({"phase": "publish", **run("publish", publish_epoch, config=config, epoch=epoch, now=now, client=read_registrations(client, lambda: now)[0], store=store, pool=pool)})
    result = run("links", issue_links, config=config, now=now, client=client, store=store)
    if result.get("status") != "not_due":
        events.append({"phase": "links", **result})
    run("window_state", update_window_state, config=config, epoch=planned, now=now, store=store)
    return events
