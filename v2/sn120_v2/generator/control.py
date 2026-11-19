"""Clock-driven control-plane operations shared by the minute timer and tests."""
from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path
from typing import Any, Callable, Mapping

from ..canonical import sha256_hex
from ..errors import ConfigError, VerifyError
from .common import PublicSink, atomic_json, json_bytes, read_json, utc_text
from .config import Config
from .epoch import epoch_deadline, epoch_id, epoch_start
from .links import GeneratorPresigner, link_files
from .constants import POOL_ID
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


def frozen_at(king_path: Path, epoch: str) -> float:
    row = read_json(king_path)
    if row.get("epoch") != epoch:
        raise VerifyError("frozen king epoch does not match the job")
    return float(row["captured_at_epoch"])


def read_registrations(client: Any, clock: Callable[[], float]) -> tuple[dict[int, str], set[str]]:
    from ..chain import Reader
    from ..links import read_activated
    snapshot = Reader(client, 120, clock=clock).snapshot()
    registered = {reg.uid: reg.hotkey for reg in snapshot.registrations}
    return registered, read_activated(client, registered.values(), 120)


def _write_links(sink: PublicSink, files: Mapping[str, bytes]) -> None:
    for name in sorted(files, key=lambda name: name == "links.json"):
        relative = name.removeprefix("generator/")
        if relative.startswith("links/") or relative == "links.json":
            sink.put(relative, files[name])


def issue_links(config: Config, *, now: float, client: Any, store: Any) -> dict[str, Any]:
    from .links import GeneratorPresigner, eligible, link_files
    state_path = config.work_dir / "links-state.json"
    state = read_json(state_path) if state_path.is_file() else {}
    if now - float(state.get("issued_at", 0)) < 600:
        return {"status": "not_due"}
    from .control import read_registrations
    registered, activated = read_registrations(client, lambda: now)
    hotkeys = sorted({hotkey for uid, hotkey in registered.items() if uid != 0 and hotkey in activated})
    files = link_files(hotkeys, GeneratorPresigner(store), now)
    sink = PublicSink(config.site_dir, store, config.public_url)
    _write_links(sink, files)
    epoch = epoch_id(now)
    document = {"epoch": epoch, "issued_at": now, "hotkeys": hotkeys}
    atomic_json(state_path, document)
    return {"status": "issued", "epoch": epoch, "count": len(hotkeys)}


def snapshot_epoch(config: Config, *, epoch: str, now: float, client: Any, store: Any, pool: Any) -> dict[str, Any]:
    sink = PublicSink(config.site_dir, store, config.public_url)
    manifest_rel = f"epochs/{epoch}/manifest.json"
    job_path = config.work_dir / "jobs" / epoch / "evidence.json"
    if sink.exists(manifest_rel) and job_path.is_file():
        return {"status": "already_complete", "epoch": epoch}
    if now < epoch_deadline(epoch).timestamp() + 60:
        return {"status": "not_due", "epoch": epoch}
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
    )
    atomic_json(job_path, job)
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
        }, private=True)
    sink.put(manifest_rel, json_bytes(manifest))
    update_window_state(config, epoch=epoch_id(now), now=now, store=store)
    return {"status": "snapshot", "epoch": epoch, "tasks": len(job["tasks"])}


def update_window_state(config: Config, *, epoch: str, now: float, store: Any) -> None:
    """Update only the open-window fields, preserving published_epochs."""
    sink = PublicSink(config.site_dir, store, config.public_url)
    state = read_json(sink.path("state.json")) if sink.exists("state.json") else {}
    seed = read_json(config.work_dir / "seeds" / f"{epoch}.json")
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
    if isinstance(state.get("links"), dict) and not state["links"].get("epochs"):
        state["links"]["epochs"] = [{"epoch": row["epoch"], "manifest": sink.url(f"epochs/{row['epoch']}/manifest.json")} for row in state.get("published_epochs", [])]
    sink.json("state.json", state)


def publish_epoch(config: Config, *, epoch: str, now: float, client: Any, store: Any, pool: Any) -> dict[str, Any]:
    from .publish import publish
    seed = read_json(config.work_dir / "seeds" / f"{epoch}.json")
    frozen = frozen_at(config.work_dir / "king" / f"{epoch}.json", epoch)
    job = read_json(config.work_dir / "jobs" / epoch / "evidence.json")
    out_dir = config.work_dir / "epochs" / epoch / "out"
    output = read_json(out_dir / "results.json")
    snapshot = client if isinstance(client, Mapping) else {reg.uid: reg.hotkey for reg in client}
    plan = publish(
        sink=PublicSink(config.site_dir, store, config.public_url), job=job, output=output, out_dir=out_dir,
        secret=seed["secret"], registered=snapshot, now=now, pool_rows=pool, frozen_at=frozen,
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
    run = lambda name, default, **kwargs: actions.get(name, default)(**kwargs)
    current = epoch_id(now)
    events: list[dict[str, Any]] = []
    write_seed(config.work_dir, current)
    king_path = config.work_dir / "king" / f"{current}.json"
    if not king_path.is_file() and now >= epoch_deadline(current).timestamp():
        fetch_head(config.head_url, king_path, now=now)
        row = read_json(king_path)
        row.update({"epoch": current, "captured_at_epoch": epoch_deadline(current).timestamp()})
        atomic_json(king_path, row)
    # The operator's issuer has its own 10-minute guard in links.issue.
    result = run("links", issue_links, config=config, now=now, client=client, store=store)
    if result.get("status") not in ("not_due", "unchanged"):
        events.append({"phase": "links", **result})
    update_window_state(config, epoch=current, now=now, store=store)
    for number in range(3):
        deadline = epoch_deadline(current).timestamp() - number * 4 * 3600
        epoch = epoch_id(deadline - 1)
        job_path = config.work_dir / "jobs" / epoch / "evidence.json"
        results_path = config.work_dir / "epochs" / epoch / "out" / "results.json"
        intake_path = config.site_dir / "generator" / "epochs" / epoch / "intake.sha256"
        if now < deadline + 60 or intake_path.is_file():
            continue
        if not job_path.is_file() and now <= deadline + 3 * 3600 + 45 * 60:
            events.append({"phase": "snapshot", **run("snapshot", snapshot_epoch, config=config, epoch=epoch, now=now, client=client, store=store, pool=pool)})
        if job_path.is_file() and not results_path.is_file() and now <= deadline + 3 * 3600 + 45 * 60:
            if remote is None:
                raise ConfigError("Generator ssh config is required for dispatch and collect")
            events.append({"phase": "dispatch", **run("dispatch", remote.dispatch, epoch=epoch).to_dict()})
            result = run("collect", remote.collect, epoch=epoch)
            events.append({"phase": "collect", **result.to_dict()})
        if results_path.is_file():
            if now > deadline + 3 * 3600 + 45 * 60:
                raise ConfigError(f"scoring results for {epoch} missed the results deadline")
            events.append({"phase": "publish", **run("publish", publish_epoch, config=config, epoch=epoch, now=now, client=read_registrations(client, lambda: now)[0], store=store, pool=pool)})
    return events
