"""Continuous datagen service, pipelined:

  main thread   : select batch -> prefetch next batch's images (background)
                  -> shard across the provider pool -> mini-swe-agent rollouts
  eval thread   : swebench eval (telemetry) + slice turns from all trajs
                  trajectories -> slice per-turn records -> upload shards

Generation is API-bound and eval is docker/CPU-bound, so batch N evaluates
while batch N+1 generates. Restart-safe: processed instance ids live in
state.jsonl (written only after eval), sliced-but-not-uploaded turns survive
in pending_turns.jsonl / outbox/, and a crash mid-pipeline just means the
un-marked instances are re-selected and regenerated.

  python -m datagen.loop            # the service (supervised by bootstrap.sh)
  python -m datagen.loop --once     # one batch through eval, then exit
  python -m datagen.loop --dry-run  # select + materialize + config + planned
                                    # provider assignment; stops before any
                                    # model API call
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

from .agentrun import (
    agent_cmd,
    classify_traj,
    merge_predictions,
    parse_resolved_ids,
    prefetch_images,
    prune_images,
    reap_containers_for_images,
    reap_stale_containers,
    row_image,
    run_agent_phase,
    run_eval_phase,
    traj_usage,
    write_agent_config,
)
from .config import DatagenConfig, load_config
from .instances import ProcessedState, load_pool, materialize_subset, select_batch
from .providers import Provider, ProviderPool, load_providers
from .slicer import slice_traj_file
from .uploader import TurnUploader, shard_name, shard_sha256

log = logging.getLogger("datagen.loop")

POOL_EXHAUSTED_SLEEP_S = 6 * 3600
MAX_ASSIGN_WAVES = 2
EVAL_RETRIES = 2


def _utc_tag() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _append_turns(pending: Path, records: list[dict]) -> None:
    pending.parent.mkdir(parents=True, exist_ok=True)
    with open(pending, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


# -- generation (main thread) ------------------------------------------------

def _run_provider_group(cfg: DatagenConfig, provider: Provider,
                        rows: list[dict], run_dir: Path,
                        ) -> dict[str, tuple[str, str]]:
    """One provider's share of a wave. Returns iid -> (kind, detail) from
    classify_traj; a group-level failure marks every instance "provider"."""
    t0 = time.time()
    gdir = run_dir / provider.name
    subset_dir = materialize_subset(rows, gdir / "subset")
    agent_cfg = gdir / "agent.yaml"
    write_agent_config(cfg, provider, agent_cfg)
    preds_dir = gdir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    code, out = run_agent_phase(cfg, provider, subset_dir, agent_cfg,
                                preds_dir, gdir, len(rows))
    if code == -2:
        # Killing the process group orphans its containers (observed live);
        # reap only this group's, by instance image.
        reap_containers_for_images([row_image(r) for r in rows])
        # Salvage: a single hung straggler must not void its groupmates'
        # finished rollouts (observed live: 10/11 engy trajs complete at
        # timeout, all thrown away and re-billed on another lane). Completed
        # trajectories classify normally; only the stragglers requeue.
        results = {}
        for r in rows:
            iid = r["instance_id"]
            kind, detail = classify_traj(preds_dir / iid / f"{iid}.traj.json")
            if kind == "provider":
                detail = f"agent timeout ({detail})"
            results[iid] = (kind, detail)
        n_salvaged = sum(1 for kind, _ in results.values()
                         if kind == "attempted")
        log.warning("provider %s group timed out after %.0fs; salvaged %d/%d "
                    "finished trajectories", provider.name, time.time() - t0,
                    n_salvaged, len(rows))
        return results
    if code != 0:
        detail = f"agent exit {code}"
        log.error("provider %s group failed after %.0fs (%s): %s",
                  provider.name, time.time() - t0, detail, (out or "")[-1500:])
        return {r["instance_id"]: ("provider", detail) for r in rows}
    results = {r["instance_id"]:
               classify_traj(preds_dir / r["instance_id"]
                             / f"{r['instance_id']}.traj.json")
               for r in rows}
    n_att = sum(1 for kind, _ in results.values() if kind == "attempted")
    log.info("provider %s: %d/%d attempted in %.0fs", provider.name, n_att,
             len(rows), time.time() - t0)
    return results


def _run_waves(cfg: DatagenConfig, pool: ProviderPool, batch: list[dict],
               run_dir: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Shard the batch across providers; requeue provider-failures once on a
    different provider. Returns (attempted: iid -> {provider, preds_dir},
    errors: iid -> detail)."""
    rows_by_id = {r["instance_id"]: r for r in batch}
    attempted: dict[str, dict] = {}
    errors: dict[str, str] = {}
    queue_ids = [r["instance_id"] for r in batch]
    banned: dict[str, set[str]] = {}

    for wave in range(MAX_ASSIGN_WAVES):
        if not queue_ids:
            break
        groups, unassigned = pool.assign(queue_ids, banned)
        for iid in unassigned:
            errors[iid] = "no provider available"
        if not groups:
            break
        log.info("wave %d assignment: %s", wave + 1,
                 {name: len(ids) for name, ids in groups.items()})
        with concurrent.futures.ThreadPoolExecutor(len(groups)) as ex:
            futures = {
                ex.submit(_run_provider_group, cfg, pool.by_name[name],
                          [rows_by_id[i] for i in ids], run_dir / f"wave{wave}")
                : name
                for name, ids in groups.items()
            }
            results: dict[str, dict[str, tuple[str, str]]] = {}
            for fut in concurrent.futures.as_completed(futures):
                results[futures[fut]] = fut.result()

        queue_ids = []
        for name, group_results in results.items():
            provider = pool.by_name[name]
            fails = [iid for iid, (kind, _) in group_results.items()
                     if kind == "provider"]
            if fails:
                pool.cooldown(name, group_results[fails[0]][1])
            else:
                pool.mark_ok(name)
            for iid, (kind, detail) in group_results.items():
                if kind == "attempted":
                    attempted[iid] = {
                        "provider": provider.label,
                        "preds_dir": run_dir / f"wave{wave}" / name / "preds",
                        "detail": detail,
                    }
                elif kind == "provider":
                    banned.setdefault(iid, set()).add(name)
                    queue_ids.append(iid)
                else:
                    errors[iid] = detail

    for iid in queue_ids:  # provider-failed on every wave
        errors[iid] = "provider failure after requeue"
    return attempted, errors


# -- eval + finalize (worker thread) ------------------------------------------

def _eval_batch(cfg: DatagenConfig, batch: list[dict], eval_ids: set[str],
                preds_jsonl: Path, run_dir: Path) -> set[str] | None:
    """swebench eval, grouped by source dataset, retried per group. Returns
    the union of resolved ids, or None on a persistent eval failure."""
    by_source: dict[tuple[str, str], list[str]] = {}
    for row in batch:
        iid = row["instance_id"]
        if iid in eval_ids:
            key = (row["_source_dataset"], row["_source_split"])
            by_source.setdefault(key, []).append(iid)

    resolved: set[str] = set()
    for (dataset, split), ids in sorted(by_source.items()):
        got = None
        for attempt in range(EVAL_RETRIES):
            run_id = f"datagen-{int(time.time())}"
            code, out = run_eval_phase(cfg, dataset, split, preds_jsonl, ids,
                                       run_id, run_dir)
            if code == 0:
                got = parse_resolved_ids(run_dir, run_id)
                if got is not None:
                    break
                log.error("eval finished for %s[%s] but no resolved_ids "
                          "report (attempt %d)", dataset, split, attempt + 1)
            else:
                log.error("eval failed for %s[%s] (exit %s, attempt %d): %s",
                          dataset, split, code, attempt + 1,
                          (out or "")[-2000:])
            time.sleep(30)
        if got is None:
            return None
        resolved |= set(got)
    return resolved


def _finalize_batch(cfg: DatagenConfig, job: dict, state: ProcessedState,
                    pending: Path) -> None:
    """Eval + slice + mark for one generated batch. Runs on the eval thread
    while the next batch generates. Eval hard-failure marks the attempted
    instances as error — regenerating them would double the API spend."""
    batch, attempted, run_dir = job["batch"], job["attempted"], job["run_dir"]
    t0 = time.time()
    preds_dirs = sorted({info["preds_dir"] for info in attempted.values()})
    preds_jsonl = merge_predictions(preds_dirs, run_dir / "predictions.jsonl")

    with_patch: set[str] = set()
    if preds_jsonl is not None:
        with open(preds_jsonl, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("model_patch"):
                    with_patch.add(row["instance_id"])
    # No submitted patch -> cannot resolve; skip the docker eval entirely.
    no_patch = set(attempted) - with_patch
    eval_ids = set(attempted) & with_patch

    resolved: set[str] = set()
    if eval_ids:
        got = _eval_batch(cfg, batch, eval_ids, preds_jsonl, run_dir)
        if got is None:
            for iid, info in attempted.items():
                state.mark(iid, "error", detail="eval failure",
                           provider=info["provider"])
            shutil.rmtree(run_dir, ignore_errors=True)
            return
        resolved = got

    traj_archive = cfg.data_dir / "trajs"
    traj_archive.mkdir(parents=True, exist_ok=True)
    n_turns_total = 0
    # Reason v3: D is teacher-continuable prefixes. Keep turns from any
    # trajectory that slices cleanly — resolved is telemetry, not a gate.
    for iid, info in attempted.items():
        traj_path = info["preds_dir"] / iid / f"{iid}.traj.json"
        usage = traj_usage(traj_path)
        records: list[dict] = []
        if traj_path.is_file():
            records = slice_traj_file(traj_path, model_label=info["provider"])
            if records:
                _append_turns(pending, records)
                shutil.copy2(traj_path, traj_archive / traj_path.name)
                n_turns_total += len(records)
        if iid in resolved:
            outcome = "resolved"
            detail = info["detail"]
        elif iid in no_patch:
            outcome = "unresolved"
            detail = f"no patch ({info['detail']})"
        else:
            outcome = "unresolved"
            detail = info["detail"]
        state.mark(iid, outcome, n_turns=len(records),
                   provider=info["provider"], detail=detail, **usage)

    log.info("eval done in %.0fs: %d/%d resolved (%d attempted, %d with "
             "patch), %d turns sliced", time.time() - t0, len(resolved),
             len(batch), len(attempted), len(eval_ids), n_turns_total)
    shutil.rmtree(run_dir, ignore_errors=True)
    if cfg.prune_images:
        prune_images([row_image(r) for r in batch])


def _eval_worker(cfg: DatagenConfig, state: ProcessedState, pending: Path,
                 uploader: TurnUploader, jobs: queue.Queue,
                 inflight: set[str], lock: threading.Lock) -> None:
    while True:
        job = jobs.get()
        if job is None:
            jobs.task_done()
            return
        try:
            _finalize_batch(cfg, job, state, pending)
            flush_uploads(cfg, pending, uploader)
        except Exception:
            log.exception("eval worker failed for %s (instances stay "
                          "unmarked and will be re-selected)", job["run_dir"])
        finally:
            with lock:
                inflight.difference_update(
                    r["instance_id"] for r in job["batch"])
            jobs.task_done()


# -- upload -------------------------------------------------------------------

def flush_uploads(cfg: DatagenConfig, pending: Path,
                  uploader: TurnUploader) -> None:
    """Cut pending turns into a shard when big or old enough, then push every
    queued shard. The pending->outbox rename is atomic, so a crash or upload
    failure never loses or duplicates turns."""
    outbox = cfg.data_dir / "outbox"
    cut_state = cfg.data_dir / "upload_state.json"
    last_cut = 0.0
    if cut_state.exists():
        last_cut = float(json.loads(cut_state.read_text()).get("last_cut", 0))

    n_pending = _count_lines(pending)
    if n_pending and (n_pending >= cfg.upload_min_turns
                      or time.time() - last_cut >= cfg.upload_interval_s):
        outbox.mkdir(parents=True, exist_ok=True)
        dest = outbox / shard_name(shard_sha256(pending))
        pending.rename(dest)
        cut_state.write_text(json.dumps({"last_cut": time.time()}))
        log.info("cut shard %s (%d turns)", dest.name, n_pending)

    shards = sorted(outbox.glob("*.jsonl")) if outbox.is_dir() else []
    for shard in shards:
        try:
            uploader.upload_shard(shard)
        except Exception:
            log.warning("upload of %s failed; will retry next cycle",
                        shard.name, exc_info=True)
            break
        shard.unlink()


# -- main ---------------------------------------------------------------------

def _dry_run_batch(cfg: DatagenConfig, pool: ProviderPool,
                   batch: list[dict], run_dir: Path) -> None:
    ids = [r["instance_id"] for r in batch]
    groups, unassigned = pool.assign(ids)
    plan = {iid: name for name, gids in groups.items() for iid in gids}
    for iid in ids:
        provider = pool.by_name.get(plan.get(iid, ""))
        log.info("dry-run: %-45s -> %s", iid,
                 provider.label if provider else "UNASSIGNED")
    for name, gids in sorted(groups.items()):
        provider = pool.by_name[name]
        gdir = run_dir / "wave0" / name
        subset_dir = materialize_subset(
            [r for r in batch if r["instance_id"] in set(gids)],
            gdir / "subset")
        agent_cfg = gdir / "agent.yaml"
        write_agent_config(cfg, provider, agent_cfg)
        log.info("dry-run: %s agent cmd: %s", name,
                 " ".join(agent_cmd(provider, subset_dir, agent_cfg,
                                    gdir / "preds", len(gids))))
    if unassigned:
        log.warning("dry-run: %d instances unassignable", len(unassigned))
    log.info("dry-run: stopping before any model API call")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="stop before any model API call or upload")
    ap.add_argument("--once", action="store_true",
                    help="process a single batch through eval, then exit")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config()
    providers = load_providers()
    if not providers:
        sys.exit("no inference provider has its API key set (fail-closed); "
                 "set at least one of the key_env vars from DATAGEN_PROVIDERS")
    if not args.dry_run and not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN missing (fail-closed: sliced turns could never "
                 "be uploaded)")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    # runs/ is scratch: anything in it belongs to un-marked (re-selectable)
    # instances from a previous life of the process.
    shutil.rmtree(cfg.data_dir / "runs", ignore_errors=True)
    ppool = ProviderPool(providers)
    log.info("datagen starting: providers=%s datasets=%s batch=%d "
             "step_limit=%d hf_repo=%s data_dir=%s dry_run=%s",
             [f"{p.label}(x{p.concurrency})" for p in providers],
             [f"{d}[{s}]" for d, s in cfg.datasets], cfg.batch_size,
             cfg.step_limit, cfg.hf_repo, cfg.data_dir, args.dry_run)

    state = ProcessedState(cfg.data_dir / "state.jsonl")
    pending = cfg.data_dir / "pending_turns.jsonl"
    uploader = TurnUploader(cfg.hf_repo, cfg.hf_private)
    pool = load_pool(cfg.datasets)

    inflight: set[str] = set()
    lock = threading.Lock()
    jobs: queue.Queue = queue.Queue(maxsize=1)
    worker = None
    if not args.dry_run:
        worker = threading.Thread(
            target=_eval_worker, name="datagen-eval",
            args=(cfg, state, pending, uploader, jobs, inflight, lock),
            daemon=True)
        worker.start()

    batch_fails = 0
    prefetcher: threading.Thread | None = None
    while True:
        with lock:
            busy = state.done | inflight
        batch = select_batch(pool, busy, cfg.batch_size, cfg.seed)
        if not batch:
            log.info("pool exhausted (%d processed); reloading in %ds",
                     len(state.done), POOL_EXHAUSTED_SLEEP_S)
            if args.once or args.dry_run:
                break
            jobs.join()
            flush_uploads(cfg, pending, uploader)
            time.sleep(POOL_EXHAUSTED_SLEEP_S)
            pool = load_pool(cfg.datasets)
            continue

        ids = [r["instance_id"] for r in batch]
        run_dir = cfg.data_dir / "runs" / f"batch-{_utc_tag()}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if args.dry_run:
            _dry_run_batch(cfg, ppool, batch, run_dir)
            break

        # Prefetch the NEXT batch's images while this one generates.
        nxt = select_batch(pool, busy | set(ids), cfg.batch_size, cfg.seed)
        if nxt and (prefetcher is None or not prefetcher.is_alive()):
            prefetcher = threading.Thread(target=prefetch_images, args=(nxt,),
                                          name="datagen-prefetch", daemon=True)
            prefetcher.start()

        reap_stale_containers()
        t0 = time.time()
        attempted, errors = _run_waves(cfg, ppool, batch, run_dir)
        log.info("generation done in %.0fs: %d attempted, %d errors of %d",
                 time.time() - t0, len(attempted), len(errors), len(ids))

        if not attempted:
            batch_fails += 1
            if batch_fails >= cfg.max_batch_retries:
                log.error("generation failed %d times; marking %d instances "
                          "as error and moving on", batch_fails, len(ids))
                for iid in ids:
                    state.mark(iid, "error", detail="batch-level failure")
                batch_fails = 0
            else:
                log.warning("no instance attempted; retrying batch (%d/%d)",
                            batch_fails, cfg.max_batch_retries)
                time.sleep(60)
            shutil.rmtree(run_dir, ignore_errors=True)
            continue

        batch_fails = 0
        for iid, detail in errors.items():
            state.mark(iid, "error", detail=detail)
        with lock:
            inflight.update(attempted)
        # Blocks while the previous batch is still evaluating (queue depth 1)
        # — generation never runs more than one batch ahead.
        jobs.put({"batch": batch, "attempted": attempted, "run_dir": run_dir})

        if args.once:
            break

    if worker is not None:
        jobs.join()
        jobs.put(None)
        worker.join(timeout=60)


if __name__ == "__main__":
    main()
