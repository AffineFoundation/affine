"""Attempt pre-pass: roll every task of ONE source a fixed number of times per
seat, publishing into the LIVE trace prefix under the source's own policy ids.

    python -m rollouts.prepass --source affine_science \
        --policy teacher_boxed:3 --policy king_boxed:2 --budget-usd 130

Why (aa-gap-fill-plan item 1, 2026-09-22): the fold's `[band_filter.<source>]`
keeps a task only inside the signal band — teacher solved >= k of n attempts,
king seat solved <= m of >= 2 attempts — and counts the attempts from the
traces. The live supervisor rolls each task ONCE per seat (the scheduler's
"done" set), so a band that needs n = 3 teacher attempts never fills by
itself. This driver walks the catalog pass by pass (every task once, then
twice, ...) until each task has `--policy <id>:<attempts>` graded rollouts
per policy in its OWN state; envelopes carry the plain policy id
(`teacher_boxed`, `king_boxed`), so the fold's count sees them like any
supervisor rollout. Attempts published before it started are not
subtracted (a task may end with n + 1).

Runs next to the supervisor on a datagen pod (own data dir, `ROLLOUTS_DATA_DIR`;
the R2 manifest merge is a union, so two publishers on `traces/` are fine).
Refuses `traces-backfill/` — these rows are meant for the fold. `--shard i/n`
splits the pool across pods. `--budget-usd` sums `cost_usd` over the
teacher-side rows (Engy invoice rates from policies.toml [pricing]) and stops
the paid seat at the cap; king rows cost nothing (our own box).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from affine.toolbake import ToolBaker

from rollouts.backfill import GRADED, Worker
from rollouts.config import RolloutsConfig, load_config
from rollouts.king import refresh_king_env
from rollouts.panel import panel_keys
from rollouts.r2mirror import R2TraceMirror
from rollouts.registry import KING_POLICY_PREFIX, load_registry
from rollouts.run import load_catalog_or_empty, ordered_rows, process_batch
from rollouts.runners.base import EndpointHealth

log = logging.getLogger("rollouts.prepass")

BACKFILL_PREFIX = "traces-backfill/"
PREFLIGHT_SLEEP_S = 60
FAIL_SLEEP_S = 300
MAX_CONSECUTIVE_FAILS = 3


def attempt_counts(state_paths: list[Path], source: str, policy_ids: set[str]) -> dict[str, Counter]:
    """policy_id -> Counter(uid -> graded attempts) over every state file.
    Errored rows without a model call are infrastructure, not attempts; an
    errored row WITH calls counts (the model had its try)."""
    out: dict[str, Counter] = {pid: Counter() for pid in policy_ids}
    for p in state_paths:
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("source") != source or rec.get("policy_id") not in policy_ids:
                    continue
                if rec.get("outcome") == "error" and not rec.get("n_calls"):
                    continue
                out[rec["policy_id"]][rec["uid"]] += 1
    return out


def spend_usd(state_paths: list[Path], source: str, policy_ids: set[str]) -> float:
    total = 0.0
    for p in state_paths:
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("source") == source and rec.get("policy_id") in policy_ids:
                    total += float(rec.get("cost_usd") or 0.0)
    return total


def in_shard(uid: str, shard: tuple[int, int]) -> bool:
    i, n = shard
    return n <= 1 or int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16) % n == i


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--policy", action="append", required=True, metavar="ID:N",
                    help="policy id and graded attempts per task (repeatable), e.g. teacher_boxed:3")
    ap.add_argument("--budget-usd", type=float, default=0.0,
                    help="stop the teacher-side (paid) policies once their summed cost_usd reaches this (0 = no cap)")
    ap.add_argument("--shard", default="0/1", help="i/n: roll only the tasks hashed into shard i (split across pods)")
    ap.add_argument("--parallel", type=int, default=2, help="worker threads (batches in flight)")
    ap.add_argument("--worker-batch", type=int, default=0, help="rollouts per batch (default: source.max_batch or ROLLOUTS_BATCH_SIZE)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg: RolloutsConfig = load_config()
    if cfg.r2_prefix == BACKFILL_PREFIX:
        sys.exit(f"ROLLOUTS_R2_PREFIX is {BACKFILL_PREFIX!r}: a pre-pass must publish to the live prefix (the fold reads it)")
    if not (cfg.r2_endpoint and cfg.r2_access_key_id and cfg.r2_secret_access_key):
        sys.exit("ROLLOUTS_R2_* missing (fail-closed: traces could never publish)")
    registry = load_registry()
    if args.source not in registry.sources:
        sys.exit(f"unknown source {args.source!r}")
    source = registry.sources[args.source]
    targets: dict[str, int] = {}
    for spec in args.policy:
        pid, _, n = spec.partition(":")
        if pid not in registry.policies:
            sys.exit(f"unknown policy {pid!r}")
        if pid not in source.policies:
            sys.exit(f"policy {pid!r} is not on source {args.source} (policies: {list(source.policies)})")
        targets[pid] = int(n or 1)
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    paid = {pid for pid in targets if not pid.startswith(KING_POLICY_PREFIX)}

    env = dict(os.environ)
    refresh_king_env(env)
    pool = [r for r in ordered_rows(cfg, args.source, load_catalog_or_empty(cfg, source))
            if in_shard(r["uid"], (shard_i, shard_n))]
    log.info("prepass %s: %d tasks in shard %d/%d, targets %s, budget USD %.0f, prefix %s",
             args.source, len(pool), shard_i, shard_n, targets, args.budget_usd, cfg.r2_prefix)
    if args.dry_run:
        print(json.dumps({"source": args.source, "tasks": len(pool), "targets": targets,
                          "rollouts": sum(len(pool) * n for n in targets.values())}, indent=1))
        return

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(cfg.data_dir / "runs", ignore_errors=True)
    harness_of = {p.id: p.harness for p in registry.policies.values()}
    health = EndpointHealth()
    r2 = R2TraceMirror(bucket=cfg.r2_bucket, endpoint=cfg.r2_endpoint,
                       access_key_id=cfg.r2_access_key_id,
                       secret_access_key=cfg.r2_secret_access_key,
                       prefix=cfg.r2_prefix)
    panel = panel_keys()
    baker = ToolBaker.from_pretrained() if any(
        registry.policies[pid].action_kind == "tool_call" for pid in targets) else None

    n_workers = max(1, args.parallel)
    batch_size = args.worker_batch or source.max_batch or cfg.batch_size
    workers = [Worker(k, dataclasses.replace(cfg, data_dir=cfg.data_dir / f"w{k}", batch_size=batch_size),
                      env, health, harness_of) for k in range(n_workers)]
    state_paths = [w.cfg.state_path for w in workers]
    lock = threading.Lock()
    mirror_lock = threading.Lock()
    in_flight: set[str] = set()     # uids being rolled right now (any policy)
    stopped: set[str] = set()       # policies over budget / exhausted
    log.info("prepass %s: %d worker(s), %d rollouts per batch", args.source, n_workers, batch_size)

    def plan_batch() -> tuple[str, list[dict]] | None:
        """(policy, batch) for the policy with the most remaining attempts;
        tasks with the fewest attempts first so every pass covers the set."""
        counts = attempt_counts(state_paths, args.source, set(targets))
        if args.budget_usd > 0 and paid - stopped:
            usd = spend_usd(state_paths, args.source, paid)
            if usd >= args.budget_usd:
                log.warning("budget reached: USD %.2f >= %.2f; paid policies %s stop", usd, args.budget_usd, sorted(paid))
                stopped.update(paid)
        best: tuple[int, str, list[dict]] | None = None
        for pid, n in targets.items():
            if pid in stopped:
                continue
            c = counts[pid]
            pending = [r for r in pool if c[r["uid"]] < n and r["uid"] not in in_flight]
            remaining = sum(n - c[r["uid"]] for r in pool if c[r["uid"]] < n)
            if not pending:
                if remaining == 0:
                    stopped.add(pid)
                continue
            pending.sort(key=lambda r: c[r["uid"]])
            if best is None or remaining > best[0]:
                best = (remaining, pid, pending[:batch_size])
        if best is None:
            return None
        return best[1], best[2]

    def run_one(w: Worker, pid: str, batch: list[dict]) -> None:
        policy = registry.policies[pid]
        if not health.preflight(policy, env):
            log.warning("w%d policy %s: no endpoint passed preflight; sleeping %ds", w.k, pid, PREFLIGHT_SLEEP_S)
            time.sleep(PREFLIGHT_SLEEP_S)
            return
        counts = attempt_counts(state_paths, args.source, set(targets))
        done_tasks = sum(1 for r in pool if all(counts[p][r["uid"]] >= n for p, n in targets.items()))
        log.info("cycle: source=%s policy=%s worker=w%d batch=%d tasks_complete=%d/%d spend_usd=%.2f",
                 args.source, pid, w.k, len(batch), done_tasks, len(pool),
                 spend_usd(state_paths, args.source, paid) if paid else 0.0)
        ok, _ = process_batch(w.cfg, source, policy, batch, w.runners, w.state, w.store, w.index, panel, baker)
        w.fails = 0 if ok else w.fails + 1
        if w.fails >= MAX_CONSECUTIVE_FAILS:
            log.error("w%d: %d consecutive batch failures; sleeping %ds", w.k, w.fails, FAIL_SLEEP_S)
            time.sleep(FAIL_SLEEP_S)
            w.fails = 0
        try:
            with mirror_lock:
                r2.mirror(w.store)
        except Exception:
            log.warning("R2 publish failed; will retry next cycle", exc_info=True)

    free = list(workers)
    futures: dict = {}
    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="prepass") as ex:
        while True:
            refresh_king_env(env)
            with lock:
                while free:
                    planned = plan_batch()
                    if planned is None:
                        break
                    pid, batch = planned
                    in_flight.update(r["uid"] for r in batch)
                    w = free.pop()
                    futures[ex.submit(run_one, w, pid, batch)] = (w, batch)
            if not futures:
                for w in workers:
                    try:
                        with mirror_lock:
                            r2.mirror(w.store)
                    except Exception:
                        log.warning("final R2 publish failed for w%d", w.k, exc_info=True)
                counts = attempt_counts(state_paths, args.source, set(targets))
                summary = {pid: {"tasks_at_target": sum(1 for r in pool if counts[pid][r["uid"]] >= n),
                                 "attempts": sum(counts[pid].values())} for pid, n in targets.items()}
                log.info("prepass %s complete: %s spend_usd=%.2f", args.source, summary,
                         spend_usd(state_paths, args.source, paid) if paid else 0.0)
                return
            done_set, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for fut in done_set:
                w, batch = futures.pop(fut)
                with lock:
                    in_flight.difference_update(r["uid"] for r in batch)
                    free.append(w)
                try:
                    fut.result()
                except Exception:
                    log.exception("w%d: batch crashed; tasks return to the pool", w.k)


if __name__ == "__main__":
    main()
