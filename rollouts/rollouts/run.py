"""Unified rollout datagen supervisor.

Cycle: pick source by kept-turn deficit -> pick policy by per-policy
deficit -> next unprocessed tasks (seed-deterministic shuffle, restart-
safe) -> runner (verifiers eval or mini-swe agent) -> store envelopes
(system of record) + parquet index -> derive duel_turns@v3 -> validate ->
queue + upload turn shards -> mirror trace chunks -> mark state.

Restart-safe: outcomes live in state.jsonl (written only after a batch's
traces are parsed), sliced-but-not-uploaded turns survive in
pending_turns.jsonl / outbox/, stored chunks are immutable, and a crash
mid-batch just means the un-marked tasks are re-selected.

  python -m rollouts.run           # the service (supervised by bootstrap.sh)
  python -m rollouts.run --once    # one batch, then exit
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

from datagen.uploader import TurnUploader, shard_sha256

from rollouts.catalog import load_catalog
from rollouts.config import RolloutsConfig, load_config
from rollouts.index import RolloutIndex
from rollouts.panel import panel_keys
from rollouts.registry import Registry, load_registry
from rollouts.runners.base import BatchResult, EndpointHealth
from rollouts.runners.mini_swe import MiniSweRunner
from rollouts.runners.verifiers import VerifiersRunner
from rollouts.scheduler import Scheduler, UnifiedState
from rollouts.store import TraceStore
from rollouts.uploader import TraceMirror
from rollouts.views.duel_turns import derive_turns, validate_records

log = logging.getLogger("rollouts.run")

MAX_CONSECUTIVE_FAILS = 3
FAIL_SLEEP_S = 600
POOL_EXHAUSTED_SLEEP_S = 6 * 3600


def _utc_tag() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def ordered_rows(cfg: RolloutsConfig, name: str,
                 catalog: list[dict]) -> list[dict]:
    rows = [r for r in catalog
            if cfg.langs is None or (r.get("language") or "") in cfg.langs]
    rng = random.Random(f"{cfg.seed}:{name}")
    rng.shuffle(rows)
    return rows


def _outcome(row: dict) -> str:
    if row.get("error"):
        return "error"
    score = row.get("resolved")
    if score is None:
        return "unscored"
    return "resolved" if score == 1.0 else "unresolved"


def process_batch(cfg: RolloutsConfig, source, policy, batch: list[dict],
                  runners: dict, state: UnifiedState, store: TraceStore,
                  index: RolloutIndex, panel) -> tuple[bool, int]:
    """One batch end to end. Returns (produced_output, kept_turns)."""
    tag = f"{source.name}-{_utc_tag()}"
    run_dir = cfg.data_dir / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_by_uid = {r["uid"]: r for r in batch}
    t0 = time.time()

    result: BatchResult = runners[source.runner].run_batch(
        source, policy, batch, run_dir)

    if not result.per_task and not result.envelopes:
        shutil.rmtree(run_dir, ignore_errors=True)
        return result.produced_output, 0

    # Derive + validate the duel_turns view; count survivors per rollout.
    records: list[dict] = []
    sid_to_rollout: dict[str, str] = {}
    for env in result.envelopes:
        sid_to_rollout[env["task"]["sid"]] = env["rollout_id"]
        records.extend(derive_turns(env, panel=panel))
    kept, drops = validate_records(records, panel)
    if drops:
        log.info("validation drops: %s", drops)
    kept_by_rollout: dict[str, int] = {}
    kept_by_sid: dict[str, int] = {}
    for rec in kept:
        sid = rec["instance_id"]
        kept_by_sid[sid] = kept_by_sid.get(sid, 0) + 1
        rid = sid_to_rollout.get(sid)
        if rid:
            kept_by_rollout[rid] = kept_by_rollout.get(rid, 0) + 1
    if kept:
        cfg.pending_turns.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.pending_turns, "a", encoding="utf-8") as f:
            for rec in kept:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # System of record + index, then (and only then) mark state.
    chunk_key = store.append_batch(result.envelopes, tag)
    if chunk_key:
        index.append(result.envelopes, chunk_key, kept_by_rollout)

    endpoint_label = result.endpoint.label if result.endpoint else ""
    for row in result.per_task:
        uid = row["uid"]
        sid = meta_by_uid.get(uid, {}).get("sid", "")
        state.mark(
            source.name, uid, _outcome(row),
            policy_id=policy.id,
            n_turns=kept_by_sid.get(sid, 0),
            provider=endpoint_label,
            detail=row.get("error") or row.get("stop") or "",
            cost_usd=row.get("cost_usd", 0.0),
            prompt_tokens=row.get("prompt_tokens", 0),
            completion_tokens=row.get("completion_tokens", 0),
            n_calls=row.get("n_calls", 0),
            agent_wall_s=row.get("agent_wall_s"))
    missing = [u for u in meta_by_uid
               if u not in {r["uid"] for r in result.per_task}]
    if missing:
        log.info("%d task(s) produced no trace; will be re-selected",
                 len(missing))

    log.info("batch %s [%s/%s]: %d rollouts, %d kept turns in %.0fs",
             tag, policy.id, endpoint_label, len(result.envelopes),
             len(kept), time.time() - t0)
    shutil.rmtree(run_dir, ignore_errors=True)
    return result.produced_output, len(kept)


def flush_uploads(cfg: RolloutsConfig, uploader: TurnUploader,
                  mirror: TraceMirror | None, store: TraceStore) -> None:
    """Cut pending turns into a shard when big or old enough, push queued
    shards, and mirror unmirrored trace chunks. Atomic renames: a crash or
    upload failure never loses or duplicates turns."""
    outbox = cfg.data_dir / "outbox"
    cut_state = cfg.data_dir / "upload_state.json"
    last_cut = 0.0
    if cut_state.exists():
        last_cut = float(json.loads(cut_state.read_text()).get("last_cut", 0))
    pending = cfg.pending_turns
    n_pending = 0
    if pending.exists():
        with open(pending, "rb") as f:
            n_pending = sum(1 for line in f if line.strip())
    if n_pending and (n_pending >= cfg.upload_min_turns
                      or time.time() - last_cut >= cfg.upload_interval_s):
        outbox.mkdir(parents=True, exist_ok=True)
        sha = shard_sha256(pending)
        dest = outbox / f"rollout-turns-{_utc_tag()}-{sha[:12]}.jsonl"
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
    if mirror is not None:
        try:
            mirror.mirror(store)
        except Exception:
            log.warning("trace mirror failed; will retry next cycle",
                        exc_info=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--once", action="store_true",
                    help="process a single batch, then exit")
    ap.add_argument("--source", default=None,
                    help="bypass the scheduler and force this source "
                         "every cycle (diagnostics)")
    ap.add_argument("--no-mirror", action="store_true",
                    help="skip the trace-chunk HF mirror")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = load_config()
    registry: Registry = load_registry()
    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN missing (fail-closed: turns could never upload)")
    if not any(p.available_endpoints(os.environ)
               for p in registry.policies.values()):
        sys.exit("no policy endpoint has its key env set (fail-closed)")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(cfg.data_dir / "runs", ignore_errors=True)

    state = UnifiedState(cfg.state_path)
    scheduler = Scheduler(registry, state)
    store = TraceStore(cfg.store_dir)
    index = RolloutIndex(cfg.store_dir)
    health = EndpointHealth()
    env = dict(os.environ)
    runners = {
        "verifiers": VerifiersRunner(cfg, health, env),
        "mini_swe": MiniSweRunner(cfg, health, env),
    }
    uploader = TurnUploader(cfg.turns_hf_repo, private=True)
    mirror = None if args.no_mirror else TraceMirror(cfg.traces_hf_repo)
    panel = panel_keys()

    log.info("rollouts starting: sources=%s targets=%s batch=%d "
             "containers=%d turns_repo=%s",
             sorted(registry.sources), {k: round(v, 3) for k, v in
                                        registry.target_shares().items()},
             cfg.batch_size, cfg.max_containers, cfg.turns_hf_repo)

    pools = {name: ordered_rows(cfg, name, load_catalog(cfg, src))
             for name, src in registry.sources.items()}
    for name, rows in pools.items():
        log.info("source %s: %d selectable tasks (%d already processed)",
                 name, len(rows), len(state.done_for(name)))

    fails = 0
    while True:
        remaining = {
            name: sum(1 for r in rows
                      if r["uid"] not in state.done_for(name))
            for name, rows in pools.items()
        }
        if args.source:
            name = args.source if remaining.get(args.source) else None
        else:
            name = scheduler.pick_source(remaining)
        if name is None:
            log.info("all pools exhausted or cooling; sleeping %ds",
                     POOL_EXHAUSTED_SLEEP_S)
            flush_uploads(cfg, uploader, mirror, store)
            if args.once:
                break
            time.sleep(POOL_EXHAUSTED_SLEEP_S)
            pools = {n: ordered_rows(cfg, n, load_catalog(cfg, s))
                     for n, s in registry.sources.items()}
            continue
        source = registry.sources[name]
        policy = scheduler.pick_policy(name)
        done = state.done_for(name)
        batch = [r for r in pools[name]
                 if r["uid"] not in done][: cfg.batch_size]
        log.info("cycle: source=%s policy=%s batch=%d remaining=%s",
                 name, policy.id, len(batch), remaining)

        ok, kept_turns = process_batch(
            cfg, source, policy, batch, runners, state, store, index, panel)
        scheduler.record_batch_yield(name, kept_turns)
        if ok:
            fails = 0
        else:
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILS:
                log.error("%d consecutive batch failures; sleeping %ds",
                          fails, FAIL_SLEEP_S)
                time.sleep(FAIL_SLEEP_S)
                fails = 0
        flush_uploads(cfg, uploader, mirror, store)
        if args.once:
            break


if __name__ == "__main__":
    main()
