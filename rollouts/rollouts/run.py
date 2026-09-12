"""Unified rollout datagen supervisor.

Cycle: pick source by kept-turn deficit -> pick policy by per-policy
deficit -> next unprocessed tasks (seed-deterministic shuffle, restart-
safe) -> runner (verifiers eval or mini-swe agent) -> store envelopes
(system of record) + parquet index -> derive duel_turns (yield accounting
only) -> publish trace chunks to data.affine.io -> mark state.

Traces are canonical (2026-09-02): the pod no longer stages turn shards;
D is derived from the published traces by ops/corpus_build.py on the
validator box. Restart-safe: outcomes live in state.jsonl (written only
after a batch's traces are parsed), stored chunks are immutable and marked
mirrored only after their put succeeded, and a crash mid-batch just means
the un-marked tasks are re-selected.

  python -m rollouts.run           # the service (supervised by bootstrap.sh)
  python -m rollouts.run --once    # one batch, then exit
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

from affine.toolbake import ToolBaker

from rollouts.catalog import load_catalog
from rollouts.config import RolloutsConfig, load_config
from rollouts.index import RolloutIndex
from rollouts.king import refresh_king_env
from rollouts.panel import panel_keys
from rollouts.r2mirror import R2TraceMirror
from rollouts.registry import Registry, load_registry
from rollouts.runners.base import BatchResult, EndpointHealth
from rollouts.runners.mini_swe import MiniSweRunner
from rollouts.runners.verifiers import (
    VerifiersChatRunner,
    VerifiersRunner,
    reap_all_verifiers_containers,
)
from rollouts.scheduler import Scheduler, UnifiedState
from rollouts.store import TraceStore
from rollouts.uploader import TraceMirror
from rollouts.views.duel_turns import derive_turns, validate_records

log = logging.getLogger("rollouts.run")

MAX_CONSECUTIVE_FAILS = 3
FAIL_SLEEP_S = 600
POOL_EXHAUSTED_SLEEP_S = 6 * 3600
PREFLIGHT_SKIP_SLEEP_S = 5
# Touch this file to make the supervisor exit cleanly between batches (the
# bootstrap loop relaunches it) — a redeploy without killing a running batch.
RESTART_FLAG = Path(os.environ.get("ROLLOUTS_RESTART_FLAG", "/root/rollouts/RESTART"))


def _utc_tag() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def owns_task(shard: tuple[int, int], uid: str) -> bool:
    """Shard ownership is a pure function of the uid, so every pod in the
    fleet computes the same partition from the same catalog."""
    i, n = shard
    if n == 1:
        return True
    h = hashlib.blake2b(uid.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % n == i


def ordered_rows(cfg: RolloutsConfig, name: str,
                 catalog: list[dict]) -> list[dict]:
    rows = [r for r in catalog
            if (cfg.langs is None or (r.get("language") or "") in cfg.langs)
            and owns_task(cfg.shard, r["uid"])]
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
                  index: RolloutIndex, panel, baker=None) -> tuple[bool, int]:
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

    # Derive + validate the duel_turns view for yield accounting only (the
    # scheduler's deficit is in kept turns); the fold re-derives from the
    # published traces, so nothing here is stored.
    records: list[dict] = []
    sid_to_rollout: dict[str, str] = {}
    for env in result.envelopes:
        sid_to_rollout[env["task"]["sid"]] = env["rollout_id"]
        records.extend(derive_turns(env, panel=panel, baker=baker))
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
            harness=policy.harness,
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

    stops: dict[str, int] = {}
    for env in result.envelopes:
        stop = str(env["trace"].get("stop_condition") or "none")
        stops[stop] = stops.get(stop, 0) + 1
    log.info("batch %s [%s/%s]: %d rollouts, %d kept turns in %.0fs; "
             "stops=%s", tag, policy.id, endpoint_label,
             len(result.envelopes), len(kept), time.time() - t0,
             dict(sorted(stops.items())))
    shutil.rmtree(run_dir, ignore_errors=True)
    return result.produced_output, len(kept)


def flush_uploads(r2: R2TraceMirror, hf: TraceMirror | None,
                  store: TraceStore) -> None:
    """Publish unmirrored trace chunks to data.affine.io (chunks, then the
    immutable manifest, then the pointer), and refresh the HF cold copy. A
    failure leaves the chunks unmarked so the next cycle retries."""
    try:
        r2.mirror(store)
    except Exception:
        log.warning("R2 trace publish failed; will retry next cycle",
                    exc_info=True)
    if hf is not None:
        try:
            hf.mirror(store)
        except Exception:
            log.warning("HF cold copy failed; will retry next cycle",
                        exc_info=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--once", action="store_true",
                    help="process a single batch, then exit")
    ap.add_argument("--source", default=None,
                    help="bypass the scheduler and force this source "
                         "every cycle (diagnostics)")
    ap.add_argument("--policy", default=None,
                    help="bypass the policy pick and force this policy id "
                         "every cycle (diagnostics; with --source)")
    ap.add_argument("--no-mirror", action="store_true",
                    help="skip the HF cold copy of trace chunks")
    ap.add_argument("--reap-all", action="store_true",
                    help="at start-up, remove EVERY container in the verifiers "
                         "image namespaces, whoever created it (pod-start "
                         "orphan sweep; the per-batch reaper only touches this "
                         "supervisor's and dead supervisors' containers)")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = load_config()
    registry: Registry = load_registry()
    if not (cfg.r2_endpoint and cfg.r2_access_key_id and cfg.r2_secret_access_key):
        sys.exit("ROLLOUTS_R2_* missing (fail-closed: traces could never publish)")
    hf_mirror = cfg.hf_trace_mirror and not args.no_mirror
    if hf_mirror and not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN missing while ROLLOUTS_HF_TRACE_MIRROR is on")
    # One env dict shared by the scheduler and every runner; the king seat
    # vars are refreshed into it each cycle (rollouts.king).
    env = dict(os.environ)
    refresh_king_env(env)
    if not any(p.available_endpoints(env)
               for p in registry.policies.values()):
        sys.exit("no policy endpoint has its key env set (fail-closed)")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(cfg.data_dir / "runs", ignore_errors=True)
    if args.reap_all:
        reap_all_verifiers_containers()

    state = UnifiedState(cfg.state_path, harness_of={
        pid: p.harness for pid, p in registry.policies.items()})
    health = EndpointHealth()
    scheduler = Scheduler(registry, state, env, health)
    store = TraceStore(cfg.store_dir)
    index = RolloutIndex(cfg.store_dir)
    runners = {
        "verifiers": VerifiersRunner(cfg, health, env),
        "verifiers_chat": VerifiersChatRunner(cfg, health, env),
        "mini_swe": MiniSweRunner(cfg, health, env),
    }
    r2 = R2TraceMirror(bucket=cfg.r2_bucket, endpoint=cfg.r2_endpoint,
                       access_key_id=cfg.r2_access_key_id,
                       secret_access_key=cfg.r2_secret_access_key,
                       prefix=cfg.r2_prefix)
    hf = TraceMirror(cfg.traces_hf_repo) if hf_mirror else None
    panel = panel_keys()
    # Tool-use traces need the teacher's chat template to bake tool schemas /
    # calls / results into plain prefixes (and to prove byte parity). Only
    # loaded when a tool_call policy is enabled; a bash/boxed-only registry
    # never touches the tokenizer.
    baker = None
    if any(p.action_kind == "tool_call" for p in registry.policies.values()):
        baker = ToolBaker.from_pretrained()
        log.info("tool baker ready: %s", baker.tok.name_or_path)

    log.info("rollouts starting: sources=%s targets=%s batch=%d "
             "containers=%d shard=%d/%d traces=%s/%s",
             sorted(registry.sources), {k: round(v, 3) for k, v in
                                        registry.target_shares().items()},
             cfg.batch_size, cfg.max_containers, cfg.shard[0], cfg.shard[1],
             cfg.r2_bucket, cfg.r2_prefix)

    pools = {name: ordered_rows(cfg, name, load_catalog(cfg, src))
             for name, src in registry.sources.items()}
    for name, rows in pools.items():
        log.info("source %s: %d selectable tasks (%d processed by some "
                 "teacher-side seat)", name, len(rows), len(state.done_for(name)))

    fails = 0
    while True:
        if RESTART_FLAG.exists():
            # Graceful redeploy: exit between batches; bootstrap.sh relaunches
            # the supervisor on the freshly deployed code.
            RESTART_FLAG.unlink(missing_ok=True)
            flush_uploads(r2, hf, store)
            log.info("restart flag %s seen; exiting for relaunch", RESTART_FLAG)
            return
        refresh_king_env(env)
        # Per source: tasks some usable seat still has to run (the king seat
        # replays the teacher's tasks, so an exhausted teacher pool is not
        # an exhausted source while the king is up).
        remaining = {name: scheduler.remaining(name, rows)
                     for name, rows in pools.items()}
        if args.source:
            name = args.source if remaining.get(args.source) else None
        else:
            name = scheduler.pick_source(remaining)
        if name is None:
            log.info("all pools exhausted or cooling; sleeping %ds",
                     POOL_EXHAUSTED_SLEEP_S)
            flush_uploads(r2, hf, store)
            if args.once:
                break
            time.sleep(POOL_EXHAUSTED_SLEEP_S)
            pools = {n: ordered_rows(cfg, n, load_catalog(cfg, s))
                     for n, s in registry.sources.items()}
            continue
        source = registry.sources[name]
        if args.policy:
            policy = registry.policies[args.policy]
        else:
            policy = scheduler.pick_policy(name, pools[name])
        pending = scheduler.pending(name, pools[name], policy)
        batch = pending[: min(cfg.batch_size, source.max_batch or cfg.batch_size)]
        log.info("cycle: source=%s policy=%s batch=%d seat_pending=%d "
                 "remaining=%s", name, policy.id, len(batch), len(pending),
                 remaining)
        if not health.preflight(policy, env):
            # A dynamic endpoint (the king seat) does not answer: struck, so
            # the scheduler prefers another policy while it cools. Not a
            # batch failure and not a zero-yield strike on the source.
            log.warning("policy %s: no endpoint passed preflight; skipping "
                        "this cycle", policy.id)
            time.sleep(PREFLIGHT_SKIP_SLEEP_S)
            continue

        ok, kept_turns = process_batch(
            cfg, source, policy, batch, runners, state, store, index, panel,
            baker)
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
        flush_uploads(r2, hf, store)
        if args.once:
            break


if __name__ == "__main__":
    main()
