"""Coverage backfill: N graded rollouts on EVERY datagen source for one model.

Operator directive 2026-09-15: every king, the genesis and the teacher must
have a score on every environment of affine.io/#kings, on >= 50 rollouts. The
live king seat fills environments by deficit, so a short reign covers 9-13 of
36; past kings and the genesis were never served at all. This driver replaces
the deficit scheduler with a coverage loop and otherwise reuses the datagen
supervisor end to end (runners, TraceStore, RolloutIndex, R2 publish).

    python -m rollouts.backfill --digest12 995ad96eacd9 --n 50
    python -m rollouts.backfill --teacher --n 50 --sources nl2repobench

Policies are the `king_*` policies of each source (or `teacher_*` with
--teacher) cloned under the id `backfill_<digest12>_<harness>` (teacher:
`backfill_teacher_<harness>`); endpoints, sampling (T = 0.8), loop guard and
action kind are unchanged, so a backfill rollout is the seat's rollout with
another label. The model comes from the same `.king_env` file the seat
uses (KING_BASE_URL / KING_MODEL / KING_KEY) — point it at the backfill
box, never at the live seat.

Task pick: the source pool in `ordered_rows` order (ROLLOUTS_SEED; use the
same seed for every model so all rows see the same tasks), skipping tasks
this policy's seat already rolled (restart-safe through state.jsonl).
Stop rule per source: graded (resolved + unresolved) >= N, or attempts >=
--max-attempts (errored rollouts do not count as graded).

Concurrency (2026-09-21): --parallel W worker threads (default 4) each run
ONE source at a time — the most-behind source nobody else holds — with
their own data dir (`<data_dir>/w<k>`: state, trace store, index, run
scratch), so a 25-source row fills W sources at once instead of one
1-3 h cycle after another. Graded counts and "done" task sets are read
across every worker's state file (plus the pre-concurrency `state.jsonl`
at the data-dir root), so a restart with a different W resumes cleanly.
Per-worker batch = --worker-batch (default ceil(2 * ROLLOUTS_BATCH_SIZE /
W), min 4) so the serving box sees ~2 batches' worth in flight, and the
docker container cap is split the same way (the driver box has 24 cores;
diskgc runs before every batch as before).

Publish: ROLLOUTS_R2_PREFIX must be `traces-backfill/` (fail-closed here):
the fold reads `traces/manifest.json` only, so D never sees these; the
kingboard ingests `traces-backfill/manifest.json` and scores them.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shlex
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from affine.toolbake import ToolBaker

from rollouts.config import RolloutsConfig, load_config
from rollouts.index import RolloutIndex
from rollouts.king import refresh_king_env
from rollouts.panel import panel_keys
from rollouts.r2mirror import R2TraceMirror
from rollouts.registry import KING_POLICY_PREFIX, Registry, load_registry
from rollouts.run import (
    load_catalog_or_empty,
    ordered_rows,
    process_batch,
)
from rollouts.runners.base import EndpointHealth
from rollouts.runners.mini_swe import MiniSweRunner
from rollouts.runners.verifiers import VerifiersChatRunner, VerifiersRunner
from rollouts.scheduler import UnifiedState, policy_seat
from rollouts.schema import Policy
from rollouts.store import TraceStore

log = logging.getLogger("rollouts.backfill")

BACKFILL_PREFIX = "backfill_"
TEACHER_POLICY_PREFIX = "teacher_"
GREEDY_SUFFIX = "_greedy"          # king_*_greedy: T = 0 variants, not the standard sampling
REQUIRED_R2_PREFIX = "traces-backfill/"
GRADED = ("resolved", "unresolved")
PREFLIGHT_SLEEP_S = 60
FAIL_SLEEP_S = 300
MAX_CONSECUTIVE_FAILS = 3


def backfill_policies(registry: Registry, source_name: str, tag: str,
                      teacher: bool) -> list[Policy]:
    """The source's seat policies (king_* or teacher_*) cloned under the
    backfill ids, greedy variants excluded, in the source's policy order."""
    prefix = TEACHER_POLICY_PREFIX if teacher else KING_POLICY_PREFIX
    out = []
    for pid in registry.sources[source_name].policies:
        if not pid.startswith(prefix) or pid.endswith(GREEDY_SUFFIX):
            continue
        base = registry.policies[pid]
        harness_tag = pid[len(prefix):]
        out.append(dataclasses.replace(base, id=f"{BACKFILL_PREFIX}{tag}_{harness_tag}"))
    return out


def graded_counts(state_paths: Path | list[Path], policy_ids: set[str]) -> dict[str, dict[str, int]]:
    """source -> {graded, attempts} over the backfill policies' state rows,
    summed over every state file given (the workers' plus the legacy root)."""
    out: dict[str, dict[str, int]] = {}
    paths = [state_paths] if isinstance(state_paths, Path) else list(state_paths)
    for state_path in paths:
        if not state_path.exists():
            continue
        with state_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("policy_id") not in policy_ids:
                    continue
                c = out.setdefault(rec["source"], {"graded": 0, "attempts": 0, "infra": 0})
                # An errored row with no model call (n_calls 0: a missing
                # image, a harness crash before the first request) is an
                # infrastructure failure, not one of the model's attempts;
                # 156 such rows had used up terminal_lego's --max-attempts.
                # Counted apart ("infra") and capped at 2 x --max-attempts
                # so a permanently broken harness cannot spin forever.
                if rec.get("outcome") == "error" and not rec.get("n_calls"):
                    c["infra"] += 1
                    continue
                c["attempts"] += 1
                if rec.get("outcome") in GRADED:
                    c["graded"] += 1
    return out


def done_across(state_paths: list[Path], harness_of: dict[str, str], source: str, seat: str) -> set[str]:
    """Tasks `seat` already rolled on `source`, over every state file (a
    source may move between workers across restarts / W changes)."""
    done: set[str] = set()
    for path in state_paths:
        if path.exists():
            done |= UnifiedState(path, harness_of=harness_of).done_for(source, seat)
    return done


class Worker:
    """One thread's private pipeline: its own data dir, state, store, index
    and runner instances (run scratch under <data_dir>/runs). Shared with
    the others: registry, plan, pools, endpoint health, tool baker, env."""

    def __init__(self, k: int, cfg: RolloutsConfig, env: dict, health: EndpointHealth,
                 harness_of: dict[str, str]):
        self.k = k
        self.cfg = cfg
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.cfg.data_dir / "runs", ignore_errors=True)
        self.state = UnifiedState(self.cfg.state_path, harness_of=harness_of)
        self.store = TraceStore(self.cfg.store_dir)
        self.index = RolloutIndex(self.cfg.store_dir)
        self.runners = {
            "verifiers": VerifiersRunner(self.cfg, health, env),
            "verifiers_chat": VerifiersChatRunner(self.cfg, health, env),
            "mini_swe": MiniSweRunner(self.cfg, health, env),
        }
        self.fails = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--digest12", help="model digest prefix (12 hex) the policies are labelled with")
    who.add_argument("--teacher", action="store_true", help="backfill the teacher (teacher_* policies, backfill_teacher_* ids)")
    ap.add_argument("--n", type=int, default=50, help="graded rollouts per source")
    ap.add_argument("--max-attempts", type=int, default=150, help="attempts per source before giving up")
    ap.add_argument("--sources", default="all", help="comma list, or all = every LIVE source (share > 0)")
    ap.add_argument("--skip", default="affine_wiki", help="comma list of sources to leave out (default: the env with no grader)")
    ap.add_argument("--parallel", type=int, default=int(os.environ.get("ROLLOUTS_BACKFILL_PARALLEL", "4")),
                    help="sources rolled at once, one worker thread each (default 4)")
    ap.add_argument("--worker-batch", type=int, default=0,
                    help="rollouts per worker batch (default ceil(2 * ROLLOUTS_BATCH_SIZE / parallel), min 4)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg: RolloutsConfig = load_config()
    if cfg.r2_prefix != REQUIRED_R2_PREFIX:
        sys.exit(f"ROLLOUTS_R2_PREFIX must be {REQUIRED_R2_PREFIX!r} for a backfill (got {cfg.r2_prefix!r}); "
                 "the fold must never see these rollouts")
    if not (cfg.r2_endpoint and cfg.r2_access_key_id and cfg.r2_secret_access_key):
        sys.exit("ROLLOUTS_R2_* missing (fail-closed: traces could never publish)")
    registry = load_registry()
    tag = "teacher" if args.teacher else args.digest12
    if not args.teacher and not (len(tag) == 12 and all(c in "0123456789abcdef" for c in tag)):
        sys.exit("--digest12 must be 12 lowercase hex characters")

    env = dict(os.environ)
    refresh_king_env(env)
    if not args.teacher:
        served = env.get("KING_MODEL") or ""
        if not served.endswith(tag):
            sys.exit(f"KING_MODEL={served!r} does not serve king-{tag}: refusing (is .king_env pointing at the backfill box?)")

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    wanted = ([n for n, src in registry.sources.items() if src.share > 0] if args.sources == "all"
              else [s.strip() for s in args.sources.split(",") if s.strip()])
    plan: dict[str, list[Policy]] = {}
    for name in wanted:
        if name in skip:
            continue
        if name not in registry.sources:
            sys.exit(f"unknown source {name!r}")
        pols = backfill_policies(registry, name, tag, args.teacher)
        if not pols:
            log.warning("source %s: no %s policy; skipped", name, "teacher_*" if args.teacher else "king_*")
            continue
        plan[name] = pols
    all_ids = {p.id for pols in plan.values() for p in pols}
    log.info("backfill %s: %d sources, target %d graded each, policies %s",
             tag, len(plan), args.n, sorted(all_ids))
    if args.dry_run:
        for name, pols in plan.items():
            print(name, [p.id for p in pols])
        return

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(cfg.data_dir / "runs", ignore_errors=True)
    harness_of = {p.id: p.harness for p in registry.policies.values()}
    harness_of.update({p.id: p.harness for pols in plan.values() for p in pols})
    health = EndpointHealth()
    r2 = R2TraceMirror(bucket=cfg.r2_bucket, endpoint=cfg.r2_endpoint,
                       access_key_id=cfg.r2_access_key_id,
                       secret_access_key=cfg.r2_secret_access_key,
                       prefix=cfg.r2_prefix)
    panel = panel_keys()
    baker = ToolBaker.from_pretrained() if any(
        p.action_kind == "tool_call" for pols in plan.values() for p in pols) else None

    pools = {name: ordered_rows(cfg, name, load_catalog_or_empty(cfg, registry.sources[name]))
             for name in plan}

    # Launch record for rollouts/scripts/backfill_relaunch.sh (a container
    # restart kills every driver; the record recreates the tmux session).
    # Two drivers can share a digest (the coverage queue's second-box `-b`
    # run: data dir `rollouts-data-<d12>-b`, wrapper `run_backfill_<d12>_b.sh`,
    # session `backfill-<d12>-b`); the data-dir suffix tells them apart.
    drivers_dir = Path("/root/rollouts/drivers")
    suffix = cfg.data_dir.name.removeprefix(f"rollouts-data-{tag}") if cfg.data_dir.name.startswith(f"rollouts-data-{tag}") else ""
    record_path = drivers_dir / f"{tag}{suffix}.json"
    try:
        drivers_dir.mkdir(parents=True, exist_ok=True)
        wrapper = Path(f"/root/rollouts/run_backfill_{tag}{suffix.replace('-', '_')}.sh")
        if not wrapper.exists():
            wrapper = Path("/root/rollouts/run_backfill.sh")
        log_path = f"/root/logs/backfill_{tag}{suffix.replace('-', '_')}.log"
        env_bits = " ".join(f"{k}={shlex.quote(os.environ[k])}" for k in
                            ("ROLLOUTS_DATA_DIR", "ROLLOUTS_KING_ENV", "ROLLOUTS_MAX_CONTAINERS", "ROLLOUTS_BATCH_SIZE")
                            if k in os.environ)
        record_path.write_text(json.dumps({
            "tag": tag, "tmux": f"backfill-{tag}{suffix}", "wrapper": str(wrapper), "log": log_path,
            "cmd": f"{env_bits} {wrapper} {shlex.join(sys.argv[1:])} >> {log_path} 2>&1".strip(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid(),
        }, indent=1))
    except OSError:
        log.warning("could not write the driver record %s", record_path, exc_info=True)

    n_workers = max(1, min(args.parallel, len(plan)))
    worker_batch = args.worker_batch or max(4, math.ceil(2 * cfg.batch_size / n_workers))
    worker_containers = max(worker_batch, math.ceil(cfg.max_containers / n_workers))
    workers = [Worker(k, dataclasses.replace(cfg, data_dir=cfg.data_dir / f"w{k}",
                                             batch_size=worker_batch,
                                             max_containers=worker_containers),
                      env, health, harness_of) for k in range(n_workers)]
    # Every state file that can hold this model's rows: the workers' and the
    # pre-concurrency driver's at the data-dir root (resume across versions).
    state_paths = [w.cfg.state_path for w in workers] + [cfg.state_path]
    log.info("backfill %s: %d worker(s), %d rollouts per worker batch, %d containers per worker",
             tag, n_workers, worker_batch, worker_containers)

    lock = threading.Lock()
    in_flight: dict[str, int] = {}          # source -> worker k
    round_robin: dict[str, int] = {}
    exhausted: set[str] = set()
    mirror_lock = threading.Lock()

    def run_one(w: Worker, name: str) -> None:
        source = registry.sources[name]
        pols = plan[name]
        with lock:
            i = round_robin.get(name, 0)
            round_robin[name] = i + 1
            counts = graded_counts(state_paths, all_ids)
        policy = pols[i % len(pols)]
        if not health.preflight(policy, env):
            log.warning("w%d policy %s: no endpoint passed preflight; sleeping %ds",
                        w.k, policy.id, PREFLIGHT_SLEEP_S)
            time.sleep(PREFLIGHT_SLEEP_S)
            return
        done = done_across(state_paths, harness_of, name, policy_seat(policy, env))
        pending = [r for r in pools[name] if r["uid"] not in done]
        have = counts.get(name, {}).get("graded", 0)
        need = args.n - have
        size = min(w.cfg.batch_size, source.max_batch or w.cfg.batch_size, max(1, need))
        batch = pending[:size]
        if not batch:
            log.warning("source %s: pool exhausted for %s at %s", name, policy.id, counts.get(name))
            with lock:
                exhausted.add(name)
            return
        log.info("cycle: source=%s policy=%s worker=w%d batch=%d graded=%d/%d attempts=%d",
                 name, policy.id, w.k, len(batch), have, args.n,
                 counts.get(name, {}).get("attempts", 0))
        ok, _ = process_batch(w.cfg, source, policy, batch, w.runners, w.state, w.store,
                              w.index, panel, baker)
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

    def pick(counts: dict) -> str | None:
        todo = [name for name in plan
                if name not in exhausted and name not in in_flight
                and counts.get(name, {}).get("graded", 0) < args.n
                and counts.get(name, {}).get("attempts", 0) < args.max_attempts
                and counts.get(name, {}).get("infra", 0) < 2 * args.max_attempts]
        if not todo:
            return None
        # most-behind source first (relative gap), so every env climbs together
        todo.sort(key=lambda n: counts.get(n, {}).get("graded", 0) / args.n)
        return todo[0]

    free = list(workers)
    futures = {}
    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="backfill") as ex:
        while True:
            refresh_king_env(env)
            counts = graded_counts(state_paths, all_ids)
            while free:
                name = pick(counts)
                if name is None:
                    break
                w = free.pop()
                in_flight[name] = w.k
                futures[ex.submit(run_one, w, name)] = (w, name)
            if not futures:
                remaining = pick(counts)
                if remaining is None:
                    for w in workers:
                        try:
                            with mirror_lock:
                                r2.mirror(w.store)
                        except Exception:
                            log.warning("final R2 publish failed for w%d", w.k, exc_info=True)
                    log.info("backfill %s complete: %s", tag,
                             {n: counts.get(n, {"graded": 0}) for n in plan})
                    try:
                        rec = json.loads(record_path.read_text())
                        rec["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        record_path.write_text(json.dumps(rec, indent=1))
                    except (OSError, ValueError):
                        pass
                    return
                time.sleep(5)
                continue
            done_set, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for fut in done_set:
                w, name = futures.pop(fut)
                in_flight.pop(name, None)
                free.append(w)
                try:
                    fut.result()
                except Exception:
                    log.exception("w%d: source %s cycle crashed; source stays in the plan", w.k, name)


if __name__ == "__main__":
    main()
