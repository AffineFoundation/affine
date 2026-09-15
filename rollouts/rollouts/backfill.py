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

Publish: ROLLOUTS_R2_PREFIX must be `traces-backfill/` (fail-closed here):
the fold reads `traces/manifest.json` only, so D never sees these; the
kingboard ingests `traces-backfill/manifest.json` and scores them.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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


def graded_counts(state_path: Path, policy_ids: set[str]) -> dict[str, dict[str, int]]:
    """source -> {graded, attempts} over the backfill policies' state rows."""
    out: dict[str, dict[str, int]] = {}
    if not state_path.exists():
        return out
    with state_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("policy_id") not in policy_ids:
                continue
            c = out.setdefault(rec["source"], {"graded": 0, "attempts": 0})
            c["attempts"] += 1
            if rec.get("outcome") in GRADED:
                c["graded"] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--digest12", help="model digest prefix (12 hex) the policies are labelled with")
    who.add_argument("--teacher", action="store_true", help="backfill the teacher (teacher_* policies, backfill_teacher_* ids)")
    ap.add_argument("--n", type=int, default=50, help="graded rollouts per source")
    ap.add_argument("--max-attempts", type=int, default=150, help="attempts per source before giving up")
    ap.add_argument("--sources", default="all", help="comma list, or all = every LIVE source (share > 0)")
    ap.add_argument("--skip", default="affine_wiki", help="comma list of sources to leave out (default: the env with no grader)")
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
    state = UnifiedState(cfg.state_path, harness_of=harness_of)
    health = EndpointHealth()
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
    panel = panel_keys()
    baker = ToolBaker.from_pretrained() if any(
        p.action_kind == "tool_call" for pols in plan.values() for p in pols) else None

    pools = {name: ordered_rows(cfg, name, load_catalog_or_empty(cfg, registry.sources[name]))
             for name in plan}
    fails = 0
    round_robin: dict[str, int] = {}
    while True:
        counts = graded_counts(cfg.state_path, all_ids)
        todo = [name for name in plan
                if counts.get(name, {}).get("graded", 0) < args.n
                and counts.get(name, {}).get("attempts", 0) < args.max_attempts]
        if not todo:
            r2.mirror(store)
            log.info("backfill %s complete: %s", tag,
                     {n: counts.get(n, {"graded": 0}) for n in plan})
            return
        # most-behind source first (relative gap), so every env climbs together
        todo.sort(key=lambda n: counts.get(n, {}).get("graded", 0) / args.n)
        name = todo[0]
        source = registry.sources[name]
        pols = plan[name]
        i = round_robin.get(name, 0)
        policy = pols[i % len(pols)]
        round_robin[name] = i + 1
        refresh_king_env(env)
        if not health.preflight(policy, env):
            log.warning("policy %s: no endpoint passed preflight; sleeping %ds", policy.id, PREFLIGHT_SLEEP_S)
            time.sleep(PREFLIGHT_SLEEP_S)
            continue
        done = state.done_for(name, policy_seat(policy, env))
        pending = [r for r in pools[name] if r["uid"] not in done]
        need = args.n - counts.get(name, {}).get("graded", 0)
        size = min(cfg.batch_size, source.max_batch or cfg.batch_size, max(1, need))
        batch = pending[:size]
        if not batch:
            log.warning("source %s: pool exhausted for %s at %s", name, policy.id, counts.get(name))
            # mark the source finished by exhausting attempts
            counts.setdefault(name, {"graded": 0, "attempts": 0})["attempts"] = args.max_attempts
            plan.pop(name, None)
            continue
        log.info("cycle: source=%s policy=%s batch=%d graded=%d/%d attempts=%d",
                 name, policy.id, len(batch), counts.get(name, {}).get("graded", 0), args.n,
                 counts.get(name, {}).get("attempts", 0))
        ok, _ = process_batch(cfg, source, policy, batch, runners, state, store, index, panel, baker)
        fails = 0 if ok else fails + 1
        if fails >= MAX_CONSECUTIVE_FAILS:
            log.error("%d consecutive batch failures; sleeping %ds", fails, FAIL_SLEEP_S)
            time.sleep(FAIL_SLEEP_S)
            fails = 0
        try:
            r2.mirror(store)
        except Exception:
            log.warning("R2 publish failed; will retry next cycle", exc_info=True)


if __name__ == "__main__":
    main()
