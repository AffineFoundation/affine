"""Deficit scheduling on kept turns, across sources and policies.

The scheduler owns two picks per cycle:

  source  largest deficit vs its [mix]-derived target share of kept turns,
          among sources with work remaining and not on cooldown. Deficit on
          KEPT TURNS (not attempts) is self-correcting: a low-yield source
          gets scheduled more often until it reaches target — bounded by
          the zero-yield cooldown so a dead lane cannot burn money forever.
  policy  within the picked source, largest deficit vs the policy's share,
          among policies with at least one endpoint whose key env is set.

Kept-turn counts come from the unified state (one jsonl row per processed
task, written after its batch converts) — the same numbers the Parquet
index carries, but restart-cheap to load.

Seats (2026-09-10): "done" is per (source, SEAT), not per source. The
teacher seat is every non-king policy (one pass over each task, as before).
Each king is its own seat, `king:<served model>`, so the king replays tasks
the teacher already solved — its failures on the teacher's tasks are the
whole point of the king seat (DAgger) — and a newly crowned king starts
over. Before this, the king could only pick tasks the teacher had not
reached yet: every exhausted pool (terminal, math, tool_use, nl2repo) was
closed to it and king failures came from three coding sources only.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from rollouts.registry import Registry
from rollouts.runners.base import EndpointHealth
from rollouts.schema import Policy

log = logging.getLogger("rollouts.scheduler")

ZERO_YIELD_STRIKES = 3        # consecutive zero-kept batches -> cooldown
ZERO_YIELD_COOLDOWN_S = 4 * 3600
KING_POLICY_PREFIX = "king_"
TEACHER_SEAT = "teacher"


def seat_of(policy_id: str, model: str) -> str:
    """The seat a (policy, served model) plays. `model` is the endpoint's
    model name (the `provider` label on state rows is `<endpoint>/<model>`)."""
    if policy_id.startswith(KING_POLICY_PREFIX):
        return f"king:{model or 'unknown'}"
    return TEACHER_SEAT


def policy_seat(policy: Policy, env: dict) -> str:
    eps = policy.available_endpoints(env)
    return seat_of(policy.id, eps[0].model if eps else "")


class UnifiedState:
    """Append-only jsonl keyed (source, uid): outcome + kept-turn counts.
    A task recorded here is never attempted again BY THE SAME SEAT
    (restart-safe; rows are written only after its batch's traces were
    parsed)."""

    def __init__(self, path: Path):
        self.path = path
        self.done: dict[tuple[str, str], set[str]] = {}
        self.kept_by_source: dict[str, int] = {}
        self.kept_by_policy: dict[tuple[str, str], int] = {}
        if path.exists():
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                self._absorb(rec)

    def _absorb(self, rec: dict) -> None:
        source, uid = rec["source"], rec["uid"]
        provider = str(rec.get("provider") or "")
        model = provider.split("/", 1)[1] if "/" in provider else provider
        seat = seat_of(str(rec.get("policy_id") or ""), model)
        self.done.setdefault((source, seat), set()).add(uid)
        n = int(rec.get("n_turns") or 0)
        self.kept_by_source[source] = self.kept_by_source.get(source, 0) + n
        pid = rec.get("policy_id") or ""
        key = (source, pid)
        self.kept_by_policy[key] = self.kept_by_policy.get(key, 0) + n

    def mark(self, source: str, uid: str, outcome: str, *,
             policy_id: str = "", n_turns: int = 0, **extra) -> None:
        rec = {"source": source, "uid": uid, "outcome": outcome,
               "policy_id": policy_id, "n_turns": n_turns,
               "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               **extra}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._absorb(rec)

    def done_for(self, source: str, seat: str = TEACHER_SEAT) -> set[str]:
        return self.done.get((source, seat), set())


class Scheduler:
    def __init__(self, registry: Registry, state: UnifiedState,
                 env: dict | None = None, health: EndpointHealth | None = None):
        self.registry = registry
        self.state = state
        self.env = env if env is not None else dict(os.environ)
        self.health = health
        self.targets = registry.target_shares()
        self._cooldown_until: dict[str, float] = {}
        self._zero_streak: dict[str, int] = {}

    # -- source pick -------------------------------------------------------------

    def eligible(self, remaining: dict[str, int]) -> list[str]:
        now = time.time()
        return [name for name in self.registry.sources
                if remaining.get(name, 0) > 0
                and self._cooldown_until.get(name, 0.0) <= now
                and self._usable_policies(name)]

    def pick_source(self, remaining: dict[str, int]) -> str | None:
        cands = self.eligible(remaining)
        if not cands:
            return None
        total_target = sum(self.targets[n] for n in cands)
        total_kept = sum(self.state.kept_by_source.get(n, 0)
                         for n in cands) + 1
        def deficit(name: str) -> float:
            share = self.targets[name] / total_target
            return share * total_kept - self.state.kept_by_source.get(name, 0)
        return max(cands, key=lambda n: (deficit(n), self.targets[n], n))

    # -- policy pick -------------------------------------------------------------

    def _usable_policies(self, source: str) -> list[Policy]:
        return [p for p in self.registry.policies_for(source)
                if p.available_endpoints(self.env)]

    def _healthy_policies(self, source: str) -> list[Policy]:
        """Usable policies whose endpoints are not ALL on cooldown (a struck
        king box is skipped for the cooldown instead of being re-picked by
        deficit every cycle). Falls back to every usable policy when all of
        them cool, so a source never goes unpicked while it has a route."""
        cands = self._usable_policies(source)
        if self.health is None:
            return cands
        warm = [p for p in cands if not self.health.all_cooling(p, self.env)]
        return warm or cands

    # -- seats: per-(source, seat) work ------------------------------------------

    def pending(self, source: str, rows: list[dict], policy: Policy) -> list[dict]:
        """Pool rows this policy's seat has not processed, in pool order."""
        done = self.state.done_for(source, policy_seat(policy, self.env))
        return [r for r in rows if r["uid"] not in done]

    def remaining(self, source: str, rows: list[dict]) -> int:
        """Tasks some usable policy of this source still has to run — the
        union over seats (a task the teacher finished is still work for the
        king)."""
        seats = {policy_seat(p, self.env) for p in self._usable_policies(source)}
        if not seats:
            return 0
        dones = [self.state.done_for(source, s) for s in seats]
        return sum(1 for r in rows if any(r["uid"] not in d for d in dones))

    def pick_policy(self, source: str, rows: list[dict] | None = None) -> Policy:
        cands = self._healthy_policies(source)
        if rows is not None:
            with_work = [p for p in cands if self.pending(source, rows, p)]
            if not with_work:
                # Every warm policy is finished here; fall back to any usable
                # policy with work (a cooling one still beats an idle cycle).
                with_work = [p for p in self._usable_policies(source)
                             if self.pending(source, rows, p)]
            cands = with_work
        if not cands:
            raise RuntimeError(
                f"no policy for source {source!r} has a usable endpoint "
                "with work left (fail-closed)")
        total_share = sum(p.share for p in cands)
        total_kept = sum(self.state.kept_by_policy.get((source, p.id), 0)
                         for p in cands) + 1
        def deficit(p: Policy) -> float:
            return (p.share / total_share * total_kept
                    - self.state.kept_by_policy.get((source, p.id), 0))
        return max(cands, key=lambda p: (deficit(p), p.share, p.id))

    # -- yield tracking ----------------------------------------------------------

    def record_batch_yield(self, source: str, kept_turns: int) -> None:
        if kept_turns > 0:
            self._zero_streak.pop(source, None)
            return
        streak = self._zero_streak.get(source, 0) + 1
        self._zero_streak[source] = streak
        if streak >= ZERO_YIELD_STRIKES:
            self._cooldown_until[source] = time.time() + ZERO_YIELD_COOLDOWN_S
            self._zero_streak.pop(source, None)
            log.warning("source %s: %d consecutive zero-yield batches; "
                        "cooling down %ds", source, streak,
                        ZERO_YIELD_COOLDOWN_S)

    def snapshot(self) -> dict:
        return {
            "targets": self.targets,
            "kept_by_source": dict(self.state.kept_by_source),
            "cooldowns": {n: round(t - time.time())
                          for n, t in self._cooldown_until.items()
                          if t > time.time()},
        }
