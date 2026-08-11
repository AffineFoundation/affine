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
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from rollouts.registry import Registry
from rollouts.schema import Policy

log = logging.getLogger("rollouts.scheduler")

ZERO_YIELD_STRIKES = 3        # consecutive zero-kept batches -> cooldown
ZERO_YIELD_COOLDOWN_S = 4 * 3600


class UnifiedState:
    """Append-only jsonl keyed (source, uid): outcome + kept-turn counts.
    A task recorded here is never attempted again (restart-safe; rows are
    written only after its batch's traces were parsed)."""

    def __init__(self, path: Path):
        self.path = path
        self.done: dict[str, set[str]] = {}
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
        self.done.setdefault(source, set()).add(uid)
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

    def done_for(self, source: str) -> set[str]:
        return self.done.get(source, set())


class Scheduler:
    def __init__(self, registry: Registry, state: UnifiedState,
                 env: dict | None = None):
        self.registry = registry
        self.state = state
        self.env = env if env is not None else dict(os.environ)
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

    def pick_policy(self, source: str) -> Policy:
        cands = self._usable_policies(source)
        if not cands:
            raise RuntimeError(
                f"no policy for source {source!r} has a usable endpoint "
                "(fail-closed)")
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
