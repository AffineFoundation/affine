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

Seats (2026-09-10): "done" is per (source, SEAT), not per source. Each
king is its own seat, `king:<served model>`, so the king replays tasks the
teacher already solved — its failures on the teacher's tasks are the whole
point of the king seat (DAgger) — and a newly crowned king starts over.
Before this, the king could only pick tasks the teacher had not reached
yet: every exhausted pool (terminal, math, tool_use, nl2repo) was closed to
it and king failures came from three coding sources only.

Teacher seats per (model, harness) (2026-09-11): a non-king policy's seat
is `<served model>:<harness>`. Before, every non-king policy shared ONE
seat, so a task rolled once by any of them (GLM-era `glm_textbased`, or the
teacher under mini-swe) was closed to every other teacher harness for
good: `teacher_terminus` / `teacher_kimi` / `teacher_hermes` had never run
a single task, terminal_bench_2 and nl2repobench had only GLM rollouts,
and 1,848 math problems only a GLM `boxed` answer — while the king seat
replays all of them on every harness. Now each teacher harness makes its
own pass over each source, the king's tasks first (`pending`), so every
(env, harness) the king runs gets a teacher baseline on the same tasks.

King share (2026-09-13): while the king seat is served, at least
KING_BATCH_SHARE of the picks are KING cycles — the source is chosen among
the sources that have a warm king policy with work left, ranked by the
king seat's OWN relative shortfall, and the policy pick is restricted to
king policies. Without this the king starved: the source pick ranks by
relative shortfall across ALL kept turns, so the fifteen teacher-only env
wave-2/3 sources (shortfall ~1.0, nothing generated yet) outranked every
source that carries a king policy (best 0.85), and between 2026-09-12
13:24 UTC and 2026-09-13 05:48 UTC the three pods ran ~200 teacher cycles
and zero king batches while reign 12 sat published on every pod.
"""

from __future__ import annotations

import calendar
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
KING_SEAT_PREFIX = "king:"
# Guaranteed fraction of picks that go to the king seat while it is served
# (ROLLOUTS_KING_BATCH_SHARE overrides; 0 disables the guarantee).
# 0.4 -> 0.5 on 2026-09-16 (phase 10, Jacob: "sample more from steps where the
# teacher stops but the king doesn't"): king_done needs ~120 and
# king_divergence ~340 more states, both come from king rollouts on tasks the
# teacher solved (see UnifiedState.teacher_solved / Scheduler.pending).
KING_BATCH_SHARE = float(os.environ.get("ROLLOUTS_KING_BATCH_SHARE", "0.5"))
# Window over which a source's king_rollouts_per_hour floor is measured. A
# batch is 16-48 rollouts, larger than any floor, so a 1 h window would fire
# one batch every hour whatever the floor says; over 6 h the floor sets how
# many batches land per 6 h (floor x 6 / batch size).
KING_FLOOR_WINDOW_S = 6 * 3600.0


def is_king_policy(policy_id: str) -> bool:
    return policy_id.startswith(KING_POLICY_PREFIX)


def seat_of(policy_id: str, model: str, harness: str = "") -> str:
    """The seat a (policy, served model, harness) plays. `model` is the
    endpoint's model name (the `provider` label on state rows is
    `<endpoint>/<model>`). Kings: one seat per served model, all harnesses
    (one pass over each task). Everyone else: one seat per (model,
    harness), so each teacher harness passes over every task once."""
    if policy_id.startswith(KING_POLICY_PREFIX):
        return f"{KING_SEAT_PREFIX}{model or 'unknown'}"
    return f"{model or 'unknown'}:{harness or 'unknown'}"


def is_king_seat(seat: str) -> bool:
    return seat.startswith(KING_SEAT_PREFIX)


def _row_ts(at) -> float | None:
    """State rows stamp `at` as ISO-8601 Z (mark); older rows may carry a
    float. None when unparsable."""
    if isinstance(at, (int, float)):
        return float(at)
    try:
        return calendar.timegm(time.strptime(str(at), "%Y-%m-%dT%H:%M:%SZ"))
    except (TypeError, ValueError):
        return None


def policy_seat(policy: Policy, env: dict) -> str:
    eps = policy.available_endpoints(env)
    return seat_of(policy.id, eps[0].model if eps else "", policy.harness)


class UnifiedState:
    """Append-only jsonl keyed (source, uid): outcome + kept-turn counts.
    A task recorded here is never attempted again BY THE SAME SEAT
    (restart-safe; rows are written only after its batch's traces were
    parsed)."""

    def __init__(self, path: Path, harness_of: dict[str, str] | None = None):
        # policy id -> harness, to place legacy rows (no harness column) in
        # their (model, harness) seat; a retired policy id lands in
        # `<model>:unknown`, which no live policy claims.
        self.harness_of = harness_of or {}
        self.path = path
        self.done: dict[tuple[str, str], set[str]] = {}
        # (source, harness) -> tasks some KING already rolled under that
        # harness: the teacher's first pick, so king failures get a teacher
        # baseline on the same task and harness.
        self.king_done: dict[tuple[str, str], set[str]] = {}
        self.kept_by_source: dict[str, int] = {}
        self.kept_by_policy: dict[tuple[str, str], int] = {}
        # (source, seat) -> kept turns: the king ranking reads the CURRENT
        # king's own turns (a new king starts every source from zero; before
        # this, kept_by_policy carried every earlier king's turns and a
        # source like affine_agent with 16k reign-11 turns was never picked
        # for reign 12). (source, seat) -> row timestamps of king rollouts,
        # for the per-source king floor (king_rollouts_per_hour).
        self.kept_by_seat: dict[tuple[str, str], int] = {}
        self.king_times: dict[tuple[str, str], list[float]] = {}
        # source -> tasks some TEACHER-side seat solved (outcome resolved):
        # the king's first picks (phase 10, 2026-09-16). A king that keeps
        # going after the point where the teacher stopped, or calls a tool
        # where the teacher answered, or fails where the teacher solved, is
        # the stop-state material (king_done / king_divergence / king_tooluse)
        # and it exists only on tasks the teacher finished.
        self.teacher_solved: dict[str, set[str]] = {}
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
        pid = str(rec.get("policy_id") or "")
        harness = str(rec.get("harness") or self.harness_of.get(pid, ""))
        seat = seat_of(pid, model, harness)
        self.done.setdefault((source, seat), set()).add(uid)
        if is_king_seat(seat) and harness:
            self.king_done.setdefault((source, harness), set()).add(uid)
        elif not is_king_seat(seat) and rec.get("outcome") == "resolved" \
                and not pid.startswith("backfill_"):
            self.teacher_solved.setdefault(source, set()).add(uid)
        n = int(rec.get("n_turns") or 0)
        self.kept_by_source[source] = self.kept_by_source.get(source, 0) + n
        pid = rec.get("policy_id") or ""
        key = (source, pid)
        self.kept_by_policy[key] = self.kept_by_policy.get(key, 0) + n
        self.kept_by_seat[(source, seat)] = self.kept_by_seat.get((source, seat), 0) + n
        if is_king_seat(seat):
            ts = _row_ts(rec.get("at"))
            if ts is not None:
                self.king_times.setdefault((source, seat), []).append(ts)

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

    def done_for(self, source: str, seat: str | None = None) -> set[str]:
        """Tasks `seat` has processed; seat None = the union over every
        non-king seat (the source's tasks some teacher-side policy rolled)."""
        if seat is not None:
            return self.done.get((source, seat), set())
        out: set[str] = set()
        for (src, s), uids in self.done.items():
            if src == source and not is_king_seat(s):
                out |= uids
        return out


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
        self.king_share = KING_BATCH_SHARE
        # Picks made by this supervisor (in-memory; a restart starts the
        # ratio over). `king_cycle` is the verdict of the last pick_source:
        # True = this cycle belongs to the king seat (pick_policy honours it).
        self.picks_total = 0
        self.picks_king = 0
        self.king_cycle = False

    # -- source pick -------------------------------------------------------------

    def eligible(self, remaining: dict[str, int]) -> list[str]:
        now = time.time()
        return [name for name in self.registry.sources
                if remaining.get(name, 0) > 0
                and self._cooldown_until.get(name, 0.0) <= now
                and self._usable_policies(name)]

    def _king_eligible(self, king_remaining: dict[str, int]) -> list[str]:
        """Sources with a WARM king policy and work left for the king seat
        (`king_remaining` = per-source count from `remaining(..., king_only=
        True)`). Share-0 sources (retired pools, public benches) are never
        king cycles."""
        now = time.time()
        return [name for name in self.registry.sources
                if king_remaining.get(name, 0) > 0
                and self.targets.get(name, 0.0) > 0
                and self._cooldown_until.get(name, 0.0) <= now
                and self._warm_king_policies(name)]

    def _king_due(self) -> bool:
        """Would handing this pick to the teacher drop the king's share of
        picks below the target? With share 0.4 the sequence is K T K T T
        K T K T T ... = 40 % king cycles exactly."""
        if self.king_share <= 0:
            return False
        return self.picks_king / (self.picks_total + 1) < self.king_share

    def pick_source(self, remaining: dict[str, int],
                    king_remaining: dict[str, int] | None = None) -> str | None:
        """The source furthest below its target, measured RELATIVE to the
        target: shortfall = 1 - kept / expected, where expected is the
        source's share of all kept turns so far. 1.0 = nothing generated
        yet, 0 = on target, negative = over.

        Ranking by absolute deficit (until 2026-09-12) starved every small
        source: the eight `general` sources at targets 0.005-0.009 sat at
        +2-4k turns of deficit with zero batches while terminal_bench_2 at
        +25k (already 18 % of its target) won every pick — a source's
        deficit in turns scales with its target, so the big groups always
        outranked the small ones. Absolute deficit stays the tie-break.

        King cycles (see the module docstring): when the king's share of
        picks is due and some source has a warm king policy with work
        left, the pick is made among THOSE sources only, ranked by the
        same shortfall on the king seat's own kept turns, and
        `self.king_cycle` is set so pick_policy stays on the king. The
        teacher's ranking is untouched on its own cycles."""
        self.king_cycle = False
        king_cands = (self._king_eligible(king_remaining)
                      if king_remaining is not None else [])
        # Per-source king floors come first and may take a cycle beyond the
        # 40 % share (the share is a floor, not a cap): the king_tooluse
        # feeders must see the king several times a day whatever their
        # turn target says.
        floor_due = self._king_floor_due(king_cands) if king_cands else []
        if floor_due:
            # Not counted in picks_king / picks_total: a floor cycle sits on
            # top of the 40 % share. Counting it (2026-09-15 morning) made a
            # new king's six floor batches push the ratio to 47 %, after
            # which the picker ran ~20 teacher cycles in a row to bring it
            # back — two hours with no king batch on any pod.
            self.king_cycle = True
            return floor_due[0]
        if king_cands and self._king_due():
            return self._king_pick(king_cands)
        cands = self.eligible(remaining)
        if not cands:
            # `remaining` is the teacher's work when run.py passes it
            # seat-scoped: nothing left for the teacher, so the king takes
            # the cycle if it has any (its share is a floor, not a cap).
            if king_cands:
                return self._king_pick(king_cands)
            return None
        self.picks_total += 1
        return self._rank_pick(cands, self._teacher_kept)

    def _king_pick(self, king_cands: list[str]) -> str:
        self.king_cycle = True
        self.picks_total += 1
        self.picks_king += 1
        return self._rank_pick(king_cands, self._king_kept)

    def _rank_pick(self, cands: list[str], kept_of) -> str:
        total_target = sum(self.targets[n] for n in cands)
        total_kept = sum(kept_of(n) for n in cands) + 1
        def rank(name: str) -> tuple[float, float, float, str]:
            expected = self.targets[name] / total_target * total_kept
            kept = kept_of(name)
            deficit = expected - kept
            shortfall = deficit / expected if expected > 0 else 0.0
            return (shortfall, deficit, self.targets[name], name)
        return max(cands, key=rank)

    def king_seat(self) -> str | None:
        """The seat the served king plays right now (None while unserved)."""
        for p in self.registry.policies.values():
            if is_king_policy(p.id) and p.available_endpoints(self.env):
                return policy_seat(p, self.env)
        return None

    def _king_kept(self, source: str) -> int:
        """The CURRENT king's kept turns on the source (a new king starts
        every source over; earlier kings' turns do not count against it)."""
        seat = self.king_seat()
        return self.state.kept_by_seat.get((source, seat), 0) if seat else 0

    def king_rate(self, source: str, window_s: float = 3600.0) -> float:
        """The current king's rollouts on `source` in the last `window_s`,
        per hour."""
        seat = self.king_seat()
        if not seat:
            return 0.0
        cutoff = time.time() - window_s
        times = self.state.king_times.get((source, seat), ())
        return sum(1 for t in times if t >= cutoff) * 3600.0 / window_s

    def _king_floor_due(self, king_cands: list[str]) -> list[str]:
        """Feeder sources (king_rollouts_per_hour > 0) whose current king
        rate is below their floor, most-behind first (relative gap)."""
        due = []
        for name in king_cands:
            floor = self.registry.sources[name].king_rollouts_per_hour
            if floor <= 0:
                continue
            rate = self.king_rate(name, KING_FLOOR_WINDOW_S)
            if rate < floor:
                due.append(((floor - rate) / floor, name))
        return [n for _, n in sorted(due, reverse=True)]

    def _teacher_kept(self, source: str) -> int:
        """Kept turns of the non-king seats: the teacher's own shortfall
        ranks teacher cycles (king turns on a source are not teacher supply).
        Legacy rows without a policy id count as teacher."""
        return sum(n for (src, pid), n in self.state.kept_by_policy.items()
                   if src == source and not is_king_policy(pid))

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

    def _warm_king_policies(self, source: str) -> list[Policy]:
        """King policies of the source whose endpoint is keyed (the seat is
        served) and not cooling. A struck king box makes every source
        king-ineligible for the cooldown, so a dark seat costs teacher
        cycles, never a spin."""
        cands = [p for p in self._usable_policies(source)
                 if is_king_policy(p.id)]
        if self.health is None:
            return cands
        return [p for p in cands if not self.health.all_cooling(p, self.env)]

    # -- seats: per-(source, seat) work ------------------------------------------

    def pending(self, source: str, rows: list[dict], policy: Policy) -> list[dict]:
        """Pool rows this policy's seat has not processed, in pool order —
        except that a teacher-side policy takes the tasks a king already
        rolled under the same harness FIRST (the paired baseline), then the
        rest of the pool."""
        done = self.state.done_for(source, policy_seat(policy, self.env))
        todo = [r for r in rows if r["uid"] not in done]
        if policy.id.startswith(KING_POLICY_PREFIX):
            # Teacher-solved tasks first (phase 10): the king's stop-state
            # failures need a teacher that finished the same task.
            solved = self.state.teacher_solved.get(source, set())
            if not solved:
                return todo
            return ([r for r in todo if r["uid"] in solved]
                    + [r for r in todo if r["uid"] not in solved])
        king_first = self.state.king_done.get((source, policy.harness), set())
        if not king_first:
            return todo
        return ([r for r in todo if r["uid"] in king_first]
                + [r for r in todo if r["uid"] not in king_first])

    def remaining(self, source: str, rows: list[dict],
                  king_only: bool = False, teacher_only: bool = False) -> int:
        """Tasks some usable policy of this source still has to run — the
        union over seats (a task the teacher finished is still work for the
        king). `king_only` counts the king seat's work alone, `teacher_only`
        the non-king seats' work alone."""
        pols = self._usable_policies(source)
        if king_only:
            pols = [p for p in pols if is_king_policy(p.id)]
        elif teacher_only:
            pols = [p for p in pols if not is_king_policy(p.id)]
        seats = {policy_seat(p, self.env) for p in pols}
        if not seats:
            return 0
        dones = [self.state.done_for(source, s) for s in seats]
        return sum(1 for r in rows if any(r["uid"] not in d for d in dones))

    def pick_policy(self, source: str, rows: list[dict] | None = None,
                    king_only: bool = False, teacher_only: bool = False) -> Policy:
        """Largest per-policy deficit among the source's healthy policies
        with work. Cycles are seat-scoped: `king_only` (a king cycle) keeps
        the pick on the king policies, `teacher_only` (a teacher cycle) on
        the non-king ones; if the seat has nothing runnable here the pick
        falls back to every policy rather than idling the cycle.

        Why teacher cycles are scoped too (2026-09-13 15:00 UTC): without
        it the per-policy deficit picked the KING on teacher cycles as
        well — a freshly added king policy has zero kept turns next to a
        teacher policy with thousands, so it out-deficits the teacher on
        every source until it holds 60 % of that source's turns. With king
        policies on every source since 07:20 UTC the three pods ran ~0
        teacher turns/h for eight hours (674 / 742 / 506 king turns/h)."""
        cands = self._healthy_policies(source)
        if king_only or teacher_only:
            mine = [p for p in cands if is_king_policy(p.id) == king_only]
            if rows is not None:
                mine = [p for p in mine if self.pending(source, rows, p)]
            if mine:
                cands = mine
            else:
                log.warning("%s cycle on %s: no %s policy can run; falling "
                            "back to any policy", "king" if king_only else "teacher",
                            source, "king" if king_only else "teacher")
                king_only = teacher_only = False
        if rows is not None and not (king_only or teacher_only):
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
            "picks": {"total": self.picks_total, "king": self.picks_king,
                      "king_share_target": self.king_share},
            "cooldowns": {n: round(t - time.time())
                          for n, t in self._cooldown_until.items()
                          if t > time.time()},
        }
