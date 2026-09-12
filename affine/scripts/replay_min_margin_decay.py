"""Replay stored verdicts under the decaying crown margin (staged 2026-09-12).

Reads the validator's ``history.jsonl`` and asks, for one or more decay
settings: which of the stored duels would have crowned, at what z and
margin, how far into the δ cycle, and how many kings per day that makes.

    source .venv/bin/activate
    python affine/scripts/replay_min_margin_decay.py affine/state/history.jsonl
    python affine/scripts/replay_min_margin_decay.py history.jsonl \
        --decay-hours 24 48 --shape linear exponential --min-z 0 2.5 3.0 --json out.json

What the replay is and is not
-----------------------------
* It uses each verdict's STORED margin, SE and z (the pooled numbers when
  the near-miss rule pooled) and its stored gate outcome (thought floor, B
  license). Nothing is re-scored.
* The clock is the verdict timestamp converted to blocks at 12 s/block —
  the stamp the live rule would read (``blocks_since_crown``) does not exist
  in pre-2026-09-12 rows, so this is the closest deterministic proxy.
* "chained" mode is a COUNTERFACTUAL: when the replay crowns a challenger
  the real board did not, the following stored margins were measured
  against the REAL king of the day, not the replay's king. Treat chained
  counts as "how often the bar would have been cleared", not as a
  simulation of the board that would have existed. "actual-clock" mode
  restarts the cycle only on the crowns that really happened and counts
  how many stored verdicts would have passed the decayed bar against the
  king they actually faced — a lower-noise reading of the same question.
* Verdicts without a margin (protocol-probe rejections, unservable
  checkpoints, failures) cannot crown under any δ and are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affine.score import (  # noqa: E402
    DEFAULT_K_SIGMA,
    DEFAULT_MIN_MARGIN,
    SECONDS_PER_BLOCK,
    MarginSchedule,
)

# A challenger whose paired SE is this small is scoring almost the same
# per-turn numbers as the king on every turn: the ε-copy signature (AGENTS
# §2: copy SE ≈ 0.0003 vs ≈ 0.0007 for a distinct model). Telemetry only.
NEAR_COPY_SE = 0.0005
# Below this |z| a crown is a statistical tie at the 2σ scale.
TIE_Z = 2.5
GATE_REASONS = ("thought_too_short", "causality_fail")


@dataclass
class Duel:
    """One stored verdict, reduced to what the crown test needs."""
    challenge_id: str
    at: datetime
    hotkey: str
    margin: float
    se: float
    z: float
    n_paired_turns: int
    gate_blocked: bool
    actual_win: bool
    actual_reign: int | None = None

    @property
    def near_copy(self) -> bool:
        return self.se < NEAR_COPY_SE


@dataclass
class Crown:
    challenge_id: str
    at: datetime
    hotkey: str
    margin: float
    se: float
    z: float
    delta: float
    hours_since_crown: float
    peak_before: float
    peak_after: float
    near_copy: bool
    actual_win: bool


@dataclass
class Replay:
    label: str
    schedule: MarginSchedule
    k_sigma: float
    min_z: float
    mode: str
    crowns: list[Crown] = field(default_factory=list)
    n_duels: int = 0
    span_days: float = 0.0
    n_delta_binding: int = 0     # duels where δ_eff > k_sigma·SE (δ decided)
    n_blocked_by_min_z: int = 0  # cleared the bar, lost to min_z
    n_actual_lost: int = 0       # real crowns the replay would not award

    @property
    def kings_per_day(self) -> float:
        return len(self.crowns) / self.span_days if self.span_days else 0.0

    @property
    def n_ties(self) -> int:
        return sum(1 for c in self.crowns if c.z < TIE_Z)

    @property
    def n_near_copies(self) -> int:
        return sum(1 for c in self.crowns if c.near_copy)

    @property
    def n_new(self) -> int:
        """Crowns the real board did not award."""
        return sum(1 for c in self.crowns if not c.actual_win)

    def summary(self) -> dict:
        zs = [c.z for c in self.crowns]
        ms = [c.margin for c in self.crowns]
        return {
            "label": self.label, "mode": self.mode,
            "min_margin_mode": self.schedule.mode,
            "decay_hours": self.schedule.decay_hours,
            "shape": self.schedule.shape,
            "floor": self.schedule.floor, "peak_cap": self.schedule.peak_cap,
            "double_on_crown": self.schedule.double_on_crown,
            "min_z": self.min_z,
            "n_duels": self.n_duels, "span_days": round(self.span_days, 2),
            "n_crowns": len(self.crowns),
            "kings_per_day": round(self.kings_per_day, 2),
            "n_new_vs_actual": self.n_new,
            "n_actual_lost": self.n_actual_lost,
            "n_ties_z_below_2p5": self.n_ties,
            "n_near_copy_crowns": self.n_near_copies,
            "n_blocked_by_min_z": self.n_blocked_by_min_z,
            "n_delta_binding": self.n_delta_binding,
            "z_median": round(st.median(zs), 2) if zs else None,
            "z_min": round(min(zs), 2) if zs else None,
            "margin_median": round(st.median(ms), 5) if ms else None,
            "margin_min": round(min(ms), 5) if ms else None,
        }


def load_duels(path: Path) -> list[Duel]:
    """Every stored duel with a margin, oldest first."""
    out: list[Duel] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") not in ("verdict", "crowned"):
                continue
            v = row.get("verdict")
            if not isinstance(v, dict) or v.get("margin") is None or v.get("se") is None:
                continue
            se = float(v["se"])
            if not math.isfinite(se) or se <= 0:
                continue
            out.append(Duel(
                challenge_id=str(row.get("challenge_id", "?")),
                at=datetime.fromisoformat(row["at"]).astimezone(timezone.utc),
                hotkey=str(row.get("hotkey", "")),
                margin=float(v["margin"]), se=se,
                z=float(v.get("z") if v.get("z") is not None else v["margin"] / se),
                n_paired_turns=int(v.get("n_paired_turns") or 0),
                gate_blocked=v.get("rejection_reason") in GATE_REASONS,
                actual_win=bool(row.get("event") == "crowned"),
                actual_reign=(int(row["reign_number"])
                              if row.get("reign_number") is not None else None),
            ))
    out.sort(key=lambda d: d.at)
    return out


def seed_time(path: Path) -> datetime | None:
    """When the genesis king was seeded (the first cycle's origin)."""
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "crowned" and row.get("challenge_id") == "seed":
                return datetime.fromisoformat(row["at"]).astimezone(timezone.utc)
    return None


def blocks_between(t0: datetime, t1: datetime,
                   seconds_per_block: float = SECONDS_PER_BLOCK) -> int:
    return int((t1 - t0).total_seconds() // seconds_per_block)


def run_replay(duels: list[Duel], schedule: MarginSchedule, *,
               k_sigma: float = DEFAULT_K_SIGMA, min_z: float = 0.0,
               mode: str = "chained", origin: datetime | None = None,
               label: str = "") -> Replay:
    """Walk the duels in order and apply the crown test with δ from the
    schedule. ``mode`` "chained": a replay crown restarts the cycle.
    "actual-clock": only the crowns that really happened restart it (the
    peak still follows the schedule's doubling rule at those crowns)."""
    if mode not in ("chained", "actual-clock"):
        raise ValueError(f"mode must be chained or actual-clock, got {mode!r}")
    rep = Replay(label=label or f"{schedule.mode}/{schedule.shape}/"
                 f"{schedule.decay_hours:g}h/min_z={min_z:g}/{mode}",
                 schedule=schedule, k_sigma=k_sigma, min_z=min_z, mode=mode)
    if not duels:
        return rep
    crown_at = origin or duels[0].at
    peak: float | None = None  # None = the cap (first cycle)
    for d in duels:
        since = blocks_between(crown_at, d.at, schedule.seconds_per_block)
        delta = schedule.effective(peak, since)
        bar = max(k_sigma * d.se, delta)
        if delta > k_sigma * d.se:
            rep.n_delta_binding += 1
        cleared = d.margin > bar and not d.gate_blocked
        if cleared and min_z > 0 and d.z < min_z:
            rep.n_blocked_by_min_z += 1
            cleared = False
        rep.n_duels += 1
        if d.actual_win and not cleared:
            rep.n_actual_lost += 1
        restart = cleared if mode == "chained" else d.actual_win
        if cleared:
            rep.crowns.append(Crown(
                challenge_id=d.challenge_id, at=d.at, hotkey=d.hotkey,
                margin=d.margin, se=d.se, z=d.z, delta=delta,
                hours_since_crown=since * schedule.seconds_per_block / 3600.0,
                peak_before=schedule.peak_cap if peak is None else peak,
                peak_after=schedule.next_peak(delta),
                near_copy=d.near_copy, actual_win=d.actual_win))
        if restart:
            peak = schedule.next_peak(delta)
            crown_at = d.at
    rep.span_days = (duels[-1].at - (origin or duels[0].at)).total_seconds() / 86400.0
    return rep


def format_table(replays: list[Replay]) -> str:
    cols = [("label", 44), ("n_crowns", 8), ("kings/day", 9), ("new", 4),
            ("lost", 4), ("ties<2.5", 8), ("copies", 6), ("min_z blk", 9),
            ("z_med", 6), ("z_min", 6), ("m_med", 8), ("m_min", 8)]
    head = " ".join(f"{name:<{w}}" for name, w in cols)
    lines = [head, "-" * len(head)]
    for r in replays:
        s = r.summary()
        vals = [s["label"], s["n_crowns"], s["kings_per_day"], s["n_new_vs_actual"],
                s["n_actual_lost"], s["n_ties_z_below_2p5"], s["n_near_copy_crowns"],
                s["n_blocked_by_min_z"], s["z_median"], s["z_min"],
                s["margin_median"], s["margin_min"]]
        lines.append(" ".join(f"{str(v):<{w}}" for v, (_, w) in zip(vals, cols)))
    return "\n".join(lines)


def format_crowns(rep: Replay) -> str:
    lines = [f"== {rep.label}: {len(rep.crowns)} crowns over {rep.span_days:.1f} days"]
    for c in rep.crowns:
        flags = ("ACTUAL" if c.actual_win else "new   ") + (" COPY?" if c.near_copy else "")
        lines.append(
            f"  {c.at.strftime('%m-%d %H:%M')} {c.challenge_id} {c.hotkey[:8]} "
            f"m={c.margin:+.5f} se={c.se:.5f} z={c.z:5.2f} δ={c.delta:.5f} "
            f"+{c.hours_since_crown:5.1f}h peak {c.peak_before:.4f}→{c.peak_after:.4f} {flags}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("history", type=Path)
    ap.add_argument("--decay-hours", type=float, nargs="+", default=[24.0, 48.0])
    ap.add_argument("--shape", nargs="+", default=["linear", "exponential"])
    ap.add_argument("--min-z", type=float, nargs="+", default=[0.0, 2.5, 3.0])
    ap.add_argument("--floor", type=float, default=0.0001)
    ap.add_argument("--peak-cap", type=float, default=DEFAULT_MIN_MARGIN)
    ap.add_argument("--no-double", action="store_true",
                    help="reset to the cap at every crown instead of doubling")
    ap.add_argument("--k-sigma", type=float, default=DEFAULT_K_SIGMA)
    ap.add_argument("--mode", nargs="+", default=["chained", "actual-clock"])
    ap.add_argument("--crowns", action="store_true", help="list every replay crown")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    duels = load_duels(args.history)
    origin = seed_time(args.history)
    print(f"{len(duels)} stored duels with a margin, "
          f"{duels[0].at:%Y-%m-%d} → {duels[-1].at:%Y-%m-%d}; "
          f"{sum(d.actual_win for d in duels)} actual crowns; "
          f"{sum(d.near_copy for d in duels)} near-copy signatures (SE < {NEAR_COPY_SE})")
    replays: list[Replay] = []
    baseline = MarginSchedule(min_margin=args.peak_cap, mode="fixed")
    for mz in args.min_z:
        replays.append(run_replay(duels, baseline, k_sigma=args.k_sigma,
                                  min_z=mz, mode="chained", origin=origin,
                                  label=f"fixed δ={args.peak_cap:g} min_z={mz:g}"))
    for mode in args.mode:
        for hours in args.decay_hours:
            for shape in args.shape:
                sched = MarginSchedule(
                    min_margin=args.peak_cap, mode="decay",
                    peak_cap=args.peak_cap, floor=args.floor,
                    decay_hours=hours, shape=shape,
                    double_on_crown=not args.no_double)
                for mz in args.min_z:
                    replays.append(run_replay(
                        duels, sched, k_sigma=args.k_sigma, min_z=mz,
                        mode=mode, origin=origin,
                        label=f"decay {hours:g}h {shape} min_z={mz:g} [{mode}]"))
    print()
    print(format_table(replays))
    if args.crowns:
        for r in replays:
            print()
            print(format_crowns(r))
    if args.json:
        args.json.write_text(json.dumps({
            "replays": [{**r.summary(),
                         "crowns": [{**c.__dict__, "at": c.at.isoformat()}
                                    for c in r.crowns]} for r in replays],
        }, indent=1, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
