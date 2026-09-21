"""Replay the stored duel history under the window-best crown rule.

    python affine/scripts/replay_window_best.py affine/state/history.jsonl \
        [--window-blocks 3600] [--anchor-block 9052470 --anchor-time 2026-09-12T15:45:00Z] \
        [--no-hotkey-dedupe] [--json out.json]

Rule replayed (crown_mode = "window_best"): fixed windows of W blocks aligned
on the block number; every scored verdict inside a window is a candidate
when its margin is finite, > 0 and no gate rejected it; one candidate per
hotkey (its best margin); the largest margin wins the window; the winner
must confirm on a fresh slice (pooled margin over both slices > 0), else
the next-best is tried, up to crown_confirm_max = 2.

What the replay can and cannot know — stated up front:
  * Stored verdicts have no block stamp. Wall time is mapped to blocks at
    12 s/block from an anchor (block, time) pair, and each duel's DISPATCH
    time is `at − duration_s`; the window is the dispatch window, as live.
  * Every stored margin was measured against the REAL king of that hour.
    The rule freezes the king per window, so a window's later margins would
    have been measured against the window king — which the real board may
    have replaced mid-window. Read the counts as "how often the rule would
    have picked a winner", not as the board that would have existed.
  * Confirmation: where the near-miss rule already drew a second slice
    (`verdict.near_miss.slices`, since 2026-09-11) that slice IS a fresh
    disjoint draw — the replay uses it: confirmation passes iff the pooled
    margin over slice 0 + slice 1 is > 0. For every other winner there is
    no second draw. The replay then uses the directive's rule: z ≥ 1 →
    assumed to confirm; 0 < z < 1 ("near-zero margin") → a 50 % coin
    (seeded, reported both as the seeded outcome and as the expectation).
    The analytic probability Φ(z) under "true margin = observed" is printed
    next to it; it is optimistic (winner's curse) and is not used.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SECONDS_PER_BLOCK = 12.0
CONFIRM_MAX = 2
TIE_Z = 2.0
COIN_Z = 1.0


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def load_duels(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("event") not in ("verdict", "crowned"):
                continue
            v = r.get("verdict") or {}
            if v.get("via") == "window_best" or v.get("verdict") == "crown_earlier":
                continue
            m = v.get("margin")
            slices = ((v.get("near_miss") or {}).get("slices") or [])
            out.append({
                "challenge_id": r.get("challenge_id"), "hotkey": r.get("hotkey"),
                "at": r.get("at"), "duration_s": float(r.get("duration_s") or 0.0),
                "margin": float(m) if isinstance(m, (int, float)) else None,
                "se": v.get("se"), "z": v.get("z"),
                "rejection_reason": v.get("rejection_reason"),
                "real_crown": r.get("event") == "crowned",
                "reign": r.get("reign_number"),
                "slices": [{"margin": s.get("margin"), "n": s.get("n_paired_turns") or s.get("n"),
                            "se": s.get("se")} for s in slices],
            })
    return out


def block_of(ts: str, duration_s: float, anchor_block: int, anchor_time: datetime) -> int:
    t = _iso(ts).timestamp() - duration_s
    return int(round(anchor_block + (t - anchor_time.timestamp()) / SECONDS_PER_BLOCK))


def confirm_outcome(d: dict, rng: random.Random) -> tuple[bool, str, float]:
    """(passed, how, expectation) for one winner."""
    if len(d["slices"]) >= 2 and all(
            isinstance(s.get("margin"), (int, float)) and s.get("n") for s in d["slices"][:2]):
        s0, s1 = d["slices"][0], d["slices"][1]
        pooled = (s0["margin"] * s0["n"] + s1["margin"] * s1["n"]) / (s0["n"] + s1["n"])
        return pooled > 0, "second_slice_record", 1.0 if pooled > 0 else 0.0
    z = float(d["z"] or 0.0)
    if z >= COIN_Z:
        return True, "assumed_pass(z>=1)", 1.0
    return rng.random() < 0.5, "coin(z<1)", 0.5


def replay(duels: list[dict], window_blocks: int, anchor_block: int,
           anchor_time: datetime, dedupe_hotkey: bool, seed: int = 0) -> dict:
    rng = random.Random(seed)
    by_window: dict[int, list[dict]] = defaultdict(list)
    for d in duels:
        if d["margin"] is None or not math.isfinite(d["margin"]):
            continue
        b = block_of(d["at"], d["duration_s"], anchor_block, anchor_time)
        d = dict(d, block=b, window_id=b // window_blocks)
        by_window[d["window_id"]].append(d)
    if not by_window:
        return {"windows": [], "summary": {}}
    first, last = min(by_window), max(by_window)
    windows = []
    for wid in range(first, last + 1):
        vs = by_window.get(wid, [])
        cands = [d for d in vs if d["margin"] > 0 and not d.get("rejection_reason")]
        cands.sort(key=lambda d: (-d["margin"], -(d["z"] or -1e9), d["challenge_id"]))
        if dedupe_hotkey:
            seen, kept = set(), []
            for d in cands:
                if d["hotkey"] in seen:
                    continue
                seen.add(d["hotkey"]); kept.append(d)
            cands = kept
        confirmations = []
        winner = None
        exp_crown = 0.0
        remaining = 1.0
        for d in cands[:CONFIRM_MAX]:
            passed, how, p = confirm_outcome(d, rng)
            confirmations.append({"challenge_id": d["challenge_id"], "margin": d["margin"],
                                  "z": d["z"], "passed": passed, "how": how, "p_pass": p})
            exp_crown += remaining * p
            remaining *= (1 - p)
            if passed and winner is None:
                winner = d
        windows.append({
            "window_id": wid,
            "start_utc": datetime.fromtimestamp(
                anchor_time.timestamp() + (wid * window_blocks - anchor_block) * SECONDS_PER_BLOCK,
                tz=timezone.utc).strftime("%m-%d %H:%M"),
            "n_verdicts": len(vs), "n_candidates": len(cands),
            "best": (cands[0]["challenge_id"] if cands else None),
            "best_margin": (cands[0]["margin"] if cands else None),
            "best_z": (cands[0]["z"] if cands else None),
            "winner": winner["challenge_id"] if winner else None,
            "winner_hotkey": winner["hotkey"][:8] if winner else None,
            "winner_margin": winner["margin"] if winner else None,
            "winner_z": winner["z"] if winner else None,
            "winner_real_crown": bool(winner and winner["real_crown"]),
            "confirmations": confirmations,
            "expected_crown_prob": exp_crown,
            "real_crowns_in_window": [d["challenge_id"] for d in vs if d["real_crown"]],
        })
    span_days = ((last + 1 - first) * window_blocks * SECONDS_PER_BLOCK) / 86400.0
    crowns = [w for w in windows if w["winner"]]
    no_conf = [w for w in windows if w["best"]]
    real = [d for d in duels if d["real_crown"]]
    summary = {
        "window_blocks": window_blocks, "window_hours": window_blocks * SECONDS_PER_BLOCK / 3600,
        "n_windows": len(windows), "span_days": round(span_days, 2),
        "windows_with_candidates": len(no_conf),
        "crowns_no_confirmation": len(no_conf),
        "crowns_with_confirmation_seeded": len(crowns),
        "crowns_with_confirmation_expected": round(sum(w["expected_crown_prob"] for w in windows), 2),
        "kings_per_day_no_confirmation": round(len(no_conf) / span_days, 2),
        "kings_per_day_with_confirmation": round(len(crowns) / span_days, 2),
        "ties_among_winners(z<2)": sum(1 for w in crowns if (w["winner_z"] or 0) < TIE_Z),
        "ties_among_best_no_confirmation(z<2)": sum(1 for w in no_conf if (w["best_z"] or 0) < TIE_Z),
        "winners_that_were_real_crowns": sum(1 for w in crowns if w["winner_real_crown"]),
        "real_crowns_in_history": len(real),
        "real_crowns_not_window_winners": [
            d["challenge_id"] for d in real
            if d["challenge_id"] not in {w["winner"] for w in crowns}],
        "confirmations_tried": sum(len(w["confirmations"]) for w in windows),
        "confirmations_from_second_slice_records": sum(
            1 for w in windows for c in w["confirmations"] if c["how"] == "second_slice_record"),
        "confirmations_coin(z<1)": sum(
            1 for w in windows for c in w["confirmations"] if c["how"].startswith("coin")),
        "confirmations_assumed(z>=1)": sum(
            1 for w in windows for c in w["confirmations"] if c["how"].startswith("assumed")),
        "confirmations_failed_seeded": sum(
            1 for w in windows for c in w["confirmations"] if not c["passed"]),
        "confirmations_expected_failures": round(sum(
            1 - c["p_pass"] for w in windows for c in w["confirmations"]), 2),
    }
    return {"windows": windows, "summary": summary}


def print_table(res: dict, only_with_candidates: bool = True) -> None:
    s = res["summary"]
    print(f"\n== window = {s['window_hours']:g} h ({s['window_blocks']} blocks), "
          f"{s['n_windows']} windows over {s['span_days']} days ==")
    print(f"{'win':>5} {'start(UTC)':>11} {'verd':>4} {'cand':>4} {'best':>10} {'margin':>8} "
          f"{'z':>5} {'winner':>10} {'w.z':>5} {'confirm':>26} {'real?':>5}")
    for w in res["windows"]:
        if only_with_candidates and not w["best"]:
            continue
        conf = "; ".join(f"{c['challenge_id'][-5:]}:{'P' if c['passed'] else 'F'}({c['how'].split('(')[0][:6]})"
                         for c in w["confirmations"]) or "-"
        print(f"{w['window_id']:>5} {w['start_utc']:>11} {w['n_verdicts']:>4} {w['n_candidates']:>4} "
              f"{(w['best'] or '-'):>10} {(w['best_margin'] or 0):>8.5f} {(w['best_z'] or 0):>5.2f} "
              f"{(w['winner'] or '-'):>10} {(w['winner_z'] or 0):>5.2f} {conf:>26} "
              f"{'yes' if w['winner_real_crown'] else ''}")
    print("summary:")
    for k, v in s.items():
        print(f"  {k}: {v}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("history", type=Path)
    ap.add_argument("--window-blocks", type=int, action="append",
                    help="window size in blocks (repeatable; default 3600 and 7200)")
    ap.add_argument("--anchor-block", type=int, default=9_052_470)
    ap.add_argument("--anchor-time", default="2026-09-12T15:45:00Z")
    ap.add_argument("--no-hotkey-dedupe", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-windows", action="store_true", help="print empty windows too")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    duels = load_duels(args.history)
    scored = [d for d in duels if d["margin"] is not None]
    print(f"{len(duels)} verdict/crowned rows, {len(scored)} with a margin, "
          f"{sum(1 for d in duels if d['real_crown'])} real crowns")
    anchor = _iso(args.anchor_time)
    results = {}
    for W in (args.window_blocks or [3600, 7200]):
        res = replay(scored, W, args.anchor_block, anchor, not args.no_hotkey_dedupe, args.seed)
        results[str(W)] = res
        print_table(res, only_with_candidates=not args.all_windows)
    if args.json:
        args.json.write_text(json.dumps(results, indent=1, default=str))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
