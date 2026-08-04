"""RT-6 memorization / dataset-sniping probe (offline).

A memorizer overfits to specific public trajectories: its per-turn score should be
(a) spikier — a few turns carry the mean, (b) repo/trajectory-clustered, and
(c) inflated on turns containing agent-artifact strings it can key on.

We have no memorizing miner to test, so this establishes the *honest baseline*
distribution of these three signals across the 19 production kings, so the subnet
can flag a future miner that sits in the tail. Pure offline over stored pairs +
the public turn corpus.

Usage: source .venv/bin/activate && python scripts/rt6_memorization.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.config import KING_BENCH  # noqa: E402
from scripts.rt3_calibration import PAIR_FILES, mix  # noqa: E402

RESULTS = ROOT / "results"
# Distinctive agent-scaffold artifacts a memorizer could key on. Excludes
# ubiquitous tokens (```bash, "submit") that appear in every turn by construction.
ARTIFACT_STRINGS = [
    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "SUBMIT_FINAL", "<final>",
    "TASK COMPLETE", "FINAL_OUTPUT", "OBSERVATION:", "SCRATCHPAD",
    "def test_", "assert ", "pytest",
]


def per_turn_scores() -> dict[str, dict[str, float]]:
    """by[miner][turn_id] = mean mix over that turn's pairs."""
    by: dict[str, dict[str, float]] = defaultdict(dict)
    for pf in PAIR_FILES:
        if not pf.exists():
            continue
        for line in open(pf):
            r = json.loads(line)
            if not (r.get("valid") and "pairs" in r):
                continue
            if r["turn_id"] in by[r["miner"]]:
                continue
            good = [p for p in r["pairs"] if "lpA_yc_za" in p]
            if good:
                by[r["miner"]][r["turn_id"]] = st.mean(mix(p, "l1lift") for p in good)
    return by


def gini(xs: list[float]) -> float:
    """Gini of a non-negative array (we shift to non-negative first)."""
    if not xs:
        return 0.0
    m = min(xs)
    ys = sorted(x - m for x in xs)  # shift so min=0
    n = len(ys)
    s = sum(ys)
    if s == 0:
        return 0.0
    cum = 0.0
    for i, y in enumerate(ys, 1):
        cum += i * y
    return (2 * cum) / (n * s) - (n + 1) / n


def concentration(by: dict[str, dict[str, float]]) -> None:
    print("== per-turn S concentration (memorizer = spiky, high Gini / top-10% share) ==")
    print(f"{'king':8s} {'swe':>5s} {'nturn':>5s} {'mean':>8s} {'gini':>6s} "
          f"{'top10%share':>11s} {'top1turn':>9s}")
    rows = []
    for m, turns in by.items():
        suf = m.removeprefix("king-")
        if suf not in KING_BENCH or not turns:
            continue
        vals = list(turns.values())
        mean = st.mean(vals)
        base = min(vals)
        mass = [v - base for v in vals]
        total = sum(mass) or 1.0
        k = max(1, len(vals) // 10)
        top10 = sum(sorted(mass, reverse=True)[:k]) / total
        top1 = max(mass) / total
        g = gini(vals)
        rows.append((suf, KING_BENCH[suf], len(vals), mean, g, top10, top1))
    for r in sorted(rows, key=lambda x: -x[4]):
        print(f"{r[0]:8s} {r[1]:5.1f} {r[2]:5d} {r[3]:+8.4f} {r[4]:6.3f} "
              f"{r[5]*100:10.1f}% {r[6]*100:8.1f}%")
    ginis = [r[4] for r in rows]
    print(f"\nfleet Gini: mean={st.mean(ginis):.3f} sd={st.pstdev(ginis):.3f} "
          f"max={max(ginis):.3f} ({[r[0] for r in rows if r[4]==max(ginis)][0]})")
    print("Flag rule candidate: Gini > mean+3sd would mark a memorizer.")


def per_repo(by: dict[str, dict[str, float]]) -> None:
    print("\n== per-repo S dispersion (memorizer = one repo family carries the mean) ==")
    print(f"{'king':8s} {'best_repo':>16s} {'best-worst repo gap':>20s}")
    for m, turns in sorted(by.items()):
        suf = m.removeprefix("king-")
        if suf not in KING_BENCH or not turns:
            continue
        byrepo: dict[str, list[float]] = defaultdict(list)
        for tid, v in turns.items():
            repo = tid.split("__")[0]
            byrepo[repo].append(v)
        means = {r: st.mean(v) for r, v in byrepo.items() if len(v) >= 3}
        if len(means) < 2:
            continue
        best = max(means, key=means.get)
        worst = min(means, key=means.get)
        gap = means[best] - means[worst]
        print(f"{suf:8s} {best[:16]:>16s} {gap:20.4f}")


def artifact_keying(by: dict[str, dict[str, float]]) -> None:
    print("\n== artifact-string keying (does S spike on turns w/ agent artifacts?) ==")
    prefix_has: dict[str, bool] = {}
    corpus = ROOT / "data" / "turns_minicoder.jsonl"
    if not corpus.exists():
        print("  (turn corpus missing; skipping)")
        return
    wanted = set()
    for turns in by.values():
        wanted.update(turns)
    for line in open(corpus):
        r = json.loads(line)
        tid = f"{r['traj_id']}:{r['turn_idx']}"
        if tid not in wanted:
            continue
        text = " ".join(str(x) for x in r.get("prefix", [])) + " " + str(r.get("reference_turn", ""))
        prefix_has[tid] = any(s in text for s in ARTIFACT_STRINGS)
    n_flag = sum(prefix_has.values())
    print(f"  matched {len(prefix_has)} turns; {n_flag} contain an artifact string")
    if n_flag == 0 or n_flag == len(prefix_has):
        print("  (degenerate split; artifact strings ubiquitous or absent)")
    print(f"{'king':8s} {'S|artifact':>11s} {'S|clean':>9s} {'delta':>8s}")
    for m, turns in sorted(by.items()):
        suf = m.removeprefix("king-")
        if suf not in KING_BENCH:
            continue
        a = [v for tid, v in turns.items() if prefix_has.get(tid)]
        c = [v for tid, v in turns.items() if prefix_has.get(tid) is False]
        if len(a) >= 3 and len(c) >= 3:
            print(f"{suf:8s} {st.mean(a):+11.4f} {st.mean(c):+9.4f} "
                  f"{st.mean(a)-st.mean(c):+8.4f}")


def synthetic_memorizer(by: dict[str, dict[str, float]]) -> None:
    """Validate the concentration flag: take the worst honest king and let it
    'memorize' a fraction f of turns (score jumps to genesis level on those,
    honest elsewhere). Show that reaching a competitive mean forces Gini out of
    the honest band — i.e. the detector fires before the attack pays off."""
    print("\n== synthetic memorizer validation (base=XLVI, memorized turns -> genesis-level) ==")
    base_turns = by["king-XLVI"]
    gen = by["king-genesis"]
    fleet = [gini(list(t.values())) for m, t in by.items()
             if m.removeprefix("king-") in KING_BENCH and t]
    hi = st.mean(fleet) + 3 * st.pstdev(fleet)
    genesis_mean = st.mean(gen.values())
    print(f"honest Gini band max (mean+3sd) = {hi:.3f}; genesis mean S = {genesis_mean:+.4f}")
    print(f"{'frac_memorized':>14s} {'mean_S':>8s} {'gini':>6s} {'top10%':>7s} {'flagged?':>9s}")
    common = [t for t in base_turns if t in gen]
    for f in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        k = int(len(common) * f)
        memo = set(common[:k])
        vals = [gen[t] if t in memo else base_turns[t] for t in base_turns]
        mean = st.mean(vals)
        g = gini(vals)
        base = min(vals)
        mass = [v - base for v in vals]
        tot = sum(mass) or 1.0
        top10 = sum(sorted(mass, reverse=True)[: max(1, len(vals)//10)]) / tot
        print(f"{f:14.2f} {mean:+8.4f} {g:6.3f} {top10*100:6.1f}% "
              f"{'YES' if g > hi else 'no':>9s}")


def main() -> None:
    by = per_turn_scores()
    concentration(by)
    per_repo(by)
    artifact_keying(by)
    synthetic_memorizer(by)


if __name__ == "__main__":
    main()
