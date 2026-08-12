"""Calibrate an absolute crown-margin floor δ for Reason v3 duels.

Question: what δ makes copy/noise crowns (true margin ≈ 0) negligible while
letting every genuine improvement through, at the live duel size n≈2080, k=2?

Data: local validator eval artifacts (affine/state/evals/chal-*.json.gz),
which store per-turn pair records with all lp components. Reason-only margins
are recomputed uniformly for every era (pre-fork duels stored mix-based
margins; the lp fields allow exact Reason recomputation).

Null model (an ε-copy of the king): per-turn paired diff has true mean 0.
We estimate the sampling spread of a 2080-turn mean three ways:
  1. iid bootstrap of duel-centered per-turn diffs (lower bound),
  2. trajectory-cluster bootstrap (turns from one trajectory are correlated),
  3. king cross-duel overdispersion (same king re-measured across duels with
     fresh slices + fresh teacher refs: catches between-duel variance that
     within-duel bootstraps cannot see; upper-bound flavored because pairing
     cancels turn-difficulty effects that the king-alone spread includes).

Signal: recomputed Reason margins of all historical duels, split by verdict.

Outputs: results/delta_calibration.json + .txt (run from research/).
"""

from __future__ import annotations

import glob
import gzip
import json
import math
import os
import statistics as st
from collections import Counter, defaultdict

import numpy as np

EVALS_GLOB = "../affine/state/evals/chal-*.json.gz"
OUT_JSON = "results/delta_calibration.json"
OUT_TXT = "results/delta_calibration.txt"

N_LIVE = 2080          # live n_turns
K_LIVE = 2.0           # live k_sigma
N_BOOT = 20000         # bootstrap replicates per duel
DELTAS = [0.002, 0.003, 0.005, 0.0075, 0.010, 0.015, 0.020]
RNG = np.random.default_rng(120)


def reason_of_pair(p: dict) -> float | None:
    a, e = p.get("lpC_yc_za"), p.get("lpC_yc_e")
    if a is None or e is None:
        return None
    v = a - e
    return v if math.isfinite(v) else None


def turn_reason(row: dict) -> float | None:
    vals = [r for p in row.get("pairs", []) if (r := reason_of_pair(p)) is not None]
    return float(np.mean(vals)) if vals else None


def load_duel(path: str) -> dict | None:
    try:
        d = json.load(gzip.open(path))
    except Exception:
        return None
    v = d.get("verdict") or {}
    if not d.get("king_rows") or not d.get("challenger_rows"):
        return None
    king = {r["turn_id"]: turn_reason(r) for r in d["king_rows"]}
    chal = {r["turn_id"]: turn_reason(r) for r in d["challenger_rows"]}
    tids = [t for t in king if t in chal and king[t] is not None and chal[t] is not None]
    if len(tids) < 40:
        return None
    diffs = np.array([chal[t] - king[t] for t in tids])
    king_means = np.array([king[t] for t in tids])
    return {
        "file": os.path.basename(path),
        "n": len(tids),
        "turn_ids": tids,
        "diffs": diffs,
        "king_mean": float(king_means.mean()),
        "king_turn_vals": king_means,
        "reason_margin": float(diffs.mean()),
        "reason_se": float(diffs.std(ddof=1) / math.sqrt(len(diffs))),
        "turn_std": float(diffs.std(ddof=1)),
        "published_margin": v.get("margin"),
        "published_se": v.get("se"),
        "published_n": v.get("n_paired_turns"),
        "k_sigma": v.get("k_sigma"),
        "wins": bool(v.get("challenger_wins")),
        "pre_fork": "gates" in v,
        "king_repo": (d.get("request") or {}).get("king_repo"),
        "king_rev": (d.get("request") or {}).get("king_revision"),
        "chal_repo": (d.get("request") or {}).get("challenger_repo"),
        "chal_rev": (d.get("request") or {}).get("challenger_revision"),
    }


def cluster_of(turn_id: str) -> str:
    return turn_id.split(":")[0]


def boot_sigma_iid(diffs: np.ndarray, n_target: int) -> float:
    centered = diffs - diffs.mean()
    idx = RNG.integers(0, len(centered), size=(N_BOOT, n_target))
    return float(centered[idx].mean(axis=1).std(ddof=1))


def boot_sigma_cluster(diffs: np.ndarray, tids: list[str], n_target: int) -> float:
    centered = diffs - diffs.mean()
    groups = defaultdict(list)
    for i, t in enumerate(tids):
        groups[cluster_of(t)].append(i)
    clusters = [np.array(ix) for ix in groups.values()]
    n_clusters = len(clusters)
    mean_cluster_size = len(diffs) / n_clusters
    draws_needed = int(math.ceil(n_target / mean_cluster_size))
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        picks = RNG.integers(0, n_clusters, size=draws_needed)
        vals = np.concatenate([centered[clusters[p]] for p in picks])[:n_target]
        means[b] = vals.mean()
    return float(means.std(ddof=1))


def main() -> None:
    duels = [d for p in sorted(glob.glob(EVALS_GLOB)) if (d := load_duel(p))]
    post = [d for d in duels if not d["pre_fork"]]
    pre = [d for d in duels if d["pre_fork"]]
    print(f"loaded {len(duels)} duels ({len(pre)} pre-fork, {len(post)} post-fork)")

    # ---- null spread at n=2080 -------------------------------------------
    # Post-fork duels are the live regime (1 teacher ref, current corpus);
    # compute per-duel bootstrap sigmas and pool.
    per_duel = []
    for d in post:
        sig_iid = boot_sigma_iid(d["diffs"], N_LIVE)
        sig_cl = boot_sigma_cluster(d["diffs"], d["turn_ids"], N_LIVE)
        n_clusters = len({cluster_of(t) for t in d["turn_ids"]})
        per_duel.append({
            "file": d["file"], "n": d["n"], "wins": d["wins"],
            "reason_margin": d["reason_margin"],
            "published_se": d["published_se"],
            "turn_std": d["turn_std"],
            "sigma_iid_2080": sig_iid,
            "sigma_cluster_2080": sig_cl,
            "n_clusters": n_clusters,
        })
    sig_iid_med = float(np.median([r["sigma_iid_2080"] for r in per_duel]))
    sig_cl_med = float(np.median([r["sigma_cluster_2080"] for r in per_duel]))
    sig_cl_max = float(np.max([r["sigma_cluster_2080"] for r in per_duel]))
    inflation = sig_cl_med / sig_iid_med if sig_iid_med else float("nan")

    # ---- between-duel overdispersion via king repeats ---------------------
    # Same king (repo, revision) measured across duels with fresh slices and
    # fresh refs. Compare cross-duel std of the king's mean Reason with the
    # within-duel expectation std/sqrt(n).
    king_groups = defaultdict(list)
    for d in post:
        king_groups[(d["king_repo"], d["king_rev"])].append(d)
    overdisp = []
    for (repo, rev), ds in king_groups.items():
        if len(ds) < 4:
            continue
        means = np.array([d["king_mean"] for d in ds])
        within = np.array([d["king_turn_vals"].std(ddof=1) / math.sqrt(d["n"]) for d in ds])
        phi = float(means.std(ddof=1) / np.mean(within)) if np.mean(within) else float("nan")
        overdisp.append({"king": repo, "rev": (rev or "")[:8], "duels": len(ds),
                         "cross_duel_std": float(means.std(ddof=1)),
                         "mean_within_se": float(np.mean(within)),
                         "phi": phi})

    # ---- signal: recomputed Reason margins --------------------------------
    margins_all = sorted(d["reason_margin"] for d in duels)
    crowns = [d for d in duels if d["wins"]]
    crown_margins = sorted(d["reason_margin"] for d in crowns)
    post_margins = sorted(d["reason_margin"] for d in post)

    # ---- delta table -------------------------------------------------------
    # Noise-crown probability for a true-zero challenger, Gaussian tail with
    # the low and high sigma estimates (crown needs margin > max(k*SE, delta);
    # k*SE at n=2080 is ~0.003, so delta dominates for delta > 0.003).
    def tail(delta: float, sigma: float) -> float:
        if sigma <= 0:
            return 0.0
        z = delta / sigma
        return 0.5 * math.erfc(z / math.sqrt(2))

    k_se_live = K_LIVE * sig_cl_med  # the statistical bar itself
    table = []
    for dl in DELTAS:
        eff = max(dl, k_se_live)
        p_lo = tail(eff, sig_iid_med)
        p_hi = tail(eff, sig_cl_max)
        blocked = [m for m in crown_margins if 0 < m <= dl]
        table.append({
            "delta": dl,
            "noise_crown_p_lo": p_lo,
            "noise_crown_p_hi": p_hi,
            "duels_per_noise_crown_lo": (1 / p_lo) if p_lo > 0 else float("inf"),
            "duels_per_noise_crown_hi": (1 / p_hi) if p_hi > 0 else float("inf"),
            "historical_crowns_blocked": len(blocked),
        })

    out = {
        "generated": "2026-08-12",
        "inputs": {"n_duels": len(duels), "pre_fork": len(pre), "post_fork": len(post),
                   "n_live": N_LIVE, "k_live": K_LIVE, "n_boot": N_BOOT},
        "null": {
            "sigma_iid_median": sig_iid_med,
            "sigma_cluster_median": sig_cl_med,
            "sigma_cluster_max": sig_cl_max,
            "cluster_inflation": inflation,
            "k_se_live_bar": k_se_live,
            "per_duel": per_duel,
            "king_overdispersion": overdisp,
        },
        "signal": {
            "all_margins_reason": margins_all,
            "post_fork_margins": post_margins,
            "crown_margins_reason": crown_margins,
            "crowns": [{"file": d["file"], "chal": d["chal_repo"],
                        "margin_reason": d["reason_margin"],
                        "pre_fork": d["pre_fork"]} for d in crowns],
        },
        "delta_table": table,
    }
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1)

    lines = []
    lines.append("δ calibration for Reason v3 (n=2080, k=2) — generated 2026-08-12")
    lines.append(f"duels: {len(duels)} ({len(pre)} pre-fork n=80, {len(post)} post-fork n≈2080)")
    lines.append("")
    lines.append("NULL (spread of a zero-edge challenger's measured margin @ n=2080):")
    lines.append(f"  sigma iid bootstrap (median across post-fork duels)      {sig_iid_med:.5f}")
    lines.append(f"  sigma trajectory-cluster bootstrap (median)              {sig_cl_med:.5f}")
    lines.append(f"  sigma trajectory-cluster bootstrap (max, conservative)   {sig_cl_max:.5f}")
    lines.append(f"  cluster inflation factor                                 {inflation:.2f}x")
    lines.append(f"  statistical bar k*SE at live settings                    {k_se_live:.5f}")
    for o in overdisp:
        lines.append(f"  king overdispersion {o['king'][:40]} ({o['duels']} duels): "
                     f"cross-duel std {o['cross_duel_std']:.5f} vs within SE {o['mean_within_se']:.5f} "
                     f"(phi={o['phi']:.2f})")
    lines.append("")
    lines.append("SIGNAL (Reason-only margins recomputed from pair records):")
    lines.append(f"  all duels    n={len(margins_all)} min={margins_all[0]:+.4f} "
                 f"p25={np.percentile(margins_all,25):+.4f} med={np.percentile(margins_all,50):+.4f} "
                 f"p75={np.percentile(margins_all,75):+.4f} max={margins_all[-1]:+.4f}")
    if crown_margins:
        lines.append(f"  crowned      {['%+.4f' % m for m in crown_margins]}")
    lines.append("")
    lines.append("DELTA TABLE (noise-crown prob per duel for a true-zero challenger):")
    lines.append("  delta   p_noise[lo]   p_noise[hi]   1-in-N[lo]   1-in-N[hi]  crowns_blocked")
    for r in table:
        lines.append(f"  {r['delta']:.4f}  {r['noise_crown_p_lo']:.2e}     {r['noise_crown_p_hi']:.2e}"
                     f"     {r['duels_per_noise_crown_lo']:.3g}       {r['duels_per_noise_crown_hi']:.3g}"
                     f"        {r['historical_crowns_blocked']}")
    txt = "\n".join(lines)
    open(OUT_TXT, "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
