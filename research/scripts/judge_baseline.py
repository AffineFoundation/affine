"""E-BASELINE: quantify how badly the LLM-judge duel score tracks real benchmarks.

Inputs (already downloaded):
  data/albedo/albedo_dash.json   - eval_runs duel history (last 200 duels, king versions 91-103)
  data/albedo/albedo_scores.json - 500-task swe-rebench scores for ~30 kings + genesis + GLM-5.2 ref
  data/albedo/albedo_bench.json  - per-king latest_scores: tau2_{airline,retail,telecom} + 110-task swe_rebench subset

Outputs: results/judge_baseline/{baseline.json, baseline.md, king_trajectory.png, judge_vs_benchmark.png}
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "albedo"
OUT = ROOT / "results" / "judge_baseline"
OUT.mkdir(parents=True, exist_ok=True)

ROMAN_VALUES = [
    ("CM", 900), ("M", 1000), ("CD", 400), ("D", 500), ("XC", 90), ("C", 100),
    ("XL", 40), ("L", 50), ("IX", 9), ("X", 10), ("IV", 4), ("V", 5), ("I", 1),
]


def roman_to_int(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        v = vals[ch]
        if i + 1 < len(s) and vals[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total


def king_version_from_name(name: str) -> int | None:
    """'king-XCIX' -> 99, 'king-genesis' -> 0, 'King XCIX' -> 99."""
    m = re.search(r"king[-\s]+([IVXLCDM]+)$", name, re.IGNORECASE)
    if m:
        return roman_to_int(m.group(1).upper())
    if "genesis" in name.lower():
        return 0
    return None


dash = json.load(open(DATA / "albedo_dash.json"))
scores_500 = json.load(open(DATA / "albedo_scores.json"))
bench = json.load(open(DATA / "albedo_bench.json"))

# ---------- 1. Judge scores per king version, from eval_runs ----------
runs = dash["eval_runs"]
# Judge score at coronation: score_challenger of the duel the future king won.
judge_at_win = {}  # king_version -> score_challenger
for r in runs:
    if r.get("coronated"):
        judge_at_win[r["king_version"]] = r["score_challenger"]

# Mean judge score while defending the throne (denser signal, includes king 91).
defense_scores = defaultdict(list)  # king_version -> [score_king, ...]
for r in runs:
    kv = (r.get("king") or {}).get("king_version")
    if kv is not None and r.get("score_king") is not None:
        defense_scores[kv].append(r["score_king"])
judge_defense_mean = {kv: float(np.mean(v)) for kv, v in defense_scores.items()}

# ---------- 2. Benchmark scores per king version ----------
# 500-task swe-rebench (primary, from albedo_scores.json), percent scale.
swe500 = {}
glm_ref = None
for row in scores_500:
    v = king_version_from_name(row["model"])
    if v is not None:
        swe500[v] = row["score"]
    elif "glm" in row["model"].lower():
        glm_ref = row["score"]

# Suite scores from albedo_bench.json models[].latest_scores (fraction scale).
SUITES = ["tau2_airline", "tau2_retail", "tau2_telecom", "swe_rebench_2026_03"]
bench_scores = {s: {} for s in SUITES}
for m in bench["models"]:
    label = m.get("label") or m["id"]
    v = king_version_from_name(label)
    if v is None:
        continue
    for suite, rec in (m.get("latest_scores") or {}).items():
        if suite in bench_scores and rec.get("score") is not None:
            bench_scores[suite][v] = rec["score"]

tau2_mean = {}
for v in set().union(*(bench_scores[s] for s in SUITES[:3])):
    vals = [bench_scores[s][v] for s in SUITES[:3] if v in bench_scores[s]]
    if len(vals) == 3:
        tau2_mean[v] = float(np.mean(vals))


def corr(judge: dict, benchmark: dict) -> dict:
    common = sorted(set(judge) & set(benchmark))
    x = [judge[v] for v in common]
    y = [benchmark[v] for v in common]
    out = {"n": len(common), "king_versions": common}
    if len(common) >= 3:
        sp = stats.spearmanr(x, y)
        pe = stats.pearsonr(x, y)
        out.update(
            spearman_r=round(float(sp.statistic), 4), spearman_p=round(float(sp.pvalue), 4),
            pearson_r=round(float(pe.statistic), 4), pearson_p=round(float(pe.pvalue), 4),
        )
    return out


correlations = {
    "judge_at_win__vs__swe_rebench_500task": corr(judge_at_win, swe500),
    "judge_at_win__vs__tau2_mean": corr(judge_at_win, tau2_mean),
    "judge_at_win__vs__swe_rebench_110task_subset": corr(judge_at_win, bench_scores["swe_rebench_2026_03"]),
    "judge_defense_mean__vs__swe_rebench_500task": corr(judge_defense_mean, swe500),
    "judge_defense_mean__vs__tau2_mean": corr(judge_defense_mean, tau2_mean),
}
for suite in SUITES[:3]:
    correlations[f"judge_at_win__vs__{suite}"] = corr(judge_at_win, bench_scores[suite])

# ---------- 3. Trajectory over successive kings ----------
def trajectory(benchmark: dict, exclude_genesis: bool = False) -> dict:
    pts = sorted((v, s) for v, s in benchmark.items() if not (exclude_genesis and v == 0))
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    lr = stats.linregress(x, y)
    sp = stats.spearmanr(x, y)
    return {
        "n": len(pts),
        "king_version_range": [int(x.min()), int(x.max())],
        "slope_per_king": round(float(lr.slope), 4),
        "slope_p": round(float(lr.pvalue), 6),
        "intercept": round(float(lr.intercept), 4),
        "spearman_vs_order": round(float(sp.statistic), 4),
        "spearman_p": round(float(sp.pvalue), 6),
        "first_score": y[0], "last_score": y[-1],
    }


trajectories = {
    "swe_rebench_500task_incl_genesis": trajectory(swe500),
    "swe_rebench_500task_kings_only": trajectory(swe500, exclude_genesis=True),
    "tau2_mean_kings_only": trajectory(tau2_mean, exclude_genesis=True),
}

# ---------- 4. Dethronement transitions (consecutive kings only) ----------
def transitions(benchmark: dict, tol: float = 0.0) -> dict:
    versions = sorted(benchmark)
    improved, worsened, unchanged, pairs = 0, 0, 0, []
    for a, b in zip(versions, versions[1:]):
        if b - a != 1:
            continue  # only true dethronements king_n -> king_{n+1}
        delta = benchmark[b] - benchmark[a]
        pairs.append({"from": a, "to": b, "delta": round(delta, 4)})
        if delta > tol:
            improved += 1
        elif delta < -tol:
            worsened += 1
        else:
            unchanged += 1
    return {"n_transitions": len(pairs), "improved": improved, "worsened": worsened,
            "unchanged": unchanged, "pairs": pairs}


transition_stats = {
    "swe_rebench_500task": transitions(swe500),
    "tau2_mean": transitions(tau2_mean),
    "swe_rebench_110task_subset": transitions(bench_scores["swe_rebench_2026_03"]),
}

# ---------- 5. Duel-level sanity numbers ----------
coronation_margins = [r["win_margin"] for r in runs if r.get("coronated")]
duel_stats = {
    "n_eval_runs_in_window": len(runs),
    "window": [runs[-1]["finished_at"][:10], runs[0]["finished_at"][:10]],
    "n_coronations_in_window": len(coronation_margins),
    "king_versions_covered": sorted(judge_at_win),
    "mean_coronation_win_margin": round(float(np.mean(coronation_margins)), 4),
    "required_win_margin": runs[0]["required_win_margin"],
}

results = {
    "experiment": "E-BASELINE judge-score vs real benchmarks (Albedo SN97)",
    "date": "2026-08-02",
    "references": {"swe_rebench_500task_genesis": swe500.get(0), "swe_rebench_500task_glm_5_2": glm_ref},
    "duel_window": duel_stats,
    "judge_at_win": {str(k): round(v, 4) for k, v in sorted(judge_at_win.items())},
    "correlations": correlations,
    "trajectories": trajectories,
    "transitions": transition_stats,
    "notes": [
        "eval_runs only retains the most recent 200 duels (2026-07-23..2026-08-02), so judge-score-at-win exists only for king versions 92-103.",
        "swe_rebench_500task = albedo_scores.json (percent, 500 tasks); swe_rebench_110task_subset = albedo_bench.json suite swe_rebench_2026_03 (fraction, 110 tasks, different task set; kings LI-LXXIII score 0.0 on it).",
        "tau2_mean = unweighted mean of tau2 airline/retail/telecom pass rates from albedo_bench.json.",
        "judge_defense_mean = mean score_king over all duels a king defended within the 200-run window.",
    ],
}
json.dump(results, open(OUT / "baseline.json", "w"), indent=2)

# ---------- 6. Plots ----------
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

fig, ax1 = plt.subplots(figsize=(8, 4.5))
xs = sorted(swe500)
ax1.plot(xs, [swe500[v] for v in xs], "o-", color="#1f77b4", ms=4, lw=1,
         label="swe-rebench (500 tasks, %)")
ax1.axhline(swe500[0], color="#1f77b4", ls="--", lw=0.8, alpha=0.6)
ax1.annotate(f"genesis {swe500[0]:.1f}", (xs[-1], swe500[0]), fontsize=7,
             color="#1f77b4", va="bottom", ha="right")
if glm_ref:
    ax1.axhline(glm_ref, color="gray", ls=":", lw=0.8)
    ax1.annotate(f"GLM-5.2 ref {glm_ref:.1f}", (xs[-1], glm_ref), fontsize=7,
                 color="gray", va="bottom", ha="right")
ax1.set_xlabel("king version (coronation order)")
ax1.set_ylabel("swe-rebench score (%)", color="#1f77b4")
ax2 = ax1.twinx()
xt = sorted(tau2_mean)
ax2.plot(xt, [tau2_mean[v] * 100 for v in xt], "s-", color="#d62728", ms=3, lw=1,
         alpha=0.7, label="tau2 mean (airline/retail/telecom, %)")
ax2.set_ylabel("tau2 mean pass rate (%)", color="#d62728")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)
ax1.set_title("Albedo SN97: real benchmark scores across successive kings")
fig.tight_layout()
fig.savefig(OUT / "king_trajectory.png")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
c = correlations["judge_at_win__vs__swe_rebench_500task"]
common = sorted(set(judge_at_win) & set(swe500))
axes[0].scatter([judge_at_win[v] for v in common], [swe500[v] for v in common], c="#1f77b4")
for v in common:
    axes[0].annotate(str(v), (judge_at_win[v], swe500[v]), fontsize=7,
                     textcoords="offset points", xytext=(4, 3))
axes[0].set_xlabel("judge score at coronation")
axes[0].set_ylabel("swe-rebench (500 tasks, %)")
axes[0].set_title(
    f"vs swe-rebench (n={c['n']})\nSpearman={c.get('spearman_r', float('nan')):.2f}, "
    f"Pearson={c.get('pearson_r', float('nan')):.2f}", fontsize=9)

c = correlations["judge_at_win__vs__tau2_mean"]
common = sorted(set(judge_at_win) & set(tau2_mean))
axes[1].scatter([judge_at_win[v] for v in common], [tau2_mean[v] * 100 for v in common], c="#d62728")
for v in common:
    axes[1].annotate(str(v), (judge_at_win[v], tau2_mean[v] * 100), fontsize=7,
                     textcoords="offset points", xytext=(4, 3))
axes[1].set_xlabel("judge score at coronation")
axes[1].set_ylabel("tau2 mean pass rate (%)")
axes[1].set_title(
    f"vs tau2 mean (n={c['n']})\nSpearman={c.get('spearman_r', float('nan')):.2f}, "
    f"Pearson={c.get('pearson_r', float('nan')):.2f}", fontsize=9)
fig.suptitle("LLM-judge score at win vs real benchmarks (king versions annotated)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "judge_vs_benchmark.png")
plt.close(fig)

print(json.dumps({k: results[k] for k in ["correlations", "trajectories", "transitions", "duel_window"]}, indent=2))
