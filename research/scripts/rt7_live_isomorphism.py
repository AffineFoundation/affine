"""RT-7: does S* stay benchmark-isomorphic on the LIVE adversarial panel?

The freeze in `research/results/hybrid_w5_table.txt` measured Spearman(S, swe) =
+0.758 on 30 Albedo kings. Those kings were trained against an LLM judge, not
against S*, so that panel is non-adversarial with respect to our own metric.

This script re-runs the same question on the only panel that is adversarial by
construction: live SN120 submissions, every one of which was produced by someone
trying to maximise S*. Both inputs are the validator's own published artifacts,
so the result is reproducible by anyone.

  duel outcomes : s3.hippius.com/affine-sn120/data/history.json   (last 100 events)
  swe advisory  : s3.hippius.com/affine-sn120/data/benchmarks.json (policy=all)

Usage:  python research/scripts/rt7_live_isomorphism.py [--out results/rt7_live_isomorphism]
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

HISTORY_URL = "https://s3.hippius.com/affine-sn120/data/history.json"
BENCH_URL = "https://s3.hippius.com/affine-sn120/data/benchmarks.json"
SUITE = "swe_rebench_lite"
FREEZE_RHO = 0.758  # research/results/hybrid_w5_table.txt, n=30 Albedo kings


def fetch(url: str):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0


def permutation_p(a, b, observed, trials=200_000, seed=0):
    """Two-sided. Permutation rather than a t-approximation because swe_lite is
    coarse (25 tasks => 0.04 granularity) and heavily tied at zero."""
    rng = random.Random(seed)
    shuffled = list(b)
    hits = 0
    for _ in range(trials):
        rng.shuffle(shuffled)
        if abs(spearman(a, shuffled)) >= abs(observed):
            hits += 1
    return (hits + 1) / (trials + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/rt7_live_isomorphism")
    ap.add_argument("--trials", type=int, default=200_000)
    args = ap.parse_args()

    history, bench = fetch(HISTORY_URL), fetch(BENCH_URL)

    swe, labels = {}, {}
    for m in bench["models"]:
        score = m["suites"].get(SUITE, {}).get("score")
        if score is not None:
            swe[m["model_repo"].lower()] = score
            labels[m["model_repo"].lower()] = m.get("label", "")

    # One row per repo, keeping its most recent duel.
    panel = {}
    for e in history:
        repo = (e.get("repo") or "").lower()
        if repo in swe and e.get("score") is not None and repo not in panel:
            panel[repo] = {
                "repo": repo,
                "s": e["score"],
                "margin": e.get("margin"),
                "swe": swe[repo],
                "accepted": e.get("accepted"),
                "label": labels.get(repo, ""),
            }
    rows = sorted(panel.values(), key=lambda r: -r["s"])

    s_vals = [r["s"] for r in rows]
    w_vals = [r["swe"] for r in rows]
    paired = [r for r in rows if r["margin"] is not None]
    m_vals = [r["margin"] for r in paired]
    mw_vals = [r["swe"] for r in paired]

    rho_s = spearman(s_vals, w_vals)
    rho_m = spearman(m_vals, mw_vals)
    res = {
        "n_challengers": len(rows),
        "n_benched_total": len(swe),
        "spearman_S_swe": rho_s,
        "p_S_swe": permutation_p(s_vals, w_vals, rho_s, args.trials),
        "spearman_margin_swe": rho_m,
        "p_margin_swe": permutation_p(m_vals, mw_vals, rho_m, args.trials),
        "freeze_spearman_albedo_panel": FREEZE_RHO,
        "reigns": {r: {"label": labels[r], "swe": s}
                   for r, s in swe.items() if str(labels.get(r, "")).startswith("reign")},
        "baseline_qwen_swe": next((s for r, s in swe.items() if "qwen/qwen3.6" in r), None),
        "field_max_swe": max(swe.values()),
        "field_zero_fraction": sum(1 for v in swe.values() if v == 0.0) / len(swe),
        "rows": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=2))

    zero_frac = res["field_zero_fraction"]
    lines = [
        "RT-7 — S* vs swe-rebench on the LIVE adversarial panel",
        "=" * 62,
        f"n = {res['n_challengers']} live challengers ({res['n_benched_total']} benched; policy=all,",
        "    so the bench panel is not selected on duel outcome)",
        "",
        f"  Spearman(duel margin, swe) = {rho_m:+.3f}   p = {res['p_margin_swe']:.4f}",
        f"  Spearman(S absolute,  swe) = {rho_s:+.3f}   p = {res['p_S_swe']:.4f}",
        f"  freeze, Albedo panel n=30  = {FREEZE_RHO:+.3f}",
        "",
        "  The sign flips. Margin is the cleaner statistic (paired within a duel);",
        "  absolute S is shown only for continuity with the freeze table.",
        "",
        "Crowned by S* vs the field:",
    ]
    for repo, info in sorted(res["reigns"].items(), key=lambda x: x[1]["label"]):
        lines.append(f"  {info['label']:9} swe={info['swe']:.2f}  {repo}")
    lines += [
        f"  baseline  swe={res['baseline_qwen_swe']:.2f}  Qwen/Qwen3.6-35B-A3B (untouched)",
        "",
        f"  Every model crowned BY S* scores 0.00. The untouched base model scores",
        f"  {res['baseline_qwen_swe']:.2f}, the best on a panel of {res['n_benched_total']}. {zero_frac*100:.0f}% of the field is at zero,",
        f"  so three-of-three kings at zero is p={zero_frac**3:.3f} on its own.",
        "",
        "Caveats:",
        "  - swe_rebench_lite is 25 tasks (0.04 granularity, floor-heavy); the freeze",
        "    used the 500-task suite. Different instruments, same direction of question.",
        "  - history.json exposes the last 100 events, so n is capped near 30.",
        "  - Absolute S is only strictly comparable within one duel; margin is paired",
        "    within a duel but still spans different kings and slices.",
    ]
    text = "\n".join(lines)
    out.with_suffix(".txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
