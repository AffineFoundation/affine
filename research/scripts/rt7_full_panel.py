"""RT-7 re-measured on the full local panel, with a noise-robust statistic.

`rt7_live_isomorphism.py` joins two *presentation* files — history.json (last
100 events) and benchmarks.json (latest run per repo) — and lands on n=29. The
validator's own durable stores are far larger. Joining bench_history.jsonl
against the local eval artifacts gives n≈163 on the same question.

Four changes beyond the bigger n, all forced by properties of the data:

  1. Bench scores are POOLED per repo (sum resolved / sum attempted) instead of
     taking the latest run. Genesis scored 0/25 then 5/25 on one revision, so a
     single run is not a measurement. Repeatability is measured here, not
     assumed.
  2. The headline statistic is AUC on "resolved >= 1", not Spearman.
     swe_rebench_lite is 25 tasks with ~half the field at 0/25, so rank
     correlation is attenuated toward zero. The binary split is near balanced
     and survives the noise. Spearman is still reported for continuity.
  3. Reason is recomputed from the stored pair records with the production v4
     scorer (harness.score.turn_reason, tau=0.03), so every era is scored
     uniformly instead of trusting whatever margin each duel published.
  4. **Everything is STRATIFIED.** This is not optional. Pooling the panel
     naively produces Simpson's paradox: pooled Spearman(S, swe) comes out
     +0.28 while three of the four largest corpus epochs are individually
     NEGATIVE. The mechanism is a shared confound — later epochs have harder
     slices (rho(epoch, S) = -0.43) and slightly weaker models
     (rho(epoch, swe) = -0.14), so the two decline together and fake a
     positive association. Absolute S is only comparable within a fixed slice
     distribution, exactly as rt7_live_isomorphism warned.

     So: S is stratified by corpus_epoch, margin by king_repo (margin is
     paired within a duel, so its confound is which king you faced, not which
     slice you drew). Strata are combined by Fisher-z for Spearman and by the
     standard n1*n0 weighting for AUC. Permutations shuffle WITHIN strata.
     The naive pooled numbers are still reported, labelled as confounded, so
     nobody re-derives them later and thinks they found something.

This is the CONTROL for any proposed replacement or secondary metric. A new
term is only interesting if it beats these numbers on this panel, measured the
same way, stratified the same way.

Usage:  cd research && python scripts/rt7_full_panel.py
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.score import DEFAULT_TEMPER_TAU, turn_reason  # noqa: E402

EVALS_GLOB = "../affine/state/evals/chal-*.json.gz"
EVALS_INDEX = "../affine/state/evals/index.jsonl"
BENCH_HISTORY = "../affine/state/bench_history.jsonl"
SUITE = "swe_rebench_lite"

# Frozen prior measurements, for the comparison table.
FREEZE_RHO = 0.758        # hybrid_w5_table.txt, n=30 Albedo kings (non-adversarial)
RT7_RHO_MARGIN = -0.421   # rt7_live_isomorphism.json, n=29 live challengers
RT7_N = 29


# --------------------------------------------------------------------------
# statistics (self-contained, matching rt7_live_isomorphism conventions)
# --------------------------------------------------------------------------

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


def _rho_from_ranks(ra, rb):
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0


def spearman(a, b):
    return _rho_from_ranks(rankdata(a), rankdata(b))


def spearman_perm_p(a, b, observed, trials, seed=0):
    """Permuting values is equivalent to permuting their ranks, so rank once."""
    ra, rb = rankdata(a), rankdata(b)
    rng = random.Random(seed)
    shuffled = list(rb)
    hits = 0
    for _ in range(trials):
        rng.shuffle(shuffled)
        if abs(_rho_from_ranks(ra, shuffled)) >= abs(observed):
            hits += 1
    return (hits + 1) / (trials + 1)


def auc(scores, labels):
    """Mann-Whitney AUC of `scores` against binary `labels`. Ties averaged.

    0.5 is chance. Below 0.5 means the score ranks failures ABOVE successes,
    which is the inversion RT-7 is about.
    """
    ranks = rankdata(scores)
    pos = [r for r, l in zip(ranks, labels) if l]
    n1, n0 = len(pos), len(labels) - len(pos)
    if not n1 or not n0:
        return float("nan")
    return (sum(pos) - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_perm_p(scores, labels, observed, trials, seed=0):
    ranks = rankdata(scores)
    n1 = sum(1 for l in labels if l)
    n0 = len(labels) - n1
    rng = random.Random(seed)
    lab = list(labels)
    hits = 0
    for _ in range(trials):
        rng.shuffle(lab)
        s = sum(r for r, l in zip(ranks, lab) if l)
        a = (s - n1 * (n1 + 1) / 2) / (n1 * n0)
        if abs(a - 0.5) >= abs(observed - 0.5):
            hits += 1
    return (hits + 1) / (trials + 1)


# --------------------------------------------------------------------------
# stratified combination — see the Simpson's paradox note in the docstring
# --------------------------------------------------------------------------

MIN_STRATUM = 8


def _fisher_z(r):
    r = max(-0.999999, min(0.999999, r))
    return 0.5 * math.log((1 + r) / (1 - r))


def _inv_fisher_z(z):
    e = math.exp(2 * z)
    return (e - 1) / (e + 1)


def usable_strata(groups):
    return [(v, w) for v, w in groups if len(v) >= MIN_STRATUM]


def stratified_spearman(groups):
    """Fisher-z combination of within-stratum rho, weighted by (n - 3)."""
    num = den = 0.0
    per = []
    for vals, w in groups:
        n = len(vals)
        if n < MIN_STRATUM or n <= 3:
            continue
        r = spearman(vals, w)
        num += (n - 3) * _fisher_z(r)
        den += n - 3
        per.append({"n": n, "rho": r})
    return (_inv_fisher_z(num / den) if den else float("nan")), per


def stratified_auc(groups):
    """Standard stratified Mann-Whitney: weight each stratum by n1 * n0."""
    num = den = 0.0
    per = []
    for vals, y in groups:
        n1 = sum(1 for v in y if v)
        n0 = len(y) - n1
        if len(y) < MIN_STRATUM or not n1 or not n0:
            continue
        a = auc(vals, y)
        num += n1 * n0 * a
        den += n1 * n0
        per.append({"n": len(y), "auc": a, "pos": n1})
    return (num / den if den else float("nan")), per


def stratified_perm_p(groups, observed, kind, trials, seed=0):
    """Null shuffles the outcome WITHIN each stratum, preserving the confound."""
    if math.isnan(observed):
        return float("nan")
    gs = usable_strata(groups)
    if not gs:
        return float("nan")
    rng = random.Random(seed)
    center = 0.5 if kind == "auc" else 0.0
    fn = stratified_auc if kind == "auc" else stratified_spearman
    hits = 0
    for _ in range(trials):
        perm = []
        for v, w in gs:
            sw = list(w)
            rng.shuffle(sw)
            perm.append((v, sw))
        stat, _ = fn(perm)
        if not math.isnan(stat) and abs(stat - center) >= abs(observed - center):
            hits += 1
    return (hits + 1) / (trials + 1)


def make_groups(rows, metric_key, outcome_key, stratum_key):
    by = defaultdict(list)
    for r in rows:
        by[r.get(stratum_key)].append(r)
    return [([r[metric_key] for r in rs], [r[outcome_key] for r in rs])
            for rs in by.values()]


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def load_bench():
    """repo -> pooled swe outcome across every ok run of the suite."""
    agg = defaultdict(lambda: {"resolved": 0, "attempted": 0, "runs": [], "label": ""})
    for line in open(BENCH_HISTORY):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("suite") != SUITE:
            continue
        res = row.get("result") or {}
        if not res.get("ok"):
            continue
        n_inst = res.get("n_instances")
        n_res = res.get("n_resolved")
        if not n_inst:
            continue
        if n_res is None:
            score = res.get("score")
            if score is None:
                continue
            n_res = round(score * n_inst)
        repo = (row.get("repo") or "").lower()
        a = agg[repo]
        a["resolved"] += int(n_res)
        a["attempted"] += int(n_inst)
        a["runs"].append(int(n_res) / int(n_inst))
        a["label"] = a["label"] or row.get("label", "")
    return {r: v for r, v in agg.items() if v["attempted"]}


def turn_score(row, tau):
    """Production v4 turn score, or None if the turn is unusable."""
    if not row.get("valid", True):
        return None
    pairs = [p for p in (row.get("pairs") or [])
             if p.get("lpC_yc_za") is not None and p.get("lpC_yc_e") is not None]
    if not pairs:
        return None
    v = turn_reason(pairs, tau)
    return v if math.isfinite(v) else None


def load_duels(tau):
    """One record per duel with Reason recomputed uniformly under v4."""
    at_by_chal = {}
    try:
        for line in open(EVALS_INDEX):
            row = json.loads(line)
            if row.get("challenge_id"):
                at_by_chal[row["challenge_id"]] = row.get("at", "")
    except FileNotFoundError:
        pass

    out = []
    for path in sorted(glob.glob(EVALS_GLOB)):
        try:
            d = json.loads(gzip.open(path).read())
        except Exception:
            continue
        if not d.get("challenger_rows") or not d.get("king_rows"):
            continue
        chal = {r["turn_id"]: s for r in d["challenger_rows"]
                if (s := turn_score(r, tau)) is not None}
        king = {r["turn_id"]: s for r in d["king_rows"]
                if (s := turn_score(r, tau)) is not None}
        shared = [t for t in chal if t in king]
        if len(shared) < 40:
            continue
        req = d.get("request") or {}
        cid = Path(path).name.split(".")[0]
        diffs = [chal[t] - king[t] for t in shared]
        out.append({
            "challenge_id": cid,
            "at": at_by_chal.get(cid, ""),
            "repo": (req.get("challenger_repo") or "").lower(),
            "revision": req.get("challenger_revision") or "",
            "king_repo": (req.get("king_repo") or "").lower(),
            "n_paired": len(shared),
            "s": sum(chal[t] for t in shared) / len(shared),
            "margin": sum(diffs) / len(diffs),
            "wins": bool((d.get("verdict") or {}).get("challenger_wins")),
            "corpus_epoch": (d.get("slice") or {}).get("corpus_epoch"),
        })
    return out


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/rt7_full_panel")
    ap.add_argument("--trials", type=int, default=50_000)
    ap.add_argument("--tau", type=float, default=DEFAULT_TEMPER_TAU)
    args = ap.parse_args()

    bench = load_bench()
    duels = load_duels(args.tau)
    print(f"loaded {len(duels)} duels, {len(bench)} benched repos")

    # One row per repo: its most recent duel as challenger, pooled bench outcome.
    by_repo = {}
    dup = 0
    for d in sorted(duels, key=lambda x: x["at"], reverse=True):
        repo = d["repo"]
        if not repo or repo not in bench:
            continue
        if repo in by_repo:
            dup += 1
            continue
        b = bench[repo]
        by_repo[repo] = {
            **{k: d[k] for k in ("challenge_id", "at", "s", "margin", "wins",
                                 "n_paired", "corpus_epoch", "revision",
                                 "king_repo")},
            "repo": repo,
            "swe": b["resolved"] / b["attempted"],
            "resolved": b["resolved"],
            "attempted": b["attempted"],
            "bench_runs": len(b["runs"]),
            "label": b["label"],
            "solved_any": int(b["resolved"] >= 1),
        }
    rows = sorted(by_repo.values(), key=lambda r: -r["margin"])
    if len(rows) < 10:
        raise SystemExit(f"panel too small (n={len(rows)}) — check input paths")

    s_vals = [r["s"] for r in rows]
    m_vals = [r["margin"] for r in rows]
    w_vals = [r["swe"] for r in rows]
    y_bin = [r["solved_any"] for r in rows]

    # S is confounded by slice difficulty -> stratify by corpus epoch.
    # margin is paired within a duel, so its confound is the opponent -> by king.
    stratum_for = {"S": "corpus_epoch", "margin": "king_repo"}
    metric_key = {"S": "s", "margin": "margin"}

    stats = {}
    for name in ("S", "margin"):
        key, strat = metric_key[name], stratum_for[name]
        vals = [r[key] for r in rows]
        g_rho = make_groups(rows, key, "swe", strat)
        g_auc = make_groups(rows, key, "solved_any", strat)
        rho_s, rho_per = stratified_spearman(g_rho)
        auc_s, auc_per = stratified_auc(g_auc)
        stats[name] = {
            "stratum": strat,
            "n_strata_used": len(rho_per),
            "n_rows_used": sum(p["n"] for p in rho_per),
            "spearman_stratified": rho_s,
            "spearman_stratified_p": stratified_perm_p(g_rho, rho_s, "rho", args.trials),
            "auc_stratified": auc_s,
            "auc_stratified_p": stratified_perm_p(g_auc, auc_s, "auc", args.trials),
            "spearman_naive_confounded": spearman(vals, w_vals),
            "auc_naive_confounded": auc(vals, y_bin),
            "per_stratum_rho": sorted(rho_per, key=lambda p: -p["n"]),
            "per_stratum_auc": sorted(auc_per, key=lambda p: -p["n"]),
        }

    # The confound itself, so the paradox is auditable rather than asserted.
    epochs = [r["corpus_epoch"] or 0 for r in rows]
    confound = {
        "spearman_epoch_S": spearman(epochs, s_vals),
        "spearman_epoch_swe": spearman(epochs, w_vals),
        "spearman_epoch_margin": spearman(epochs, m_vals),
    }

    # Bench repeatability, measured rather than assumed.
    repeats = [(r, b) for r, b in bench.items() if len(b["runs"]) > 1]
    spreads = [max(b["runs"]) - min(b["runs"]) for _, b in repeats]

    n_pos = sum(y_bin)
    res = {
        "n_panel": len(rows),
        "n_duels_loaded": len(duels),
        "n_benched_repos": len(bench),
        "n_repos_with_extra_duels": dup,
        "tau": args.tau,
        "class_balance": {"solved_any": n_pos, "zero": len(rows) - n_pos},
        "min_stratum": MIN_STRATUM,
        "stats": stats,
        "confound": confound,
        "prior": {
            "rt7_live_spearman_margin": RT7_RHO_MARGIN, "rt7_live_n": RT7_N,
            "freeze_albedo_spearman": FREEZE_RHO,
        },
        "bench_repeatability": {
            "repos_with_repeat_runs": len(repeats),
            "max_spread_instances": (max(spreads) * 25 if spreads else None),
            "mean_spread": (sum(spreads) / len(spreads) if spreads else None),
        },
        "field": {
            "pooled_resolved": sum(r["resolved"] for r in rows),
            "pooled_attempted": sum(r["attempted"] for r in rows),
            "max_swe": max(w_vals),
        },
        "rows": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=2))

    st_s, st_m = stats["S"], stats["margin"]
    lines = [
        "RT-7 on the FULL local panel — Reason v4 vs swe_rebench_lite",
        "=" * 66,
        f"n = {len(rows)} repos (vs n={RT7_N} in rt7_live_isomorphism, which joined the",
        f"    last-100 presentation files). {len(duels)} duels loaded, {len(bench)} repos benched.",
        f"    {dup} repos had extra duels; most recent kept.",
        f"Reason recomputed uniformly with production v4, tau={args.tau}.",
        f"Binary split: {n_pos} repos solved >=1 instance, {len(rows) - n_pos} solved 0.",
        "",
        "STRATIFIED (the numbers to use)",
        "  " + "-" * 62,
        f"  {'statistic':<34}{'value':>9}{'perm p':>10}",
        f"  {'AUC(margin, solved>=1) | king':<34}{st_m['auc_stratified']:>+9.3f}"
        f"{st_m['auc_stratified_p']:>10.4f}",
        f"  {'AUC(S, solved>=1) | epoch':<34}{st_s['auc_stratified']:>+9.3f}"
        f"{st_s['auc_stratified_p']:>10.4f}",
        f"  {'Spearman(margin, swe) | king':<34}{st_m['spearman_stratified']:>+9.3f}"
        f"{st_m['spearman_stratified_p']:>10.4f}",
        f"  {'Spearman(S, swe) | epoch':<34}{st_s['spearman_stratified']:>+9.3f}"
        f"{st_s['spearman_stratified_p']:>10.4f}",
        "",
        f"  strata used: S {st_s['n_strata_used']} epochs covering {st_s['n_rows_used']} repos; "
        f"margin {st_m['n_strata_used']} kings covering {st_m['n_rows_used']} repos",
        f"  (strata smaller than {MIN_STRATUM} repos are dropped)",
        "",
        "NAIVE POOLED — CONFOUNDED, shown only so nobody re-derives it",
        "  " + "-" * 62,
        f"  Spearman(S, swe)      pooled  {st_s['spearman_naive_confounded']:+.3f}   "
        f"vs stratified {st_s['spearman_stratified']:+.3f}",
        f"  Spearman(margin, swe) pooled  {st_m['spearman_naive_confounded']:+.3f}   "
        f"vs stratified {st_m['spearman_stratified']:+.3f}",
        "",
        "  Simpson's paradox. The shared confound is corpus epoch:",
        f"    Spearman(epoch, S)      = {confound['spearman_epoch_S']:+.3f}  (later slices are harder)",
        f"    Spearman(epoch, swe)    = {confound['spearman_epoch_swe']:+.3f}  (later models slightly weaker)",
        f"    Spearman(epoch, margin) = {confound['spearman_epoch_margin']:+.3f}",
        "  Both decline with epoch, which fakes a positive pooled association.",
        "",
        "PER-STRATUM detail (S by corpus epoch)",
    ]
    for p in st_s["per_stratum_rho"][:8]:
        lines.append(f"    n={p['n']:>3}   rho(S, swe) = {p['rho']:+.3f}")
    lines += [
        "",
        f"  prior: rt7_live Spearman(margin,swe) = {RT7_RHO_MARGIN:+.3f} at n={RT7_N} (pooled, unstratified)",
        f"  prior: freeze Albedo panel Spearman  = {FREEZE_RHO:+.3f} at n=30 (non-adversarial panel)",
        "",
        "How to read AUC: 0.5 is chance. Below 0.5 means the metric ranks models",
        "that resolve NOTHING above models that resolve something — the RT-7",
        "inversion. Above 0.5 means real signal about coding ability. A value",
        "near 0.5 with a large p means the metric is simply uninformative here,",
        "which is a weaker claim than inversion but still disqualifying.",
        "",
        f"Field: {res['field']['pooled_resolved']} resolved / {res['field']['pooled_attempted']} attempted pooled; "
        f"best repo {res['field']['max_swe']:.2f}.",
    ]
    if repeats:
        lines += [
            "",
            f"Bench repeatability (measured): {len(repeats)} of {len(bench)} repos have >1 run; "
            f"worst spread {res['bench_repeatability']['max_spread_instances']:.0f}/25 instances.",
            "  Pooling runs is why this panel is a better instrument than the",
            "  latest-run-per-repo view in benchmarks.json. Note repeat runs are not",
            "  randomly assigned (kings get re-benched), so that subset is selected.",
        ]
    lines += [
        "",
        "Caveats:",
        "  - Stratifying by epoch does not fully fix absolute S: every duel draws its",
        "    OWN slice, so difficulty varies within an epoch too. Margin is immune to",
        "    slice difficulty by construction and is the statistic to trust.",
        "  - Margin's own confound is the opponent, hence stratification by king. It",
        "    still measures 'better than the king of the day', not absolute ability.",
        "  - swe_rebench_lite is 25 tasks. Outcome noise attenuates Spearman toward",
        "    zero, so a significant result is a conservative floor; a null result is",
        "    NOT evidence of no effect, only of no detectable effect at this power.",
        "  - Repos are benched under policy=all, so the panel is not selected on",
        "    duel outcome.",
    ]
    text = "\n".join(lines)
    out.with_suffix(".txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
