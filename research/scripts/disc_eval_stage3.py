#!/usr/bin/env python3
"""Stage 3: does the discriminator's verdict track real coding ability?

Stage 2 trains D(prefix, action) so that teacher actions outrank miner actions.
Per test pair we get delta = D(prefix, y_teacher) - D(prefix, y_miner):
how easily the discriminator tells the miner apart from the teacher.

A miner the discriminator *cannot* separate is imitating the teacher well, so
we score each checkpoint by closeness = -mean(delta) and ask whether closeness
predicts swe_rebench_lite.

Statistics are imported from rt7_full_panel so this is directly comparable to
the control baseline measured there (where Reason's own S and margin came out
uninformative, all stratified AUC ~0.5).

Stratification by corpus_epoch is mandatory, not cosmetic: pooling epochs of
different difficulty produced a Simpson's-paradox sign flip in the control.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rt7_full_panel as ctl  # noqa: E402  (shared stats + bench loader)

PERM_TRIALS = 20000


def load_repo_epoch(data_dir):
    """repo -> corpus_epoch, recovered from the Stage 1 pair shards."""
    out = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "pairs_*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                repo = (r.get("repo") or "").lower()
                if repo and repo not in out and r.get("corpus_epoch") is not None:
                    out[repo] = r["corpus_epoch"]
    return out


def load_scores(path):
    """repo -> list of per-pair deltas."""
    by = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            repo = (r.get("repo") or "").lower()
            if repo and r.get("delta") is not None:
                by[repo].append(float(r["delta"]))
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="test_scores.jsonl from Stage 2")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "disc_pairs"))
    ap.add_argument("--min-pairs", type=int, default=20,
                    help="drop repos with too few scored turns to be stable")
    args = ap.parse_args()

    # rt7_full_panel resolves BENCH_HISTORY relative to research/, its own cwd
    os.chdir(os.path.join(HERE, ".."))

    bench = ctl.load_bench()
    epoch = load_repo_epoch(args.data)
    scores = load_scores(args.scores)

    rows = []
    for repo, deltas in scores.items():
        if len(deltas) < args.min_pairs:
            continue
        b = bench.get(repo)
        if not b:
            continue
        mean_delta = sum(deltas) / len(deltas)
        sep_rate = sum(1 for d in deltas if d > 0) / len(deltas)
        rows.append({
            "repo": repo,
            "n_pairs": len(deltas),
            # closeness to teacher: higher = harder to tell apart = (hypothesis) better
            "closeness": -mean_delta,
            "sep_rate": sep_rate,
            "swe_rate": b["resolved"] / b["attempted"],
            "solved": 1 if b["resolved"] >= 1 else 0,
            "corpus_epoch": epoch.get(repo),
        })

    print("=" * 74)
    print("Stage 3 — discriminator closeness vs swe_rebench_lite")
    print("=" * 74)
    print(f"scored repos with bench labels : {len(rows)}")
    if not rows:
        print("\nNo overlap between scored repos and benched repos. "
              "Nothing to test; widen the test split or wait for more bench runs.")
        return
    pos = sum(r["solved"] for r in rows)
    print(f"  solved >= 1 task             : {pos}   (floor: {len(rows) - pos} at 0/25)")
    print(f"  median scored turns per repo : "
          f"{sorted(r['n_pairs'] for r in rows)[len(rows)//2]}")
    eps = sorted({r["corpus_epoch"] for r in rows})
    print(f"  corpus epochs represented    : {eps}")

    for metric, direction in (("closeness", "higher = closer to teacher"),
                              ("sep_rate", "higher = easier to separate")):
        print("\n" + "-" * 74)
        print(f"metric = {metric}  ({direction})")
        print("-" * 74)

        # pooled (reported only to expose the confound, never as the verdict)
        vals = [r[metric] for r in rows]
        rho_p = ctl.spearman(vals, [r["swe_rate"] for r in rows])
        auc_p = ctl.auc(vals, [r["solved"] for r in rows])
        print(f"  POOLED   rho={rho_p:+.3f}   auc={auc_p:.3f}   (confounded, see note)")

        g_rho = ctl.make_groups(rows, metric, "swe_rate", "corpus_epoch")
        g_auc = ctl.make_groups(rows, metric, "solved", "corpus_epoch")
        rho, per_rho = ctl.stratified_spearman(g_rho)
        a, per_auc = ctl.stratified_auc(g_auc)
        p_rho = ctl.stratified_perm_p(g_rho, rho, "spearman", PERM_TRIALS)
        p_auc = ctl.stratified_perm_p(g_auc, a, "auc", PERM_TRIALS)
        print(f"  STRATIFIED by corpus_epoch (min stratum n={ctl.MIN_STRATUM})")
        print(f"    Spearman rho = {rho:+.3f}   perm p = {p_rho:.4f}   "
              f"({len(per_rho)} usable strata)")
        print(f"    AUC          = {a:.3f}   perm p = {p_auc:.4f}   "
              f"({len(per_auc)} usable strata)")
        for s in per_auc:
            print(f"      stratum n={s['n']:<4} pos={s['pos']:<4} auc={s['auc']:.3f}")

    print("\n" + "=" * 74)
    print("How to read this")
    print("=" * 74)
    print("  * The verdict is the STRATIFIED row. Pooled numbers mix corpus epochs")
    print("    of different difficulty and flipped sign in the control (Simpson's")
    print("    paradox), so they are printed only for contrast.")
    print("  * Control baseline (rt7_full_panel): Reason's own S and margin were")
    print("    uninformative about swe_rebench_lite, stratified AUC ~0.5, p ~ n.s.")
    print("    The discriminator only earns its keep if it beats that.")
    print("  * swe_rebench_lite is 25 tasks with a heavy floor at 0/25, so rank")
    print("    correlation is attenuated; AUC on solved>=1 is the robust read.")
    print("  * A null result here does NOT say GAD cannot work. It says this")
    print("    discriminator, at this size and training budget, carries no")
    print("    ability signal beyond what length alone explains.")


if __name__ == "__main__":
    main()
