#!/usr/bin/env python3
"""Test the abstraction-ladder hypothesis.

Claim under test: miners first converge on the teacher's STYLE, and once style
is exhausted the only signal a discriminator can still use is SUBSTANCE, so the
adversarial process eventually selects for performance rather than style.

If that is right, then among pairs where the miner's thought is ALREADY very
close to the teacher's in surface form (style converged), whatever the judge can
still detect should track real coding ability more strongly -- and with the sign
GAD assumes (closer to teacher => better).

So: bucket pairs by surface similarity, and inside each bucket ask whether the
judge's verdict predicts swe_rebench_lite. Rising, correctly-signed correlation
as similarity increases supports the ladder. Flat or inverted does not.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rt7_full_panel as ctl  # noqa: E402
from disc_text import as_list, normalize  # noqa: E402

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S")
PERM = 20000


def toks(s):
    return set(TOKEN_RE.findall(s.lower()))


def jaccard(a, b):
    ta, tb = toks(a), toks(b)
    if not ta and not tb:
        return 1.0
    u = ta | tb
    return len(ta & tb) / len(u) if u else 0.0


def load_thought_pairs(data_dir):
    """(repo, turn_id) -> surface similarity between miner and teacher thought."""
    sim = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "pairs_*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                mine = normalize(r.get("z_a"))
                refs = [normalize(x) for x in as_list(r.get("teacher_z"))]
                ref = next((c for c in refs if c and c != mine), None)
                if not ref or not mine:
                    continue
                key = ((r.get("repo") or "").lower(), r["turn_id"])
                sim[key] = (jaccard(mine, ref), r.get("corpus_epoch"))
    return sim


def stratified(rows, metric, outcome, kind):
    g = ctl.make_groups(rows, metric, outcome, "corpus_epoch")
    if kind == "auc":
        stat, per = ctl.stratified_auc(g)
    else:
        stat, per = ctl.stratified_spearman(g)
    p = ctl.stratified_perm_p(g, stat, kind, PERM)
    return stat, p, len(per)


def analyse(rows, label):
    print(f"\n  {label}")
    print(f"    repos={len(rows)}  solved>=1: {sum(r['solved'] for r in rows)}")
    if len(rows) < ctl.MIN_STRATUM:
        print("    too few repos to stratify")
        return
    for metric, arrow in (("closeness", "GAD expects POSITIVE"),
                          ("sep_rate", "inverse of closeness")):
        a, pa, ns = stratified(rows, metric, "solved", "auc")
        rho, pr, _ = stratified(rows, metric, "swe_rate", "spearman")
        print(f"    {metric:<10} AUC={a:.3f} (p={pa:.3f})  rho={rho:+.3f} "
              f"(p={pr:.3f})  strata={ns}   [{arrow}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "disc_pairs"))
    ap.add_argument("--min-pairs", type=int, default=15)
    args = ap.parse_args()
    os.chdir(os.path.join(HERE, ".."))

    bench = ctl.load_bench()
    sim = load_thought_pairs(args.data)

    recs = []
    with open(args.scores) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = ((r.get("repo") or "").lower(), r.get("turn_id"))
            s = sim.get(key)
            if s is None:
                continue
            recs.append({"repo": key[0], "delta": r["delta"], "sim": s[0],
                         "corpus_epoch": s[1]})

    print("=" * 74)
    print("Abstraction-ladder test: does the signal track ability where style")
    print("has already converged?")
    print("=" * 74)
    print(f"scored pairs joined to similarity : {len(recs)}")
    sims = sorted(r["sim"] for r in recs)
    q = lambda p: sims[int(len(sims) * p)]
    print(f"surface similarity (Jaccard)      : p10={q(.1):.3f} p50={q(.5):.3f} "
          f"p90={q(.9):.3f}")

    # global panel, then similarity terciles
    cuts = [(0.0, 1.01, "ALL pairs"),
            (0.0, q(1 / 3), f"LOW similarity  (<{q(1/3):.3f}) - style NOT converged"),
            (q(1 / 3), q(2 / 3), f"MID similarity  ({q(1/3):.3f}-{q(2/3):.3f})"),
            (q(2 / 3), 1.01, f"HIGH similarity (>{q(2/3):.3f}) - style converged")]

    for lo, hi, label in cuts:
        by_repo = defaultdict(list)
        ep = {}
        for r in recs:
            if lo <= r["sim"] < hi:
                by_repo[r["repo"]].append(r["delta"])
                ep[r["repo"]] = r["corpus_epoch"]
        rows = []
        for repo, ds in by_repo.items():
            if len(ds) < args.min_pairs:
                continue
            b = bench.get(repo)
            if not b:
                continue
            rows.append({
                "repo": repo,
                "closeness": -sum(ds) / len(ds),
                "sep_rate": sum(1 for d in ds if d > 0) / len(ds),
                "swe_rate": b["resolved"] / b["attempted"],
                "solved": 1 if b["resolved"] >= 1 else 0,
                "corpus_epoch": ep[repo],
            })
        analyse(rows, label)

    print("\n" + "=" * 74)
    print("Reading the ladder claim")
    print("=" * 74)
    print("  Supports it : closeness AUC rises above 0.5 and strengthens as")
    print("                similarity increases (style gone, substance left).")
    print("  Against it  : closeness AUC stays at/below 0.5 in the HIGH bucket.")
    print("                Then what survives style convergence still is not")
    print("                ability -- it is teacher-specific idiosyncrasy.")


if __name__ == "__main__":
    main()
