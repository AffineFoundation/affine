"""RT-6 temporal / leave-repo holdout probe (offline).

D's traj_ids embed `org__repo.<commit>` — one commit per repo in the current
minicoder corpus (20 repos). We date those commits via the GitHub API and split
D into early/late halves. Then recompute each king's mean S on the scored turns
that fall in each half and check whether the swe-rebench Spearman survives.

A memorizer that sniped the public trajectories available at train time should
lose rank on the *later* half; an honest capability meter should keep ρ high on
both halves. Pure offline over stored pair files + GH commit metadata.

Usage:
  source .venv/bin/activate
  GITHUB_TOKEN=$(doppler secrets get GITHUB_TOKEN --plain) \\
    python scripts/rt6_temporal_holdout.py
"""

from __future__ import annotations

import json
import os
import re
import statistics as st
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.config import KING_BENCH  # noqa: E402
from scripts.rt3_calibration import PAIR_FILES, mix  # noqa: E402

RESULTS = ROOT / "results"
OUT_JSON = RESULTS / "rt6_temporal_holdout.json"
OUT_TXT = RESULTS / "rt6_temporal_holdout.txt"
TID_RE = re.compile(r"^([^./]+)__([^.]+)\.([0-9a-f]{7,40})\.")


def parse_repo_commit(traj_or_turn: str) -> tuple[str, str] | None:
    base = traj_or_turn.split(":")[0]
    m = TID_RE.match(base)
    if not m:
        return None
    return f"{m.group(1)}/{m.group(2)}", m.group(3)


def gh_commit_date(repo: str, sha: str, token: str) -> str | None:
    url = f"https://api.github.com/repos/{repo}/commits/{sha}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "affv-rt6",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        print(f"  gh fail {repo}@{sha[:8]}: {e.code}", file=sys.stderr)
        return None
    return (data.get("commit") or {}).get("committer", {}).get("date") or (
        (data.get("commit") or {}).get("author", {}).get("date")
    )


def load_per_turn() -> dict[str, dict[str, float]]:
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
                by[r["miner"]][r["turn_id"]] = st.mean(
                    mix(p, "l1lift") for p in good)
    return by


def spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if len(xs) < 3:
        return float("nan"), float("nan")
    r = stats.spearmanr(xs, ys)
    return float(r.statistic), float(r.pvalue)


def main() -> None:
    token = (
        os.environ.get("GITHUB_PR_SCREENING_TOKEN", "").strip()
        or os.environ.get("GITHUB_TOKEN", "").strip()
    )
    if not token:
        print("WARN: no GH token; using unauthenticated API", file=sys.stderr)

    by = load_per_turn()
    # Collect unique (repo, sha) from scored turns.
    commits: dict[tuple[str, str], str | None] = {}
    for turns in by.values():
        for tid in turns:
            pc = parse_repo_commit(tid)
            if pc:
                commits[pc] = None

    print(f"== dating {len(commits)} unique repo@commit from scored turns ==")
    cache_path = RESULTS / "rt6_commit_dates.json"
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    for (repo, sha) in sorted(commits):
        key = f"{repo}@{sha}"
        if key in cache:
            commits[(repo, sha)] = cache[key]
            continue
        d = gh_commit_date(repo, sha, token)
        if d:
            commits[(repo, sha)] = d
            cache[key] = d
            print(f"  {key} → {d}")
        else:
            print(f"  {key} → MISSING")
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")

    dated = [(k, v) for k, v in commits.items() if v]
    dated.sort(key=lambda kv: kv[1])
    if len(dated) < 4:
        raise SystemExit(f"too few dated commits ({len(dated)}) to split")

    mid = len(dated) // 2
    early_set = {repo for (repo, _), _ in dated[:mid]}
    late_set = {repo for (repo, _), _ in dated[mid:]}
    # if odd mid shares, ensure disjoint by assigning middle to late
    early_set -= late_set

    lines: list[str] = []
    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("== commit timeline (scored D) ==")
    for (repo, sha), d in dated:
        half = "early" if repo in early_set else "late"
        log(f"  {d}  {half:5s}  {repo}@{sha[:8]}")
    log(f"early repos ({len(early_set)}): {sorted(early_set)}")
    log(f"late  repos ({len(late_set)}): {sorted(late_set)}")

    def half_mean(turns: dict[str, float], repos: set[str]) -> tuple[float | None, int]:
        xs = []
        for tid, s in turns.items():
            pc = parse_repo_commit(tid)
            if pc and pc[0] in repos:
                xs.append(s)
        if not xs:
            return None, 0
        return st.mean(xs), len(xs)

    rows = []
    log("")
    log(f"{'king':12s} {'swe':>5s} {'S_all':>8s} {'n':>4s} "
        f"{'S_early':>8s} {'ne':>4s} {'S_late':>8s} {'nl':>4s} {'Δlate':>7s}")
    for miner, turns in sorted(by.items()):
        suf = miner.removeprefix("king-")
        swe = KING_BENCH.get(suf)
        if swe is None:
            continue
        s_all = st.mean(turns.values()) if turns else None
        s_e, n_e = half_mean(turns, early_set)
        s_l, n_l = half_mean(turns, late_set)
        if s_all is None or s_e is None or s_l is None:
            continue
        delta = s_l - s_e
        log(f"{miner:12s} {swe:5.1f} {s_all:8.4f} {len(turns):4d} "
            f"{s_e:8.4f} {n_e:4d} {s_l:8.4f} {n_l:4d} {delta:+7.4f}")
        rows.append({
            "miner": miner, "swe": swe, "S_all": s_all, "n_all": len(turns),
            "S_early": s_e, "n_early": n_e, "S_late": s_l, "n_late": n_l,
            "delta_late_minus_early": delta,
        })

    def pack(key: str) -> tuple[list[float], list[float], list[str]]:
        xs, ys, names = [], [], []
        for r in rows:
            if r["n_early"] < 5 or r["n_late"] < 5:
                continue
            xs.append(r[key])
            ys.append(r["swe"])
            names.append(r["miner"])
        return xs, ys, names

    log("")
    log("== Spearman(S_half, swe) ==")
    summary = {}
    for label, key in [("all", "S_all"), ("early", "S_early"), ("late", "S_late")]:
        xs, ys, names = pack(key) if key != "S_all" else (
            [r["S_all"] for r in rows], [r["swe"] for r in rows],
            [r["miner"] for r in rows])
        if key == "S_all":
            xs, ys, names = [r["S_all"] for r in rows], [r["swe"] for r in rows], [r["miner"] for r in rows]
        else:
            xs, ys, names = pack(key)
        rho, p = spearman(xs, ys)
        summary[label] = {"rho": rho, "p": p, "n": len(xs)}
        log(f"  S_{label:5s} vs swe: ρ={rho:+.3f}  p={p:.3g}  n={len(xs)}")

    # Rank stability early↔late
    early_rank = {r["miner"]: r["S_early"] for r in rows if r["n_early"] >= 5 and r["n_late"] >= 5}
    late_vals = [rows[[x["miner"] for x in rows].index(m)]["S_late"] for m in early_rank]
    # cleaner:
    common = [r for r in rows if r["n_early"] >= 5 and r["n_late"] >= 5]
    rho_el, p_el = spearman([r["S_early"] for r in common], [r["S_late"] for r in common])
    summary["early_vs_late"] = {"rho": rho_el, "p": p_el, "n": len(common)}
    log(f"  S_early vs S_late: ρ={rho_el:+.3f}  p={p_el:.3g}  n={len(common)}")

    # Biggest late drops (candidate snipers if any)
    drops = sorted(common, key=lambda r: r["delta_late_minus_early"])
    log("")
    log("== largest late drops (S_late − S_early); sniper would drop) ==")
    for r in drops[:8]:
        log(f"  {r['miner']:12s} Δ={r['delta_late_minus_early']:+.4f}  "
            f"swe={r['swe']:.1f}")

    verdict = (
        "STABLE" if summary.get("late", {}).get("rho", 0) > 0.7
        and summary.get("early_vs_late", {}).get("rho", 0) > 0.8
        else "UNSTABLE"
    )
    log("")
    log(f"VERDICT: {verdict} — late-half swe isomorphism "
        f"ρ={summary.get('late', {}).get('rho', float('nan')):+.3f}; "
        f"early↔late ρ={rho_el:+.3f}.")
    log("Interpretation: with one commit/repo, 'temporal' = dated leave-repo-out. "
        "Stable ρ ⇒ S is not carried by a few early-repo trajectories. "
        "Does not replace corpus refresh at deploy time.")

    payload = {
        "early_repos": sorted(early_set),
        "late_repos": sorted(late_set),
        "commits": {f"{r}@{s}": d for (r, s), d in commits.items() if d},
        "rows": rows,
        "summary": summary,
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_JSON} and {OUT_TXT}")


if __name__ == "__main__":
    main()
