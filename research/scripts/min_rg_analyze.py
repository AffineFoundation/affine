"""Phase 4 of the min(R, G) upgrade test: combine centered-Reason with the
grounding band and replay the selected duels end to end.

Inputs:
  research/data/echo_results.jsonl    lpC(z|x)/byte per (duel, turn, kind)
                                      kinds: ref0..ref2, king, chall,
                                      and for chal-01093: parrot/boiler/swap
  affine/state/evals/chal-*.json.gz   stored duel records (centered R legs)

Per turn:
  band   mu = mean(ref lp/byte), sd = stdev(ref lp/byte)
  G(m)   = min(m - (mu - W), (mu + W) - m)   W = max(C_SIGMA*sd, W_MIN)
           positive inside the corridor, negative (nats/byte) outside
  turn   = min(centered R_turn, G)

Outputs research/results/min_rg_replay.{json,txt}:
  1. adversary panel on chal-01093 (m - mu distributions per kind,
     leave-one-out check for the refs themselves)
  2. duel table: margins/z under old rule, centered-R only, min(R, G)
"""

from __future__ import annotations

import gzip
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))

from affine.score import reason, turn_reason  # noqa: E402

EVALS_DIR = REPO / "affine" / "state" / "evals"
ECHOES = REPO / "research" / "data" / "echo_results.jsonl"
OUT_JSON = REPO / "research" / "results" / "min_rg_replay.json"
OUT_TXT = REPO / "research" / "results" / "min_rg_replay.txt"

DUELS = ["chal-01074", "chal-01093", "chal-01113", "chal-01115"]
TAU = 0.03
C_SIGMA = 2.0     # band half-width in ref-sd units
W_MIN = 0.002     # floor on half-width (nats/byte): k=3 sd is noisy
K_SIGMA = 2.0
REF_KINDS = ("ref0", "ref1", "ref2")


def load_echoes() -> dict[tuple[str, str], dict[str, float]]:
    """(challenge_id, turn_id) -> kind -> lp_per_byte."""
    out: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for line in open(ECHOES):
        r = json.loads(line)
        if r.get("lp_per_byte") is not None:
            out[(r["challenge_id"], r["turn_id"])][r["kind"]] = r["lp_per_byte"]
    return out


def band(refs: list[float]) -> tuple[float, float]:
    mu = st.mean(refs)
    sd = st.stdev(refs) if len(refs) > 1 else 0.0
    return mu, max(C_SIGMA * sd, W_MIN)


def g_score(m: float, mu: float, w: float) -> float:
    return min(m - (mu - w), (mu + w) - m)


def centered_turns(rows: list[dict]) -> dict[str, float]:
    out = {}
    for r in rows:
        if r.get("valid") and r.get("pairs"):
            a = [reason(p) for p in r["pairs"]]
            out[r["turn_id"]] = turn_reason(r["pairs"], TAU) - st.mean(a)
    return out


def paired_stats(diffs: list[float]) -> dict:
    n = len(diffs)
    mean = st.mean(diffs)
    se = st.stdev(diffs) / math.sqrt(n)
    return {"n": n, "margin": mean, "se": se,
            "z": mean / se if se > 0 else math.inf}


def main() -> None:
    echoes = load_echoes()
    lines: list[str] = []
    report: dict = {"panel": {}, "duels": []}

    # ---- adversary panel on chal-01093 ------------------------------------
    panel_kinds = ("king", "chall", "parrot", "boiler", "swap")
    dev: dict[str, list[float]] = defaultdict(list)   # m - mu per kind
    gval: dict[str, list[float]] = defaultdict(list)  # banded G per kind
    loo_dev: list[float] = []                         # ref leave-one-out
    loo_g: list[float] = []
    for (cid, _tid), kinds in echoes.items():
        if cid != "chal-01093":
            continue
        refs = [kinds[k] for k in REF_KINDS if k in kinds]
        if len(refs) < 3:
            continue
        mu, w = band(refs)
        for k in panel_kinds:
            if k in kinds:
                dev[k].append(kinds[k] - mu)
                gval[k].append(g_score(kinds[k], mu, w))
        for i in range(3):
            others = [refs[j] for j in range(3) if j != i]
            mu2, w2 = band(others)
            loo_dev.append(refs[i] - mu2)
            loo_g.append(g_score(refs[i], mu2, w2))

    lines.append(f"ADVERSARY PANEL chal-01093 (n_turns={len(dev['king'])}, "
                 f"band = mu +/- max({C_SIGMA}*sd, {W_MIN}))")
    lines.append(f"{'kind':<8} {'median m-mu':>12} {'median G':>10} "
                 f"{'%G>0':>6} {'mean G':>10}")
    rows_p = [("ref-LOO", loo_dev, loo_g)] + [
        (k, dev[k], gval[k]) for k in panel_kinds if dev[k]]
    for name, ds, gs in rows_p:
        frac = sum(1 for g in gs if g > 0) / len(gs)
        lines.append(f"{name:<8} {st.median(ds):>12.5f} "
                     f"{st.median(gs):>10.5f} {100 * frac:>5.1f}% "
                     f"{st.mean(gs):>10.5f}")
        report["panel"][name] = {
            "median_dev": st.median(ds), "median_g": st.median(gs),
            "frac_g_pos": frac, "mean_g": st.mean(gs), "n": len(gs)}

    # ---- duel replays under min(centered R, G) ----------------------------
    lines.append("")
    lines.append(f"DUEL REPLAYS  turn = min(centered R, G)   "
                 f"(delta ref: 0.002; z at {K_SIGMA} sigma)")
    lines.append(f"{'challenge':<11} {'challenger':<22} {'stored':>7} "
                 f"{'cenR z':>8} {'minRG margin':>13} {'minRG z':>8}")
    for cid in DUELS:
        d = json.loads(gzip.decompress(
            (EVALS_DIR / f"{cid}.json.gz").read_bytes()))
        req = d.get("request") or {}
        v = d.get("verdict") or {}
        ck = centered_turns(d["challenger_rows"])
        kk = centered_turns(d["king_rows"])
        diffs_cen, diffs_min = [], []
        for tid in sorted(set(ck) & set(kk)):
            kinds = echoes.get((cid, tid), {})
            refs = [kinds[k] for k in REF_KINDS if k in kinds]
            if len(refs) < 3 or "king" not in kinds or "chall" not in kinds:
                continue
            mu, w = band(refs)
            diffs_cen.append(ck[tid] - kk[tid])
            diffs_min.append(min(ck[tid], g_score(kinds["chall"], mu, w))
                             - min(kk[tid], g_score(kinds["king"], mu, w)))
        if len(diffs_min) < 2:
            lines.append(f"{cid:<11} (insufficient echo coverage)")
            continue
        s_cen = paired_stats(diffs_cen)
        s_min = paired_stats(diffs_min)
        chall = (req.get("challenger_repo") or req.get("repo") or "?")
        lines.append(
            f"{cid:<11} {chall.split('/')[0][:22]:<22} "
            f"{'WIN' if v.get('challenger_wins') else 'lose':>7} "
            f"{s_cen['z']:>8.2f} {s_min['margin']:>13.5f} {s_min['z']:>8.2f}")
        report["duels"].append({"challenge_id": cid, "challenger": chall,
                                "stored_wins": v.get("challenger_wins"),
                                "centered": s_cen, "min_rg": s_min})

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=1))
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
