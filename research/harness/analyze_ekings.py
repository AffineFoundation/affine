"""Offline E-KINGS merge + Spearman audit against swe-rebench truth.

Loads results/ekings_v2_all.jsonl plus optional wave jsonl paths, merges bank
rescoring from bank_w1.jsonl and optional bank_w2_fullz.jsonl, prints per-miner
stats and Spearman correlations, writes results/ekings_merged_table.txt.

Usage:
  python -m harness.analyze_ekings
  python -m harness.analyze_ekings results/ekings_w3_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

from .score import DEFAULT_GAMMA, DEFAULT_GAMMA_BANK, gate_pass, lambda2, rank_term

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

KINGS_TRUTH = {
    "king-genesis": 58.2, "king-I": 38.4, "king-XCIX": 39.8, "king-VIII": 36.2,
    "king-XI": 33.6, "king-V": 32.0, "king-XLV": 26.0, "king-XLVI": 13.2,
    "king-LI": 11.6, "king-CI": 12.4,
    "king-II": 37.2, "king-III": 32.8, "king-VII": 33.2,
    "king-XCIV": 36.2, "king-XLI": 34.2,
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return RESULTS / path.name
    return path


def bank_fracs(*paths: Path) -> dict[str, float]:
    by_miner: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        for r in load_jsonl(path):
            by_miner[r["miner"]].append(1.0 if r["L2_bank"] > 0 else 0.0)
    return {m: st.mean(vs) for m, vs in by_miner.items()}


def per_miner_stats(rows: list[dict]) -> dict[str, dict]:
    pairs: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r.get("valid") or "pairs" not in r:
            continue
        for p in r["pairs"]:
            pairs[r["miner"]].append(p)
    out = {}
    for m, ps in pairs.items():
        out[m] = {
            "mix": st.mean(rank_term(p) for p in ps),
            "Lambda2": st.mean(lambda2(p) for p in ps),
            "gate_pass": st.mean(1.0 if gate_pass(p) else 0.0 for p in ps),
            "n_pairs": len(ps),
        }
    return out


def spearman(scores: list[float], truth: list[float]) -> tuple[float, float]:
    rho, p = stats.spearmanr(scores, truth)
    return float(rho), float(p)


def run(args) -> None:
    base = resolve_path(args.base)
    paths = [base] + [resolve_path(p) for p in args.waves]
    rows: list[dict] = []
    for p in paths:
        if not p.exists():
            print(f"warning: missing {p}", file=sys.stderr)
            continue
        rows.extend(load_jsonl(p))

    bank_w1 = resolve_path(args.bank_w1)
    bank_paths = [bank_w1]
    if args.bank_w2:
        bank_paths.append(resolve_path(args.bank_w2))
    bf = bank_fracs(*bank_paths)

    stats_m = per_miner_stats(rows)
    miners = sorted(
        (m for m in KINGS_TRUTH if m in stats_m),
        key=lambda m: -stats_m[m]["mix"],
    )

    lines: list[str] = []
    lines.append("E-KINGS merged table (mix = Λ2 + 0.5·L1lift)")
    lines.append(
        f"gates: γ={DEFAULT_GAMMA:.2f}, γ_bank={DEFAULT_GAMMA_BANK:.2f}\n"
    )
    header = f"{'king':<12} {'mix':>8} {'Λ2':>8} {'gate':>6} {'bank_frac':>10} {'swe':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    for m in miners:
        s = stats_m[m]
        b = bf.get(m)
        bank_s = f"{b:.3f}" if b is not None else "—"
        lines.append(
            f"{m.removeprefix('king-'):<12} "
            f"{s['mix']:+.4f} {s['Lambda2']:+.4f} "
            f"{s['gate_pass']:>5.0%} {bank_s:>10} "
            f"{KINGS_TRUTH[m]:>6.1f}"
        )

    truth = [KINGS_TRUTH[m] for m in miners]
    lines.append("")
    lines.append(f"Spearman vs swe-rebench (n={len(miners)} kings with data)")
    for label, key in [
        ("raw Λ2", "Lambda2"),
        ("mix (Λ2 + 0.5·L1lift)", "mix"),
    ]:
        scores = [stats_m[m][key] for m in miners]
        rho, p = spearman(scores, truth)
        lines.append(f"  {label:<28} ρ={rho:+.3f}  p={p:.3g}")
        print(f"{label:<28} Spearman={rho:+.3f}  p={p:.3g}  n={len(miners)}")

    g_mix = stats_m.get("king-genesis", {}).get("mix")
    i_mix = stats_m.get("king-I", {}).get("mix")
    if g_mix is not None and i_mix is not None:
        order = "genesis > I" if g_mix > i_mix else "I > genesis"
        lines.append(f"  mix ordering: {order}")
        print(f"mix ordering: {order}")

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("waves", nargs="*", default=[],
                    help="extra wave jsonl files to merge")
    ap.add_argument("--base", default="ekings_v2_all.jsonl",
                    help="primary ekings jsonl (default: results/ekings_v2_all.jsonl)")
    ap.add_argument("--bank-w1", default="bank_w1.jsonl")
    ap.add_argument("--bank-w2", default=None,
                    help="optional bank_w2_fullz.jsonl path")
    ap.add_argument("--out", default="ekings_merged_table.txt")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
