"""Re-freeze hybrid S* Spearman + duel slices under production S* v2.

Uses clip(L1lift, ±0.1), calibration-ratio gate r∈[1.0, 4.0], duel min_margin=0.05.

Usage: source .venv/bin/activate && python scripts/freeze_sstar_v2.py
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
from harness.score import (  # noqa: E402
    DEFAULT_GAMMA,
    DEFAULT_GAMMA_BANK,
    DEFAULT_L1_CLIP,
    DEFAULT_MIN_MARGIN,
    DEFAULT_R_HI,
    DEFAULT_R_LO,
    calibration_ratio,
    duel,
    gate_pass,
    rank_term,
    score_miner,
)

RESULTS = ROOT / "results"
TRUTH = {
    "king-genesis": 58.2, "king-XCIX": 39.8, "king-I": 38.4, "king-II": 37.2,
    "king-VIII": 36.2, "king-XCIV": 36.2, "king-XLI": 34.2, "king-XI": 33.6,
    "king-XLII": 33.4, "king-VII": 33.2, "king-III": 32.8, "king-XL": 32.2,
    "king-C": 32.0, "king-V": 32.0, "king-XCVI": 32.0, "king-XLV": 26.0,
    "king-XLVI": 13.2, "king-CI": 12.4, "king-LI": 11.6,
}
PAIR_FILES = [
    RESULTS / "ekings_v2_all.jsonl",
    RESULTS / "ekings_w2_v2_fullz.jsonl",
    RESULTS / "ekings_w3_v2.jsonl",
    RESULTS / "ekings_w3_VII.jsonl",
    RESULTS / "ekings_w3_all.jsonl",
    RESULTS / "ekings_w4.jsonl",
]
DUELS = {
    "I": RESULTS / "duel_genesis_I.jsonl",
    "II": RESULTS / "duel_genesis_II.jsonl",
    "VII": RESULTS / "duel_genesis_VII.jsonl",
    "XCIX": RESULTS / "duel_genesis_XCIX.jsonl",
    "LI": RESULTS / "duel_genesis_LI.jsonl",
}


def load_bank() -> tuple[dict[str, float], dict[str, int]]:
    bf_lists: dict[str, list[float]] = defaultdict(list)
    fullz: set[str] = set()
    if (RESULTS / "bank_w2_fullz.jsonl").exists():
        for line in open(RESULTS / "bank_w2_fullz.jsonl"):
            r = json.loads(line)
            bf_lists[r["miner"]].append(1.0 if r["L2_bank"] > 0 else 0.0)
            fullz.add(r["miner"])
    for p in [RESULTS / "bank_w1.jsonl", RESULTS / "bank_w3.jsonl",
              RESULTS / "bank_w4.jsonl", RESULTS / "bank_w2.jsonl"]:
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            if p.name == "bank_w2.jsonl" and r["miner"] in fullz:
                continue
            bf_lists[r["miner"]].append(1.0 if r["L2_bank"] > 0 else 0.0)
    if (RESULTS / "bank_w4.jsonl").exists():
        w4: dict[str, list[float]] = defaultdict(list)
        for line in open(RESULTS / "bank_w4.jsonl"):
            r = json.loads(line)
            w4[r["miner"]].append(1.0 if r["L2_bank"] > 0 else 0.0)
        for m, vs in w4.items():
            bf_lists[m] = vs
    bf = {m: st.mean(v) for m, v in bf_lists.items()}
    bn = {m: len(v) for m, v in bf_lists.items()}
    return bf, bn


def load_pairs() -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for pf in PAIR_FILES:
        if not pf.exists():
            continue
        for line in open(pf):
            r = json.loads(line)
            if not (r.get("valid") and "pairs" in r):
                continue
            key = (r["turn_id"], r["miner"])
            if key in seen:
                continue
            seen.add(key)
            by[r["miner"]].extend(r["pairs"])
    return by


def hybrid_valid(gate: float, bank: float | None, calib: float | None) -> bool:
    if gate < DEFAULT_GAMMA:
        return False
    if bank is not None and bank < DEFAULT_GAMMA_BANK:
        return False
    if calib is None or not (DEFAULT_R_LO <= calib <= DEFAULT_R_HI):
        return False
    return True


def freeze_hybrid() -> dict:
    by = load_pairs()
    bf, bn = load_bank()
    scores_u, truth_u, scores_h, truth_h, rejected, rows = [], [], [], [], [], []
    for m, swe in sorted(TRUTH.items(), key=lambda x: -x[1]):
        ps = by.get(m)
        if not ps:
            print("MISSING", m)
            continue
        mix = st.mean(rank_term(p) for p in ps)
        gate = st.mean(1.0 if gate_pass(p) else 0.0 for p in ps)
        bank = bf.get(m)
        calib = calibration_ratio(ps)
        valid = hybrid_valid(gate, bank, calib)
        rows.append((m.removeprefix("king-"), mix, gate, bank, bn.get(m, 0),
                     calib, valid, swe))
        scores_u.append(mix)
        truth_u.append(swe)
        if not valid:
            rejected.append(m.removeprefix("king-"))
        else:
            scores_h.append(mix)
            truth_h.append(swe)

    ru, pu = stats.spearmanr(scores_u, truth_u)
    rh, ph = stats.spearmanr(scores_h, truth_h)
    ranked = sorted(rows, key=lambda x: -x[1])
    genesis_rank = next(i + 1 for i, r in enumerate(ranked) if r[0] == "genesis")

    out = RESULTS / "hybrid_sstar_v2_table.txt"
    with open(out, "w") as f:
        f.write(
            "S* v2: clip(L1lift,±0.1) + gates γ=0.3 γ_bank=0.08 r∈[1.0,4.0] "
            "(w1+w3+w4 banks)\n"
        )
        f.write("king        S_mix   gate   bank  nbank     r valid   swe\n")
        for lab, mix, gate, bank, nbank, calib, valid, swe in rows:
            bs = f"{bank:.3f}" if bank is not None else "—"
            rs = f"{calib:.3f}" if calib is not None else "—"
            f.write(
                f"{lab:8s} {mix:8.4f} {gate*100:5.0f}% {bs:>6} "
                f"{nbank:5d} {rs:>6} {str(valid):5} {swe:5.1f}\n"
            )
        f.write(f"\nungated Spearman={ru:+.3f} (n={len(scores_u)}, p={pu:.4g})\n")
        f.write(f"hybrid  Spearman={rh:+.3f} (n={len(scores_h)}, p={ph:.4g})\n")
        f.write(f"genesis rank (ungated S_mix): #{genesis_rank} of {len(rows)}\n")
        f.write(f"rejected: {rejected}\n")
    print(out.read_text())
    return {
        "ungated_spearman": float(ru),
        "ungated_p": float(pu),
        "ungated_n": len(scores_u),
        "hybrid_spearman": float(rh),
        "hybrid_p": float(ph),
        "hybrid_n": len(scores_h),
        "genesis_rank": genesis_rank,
        "rejected": rejected,
        "rows": [
            {"king": r[0], "S_mix": r[1], "gate": r[2], "bank": r[3],
             "calib_ratio": r[5], "valid": r[6], "swe": r[7]}
            for r in rows
        ],
    }


def _mean_bank(rows: list[dict]) -> float | None:
    vals = [r["bank_frac"] for r in rows if r.get("valid") and "bank_frac" in r]
    return st.mean(vals) if vals else None


def _load_duel_rows(path: Path) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r.get("valid") and "pairs" in r:
            by[r["miner"]].append(r)
    return by


def rescore_duels() -> dict:
    results: dict[str, dict] = {}
    print("\n=== duel re-score (clip0.1, r gate, min_margin=0.05) ===")
    print(f"{'chall':6s} {'margin':>9s} {'se':>8s} {'z':>7s} {'wins':>5s}  "
          f"{'c_valid':>7s} {'k_valid':>7s} {'n':>3s}")
    for name, path in DUELS.items():
        if not path.exists():
            print(f"MISSING {path}")
            continue
        by = _load_duel_rows(path)
        king_rows = by.get("king-genesis", [])
        chall_rows = by.get(f"king-{name}", [])
        kb = _mean_bank(king_rows)
        cb = _mean_bank(chall_rows)
        dr = duel(
            chall_rows, king_rows,
            challenger_bank_frac=cb,
            king_bank_frac=kb,
            min_margin=DEFAULT_MIN_MARGIN,
        )
        ks = score_miner(king_rows, bank_frac=kb)
        cs = score_miner(chall_rows, bank_frac=cb)
        entry = {
            "king": {
                "name": ks.miner,
                "valid": ks.valid,
                "S": ks.S,
                "gate_pass_rate": ks.gate_pass_rate,
                "bank_frac": ks.bank_frac,
                "calib_ratio": ks.calib_ratio,
                "n_turns": ks.n_turns,
            },
            "challenger": {
                "name": cs.miner,
                "valid": cs.valid,
                "S": cs.S,
                "gate_pass_rate": cs.gate_pass_rate,
                "bank_frac": cs.bank_frac,
                "calib_ratio": cs.calib_ratio,
                "n_turns": cs.n_turns,
            },
            "duel": {
                "margin": dr.margin,
                "se": dr.se,
                "z": dr.z,
                "k_sigma": dr.k_sigma,
                "min_margin": dr.min_margin,
                "challenger_wins": dr.challenger_wins,
                "n_paired_turns": dr.n_paired_turns,
            },
        }
        results[name] = entry
        print(
            f"{name:6s} {dr.margin:+9.5f} {dr.se:8.5f} {dr.z:+7.2f} "
            f"{str(dr.challenger_wins):>5}  {str(cs.valid):>7} {str(ks.valid):>7} "
            f"{dr.n_paired_turns:3d}"
        )
    out = RESULTS / "duel_sstar_v2.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out}")
    return results


def main() -> None:
    hybrid = freeze_hybrid()
    duels = rescore_duels()
    meta = {"hybrid": hybrid, "duels": duels}
    (RESULTS / "hybrid_sstar_v2_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
