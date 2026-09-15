#!/usr/bin/env python
"""Re-score the last N stored verdicts under a group share vector.

Method (docs/duel-signal-by-group.md §7): per verdict and group g take the
mean m_g and variance v_g of the paired turn difference d = challenger −
king and the kept fraction k_g (turns scored on both sides / turns drawn).
Under weights w_g (renormalised over the groups present in that slice):

    keep    = Σ w_g k_g ;  u_g = w_g k_g / keep
    margin' = Σ u_g m_g
    var'    = Σ u_g (v_g + (m_g − margin')²)
    SE'     = sqrt(var' / (n · keep)) ;  z' = margin' / SE'

No turn is re-echoed; the counterfactual asks only "had the slice been
composed like this, with the turns we already scored". Stage-3 item 3:
mean |z'| within ±10 % of the stored mean |z| and no sign change among the
verdicts with |z| ≥ 2.

    python ops/curriculum/counterfactual.py --rows <sha>.rows.parquet --groups groups.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SLICE_N, clean_float, write_json  # noqa: E402


def load_verdict_groups(rows_path: Path, n_last: int) -> dict[str, dict]:
    cols = ["challenge_id", "side", "group", "joined", "scored", "d", "z", "margin"]
    t = pq.read_table(rows_path, columns=cols)
    by: dict[str, dict] = {}
    for r in t.to_pylist():
        if r["side"] != "challenger":
            continue
        v = by.setdefault(r["challenge_id"], {"z": r["z"], "margin": r["margin"], "groups": {}})
        g = r["group"] if r["joined"] else "(unjoined)"
        s = v["groups"].setdefault(g, {"n": 0, "d": []})
        s["n"] += 1
        if r["scored"] and r["d"] is not None:
            s["d"].append(float(r["d"]))
    cids = sorted(by)[-n_last:]
    return {c: by[c] for c in cids}


def rescore(v: dict, weights: dict[str, float] | None, n: int = SLICE_N) -> dict:
    stats = {}
    n_tot = sum(s["n"] for s in v["groups"].values()) or 1
    for g, s in v["groups"].items():
        ds = s["d"]
        m = sum(ds) / len(ds) if ds else 0.0
        var = (sum((x - m) ** 2 for x in ds) / (len(ds) - 1)) if len(ds) > 1 else 0.0
        stats[g] = {"m": m, "var": var, "keep": len(ds) / s["n"], "share": s["n"] / n_tot}
    w = {g: (stats[g]["share"] if weights is None else float(weights.get(g, 0.0))) for g in stats}
    tot = sum(w.values())
    if tot <= 0:
        return {"margin": None, "se": None, "z": None}
    w = {g: x / tot for g, x in w.items()}
    keep = sum(w[g] * stats[g]["keep"] for g in w)
    if keep <= 0:
        return {"margin": None, "se": None, "z": None}
    u = {g: w[g] * stats[g]["keep"] / keep for g in w}
    margin = sum(u[g] * stats[g]["m"] for g in u)
    var = sum(u[g] * (stats[g]["var"] + (stats[g]["m"] - margin) ** 2) for g in u)
    se = math.sqrt(var / (n * keep)) if var > 0 else 0.0
    z = margin / se if se > 0 else 0.0
    return {"margin": margin, "se": se, "z": z}


def run(rows_path: Path, shares: dict[str, float], n_last: int, tol_z: float = 0.10,
        variant_max_shift: float = 0.25, amended_band: tuple[float, float] = (-0.10, 0.50)) -> dict:
    verd = load_verdict_groups(rows_path, n_last)
    per = []
    for cid, v in verd.items():
        real = rescore(v, None)
        shad = rescore(v, shares)
        per.append({"challenge_id": cid, "z_stored": clean_float(v["z"]), "margin_stored": clean_float(v["margin"]),
                    "z_realized": clean_float(real["z"]), "se_realized": clean_float(real["se"]),
                    "z_shadow": clean_float(shad["z"]), "se_shadow": clean_float(shad["se"]),
                    "margin_shadow": clean_float(shad["margin"])})
    with_z = [p for p in per if p["z_stored"] is not None and p["z_shadow"] is not None]
    mean_abs_stored = sum(abs(p["z_stored"]) for p in with_z) / len(with_z) if with_z else None
    mean_abs_shadow = sum(abs(p["z_shadow"]) for p in with_z) / len(with_z) if with_z else None
    shift = (mean_abs_shadow / mean_abs_stored - 1.0) if mean_abs_stored else None
    flips_strong = [p["challenge_id"] for p in with_z
                    if abs(p["z_stored"]) >= 2 and (p["z_stored"] > 0) != (p["z_shadow"] > 0)]
    flips_any = [p["challenge_id"] for p in with_z if (p["z_stored"] > 0) != (p["z_shadow"] > 0)]
    se_s = sorted(p["se_realized"] for p in per if p["se_realized"])
    se_h = sorted(p["se_shadow"] for p in per if p["se_shadow"])
    med = lambda xs: xs[len(xs) // 2] if xs else None  # noqa: E731
    doc = {
        "n_verdicts": len(per), "first_challenge_id": per[0]["challenge_id"] if per else None,
        "last_challenge_id": per[-1]["challenge_id"] if per else None,
        "shares": {g: clean_float(x) for g, x in sorted(shares.items())},
        "mean_abs_z_stored": clean_float(mean_abs_stored), "mean_abs_z_shadow": clean_float(mean_abs_shadow),
        "mean_abs_z_shift": clean_float(shift),
        "median_se_realized": clean_float(med(se_s)), "median_se_shadow": clean_float(med(se_h)),
        "median_se_shift": clean_float(med(se_h) / med(se_s) - 1.0) if med(se_s) and med(se_h) else None,
        "sign_flips_abs_z_ge_2": flips_strong, "sign_flips_any": flips_any,
        # plan §7.3 item 3: mean |z| within ±tol_z of stored, no sign change at |z| >= 2
        "pass": (shift is not None and abs(shift) <= tol_z and not flips_strong),
        "tolerance_abs_z_shift": tol_z,
        # operator variant (2026-09-15 00:39 UTC, printed alongside, NOT the criterion yet):
        # 0 sign flips at |z| >= 2 and mean |z| shift <= +variant_max_shift -- larger |z|
        # with zero flips is the intended effect of concentrating signal
        "pass_variant": (shift is not None and shift <= variant_max_shift and not flips_strong),
        "variant_max_abs_z_shift": variant_max_shift,
        # THE criterion since the coordinator amendment 2026-09-15 01:05 UTC: 0 sign flips at
        # |z| >= 2 AND mean |z| change within [-10 %, +50 %] -- concentrating signal is the
        # intended effect; a magnitude increase with zero decision flips is not a failure
        "pass_amended": (shift is not None and amended_band[0] <= shift <= amended_band[1] and not flips_strong),
        "amended_band": list(amended_band),
        "per_verdict": per,
    }
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", required=True, help="ledger <sha>.rows.parquet")
    ap.add_argument("--groups", required=True, help="groups.json (uses share_after_clamp)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    gdoc = json.loads(Path(args.groups).read_text())
    shares = {g: r["share_after_clamp"] for g, r in gdoc["groups"].items()}
    doc = run(Path(args.rows), shares, args.n)
    if args.out:
        write_json(doc, Path(args.out))
    print(f"counterfactual n={doc['n_verdicts']} mean|z| stored {doc['mean_abs_z_stored']} shadow "
          f"{doc['mean_abs_z_shadow']} shift {doc['mean_abs_z_shift']} flips(|z|>=2) {doc['sign_flips_abs_z_ge_2']} "
          f"pass={doc['pass']}")


if __name__ == "__main__":
    main()
