#!/usr/bin/env python
"""Metrics for E1 / E2 / E3 from results.jsonl (no GPU).

Definitions (per turn, per condition c, refs = the condition's valid refs):
  a_i(z)   = lpC(y_i|x,z) − lpC(y_i|x,∅)          (per byte)
  R_c(z)   = centered tempered LME over a_i, tau = 0.03 (affine.score)
  a_sd     = population sd of the a_i — "R spread"
  identical: all valid ref actions equal after whitespace normalization
  band_c   = mu ± max(2·sd, 0.002) over t_i = lpC(z_C^i|x) of the condition's refs
  G_c(z)   = distance of m = lpC(z|x) INTO band_c;  G_0 = same with the H0 band
  score    = min(R_c, G_0)   (the proposed design: hinted refs, unhinted band)
  forfeit  = −0.1 when the miner has no valid action
  gate on  : a condition whose hint failed the grounding gate falls back to H0
Paired separation d = score(challenger) − score(king) per turn; z = mean/SE.
Pairs: teacher_heldout vs king_live (all turns), stored_chal vs stored_king
(turns from stored duels), teacher_heldout vs recorded (king groups: the
king's own reply at the state).

  python analyze.py --run-dir RUN [--out RUN/analysis]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "affine"))

from affine import score as S  # noqa: E402

TAU = 0.03
BAND_C = 2.0
BAND_FLOOR = 0.002
FORFEIT = -0.1
WS = re.compile(r"\s+")
PAIRS = [("teacher_heldout", "king_live"), ("stored_chal", "stored_king"),
         ("teacher_heldout", "recorded")]
MIX_LABELS = {3: "3h/0u", 2: "2h/1u", 1: "1h/2u", 0: "0h/3u"}


def norm(s: str) -> str:
    return WS.sub(" ", s or "").strip()


def lme(a: list[float], tau: float = TAU) -> float:
    if not a:
        return float("nan")
    if len(a) == 1:
        return 0.0
    m = max(a)
    return m + tau * math.log(st.mean(math.exp((x - m) / tau) for x in a)) - st.mean(a)


def band(ts: list[float]) -> tuple[float, float] | None:
    if not ts:
        return None
    mu = st.mean(ts)
    sd = st.stdev(ts) if len(ts) >= 2 else 0.0
    return mu, max(BAND_C * sd, BAND_FLOOR)


def g_of(m: float, b: tuple[float, float] | None) -> float | None:
    if b is None or m is None:
        return None
    mu, w = b
    return min(m - (mu - w), (mu + w) - m)


def refs_of(row: dict, cname: str) -> list[tuple[int, dict]]:
    cond = row["conditions"].get(cname)
    if not cond:
        return []
    return [(i, r) for i, r in enumerate(cond["refs"]) if r.get("valid") and r.get("lp_empty") is not None]


def miner_a(miner: dict, cname: str, refs: list[tuple[int, dict]]) -> list[float]:
    vals = (miner.get("lpC_yc_za") or {}).get(cname) or []
    out = []
    for i, r in refs:
        if i < len(vals) and vals[i] is not None:
            out.append(vals[i] - r["lp_empty"])
    return out


def turn_condition(row: dict, cname: str, mix: int | None = None, gate: bool = False) -> dict | None:
    """Per-turn measurement of one condition (optionally an E2 mixture with
    `mix` hinted refs + (3 − mix) unhinted refs; gate=True falls back to H0
    when the hint is ungrounded)."""
    cond = row["conditions"].get(cname)
    h0 = row["conditions"].get("H0")
    if cond is None or h0 is None:
        return None
    effective = cname
    if gate and cname != "H0" and cond.get("grounded") is False:
        effective = "H0"
    if mix is None or effective == "H0":
        refs = [(effective, i, r) for i, r in refs_of(row, effective)]
    else:
        hinted = [(effective, i, r) for i, r in refs_of(row, effective)][:mix]
        unh = [("H0", i, r) for i, r in refs_of(row, "H0")][: 3 - mix]
        refs = hinted + unh
    b0 = band([r["lp_thought"] for _, _, r in [("H0", i, r) for i, r in refs_of(row, "H0")]
               if r.get("lp_thought") is not None])
    bc = band([r["lp_thought"] for _, _, r in refs if r.get("lp_thought") is not None])
    acts = [norm(r["y"]) for _, _, r in refs]
    out = {
        "turn_id": row["turn_id"], "group": row["group"], "action_kind": row["action_kind"],
        "harness": row.get("harness"), "cond": cname, "effective": effective, "mix": mix,
        "n_valid": len(refs), "n_sampled": len(cond["refs"]),
        "identical": (len(acts) >= 2 and len(set(acts)) == 1),
        "r_dead": len(acts) < 2 or len(set(acts)) == 1,
        "grounded": cond.get("grounded"), "leaks_future": cond.get("leaks_future"),
        "hint_len": len(cond.get("hint") or "") if cond.get("hint") else None,
        "cap_hit": sum(1 for r in cond["refs"] if not r.get("think_closed")) / max(1, len(cond["refs"])),
        "ref_z_len": st.mean(len(r["z"]) for r in cond["refs"] if r.get("valid")) if any(r.get("valid") for r in cond["refs"]) else None,
        "ref_leak": (st.mean(1.0 if S.leakage(r["z"], r["y"]) else 0.0 for r in cond["refs"] if r.get("valid"))
                     if any(r.get("valid") for r in cond["refs"]) else None),
        "t_in_band0": None, "t_pos0": None, "band0": b0, "bandc": bc,
        "miners": {},
    }
    if b0 and effective != "H0":
        pos = [(r["lp_thought"] - b0[0]) / b0[1] for _, _, r in refs if r.get("lp_thought") is not None]
        if pos:
            out["t_pos0"] = st.mean(pos)
            out["t_in_band0"] = st.mean(1.0 if abs(p) <= 1 else 0.0 for p in pos)
    for mname, m in row["miners"].items():
        if not m.get("valid") or m.get("m") is None:
            out["miners"][mname] = {"forfeit": True, "score": FORFEIT}
            continue
        a = []
        for src, i, r in refs:
            vals = (m.get("lpC_yc_za") or {}).get(src) or []
            if i < len(vals) and vals[i] is not None:
                a.append(vals[i] - r["lp_empty"])
        R = lme(a) if a else None
        G0 = g_of(m["m"], b0)
        Gc = g_of(m["m"], bc)
        score = min(R, G0) if (R is not None and G0 is not None and not math.isnan(R)) else None
        out["miners"][mname] = {
            "forfeit": False, "R": R, "a_sd": (st.pstdev(a) if len(a) >= 2 else 0.0),
            "a": a, "G0": G0, "Gc": Gc, "m": m["m"], "score": score,
            "B": (m["lpC_ya_za"] - m["lpC_ya_e"]) if m.get("lpC_ya_za") is not None else None,
            "len_z": len(m.get("z") or ""),
        }
    return out


def mean_se(xs: list[float]) -> tuple[float, float, int]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else float("nan")), float("nan"), n
    return st.mean(xs), st.stdev(xs) / math.sqrt(n), n


def paired(tcs: list[dict], a: str, b: str) -> list[float]:
    out = []
    for tc in tcs:
        ma, mb = tc["miners"].get(a), tc["miners"].get(b)
        if ma is None or mb is None:
            continue
        if ma.get("score") is None and not ma.get("forfeit"):
            continue
        if mb.get("score") is None and not mb.get("forfeit"):
            continue
        out.append(ma["score"] - mb["score"])
    return out


def zstat(d: list[float]) -> tuple[float, float, float, int]:
    m, se, n = mean_se(d)
    z = m / se if (n >= 2 and se > 0) else float("nan")
    return m, se, z, n


def bootstrap_dz(d1: list[float], d0: list[float], n_boot: int = 1000, seed: int = 1) -> tuple[float, float]:
    """95% interval of z(d1) − z(d0) over paired turn resamples (both lists
    aligned by turn)."""
    n = min(len(d1), len(d0))
    if n < 5:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        z1 = zstat([d1[i] for i in idx])[2]
        z0 = zstat([d0[i] for i in idx])[2]
        if not (math.isnan(z1) or math.isnan(z0)):
            vals.append(z1 - z0)
    if not vals:
        return float("nan"), float("nan")
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def frac(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(1.0 if x else 0.0 for x in xs) if xs else float("nan")


def summarize(tcs_by_cond: dict[str, list[dict]], groups: list[str], label: str) -> list[dict]:
    """One table row per (condition, group)."""
    rows = []
    for cname, tcs in tcs_by_cond.items():
        for grp in groups:
            sub = [t for t in tcs if grp == "all" or t["group"] == grp]
            if not sub:
                continue
            base = [t for t in tcs_by_cond.get("H0", []) if grp == "all" or t["group"] == grp]
            base_by = {t["turn_id"]: t for t in base}
            row = {"table": label, "cond": cname, "group": grp, "n_turns": len(sub),
                   "n_hinted": sum(1 for t in sub if t["effective"] != "H0"),
                   "grounded_frac": frac([t["grounded"] for t in sub if t["cond"] != "H0"]),
                   "leak_frac": frac([t["leaks_future"] for t in sub if t["cond"] != "H0"]),
                   "ref_yield": st.mean(t["n_valid"] / 3 for t in sub),
                   "cap_hit": st.mean(t["cap_hit"] for t in sub),
                   "identical_frac": frac([t["identical"] for t in sub if t["n_valid"] >= 2]),
                   "r_dead_frac": frac([t["r_dead"] for t in sub]),
                   "ref_z_len": st.mean(t["ref_z_len"] for t in sub if t["ref_z_len"] is not None) if any(t["ref_z_len"] is not None for t in sub) else float("nan"),
                   "ref_leak": st.mean(t["ref_leak"] for t in sub if t["ref_leak"] is not None) if any(t["ref_leak"] is not None for t in sub) else float("nan"),
                   "t_in_band0": st.mean(t["t_in_band0"] for t in sub if t["t_in_band0"] is not None) if any(t["t_in_band0"] is not None for t in sub) else float("nan"),
                   "t_pos0": st.mean(t["t_pos0"] for t in sub if t["t_pos0"] is not None) if any(t["t_pos0"] is not None for t in sub) else float("nan"),
                   }
            for mname in ("teacher_heldout", "king_live", "recorded", "stored_king", "stored_chal"):
                ms = [t["miners"][mname] for t in sub if mname in t["miners"] and not t["miners"][mname]["forfeit"]]
                if ms:
                    row[f"R_{mname}"] = st.mean(m["R"] for m in ms if m["R"] is not None)
                    row[f"asd_{mname}"] = st.mean(m["a_sd"] for m in ms)
                    row[f"G0_{mname}"] = st.mean(m["G0"] for m in ms if m["G0"] is not None) if any(m["G0"] is not None for m in ms) else float("nan")
                    row[f"score_{mname}"] = st.mean(m["score"] for m in ms if m["score"] is not None) if any(m["score"] is not None for m in ms) else float("nan")
                    row[f"forfeit_{mname}"] = 1 - len(ms) / sum(1 for t in sub if mname in t["miners"])
            for a, b in PAIRS:
                d = paired(sub, a, b)
                m, se, z, n = zstat(d)
                key = f"{a[:8]}-{b[:8]}"
                row[f"d_{key}"], row[f"se_{key}"], row[f"z_{key}"], row[f"n_{key}"] = m, se, z, n
                if cname != "H0" and base_by:
                    # paired bootstrap on the same turns
                    tids = [t["turn_id"] for t in sub if t["turn_id"] in base_by]
                    d1 = paired([t for t in sub if t["turn_id"] in base_by], a, b)
                    d0 = paired([base_by[t] for t in tids], a, b)
                    z0 = zstat(d0)[2]
                    lo, hi = bootstrap_dz(d1, d0)
                    row[f"dz_{key}"] = z - z0 if not math.isnan(z0) else float("nan")
                    row[f"dz_lo_{key}"], row[f"dz_hi_{key}"] = lo, hi
            rows.append(row)
    return rows


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return "–"
        return f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"
    return str(v)


def md_table(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    run = Path(args.run_dir)
    out = Path(args.out or run / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(run / "results.jsonl")]
    ok = [r for r in rows if not r.get("failed") and "H0" in r.get("conditions", {})]
    print(f"{len(rows)} rows, {len(ok)} usable, {len(rows) - len(ok)} failed", file=sys.stderr)
    conds = sorted({c for r in ok for c in r["conditions"]}, key=lambda c: (c != "H0", c))
    groups = ["all", "king_loop_onset", "king_pivot", "completion"]

    # E1 / E3: every condition, gate off and gate on.
    e1_off = {c: [tc for r in ok if (tc := turn_condition(r, c)) is not None] for c in conds}
    e1_on = {c: [tc for r in ok if (tc := turn_condition(r, c, gate=True)) is not None] for c in conds if c != "H0"}
    e1_on["H0"] = e1_off["H0"]
    tables = summarize(e1_off, groups, "E1_gate_off") + summarize(e1_on, groups, "E1_gate_on")
    # E1 by dialect (gate off)
    by_kind: dict[str, list[dict]] = collections.defaultdict(list)
    for c, tcs in e1_off.items():
        for t in tcs:
            by_kind[f"{c}|{t['action_kind']}"].append(t)
    kind_rows = []
    for key, tcs in sorted(by_kind.items()):
        c, kind = key.split("|")
        kind_rows += [dict(r, action_kind=kind) for r in summarize({c: tcs, "H0": [t for t in e1_off["H0"] if t["action_kind"] == kind]}, ["all"], "E1_by_dialect") if r["cond"] == c]
    tables += kind_rows
    # E2: mixtures for every hinted condition.
    e2 = {}
    for c in conds:
        if c == "H0":
            continue
        for mix in (3, 2, 1):
            e2[f"{c}:{MIX_LABELS[mix]}"] = [tc for r in ok if (tc := turn_condition(r, c, mix=mix)) is not None]
    e2["H0"] = e1_off["H0"]
    tables += summarize(e2, groups, "E2_mix")

    with open(out / "tables.json", "w") as f:
        json.dump(tables, f, indent=1)
    cols = sorted({k for r in tables for k in r})
    with open(out / "tables.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in tables:
            w.writerow(r)
    with open(out / "turn_conditions.jsonl", "w") as f:
        for c, tcs in e1_off.items():
            for t in tcs:
                f.write(json.dumps(t) + "\n")

    main_cols = ["cond", "group", "n_turns", "grounded_frac", "leak_frac", "ref_yield", "cap_hit",
                 "identical_frac", "r_dead_frac", "t_in_band0", "t_pos0", "ref_leak",
                 "asd_teacher_heldout", "R_teacher_heldout", "R_king_live", "G0_teacher_heldout",
                 "d_teacher_-king_liv", "z_teacher_-king_liv", "n_teacher_-king_liv", "dz_teacher_-king_liv",
                 "dz_lo_teacher_-king_liv", "dz_hi_teacher_-king_liv",
                 "d_stored_c-stored_k", "z_stored_c-stored_k", "n_stored_c-stored_k",
                 "d_teacher_-recorded", "z_teacher_-recorded", "n_teacher_-recorded"]
    md = ["# Hinted-teacher probe — tables\n"]
    for label in ("E1_gate_off", "E1_gate_on", "E2_mix"):
        md.append(f"\n## {label}\n")
        md.append(md_table([r for r in tables if r["table"] == label], main_cols))
    md.append("\n## E1 by dialect (gate off)\n")
    md.append(md_table([r for r in tables if r["table"] == "E1_by_dialect"],
                       ["cond", "action_kind", "n_turns", "ref_yield", "identical_frac", "r_dead_frac",
                        "t_in_band0", "asd_teacher_heldout", "d_teacher_-king_liv", "z_teacher_-king_liv",
                        "n_teacher_-king_liv", "dz_teacher_-king_liv"]))
    (out / "tables.md").write_text("\n".join(md) + "\n")
    print((out / "tables.md").read_text())


if __name__ == "__main__":
    main()
