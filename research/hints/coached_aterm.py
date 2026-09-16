#!/usr/bin/env python
"""Coached A-term probe read-out (2026-09-16): credit the miner for taking
the action the COACHED teacher takes.

Input: a run_probe results.jsonl with the arms `H0_4096` (plain references,
k=3 + held-out) and `coached_4096` (references sampled WITH the stored coach
note, k=3 + a coached held-out sample) and the miners
  king_live       the live king sampled on its box
  recorded        the king's own reply recorded at this state
  stored_king / stored_chal[, 2, 3]   both sides of stored verdicts on this turn
  teacher_heldout the plain teacher's 4th sample
  coached_coached_4096   the coached teacher's 4th sample
  coached_stored  the first reply of the decisive coached continuation (y*)

Per miner and per turn:
  A_P, A_C   fraction of the valid plain / coached refs whose normalised action
             equals the miner's; pair_P / pair_C = the refs' own pairwise agreement
  A*         1[miner action == y*]; p* = fraction of plain refs equal to y*
  S0         min(R, G) with R over the PLAIN refs and G from the PLAIN band
             (the live rule, at the 4,096 cap)
  V1_w       S0 + w·(A_C − pair_C)     fresh coached refs as the target
  V2_w       S0 + w·(A* − p*)          the single stored coached action as target
  VP_w       S0 + w·(A_P − pair_P)     the live A term (plain refs), for contrast
A forfeit (no parseable action / no thought) scores −0.1 under every rule.

  python coached_aterm.py --run-dir RUN --turns turns.jsonl --out RUN/analysis
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "affine"))

import analyze as A  # noqa: E402
from amatch import norm_action, pairwise, synthetic_actions  # noqa: E402
from aterm_live import is_generic  # noqa: E402

FORFEIT = -0.1
WEIGHTS = (0.005, 0.01, 0.02)
P, C = "H0_4096", "coached_4096"
MINERS = ["coached_coached_4096", "coached_stored", "teacher_heldout", "king_live", "recorded",
          "stored_king", "stored_chal", "stored_chal2", "stored_chal3"]
PAIRS = [("coached_coached_4096", "king_live"), ("coached_stored", "king_live"),
         ("coached_coached_4096", "recorded"), ("teacher_heldout", "king_live"),
         ("coached_coached_4096", "teacher_heldout"),
         ("stored_chal", "stored_king"), ("stored_chal2", "stored_king"), ("stored_chal3", "stored_king"),
         ("king_live", "recorded"), ("king_live", "stored_king")]
RULES = ["S0", "A_P", "Ac_P", "A_C", "Ac_C", "A_star", "Ac_star", "R_P", "G_P", "R_C"] + \
        [f"V1_{w}" for w in WEIGHTS] + [f"V2_{w}" for w in WEIGHTS] + [f"VP_{w}" for w in WEIGHTS]


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:+.{d}f}" if isinstance(v, float) else str(v)


def zstat(d):
    d = [x for x in d if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(d) < 2:
        return {"mean": st.mean(d) if d else None, "z": None, "n": len(d), "pos": None}
    m = st.mean(d)
    se = st.stdev(d) / math.sqrt(len(d))
    return {"mean": m, "z": m / se if se > 0 else None, "n": len(d), "pos": sum(1 for x in d if x > 0) / len(d)}


def refs_of(row, cname):
    return A.refs_of(row, cname)


def norm_refs(row, cname):
    kind = row["action_kind"]
    out = []
    for i, r in refs_of(row, cname):
        out.append((i, r, norm_action(r["y"], kind)))
    return [(i, r, a) for i, r, a in out if a is not None]


def majority(acts):
    if not acts:
        return None
    c = collections.Counter(acts)
    top, n = c.most_common(1)[0]
    return top if n > 1 or len(acts) == 1 else None


def miner_terms(row, mname, m, refsP, refsC, bandP, ystar, pstar, subset_label):
    kind = row["action_kind"]
    t = {"turn_id": row["turn_id"], "miner": mname, "action_kind": kind, "subset": subset_label,
         "forfeit": not (m.get("valid") and m.get("m") is not None and m.get("z"))}
    if t["forfeit"]:
        for r in RULES:
            t[r] = FORFEIT if r.startswith(("S0", "V")) else None
        t["A_P"] = t["A_C"] = t["A_star"] = None
        t["generic"] = None
        return t
    a = norm_action(m["y"], kind)
    t["generic"] = is_generic(m["y"], kind)
    t["len_y"] = len(m["y"])
    actsP = [x for _, _, x in refsP]
    actsC = [x for _, _, x in refsC]
    pairP, pairC = pairwise(actsP), pairwise(actsC)
    t["A_P"] = (sum(1 for x in actsP if x == a) / len(actsP)) if actsP and a else (0.0 if actsP else None)
    t["A_C"] = (sum(1 for x in actsC if x == a) / len(actsC)) if actsC and a else (0.0 if actsC else None)
    t["Ac_P"] = (t["A_P"] - pairP) if (t["A_P"] is not None and pairP is not None) else None
    t["Ac_C"] = (t["A_C"] - pairC) if (t["A_C"] is not None and pairC is not None) else None
    t["A_star"] = (1.0 if (a and ystar and a == ystar) else 0.0) if ystar else None
    t["Ac_star"] = (t["A_star"] - pstar) if (t["A_star"] is not None and pstar is not None) else None
    t["match_generic_C"] = bool(a and t["generic"] and t["A_C"])
    # R over plain refs / coached refs; G from the plain band
    aP = []
    for i, r, _ in refsP:
        vals = (m.get("lpC_yc_za") or {}).get(P) or []
        if i < len(vals) and vals[i] is not None:
            aP.append(vals[i] - r["lp_empty"])
    aC = []
    for i, r, _ in refsC:
        vals = (m.get("lpC_yc_za") or {}).get(C) or []
        if i < len(vals) and vals[i] is not None:
            aC.append(vals[i] - r["lp_empty"])
    t["R_P"] = A.lme(aP) if aP else None
    t["R_C"] = A.lme(aC) if aC else None
    t["G_P"] = A.g_of(m["m"], bandP)
    if t["R_P"] is None or t["G_P"] is None or (isinstance(t["R_P"], float) and math.isnan(t["R_P"])):
        t["S0"] = None
    else:
        t["S0"] = min(t["R_P"], t["G_P"])
    for w in WEIGHTS:
        t[f"V1_{w}"] = (t["S0"] + w * t["Ac_C"]) if (t["S0"] is not None and t["Ac_C"] is not None) else t["S0"]
        t[f"V2_{w}"] = (t["S0"] + w * t["Ac_star"]) if (t["S0"] is not None and t["Ac_star"] is not None) else t["S0"]
        t[f"VP_{w}"] = (t["S0"] + w * t["Ac_P"]) if (t["S0"] is not None and t["Ac_P"] is not None) else t["S0"]
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--turns", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    turns = {json.loads(l)["turn_id"]: json.loads(l) for l in open(args.turns)}
    rows = [json.loads(l) for l in open(Path(args.run_dir) / "results.jsonl")]
    rows = [r for r in rows if not r.get("failed") and P in r.get("conditions", {}) and C in r.get("conditions", {})]
    per_turn, terms = [], []
    for row in rows:
        turn = turns.get(row["turn_id"], {})
        kind = row["action_kind"]
        refsP, refsC = norm_refs(row, P), norm_refs(row, C)
        actsP, actsC = [x for _, _, x in refsP], [x for _, _, x in refsC]
        bandP = A.band([r["lp_thought"] for _, r in refs_of(row, P) if r.get("lp_thought") is not None])
        bandC = A.band([r["lp_thought"] for _, r in refs_of(row, C) if r.get("lp_thought") is not None])
        cs = (row["miners"].get("coached_stored") or {})
        ystar = norm_action(cs["y"], kind) if cs.get("y") else None
        pstar = (sum(1 for x in actsP if x == ystar) / len(actsP)) if (actsP and ystar) else None
        majP, majC = majority(actsP), majority(actsC)
        coached_meta = (turn.get("coached") or {})
        subset = f"{coached_meta.get('set')}:{coached_meta.get('arm')}:{coached_meta.get('label6')}"
        pt = {
            "turn_id": row["turn_id"], "action_kind": kind, "harness": row.get("harness"), "subset": subset,
            "n_refs_P": len(actsP), "n_refs_C": len(actsC),
            "pair_P": pairwise(actsP), "pair_C": pairwise(actsC),
            "identical_P": len(actsP) >= 2 and len(set(actsP)) == 1,
            "identical_C": len(actsC) >= 2 and len(set(actsC)) == 1,
            "overlap_C_in_P": (sum(1 for x in actsC if x in set(actsP)) / len(actsC)) if actsC else None,
            "majority_differs": (majP is not None and majC is not None and majP != majC),
            "no_shared_action": bool(actsC and actsP and not (set(actsC) & set(actsP))),
            "ystar_in_C": (ystar in set(actsC)) if (ystar and actsC) else None,
            "ystar_in_P": (ystar in set(actsP)) if (ystar and actsP) else None,
            "p_star": pstar,
            "generic_refs_P": (sum(1 for _, r, _ in refsP if is_generic(r["y"], kind)) / len(refsP)) if refsP else None,
            "generic_refs_C": (sum(1 for _, r, _ in refsC if is_generic(r["y"], kind)) / len(refsC)) if refsC else None,
            "cap_hit_C": sum(1 for r in row["conditions"][C]["refs"] if r.get("think_closed") is False) / max(1, len(row["conditions"][C]["refs"])),
            "cap_hit_P": sum(1 for r in row["conditions"][P]["refs"] if r.get("think_closed") is False) / max(1, len(row["conditions"][P]["refs"])),
            "coached_refs_in_bandP": None,
            "synthetic": {},
        }
        if bandP:
            pos = [(r["lp_thought"] - bandP[0]) / bandP[1] for _, r in refs_of(row, C) if r.get("lp_thought") is not None]
            if pos:
                pt["coached_refs_in_bandP"] = st.mean(1.0 if abs(p) <= 1 else 0.0 for p in pos)
        for sname, sact in synthetic_actions(turn, kind).items() if turn else []:
            if sact is None:
                continue
            pt["synthetic"][sname] = {
                "A_C": (sum(1 for x in actsC if x == sact) / len(actsC)) if actsC else None,
                "A_P": (sum(1 for x in actsP if x == sact) / len(actsP)) if actsP else None,
            }
        per_turn.append(pt)
        for mname, m in row["miners"].items():
            if mname not in MINERS:
                continue
            terms.append(miner_terms(row, mname, m, refsP, refsC, bandP, ystar, pstar, subset))
    with open(out / "per_turn.jsonl", "w") as f:
        for pt in per_turn:
            f.write(json.dumps(pt) + "\n")
    with open(out / "terms.jsonl", "w") as f:
        for t in terms:
            f.write(json.dumps(t) + "\n")

    by_turn = collections.defaultdict(dict)
    for t in terms:
        by_turn[t["turn_id"]][t["miner"]] = t
    kinds = sorted({pt["action_kind"] for pt in per_turn})
    subsets = sorted({pt["subset"] for pt in per_turn})
    L = []
    L.append(f"# Coached A-term probe — read-out ({len(rows)} turns)\n")

    # 1. ref-set diagnostics
    L.append("## 1. Reference sets: plain (H0_4096) vs coached (coached_4096)\n")
    L.append("| slice | n | refs/turn P | refs/turn C | pair_P | pair_C | identical_P | identical_C | coached refs in plain band | majority differs | no shared action | y* in C refs | y* in P refs | p* | cap-hit P | cap-hit C | generic refs P | generic refs C |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    def refrow(label, sub):
        if not sub:
            return
        mean = lambda k: A.smean([x[k] for x in sub if x.get(k) is not None])  # noqa: E731
        frac = lambda k: A.smean([1.0 if x[k] else 0.0 for x in sub if x.get(k) is not None])  # noqa: E731
        L.append(f"| {label} | {len(sub)} | {mean('n_refs_P'):.2f} | {mean('n_refs_C'):.2f} | {fmt(mean('pair_P'),2)} | {fmt(mean('pair_C'),2)} | "
                 f"{fmt(frac('identical_P'),2)} | {fmt(frac('identical_C'),2)} | {fmt(mean('coached_refs_in_bandP'),2)} | {fmt(frac('majority_differs'),2)} | "
                 f"{fmt(frac('no_shared_action'),2)} | {fmt(frac('ystar_in_C'),2)} | {fmt(frac('ystar_in_P'),2)} | {fmt(mean('p_star'),2)} | "
                 f"{fmt(mean('cap_hit_P'),2)} | {fmt(mean('cap_hit_C'),2)} | {fmt(mean('generic_refs_P'),2)} | {fmt(mean('generic_refs_C'),2)} |")
    refrow("all", per_turn)
    for k in kinds:
        refrow(f"dialect {k}", [x for x in per_turn if x["action_kind"] == k])
    for s in subsets:
        refrow(f"set {s}", [x for x in per_turn if x["subset"] == s])
    L.append("")

    # 2. levels per miner
    L.append("## 2. Per-miner levels (answered turns; forfeits counted separately)\n")
    L.append("| miner | n turns | forfeit | A_P | A_C | Ac_P | Ac_C | A* | Ac* | R_P | G_P | S0 = min(R,G) | generic actions | share of A_C matches that are generic |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mname in MINERS:
        ts = [t for t in terms if t["miner"] == mname]
        if not ts:
            continue
        ans = [t for t in ts if not t["forfeit"]]
        mean = lambda k: A.smean([t[k] for t in ans if t.get(k) is not None])  # noqa: E731
        gm = [t for t in ans if t.get("A_C")]
        gshare = (sum(1 for t in gm if t["generic"]) / len(gm)) if gm else float("nan")
        L.append(f"| {mname} | {len(ts)} | {sum(t['forfeit'] for t in ts)/len(ts):.2f} | {fmt(mean('A_P'),2)} | {fmt(mean('A_C'),2)} | {fmt(mean('Ac_P'),2)} | {fmt(mean('Ac_C'),2)} | "
                 f"{fmt(mean('A_star'),2)} | {fmt(mean('Ac_star'),2)} | {fmt(mean('R_P'),4)} | {fmt(mean('G_P'),4)} | {fmt(mean('S0'),4)} | "
                 f"{fmt(A.smean([1.0 if t['generic'] else 0.0 for t in ans]),2)} | {fmt(gshare,2)} |")
    L.append("")

    # 3. paired separations
    L.append("## 3. Paired separations (miner a − miner b, same turn; both answering unless noted)\n")
    L.append("Rules: `S0` = live min(R,G) at the 4,096 cap; `A_C`/`Ac_C` = agreement with coached refs (raw / centred); "
             "`A_P`/`Ac_P` = with plain refs; `A*`/`Ac*` = with the stored coached action; `V1_w` = S0 + w·Ac_C; "
             "`V2_w` = S0 + w·Ac*; `VP_w` = S0 + w·Ac_P. `z` = paired mean / SE. `forfeits in` = forfeits kept at −0.1 (S0 and V rules only).\n")
    L.append("| a − b | rule | n | mean | z | share > 0 | n (forfeits in) | mean (forfeits in) | z (forfeits in) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    pair_out = {}
    for a_name, b_name in PAIRS:
        for rule in ["S0", "R_P", "G_P", "A_P", "Ac_P", "A_C", "Ac_C", "A_star", "Ac_star"] + [f"V1_{w}" for w in WEIGHTS] + [f"V2_{w}" for w in WEIGHTS] + [f"VP_{w}" for w in WEIGHTS]:
            d, d_all = [], []
            for tid, ms in by_turn.items():
                ta, tb = ms.get(a_name), ms.get(b_name)
                if not ta or not tb:
                    continue
                if rule.startswith(("S0", "V")) and ta.get(rule) is not None and tb.get(rule) is not None:
                    d_all.append(ta[rule] - tb[rule])
                if ta["forfeit"] or tb["forfeit"]:
                    continue
                if ta.get(rule) is None or tb.get(rule) is None:
                    continue
                d.append(ta[rule] - tb[rule])
            s, s_all = zstat(d), zstat(d_all)
            pair_out[f"{a_name}-{b_name}:{rule}"] = {"answered": s, "forfeits_in": s_all}
            if s["n"] == 0 and s_all["n"] == 0:
                continue
            L.append(f"| {a_name} − {b_name} | {rule} | {s['n']} | {fmt(s['mean'],4)} | {fmt(s['z'],2)} | {fmt(s['pos'],2)} | "
                     f"{s_all['n'] or ''} | {fmt(s_all['mean'],4) if s_all['n'] else ''} | {fmt(s_all['z'],2) if s_all['n'] else ''} |")
    L.append("")

    # 4. breakdowns of the headline pairs by dialect / subset / "term can matter"
    L.append("## 4. Headline pairs by dialect, state set, and whether coached refs differ from plain refs\n")
    L.append("| pair | slice | n | S0 z | Ac_C z | V1_0.01 z | Ac* z | V2_0.01 z | Ac_P z | VP_0.01 z |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    pt_by = {pt["turn_id"]: pt for pt in per_turn}
    slices = [("all", lambda pt: True)] + [(f"dialect {k}", (lambda k: lambda pt: pt["action_kind"] == k)(k)) for k in kinds] + \
             [(f"set {s}", (lambda s: lambda pt: pt["subset"] == s)(s)) for s in subsets] + \
             [("coached majority differs from plain", lambda pt: pt["majority_differs"]),
              ("coached majority same as plain", lambda pt: not pt["majority_differs"]),
              ("no action shared between C and P refs", lambda pt: pt["no_shared_action"]),
              ("some action shared", lambda pt: not pt["no_shared_action"])]
    for a_name, b_name in [("coached_coached_4096", "king_live"), ("coached_stored", "king_live"), ("teacher_heldout", "king_live"), ("stored_chal", "stored_king")]:
        for label, pred in slices:
            zs = {}
            for rule in ("S0", "Ac_C", "V1_0.01", "Ac_star", "V2_0.01", "Ac_P", "VP_0.01"):
                d = []
                for tid, ms in by_turn.items():
                    if not pred(pt_by[tid]):
                        continue
                    ta, tb = ms.get(a_name), ms.get(b_name)
                    if not ta or not tb or ta["forfeit"] or tb["forfeit"] or ta.get(rule) is None or tb.get(rule) is None:
                        continue
                    d.append(ta[rule] - tb[rule])
                zs[rule] = zstat(d)
            n = zs["S0"]["n"]
            if n == 0:
                continue
            L.append(f"| {a_name} − {b_name} | {label} | {n} | " + " | ".join(fmt(zs[r]["z"], 2) for r in ("S0", "Ac_C", "V1_0.01", "Ac_star", "V2_0.01", "Ac_P", "VP_0.01")) + " |")
    L.append("")

    # 5. generic credit / synthetic miners
    L.append("## 5. Where the credit goes: generic actions and synthetic miners\n")
    L.append("| slice | n turns | generic refs C | generic refs P | A_C of generic `ls -la` miner | A_P of generic miner | A_C of repeat-last-action miner | A_P of repeat-last | A_C of shrink miner | mean pair_C (mode-guess ceiling) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for label, pred in [("all", lambda pt: True)] + [(f"dialect {k}", (lambda k: lambda pt: pt["action_kind"] == k)(k)) for k in kinds]:
        sub = [pt for pt in per_turn if pred(pt)]
        if not sub:
            continue
        g = lambda s, k: A.smean([pt["synthetic"][s][k] for pt in sub if s in pt["synthetic"] and pt["synthetic"][s].get(k) is not None])  # noqa: E731
        L.append(f"| {label} | {len(sub)} | {fmt(A.smean([pt['generic_refs_C'] for pt in sub if pt['generic_refs_C'] is not None]),2)} | "
                 f"{fmt(A.smean([pt['generic_refs_P'] for pt in sub if pt['generic_refs_P'] is not None]),2)} | "
                 f"{fmt(g('generic','A_C'),2)} | {fmt(g('generic','A_P'),2)} | {fmt(g('repeat_last','A_C'),2)} | {fmt(g('repeat_last','A_P'),2)} | "
                 f"{fmt(g('shrink','A_C'),2)} | {fmt(A.smean([pt['pair_C'] for pt in sub if pt['pair_C'] is not None]),2)} |")
    L.append("")
    (out / "tables.md").write_text("\n".join(L))
    json.dump({"n_turns": len(rows), "pairs": pair_out}, open(out / "summary.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
