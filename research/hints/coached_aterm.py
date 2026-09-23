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
P, C = "H0_4096", "coached_4096"  # overridden by --plain / --coached
# Miner families: every held-out draw of an arm is one member; a family's
# per-turn value is the mean over its answered members (pooled draws).
FAMILIES = ["coached_heldout", "coached_stored", "teacher_heldout", "king_live", "recorded",
            "stored_king", "stored_chal", "stored_chal2", "stored_chal3"]
PAIRS = [("coached_heldout", "king_live"), ("coached_stored", "king_live"),
         ("coached_heldout", "recorded"), ("coached_stored", "recorded"), ("teacher_heldout", "king_live"),
         ("teacher_heldout", "recorded"), ("coached_heldout", "teacher_heldout"),
         ("stored_chal", "stored_king"), ("stored_chal2", "stored_king"), ("stored_chal3", "stored_king"),
         ("king_live", "recorded"), ("king_live", "stored_king")]
AGREE = ["A_P", "Ac_P", "A_C", "Ac_C", "A_star", "Ac_star",
         "As_P", "Acs_P", "As_C", "Acs_C", "As_star", "Acs_star", "J_C", "J_P"]
RULES = ["S0", "R_P", "G_P", "R_C"] + AGREE + \
        [f"V1_{w}" for w in WEIGHTS] + [f"V2_{w}" for w in WEIGHTS] + [f"VP_{w}" for w in WEIGHTS] + \
        [f"V1s_{w}" for w in WEIGHTS] + [f"V2s_{w}" for w in WEIGHTS]
TOK = __import__("re").compile(r"[\s|;&()<>'\"=,\[\]{}:]+")
SOFT_J = 0.5


def family_of(mname: str) -> str | None:
    if mname.startswith("coached_") and mname != "coached_stored":
        return "coached_heldout"
    if mname.startswith("teacher_heldout"):
        return "teacher_heldout"
    return mname if mname in FAMILIES else None


def toks(a: str, kind: str) -> set[str]:
    """Token set of a normalised action for soft agreement. tool_call: tool
    names count as tokens too; terminus: the joined command batch."""
    if kind == "tool_call":
        try:
            calls = json.loads(a)
            out = set()
            for name, args in calls:
                out.add(f"tool:{name}")
                out |= {t for v in args.values() for t in TOK.split(str(v)) if t}
            return out
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    if kind == "terminus_json":
        try:
            o = json.loads(a)
            return {t for c in o.get("commands", []) for t in TOK.split(c) if t} | ({"done"} if o.get("done") else set())
        except (json.JSONDecodeError, AttributeError):
            pass
    return {t for t in TOK.split(a) if t}


def jaccard(a: str | None, b: str | None, kind: str) -> float | None:
    if not a or not b:
        return None
    ta, tb = toks(a, kind), toks(b, kind)
    if not ta or not tb:
        return 1.0 if a == b else 0.0
    return len(ta & tb) / len(ta | tb)


def soft_eq(a, b, kind) -> bool:
    j = jaccard(a, b, kind)
    return bool(j is not None and j >= SOFT_J)


def soft_pairwise(acts: list[str], kind: str) -> float | None:
    if len(acts) < 2:
        return None
    n = len(acts)
    return sum(1 for i in range(n) for j in range(i + 1, n) if soft_eq(acts[i], acts[j], kind)) / (n * (n - 1) / 2)


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


def miner_terms(row, mname, m, refsP, refsC, bandP, ystar, pstar, subset_label, extra=None):
    extra = extra or {}
    kind = row["action_kind"]
    t = {"turn_id": row["turn_id"], "miner": mname, "family": family_of(mname), "action_kind": kind, "subset": subset_label,
         "forfeit": not (m.get("valid") and m.get("m") is not None and m.get("z"))}
    if t["forfeit"]:
        for r in RULES:
            t[r] = FORFEIT if r.startswith(("S0", "V")) else None
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
    # soft agreement (token Jaccard >= SOFT_J) and the graded mean Jaccard
    pairPs, pairCs = soft_pairwise(actsP, kind), soft_pairwise(actsC, kind)
    t["As_P"] = (sum(1 for x in actsP if soft_eq(x, a, kind)) / len(actsP)) if actsP and a else (0.0 if actsP else None)
    t["As_C"] = (sum(1 for x in actsC if soft_eq(x, a, kind)) / len(actsC)) if actsC and a else (0.0 if actsC else None)
    t["Acs_P"] = (t["As_P"] - pairPs) if (t["As_P"] is not None and pairPs is not None) else None
    t["Acs_C"] = (t["As_C"] - pairCs) if (t["As_C"] is not None and pairCs is not None) else None
    t["As_star"] = (1.0 if (a and ystar and soft_eq(a, ystar, kind)) else 0.0) if ystar else None
    pstar_s = extra.get("pstar_s")
    t["Acs_star"] = (t["As_star"] - pstar_s) if (t["As_star"] is not None and pstar_s is not None) else None
    t["J_C"] = A.smean([jaccard(x, a, kind) for x in actsC]) if (actsC and a) else (0.0 if actsC else None)
    t["J_P"] = A.smean([jaccard(x, a, kind) for x in actsP]) if (actsP and a) else (0.0 if actsP else None)
    t["match_generic_Cs"] = bool(a and t["generic"] and t["As_C"])
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
        t[f"V1s_{w}"] = (t["S0"] + w * t["Acs_C"]) if (t["S0"] is not None and t["Acs_C"] is not None) else t["S0"]
        t[f"V2s_{w}"] = (t["S0"] + w * t["Acs_star"]) if (t["S0"] is not None and t["Acs_star"] is not None) else t["S0"]
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, nargs="+",
                    help="one or more run dirs; several = replicate runs on the same turns, pooled per turn")
    ap.add_argument("--turns", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plain", default="H0_4096")
    ap.add_argument("--coached", default="coached_4096")
    args = ap.parse_args()
    global P, C
    P, C = args.plain, args.coached
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    turns = {json.loads(l)["turn_id"]: json.loads(l) for l in open(args.turns)}
    rows = []
    for rd in args.run_dir:
        for l in open(Path(rd) / "results.jsonl"):
            r = json.loads(l)
            r["run_dir"] = rd
            rows.append(r)
    rows = [r for r in rows if not r.get("failed") and P in r.get("conditions", {}) and C in r.get("conditions", {})]
    n_states = len({r["turn_id"] for r in rows})
    per_turn, terms = [], []
    for row in rows:
        turn = turns.get(row["turn_id"], {})
        kind = row["action_kind"]
        refsP, refsC = norm_refs(row, P), norm_refs(row, C)
        actsP, actsC = [x for _, _, x in refsP], [x for _, _, x in refsC]
        bandP = A.band([r["lp_thought"] for _, r in refs_of(row, P) if r.get("lp_thought") is not None])
        bandC = A.band([r["lp_thought"] for _, r in refs_of(row, C) if r.get("lp_thought") is not None])
        # a held-out of the unhinted arm is the plain teacher's held-out
        for mname in list(row["miners"]):
            if mname == f"coached_{P}" or mname.startswith(f"coached_{P}_"):
                new = "teacher_heldout" if mname == f"coached_{P}" else "teacher_heldout" + mname.rsplit("_", 1)[1]
                row["miners"][new] = row["miners"].pop(mname)
        cs = (row["miners"].get("coached_stored") or {})
        ystar = norm_action(cs["y"], kind) if cs.get("y") else None
        pstar = (sum(1 for x in actsP if x == ystar) / len(actsP)) if (actsP and ystar) else None
        pstar_s = (sum(1 for x in actsP if soft_eq(x, ystar, kind)) / len(actsP)) if (actsP and ystar) else None
        majP, majC = majority(actsP), majority(actsC)
        coached_meta = (turn.get("coached") or {})
        subset = f"{coached_meta.get('set')}:{coached_meta.get('arm')}:{coached_meta.get('label6')}"
        pt = {
            "turn_id": row["turn_id"], "action_kind": kind, "harness": row.get("harness"), "subset": subset,
            "n_refs_P": len(actsP), "n_refs_C": len(actsC),
            "pair_P": pairwise(actsP), "pair_C": pairwise(actsC),
            "pair_Ps": soft_pairwise(actsP, kind), "pair_Cs": soft_pairwise(actsC, kind),
            "overlap_Cs_in_P": (sum(1 for x in actsC if any(soft_eq(x, y, kind) for y in actsP)) / len(actsC)) if (actsC and actsP) else None,
            "no_shared_action_soft": bool(actsC and actsP and not any(soft_eq(x, y, kind) for x in actsC for y in actsP)),
            "ystar_in_Cs": any(soft_eq(x, ystar, kind) for x in actsC) if (ystar and actsC) else None,
            "ystar_in_Ps": any(soft_eq(x, ystar, kind) for x in actsP) if (ystar and actsP) else None,
            "p_star_s": pstar_s,
            "draw_yield": row.get("draw_yield"),
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
                "As_C": (sum(1 for x in actsC if soft_eq(x, sact, kind)) / len(actsC)) if actsC else None,
                "As_P": (sum(1 for x in actsP if soft_eq(x, sact, kind)) / len(actsP)) if actsP else None,
            }
        per_turn.append(pt)
        for mname, m in row["miners"].items():
            if family_of(mname) is None:
                continue
            terms.append(miner_terms(row, mname, m, refsP, refsC, bandP, ystar, pstar, subset, {"pstar_s": pstar_s}))
    with open(out / "per_turn.jsonl", "w") as f:
        for pt in per_turn:
            f.write(json.dumps(pt) + "\n")
    with open(out / "terms.jsonl", "w") as f:
        for t in terms:
            f.write(json.dumps(t) + "\n")

    # family aggregation per turn: answered-mean of each rule; forfeit share
    fam_members = collections.defaultdict(list)
    for t in terms:
        fam_members[(t["turn_id"], t["family"])].append(t)
    by_turn = collections.defaultdict(dict)
    for (tid, fam), ms in fam_members.items():
        ans = [t for t in ms if not t["forfeit"]]
        agg = {"n_members": len(ms), "n_answered": len(ans), "forfeit": not ans,
               "forfeit_frac": 1 - len(ans) / len(ms), "generic": A.smean([1.0 if t["generic"] else 0.0 for t in ans]) if ans else None}
        for r in RULES:
            vals = [t[r] for t in ans if t.get(r) is not None]
            agg[r] = st.mean(vals) if vals else (FORFEIT if (not ans and r.startswith(("S0", "V"))) else None)
            if r.startswith(("S0", "V")):
                # forfeits-in value: every member counts, forfeits at the floor
                allv = [t[r] if not t["forfeit"] else FORFEIT for t in ms if (t["forfeit"] or t.get(r) is not None)]
                agg[r + "__all"] = st.mean(allv) if allv else None
        by_turn[tid][fam] = agg
    kinds = sorted({pt["action_kind"] for pt in per_turn})
    subsets = sorted({pt["subset"] for pt in per_turn})
    L = []
    L.append(f"# Coached A-term probe — read-out ({n_states} states, {len(rows)} state×run rows over {len(args.run_dir)} run(s); plain arm `{P}`, coached arm `{C}`)\n")
    L.append("Several runs = replicate draws on the same states; per-turn family values pool the draws of every run, so `n` in the paired tables counts STATES, not draws.\n")
    L.append("Agreement: `A` = exact match of the dialect-normalised action; `As` = soft match (token Jaccard ≥ 0.5); "
             "`J` = mean token Jaccard. `_P` = against the plain refs, `_C` = against the coached refs, `_star` = against the "
             "stored coached action y*. `Ac`/`Acs` = centred against the refs' own (exact / soft) pairwise agreement "
             "(for y*: against the plain refs' rate of hitting y*).\n")

    # 1. ref-set diagnostics
    L.append("## 1. Reference sets: plain vs coached\n")
    L.append("| slice | n | valid refs/turn P | C | draws valid P | C | draws cap-hit P | C | pair_P | pair_C | soft pair_P | soft pair_C | identical_C | coached refs inside plain band | majority differs | no shared action (exact) | no shared action (soft) | y* in C refs (soft) | y* in P refs (soft) | p* soft | generic refs P | C |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    def refrow(label, sub):
        if not sub:
            return
        mean = lambda k: A.smean([x[k] for x in sub if x.get(k) is not None])  # noqa: E731
        frac = lambda k: A.smean([1.0 if x[k] else 0.0 for x in sub if x.get(k) is not None])  # noqa: E731
        dy = lambda arm, k: A.smean([x["draw_yield"][arm][k] / x["draw_yield"][arm]["n_draws"] for x in sub if x.get("draw_yield") and arm in x["draw_yield"]])  # noqa: E731
        L.append(f"| {label} | {len(sub)} | {mean('n_refs_P'):.2f} | {mean('n_refs_C'):.2f} | {fmt(dy(P,'n_valid'),2)} | {fmt(dy(C,'n_valid'),2)} | {fmt(dy(P,'n_cap'),2)} | {fmt(dy(C,'n_cap'),2)} | "
                 f"{fmt(mean('pair_P'),2)} | {fmt(mean('pair_C'),2)} | {fmt(mean('pair_Ps'),2)} | {fmt(mean('pair_Cs'),2)} | {fmt(frac('identical_C'),2)} | {fmt(mean('coached_refs_in_bandP'),2)} | {fmt(frac('majority_differs'),2)} | "
                 f"{fmt(frac('no_shared_action'),2)} | {fmt(frac('no_shared_action_soft'),2)} | {fmt(frac('ystar_in_Cs'),2)} | {fmt(frac('ystar_in_Ps'),2)} | {fmt(mean('p_star_s'),2)} | "
                 f"{fmt(mean('generic_refs_P'),2)} | {fmt(mean('generic_refs_C'),2)} |")
    refrow("all", per_turn)
    for k in kinds:
        refrow(f"dialect {k}", [x for x in per_turn if x["action_kind"] == k])
    for s_ in subsets:
        refrow(f"set {s_}", [x for x in per_turn if x["subset"] == s_])
    L.append("")

    # 2. levels per family
    L.append("## 2. Per-miner levels (answered draws; forfeit = share of draws with no parseable action / no thought)\n")
    L.append("| miner | turns | draws | forfeit | A_P | A_C | A* | As_P | As_C | As* | Acs_P | Acs_C | Acs* | J_P | J_C | R_P | G_P | S0 = min(R,G) | generic actions | share of soft C-matches that are generic |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for fam in FAMILIES:
        ts = [t for t in terms if t["family"] == fam]
        if not ts:
            continue
        ans = [t for t in ts if not t["forfeit"]]
        mean = lambda k: A.smean([t[k] for t in ans if t.get(k) is not None])  # noqa: E731
        gm = [t for t in ans if t.get("As_C")]
        gshare = (sum(1 for t in gm if t["generic"]) / len(gm)) if gm else float("nan")
        L.append(f"| {fam} | {len({t['turn_id'] for t in ts})} | {len(ts)} | {sum(t['forfeit'] for t in ts)/len(ts):.2f} | "
                 f"{fmt(mean('A_P'),2)} | {fmt(mean('A_C'),2)} | {fmt(mean('A_star'),2)} | {fmt(mean('As_P'),2)} | {fmt(mean('As_C'),2)} | {fmt(mean('As_star'),2)} | "
                 f"{fmt(mean('Acs_P'),2)} | {fmt(mean('Acs_C'),2)} | {fmt(mean('Acs_star'),2)} | {fmt(mean('J_P'),2)} | {fmt(mean('J_C'),2)} | "
                 f"{fmt(mean('R_P'),4)} | {fmt(mean('G_P'),4)} | {fmt(mean('S0'),4)} | "
                 f"{fmt(A.smean([1.0 if t['generic'] else 0.0 for t in ans]),2)} | {fmt(gshare,2)} |")
    L.append("")

    # 3. paired separations
    HEAD_RULES = ["S0", "R_P", "G_P", "A_C", "Ac_C", "As_C", "Acs_C", "J_C", "A_P", "Ac_P", "As_P", "Acs_P", "J_P", "A_star", "Ac_star", "As_star", "Acs_star"] + \
                 [f"V1_{w}" for w in WEIGHTS] + [f"V1s_{w}" for w in WEIGHTS] + [f"V2_{w}" for w in WEIGHTS] + [f"V2s_{w}" for w in WEIGHTS] + [f"VP_{w}" for w in WEIGHTS]
    L.append("## 3. Paired separations (family a − family b on the same turn; a family's value = mean over its answered draws)\n")
    L.append("`S0` = live min(R,G) at the 4,096 cap (R over plain refs, G from the plain band); `V1_w` = S0 + w·Ac_C, `V1s_w` = S0 + w·Acs_C "
             "(fresh coached refs as target); `V2_w` = S0 + w·Ac*, `V2s_w` = S0 + w·Acs* (single stored coached action as target); "
             "`VP_w` = S0 + w·Ac_P (the live A term). `z` = paired mean / SE over turns. Right block: forfeits kept at −0.1.\n")
    L.append("| a − b | rule | n | mean | z | share > 0 | n (forfeits in) | mean (forfeits in) | z (forfeits in) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    pair_out = {}
    for a_name, b_name in PAIRS:
        for rule in HEAD_RULES:
            d, d_all = [], []
            for tid, ms in by_turn.items():
                ta, tb = ms.get(a_name), ms.get(b_name)
                if not ta or not tb:
                    continue
                if rule.startswith(("S0", "V")) and ta.get(rule + "__all") is not None and tb.get(rule + "__all") is not None:
                    d_all.append(ta[rule + "__all"] - tb[rule + "__all"])
                if ta["forfeit"] or tb["forfeit"] or ta.get(rule) is None or tb.get(rule) is None:
                    continue
                d.append(ta[rule] - tb[rule])
            s_, s_all = zstat(d), zstat(d_all)
            pair_out[f"{a_name}-{b_name}:{rule}"] = {"answered": s_, "forfeits_in": s_all}
            if s_["n"] == 0 and s_all["n"] == 0:
                continue
            L.append(f"| {a_name} − {b_name} | {rule} | {s_['n']} | {fmt(s_['mean'],4)} | {fmt(s_['z'],2)} | {fmt(s_['pos'],2)} | "
                     f"{s_all['n'] or ''} | {fmt(s_all['mean'],4) if s_all['n'] else ''} | {fmt(s_all['z'],2) if s_all['n'] else ''} |")
    L.append("")

    # 4. breakdowns
    SHOW = ["S0", "Acs_C", "V1s_0.01", "Acs_star", "V2s_0.01", "Acs_P", "VP_0.01", "J_C", "J_P"]
    L.append("## 4. Headline pairs by dialect, state set, and whether the coached refs differ from the plain refs (z, answered)\n")
    L.append("| pair | slice | n | " + " | ".join(SHOW) + " |")
    L.append("|---|---|---|" + "---|" * len(SHOW))
    pt_by = {pt["turn_id"]: pt for pt in per_turn}
    slices = [("all", lambda pt: True)] + [(f"dialect {k}", (lambda k: lambda pt: pt["action_kind"] == k)(k)) for k in kinds] + \
             [(f"set {s_}", (lambda s_: lambda pt: pt["subset"] == s_)(s_)) for s_ in subsets] + \
             [("coached majority differs from plain", lambda pt: pt["majority_differs"]),
              ("no soft-shared action between C and P refs", lambda pt: pt["no_shared_action_soft"]),
              ("some soft-shared action", lambda pt: not pt["no_shared_action_soft"])]
    for a_name, b_name in [("coached_heldout", "king_live"), ("coached_stored", "king_live"), ("coached_heldout", "recorded"),
                           ("coached_stored", "recorded"), ("teacher_heldout", "king_live"), ("stored_chal", "stored_king")]:
        for label, pred in slices:
            zs = {}
            for rule in SHOW:
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
            L.append(f"| {a_name} − {b_name} | {label} | {n} | " + " | ".join(fmt(zs[r]["z"], 2) for r in SHOW) + " |")
    L.append("")

    # 5. generic credit / synthetic miners
    L.append("## 5. Where the credit goes: generic actions and synthetic miners (soft agreement)\n")
    L.append("| slice | n turns | generic refs C | generic refs P | generic `ls -la` miner: As_C | As_P | repeat-last-action miner: As_C | As_P | shrink miner: As_C | soft pair_C (mode-guess ceiling) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for label, pred in [("all", lambda pt: True)] + [(f"dialect {k}", (lambda k: lambda pt: pt["action_kind"] == k)(k)) for k in kinds]:
        sub = [pt for pt in per_turn if pred(pt)]
        if not sub:
            continue
        g = lambda s_, k: A.smean([pt["synthetic"][s_][k] for pt in sub if s_ in pt["synthetic"] and pt["synthetic"][s_].get(k) is not None])  # noqa: E731
        L.append(f"| {label} | {len(sub)} | {fmt(A.smean([pt['generic_refs_C'] for pt in sub if pt['generic_refs_C'] is not None]),2)} | "
                 f"{fmt(A.smean([pt['generic_refs_P'] for pt in sub if pt['generic_refs_P'] is not None]),2)} | "
                 f"{fmt(g('generic','As_C'),2)} | {fmt(g('generic','As_P'),2)} | {fmt(g('repeat_last','As_C'),2)} | {fmt(g('repeat_last','As_P'),2)} | "
                 f"{fmt(g('shrink','As_C'),2)} | {fmt(A.smean([pt['pair_Cs'] for pt in sub if pt['pair_Cs'] is not None]),2)} |")
    L.append("")
    (out / "tables.md").write_text("\n".join(L))
    json.dump({"n_states": n_states, "n_rows": len(rows), "runs": args.run_dir, "pairs": pair_out}, open(out / "summary.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
