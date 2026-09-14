#!/usr/bin/env python
"""E5 diagnostics on the recorded echoes (no GPU).

1. Action agreement. Per arm: does the coached held-out sample's ACTION
   (whitespace-normalized) equal the majority action of the arm's hinted
   refs? Same for the king's live action and the unhinted held-out. Per
   group, fuzzy match of the coached / king action against the pivot judge's
   `should_have` sentence where it exists (token containment >= 0.5 of the
   action's alphanumeric tokens of >= 4 chars — a weak proxy, flagged).
2. Meter check. Unhinted teacher held-out vs king on the R leg alone under
   H0 refs, per group, and length-matched (pairs whose thoughts differ in
   length by <= 25 %); thought-length distributions for coached / king /
   unhinted teacher / recorded / refs.

  python e5_diagnostics.py --run-dir RUN --turns turns.jsonl --out RUN/analysis
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze as A  # noqa: E402

ARMS = ["fact_1792", "fact_4096", "fact_nothink", "mix3", "coached_1792"]
GROUPS = ["all", "king_loop_onset", "king_pivot", "king_fail", "king_recoverable", "king_done",
          "completion_pre", "completion"]
WS = re.compile(r"\s+")
FENCE = re.compile(r"```mswea_bash_command[ \t]*\n")


def norm(y: str) -> str:
    return WS.sub(" ", FENCE.sub("```bash\n", y or "")).strip()


def majority(actions: list[str]) -> str | None:
    acts = [a for a in actions if a]
    if not acts:
        return None
    c = collections.Counter(acts).most_common()
    return c[0][0] if c[0][1] >= 2 or len(acts) == 1 else None


def fuzzy_in(action: str, text: str) -> bool | None:
    if not text or not action:
        return None
    toks = [t for t in re.findall(r"[A-Za-z0-9_]{4,}", action.lower()) if not t.isdigit()]
    if len(toks) < 2:
        return None
    return sum(1 for t in toks if t in text.lower()) / len(toks) >= 0.5


def zstat(d):
    return A.zstat(d)


def fmt(v, d=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--turns", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    turns = {}
    for line in open(args.turns):
        t = json.loads(line)
        turns[t["turn_id"]] = {"pivot": t.get("pivot"), "reference_action": t.get("reference_action")}
    rows = [json.loads(l) for l in open(Path(args.run_dir) / "results.jsonl")]
    rows = [r for r in rows if not r.get("failed") and "H0" in r.get("conditions", {})]
    lines = [f"# E5 diagnostics ({len(rows)} turns)\n"]

    # ---------------------------------------------------------- 1. action agreement
    lines.append("## 1. Action agreement\n")
    lines.append("Majority action = an action shared by >= 2 of the arm's 3 valid refs (else none). "
                 "`= king` compares the coached action with the king's live action. "
                 "`~ should_have` = >= 50 % of the action's tokens (>= 4 chars) appear in the pivot judge's sentence — a weak proxy.\n")
    lines.append("| arm | group | turns with a majority | coached = majority | king = majority | unhinted held-out = majority (H0 refs) | coached ∈ refs | king ∈ refs | coached = king | coached ~ should_have (n) | king ~ should_have (n) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    agree_rows = []
    for arm in ["H0"] + ARMS:
        for g in GROUPS:
            sub = [r for r in rows if (g == "all" or r["group"] == g) and arm in r["conditions"]]
            if not sub:
                continue
            c_maj = k_maj = t_maj = c_in = k_in = c_eq_k = 0
            n_maj = n_c = n_k = n_t = 0
            c_sh = []; k_sh = []
            for r in sub:
                refs = [norm(x["y"]) for x in r["conditions"][arm]["refs"] if x.get("valid")]
                maj = majority(refs)
                coached = r["miners"].get(f"coached_{arm}") if arm != "H0" else r["miners"].get("teacher_heldout")
                king = r["miners"].get("king_live")
                th = r["miners"].get("teacher_heldout")
                ca = norm(coached["y"]) if coached and coached.get("valid") else None
                ka = norm(king["y"]) if king and king.get("valid") else None
                ta = norm(th["y"]) if th and th.get("valid") else None
                if maj is not None:
                    n_maj += 1
                    if ca is not None:
                        n_c += 1; c_maj += (ca == maj); c_in += (ca in refs)
                    if ka is not None:
                        n_k += 1; k_maj += (ka == maj); k_in += (ka in refs)
                    h0refs = [norm(x["y"]) for x in r["conditions"]["H0"]["refs"] if x.get("valid")]
                    m0 = majority(h0refs)
                    if ta is not None and m0 is not None:
                        n_t += 1; t_maj += (ta == m0)
                if ca is not None and ka is not None:
                    c_eq_k += (ca == ka)
                sh = ((turns.get(r["turn_id"]) or {}).get("pivot") or {}).get("should_have")
                if sh:
                    v = fuzzy_in(ca or "", sh)
                    if v is not None:
                        c_sh.append(v)
                    v = fuzzy_in(ka or "", sh)
                    if v is not None:
                        k_sh.append(v)
            row = {"arm": arm, "group": g, "n_turns": len(sub), "n_majority": n_maj,
                   "coached_eq_majority": c_maj / n_c if n_c else float("nan"),
                   "king_eq_majority": k_maj / n_k if n_k else float("nan"),
                   "unhinted_eq_majority_h0": t_maj / n_t if n_t else float("nan"),
                   "coached_in_refs": c_in / n_c if n_c else float("nan"),
                   "king_in_refs": k_in / n_k if n_k else float("nan"),
                   "coached_eq_king": c_eq_k / len(sub),
                   "coached_sh": (st.mean(1.0 if x else 0.0 for x in c_sh) if c_sh else float("nan")), "n_c_sh": len(c_sh),
                   "king_sh": (st.mean(1.0 if x else 0.0 for x in k_sh) if k_sh else float("nan")), "n_k_sh": len(k_sh)}
            agree_rows.append(row)
            lines.append(f"| {arm} | {g} | {n_maj}/{len(sub)} | {fmt(row['coached_eq_majority'])} | {fmt(row['king_eq_majority'])} | "
                         f"{fmt(row['unhinted_eq_majority_h0'])} | {fmt(row['coached_in_refs'])} | {fmt(row['king_in_refs'])} | {fmt(row['coached_eq_king'])} | "
                         f"{fmt(row['coached_sh'])} ({row['n_c_sh']}) | {fmt(row['king_sh'])} ({row['n_k_sh']}) |")

    # ------------------------------------------------- 2. meter check: R leg, H0 refs
    lines.append("\n## 2. Unhinted teacher held-out vs king on the R leg (H0 refs)\n")
    lines.append("| group | n | mean R teacher | mean R king | d_R | z_R | length-matched n (|Δlen| ≤ 25 %) | d_R matched | z_R matched | teacher len p50 | king len p50 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    meter_rows = []
    for g in GROUPS:
        sub = [r for r in rows if g == "all" or r["group"] == g]
        d = []; dm = []; rt = []; rk = []; lt = []; lk = []
        for r in sub:
            tc = A.turn_condition(r, "H0")
            if not tc:
                continue
            a, b = tc["miners"].get("teacher_heldout"), tc["miners"].get("king_live")
            if not a or not b or a["forfeit"] or b["forfeit"] or a.get("R") is None or b.get("R") is None:
                continue
            if math.isnan(a["R"]) or math.isnan(b["R"]):
                continue
            d.append(a["R"] - b["R"]); rt.append(a["R"]); rk.append(b["R"])
            la, lb = a["len_z"], b["len_z"]
            lt.append(la); lk.append(lb)
            if max(la, lb) > 0 and abs(la - lb) / max(la, lb) <= 0.25:
                dm.append(a["R"] - b["R"])
        if len(d) < 3:
            continue
        m, se, z, n = zstat(d)
        mm, sem, zm, nm = zstat(dm) if len(dm) >= 3 else (float("nan"), float("nan"), float("nan"), len(dm))
        meter_rows.append({"group": g, "n": n, "R_teacher": st.mean(rt), "R_king": st.mean(rk), "d_R": m, "z_R": z,
                           "n_matched": nm, "d_R_matched": mm, "z_R_matched": zm,
                           "len_teacher_p50": st.median(lt), "len_king_p50": st.median(lk)})
        lines.append(f"| {g} | {n} | {fmt(st.mean(rt), 4)} | {fmt(st.mean(rk), 4)} | {fmt(m, 4)} | {fmt(z)} | {nm} | {fmt(mm, 4)} | {fmt(zm)} | {fmt(st.median(lt), 0)} | {fmt(st.median(lk), 0)} |")

    # R by thought-length bin (both sides pooled): is R itself a function of length?
    lines.append("\n### R leg vs thought length (H0 refs; teacher held-out and king pooled)\n")
    lines.append("| thought chars | n thoughts | mean R | mean G_0 | share G_0 < 0 |")
    lines.append("|---|---|---|---|---|")
    bins = [(0, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 10**9)]
    binned = collections.defaultdict(list)
    for r in rows:
        tc = A.turn_condition(r, "H0")
        if not tc:
            continue
        for mname in ("teacher_heldout", "king_live"):
            m = tc["miners"].get(mname)
            if m and not m["forfeit"] and m.get("R") is not None and not math.isnan(m["R"]):
                for lo, hi in bins:
                    if lo <= m["len_z"] < hi:
                        binned[(lo, hi)].append((m["R"], m.get("G0")))
    for (lo, hi), vals in sorted(binned.items()):
        g0 = [g for _, g in vals if g is not None]
        lines.append(f"| {lo}–{hi if hi < 10**8 else '∞'} | {len(vals)} | {fmt(st.mean(v for v, _ in vals), 4)} | "
                     f"{fmt(st.mean(g0), 4) if g0 else '–'} | {fmt(st.mean(1.0 if g < 0 else 0.0 for g in g0)) if g0 else '–'} |")

    # ------------------------------------------------- 3. thought lengths
    lines.append("\n## 3. Thought length (chars of z) — p25 / p50 / p75, and share < 80 chars (the contract's median floor)\n")
    lines.append("| thought | n | p25 | p50 | p75 | share < 80 |")
    lines.append("|---|---|---|---|---|---|")
    lens = collections.defaultdict(list)
    for r in rows:
        for mname, m in r["miners"].items():
            if m.get("valid"):
                lens[mname].append(len((m.get("z") or "").strip()))
        for cname, cd in r["conditions"].items():
            for x in cd["refs"]:
                if x.get("valid"):
                    lens[f"refs:{cname}"].append(len((x.get("z") or "").strip()))
    for k in sorted(lens, key=lambda k: (k.startswith("refs"), k)):
        v = lens[k]
        lines.append(f"| {k} | {len(v)} | {fmt(pct(v, .25), 0)} | {fmt(pct(v, .5), 0)} | {fmt(pct(v, .75), 0)} | {fmt(st.mean(1.0 if x < 80 else 0.0 for x in v))} |")

    (out / "e5_diagnostics.md").write_text("\n".join(lines) + "\n")
    json.dump({"agreement": agree_rows, "meter": meter_rows}, open(out / "e5_diagnostics.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
