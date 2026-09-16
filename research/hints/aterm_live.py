#!/usr/bin/env python
"""A-term analysis on stored verdicts (offline, no GPU).

Inputs: a directory of `chal-*.json.gz` verdict records (king_rows,
challenger_rows, teacher_refs). wvk-18 rows carry per-turn `a_match` /
`ref_pair`; older rows get the E5-style offline A_match (research/hints/amatch.py
normalization) from `y_a` and the stored ref actions.

  python aterm_live.py --evals DIR --out OUTDIR
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import math
import os
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "affine"))
sys.path.insert(0, str(HERE))
from affine import score as S  # noqa: E402
from amatch import norm_action, pairwise  # noqa: E402
from e4_analyze import spearman  # noqa: E402

WEIGHTS = (0.005, 0.01, 0.02)
FORFEIT = -0.1
GENERIC = re.compile(r"^(ls( -[a-z]+)?( \S+)?|pwd|cat \S+|git (status|diff|log)( [^|;&]*)?|echo .*|head [^|;&]*|tail [^|;&]*)$")


def kind_of(y: str) -> str:
    y = (y or "").lstrip()
    if y.startswith("```"):
        return "bash"
    if "<tool_call>" in y or y.startswith("[{") or (y.startswith("{") and '"name"' in y[:200]):
        return "tool_call"
    if "\\boxed" in y:
        return "boxed"
    if y.startswith("{") and '"commands"' in y:
        return "terminus_json"
    return "text"


def wvk_of(d: dict) -> str:
    dp = (d.get("verdict") or {}).get("duel_params") or {}
    rows = d.get("king_rows") or []
    if rows and "a_match" in rows[0]:
        return "18"
    if float(dp.get("band_c", 2.0)) >= 4.0:
        return "17"
    return "16"


def side_terms(row, refs, dp):
    tau, bc, bf = dp.get("tau", 0.03), dp.get("band_c", 2.0), dp.get("band_floor", 0.002)
    if S.is_forfeit(row):
        return {"forfeit": True, "S0": FORFEIT, "R": None, "G": None, "A": None, "pair": None,
                "kind": None, "len_y": None, "len_z": None}
    pairs = row["pairs"]
    R = S.centered_reason(pairs, tau)
    G = S.grounding(pairs, bc, bf)
    y = pairs[0].get("y_a") or ""
    kind = kind_of(y)
    A, pair = row.get("a_match"), row.get("ref_pair")
    if A is None and refs and kind != "text":
        ra = [norm_action(r["y"], kind) for r in refs]
        ra = [a for a in ra if a]
        ya = norm_action(y, kind)
        if ra and ya:
            A = sum(1 for a in ra if a == ya) / len(ra)
            pair = pairwise(ra)
    if pair is None and refs and kind != "text":
        ra = [a for a in (norm_action(r["y"], kind) for r in refs) if a]
        pair = pairwise(ra)
    return {"forfeit": False, "S0": min(R, G) if G is not None else R, "R": R, "G": G,
            "A": A, "pair": pair, "kind": kind, "len_y": len(y), "len_z": len(pairs[0].get("z_a") or ""),
            "generic": is_generic(y, kind)}


def is_generic(y: str, kind: str) -> bool:
    """A cheap, task-blind action: bare ls/pwd/cat/git status.../echo/head/tail (bash or a
    bash tool call), or a Terminus batch of one such command."""
    try:
        if kind == "bash":
            cmd = norm_action(y, "bash") or ""
            return bool(GENERIC.match(cmd))
        if kind == "tool_call":
            n = norm_action(y, "tool_call") or ""
            m = re.search(r'"command":\s*"([^"]*)"', n)
            return bool(m and GENERIC.match(m.group(1).strip()))
        if kind == "terminus_json":
            n = json.loads(norm_action(y, "terminus_json") or "{}")
            cmds = [c for c in n.get("commands", []) if c.strip()]
            return len(cmds) <= 1 and (not cmds or bool(GENERIC.match(cmds[0].strip())))
    except Exception:  # noqa: BLE001
        return False
    return False


def scored(t, w):
    if t["forfeit"]:
        return FORFEIT
    if t["A"] is None or t["pair"] is None:
        return t["S0"]
    return t["S0"] + w * (t["A"] - t["pair"])


def zstat(d):
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    m = st.mean(d); se = st.stdev(d) / math.sqrt(n)
    return m, se, (m / se if se > 0 else float("nan")), n


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    verdicts = []
    turns_all = []
    for f in sorted(glob.glob(os.path.join(args.evals, "chal-*.json.gz"))):
        cid = os.path.basename(f)[:-8]
        try:
            d = json.load(gzip.open(f, "rt"))
        except Exception:
            continue
        if not d.get("king_rows") or not d.get("challenger_rows"):
            continue
        v = d["verdict"]; dp = v.get("duel_params") or {}
        wvk = wvk_of(d)
        kr = {r["turn_id"]: r for r in d["king_rows"]}
        cr = {r["turn_id"]: r for r in d["challenger_rows"]}
        refs = d.get("teacher_refs") or {}
        per = []
        for tid in sorted(set(kr) & set(cr)):
            tk = side_terms(kr[tid], refs.get(tid), dp)
            tc = side_terms(cr[tid], refs.get(tid), dp)
            if tk["forfeit"] and tc["forfeit"]:
                continue
            per.append((tid, tk, tc))
            turns_all.append({"cid": cid, "wvk": wvk, "tid": tid, "k": tk, "c": tc})
        stored_margin, stored_se, stored_z = v.get("margin"), v.get("se"), v.get("z")
        stored_win = bool(v.get("challenger_wins"))
        gates_ok = not (v.get("thought_floor_blocked") or v.get("causality_blocked"))
        res = {"cid": cid, "wvk": wvk, "n": len(per), "stored_margin": stored_margin, "stored_z": stored_z,
               "stored_win": stored_win, "min_margin": dp.get("min_margin", 0.002), "k_sigma": dp.get("k_sigma", 2.0)}
        for w in (0.0,) + WEIGHTS:
            dlist = [scored(tc, w) - scored(tk, w) for _, tk, tc in per]
            m, se, z, n = zstat(dlist)
            win = (m > max(res["k_sigma"] * se, res["min_margin"])) and gates_ok
            res[f"m_{w}"], res[f"z_{w}"], res[f"win_{w}"] = m, z, win
        # A separation
        da = [(tc["A"] - tc["pair"]) - (tk["A"] - tk["pair"]) for _, tk, tc in per
              if not tk["forfeit"] and not tc["forfeit"] and tc["A"] is not None and tk["A"] is not None
              and tc["pair"] is not None and tk["pair"] is not None]
        res["zA"] = zstat(da)[2]; res["dA"] = zstat(da)[0]; res["nA"] = len(da)
        for side, key in (("k", 1), ("c", 2)):
            ts = [p[key] for p in per if not p[key]["forfeit"]]
            res[f"R_{side}"] = st.mean(t["R"] for t in ts if t["R"] is not None) if ts else None
            res[f"G_{side}"] = st.mean(t["G"] for t in ts if t["G"] is not None) if ts else None
            aa = [t["A"] for t in ts if t["A"] is not None]
            res[f"A_{side}"] = st.mean(aa) if aa else None
            ac = [t["A"] - t["pair"] for t in ts if t["A"] is not None and t["pair"] is not None]
            res[f"Ac_{side}"] = st.mean(ac) if ac else None
            res[f"forfeit_{side}"] = 1 - len(ts) / len(per) if per else None
        verdicts.append(res)
    json.dump(verdicts, open(out / "aterm_verdicts.json", "w"), indent=1)

    L = [f"# A-term analysis on stored verdicts ({len(verdicts)} scored duels: " +
         ", ".join(f"wvk {k} × {v}" for k, v in sorted(collections.Counter(r['wvk'] for r in verdicts).items())) + ")\n",
         "Rules: S0 = min(R, G) (live); S_w = S0 + w·(A_match − pair) where defined (text turns keep S0); forfeits −0.1. "
         "Decision = margin > max(2·SE, 0.002) with the stored gates. wvk-18 A_match/pair are the live per-turn telemetry; wvk 16/17 use the E5-style offline normalization.\n"]

    # (1) wvk 18 per side / dialect
    L.append("## 1. wvk-18 verdicts — A_match per side and dialect, and what it correlates with\n")
    L.append("| dialect | side | n turns | mean A_match | mean pair | mean centred | ρ(centred, min(R,G)) | ρ(centred, R) | ρ(centred, G) | ρ(centred, len y) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    t18 = [t for t in turns_all if t["wvk"] == "18"]
    for kind in ("bash", "tool_call", "terminus_json", "boxed", "all"):
        for side in ("k", "c"):
            ts = [t[side] for t in t18 if not t[side]["forfeit"] and t[side]["A"] is not None and t[side]["pair"] is not None
                  and (kind == "all" or t[side]["kind"] == kind)]
            if len(ts) < 20:
                continue
            c = [x["A"] - x["pair"] for x in ts]
            L.append(f"| {kind} | {'king' if side == 'k' else 'challenger'} | {len(ts)} | {fmt(st.mean(x['A'] for x in ts))} | {fmt(st.mean(x['pair'] for x in ts))} | {fmt(st.mean(c))} | "
                     f"{fmt(spearman(c, [x['S0'] for x in ts]), 2)} | {fmt(spearman(c, [x['R'] for x in ts]), 2)} | {fmt(spearman(c, [x['G'] for x in ts]), 2)} | {fmt(spearman(c, [x['len_y'] for x in ts]), 2)} |")
    L.append("\n### Does centred A_match separate challenger from king where min(R,G) does not? (per wvk-18 verdict)\n")
    L.append("| verdict | n | stored z (min(R,G)) | z of Δcentred A_match (chal − king) | n with A | mean ΔA | king A / centred | chal A / centred |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in verdicts:
        if r["wvk"] != "18":
            continue
        L.append(f"| {r['cid']} | {r['n']} | {fmt(r['stored_z'], 2)} | {fmt(r['zA'], 2)} | {r['nA']} | {fmt(r['dA'], 4)} | {fmt(r['A_k'])} / {fmt(r['Ac_k'])} | {fmt(r['A_c'])} / {fmt(r['Ac_c'])} |")

    # (2) re-score
    L.append("\n## 2. Re-scoring every verdict under S_w\n")
    L.append("| verdict | wvk | n | stored win | z S0 | z w=0.005 | z w=0.01 | z w=0.02 | margin S0 → w=0.01 → w=0.02 | decision w=0.005 / 0.01 / 0.02 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    flips = collections.Counter()
    for r in verdicts:
        dec = " / ".join("WIN" if r[f"win_{w}"] else "lose" for w in WEIGHTS)
        for w in WEIGHTS:
            flips[(r["wvk"], w)] += int(r[f"win_{w}"] != r["stored_win"])
        L.append(f"| {r['cid']} | {r['wvk']} | {r['n']} | {'WIN' if r['stored_win'] else 'lose'} | {fmt(r['z_0.0'], 2)} | {fmt(r['z_0.005'], 2)} | {fmt(r['z_0.01'], 2)} | {fmt(r['z_0.02'], 2)} | "
                 f"{fmt(r['m_0.0'], 4)} → {fmt(r['m_0.01'], 4)} → {fmt(r['m_0.02'], 4)} | {dec} |")
    L.append("\nDecision flips vs the stored decision: " + "; ".join(f"wvk {k} w={w}: {n}" for (k, w), n in sorted(flips.items())) + ".")
    for wvk in ("16", "17", "18"):
        rs = [r for r in verdicts if r["wvk"] == wvk]
        if not rs:
            continue
        L.append(f"- wvk {wvk} ({len(rs)} duels): mean |z| S0 {fmt(st.mean(abs(r['z_0.0']) for r in rs), 2)} → w 0.01 {fmt(st.mean(abs(r['z_0.01']) for r in rs), 2)} → w 0.02 {fmt(st.mean(abs(r['z_0.02']) for r in rs), 2)}; "
                 f"mean Δz (w 0.02 − S0) {fmt(st.mean(r['z_0.02'] - r['z_0.0'] for r in rs), 2)}; duels where the A term helps the challenger {sum(1 for r in rs if r['z_0.02'] > r['z_0.0'])} / hurts {sum(1 for r in rs if r['z_0.02'] < r['z_0.0'])}.")

    # (3) attack check on live data
    L.append("\n## 3. Attack check on live data — does the centred term reward generic or repeated actions?\n")
    L.append("Generic = a task-blind action: bare `ls` / `pwd` / `cat f` / `git status|diff|log` / `echo` / `head` / `tail` (bash, or a bash tool call), or a Terminus batch of ≤ 1 such command. "
             "`pair` is the ceiling for a miner who guesses the teacher's modal action; the centring subtracts it.\n")
    L.append("| dialect | side | n | share generic | mean centred on generic turns | mean centred on other turns | generic turns with centred > 0 | share of all centred-positive turns that are generic | mean pair |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for kind in ("bash", "tool_call", "terminus_json"):
        for side in ("k", "c"):
            rows = [t[side] for t in turns_all if not t[side]["forfeit"] and t[side]["A"] is not None and t[side]["pair"] is not None and t[side]["kind"] == kind]
            if len(rows) < 20:
                continue
            cen = [x["A"] - x["pair"] for x in rows]; gen = [bool(x.get("generic")) for x in rows]
            pos = [(c, g) for c, g in zip(cen, gen) if c > 0]
            L.append(f"| {kind} | {'king' if side == 'k' else 'challenger'} | {len(rows)} | {fmt(st.mean(1.0 if g else 0 for g in gen))} | "
                     f"{fmt(st.mean(c for c, g in zip(cen, gen) if g)) if any(gen) else '–'} | {fmt(st.mean(c for c, g in zip(cen, gen) if not g))} | "
                     f"{fmt(st.mean(1.0 if c > 0 else 0 for c, g in zip(cen, gen) if g)) if any(gen) else '–'} | "
                     f"{fmt(st.mean(1.0 if g else 0 for c, g in pos)) if pos else '–'} | {fmt(st.mean(x['pair'] for x in rows))} |")

    # (4) Jacob's diagnosis: sound like vs decide like
    L.append("\n## 4. \"Sounds like the teacher, does not decide like it\" — G up, R down?\n")
    L.append("| wvk | duels | challengers with G_c > G_k | of those, R_c < R_k | ρ across duels (G_c − G_k, R_c − R_k) | mean R_c − R_k when G_c > G_k | when G_c ≤ G_k |")
    L.append("|---|---|---|---|---|---|---|")
    for wvk in ("16", "17", "18", "all"):
        rs = [r for r in verdicts if (wvk == "all" or r["wvk"] == wvk) and r["G_c"] is not None and r["G_k"] is not None]
        if not rs:
            continue
        hi = [r for r in rs if r["G_c"] > r["G_k"]]
        lo = [r for r in rs if r["G_c"] <= r["G_k"]]
        L.append(f"| {wvk} | {len(rs)} | {len(hi)} | {sum(1 for r in hi if r['R_c'] < r['R_k'])} | "
                 f"{fmt(spearman([r['G_c'] - r['G_k'] for r in rs], [r['R_c'] - r['R_k'] for r in rs]), 2)} | "
                 f"{fmt(st.mean(r['R_c'] - r['R_k'] for r in hi), 4) if hi else '–'} | {fmt(st.mean(r['R_c'] - r['R_k'] for r in lo), 4) if lo else '–'} |")
    # per-turn within-side correlation of G and R
    for side in ("k", "c"):
        ts = [t[side] for t in turns_all if not t[side]["forfeit"] and t[side]["G"] is not None and t[side]["R"] is not None]
        L.append(f"- per-turn Spearman(G, R) within the {'king' if side == 'k' else 'challenger'} side, all duels: {fmt(spearman([x['G'] for x in ts], [x['R'] for x in ts]), 2)} (n {len(ts)}); "
                 f"Spearman(G, centred A_match): {fmt(spearman([x['G'] for x in ts if x['A'] is not None and x['pair'] is not None], [x['A'] - x['pair'] for x in ts if x['A'] is not None and x['pair'] is not None]), 2)}.")
    (out / "aterm_tables.md").write_text("\n".join(L) + "\n")
    json.dump([{"cid": t["cid"], "wvk": t["wvk"], "tid": t["tid"],
                "k": {k: v for k, v in t["k"].items()}, "c": {k: v for k, v in t["c"].items()}} for t in turns_all],
              open(out / "aterm_turns.json", "w"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
