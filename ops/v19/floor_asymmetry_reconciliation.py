"""Reconcile the floor-asymmetry decomposition on the wvk-22/23 crowns and a fully
matched (2-ref) R/A teacher control."""
import gzip, json, math, statistics as st, sys
import numpy as np
sys.path.insert(0, "/home/const/wvk24_stage/affine")
from evalsrv import sdmeter
from affine import dialects
from affine.score import reason, action_lift

rows = [json.loads(l) for l in open("/home/const/subnet120/affine/state/history.jsonl") if l.strip()]
byc = {}
for r in rows:
    if isinstance(r.get("verdict"), dict): byc[r["challenge_id"]] = r
CROWNS = [("16", "chal-00588"), ("17", "chal-00598"), ("18", "chal-00602"), ("19", "chal-00606"), ("20", "chal-00614"), ("21", "chal-00662")]
RECENT = [r["challenge_id"] for r in rows if isinstance(r.get("verdict"), dict) and (r["verdict"].get("duel_params") or {}).get("score_mode") == "sd_min_rga"][-30:]

def kind_of(refs, tid):
    ys = [r["y"] for r in refs.get(tid, [])]
    for k in ("tool_call", "boxed", "terminus_json", "bash"):
        if ys and all(dialects.count_actions(y, k) > 0 for y in ys): return k
    return "text"

def lme(v, tau):
    m = max(v); return m + tau * math.log(st.mean(math.exp((x - m) / tau) for x in v))

def analyse(cid, floor, sym):
    """Return per-turn diffs under: floor (−12 / −6) and sym = symmetric miner empty-thought rule
    (typ leg dropped when < 10 content tokens instead of floored)."""
    r = byc[cid]; v = r["verdict"]
    art = json.load(gzip.open(f"/home/const/subnet120/affine/state/evals/{cid}.json.gz")); refs = art["teacher_refs"]
    kinds = {t: kind_of(refs, t) for t in refs}
    cfg = sdmeter.settings({"sd_meter": {**v["duel_params"]["sd_meter"], "shadow": True, "forfeit_sd": floor,
                                          "content_min_tokens": (0 if sym else 10)}})
    tau = v["duel_params"]["tau"]; a_norm = cfg["a_norm_bytes"]
    loo = sdmeter.loo_anchors(refs, kinds, tau, a_norm, cfg["ref_min_content"], cfg["typ_min_refs"])
    def sc(row):
        t = row["turn_id"]; legs = sdmeter.side_legs(row, tau, a_norm)
        if sym and legs is not None and (legs.get("n_content") or 0) < 10:
            legs = dict(legs); legs["mc"] = None          # typ leg dropped → min(z_R, z_A)
        return sdmeter.turn_score(legs, loo.mu.get(t), loo.sigma.get(kinds.get(t)), cfg), legs
    cs = {x["turn_id"]: sc(x) for x in art["challenger_rows"]}; ks = {x["turn_id"]: sc(x) for x in art["king_rows"]}
    diffs, cat = [], []
    for t in sorted(set(cs) & set(ks)):
        (a, la), (b, lb) = cs[t], ks[t]
        if a["score"] is None or b["score"] is None: continue
        af, bf = a["bind"] == "forfeit", b["bind"] == "forfeit"
        ae = (la is not None and (la.get("n_content") or 0) < 10); be = (lb is not None and (lb.get("n_content") or 0) < 10)
        d = 0.0 if (af and bf) else a["score"] - b["score"]
        diffs.append(d); cat.append("forfeit" if (af or bf) else "empty" if (ae or be) else "normal")
        if ae and not af: pass
    diffs = np.array(diffs); cat = np.array(cat)
    n = len(diffs); m = diffs.mean(); se = diffs.std(ddof=1) / math.sqrt(n)
    contrib = {c: diffs[cat == c].sum() / n for c in ("forfeit", "empty", "normal")}
    counts = {c: int((cat == c).sum()) for c in ("forfeit", "empty", "normal")}
    # empty-thought counts per side
    ce = sum(1 for a, la in cs.values() if la is not None and (la.get("n_content") or 0) < 10)
    ke = sum(1 for b, lb in ks.values() if lb is not None and (lb.get("n_content") or 0) < 10)
    return dict(margin=m, se=se, z=m / se, crown=m > max(2 * se, 0.2), contrib=contrib, counts=counts, empty=(ce, ke),
                forfeits=(sum(1 for a, _ in cs.values() if a["bind"] == "forfeit"), sum(1 for b, _ in ks.values() if b["bind"] == "forfeit")))

L = []
P = lambda *a: (print(*a), L.append(" ".join(str(x) for x in a)))
P("## A. Crown margins decomposed by turn class (paired chal − king, sd units; contribution = class sum / n)\n")
P("| reign | duel | uid | live stamp | rule replayed | margin / SE / z | crown | forfeit turns (c/k) → contrib | empty-thought turns (c/k) → contrib | normal turns contrib |")
P("|---|---|---|---|---|---|---|---|---|---|")
for reign, cid in CROWNS:
    v = byc[cid]["verdict"]
    for label, floor, sym in (("as judged (floor −12)", -12.0, False), ("wvk-24 floor −6", -6.0, False), ("−6 + symmetric empty-thought rule", -6.0, True)):
        a = analyse(cid, floor, sym)
        P(f"| {reign} | {cid} | {byc[cid].get('uid')} | {v['margin']:+.3f} z {v['z']:+.2f} | {label} | {a['margin']:+.3f} / {a['se']:.3f} / {a['z']:+.2f} | {'**yes**' if a['crown'] else 'no'} | {a['forfeits'][0]}/{a['forfeits'][1]} → {a['contrib']['forfeit']:+.3f} | {a['empty'][0]}/{a['empty'][1]} ({a['counts']['empty']} paired) → {a['contrib']['empty']:+.3f} | {a['contrib']['normal']:+.3f} |")

# B. fully matched R / A control on the recent window + crowns
P("\n## B. Teacher-vs-king control on R and A: k-mismatched vs fully matched (2 references on both sides)\n")
P("| duel | R: king(3 refs) − teacher(2 refs) | R: both 2 refs | A: king(3) − teacher(2) | A: both 2 refs |")
P("|---|---|---|---|---|")
agg = {k: [] for k in ("R3", "R2", "A3", "A2")}
for cid in sorted(set(RECENT) | {c for _, c in CROWNS}):
    r = byc[cid]; v = r["verdict"]; tau = v["duel_params"]["tau"]; a_norm = v["duel_params"]["sd_meter"]["a_norm_bytes"]
    art = json.load(gzip.open(f"/home/const/subnet120/affine/state/evals/{cid}.json.gz")); refs = art["teacher_refs"]
    kinds = {t: kind_of(refs, t) for t in refs}
    loo = sdmeter.loo_anchors(refs, kinds, tau, a_norm)
    d = {k: [] for k in agg}
    for row in art["king_rows"]:
        t = row["turn_id"]; rl = refs.get(t); lt = sdmeter.ref_loo_terms(rl, tau, a_norm) if rl else None
        if not row.get("valid") or lt is None or t not in loo.mu: continue
        sig = loo.sigma.get(kinds.get(t)) or {}
        if not sig.get("R") or not sig.get("A"): continue
        pairs = row["pairs"]; k = len(pairs)
        if k != 3: continue
        a_i = [reason(p) for p in pairs]; b_i = [action_lift(p, a_norm) for p in pairs]
        if any(b is None for b in b_i): continue
        R3 = lme(a_i, tau) - st.mean(a_i); A3 = lme(b_i, tau)
        for j in range(3):
            oth = [i for i in range(3) if i != j]
            muR = st.mean(lt["R"][i] for i in oth); muA = st.mean(lt["A"][i] for i in oth)
            tR = (lt["R"][j] - muR) / sig["R"]; tA = (lt["A"][j] - muA) / sig["A"]
            kR3 = (R3 - muR) / sig["R"]; kA3 = (A3 - muA) / sig["A"]
            a2 = [a_i[i] for i in oth]; b2 = [b_i[i] for i in oth]
            kR2 = ((lme(a2, tau) - st.mean(a2)) - muR) / sig["R"]; kA2 = (lme(b2, tau) - muA) / sig["A"]
            d["R3"].append(tR - kR3); d["R2"].append(tR - kR2); d["A3"].append(tA - kA3); d["A2"].append(tA - kA2)
    cells = []
    for k in ("R3", "R2", "A3", "A2"):
        x = np.array(d[k]); m = x.mean(); z = m / (x.std(ddof=1) / math.sqrt(len(x))); agg[k].append((m, z)); cells.append(f"{m:+.3f} (z {z:+.1f})")
    P(f"| {cid}{' (crown)' if cid in dict((c, r) for r, c in CROWNS) else ''} | " + " | ".join(cells) + " |")
P("\nmedians over the table: " + "; ".join(f"{k}: {st.median(m for m, _ in agg[k]):+.3f} (z {st.median(z for _, z in agg[k]):+.1f}, {sum(1 for m, _ in agg[k] if m > 0)} pos / {sum(1 for m, _ in agg[k] if m < 0)} neg)" for k in agg))
open("/home/const/subnet120/ops/v19/floor_asymmetry_reconciliation.md", "w").write("\n".join(L) + "\n")
