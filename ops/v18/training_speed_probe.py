"""Training-speed probe over stored wvk-22/23 verdicts: (1) reward density of
min(z_R, typ_c, z_A) vs soft-min / mean; (2) sequential stopping replay."""
import gzip, json, math, statistics as st, sys, collections, random
import numpy as np
sys.path.insert(0, "/home/const/subnet120/affine")
from evalsrv import sdmeter
from affine import dialects

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
rows = [json.loads(l) for l in open("/home/const/subnet120/affine/state/history.jsonl") if l.strip()]
vs = [r for r in rows if isinstance(r.get("verdict"), dict) and (r["verdict"].get("duel_params") or {}).get("score_mode") == "sd_min_rga"][-N:]
FLOOR = -12.0; DELTA = 0.2

def kind_of(refs, tid):
    ys = [r["y"] for r in refs.get(tid, [])]
    for k in ("tool_call", "boxed", "terminus_json", "bash"):
        if ys and all(dialects.count_actions(y, k) > 0 for y in ys): return k
    return "text"

def legs_of(row, loo, kinds, cfg, tau):
    t = row["turn_id"]
    s = sdmeter.turn_score(sdmeter.side_legs(row, tau, cfg["a_norm_bytes"]), loo.mu.get(t), loo.sigma.get(kinds.get(t)), cfg)
    if s["bind"] == "forfeit": return None
    if s["z_R"] is None or s["typ_c"] is None or s["z_A"] is None: return None
    return (s["z_R"], s["typ_c"], s["z_A"])

def combine(legs, mode, tau=0.5):
    if legs is None: return FLOOR
    x = np.array(legs)
    if mode == "min": return float(x.min())
    if mode == "mean": return float(x.mean())
    if mode == "softmin":
        m = x.min(); return float(m - tau * math.log(np.mean(np.exp(-(x - m) / tau))))
    raise ValueError

# ---------- part 1 + data for part 2
gaps, bind, ge1, wmax = [], collections.Counter(), 0, []
ss_forfeit = ss_total = 0.0; n_forf = n_all = 0
per_verdict = []
for r in vs:
    cid = r["challenge_id"]; v = r["verdict"]
    art = json.load(gzip.open(f"/home/const/subnet120/affine/state/evals/{cid}.json.gz"))
    refs = art["teacher_refs"]; kinds = {t: kind_of(refs, t) for t in refs}
    cfg = sdmeter.settings({"sd_meter": {**v["duel_params"]["sd_meter"], "shadow": True}}); tau = v["duel_params"]["tau"]
    loo = sdmeter.loo_anchors(refs, kinds, tau, cfg["a_norm_bytes"])
    order = [t for t in art["turn_ids"]]
    cl = {x["turn_id"]: legs_of(x, loo, kinds, cfg, tau) if x.get("valid") else None for x in art["challenger_rows"]}
    kl = {x["turn_id"]: legs_of(x, loo, kinds, cfg, tau) if x.get("valid") else None for x in art["king_rows"]}
    cvalid = {x["turn_id"]: bool(x.get("valid")) for x in art["challenger_rows"]}; kvalid = {x["turn_id"]: bool(x.get("valid")) for x in art["king_rows"]}
    # density stats on valid turns (both sides)
    scores_all = []
    for side, valid in ((cl, cvalid), (kl, kvalid)):
        for t, L in side.items():
            sc = combine(L, "min"); scores_all.append((sc, valid.get(t, False) and L is not None))
            if L is None: continue
            x = sorted(L); gap = x[1] - x[0]; gaps.append(gap); ge1 += gap >= 1.0
            bind[("R", "Gc", "A")[int(np.argmin(L))]] += 1
            w = np.exp(-(np.array(L) - min(L)) / 0.5); wmax.append(float(w.max() / w.sum()))
    mu_all = st.mean(s for s, _ in scores_all)
    for s, ok in scores_all:
        ss_total += (s - mu_all) ** 2; n_all += 1
        if not ok: ss_forfeit += (s - mu_all) ** 2; n_forf += 1
    # paired diffs in slice order per combiner (drop turns missing legs on a valid side; both-forfeit = 0)
    diffs = {m: [] for m in ("min", "softmin", "mean")}
    for t in order:
        if t not in cl or t not in kl: continue
        cf, kf = not cvalid.get(t, False), not kvalid.get(t, False)
        if (not cf and cl[t] is None) or (not kf and kl[t] is None): continue
        for m in diffs:
            a = FLOOR if cf else combine(cl[t], m); b = FLOOR if kf else combine(kl[t], m)
            diffs[m].append(0.0 if (cf and kf) else a - b)
    per_verdict.append((cid, r.get("uid"), r["event"], v["margin"], v["se"], v["z"], v["challenger_wins"], diffs))

def decide(d):
    n = len(d); m = st.mean(d); se = st.stdev(d) / math.sqrt(n); return m, se, m > max(2 * se, DELTA)

lines = []
P = lambda *a: (print(*a), lines.append(" ".join(str(x) for x in a)))
P(f"# Training-speed probe — {len(vs)} wvk-22/23 verdicts ({vs[0]['challenge_id']} … {vs[-1]['challenge_id']}), {n_all} side-turns")
P("\n## 1. Reward density under min(z_R, typ_c, z_A)\n")
g = np.array(gaps)
P(f"- binding leg (valid side-turns, n={len(gaps)}): R {bind['R']/len(gaps):.2f} / Gc {bind['Gc']/len(gaps):.2f} / A {bind['A']/len(gaps):.2f}")
P(f"- margin of the binding leg over the next leg: p25 {np.quantile(g,.25):.2f} / p50 {np.quantile(g,.5):.2f} / p75 {np.quantile(g,.75):.2f} / p90 {np.quantile(g,.9):.2f} sd; **gap ≥ 1 sd on {ge1/len(gaps):.2f} of turns** (the other two legs give zero gradient there); gap < 0.25 sd on {np.mean(g<0.25):.2f}")
P(f"- soft-min τ = 0.5: mean weight on the binding leg {st.mean(wmax):.2f} (min = 1.00, mean = 0.33)")
P(f"- forfeits: {n_forf/n_all:.4f} of side-turns carry {ss_forfeit/ss_total:.2f} of the total per-turn score variance (floor −12)")
P("\n| combiner | sd of valid per-turn score | sd of paired diff | verdict flips vs live (of %d) | crowns |" % len(per_verdict))
P("|---|---|---|---|---|")
for m in ("min", "softmin", "mean"):
    sds, sdd, flips, crowns = [], [], 0, 0
    for cid, uid, ev, lm, lse, lz, lw, diffs in per_verdict:
        d = diffs[m]; mm, se, w = decide(d); sdd.append(st.stdev(d)); crowns += w
        live_w = decide(diffs["min"])[2]; flips += (w != live_w)
    # valid per-turn score sd: recompute cheaply from diffs is wrong; use side scores: approximate via combine on stored legs
    P(f"| {m} | see below | {st.mean(sdd):.2f} | {flips} | {crowns} |")
# per-turn valid score sd per combiner (both sides)
vals = {m: [] for m in ("min", "softmin", "mean")}
for r in vs:
    pass
P("\nPer-verdict decisions (margin / z under each combiner; live rule = min):\n")
P("| duel | uid | live | min | soft-min τ0.5 | mean |")
P("|---|---|---|---|---|---|")
for cid, uid, ev, lm, lse, lz, lw, diffs in per_verdict:
    cells = []
    for m in ("min", "softmin", "mean"):
        mm, se, w = decide(diffs[m]); cells.append(f"{mm:+.3f} z {mm/se:+.1f}{' **crown**' if w else ''}")
    P(f"| {cid} | {uid} | {lm:+.3f} z {lz:+.1f}{' **crown**' if lw else ''} | " + " | ".join(cells) + " |")

# ---------- part 2: sequential stopping
P("\n## 2. Sequential stopping (looks every 100 turns, slice order)\n")
LOOKS = list(range(100, 1001, 100))
def seq(d, k):
    for n in LOOKS:
        if n > len(d): break
        x = d[:n]; m = st.mean(x); se = st.stdev(x) / math.sqrt(n); bar = max(k * se, DELTA)
        if m > bar: return n, True
        if m + k * se < DELTA: return n, False   # futility: bar unreachable
    m, se, w = decide(d); return len(d), w
rng = random.Random(0)
P("| k per look | turns-to-decision p50 / p90 | agreement with full-1000 | crowns seq / full | false-crown rate under null (sign-flip permutations) | duel min p50 / p90 (37 min per 1000 + 8 load) | verdicts/day |")
P("|---|---|---|---|---|---|---|")
for k in (2.0, 2.6, 2.83):
    ns, agree, cs, cf, fp, trials = [], 0, 0, 0, 0, 0
    for cid, uid, ev, lm, lse, lz, lw, diffs in per_verdict:
        d = diffs["min"]; n, w = seq(d, k); full = decide(d)[2]
        ns.append(n); agree += (w == full); cs += w; cf += full
        for _ in range(40):  # null: random sign flips of the paired diffs
            dd = [x * rng.choice((-1, 1)) for x in d]; trials += 1; fp += seq(dd, k)[1]
    p50, p90 = np.quantile(ns, .5), np.quantile(ns, .9)
    P(f"| {k} | {p50:.0f} / {p90:.0f} | {agree}/{len(per_verdict)} | {cs} / {cf} | {fp/trials:.3f} per duel | {8+37*p50/1000:.0f} / {8+37*p90/1000:.0f} | {1440/(8+37*st.mean(ns)/1000):.0f} (full 1000: {1440/45:.0f}) |")
P(f"\nnull false-crown for the full-1000 rule (same permutations): " + str(round(sum(decide([x * rng.choice((-1,1)) for x in pv[7]['min']])[2] for pv in per_verdict for _ in range(40)) / (40 * len(per_verdict)), 3)))
open("/home/const/subnet120/ops/v18/training_speed_probe.md", "w").write("\n".join(lines) + "\n")
