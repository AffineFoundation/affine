#!/usr/bin/env python3
"""Re-derive the held-out gate run-noise (sigma_run) from production head replicate blocks (RA/20261031-gate-noise-floor, Jacob 2026-10-31).
Per round: blocks 1-3 = 3 independent k4 runs of the head on the kept tasks. (a) pooled run-level sd of the weighted mean rate,
(b) leave-one-block-out A/A: block j (k4, candidate role) vs p_i from the other two blocks, z with se_task only and with the floor."""
import glob, json, os, sys, statistics as S
GC = "/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate"
sys.path.insert(0, GC + "/code"); import heldout as HO
H = GC + "/state/heldout"
out = dict(rounds={}); sv_minus = []; aa = []
for rd in sorted(glob.glob(H + "/r2*")):
    rid = os.path.basename(rd)
    if not os.path.exists(rd + "/BASELINE.private.json"): continue
    b = json.load(open(rd + "/BASELINE.private.json")); kept = b["kept"]; sel = b["selection"]
    blk = {i: json.load(open(rd + "/BLOCK-%d.private.json" % i))["per_task"] for i in (1, 2, 3)}
    ms, vs = [], []
    for pt in blk.values():
        num = W = v = 0.0
        for t in kept:
            sc = sel.get(str(t)); c = pt.get(str(t))
            if not (sc and c and c.get("n")): continue
            w = HO.weight(sc); x = c["c"] / c["n"]; num += w * x; W += w; v += w * w * x * (1 - x) / max(1, c["n"] - 1)
        ms.append(num / W); vs.append(v / (W * W))
    m = sum(ms) / 3; sv = sum((x - m) ** 2 for x in ms) / 2; d = sv - sum(vs) / 3; sv_minus.append(d)
    lo = []
    for j in (1, 2, 3):
        rest = [k for k in (1, 2, 3) if k != j]
        pb = HO.merged_baseline({str(k): blk[k] for k in rest}, kept)
        r0 = HO.score_ivw(sel, pb, blk[j], kept, [], env="affine_math", set_size=1000, sigma_run=0.0, n_p_blocks=2)["evaluated"]
        r1 = HO.score_ivw(sel, pb, blk[j], kept, [], env="affine_math", set_size=1000, sigma_run=0.022, n_p_blocks=2)["evaluated"]
        lo.append(dict(block=j, gain_pp=round(100 * r0["gain"], 3), se_task_pp=round(100 * r0["se_task"], 3), z_task=round(r0["z"], 3), z_floor022=round(r1["z"], 3)))
        aa.append(lo[-1])
    out["rounds"][rid] = dict(kept=len(kept), block_means_pct=[round(100 * x, 3) for x in ms], within_var_sd_pp=round(100 * (sum(vs) / 3) ** 0.5, 3),
                             between_sd_pp=round(100 * sv ** 0.5, 3), sigma_run_est_pp=round(100 * max(0, d) ** 0.5, 3), loo=lo)
pooled = sum(sv_minus) / len(sv_minus)
zt = [a["z_task"] for a in aa]; zf = [a["z_floor022"] for a in aa]; gp = [a["gain_pp"] for a in aa]; st = [a["se_task_pp"] for a in aa]
out["pooled"] = dict(rounds=len(sv_minus), df=2 * len(sv_minus), pooled_excess_var_pp2=round(1e4 * pooled, 4),
                     sigma_run_pooled_pp=round(100 * max(0.0, pooled) ** 0.5, 3),
                     ratio_between_over_within=round(S.mean([1e4 * (x) for x in sv_minus]), 4))
out["aa_loo"] = dict(n=len(aa), gain_sd_pp=round((sum(g * g for g in gp) / len(gp)) ** 0.5, 3), se_task_mean_pp=round(S.mean(st), 3),
                     z_task_mean=round(S.mean(zt), 3), z_task_rms=round((sum(z * z for z in zt) / len(zt)) ** 0.5, 3), z_task_gt1=sum(z > 1 for z in zt),
                     z_floor_rms=round((sum(z * z for z in zf) / len(zf)) ** 0.5, 3), z_floor_gt1=sum(z > 1 for z in zf))
print(json.dumps(out, indent=1))
