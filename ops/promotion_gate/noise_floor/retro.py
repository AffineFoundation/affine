#!/usr/bin/env python3
"""Retro check: re-decide every decided held-out turn with sigma_run = max(FLOOR, per-round estimate) instead of max(0.022, est).
Uses the exact recorded gain / se_task / cap / n of each turn (TURNS.private.jsonl), bar z > 1.0, gain > 0, cap < 0.20, n >= 100."""
import glob, json, os, sys
FLOOR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
H = "/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/heldout"
rows = []
for rd in sorted(glob.glob(H + "/r2*")):
    b = rd + "/BASELINE.private.json"
    if not os.path.exists(b) or not os.path.exists(rd + "/TURNS.private.jsonl"): continue
    est = float(json.load(open(b)).get("sigma_run_est") or 0.0)
    for ln in open(rd + "/TURNS.private.jsonl"):
        t = json.loads(ln); ev = t.get("evaluated") or {}
        if ev.get("se_task") is None: continue
        npb = int(ev.get("n_p_blocks") or 3); sr = max(FLOOR, est)
        se = (ev["se_task"] ** 2 + sr ** 2 * (1 + 1 / npb)) ** 0.5; z = ev["gain"] / se
        new = "green" if (ev["gain"] > 0 and z > 1.0 and (ev.get("c_cap_rate") or 1) < 0.20 and ev["n"] >= 100) else "red"
        rows.append(dict(epoch=t["epoch"], round=os.path.basename(rd), old=t["decision"], new=new, gain_pp=round(100 * ev["gain"], 2),
                         se_old_pp=round(100 * ev["se"], 2), se_task_pp=round(100 * ev["se_task"], 2), sigma_run_new_pp=round(100 * sr, 2),
                         se_new_pp=round(100 * se, 2), z_old=round(ev["z"], 3), z_new=round(z, 3)))
rows.sort(key=lambda r: r["epoch"])
over = [r for r in rows if r["epoch"] >= 293]
flips = [r for r in over if r["old"] == "red" and r["new"] == "green"]
allflips = [r for r in rows if r["old"] != r["new"]]
import statistics as S
print(json.dumps(dict(floor=FLOOR, turns=len(rows), overnight_from_e293=dict(turns=len(over), red=sum(r["old"] == "red" for r in over), flips_to_green=len(flips), flips=flips),
      all_changes=allflips, se_old_median_pp=S.median(r["se_old_pp"] for r in rows), se_new_median_pp=S.median(r["se_new_pp"] for r in rows),
      se_old_range_pp=[min(r["se_old_pp"] for r in rows), max(r["se_old_pp"] for r in rows)], se_new_range_pp=[min(r["se_new_pp"] for r in rows), max(r["se_new_pp"] for r in rows)],
      rows=rows), indent=1))
