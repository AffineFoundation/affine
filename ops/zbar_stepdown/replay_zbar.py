#!/usr/bin/env python3
"""Offline replay of prior-v1 DECISIONs under the z bar step-down (clock = candidate TRAINING-RESULT time - last GREEN decided_at).
Uses the new code dir's estimate with z_bar; prints aggregates only. usage: replay_zbar.py NEW_CODE_DIR EPOCH..."""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1]); sys.path.insert(1, '/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/code')
import baseline_v1 as BV, prior_gate as PG
GS = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state'); rj = lambda p: json.load(open(p))
pv0 = dict(z_bar_schedule=BV.Z_BAR_SCHEDULE, z_bar_clock='since_last_green'); flips = []
for e in map(int, sys.argv[2:]):
    d = GS / 'epochs' / str(e); dec = rj(d / 'DECISION.json'); rid = dec['heldout']['round_id']; rd = GS / 'heldout' / rid
    r = rj(rd / 'ROUND.private.json'); pv = dict(r['cfg']); PI = rj(rd / 'PI.private.json'); tl = rj(d / 'TIMELINE.json')
    drawn = {t: dict(env=PI['env'][t], pi=PI['pi'][t]) for t in r['ids']}; bl = rj(rd / 'BASELINE.private.json')['per_task']
    cpt = {BV.task_of(int(k), pv): v for k, v in rj(d / 'C-PROBE.private.json')['per_task'].items() if isinstance(v, dict) and v.get('n')}
    zb = PG.z_bar_for(pv0, GS / 'epochs', e, tl['result_mtime'], 1.0)
    n = BV.estimate(drawn, bl, cpt, r['pool']['per_env'], shares=pv['shares'], tau=float(dec['heldout']['tau_run']), sigma_run_floor=float(pv['sigma_run_floor']),
                    bar_z=zb['value'], z_inclusive=bool(pv['z_inclusive']), cap_max=BV.CAP_HIT_MAX, min_scored=int(pv['min_scored']), min_cover=float(pv['min_cover']))
    if n['decision'] != dec['decision']: flips.append(e)
    print('e%d live=%s new=%s z=%.3f z_bar=%.2f h_since_green=%.2f (last GREEN e%s) tr=%s | reasons %s' % (
        e, dec['decision'], n['decision'], n['evaluated']['z'], zb['value'], zb['hours_since_green'], zb['last_green_epoch'], PG._crs(tl['result_mtime']), n['reasons']))
print('FLIPS: %s' % (flips or 'none'))
