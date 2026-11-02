#!/usr/bin/env python3
"""Offline replay of prior-v1 DECISIONs with the old (live) and new (task-level cap, CAP_HIT_MAX) estimate. Prints aggregates only.
usage: replay_cap.py OLD_CODE_DIR NEW_CODE_DIR EPOCH..."""
import importlib.util, json, sys
from pathlib import Path
GS = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state')
def load(d, name):
    sp = importlib.util.spec_from_file_location(name, str(Path(d) / 'baseline_v1.py')); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
OLD, NEW = load(sys.argv[1], 'bv_old'), load(sys.argv[2], 'bv_new')
rj = lambda p: json.load(open(p))
ok = True
for e in map(int, sys.argv[3:]):
    d = GS / 'epochs' / str(e); dec = rj(d / 'DECISION.json'); rid = dec['heldout']['round_id']; rd = GS / 'heldout' / rid
    r = rj(rd / 'ROUND.private.json'); pv = dict(r['cfg']); PI = rj(rd / 'PI.private.json')
    drawn = {t: dict(env=PI['env'][t], pi=PI['pi'][t]) for t in r['ids']}
    bl = rj(rd / 'BASELINE.private.json')['per_task']
    cpt = {OLD.task_of(int(k), pv): v for k, v in rj(d / 'C-PROBE.private.json')['per_task'].items() if isinstance(v, dict) and v.get('n')}
    kw = dict(shares=pv['shares'], tau=float(dec['heldout']['tau_run']), sigma_run_floor=float(pv['sigma_run_floor']), bar_z=float(pv['bar_z']),
              z_inclusive=bool(pv['z_inclusive']), min_scored=int(pv['min_scored']), min_cover=float(pv['min_cover']))
    o = OLD.estimate(drawn, bl, cpt, r['pool']['per_env'], cap_max=float(pv['cap_max']), cap_envs=tuple(pv['cap_envs']), **kw)
    n = NEW.estimate(drawn, bl, cpt, r['pool']['per_env'], cap_max=NEW.CAP_HIT_MAX, **kw)
    same_old = (o['decision'] == dec['decision'] and abs(o['evaluated']['z'] - dec['evaluated']['z']) < 1e-9)
    ev = n['evaluated']; ok &= same_old and n['decision'] == dec['decision']
    print('e%d live=%s old-replay=%s(match=%s) new=%s z=%.3f | task-level cap c=%.4f (%d/%d) h=%.4f max=%.2f | per-env c/h %s | new reasons %s' % (
        e, dec['decision'], o['decision'], same_old, n['decision'], ev['z'], ev['c_cap_rate'], ev['c_cap_hits'], ev['c_rollouts'], ev['h_cap_rate'], ev['cap_hit_max'],
        {k: (round(v['c_cap'], 4), round(v['h_cap'], 4)) for k, v in ev['per_env'].items()}, n['reasons']))
print('ALL OUTCOMES UNCHANGED' if ok else 'OUTCOME CHANGE / MISMATCH')
