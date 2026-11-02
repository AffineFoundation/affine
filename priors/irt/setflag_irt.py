#!/usr/bin/env python3
"""usage: setflag_irt.py FLAG on|off "<why>" - heldout.prior_v1.weighting of the prior-v1 baseline (RA/20261102-irt-live).
on : weighting = 'irt-info' (+ IRT knobs). off: removes them (prior-v1 sqrt(E[p(1-p)]) weighting). Applies at the next round OPEN
(each round freezes its settings when it opens). History appended to heldout.irt_history."""
import json, os, sys, time
os.environ['TZ'] = 'America/Costa_Rica'; time.tzset()
p, mode, why = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(p)); h = d['heldout']; pv = h.setdefault('prior_v1', {})
K = dict(weighting='irt-info', a_shrink=0.5, a_lo=0.5, a_hi=2.0, theta_prior_sd=2.0, theta_min_rollouts=200)
if mode == 'on': pv.update(K); new = dict(K)
else:
    for k in K: pv.pop(k, None)
    new = dict(weighting='prior-v1')
hist = h.get('irt_history') or []
hist.append(dict(mode=mode, applies_to='the next prior-v1 round open (an open round keeps the weighting it opened with)',
                 set_at=time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()), set_at_cr=time.strftime('%Y-%m-%d %H:%M CR', time.localtime()), values=new, by=why,
                 authorized_by=['Jacob 2026-11-02 10:23 CR: ship the IRT information-weighted baseline draw live now, no shadow period (pt1083u)']))
h['irt_history'] = hist; d['heldout'] = h
t = p + '.tmp.irt'; open(t, 'w').write(json.dumps(d, indent=1, sort_keys=True) + '\n'); os.chmod(t, os.stat(p).st_mode & 0o777); os.replace(t, p)
print('flag weighting %s: %s' % (mode, new))
