#!/usr/bin/env python3
"""usage: setflag.py FLAG on|off "<why>"  - variance-levers keys of PROMOTION-GATE.json "heldout" (applies to rounds opened after)."""
import json, os, sys, time
os.environ['TZ'] = 'America/Costa_Rica'; time.tzset()
p, mode, why = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(p)); h = d['heldout']
new = dict(n=2000, k_cand=8, sigma_run_policy='pooled', sigma_pool_rounds=20, sigma_pool_min_rounds=5) if mode == 'on' else \
      dict(n=1000, k_cand=4, sigma_run_policy='per-round')
h.update(new)
if mode == 'off':
    for k in ('sigma_pool_rounds', 'sigma_pool_min_rounds'): h.pop(k, None)
hist = h.get('variance_levers_history') or []
hist.append(dict(mode=mode, applies_to='rounds opened after set_at', set_at=time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()),
                 set_at_cr=time.strftime('%Y-%m-%d %H:%M CR', time.localtime()), values=new, by=why,
                 authorized_by=['Jacob 2026-10-31 14:07 CR: "Go live with pooled sigma_run", "Go live with 8 candidate rollouts per task", '
                                '"Go live with 2,000 tasks per round"']))
h['variance_levers_history'] = hist; d['heldout'] = h
t = p + '.tmp.vl'; open(t, 'w').write(json.dumps(d, indent=1, sort_keys=True) + '\n'); os.chmod(t, os.stat(p).st_mode & 0o777); os.replace(t, p)
print('flag %s: %s' % (mode, new))
