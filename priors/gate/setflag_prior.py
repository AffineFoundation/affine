#!/usr/bin/env python3
"""usage: setflag_prior.py FLAG on|off "<why>"  - PROMOTION-GATE.json "heldout" keys of the prior-v1 baseline (RA/20261101-baseline-switch).
on : heldout.baseline = 'prior-v1' + heldout.prior_v1 = the round settings (baseline_v1.DEFAULTS apply to anything not listed).
off: removes both (the gate falls back to the ivw-v2 path at the next round). History is appended either way."""
import json, os, sys, time
os.environ['TZ'] = 'America/Costa_Rica'; time.tzset()
p, mode, why = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(p)); h = d['heldout']
PV = dict(N=500, k_cand=8, head_rollouts=12, head_blocks=3, scoring_mode='separate', cap_rule='task-level', cap_max=0.33, block_delay=3,
          tau_run_separate=0.0, tau_run_same=0.0, sigma_run_floor=0.003)
if mode == 'on':
    h['baseline'] = 'prior-v1'; h['prior_v1'] = dict(PV); new = dict(baseline='prior-v1', prior_v1=PV)
else:
    h.pop('baseline', None); h.pop('prior_v1', None); new = dict(baseline='ivw-v2 (default path)')
hist = h.get('prior_v1_history') or []
hist.append(dict(mode=mode, applies_to='the next gate tick (an open round of the other kind is closed)', set_at=time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()),
                 set_at_cr=time.strftime('%Y-%m-%d %H:%M CR', time.localtime()), values=new, by=why,
                 authorized_by=['Jacob 2026-11-01 14:23-14:26 CR: switch the live gate Baseline to the prior-table design (pt1019u)']))
h['prior_v1_history'] = hist; d['heldout'] = h
t = p + '.tmp.pv'; open(t, 'w').write(json.dumps(d, indent=1, sort_keys=True) + '\n'); os.chmod(t, os.stat(p).st_mode & 0o777); os.replace(t, p)
print('flag %s: %s' % (mode, new))
