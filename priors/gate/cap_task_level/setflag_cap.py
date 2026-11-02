#!/usr/bin/env python3
"""PROMOTION-GATE.json heldout.prior_v1 cap rule (RA/20261101-cap-task-level). usage: setflag_cap.py FLAG on|off FROM_EPOCH WHY
on : prior_v1.cap_rule = task-level, cap_max = 0.33, cap_envs removed.  off: cap_max = 0.20, cap_envs = [affine_math], cap_rule removed.
Appends prior_v1_history. Atomic write (caller holds FLAG.lock)."""
import json, os, sys, time
p, mode, e, why = sys.argv[1:5]
f = json.load(open(p)); h = f['heldout']; pv = h.setdefault('prior_v1', {})
if mode == 'on':
    pv.pop('cap_envs', None); pv['cap_rule'] = 'task-level'; pv['cap_max'] = 0.33
    vals = dict(cap_rule='task-level', cap_max=0.33, cap_envs=None)
else:
    pv.pop('cap_rule', None); pv['cap_max'] = 0.20; pv['cap_envs'] = ['affine_math']
    vals = dict(cap_rule=None, cap_max=0.20, cap_envs=['affine_math'])
h.setdefault('prior_v1_history', []).append(dict(
    by=why, mode=mode, from_epoch=int(e), set_at=time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()), set_at_cr=time.strftime('%Y-%m-%d %H:%M CR'),
    applies_to='gate decisions from this epoch (decision-time rule; the draw is unchanged)',
    rule=('RED if >= 33% of the candidate\'s baseline answers hit the token cap (task-level: capped / all candidate rollouts on the scored tasks)'
          if mode == 'on' else 'estimated affine_math cap-hit < 0.20'),
    authorized_by=['Jacob 2026-11-01 19:42-19:43 CR (pt1032u): "Make it a flat 33%?"; one task-level rule over the whole baseline, no env-specific rules'],
    values=vals))
t = p + '.tmp.cap'; open(t, 'w').write(json.dumps(f, indent=1, sort_keys=True) + '\n'); os.chmod(t, os.stat(p).st_mode & 0o777); os.replace(t, p)
print('flag', mode, json.dumps({k: pv.get(k) for k in ('cap_rule', 'cap_max', 'cap_envs')}))
