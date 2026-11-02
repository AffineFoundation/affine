#!/usr/bin/env python3
"""PROMOTION-GATE.json heldout.prior_v1 z bar step-down (RA/20261101-zbar-stepdown). usage: setflag_zbar.py FLAG on|off FROM_EPOCH WHY
on : prior_v1.z_bar_schedule = [[0,1.0],[6,0.5],[12,0.25]], z_bar_clock = since_last_green, z_bar_from_epoch.  off: those removed
(z bar = bar_z = 1.0). Appends prior_v1_history. Atomic write (caller holds FLAG.lock)."""
import json, os, sys, time
p, mode, e, why = sys.argv[1:5]
f = json.load(open(p)); h = f['heldout']; pv = h.setdefault('prior_v1', {})
if mode == 'on':
    pv.update(z_bar_schedule=[[0, 1.0], [6, 0.5], [12, 0.25]], z_bar_clock='since_last_green', z_bar_from_epoch=int(e))
else:
    for k in ('z_bar_schedule', 'z_bar_clock', 'z_bar_from_epoch'): pv.pop(k, None)
h.setdefault('prior_v1_history', []).append(dict(
    by=why, mode=mode, from_epoch=int(e), set_at=time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()),
    set_at_cr=time.strftime('%Y-%m-%d %H:%M CR', time.gmtime(time.time() - 6 * 3600)),
    applies_to='gate decisions from this epoch (decision-time rule; the draw is unchanged)',
    rule=('z bar step-down: GREEN needs z > z_bar; z_bar = 1.00 below 6 h, 0.50 from 6 h, 0.25 from 12 h since the last GREEN (clock: candidate '
          'TRAINING-RESULT time - last GREEN decided_at); reset to 1.00 on GREEN' if mode == 'on' else 'z > bar_z = 1.00'),
    authorized_by=['Jacob 2026-11-01 21:23 CR: "Go with the step-down: z>1, 0.5 after 6h, 0.25 after 12h without a GREEN, reset on GREEN"; 21:25 CR: "Go live with this change as soon as it is written"'],
    values=dict(z_bar_schedule=pv.get('z_bar_schedule'), z_bar_clock=pv.get('z_bar_clock'))))
t = p + '.tmp.zb'; open(t, 'w').write(json.dumps(f, indent=1, sort_keys=True) + '\n'); os.chmod(t, os.stat(p).st_mode & 0o777); os.replace(t, p)
print('flag', mode, json.dumps({k: pv.get(k) for k in ('z_bar_schedule', 'z_bar_clock', 'z_bar_from_epoch')}))
