#!/usr/bin/env python3
"""PROMOTION-GATE.json heldout.prior_v1 z bar: step-down -> linear drop (RA/20261101-zbar-linear). usage: setflag_zlin.py FLAG on|off FROM_EPOCH WHY
on : z_bar_schedule = {kind: linear, start: 1.0, floor: 0.25, hours_to_floor: 12}, z_bar_linear_from_epoch.  off: the step schedule
[[0,1.0],[6,0.5],[12,0.25]] back (z_bar_clock unchanged). Appends prior_v1_history. Atomic write (caller holds FLAG.lock)."""
import json, os, sys, time
p, mode, e, why = sys.argv[1:5]
f = json.load(open(p)); h = f['heldout']; pv = h.setdefault('prior_v1', {})
if pv.get('z_bar_clock') != 'since_last_green': raise SystemExit('z bar step-down not on (z_bar_clock missing)')
if mode == 'on': pv.update(z_bar_schedule=dict(kind='linear', start=1.0, floor=0.25, hours_to_floor=12.0), z_bar_linear_from_epoch=int(e))
else: pv['z_bar_schedule'] = [[0, 1.0], [6, 0.5], [12, 0.25]]; pv.pop('z_bar_linear_from_epoch', None)
h.setdefault('prior_v1_history', []).append(dict(
    by=why, mode=mode, from_epoch=int(e), set_at=time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()),
    set_at_cr=time.strftime('%Y-%m-%d %H:%M CR', time.gmtime(time.time() - 6 * 3600)),
    applies_to='gate decisions from this epoch (decision-time rule; the draw is unchanged)',
    rule=('z bar linear drop: z_bar = max(0.25, 1.0 - 0.75 * h / 12), h = hours since the last GREEN (clock: candidate TRAINING-RESULT time - last '
          'GREEN decided_at); reset to 1.00 on GREEN' if mode == 'on' else 'z bar step-down: 1.00 / 0.50 from 6 h / 0.25 from 12 h since the last GREEN'),
    authorized_by=['Jacob 2026-11-01 21:51 CR: replace the step-down with a linear slow drop, z_bar(h) = max(0.25, 1.0 - 0.75 * h/12), reset to 1.0 on GREEN'],
    values=dict(z_bar_schedule=pv.get('z_bar_schedule'))))
t = p + '.tmp.zl'; open(t, 'w').write(json.dumps(f, indent=1, sort_keys=True) + '\n'); os.chmod(t, os.stat(p).st_mode & 0o777); os.replace(t, p)
print('flag', mode, json.dumps({k: pv.get(k) for k in ('z_bar_schedule', 'z_bar_clock', 'z_bar_linear_from_epoch')}))
