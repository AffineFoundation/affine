#!/usr/bin/env python3
"""PROMOTION-GATE.json heldout.prior_v1 z bar: time-relaxed -> fixed z > 1 (RA/20261102-zbar-flat). usage: setflag_zflat.py FLAG on|off FROM_EPOCH WHY [BACKUP_FLAG]
on : drops z_bar_schedule / z_bar_clock / z_bar_from_epoch / z_bar_linear_from_epoch, sets bar_z = 1.0 and z_bar_fixed_from_epoch.
off: puts those keys back from BACKUP_FLAG (the pre-install copy) and drops z_bar_fixed_from_epoch. Appends prior_v1_history.
Atomic write (caller holds FLAG.lock)."""
import json, os, sys, time
p, mode, e, why = sys.argv[1:5]
KEYS = ('z_bar_schedule', 'z_bar_clock', 'z_bar_from_epoch', 'z_bar_linear_from_epoch', 'bar_z')
f = json.load(open(p)); h = f['heldout']; pv = h.setdefault('prior_v1', {})
if mode == 'on':
    for k in KEYS: pv.pop(k, None)
    pv.update(bar_z=1.0, z_bar_fixed_from_epoch=int(e))
else:
    old = json.load(open(sys.argv[5]))['heldout'].get('prior_v1', {})
    pv.pop('z_bar_fixed_from_epoch', None)
    for k in KEYS:
        pv.pop(k, None)
        if k in old: pv[k] = old[k]
h.setdefault('prior_v1_history', []).append(dict(
    by=why, mode=mode, from_epoch=int(e), set_at=time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()),
    set_at_cr=time.strftime('%Y-%m-%d %H:%M CR', time.gmtime(time.time() - 6 * 3600)),
    applies_to='gate decisions from this epoch (decision-time rule; the draw is unchanged)',
    rule=('z bar fixed: GREEN needs z > 1.00 (the time-relaxed bar is removed)' if mode == 'on' else 'z bar back to the time-relaxed linear drop'),
    authorized_by=['Jacob 2026-11-02 08:20 CR (pt1055u): remove the z bar reduction, keep it flat at 1 going forward'],
    values={k: pv.get(k) for k in KEYS + ('z_bar_fixed_from_epoch',)}))
t = p + '.tmp.zf'; open(t, 'w').write(json.dumps(f, indent=1, sort_keys=True) + '\n'); os.chmod(t, os.stat(p).st_mode & 0o777); os.replace(t, p)
print('flag', mode, json.dumps({k: pv.get(k) for k in KEYS + ('z_bar_fixed_from_epoch',)}))
