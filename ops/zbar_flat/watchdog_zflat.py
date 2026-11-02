#!/usr/bin/env python3
"""z bar FIXED watchdog (RA/20261102-zbar-flat; takes over the linear watchdog's checks, incl. the cap rule). Auto-revert of ONLY the
fixed-bar change back to the linear drop (activate_zflat.sh MODE=revert) once, then exit, if after T0:
  gate inactive >= 45 s; any Traceback in gate.log, an error event naming prior_gate/baseline_v1/cap/z_bar, or >= 2 error events in 20 min;
  an epoch >= FROM with TRAINING-RESULT +30 min and no DECISION; a fail-closed turn in an epoch >= FROM; a decision event for an
  epoch >= FROM without cap_rule task-level / cap_max 0.33, or whose bar is not exactly 1.0 with a z_bar record kind 'fixed';
  gate.json refreshed > 3 min after T0 without z_bar.current == 1.0. Acts before the prior-v1 baseline watchdog (90 s / 3 errors / 40 min).
Writes state/WATCHDOG-STATUS.json. Runs WD_HOURS (default 24). WD_DRY=1: one pass, print, no revert."""
import json, os, subprocess, sys, time
from pathlib import Path
B = Path('/home/const/subnet120-rewrite/state/root-audits/20261102-zbar-flat')
GS = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state')
LE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs')
T0 = float(sys.argv[1]); FROM = int(sys.argv[2]); END = T0 + 3600 * float(os.environ.get('WD_HOURS', '24'))
os.environ['XDG_RUNTIME_DIR'] = '/run/user/%d' % os.getuid(); os.environ['TZ'] = 'America/Costa_Rica'; time.tzset()
def log(msg):
    with open(B / 'state' / 'watchdog.log', 'a') as f: f.write('%s %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S CR'), msg))
def revert(why):
    log('TRIGGER%s: %s' % (' (WD_DRY, no revert)' if os.environ.get('WD_DRY') == '1' else '', why))
    if os.environ.get('WD_DRY') == '1': print('WOULD REVERT:', why); sys.exit(2)
    subprocess.run(['bash', str(B / 'code' / 'activate_zflat.sh')], env=dict(os.environ, MODE='revert', WHY='watchdog: ' + why, FROM_WATCHDOG='1'), timeout=600)
    sys.exit(0)
inactive = 0; log('start T0=%d FROM=e%d' % (T0, FROM))
while time.time() < END:
    if (B / 'state' / 'REVERTED').exists(): log('REVERTED marker present: exit'); break
    now = time.time(); st = dict(at_cr=time.strftime('%Y-%m-%d %H:%M:%S CR'), t0=T0, from_epoch=FROM)
    if os.environ.get('WD_DRY') != '1':
        a = subprocess.run(['systemctl', '--user', 'is-active', 'affine-promotion-gate'], capture_output=True, text=True).stdout.strip()
        inactive = 0 if a == 'active' else inactive + 1
        if inactive >= 3: revert('gate service %s for >= 45 s' % a)
    try: lines = open(GS / 'gate.log', errors='replace').readlines()[-4000:]
    except Exception: lines = []
    last_at = 0; errs = []; decided = []
    for l in lines:
        try: d = json.loads(l); last_at = d.get('at', last_at)
        except Exception:
            if 'Traceback' in l and last_at >= T0 - 5: revert('Traceback in gate.log after activation')
            continue
        if d.get('at', 0) < T0: continue
        ev = str(d.get('event'))
        if 'error' in ev:
            s = json.dumps(d)
            if any(k in s for k in ('prior_gate', 'baseline_v1', 'cap_', 'z_bar')): revert('gate error naming the gate code: %s' % s[:160])
            if d['at'] >= now - 1200: errs.append(ev)
        if ev in ('heldout_fail_closed', 'prior_fail_closed') and int(d.get('epoch') or 0) >= FROM: revert('fail-closed e%s: %s' % (d.get('epoch'), str(d.get('why'))[:150]))
        if ev == 'decision' and int(d.get('epoch') or 0) >= FROM:
            if d.get('cap_rule') != 'task-level' or abs(float(d.get('cap_max') or 0) - 0.33) > 1e-9: revert('e%s decided without the task-level cap rule' % d.get('epoch'))
            zb = d.get('z_bar') if isinstance(d.get('z_bar'), dict) else {}
            try: okz = float(d['bar']) == 1.0 and float(zb['value']) == 1.0 and zb.get('kind') == 'fixed'
            except Exception: okz = False
            if not okz: revert('e%s decided without the fixed z bar 1.0 (bar %s, z_bar %s)' % (d.get('epoch'), d.get('bar'), str(zb)[:120]))
            decided.append(dict(epoch=d['epoch'], decision=d.get('decision'), z=d.get('z'), z_bar=d.get('bar'), cap=d.get('cap'), at_cr=d.get('at_cr')))
    if len(errs) >= 2: revert('%d gate errors in 20 min: %s' % (len(errs), errs[-1]))
    eps = sorted(int(p.name) for p in LE.iterdir() if p.name.isdigit() and int(p.name) >= FROM)[-4:]
    for e in eps:
        tr = LE / str(e) / 'TRAINING-RESULT.json'
        if tr.exists() and not (GS / 'epochs' / str(e) / 'DECISION.json').exists() and now - tr.stat().st_mtime > 1800:
            revert('epoch stall: e%d TRAINING-RESULT +%d min without DECISION' % (e, (now - tr.stat().st_mtime) / 60))
    try:
        g = json.load(open('/home/const/subnet120/affine/website/gate.json'))
        if g.get('updated_at', 0) > T0 + 180 and (g.get('z_bar') or {}).get('current') != 1.0: revert('gate.json refreshed without z_bar.current 1.0')
        st['gate_json_z_bar'] = {k: (g.get('z_bar') or {}).get(k) for k in ('current', 'kind', 'from_epoch')}
    except (OSError, ValueError): pass
    st.update(errors_20min=errs[-5:], decided=decided[-10:])
    t = B / 'state' / '.WATCHDOG-STATUS.json.tmp'; t.write_text(json.dumps(st, indent=1)); os.replace(t, B / 'state' / 'WATCHDOG-STATUS.json')
    if os.environ.get('WD_DRY') == '1': print(json.dumps(st, indent=1)); break
    time.sleep(15)
log('end')
