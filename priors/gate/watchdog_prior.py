#!/usr/bin/env python3
"""prior-v1 baseline watchdog (RA/20261101-baseline-switch). Auto-revert (activate_baseline.sh MODE=revert) once, then exit, if:
  gate service inactive >= 90 s; >= 3 error events / tracebacks in gate.log within 20 min; an epoch with TRAINING-RESULT +40 min
  and no DECISION; a prior-v1 round whose draw is not done 35 min after its upload deadline passed; a prior-v1 baseline with no
  progress (BASELINE-TIMELINE last_progress) for > 35 min while not done; a fail-closed turn in a prior-v1 round.
Writes state/WATCHDOG-STATUS.json each pass. Runs 72 h (WD_HOURS)."""
import json, os, subprocess, sys, time
from pathlib import Path
B = Path('/home/const/subnet120-rewrite/state/root-audits/20261101-baseline-switch'); GS = Path(os.environ.get('HG_GATE_STATE', '/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state'))
LE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs'); H = GS / 'heldout'; STALL = 2100
T0 = float(sys.argv[1]) if len(sys.argv) > 1 else time.time(); END = T0 + 3600 * float(os.environ.get('WD_HOURS', '72'))
os.environ['XDG_RUNTIME_DIR'] = '/run/user/%d' % os.getuid(); os.environ['TZ'] = 'America/Costa_Rica'; time.tzset()
def rj(p):
    try: return json.load(open(p))
    except Exception: return None
def log(msg):
    with open(B / 'state' / 'watchdog.log', 'a') as f: f.write('%s %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S CR'), msg))
def revert(why):
    log('TRIGGER%s: %s' % (' (WD_DRY test, no revert)' if os.environ.get('WD_DRY') == '1' else '', why))
    if os.environ.get('WD_DRY') == '1': print('WOULD REVERT:', why); sys.exit(2)
    with open(B / 'PROGRESS.md', 'a') as f: f.write('- %s: prior-v1 watchdog TRIGGER -> revert: %s\n' % (time.strftime('%H:%M CR'), why))
    subprocess.run(['bash', str(B / 'code' / 'activate_baseline.sh')], env=dict(os.environ, MODE='revert', WHY='watchdog: ' + why, FROM_WATCHDOG='1'), timeout=900)
    sys.exit(0)
inactive = 0; log('start T0=%d' % T0)
while time.time() < END:
    if (B / 'state' / 'REVERTED').exists(): log('REVERTED marker present: exit'); break
    now = time.time(); st = dict(at_cr=time.strftime('%Y-%m-%d %H:%M:%S CR'), t0=T0)
    if os.environ.get('WD_DRY') != '1':
        a = subprocess.run(['systemctl', '--user', 'is-active', 'affine-promotion-gate'], capture_output=True, text=True).stdout.strip()
        inactive = 0 if a == 'active' else inactive + 1
        if inactive >= 3: revert('gate service %s for >= 90 s' % a)
    errs = []
    try: lines = open(GS / 'gate.log', errors='replace').readlines()[-4000:]
    except Exception: lines = []
    last_at = 0
    for l in lines:
        try: d = json.loads(l); last_at = d.get('at', last_at)
        except Exception:
            if 'Traceback' in l and last_at >= max(T0, now - 1200): errs.append('traceback')
            continue
        if d.get('at', 0) >= max(T0, now - 1200) and 'error' in str(d.get('event')): errs.append('%s: %s' % (d['event'], str(d.get('error'))[:120]))
    st['errors_20min'] = errs[-5:]
    if len(errs) >= 3: revert('%d gate errors in 20 min: %s' % (len(errs), errs[-1]))
    eps = sorted((int(p.name) for p in (GS / 'epochs').iterdir() if p.name.isdigit()))[-4:] if (GS / 'epochs').exists() else []
    for e in eps:
        tr = LE / str(e) / 'TRAINING-RESULT.json'
        if tr.exists() and tr.stat().st_mtime > T0 - 600 and not (GS / 'epochs' / str(e) / 'DECISION.json').exists() and now - tr.stat().st_mtime > 2400:
            revert('epoch stall: e%d TRAINING-RESULT +%d min without DECISION' % (e, (now - tr.stat().st_mtime) / 60))
    rounds = []
    for p in sorted(H.glob('r2*')):
        r = rj(p / 'ROUND.private.json') or {}
        if r.get('baseline') != 'prior-v1' or r.get('opened_at', 0) < T0: continue
        closed = (p / 'CLOSED.json').exists(); b = rj(p / 'BASELINE.private.json'); btl = rj(p / 'BASELINE-TIMELINE.json') or {}
        row = dict(id=r['id'], opened_cr=r.get('opened_at_cr'), closed=closed, drawn=r.get('n') if r.get('ids') is not None else None,
                   target_block=r.get('target_block'), baseline_done=bool(b), last_progress_min=(round((now - btl['last_progress']) / 60, 1) if btl.get('last_progress') else None))
        if not closed:
            dl = r.get('deadline')
            if dl and now > float(dl) + STALL and r.get('ids') is None: revert('round %s draw not done %d min after the upload deadline' % (r['id'], (now - float(dl)) / 60))
            if not b and btl.get('last_progress') and now - btl['last_progress'] > STALL: revert('round %s baseline: no progress for %d min' % (r['id'], (now - btl['last_progress']) / 60))
        for l in lines:
            if '"heldout_fail_closed"' in l or '"prior_fail_closed"' in l:
                try: d = json.loads(l)
                except Exception: continue
                s = rj(GS / 'epochs' / str(d.get('epoch')) / 'SAMPLE.private.json') or {}
                if s.get('round_id') == r['id'] and d.get('at', 0) >= T0: revert('fail-closed e%s in prior-v1 round %s: %s' % (d.get('epoch'), r['id'], str(d.get('why'))[:150]))
        rounds.append(row)
    st['prior_rounds'] = rounds
    t = B / 'state' / '.WATCHDOG-STATUS.json.tmp'; t.write_text(json.dumps(st, indent=1)); os.replace(t, B / 'state' / 'WATCHDOG-STATUS.json')
    if os.environ.get('WD_DRY') == '1': print(json.dumps(st, indent=1)); break
    time.sleep(30)
log('end')
