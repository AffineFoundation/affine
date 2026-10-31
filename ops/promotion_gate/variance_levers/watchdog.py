#!/usr/bin/env python3
"""Variance-levers watchdog (RA/20261031-variance-levers). Auto-revert (activate_levers.sh MODE=revert) once, then exit, if:
  gate service inactive >= 90 s; >= 3 error events / tracebacks in gate.log within 20 min; an epoch with TRAINING-RESULT +40 min and
  no DECISION; a NEW-rules round (ROUND has sigma_run_policy) whose baseline is not done 35 min after it opened, that fails closed,
  or whose decided turn did not use k_cand 8 / the round's frozen sigma_run. Writes state/WATCHDOG-STATUS.json each pass. Runs 24 h."""
import json, os, subprocess, sys, time
from pathlib import Path
VL = Path('/home/const/subnet120-rewrite/state/root-audits/20261031-variance-levers'); GS = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state')
LE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs'); H = GS / 'heldout'
T0 = float(sys.argv[1]) if len(sys.argv) > 1 else time.time(); END = T0 + 86400
os.environ['XDG_RUNTIME_DIR'] = '/run/user/%d' % os.getuid(); os.environ['TZ'] = 'America/Costa_Rica'; time.tzset()
def rj(p):
    try: return json.load(open(p))
    except Exception: return None
def log(msg):
    with open(VL / 'state' / 'watchdog.log', 'a') as f: f.write('%s %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S CR'), msg))
def revert(why):
    log('TRIGGER: ' + why)
    with open(VL / 'PROGRESS.md', 'a') as f: f.write('- %s: watchdog TRIGGER -> revert: %s\n' % (time.strftime('%H:%M CR'), why))
    subprocess.run(['bash', str(VL / 'activate_levers.sh')], env=dict(os.environ, MODE='revert', WHY='watchdog: ' + why), timeout=600)
    sys.exit(0)
inactive = 0; log('start T0=%d' % T0)
while time.time() < END:
    if (VL / 'state' / 'REVERTED').exists(): log('REVERTED marker present: exit'); break
    now = time.time(); st = dict(at_cr=time.strftime('%Y-%m-%d %H:%M:%S CR'), t0=T0)
    a = subprocess.run(['systemctl', '--user', 'is-active', 'affine-promotion-gate'], capture_output=True, text=True).stdout.strip()
    inactive = 0 if a == 'active' else inactive + 1
    if inactive >= 3: revert('gate service %s for >= 90 s' % a)
    errs = []
    try:
        lines = open(GS / 'gate.log', errors='replace').readlines()[-4000:]
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
    eps = sorted((int(p.name) for p in (GS / 'epochs').iterdir() if p.name.isdigit()))[-4:]
    for e in eps:
        tr = LE / str(e) / 'TRAINING-RESULT.json'
        if tr.exists() and tr.stat().st_mtime > T0 - 600 and not (GS / 'epochs' / str(e) / 'DECISION.json').exists() and now - tr.stat().st_mtime > 2400:
            revert('epoch stall: e%d TRAINING-RESULT +%d min without DECISION' % (e, (now - tr.stat().st_mtime) / 60))
    rounds = []
    for p in sorted(H.glob('r2*')):
        r = rj(p / 'ROUND.private.json') or {}
        if not isinstance(r.get('sigma_run_policy'), dict) or r.get('opened_at', 0) < T0: continue
        b = rj(p / 'BASELINE.private.json'); closed = (p / 'CLOSED.json').exists(); pol = r['sigma_run_policy']
        row = dict(id=r['id'], n=r['n'], k_cand=r['k_cand'], sigma_value=pol.get('value'), opened_cr=r['opened_at_cr'], closed=closed,
                   baseline_s=((b or {}).get('timing') or {}).get('total_s'), kept=len((b or {}).get('kept') or []) or None, turns=[])
        if not b and not closed and now - r['opened_at'] > 2100: revert('new round %s baseline not done %d min after open' % (r['id'], (now - r['opened_at']) / 60))
        for l in lines:
            if '"heldout_fail_closed"' in l:
                try: d = json.loads(l)
                except Exception: continue
                s = rj(GS / 'epochs' / str(d.get('epoch')) / 'SAMPLE.private.json') or {}
                if s.get('round_id') == r['id']: revert('fail-closed e%s in new round %s: %s' % (d.get('epoch'), r['id'], str(d.get('why'))[:150]))
        try: turns = [json.loads(x) for x in open(p / 'TURNS.private.jsonl')]
        except Exception: turns = []
        for t in turns:
            e = t.get('epoch'); dd = rj(GS / 'epochs' / str(e) / 'DECISION.json') or {}; ev = t.get('evaluated') or {}
            if t.get('fail_closed') or dd.get('heldout', {}).get('not_judged'): row['turns'].append(dict(epoch=e, decision=t.get('decision'), note='fail-closed/not judged')); continue
            k = (dd.get('heldout') or {}).get('k_cand'); sr = ev.get('sigma_run')
            row['turns'].append(dict(epoch=e, decision=t.get('decision'), z=ev.get('z'), gain=ev.get('gain'), se=ev.get('se'), se_task=ev.get('se_task'),
                                     sigma_run=sr, n=ev.get('n'), k_cand=k, result_to_decision_s=(dd.get('timing') or {}).get('result_to_decision_s'),
                                     run_s=(dd.get('timing') or {}).get('run_s')))
            if k != r['k_cand'] or (pol.get('value') is not None and abs(float(sr or 0) - float(pol['value'])) > 1e-12):
                revert('turn e%s of new round used k_cand %s / sigma_run %s (round: %s / %s)' % (e, k, sr, r['k_cand'], pol.get('value')))
        rounds.append(row)
    st['new_rounds'] = rounds
    t = VL / 'state' / '.WATCHDOG-STATUS.json.tmp'; t.write_text(json.dumps(st, indent=1)); os.replace(t, VL / 'state' / 'WATCHDOG-STATUS.json')
    time.sleep(30)
log('end')
