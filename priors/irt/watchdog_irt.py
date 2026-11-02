#!/usr/bin/env python3
"""irt-info draw watchdog (RA/20261102-irt-live). Auto-revert to the prior-v1 weighting (activate_irt.sh MODE=revert) once, then exit, if
after T0: a prior_irt_fallback event (IRT weights could not be computed: that round already used prior-v1); an irt-info round whose
expected draw size is not N +- 1 or whose draw has < 400 or > 600 tasks; its draw not done 30 min after the upload deadline; its head
baseline without progress for > 30 min; a fail-closed turn in it; >= 3 gate error events / tracebacks in 20 min. (The prior-v1 watchdog,
which reverts the whole baseline, uses 35 min / 3 errors: this one acts first and only on the weighting.)
Also records the first irt-info round (draw stats, baseline, first decided turn) in state/FIRST-ROUND.json. Runs 72 h (WD_HOURS)."""
import json, os, subprocess, sys, time
from pathlib import Path
I = Path('/home/const/subnet120-rewrite/state/root-audits/20261102-irt-live'); GS = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state')
H = GS / 'heldout'; STALL = 1800
T0 = float(sys.argv[1]) if len(sys.argv) > 1 else time.time(); END = T0 + 3600 * float(os.environ.get('WD_HOURS', '72'))
os.environ['XDG_RUNTIME_DIR'] = '/run/user/%d' % os.getuid(); os.environ['TZ'] = 'America/Costa_Rica'; time.tzset()
def rj(p):
    try: return json.load(open(p))
    except Exception: return None
def log(msg):
    with open(I / 'state' / 'watchdog.log', 'a') as f: f.write('%s %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S CR'), msg))
def revert(why):
    log('TRIGGER%s: %s' % (' (WD_DRY, no revert)' if os.environ.get('WD_DRY') == '1' else '', why))
    if os.environ.get('WD_DRY') == '1': print('WOULD REVERT:', why); sys.exit(2)
    subprocess.run(['bash', str(I / 'code' / 'activate_irt.sh')], env=dict(os.environ, MODE='revert', WHY='watchdog: ' + why, FROM_WATCHDOG='1'), timeout=900)
    sys.exit(0)
log('start T0=%d' % T0)
while time.time() < END:
    if (I / 'state' / 'REVERTED').exists(): log('REVERTED marker present: exit'); break
    now = time.time(); st = dict(at_cr=time.strftime('%Y-%m-%d %H:%M:%S CR'), t0=T0)
    try: lines = open(GS / 'gate.log', errors='replace').readlines()[-4000:]
    except Exception: lines = []
    errs = []; last_at = 0; fb = []; fails = []
    for l in lines:
        try: d = json.loads(l); last_at = d.get('at', last_at)
        except Exception:
            if 'Traceback' in l and last_at >= max(T0, now - 1200): errs.append('traceback')
            continue
        at = d.get('at', 0); ev = str(d.get('event'))
        if at >= max(T0, now - 1200) and 'error' in ev: errs.append('%s: %s' % (ev, str(d.get('error'))[:120]))
        if at >= T0 and ev == 'prior_irt_fallback': fb.append(d)
        if at >= T0 and ev in ('heldout_fail_closed', 'prior_fail_closed'): fails.append(d)
    st['errors_20min'] = errs[-5:]
    if fb: revert('IRT weights failed at round open (%s): %s' % (fb[-1].get('id'), str(fb[-1].get('why'))[:200]))
    if len(errs) >= 3: revert('%d gate errors in 20 min: %s' % (len(errs), errs[-1]))
    rounds = []
    for p in sorted(H.glob('r2*'))[-20:]:
        r = rj(p / 'ROUND.private.json') or {}
        if r.get('weighting') != 'irt-info' or r.get('opened_at', 0) < T0: continue
        closed = rj(p / 'CLOSED.json'); b = rj(p / 'BASELINE.private.json'); btl = rj(p / 'BASELINE-TIMELINE.json') or {}
        turns = []
        try: turns = [json.loads(l) for l in open(p / 'TURNS.private.jsonl') if l.strip()]
        except Exception: pass
        row = dict(id=r['id'], head=r['head'][:12], opened_cr=r.get('opened_at_cr'), opened_epoch=r.get('opened_epoch'), expected_n=r.get('expected_n'),
                   drawn=r.get('n'), per_env_drawn=r.get('per_env_drawn'), drawn_at_cr=r.get('drawn_at_cr'), target_block=r.get('target_block'),
                   theta_source=(r.get('irt') or {}).get('theta_source'), theta_rollouts=(r.get('irt') or {}).get('theta_rollouts'),
                   baseline_done=bool(b), baseline_done_at_cr=(b or {}).get('done_at_cr'), closed=(closed or {}).get('reason'),
                   turns=[dict(epoch=t.get('epoch'), decision=t.get('decision'), z=t.get('z'), gain=(t.get('evaluated') or {}).get('gain'),
                               n=(t.get('evaluated') or {}).get('n'), cap=(t.get('evaluated') or {}).get('c_cap_rate'), reasons=t.get('reasons')) for t in turns])
        rounds.append(row)
        if abs(float(r.get('expected_n') or 0) - float(r['cfg'].get('N', 500))) > 1: revert('round %s expected draw size %s != N' % (r['id'], r.get('expected_n')))
        if r.get('ids') is not None and not 400 <= len(r['ids']) <= 600: revert('round %s drew %d tasks' % (r['id'], len(r['ids'])))
        if not closed:
            dl = r.get('deadline')
            if dl and now > float(dl) + STALL and r.get('ids') is None: revert('round %s draw not done %d min after the upload deadline' % (r['id'], (now - float(dl)) / 60))
            if not b and btl.get('last_progress') and now - btl['last_progress'] > STALL: revert('round %s baseline: no progress for %d min' % (r['id'], (now - btl['last_progress']) / 60))
            if r.get('ids') is not None and not b and not btl.get('last_progress') and now - float(r.get('drawn_at') or now) > STALL:
                revert('round %s baseline never started %d min after the draw' % (r['id'], (now - float(r['drawn_at'])) / 60))
        for d in fails:
            s = rj(GS / 'epochs' / str(d.get('epoch')) / 'SAMPLE.private.json') or {}
            if s.get('round_id') == r['id']: revert('fail-closed e%s in irt round %s: %s' % (d.get('epoch'), r['id'], str(d.get('why'))[:150]))
    st['irt_rounds'] = rounds
    if rounds and not (I / 'state' / 'FIRST-ROUND.json').exists() and rounds[0]['baseline_done'] and any(t['decision'] in ('red', 'green') for t in rounds[0]['turns']):
        (I / 'state' / 'FIRST-ROUND.json').write_text(json.dumps(dict(rounds[0], recorded_at_cr=st['at_cr']), indent=1)); log('first irt round verified: %s' % rounds[0]['id'])
    t = I / 'state' / '.WATCHDOG-STATUS.json.tmp'; t.write_text(json.dumps(st, indent=1)); os.replace(t, I / 'state' / 'WATCHDOG-STATUS.json')
    if os.environ.get('WD_DRY') == '1': print(json.dumps(st, indent=1)[:3000]); break
    time.sleep(30)
log('end')
