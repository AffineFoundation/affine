#!/usr/bin/env python3
"""RA/20261102-12k-everywhere 12k-watchdog: auto-revert the 12K switch (MODE=revert FORCE=1 activate_12k.sh) when, from epoch FROM on,
  (a) the trainer log (pod owned-worker-v7.log after byte OFF) shows a CUDA out-of-memory error;
  (b) an epoch fails: RECOVERY-REQUIRED.json, TRAINING-RESULT status != updated, or a non-finite update norm;
  (c) a training job stays open longer than STALL_MIN (default 90) minutes.
env FROM, OFF, UNTIL (unix)."""
import json, math, os, subprocess, sys, time
from pathlib import Path
RA = Path('/home/const/subnet120-rewrite/state/root-audits'); D = RA / '20261102-12k-everywhere'; AV = D / 'activation'
E = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs')
TSH = RA / '20261026-rollback-189de412/scripts/rssh.sh'; PODLOG = '/workspace/affine-mimo-trainer-prod/owned-worker-v7.log'
FROM, OFF, UNTIL = int(os.environ['FROM']), int(os.environ.get('OFF', '0')), int(os.environ['UNTIL'])
STALL = 60 * float(os.environ.get('STALL_MIN', '90')); LOG = AV / 'watchdog.log'
def log(**k):
    # host clock is UTC: CR = UTC-6 (caps-v1 fix; time.localtime() stamped UTC as CR)
    k['at_cr'] = time.strftime('%Y-%m-%d %H:%M:%S CR', time.gmtime(time.time() - 6 * 3600)); LOG.open('a').write(json.dumps(k) + '\n'); print(json.dumps(k), flush=True)
def jl(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None
def revert(reason):
    log(event='REVERT', reason=reason)
    env = dict(os.environ, MODE='revert', FORCE='1', REASON='12k-watchdog: ' + reason, FROM_WATCHDOG='1')
    rc = subprocess.run(['bash', str(AV / 'activate_12k.sh')], env=env).returncode
    log(event='reverted', rc=rc); sys.exit(0)
def oom():
    try:
        r = subprocess.run(['bash', str(TSH), "tail -c +%d %s | grep -c -E 'OutOfMemoryError|CUDA out of memory|CUDA error: out of memory' || true" % (OFF + 1, PODLOG)],
                           capture_output=True, text=True, timeout=60)
        return int((r.stdout.strip() or '0').splitlines()[-1]) if r.returncode == 0 else 0
    except Exception: return 0
def check():
    act = jl(AV / 'ACTIVE.json')
    if not act or act.get('trainer') != 'v833m': log(event='not active - exiting'); sys.exit(0)
    n = oom()
    if n: revert('trainer log: %d CUDA out-of-memory line(s) since the switch' % n)
    good = 0
    for e in range(FROM, FROM + 2000):
        ed = E / str(e)
        if not ed.exists():
            if not (E / str(e + 1)).exists(): break
            continue
        if (ed / 'RECOVERY-REQUIRED.json').exists(): revert('e%d RECOVERY-REQUIRED' % e)
        j, r = ed / 'TRAINING-JOB.json', jl(ed / 'TRAINING-RESULT.json')
        if r is None:
            if j.exists() and time.time() - j.stat().st_mtime > STALL: revert('e%d training job open %.0f min' % (e, (time.time() - j.stat().st_mtime) / 60))
            continue
        r = r.get('payload', r); t = (r.get('report') or {}).get('training') or {}
        um = (((t.get('optimizer') or {}).get('sgd') or {}).get('fp32_master') or {}).get('update_norm_master')
        if t.get('status') != 'updated': revert('e%d status %s' % (e, t.get('status')))
        if not (isinstance(um, float) and math.isfinite(um)): revert('e%d update_norm_master %r' % (e, um))
        good += 1
    return good
if __name__ == '__main__':
    if '--once' in sys.argv: print(check()); sys.exit(0)
    log(event='armed', FROM=FROM, OFF=OFF, UNTIL=UNTIL, stall_min=STALL / 60); last = None
    while time.time() < UNTIL:
        n = check()
        if n != last: log(event='ok', good_epochs=n); last = n
        time.sleep(60)
    log(event='window over - exiting')
