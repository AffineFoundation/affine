#!/usr/bin/env python3
"""ts-watchdog (RA/20261030-trainer-speed): auto-revert trainer v8.3.3g -> v8.3.3f on
  (a) a training job running > LIMIT s (13 min; v833f cold baseline 657-710 s) -> FORCE revert (interrupt),
  (b) TRAINING-RESULT success != True, (c) owned worker not alive for 3 checks (60 s apart) while no RED restore is running,
  (d) gate decision reasons mentioning nonfinite/digest/fp32 (integrity, not quality).
Watches WATCH (3) v833g trainings from FROM, then exits. Logs to activation/watchdog.log + PROGRESS.md."""
import json, os, subprocess, time
from pathlib import Path
RA = Path('/home/const/subnet120-rewrite/state/root-audits'); D = RA / '20261030-trainer-speed'; AV = D / 'activation'
E = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs'); GE = RA / '20261026-promotion-gate/state/epochs'
FROM = int(os.environ['FROM']); WATCH = int(os.environ.get('WATCH', 3)); LIMIT = int(os.environ.get('LIMIT', 780))
POD = '/workspace/affine-mimo-trainer-prod'; G_S = (AV / 'G_S').read_text().strip()
SSH = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10', '-o',
       'UserKnownHostsFile=/home/const/.lium/known_hosts/ef2b6085-adae-43e2-8a13-39182a3b3f7a', '-p', '40022', 'root@8.28.35.244']
cr = lambda: time.strftime('%H:%M:%S CR', time.gmtime(time.time() - 6 * 3600))
def log(m, prog=False):
    open(AV / 'watchdog.log', 'a').write(f'{cr()} {m}\n')
    if prog: open(D / 'PROGRESS.md', 'a').write(f'- {cr()[:5]} CR: ts-watchdog: {m}\n')
def rj(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None
def revert(why, force):
    log(f'REVERT ({why}) force={force}', True)
    env = dict(os.environ, MODE='revert', FROM_WATCHDOG='1', REASON=why, FORCE='1' if force else '0')
    r = subprocess.run(['bash', str(AV / 'activate_ts.sh')], env=env); log(f'revert rc={r.returncode}', True); raise SystemExit(0)
def alive():
    try:
        out = subprocess.run(SSH + [f'{POD}/runtime-exact/venv/bin/python -I -B {POD}/owned-worker-v7.py --config {POD}/owned-worker-v833g.json --sha256 {G_S} --status'],
                             capture_output=True, text=True, timeout=60).stdout
        return json.loads(out).get('alive') is True
    except Exception: return None
def restoring():
    out = subprocess.run(['systemctl', '--user', 'list-units', '--no-legend', 'affine-gate-restore-*'], capture_output=True, text=True).stdout
    return 'activ' in out
log(f'armed FROM=e{FROM} WATCH={WATCH} LIMIT={LIMIT}s', True)
done, dead, last_alive, e, seen_err = 0, 0, 0, FROM, 0
while done < WATCH:
    time.sleep(10)
    job, res = E / str(e) / 'TRAINING-JOB.json', E / str(e) / 'TRAINING-RESULT.json'
    if not job.exists() and (E / str(e + 1) / 'TRAINING-JOB.json').exists(): e += 1; continue
    if job.exists() and not res.exists() and time.time() - job.stat().st_mtime > LIMIT:
        revert(f'e{e} training > {LIMIT}s', True)
    if res.exists():
        r = (rj(res) or {}).get('payload', {}).get('report', {})
        dt = res.stat().st_mtime - job.stat().st_mtime
        if r.get('success') is not True: revert(f'e{e} TRAINING-RESULT success={r.get("success")}', False)
        log(f'e{e} v833g training OK job->RESULT {dt:.0f}s (v833f: 657-710 s cold, 319-334 s warm)', True)
        dec = GE / str(e) / 'DECISION.json'
        while not dec.exists():
            time.sleep(10)
            if (E / str(e + 1) / 'TRAINING-JOB.json').exists() and not (GE / str(e) / 'DECISION.json').exists(): break
        d = rj(dec) or {}; txt = json.dumps(d.get('reasons', '')).lower()
        if any(k in txt for k in ('nonfinite', 'non-finite', 'digest', 'fp32')): revert(f'e{e} gate integrity reason', False)
        log(f'e{e} gate decision {d.get("decision")}', True); done += 1; e += 1; continue
    if time.time() - last_alive > 60:
        last_alive = time.time(); a = alive()
        dead = 0 if (a is not False or restoring()) else dead + 1
        if dead >= 3: revert('owned worker not alive 3x', True)
log(f'finished: {done} v833g trainings clean', True)
