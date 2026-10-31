#!/usr/bin/env python3
"""ff-watchdog (RA/20261030-free-file): auto-revert the free-file cutover (MODE=revert activate_ff.sh) when the change
breaks the lifecycle. Checked every 30 s from epoch FROM until WATCH_UNTIL (unix):
  (a) affine-mimo-lifecycle not active for 3 checks in a row,
  (b) an FF turn's training job (TRAIN-DISPATCH-FF.json) runs > JOB_LIMIT s without TRAINING-RESULT, or RECOVERY-REQUIRED appears
      in an FF epoch (not counted when the pod is short of disk),
  (c) the FF selection failed (FF-SELECT reason ff-selection-error) in 2 epochs,
  (d) building the FF job took > 300 s after the seal, or an FF epoch took > 30 min from its seal to FINALIZED,
  (e) no new FINALIZED epoch for STALL s while the lifecycle is active.
Logs to activation/watchdog.log + PROGRESS.md."""
import json, os, subprocess, time
from pathlib import Path
RA = Path('/home/const/subnet120-rewrite/state/root-audits'); D = RA / '20261030-free-file'; AV = D / 'activation'
E = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs')
FROM = int(os.environ['FROM']); UNTIL = float(os.environ.get('WATCH_UNTIL', time.time() + 12 * 3600))
JOB_LIMIT = int(os.environ.get('JOB_LIMIT', 1500)); STALL = int(os.environ.get('STALL', 2700)); EPOCH_LIMIT = 1800
POD = '/workspace/affine-mimo-trainer-prod'
SSH = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10', '-o',
       'UserKnownHostsFile=/home/const/.lium/known_hosts/ef2b6085-adae-43e2-8a13-39182a3b3f7a', '-p', '40022', 'root@8.28.35.244']
cr = lambda: time.strftime('%H:%M:%S CR', time.gmtime(time.time() - 6 * 3600))


def log(m, prog=False):
    open(AV / 'watchdog.log', 'a').write(f'{cr()} {m}\n')
    if prog:
        open(D / 'PROGRESS.md', 'a').write(f'- {cr()[:5]} CR: ff-watchdog: {m}\n')


def rj(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def mt(p):
    try:
        return Path(p).stat().st_mtime
    except OSError:
        return None


def disk_short():
    try:
        out = subprocess.run(SSH + [f'df -B1G --output=avail {POD} | tail -1'], capture_output=True, text=True, timeout=60).stdout
        return int(out.strip()) < 192
    except Exception:
        return False


def revert(why, force=False):
    log(f'REVERT ({why}) force={force}', True)
    env = dict(os.environ, MODE='revert', FROM_WATCHDOG='1', REASON=why, FORCE='1' if force else '0')
    r = subprocess.run(['bash', str(AV / 'activate_ff.sh')], env=env)
    log(f'revert rc={r.returncode}', True)
    raise SystemExit(0)


def active():
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'affine-mimo-lifecycle.service']).returncode == 0


log(f'armed FROM=e{FROM} until {time.strftime("%H:%M CR", time.gmtime(UNTIL - 6 * 3600))}', True)
down, sel_err, seen = 0, set(), set()
last_fin = max([mt(p) or 0 for p in E.glob('*/FINALIZED.json')] + [time.time()])
while time.time() < UNTIL:
    time.sleep(30)
    down = 0 if active() else down + 1
    if down >= 3:
        revert('lifecycle not active for 90 s')
    fins = [mt(p) or 0 for p in E.glob('*/FINALIZED.json')]
    last_fin = max(fins + [last_fin])
    if active() and time.time() - last_fin > STALL:
        revert('no FINALIZED epoch for %d s' % STALL)
    for d in sorted((p for p in E.iterdir() if p.name.isdigit() and int(p.name) >= FROM), key=lambda p: int(p.name)):
        e = int(d.name)
        s = rj(d / 'FF-SELECT.json') or {}
        if s.get('reason') == 'ff-selection-error' and e not in sel_err:
            sel_err.add(e); log(f'e{e} ff-selection-error: {s.get("error")}', True)
            if len(sel_err) >= 2:
                revert('ff selection failed in 2 epochs')
        if not (d / 'TRAIN-DISPATCH-FF.json').exists():
            continue
        job, res, sel, fin = mt(d / 'TRAINING-JOB.json'), mt(d / 'TRAINING-RESULT.json'), mt(d / 'FF-SELECT.json'), mt(d / 'FINALIZED.json')
        if (d / 'RECOVERY-REQUIRED.json').exists() and not disk_short():
            revert(f'e{e} RECOVERY-REQUIRED in an FF turn', force=True)
        if job and not res and time.time() - job > JOB_LIMIT:
            if disk_short():
                log(f'e{e} FF job > {JOB_LIMIT}s but pod disk short: not counted', True)
            else:
                revert(f'e{e} FF training job running > {JOB_LIMIT} s', force=True)
        if sel and job and job - sel > 300 and ('build', e) not in seen:
            seen.add(('build', e)); revert(f'e{e} FF job build took {job - sel:.0f} s')
        if sel and (res or time.time()) - sel > EPOCH_LIMIT and ('epoch', e) not in seen:
            seen.add(('epoch', e)); revert(f'e{e} FF epoch > {EPOCH_LIMIT} s from seal to training result')
        if res and ('ok', e) not in seen:
            seen.add(('ok', e)); log(f'e{e} FF turn trained: select->job {job - sel:.0f}s, job->result {res - job:.0f}s', True)
log('watch window over; exiting', True)
