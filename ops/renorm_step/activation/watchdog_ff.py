#!/usr/bin/env python3
"""ff-watchdog (RA/20261030-free-file): auto-revert the free-file cutover (MODE=revert activate_ff.sh) when the change
breaks the lifecycle. Checked every 30 s from epoch FROM until WATCH_UNTIL (unix):
  (a) affine-mimo-lifecycle not active for 3 checks in a row,
  (b) an FF turn's training job (TRAIN-DISPATCH-FF.json) runs > JOB_LIMIT s without TRAINING-RESULT, or RECOVERY-REQUIRED appears
      in an FF epoch (not counted when the pod is short of disk),
  (c) the FF selection failed (FF-SELECT reason ff-selection-error) in 2 epochs,
  (d) building the FF job took > 300 s after the seal, or an FF epoch took > 30 min from its seal to FINALIZED,
  (e) no new FINALIZED epoch for STALL s while the lifecycle is active.
Logs to activation/watchdog.log + PROGRESS.md.
v8.3.3i (RA/20261031-renorm-step-live, Jacob 2026-10-31 07:00 CR): while RA/20261031-renorm-step-live/activation/ACTIVE.json names trainer
v833i (until RS_UNTIL), every epoch >= its from_epoch is also checked and the TRAINER ONLY is reverted v833i -> v833h
(MODE=revert activate_rs.sh FORCE=1; lifecycle/free-file untouched) on: training job failure (RECOVERY-REQUIRED) or a job running > JOB_LIMIT s,
a non-finite loss / gradient norm / update norm, an update norm outside [RS_LO, RS_HI] x the v8.3.3h baseline (or update_norm_master off
raw_update_norm by > 1%), a result not on the normgrad step, or tests/EIGHT-VS-ONE.json verdict "mismatch" (8-GPU turn vs 1-GPU replay).
Any ff revert first reverts the trainer to v833h. The out-of-support token fraction is logged, never a revert or rejection reason."""
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
RS = RA / '20261031-renorm-step-live'; RSA = RS / 'activation'
RS_UNTIL = float(os.environ.get('RS_UNTIL', 0)); RS_BASE = float(os.environ.get('RS_BASE', 0.0224))   # v833h e330-e352: 0.02234-0.02257
RS_LO, RS_HI = 0.5, 2.0
rs_seen = set()


def rs_log(m):
    open(RSA / 'watchdog.log', 'a').write(f'{cr()} {m}\n')
    open(RS / 'PROGRESS.md', 'a').write(f'- {cr()[:5]} CR: rs-watchdog: {m}\n')


def rs_active():
    try:
        a = json.loads((RSA / 'ACTIVE.json').read_text())
    except Exception:
        return None
    return a if a.get('trainer') == 'v833i' and time.time() < RS_UNTIL else None


def rs_revert(why):
    rs_log(f'REVERT trainer v833i -> v833h ({why})')
    r = subprocess.run(['bash', str(RSA / 'activate_rs.sh')], env=dict(os.environ, MODE='revert', FORCE='1', REASON=why))
    rs_log(f'revert rc={r.returncode}')


def rs_finite(*xs):
    import math
    return all(type(x) in (int, float) and math.isfinite(x) for x in xs)


def rs_check(d, e):
    """-> True if this epoch was handled by the v833i checks (the ff checks then skip it)."""
    a = rs_active()
    if a is None or e < int(a.get('from_epoch', 1 << 30)) or not (d / 'TRAINING-JOB.json').exists():
        return False
    job, res = mt(d / 'TRAINING-JOB.json'), mt(d / 'TRAINING-RESULT.json')
    if (d / 'RECOVERY-REQUIRED.json').exists():
        rs_revert(f'e{e} trainer job failed (RECOVERY-REQUIRED)'); return True
    if job and not res and time.time() - job > JOB_LIMIT:
        rs_revert(f'e{e} trainer job running > {JOB_LIMIT} s'); return True
    if res and ('res', e) not in rs_seen:
        rs_seen.add(('res', e))
        r = rj(d / 'TRAINING-RESULT.json') or {}; r = r.get('payload', r)
        t = ((r.get('report') or {}).get('training')) or {}; s = ((t.get('optimizer') or {}).get('sgd')) or {}
        fm = s.get('fp32_master') or {}; sr = t.get('support_renorm') or {}
        um, raw = fm.get('update_norm_master'), s.get('raw_update_norm')
        msg = (f"e{e} v833i turn: dir {s.get('sgd_direction')} raw {raw} master {um} rel_capped {s.get('tensors_rel_capped')} "
               f"preclip {t.get('preclip_gradient_norm')} loss {t.get('objective_loss')} rollouts {t.get('trained_rollouts')} "
               f"outside-support {sr.get('outside_tokens')}/{sr.get('loss_tokens')} = {sr.get('outside_fraction')} (max rollout {sr.get('max_rollout_outside_fraction')}) "
               f"job->result {(res - job) if job else -1:.0f}s")
        rs_log(msg)
        if t.get('status') == 'updated':
            if not rs_finite(t.get('objective_loss'), t.get('preclip_gradient_norm'), raw, um):
                rs_revert(f'e{e} non-finite loss/gradient/update norm'); return True
            if s.get('sgd_direction') != 'normgrad':
                rs_revert(f'e{e} result not on the normgrad step ({s.get("sgd_direction")})'); return True
            if not (RS_LO * RS_BASE <= um <= RS_HI * RS_BASE) or raw <= 0 or abs(um / raw - 1) > 0.01:
                rs_revert(f'e{e} update norm {um:.6g} vs v833h {RS_BASE} (raw {raw:.6g})'); return True
    v = rj(RS / 'tests' / 'EIGHT-VS-ONE.json')
    if v and v.get('verdict') == 'mismatch' and 'evo' not in rs_seen:
        rs_seen.add('evo'); rs_revert(f"8-GPU vs 1-GPU replay mismatch (e{v.get('epoch')}: {v.get('reason')})"); return True
    return False


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
    if rs_active() is not None:      # v8.3.3i: the free-file revert expects trainer v833h
        rs_revert('before ff revert: ' + why)
    if time.time() >= UNTIL:
        log(f'(ff window over) not reverting free-file: {why}', True); return
    log(f'REVERT ({why}) force={force}', True)
    env = dict(os.environ, MODE='revert', FROM_WATCHDOG='1', REASON=why, FORCE='1' if force else '0')
    r = subprocess.run(['bash', str(AV / 'activate_ff.sh')], env=env)
    log(f'revert rc={r.returncode}', True)
    raise SystemExit(0)


def active():
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'affine-mimo-lifecycle.service']).returncode == 0


log(f'armed FROM=e{FROM} until {time.strftime("%H:%M CR", time.gmtime(UNTIL - 6 * 3600))}; v833i checks until {time.strftime("%H:%M CR", time.gmtime(RS_UNTIL - 6 * 3600))}', True)
down, sel_err, seen = 0, set(), set()
last_fin = max([mt(p) or 0 for p in E.glob('*/FINALIZED.json')] + [time.time()])
while time.time() < max(UNTIL, RS_UNTIL):
    time.sleep(30)
    if time.time() >= UNTIL:     # ff window over: only the v833i checks keep running
        for d in sorted((p for p in E.iterdir() if p.name.isdigit() and int(p.name) >= FROM), key=lambda p: int(p.name))[-6:]:
            rs_check(d, int(d.name))
        continue
    down = 0 if active() else down + 1
    if down >= 3:
        revert('lifecycle not active for 90 s')
    fins = [mt(p) or 0 for p in E.glob('*/FINALIZED.json')]
    last_fin = max(fins + [last_fin])
    if active() and time.time() - last_fin > STALL:
        revert('no FINALIZED epoch for %d s' % STALL)
    for d in sorted((p for p in E.iterdir() if p.name.isdigit() and int(p.name) >= FROM), key=lambda p: int(p.name)):
        e = int(d.name)
        if rs_check(d, e):
            continue
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
