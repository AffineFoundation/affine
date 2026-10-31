#!/usr/bin/env python3
"""tp-watchdog (RA/20261031-ff-train-params): auto-revert the per-turn lr/kl change (MODE=revert activate_tp.sh FORCE=1:
FF-TRAIN off at once, trainer v833j -> v833i, ff files back to the pre-change pins, sidecars back). Every 30 s for epochs >= FROM
until UNTIL (unix), while activation/ACTIVE.json names v833j:
  (a) trainer failure: RECOVERY-REQUIRED in an epoch, or a training job running > JOB_LIMIT s without a result
      (not counted while the pod is short of disk);
  (b) a non-finite loss / pre-clip gradient norm / update norm, or a result not on the normgrad step;
  (c) a default-param turn (no 'train', or lr 5e-7 and kl 0.02) whose update differs from v8.3.3i: train_params.step.path must be
      'v833i-exact', lr_multiplier 1, kl_applied 0.02, step raw norm == the trainer's sgd raw_update_norm, update_norm_master
      within 1% of raw and within [0.5, 2] x 0.0224 (the rs-watchdog band);
  (d) a param turn whose stats are inconsistent with the rule: lr / kl in the report != the dispatch, kl_applied != kl,
      m <= 1: raw != m x raw_default (1e-6 rel), m > 1: raw outside [raw_default, m x raw_default], master off raw by > 1%,
      any update norm above the global cap 1.0;
  (e) the free-file selection failed (ff-selection-error) in 2 epochs, or the lifecycle not active for 3 checks in a row;
  (f) RA/20261031-renorm-step-live/tests/EIGHT-VS-ONE.json verdict "mismatch" for an epoch >= FROM (taken over from rs-watchdog).
Logs to activation/watchdog.log + PROGRESS.md. Read-only except the revert call."""
import json, math, os, subprocess, time
from pathlib import Path
RA = Path('/home/const/subnet120-rewrite/state/root-audits'); D = RA / '20261031-ff-train-params'; AV = D / 'activation'
E = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs')
FROM = int(os.environ['FROM']); UNTIL = float(os.environ.get('UNTIL', time.time() + 24 * 3600))
JOB_LIMIT = int(os.environ.get('JOB_LIMIT', 1500)); BASE = float(os.environ.get('BASE', 0.0224))
DEF = dict(lr=5e-07, kl=0.02); CAP = 1.0
POD = '/workspace/affine-mimo-trainer-prod'
SSH = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10', '-o',
       'UserKnownHostsFile=/home/const/.lium/known_hosts/ef2b6085-adae-43e2-8a13-39182a3b3f7a', '-p', '40022', 'root@8.28.35.244']
cr = lambda: time.strftime('%H:%M:%S CR', time.gmtime(time.time() - 6 * 3600))


def log(m, prog=False):
    open(AV / 'watchdog.log', 'a').write(f'{cr()} {m}\n')
    if prog: open(D / 'PROGRESS.md', 'a').write(f'- {cr()[:5]} CR: tp-watchdog: {m}\n')


def rj(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None


def mt(p):
    try: return Path(p).stat().st_mtime
    except OSError: return None


def fin(*xs): return all(type(x) in (int, float) and math.isfinite(x) for x in xs)


def live():
    a = rj(AV / 'ACTIVE.json') or {}
    return a.get('trainer') == 'v833j'


def disk_short():
    try:
        out = subprocess.run(SSH + [f'df -B1G --output=avail {POD} | tail -1'], capture_output=True, text=True, timeout=60).stdout
        return int(out.strip()) < 192
    except Exception:
        return False


def revert(why):
    log(f'REVERT ({why})', True)
    r = subprocess.run(['bash', str(AV / 'activate_tp.sh')], env=dict(os.environ, MODE='revert', FORCE='1', REASON=why, FROM_WATCHDOG='1'))
    log(f'revert rc={r.returncode}', True)
    raise SystemExit(0)


def check_result(e, d):
    """-> None if fine, else the revert reason."""
    env = rj(d / 'TRAIN-DISPATCH-FF.json') or {}; pay = env.get('payload') or env
    r = rj(d / 'TRAINING-RESULT.json') or {}; r = r.get('payload', r)
    t = ((r.get('report') or {}).get('training')) or {}
    if t.get('status') != 'updated':
        log(f'e{e} result status {t.get("status")} (no update): not checked', True); return None
    s = ((t.get('optimizer') or {}).get('sgd')) or {}; fm = s.get('fp32_master') or {}
    um, raw = fm.get('update_norm_master'), s.get('raw_update_norm')
    tp = t.get('train_params'); st = (tp or {}).get('step') or {}
    want = pay.get('train')
    log(f"e{e} turn: dispatch train {None if want is None else {k: want.get(k) for k in ('lr', 'kl', 'given')}} "
        f"report {None if tp is None else {k: tp.get(k) for k in ('lr', 'kl', 'lr_multiplier', 'kl_applied', 'defaults_used')}} "
        f"step {st.get('path')} raw {st.get('raw_update_norm')} raw_default {st.get('raw_update_norm_default')} at_ceiling {st.get('tensors_at_ceiling')} "
        f"sgd raw {raw} master {um} dir {s.get('sgd_direction')} loss {t.get('objective_loss')} preclip {t.get('preclip_gradient_norm')}", True)
    if not fin(t.get('objective_loss'), t.get('preclip_gradient_norm'), raw, um):
        return f'e{e} non-finite loss / gradient / update norm'
    if s.get('sgd_direction') != 'normgrad':
        return f'e{e} result not on the normgrad step ({s.get("sgd_direction")})'
    if want is not None and tp is None:
        return f'e{e} dispatch carried train params but the trainer reported none'
    default = want is None or (want.get('lr') == DEF['lr'] and want.get('kl') == DEF['kl'])
    if raw <= 0 or (default and abs(um / raw - 1) > 0.01):
        return f'e{e} update_norm_master {um} off raw {raw} by > 1%'
    if default:
        if tp is not None and not (st.get('path') == 'v833i-exact' and tp.get('lr_multiplier') == 1.0 and tp.get('kl_applied') == DEF['kl']
                                   and tp.get('defaults_used') is True and st.get('raw_update_norm') == raw):
            return f'e{e} default-param turn not on the exact v8.3.3i path ({st.get("path")}, m {tp.get("lr_multiplier")}, kl {tp.get("kl_applied")})'
        if not (0.5 * BASE <= um <= 2.0 * BASE):
            return f'e{e} default-param update norm {um:.6g} outside [0.5, 2] x {BASE}'
        return None
    if tp.get('lr') != want.get('lr') or tp.get('kl') != want.get('kl') or tp.get('kl_applied') != want.get('kl'):
        return f'e{e} report lr/kl {tp.get("lr")}/{tp.get("kl")} (applied {tp.get("kl_applied")}) != dispatch {want.get("lr")}/{want.get("kl")}'
    m = tp.get('lr_multiplier'); rj_, rd = st.get('raw_update_norm'), st.get('raw_update_norm_default')
    if not fin(m, rj_, rd) or rj_ <= 0 or rd <= 0 or abs(m - want['lr'] / DEF['lr']) > 1e-12:
        return f'e{e} lr multiplier {m} inconsistent'
    if m == 1.0:
        if st.get('path') != 'v833i-exact' or rj_ != raw: return f'e{e} lr default with kl only: step not v8.3.3i-exact'
    else:
        if st.get('path') != 'scaled': return f'e{e} lr {want["lr"]} turn not on the scaled path'
        if m < 1.0 and abs(rj_ / (m * rd) - 1) > 1e-6: return f'e{e} m {m}: raw {rj_} != m x raw_default {m * rd}'
        if m > 1.0 and not (rd * (1 - 1e-9) <= rj_ <= m * rd * (1 + 1e-9)): return f'e{e} m {m}: raw {rj_} outside [{rd}, {m * rd}]'
    if abs(um / rj_ - 1) > 0.01: return f'e{e} update_norm_master {um} off the turn raw {rj_} by > 1%'
    if um > CAP * 1.01: return f'e{e} update norm {um} above the global cap {CAP}'
    if not (0.5 * BASE * min(m, 1.0) <= um <= 2.0 * BASE * max(m, 1.0)):
        return f'e{e} update norm {um:.6g} outside the band for m {m}'
    return None


log(f'armed FROM=e{FROM} until {time.strftime("%H:%M CR", time.gmtime(UNTIL - 6 * 3600))}', True)
down, sel_err, seen = 0, set(), set()
while time.time() < UNTIL:
    time.sleep(30)
    if not live():
        log('ACTIVE.json no longer names v833j (reverted or not live): exiting', True); break
    down = 0 if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'affine-mimo-lifecycle.service']).returncode == 0 else down + 1
    if down >= 3: revert('lifecycle not active for 90 s')
    for d in sorted((p for p in E.iterdir() if p.name.isdigit() and int(p.name) >= FROM), key=lambda p: int(p.name)):
        e = int(d.name)
        sel = rj(d / 'FF-SELECT.json') or {}
        if sel.get('reason') == 'ff-selection-error' and e not in sel_err:
            sel_err.add(e); log(f'e{e} ff-selection-error: {sel.get("error")}', True)
            if len(sel_err) >= 2: revert('ff selection failed in 2 epochs')
        if not (d / 'TRAINING-JOB.json').exists(): continue
        job, res = mt(d / 'TRAINING-JOB.json'), mt(d / 'TRAINING-RESULT.json')
        if (d / 'RECOVERY-REQUIRED.json').exists() and ('rec', e) not in seen:
            seen.add(('rec', e))
            if disk_short(): log(f'e{e} RECOVERY-REQUIRED but pod disk short: not counted', True)
            else: revert(f'e{e} trainer job failed (RECOVERY-REQUIRED)')
        if job and not res and time.time() - job > JOB_LIMIT and ('job', e) not in seen:
            seen.add(('job', e))
            if disk_short(): log(f'e{e} job > {JOB_LIMIT}s but pod disk short: not counted', True)
            else: revert(f'e{e} trainer job running > {JOB_LIMIT} s')
        if res and ('res', e) not in seen:
            seen.add(('res', e))
            why = check_result(e, d)
            if why: revert(why)
    v = rj(RA / '20261031-renorm-step-live/tests/EIGHT-VS-ONE.json')
    if v and v.get('verdict') == 'mismatch' and type(v.get('epoch')) is int and v['epoch'] >= FROM and 'evo' not in seen:
        seen.add('evo'); revert(f"8-GPU vs 1-GPU replay mismatch (e{v.get('epoch')}: {v.get('reason')})")
else:
    log('watch window over; exiting (v833j kept)', True)
