#!/usr/bin/env python3
"""RA/20261031-renorm-step-live: add the v8.3.3i auto-revert to watchdog_ff.py (ff checks unchanged)."""
import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text()
def rep(old, new):
    global s
    if s.count(old) != 1: raise SystemExit('anchor: ' + old[:60])
    s = s.replace(old, new)
rep('''Logs to activation/watchdog.log + PROGRESS.md."""''', '''Logs to activation/watchdog.log + PROGRESS.md.
v8.3.3i (RA/20261031-renorm-step-live, Jacob 2026-10-31 07:00 CR): while RA/20261031-renorm-step-live/activation/ACTIVE.json names trainer
v833i (until RS_UNTIL), every epoch >= its from_epoch is also checked and the TRAINER ONLY is reverted v833i -> v833h
(MODE=revert activate_rs.sh FORCE=1; lifecycle/free-file untouched) on: training job failure (RECOVERY-REQUIRED) or a job running > JOB_LIMIT s,
a non-finite loss / gradient norm / update norm, an update norm outside [RS_LO, RS_HI] x the v8.3.3h baseline (or update_norm_master off
raw_update_norm by > 1%), a result not on the normgrad step, or tests/EIGHT-VS-ONE.json verdict "mismatch" (8-GPU turn vs 1-GPU replay).
Any ff revert first reverts the trainer to v833h. The out-of-support token fraction is logged, never a revert or rejection reason."""''')
rep('''cr = lambda: time.strftime('%H:%M:%S CR', time.gmtime(time.time() - 6 * 3600))
''', '''cr = lambda: time.strftime('%H:%M:%S CR', time.gmtime(time.time() - 6 * 3600))
RS = RA / '20261031-renorm-step-live'; RSA = RS / 'activation'
RS_UNTIL = float(os.environ.get('RS_UNTIL', 0)); RS_BASE = float(os.environ.get('RS_BASE', 0.0224))   # v833h e330-e352: 0.02234-0.02257
RS_LO, RS_HI = 0.5, 2.0
rs_seen = set()


def rs_log(m):
    open(RSA / 'watchdog.log', 'a').write(f'{cr()} {m}\\n')
    open(RS / 'PROGRESS.md', 'a').write(f'- {cr()[:5]} CR: rs-watchdog: {m}\\n')


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
               f"job->result {(res - job) if job else None:.0f}s")
        rs_log(msg)
        if t.get('status') == 'updated':
            if not rs_finite(t.get('objective_loss'), t.get('preclip_gradient_norm'), raw, um):
                rs_revert(f'e{e} non-finite loss/gradient/update norm'); return True
            if s.get('sgd_direction') != 'normgrad':
                rs_revert(f'e{e} result not on the normgrad step ({s.get("sgd_direction")})'); return True
            if not (RS_LO * RS_BASE <= um <= RS_HI * RS_BASE) or abs(um / raw - 1) > 0.01:
                rs_revert(f'e{e} update norm {um:.6g} vs v833h {RS_BASE} (raw {raw:.6g})'); return True
    v = rj(RS / 'tests' / 'EIGHT-VS-ONE.json')
    if v and v.get('verdict') == 'mismatch' and 'evo' not in rs_seen:
        rs_seen.add('evo'); rs_revert(f"8-GPU vs 1-GPU replay mismatch (e{v.get('epoch')}: {v.get('reason')})"); return True
    return False
''')
rep('''def revert(why, force=False):
    log(f'REVERT ({why}) force={force}', True)
''', '''def revert(why, force=False):
    if rs_active() is not None:      # v8.3.3i: the free-file revert expects trainer v833h
        rs_revert('before ff revert: ' + why)
    if time.time() >= UNTIL:
        log(f'(ff window over) not reverting free-file: {why}', True); return
    log(f'REVERT ({why}) force={force}', True)
''')
rep('''log(f'armed FROM=e{FROM} until {time.strftime("%H:%M CR", time.gmtime(UNTIL - 6 * 3600))}', True)''',
    '''log(f'armed FROM=e{FROM} until {time.strftime("%H:%M CR", time.gmtime(UNTIL - 6 * 3600))}; v833i checks until {time.strftime("%H:%M CR", time.gmtime(RS_UNTIL - 6 * 3600))}', True)''')
rep('''while time.time() < UNTIL:
    time.sleep(30)
''', '''while time.time() < max(UNTIL, RS_UNTIL):
    time.sleep(30)
    if time.time() >= UNTIL:     # ff window over: only the v833i checks keep running
        for d in sorted((p for p in E.iterdir() if p.name.isdigit() and int(p.name) >= FROM), key=lambda p: int(p.name))[-6:]:
            rs_check(d, int(d.name))
        continue
''')
rep('''        e = int(d.name)
        s = rj(d / 'FF-SELECT.json') or {}''', '''        e = int(d.name)
        if rs_check(d, e):
            continue
        s = rj(d / 'FF-SELECT.json') or {}''')
p.write_text(s); print('patched', p)
