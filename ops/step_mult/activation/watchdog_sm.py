#!/usr/bin/env python3
"""RA/20261102-step-mult sm-watchdog: auto-revert train.step_mult + the v8.3.3l knobs (MODE=revert activate_sm.sh) when
  ("special" turn = step_mult > 1 or any knob not at its default)
  (a) a special turn fails: RECOVERY-REQUIRED, status != updated, or a non-finite update norm; or the first 3 v833l turns
      need recovery at all;
  (b) holdout regression: a GREEN king trained by a special turn scores below its parent on the private holdout (perf-eval, 4,976 tasks)
      by more than 3 sigma, sigma = sqrt(se_king^2 + se_parent^2) (about 1.3pp); or the newest evaluated king is 3 sigma below the head
      the switch started from while at least one special GREEN came in between.
The z > 1 promotion gate itself is unchanged; this only takes the new hparam away again. env FROM (first v833l epoch), UNTIL (unix)."""
import json, math, os, subprocess, sys, time
from pathlib import Path
RA = Path('/home/const/subnet120-rewrite/state/root-audits'); D = RA / '20261102-step-mult'; AV = D / 'activation'
E = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs'); G = RA / '20261026-promotion-gate/state/epochs'
PR = RA / '20261031-performance-eval/state/results'
FROM, UNTIL = int(os.environ['FROM']), int(os.environ['UNTIL']); NSIG = float(os.environ.get('NSIG', '3'))
LOG = AV / 'watchdog.log'
def log(**k):
    k['at_cr'] = time.strftime('%Y-%m-%d %H:%M:%S CR', time.localtime()); LOG.open('a').write(json.dumps(k) + '\n'); print(json.dumps(k), flush=True)
def jl(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None
def tp_of(e):
    r = jl(E / str(e) / 'TRAINING-RESULT.json')
    if r is None: return None, None
    r = r.get('payload', r); t = (r.get('report') or {}).get('training') or {}
    return t, (t.get('train_params') or {})
KNOB_DEFAULTS = {'lr_scale_attn': 1.0, 'lr_scale_mlp': 1.0, 'lr_scale_embed': 1.0, 'lr_scale_lm_head': 1.0, 'lr_scale_early': 1.0,
                 'lr_scale_late': 1.0, 'adam_eps': 1e-08, 'weight_decay': 0.01, 'clamp_mode': 'per_tensor', 'global_norm_cap': 1.0,
                 'loss_agg': 'seq_mean', 'adv_center': 'mean', 'adv_clip': None}
def nondefault(kn):
    return sorted(k for k, v in (kn or {}).items() if k in KNOB_DEFAULTS and v != KNOB_DEFAULTS[k])
def job_k(e):
    """-> (step_mult, non-default knob names) of the dispatched job (any 'train' object inside TRAINING-JOB.json)."""
    d = jl(E / str(e) / 'TRAINING-JOB.json')
    if d is None: return 1, []
    st = [d]; k, kn = 1, []
    while st:
        o = st.pop()
        if isinstance(o, dict):
            t = o.get('train')
            if isinstance(t, dict) and ('step_mult' in t or 'knobs' in t):
                k = max(k, int(t.get('step_mult', 1) or 1)); kn = sorted(set(kn) | set(nondefault(t.get('knobs'))))
            st.extend(o.values())
        elif isinstance(o, list): st.extend(o)
    return k, kn
def holdout(king):
    d = jl(PR / (king + '.json'))
    try: h = d['sets']['holdout']; return float(h['pass_rate']), float(h['se']), (d.get('harness') or {}).get('max_output_tokens')
    except Exception: return None
def comparable(a, b):            # only compare perf evals run with the same harness (8192 vs 12288 output cap moved pass rates)
    return bool(a and b and a[2] == b[2])
def revert(reason):
    log(event='REVERT', reason=reason)
    env = dict(os.environ, MODE='revert', REASON='sm-watchdog: ' + reason, FROM_WATCHDOG='1')
    rc = subprocess.run(['bash', str(AV / 'activate_sm.sh')], env=env).returncode
    log(event='reverted', rc=rc); sys.exit(0)
def check():
    act = jl(AV / 'ACTIVE.json')
    if not act or act.get('trainer') != 'v833l': log(event='not active - exiting'); sys.exit(0)
    start_head = None; big_green = 0; newest = None
    for e in range(FROM, FROM + 2000):
        ed = E / str(e)
        if not ed.exists() and e > FROM + 5 and not (E / str(e + 1)).exists(): break
        k, kn = job_k(e)
        if (ed / 'RECOVERY-REQUIRED.json').exists() and (k > 1 or kn or e < FROM + 3): revert('e%d RECOVERY-REQUIRED (job step_mult %d knobs %s)' % (e, k, kn))
        t, tp = tp_of(e)
        if t is None: continue
        k = int(tp.get('step_mult', 1) or 1); kn = nondefault(tp.get('knobs'))
        special = k > 1 or bool(kn)
        if special:
            sgd = ((t.get('optimizer') or {}).get('sgd') or {}); um = (sgd.get('fp32_master') or {}).get('update_norm_master')
            if t.get('status') != 'updated': revert('e%d step_mult %d knobs %s status %s' % (e, k, kn, t.get('status')))
            if not (isinstance(um, float) and math.isfinite(um)): revert('e%d step_mult %d knobs %s update_norm_master %r' % (e, k, kn, um))
        dec = jl(G / str(e) / 'DECISION.json')
        if not dec: continue
        if start_head is None: start_head = dec.get('head')
        if dec.get('decision') != 'green': continue
        king, parent = dec['candidate'], dec['head']; hk, hp = holdout(king), holdout(parent)
        if special: big_green += 1
        if hk: newest = (e, king, hk)
        if special and comparable(hk, hp):
            d, s = hk[0] - hp[0], math.hypot(hk[1], hp[1])
            if d < -NSIG * s: revert('e%d GREEN step_mult %d knobs %s holdout %.4f vs parent %.4f (%.2f sigma)' % (e, k, kn, hk[0], hp[0], d / s))
    if big_green and newest and start_head and holdout(start_head):
        e, king, hk = newest; h0 = holdout(start_head); d, s = hk[0] - h0[0], math.hypot(hk[1], h0[1])
        if comparable(hk, h0) and d < -NSIG * s: revert('newest king e%d holdout %.4f vs pre-switch head %.4f (%.2f sigma) after %d special GREEN' % (e, hk[0], h0[0], d / s, big_green))
    return big_green
if __name__ == '__main__':
    if '--once' in sys.argv: print(check()); sys.exit(0)
    log(event='armed', FROM=FROM, UNTIL=UNTIL, nsig=NSIG); last = None
    while time.time() < UNTIL:
        n = check()
        if n != last: log(event='ok', special_greens=n); last = n
        time.sleep(60)
    log(event='window over - exiting')
