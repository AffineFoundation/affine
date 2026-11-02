#!/usr/bin/env python3
"""RA/20261102-step-mult pod side driver: phases of (turn, m) -> build candidates (build_cand.py) -> eval (run_model.sh) -> delete weights.
Models are deduplicated: m=0 of a turn is its head, m=1 is its published candidate (= next head in the chain)."""
import json, os, shutil, subprocess, sys, time
R = '/data/smult'
TURNS = json.load(open(R + '/turns.json'))       # [{epoch, H, C}]
PHASES = json.load(open(R + '/phases.json'))     # [[m, ...], ...]
SEED = os.environ.get('SEED', 'sn120-smult-v1'); KH = os.environ.get('KH', '1')
def log(**k):
    k['at'] = time.strftime('%Y-%m-%d %H:%M:%S'); open(R + '/driver.log', 'a').write(json.dumps(k) + '\n')
def key(t, m):
    if m == 0: return 'ck-' + t['H'][:12]
    if m == 1: return 'ck-' + t['C'][:12]
    return 'e%d-m%g' % (t['epoch'], m)
for ph in PHASES:
    for t in TURNS:
        need = [m for m in ph if not os.path.exists('%s/evals/%s/result.json' % (R, key(t, m))) and not os.path.exists('%s/evals/%s/stalled' % (R, key(t, m)))]
        if not need: continue
        for ck in (t['H'], t['C']):        # wait for the sha-verified download of both masters
            nf = len(json.load(open('%s/manifests/%s.json' % (R, ck)))['files'])
            while len([f for f in os.listdir('%s/fp32/%s' % (R, ck)) if f.endswith('.ok')]) < nf if os.path.isdir('%s/fp32/%s' % (R, ck)) else True:
                time.sleep(30)
        base = '%s/ckpt/e%d' % (R, t['epoch'])
        t0 = time.time()
        p = subprocess.run(['/data/venv/bin/python', R + '/code/build_cand.py', t['H'], t['C'], base, R + '/bf16/' + TURNS[0]['H'],
                            R + '/manifests', ','.join('%g' % m for m in need)], capture_output=True, text=True)
        log(ev='build', epoch=t['epoch'], ms=need, rc=p.returncode, out=p.stdout[-1500:], err=p.stderr[-800:], s=round(time.time() - t0))
        if p.returncode != 0: continue
        for m in need:
            ck = '%s-m%g' % (base, m); k = key(t, m); t1 = time.time()
            q = subprocess.run(['bash', R + '/code/run_model.sh', ck, '%s/evals/%s' % (R, k), SEED, KH], capture_output=True, text=True)
            res = None
            try: res = json.load(open('%s/evals/%s/result.json' % (R, k)))['sets']['holdout']
            except Exception: pass
            log(ev='eval', epoch=t['epoch'], m=m, key=k, rc=q.returncode, out=q.stdout[-300:], s=round(time.time() - t1),
                pass_rate=(res or {}).get('pass_rate'), tasks=(res or {}).get('tasks'))
            shutil.rmtree(ck, ignore_errors=True)
log(ev='ALL-DONE')
