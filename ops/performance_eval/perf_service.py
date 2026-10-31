#!/usr/bin/env python3
"""SN120 performance eval service (host side; Jacob pt911u 2026-10-31). READ-ONLY w.r.t. the subnet: it reads the gate ledger
(LEDGER.public.jsonl) to learn the kings, fetches published king checkpoints from R2 (gate presign.py) to its OWN pod (calm-orbit-ef,
pp.sh), runs perf_eval.py there and publishes aggregates to affine.io/performance.json. It never writes gate/lifecycle/trainer/weights
state; nothing in the gate, promotions, weights or the trainer reads its output.
Scheduling: the newest king first (as soon as it is crowned); when idle, backfill older kings newest->oldest. A newer GREEN never
preempts a running newest-king eval; it preempts a backfill eval. Stale kings in between are only scored later as backfill.
Pod disk: only the king being evaluated + the next one are kept."""
import json, os, subprocess, sys, time, glob, hashlib, traceback, re, collections
HERE = os.path.dirname(os.path.abspath(__file__)); RA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from fetch_king import fetch, descriptor
ST = RA + '/state'; RES = ST + '/results'; os.makedirs(RES, exist_ok=True)
SITE = '/home/const/subnet120/affine/website'; OUTJ = SITE + '/performance.json'
LEDGER = '/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/LEDGER.public.jsonl'
FFL = '/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs'
PP = RA + '/pp.sh'; POD = 'calm-orbit-ef'; GPU = '2xH200'; COST_H = 9.52
SEED = 'sn120-perf-v1'; K = dict(full=1, holdout=1)
STALL_S = 1200; MAX_FAIL = 3
HOLD = json.load(open(RA + '/holdout.public.json'))
def now(): return time.time()
def cr(t): return time.strftime('%Y-%m-%d %H:%M:%S CR (UTC-6)', time.gmtime(t - 6 * 3600))
def log(**k):
    k['at_cr'] = cr(now()); print(json.dumps(k), flush=True)
def rj(p, d=None):
    try: return json.load(open(p))
    except Exception: return d
def wj(p, o, mode=0o644):
    t = p + '.tmp'; json.dump(o, open(t, 'w'), indent=1); os.chmod(t, mode); os.replace(t, p)
def pod(cmd, timeout=120, inp=None):
    p = subprocess.run([PP, cmd], capture_output=True, text=True, timeout=timeout, input=inp)
    return p.returncode, p.stdout, p.stderr
def kings():
    out = []
    for l in open(LEDGER):
        try: r = json.loads(l)
        except Exception: continue
        if r.get('decision') == 'green' and r.get('applied') == 'green':
            out.append(dict(king=r['candidate'], epoch=r['epoch'], parent=r['head'], crowned_at=(r.get('timestamps') or {}).get('decided_at'),
                            crowned_at_cr=r.get('decided_at_cr'), gate_z=r.get('z')))
    return out
S = rj(ST + '/service.json', {}) or {}
S.setdefault('fail', {}); S.setdefault('unavailable', [])
def save(): wj(ST + '/service.json', S)
def scored(k): return os.path.exists(f'{RES}/{k}.json')
def wdir(k): return f'/data/perf/runs/{k}'
def pick(ks):
    """(king, priority) newest first; then backfill newest -> oldest."""
    if not ks: return None, None
    newest = ks[-1]['king']
    if not scored(newest) and S['fail'].get(newest, 0) < MAX_FAIL and newest not in S['unavailable']: return newest, 'newest'
    for k in reversed(ks[:-1]):
        kk = k['king']
        if not scored(kk) and S['fail'].get(kk, 0) < MAX_FAIL and kk not in S['unavailable']: return kk, 'backfill'
    return None, None
def ensure_ckpt(k):
    rc, o, _ = pod(f'test -f /data/perf/ckpt/{k}/.verified && echo yes')
    if 'yes' in o: return True
    if descriptor(k) is None: return None
    r = fetch(k); log(event='fetch', king=k[:12], status=r.get('status'), out=r.get('out', '')[-160:])
    rc, o, _ = pod(f'test -f /data/perf/ckpt/{k}/.verified && echo yes')
    return 'yes' in o
def prune(keep):
    rc, o, _ = pod('ls /data/perf/ckpt')
    for n in o.split():
        base = n.split('.')[0]
        if re.fullmatch(r'[0-9a-f]{64}', base) and base not in keep:
            pod(f'rm -rf /data/perf/ckpt/{base} /data/perf/ckpt/{base}.partial /data/perf/ckpt/{base}.descriptor.json /data/perf/ckpt/{base}.fetch.log')
            log(event='pruned', king=base[:12])
def launch(k):
    pod(f'rm -rf {wdir(k)}; mkdir -p {wdir(k)}')
    cmd = (f"nohup setsid bash /data/perf/run_king.sh {k} {SEED} {K['full']} {K['holdout']} > /data/perf/run-{k[:8]}.log 2>&1 < /dev/null & "
           f"sleep 2; cat {wdir(k)}/run.pid")
    try: rc, o, e = pod(cmd, timeout=40)
    except subprocess.TimeoutExpired: o = ''
    S['running'] = dict(king=k, started=now(), run_pid=o.strip()); save(); log(event='launched', king=k[:12], run_pid=o.strip())
def status(k):
    rc, o, _ = pod(f"cd {wdir(k)} 2>/dev/null && echo rc=$(cat eval.rc 2>/dev/null) && echo prog=$(cat result.json.progress 2>/dev/null) && "
                   f"echo alive=$(kill -0 $(cat run.pid) 2>/dev/null && echo 1 || echo 0) && echo res=$(test -f result.json && echo 1)", timeout=60)
    d = {}
    for line in o.splitlines():
        if '=' in line: a, b = line.split('=', 1); d[a] = b
    return d
def abort(k):
    # exact PIDs only: the eval worker, then the runner (its trap stops the two vLLM servers it started, by PID)
    pod(f"cd {wdir(k)} && for f in eval.pid run.pid; do p=$(cat $f 2>/dev/null); [ -n \"$p\" ] && kill $p 2>/dev/null; done; sleep 15; "
        f"for p in $(cat serve.pids 2>/dev/null); do kill -9 $p 2>/dev/null; done; true", timeout=90)
    log(event='aborted', king=k[:12])
def collect(k, meta):
    rc, o, _ = pod(f'cat {wdir(k)}/result.json', timeout=60)
    r = json.loads(o); r['king_meta'] = meta; r['machine'] = dict(pod=POD, gpu=GPU, cost_per_hour_usd=COST_H)
    r['evaluated_at'] = now(); r['evaluated_at_cr'] = cr(now())
    rc, o2, _ = pod(f'cat {wdir(k)}/serving_at', timeout=30)
    try: r['serve_s'] = int(o2.strip()) - int(S['running']['started'])
    except Exception: pass
    wj(f'{RES}/{k}.json', r); log(event='scored', king=k[:12], full=r['sets']['full']['pass_rate'], holdout=r['sets']['holdout']['pass_rate'], wall=r['wall_seconds'])
# ---- holdout guard: are any holdout problems used as miner training prompts (FF files)? (read-only)
_HG = {}
def holdout_guard():
    if now() - _HG.get('at', 0) < 1800 and 'res' in _HG: return _HG['res']
    try:
        OV = '/home/const/subnet120-rewrite/state/root-audits/20261031-pool-overlap'
        if 'idx' not in _HG:
            if OV not in sys.path: sys.path.insert(0, OV)
            ns = {}; src = open(OV + '/overlap.py').read().split('NEAR = float')[0].replace('os.path.abspath(__file__)', repr(OV + '/overlap.py'))
            cwd = os.getcwd(); os.chdir(OV); exec(src, ns); os.chdir(cwd)
            hp = rj(RA + '/holdout.private.json')['tasks']
            H_norm = {ns['norm'](t['problem']) for t in hp}; H_sh = [ns['shingles'](t['problem']) for t in hp]
            inv = collections.defaultdict(list)
            for i, sh in enumerate(H_sh):
                for g in sh: inv[g].append(i)
            _HG['idx'] = (ns, H_norm, H_sh, inv)
        ns, H_norm, H_sh, inv = _HG['idx']
        eps = 0; prompts = 0; hits = []; similar = []
        for f in sorted(glob.glob(FFL + '/*/ff/*.json')):
            e = f.split('/')[-3]; eps += 1
            d = rj(f) or {}
            for g in d.get('groups', []):
                p, _ = ns['strip_sys'](g.get('prompt') or ''); prompts += 1
                if ns['norm'](p) in H_norm: hits.append(dict(epoch=e, kind='exact', h=None)); continue
                sh = ns['shingles'](p); cnt = collections.Counter()
                for x in sh:
                    for i in inv.get(x, ()): cnt[i] += 1
                bj, bi = 0.0, None
                for i, c in cnt.items():
                    if c >= 0.8 * len(sh):
                        j = ns['jacc'](sh, H_sh[i])
                        if j > bj: bj, bi = j, i
                if bj >= 0.9: hits.append(dict(epoch=e, kind='near', h=bi))
                elif bj >= 0.8: similar.append(dict(epoch=e, h=bi))
        _HG['res'] = dict(checked_at_cr=cr(now()), ff_files=eps, prompts=prompts, matches=len(hits),
                          match_epochs=sorted({h['epoch'] for h in hits}),
                          similar_prompts=len(similar), similar_holdout_tasks=len({x['h'] for x in similar}),
                          rule='miner training prompts (free files) vs holdout: match = exact normalised or token-3-gram Jaccard >= 0.9 '
                               '(the holdout dedupe rule); similar = Jaccard 0.8-0.9 (same template, different numbers/answer; kept in the holdout per the dedupe rule)')
        _HG['at'] = now()
    except Exception as x:
        _HG['res'] = dict(error=type(x).__name__); _HG['at'] = now()
    return _HG['res']
def publish(ks, cur=None):
    rows = []
    for k in ks:
        r = rj(f"{RES}/{k['king']}.json")
        row = dict(king=k['king'], epoch=k['epoch'], parent=k['parent'], crowned_at=k['crowned_at'], crowned_at_cr=k['crowned_at_cr'])
        if r:
            row.update(status='scored', evaluated_at_cr=r.get('evaluated_at_cr'), eval_wall_s=round(r['wall_seconds'], 1), seed=r['seed'], k=r['k'])
            for s in ('full', 'holdout'):
                x = r['sets'][s]
                row[s] = dict(pass_rate=x['pass_rate'], se=x['se'], tasks=x['tasks'], samples=x['samples'], graded_samples=x['graded_samples'],
                              correct=x['correct'], cap_rate=x['cap_rate'], indeterminate=x['indeterminate'], mean_out_tokens=round(x['mean_out_tokens'], 1))
        else:
            row['status'] = ('evaluating' if cur and cur == k['king'] else 'unavailable' if k['king'] in S['unavailable']
                             else 'failed' if S['fail'].get(k['king'], 0) >= MAX_FAIL else 'queued')
        rows.append(row)
    out = dict(version='sn120-performance-v1', updated_at=now(), updated_at_cr=cr(now()),
               about='Pass rate of each king (head) on the WHOLE public affine_math pool (7,496 MATH train problems the gate and miners draw from) '
                     'and on a private holdout (MATH test problems not in the pool; miners do not train on it and the gate never uses it). '
                     'Unlike the gate (a few hundred tasks chosen per round because the head solves them only sometimes), these are fixed sets, '
                     'so numbers are comparable across kings. Informational only: never used for promotions, weights or training.',
               sets=dict(full=dict(name='Full set', tasks=7496, source='affine_math pool = DigitalLearningGmbH/MATH-lighteval train rows with a \\boxed answer, env index order'),
                         holdout=dict(name='Holdout', tasks=HOLD['n'], sha256=HOLD['sha256'], source=HOLD['source'], dedupe_rule=HOLD['rule'],
                                      removed=HOLD['removed'], hash_rule=HOLD['hash_rule'], list_published=False)),
               method=dict(samples_per_task=K, seed=SEED, request_seed_rule='seed(set, task, j) = int(sha256(P + ":" + set + ":" + task + ":" + j)[:12], 16), P = seed',
                           harness=dict(enable_thinking=True, temperature=1.0, top_p=0.95, top_k=20, max_output_tokens=8192, max_context_tokens=12288),
                           prompt='system = the affine_math mandate, user = problem; MiMo chat template with thinking (full-set prompt ids identical to the gate probe)',
                           grader='gate probe grader: final channel after </think>, exactly one balanced \\boxed (mimo-final-channel-math-v1), math-verify 0.9.0 '
                                  'equivalence vs the reference \\boxed answer; else the probe lenient fix allan-lenient-v1-786cdb9. Capped replies (8192 tokens) count as fails.',
                           pass_rate='correct / graded samples (mean over tasks)', se='task-clustered standard error, sd(per-task pass) / sqrt(tasks)',
                           serving='vLLM 0.32.0 bf16, prefix caching, max-num-seqs 512 (probe serve.sh flags), 1 replica per GPU',
                           determinism='seeds and settings are fixed; vLLM batching is not bit-reproducible, so a re-run moves a number by about its SE'),
               machine=dict(pod=POD, gpu=GPU, cost_per_hour_usd=COST_H), holdout_guard=holdout_guard(),
               evaluating=(dict(king=cur, **({k: v for k, v in (S.get('running') or {}).items() if k in ('started',)})) if cur else None),
               kings=rows)
    wj(OUTJ, out)
def main():
    log(event='start', pid=os.getpid())
    while True:
        try:
            open(ST + '/heartbeat', 'w').write(str(now()))
            ks = kings(); km = {k['king']: k for k in ks}
            run = S.get('running')
            if run:
                k = run['king']; st = status(k)
                newest = ks[-1]['king'] if ks else None
                if st.get('res') == '1':
                    collect(k, km.get(k)); S['running'] = None; save()
                elif st.get('alive') == '0' or (now() - run['started'] > 600 and not st.get('prog') and st.get('rc')):
                    S['fail'][k] = S['fail'].get(k, 0) + 1; log(event='run-failed', king=k[:12], st=st); abort(k); S['running'] = None; save()
                else:
                    prog = json.loads(st['prog']) if st.get('prog') else None
                    stalled = (prog and now() - prog['at'] > STALL_S) or (not prog and now() - run['started'] > 2400)
                    if stalled:
                        S['fail'][k] = S['fail'].get(k, 0) + 1; log(event='stalled', king=k[:12]); abort(k); S['running'] = None; save()
                    elif run.get('priority') == 'backfill' and newest and not scored(newest) and newest != k:
                        log(event='preempt-backfill', king=k[:12], newest=newest[:12]); abort(k); S['running'] = None; save()
                    else:
                        S['progress'] = prog
            if not S.get('running'):
                k, pri = pick(ks)
                if k:
                    ok = ensure_ckpt(k)
                    if ok is None: S['unavailable'].append(k); save(); log(event='unavailable', king=k[:12])
                    elif not ok: S['fail'][k] = S['fail'].get(k, 0) + 1; save()
                    else:
                        launch(k); S['running']['priority'] = pri; save()
                        prune({k})
                        # prefetch the next one (disk: current + next only)
                        S2 = dict(S); nxt = None
                        for kk in reversed([x['king'] for x in ks]):
                            if kk != k and not scored(kk) and kk not in S['unavailable'] and S['fail'].get(kk, 0) < MAX_FAIL: nxt = kk; break
                        if nxt: ensure_ckpt(nxt); S['next'] = nxt; save()
            publish(ks, (S.get('running') or {}).get('king'))
        except Exception:
            log(event='error', tb=traceback.format_exc()[-800:])
        time.sleep(60)
if __name__ == '__main__':
    main()
