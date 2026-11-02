#!/usr/bin/env python3
"""SN120 performance eval service (host side; Jacob pt911u 2026-10-31). READ-ONLY w.r.t. the subnet: it reads the gate ledger
(LEDGER.public.jsonl) to learn the kings, fetches published king checkpoints from R2 (gate presign.py) to its OWN pod (calm-orbit-ef,
pp.sh), runs perf_eval.py there and publishes aggregates to affine.io/performance.json. It never writes gate/lifecycle/trainer/weights
state; nothing in the gate, promotions, weights or the trainer reads its output.
Scheduling: the newest king first (as soon as it is crowned); when idle, backfill older kings newest->oldest. A newer GREEN never
preempts a running newest-king eval; it preempts a backfill eval. Stale kings in between are only scored later as backfill.
Pod disk: only the king being evaluated + the next one are kept."""
import json, os, subprocess, sys, time, glob, hashlib, traceback, re, collections, math
HERE = os.path.dirname(os.path.abspath(__file__)); RA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fetch_king
from fetch_king import fetch, descriptor
ST = RA + '/state'; RES = ST + '/results'; os.makedirs(RES, exist_ok=True)
SITE = '/home/const/subnet120/affine/website'; OUTJ = SITE + '/performance.json'
# pt1130u/pt1132u 2026-11-02: the full/holdout series is private (no chart on affine.io). Full doc -> OUTP (0600);
# the public performance.json keeps only the king list (the bench daemon reads kings[] from it).
OUTP = ST + '/performance.private.json'
LEDGER = '/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/LEDGER.public.jsonl'
FFL = '/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs'
# Two eval pods (Jacob 11:10 CR): pod 1 = newest king first + history; pod 2 = history only, removed automatically when history is done.
PODS = {'calm-orbit-ef': dict(pp=RA + '/pp.sh', gpu='2xH200', cost=9.52, lium_id='3222a673-c449-4ef5-a2c0-5c3bdf950e64', role='primary'),
        'eager-wolf-98': dict(pp=RA + '/pp2.sh', gpu='1xB200', cost=6.10, lium_id='2ad91129-cb87-4070-aa9e-942ee07ac329', role='history',
                              teardown=True, xcheck='c03ad1395b0625490b6b213549ba6763b4d3b9a6ff46c07529e9e32c4474569f')}
POD = os.environ.get('PERF_POD', 'calm-orbit-ef'); CFG = PODS[POD]
PP = CFG['pp']; GPU = CFG['gpu']; COST_H = CFG['cost']; ROLE = CFG['role']; fetch_king.PP = PP
CL = ST + '/claims'; XC = ST + '/xcheck'; os.makedirs(CL, exist_ok=True); os.makedirs(XC, exist_ok=True)
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
HH = HERE + '/head_history.json'   # every head since E0 (head_history.py; lifecycle input_checkpoint per epoch, joined with the R2 listing)
def kings():
    """All heads in crowned order: the static lifecycle history (pre-gate + gate era) plus any GREEN in the gate ledger since."""
    out, seen = [], set()
    for h in rj(HH, []) or []:
        if h['era'] == 'initial' and not h['in_r2']: continue
        out.append(dict(king=h['king'], epoch=h['crowned_epoch'] if h['crowned_epoch'] is not None else h['first_head_epoch'],
                        head_from_epoch=h['first_head_epoch'], parent=None, crowned_at=h['crowned_at'],
                        crowned_at_cr=cr(h['crowned_at']) if h['crowned_at'] else None, era=h['era'], in_r2=h['in_r2']))
        seen.add(h['king'])
    for l in open(LEDGER):
        try: r = json.loads(l)
        except Exception: continue
        if r.get('decision') == 'green' and r.get('applied') == 'green':
            g = dict(parent=r['head'], crowned_at=(r.get('timestamps') or {}).get('decided_at'), crowned_at_cr=r.get('decided_at_cr'), gate_z=r.get('z'), era='gate')
            if r['candidate'] in seen:
                for o in out:
                    if o['king'] == r['candidate']: o.update(g)
            else:
                out.append(dict(king=r['candidate'], epoch=r['epoch'], head_from_epoch=r['epoch'] + 1, in_r2=True, **g)); seen.add(r['candidate'])
    return out
def tiers(ks):
    """Backfill order. A: one king per 25 epochs over the whole history (fills the chart's span fast); B: one per 10 epochs + every
    gate-era GREEN; C: every other recoverable king. Oldest -> newest within a tier."""
    def grid(step):
        sel = []
        for g in range(0, (max(k['epoch'] for k in ks) // step + 1) * step + 1, step):
            c = [k for k in ks if k['epoch'] >= g and k['in_r2']]
            if c: sel.append(min(c, key=lambda k: k['epoch'])['king'])
        return sel
    A = grid(25); B = grid(10) + [k['king'] for k in ks if k.get('era') == 'gate']; C = [k['king'] for k in ks]
    order, seen = [], set(); pos = {k['king']: i for i, k in enumerate(ks)}
    for t in (A, B, C):
        for kk in sorted(set(t) - seen, key=lambda x: pos[x]): order.append(kk); seen.add(kk)
    return order
SF = ST + '/service-%s.json' % POD
if not os.path.exists(SF) and ROLE == 'primary' and os.path.exists(ST + '/service.json'): os.replace(ST + '/service.json', SF)
S = rj(SF, {}) or {}
S.setdefault('fail', {}); S.setdefault('unavailable', [])
def save(): wj(SF, S)
# ---- shared claims: a king is scored by exactly one pod. state/claims/<king> is created with O_EXCL (atomic on the host fs).
def claim_owner(k):
    try: return json.load(open(f'{CL}/{k}')).get('pod')
    except Exception: return None
def claim(k):
    try:
        fd = os.open(f'{CL}/{k}', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, 'w') as f: json.dump(dict(pod=POD, at=now(), at_cr=cr(now())), f)
        return True
    except FileExistsError:
        return claim_owner(k) == POD
def release(k):
    if claim_owner(k) == POD:
        try: os.unlink(f'{CL}/{k}')
        except FileNotFoundError: pass
def claimed_by_other(k):
    o = claim_owner(k); return o is not None and o != POD
def scored(k): return os.path.exists(f'{RES}/{k}.json')
def wdir(k): return f'/data/perf/runs/{k}'
def pick(ks):
    """(king, priority): the newest king first; then the backfill tiers."""
    if not ks: return None, None
    ok = lambda kk: not scored(kk) and S['fail'].get(kk, 0) < MAX_FAIL and kk not in S['unavailable'] and not claimed_by_other(kk)
    newest = ks[-1]['king']
    if ROLE == 'primary' and ok(newest): return newest, 'newest'
    for kk in tiers(ks):
        if ROLE != 'primary' and kk == newest: continue   # the newest king is pod 1's
        if ok(kk) and next(k for k in ks if k['king'] == kk)['in_r2']: return kk, 'backfill'
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
CAP_SWITCH = '/home/const/subnet120-rewrite/state/root-audits/20261102-task-cap-12k/SWITCH.json'
def perf_cap(k):
    """12K task cap (2026-11-02): kings crowned at/after the first 12K gate round's opening epoch -> the gate's max_tokens; older -> 8192."""
    sw = rj(CAP_SWITCH) or {}
    if sw.get('active') is not True or sw.get('opened_epoch') is None: return 8192
    try: ep = next(x['epoch'] for x in kings() if x['king'] == k)
    except Exception: return 8192
    return int(sw['max_tokens']) if ep is not None and int(ep) >= int(sw['opened_epoch']) else 8192
def launch(k):
    pod(f'rm -rf {wdir(k)}; mkdir -p {wdir(k)}')
    mt = perf_cap(k)
    cmd = (f"nohup setsid bash /data/perf/run_king.sh {k} {SEED} {K['full']} {K['holdout']} {mt} > /data/perf/run-{k[:8]}.log 2>&1 < /dev/null & "
           f"sleep 2; cat {wdir(k)}/run.pid")
    try: rc, o, e = pod(cmd, timeout=40)
    except subprocess.TimeoutExpired: o = ''
    S['running'] = dict(king=k, started=now(), run_pid=o.strip(), max_tokens=mt); save(); log(event='launched', king=k[:12], run_pid=o.strip(), max_tokens=mt)
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
    r = json.loads(o); r['king_meta'] = meta; r['machine'] = dict(pod=POD, lium_id=CFG['lium_id'], gpu=GPU, cost_per_hour_usd=COST_H)
    r['evaluated_at'] = now(); r['evaluated_at_cr'] = cr(now())
    rc, o2, _ = pod(f'cat {wdir(k)}/serving_at', timeout=30)
    try: r['serve_s'] = int(o2.strip()) - int(S['running']['started'])
    except Exception: pass
    if (S.get('running') or {}).get('xcheck'):
        ref = rj(f'{RES}/{k}.json'); cmp = {}
        for s_ in ('full', 'holdout'):
            a, b = ref['sets'][s_], r['sets'][s_]; se = math.sqrt(a['se'] ** 2 + b['se'] ** 2)
            cmp[s_] = dict(pod1=a['pass_rate'], pod2=b['pass_rate'], diff=b['pass_rate'] - a['pass_rate'], se_diff=se,
                           z=(b['pass_rate'] - a['pass_rate']) / se, within_1se=abs(b['pass_rate'] - a['pass_rate']) <= se)
        r['cross_check'] = dict(king=k, ref_pod=ref['machine']['pod'], pod=POD, sets=cmp)
        wj(f'{XC}/{k}.{POD}.json', r); wj(XC + '/LATEST.json', r['cross_check']); S['xcheck_done'] = True; save()
        log(event='xcheck', king=k[:12], cmp=cmp); return
    wj(f'{RES}/{k}.json', r); release(k); log(event='scored', king=k[:12], full=r['sets']['full']['pass_rate'], holdout=r['sets']['holdout']['pass_rate'], wall=r['wall_seconds'])
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
def backfill_summary(ks):
    t = tiers(ks); sc = {k['king'] for k in ks if scored(k['king'])}
    per = 1050   # measured: ~15-18 min eval + ~1.2 min warm serve + fetch overlap
    left = [kk for kk in t if kk not in sc and next(k for k in ks if k['king'] == kk)['in_r2']]
    return dict(kings_total=len(ks), recoverable=sum(1 for k in ks if k['in_r2']), scored=len(sc), remaining=len(left),
                eta_hours_all=round(len(left) * per / 3600, 1), seconds_per_king_est=per,
                order='newest king first; backfill A one per 25 epochs, B one per 10 epochs + every gate GREEN, C the rest (oldest->newest in each)',
                crowned_at_note='gate era: GREEN decision time; pre-gate: completion time of the training step that produced the checkpoint (every trained checkpoint became the next head)')
def publish(ks, cur=None):
    rows = []
    for k in ks:
        r = rj(f"{RES}/{k['king']}.json")
        row = dict(king=k['king'], epoch=k['epoch'], head_from_epoch=k.get('head_from_epoch'), era=k.get('era'), parent=k.get('parent'), crowned_at=k['crowned_at'], crowned_at_cr=k['crowned_at_cr'])
        if r:
            row.update(status='scored', evaluated_at_cr=r.get('evaluated_at_cr'), eval_wall_s=round(r['wall_seconds'], 1), seed=r['seed'], k=r['k'],
                       pod=(r.get('machine') or {}).get('pod', 'calm-orbit-ef'), pod_lium_id=(r.get('machine') or {}).get('lium_id', PODS['calm-orbit-ef']['lium_id']),
                       gpu=(r.get('machine') or {}).get('gpu', '2xH200'), max_tokens=int((r.get('harness') or {}).get('max_output_tokens', 8192)))
            for s in ('full', 'holdout'):
                x = r['sets'][s]
                row[s] = dict(pass_rate=x['pass_rate'], se=x['se'], tasks=x['tasks'], samples=x['samples'], graded_samples=x['graded_samples'],
                              correct=x['correct'], cap_rate=x['cap_rate'], indeterminate=x['indeterminate'], mean_out_tokens=round(x['mean_out_tokens'], 1))
        else:
            row['status'] = ('evaluating' if (cur and cur == k['king']) or claim_owner(k['king']) else 'unavailable' if (k['king'] in S['unavailable'] or not k.get('in_r2'))
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
                           task_cap=('max_output_tokens follows the gate: 8,192 (context 12,288) for kings crowned before the first 12K gate round; '
                                     '12,288 (context 16,384) for kings crowned from it (Jacob 2026-11-02). Each king row carries its max_tokens.'),
                           prompt='system = the affine_math mandate, user = problem; MiMo chat template with thinking (full-set prompt ids identical to the gate probe)',
                           grader='gate probe grader: final channel after </think>, exactly one balanced \\boxed (mimo-final-channel-math-v1), math-verify 0.9.0 '
                                  'equivalence vs the reference \\boxed answer; else the probe lenient fix allan-lenient-v1-786cdb9. Capped replies (at the king\'s max_tokens) count as fails.',
                           pass_rate='correct / graded samples (mean over tasks)', se='task-clustered standard error, sd(per-task pass) / sqrt(tasks)',
                           serving='vLLM 0.32.0 bf16, prefix caching, max-num-seqs 512 (probe serve.sh flags), 1 replica per GPU',
                           determinism='seeds and settings are fixed; vLLM batching is not bit-reproducible, so a re-run moves a number by about its SE'),
               machine=dict(pod=POD, gpu=GPU, cost_per_hour_usd=COST_H),
               machines=[dict(pod=p, lium_id=c['lium_id'], gpu=c['gpu'], cost_per_hour_usd=c['cost'], role=c['role'],
                              removed=(rj(f'{ST}/service-{p}.json', {}) or {}).get('torn_down')) for p, c in PODS.items()],
               cross_check=rj(XC + '/LATEST.json'), holdout_guard=holdout_guard(), backfill=backfill_summary(ks),
               evaluating=(dict(king=cur, **({k: v for k, v in (S.get('running') or {}).items() if k in ('started',)})) if cur else None),
               kings=rows)
    wj(OUTP, out, 0o600)
    wj(OUTJ, dict(version='sn120-kings-v1', updated_at=out['updated_at'], updated_at_cr=out['updated_at_cr'],
                  about='King (head) list in crowned order.',
                  kings=[{f: r.get(f) for f in ('king', 'epoch', 'head_from_epoch', 'era', 'parent', 'crowned_at', 'crowned_at_cr')} for r in rows]))
def teardown():
    """History pod only: remove THIS pod (exact lium id) once no history king is left to claim. Logged; never touches other pods."""
    lid = CFG['lium_id']
    p = subprocess.run(['/home/const/.lium/bin/lium', 'rm', lid, '-y'], capture_output=True, text=True, timeout=300)
    S['torn_down'] = dict(at=now(), at_cr=cr(now()), lium_id=lid, rc=p.returncode, out=(p.stdout + p.stderr)[-300:]); save()
    log(event='teardown', pod=POD, lium_id=lid, rc=p.returncode, out=(p.stdout + p.stderr)[-300:])
    with open(RA + '/PROGRESS.md', 'a') as f:
        f.write('- %s AUTO-TEARDOWN pod %s (lium %s) rc=%s: history queue empty\n' % (cr(now()), POD, lid, p.returncode))
def main():
    log(event='start', pid=os.getpid(), pod=POD, role=ROLE)
    run0 = (S.get('running') or {}).get('king')
    for f in glob.glob(CL + '/*'):   # stale own claims from a crash (idempotent restart)
        k = os.path.basename(f)
        if claim_owner(k) == POD and k != run0: release(k); log(event='released-stale-claim', king=k[:12])
    if run0 and not (S['running'].get('xcheck')): claim(run0)
    while True:
        try:
            open(ST + '/heartbeat-' + POD, 'w').write(str(now()))
            if ROLE == 'primary': open(ST + '/heartbeat', 'w').write(str(now()))
            if S.get('torn_down'):
                time.sleep(300); continue
            ks = kings(); km = {k['king']: k for k in ks}
            run = S.get('running')
            if run:
                k = run['king']; st = status(k)
                newest = ks[-1]['king'] if ks else None
                if st.get('res') == '1':
                    collect(k, km.get(k)); S['running'] = None; save()
                elif st.get('alive') == '0' or (now() - run['started'] > 600 and not st.get('prog') and st.get('rc')):
                    S['fail'][k] = S['fail'].get(k, 0) + 1; log(event='run-failed', king=k[:12], st=st); abort(k); release(k); S['running'] = None; save()
                else:
                    prog = json.loads(st['prog']) if st.get('prog') else None
                    stalled = (prog and now() - prog['at'] > STALL_S) or (not prog and now() - run['started'] > 2400)
                    if stalled:
                        S['fail'][k] = S['fail'].get(k, 0) + 1; log(event='stalled', king=k[:12]); abort(k); release(k); S['running'] = None; save()
                    elif ROLE == 'primary' and run.get('priority') == 'backfill' and newest and not scored(newest) and newest != k \
                            and not claimed_by_other(newest):
                        log(event='preempt-backfill', king=k[:12], newest=newest[:12]); abort(k); release(k); S['running'] = None; save()
                    else:
                        S['progress'] = prog
            if not S.get('running'):
                xk = CFG.get('xcheck')
                if xk and not S.get('xcheck_done') and scored(xk) and S['fail'].get('xcheck', 0) < MAX_FAIL:
                    if ensure_ckpt(xk):
                        prune({xk}); launch(xk); S['running'].update(priority='xcheck', xcheck=True); save()
                    else: S['fail']['xcheck'] = S['fail'].get('xcheck', 0) + 1; save()
                else:
                    k, pri = pick(ks)
                    if k and claim(k):
                        ok = ensure_ckpt(k)
                        if ok is None: S['unavailable'].append(k); release(k); save(); log(event='unavailable', king=k[:12])
                        elif not ok: S['fail'][k] = S['fail'].get(k, 0) + 1; release(k); save()
                        else:
                            prune({k}); launch(k); S['running']['priority'] = pri; save()
                            nxt = None   # prefetch (disk: current + next only); not claimed, so the other pod may still take it
                            for kk in tiers(ks):
                                if ROLE != 'primary' and kk == ks[-1]['king']: continue
                                if kk != k and not scored(kk) and not claim_owner(kk) and kk not in S['unavailable'] and S['fail'].get(kk, 0) < MAX_FAIL:
                                    nxt = kk; break
                            if nxt: ensure_ckpt(nxt); S['next'] = nxt; save()
                    elif not k and ROLE != 'primary' and CFG.get('teardown') and S.get('xcheck_done'):
                        teardown()
            if ROLE == 'primary': publish(ks, (S.get('running') or {}).get('king'))
        except Exception:
            log(event='error', tb=traceback.format_exc()[-800:])
        time.sleep(60)
if __name__ == '__main__':
    main()
