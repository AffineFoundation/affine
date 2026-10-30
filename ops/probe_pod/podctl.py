"""Pod-side control for the SN120 env probe (called over ssh by the Arbos Life daemon). Never touches the evaluator:
reads its checkpoint cache read-only, runs vLLM + the probe worker in /root/affine-env-probe at nice 19 / ionice idle.
usage: podctl.py status | pause | resume | serve CKPT_ID GMU | stop-serve | start-run CKPT_ID (plan on stdin) | fetch (descriptor+urls on stdin)"""
import json, os, signal, subprocess, sys, time, urllib.request
D = '/root/affine-env-probe'; PORT = 18555; PAUSE = D + '/PAUSE'
EVPY = '/root/affine-mimo-20261013/venv/bin/python'
def pid_alive(pf):
    try: pid = int(open(pf).read().split()[0]); os.kill(pid, 0); return pid
    except Exception: return None
def ngpu():
    try: return len(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=20).stdout.strip().splitlines())
    except Exception: return 1
def replicas():
    try: return int(open(D + '/serve.replicas').read())
    except Exception: return 1
def up(i):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT + i}/v1/models', timeout=3) as r: json.load(r); return True
    except Exception: return False
def served():
    # checkpoint id only when every replica answers
    try:
        n = replicas()
        return open(D + '/serve.ckpt').read().strip() if all(up(i) for i in range(n)) else None
    except Exception: return None
def stop_all():
    for i in range(16):
        if os.path.exists(f'{D}/serve.{i}.pid'): kill(f'{D}/serve.{i}.pid')
    kill(D + '/serve.pid')
def gpu():
    try:
        rows = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        q = [[int(x) for x in r.split(',')] for r in rows]
        return dict(n=len(q), util=[r[0] for r in q], mem_used_mib=[r[1] for r in q], mem_total_mib=q[0][2])
    except Exception as e: return dict(error=str(e))
def kill(pf, sig=signal.SIGTERM):
    pid = pid_alive(pf)
    if pid:
        try: os.killpg(os.getpgid(pid), sig)
        except Exception: pass
        for _ in range(60):
            if not pid_alive(pf): break
            time.sleep(1)
        else:
            try: os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception: pass
def status():
    runs = {}
    for c in sorted(os.listdir(D + '/runs')) if os.path.isdir(D + '/runs') else []:
        rd = f'{D}/runs/{c}'; st = dict(result=os.path.exists(rd + '/result.json'), running=bool(pid_alive(rd + '/worker.pid')))
        if os.path.exists(rd + '/progress.json'):
            try: st['progress'] = json.load(open(rd + '/progress.json'))
            except Exception: pass
        if os.path.exists(rd + '/worker.err'):
            st['err_tail'] = open(rd + '/worker.err').read()[-300:]
        runs[c] = st
    ck = sorted(x for x in os.listdir(D + '/ckpt') if not x.endswith('.partial')) if os.path.isdir(D + '/ckpt') else []
    return dict(at=time.time(), paused=os.path.exists(PAUSE), hold=os.path.exists(D + '/HOLD'), replicas=replicas(), serve_procs=sum(bool(pid_alive(f'{D}/serve.{i}.pid')) for i in range(16)), serve_ckpt=(open(D + '/serve.ckpt').read().strip() if os.path.exists(D + '/serve.ckpt') else None), replicas_up=sum(up(i) for i in range(replicas())), served=served(), ckpts=ck,
                fetching=bool(pid_alive(D + '/fetch.pid')), gpu=gpu(), runs=runs)
cmd = sys.argv[1]
if cmd == 'status': print(json.dumps(status()))
elif cmd == 'pause': open(PAUSE, 'w').write(str(time.time())); print('{"paused":true}')
elif cmd == 'resume':
    if os.path.exists(PAUSE): os.remove(PAUSE)
    print('{"paused":false}')
elif cmd == 'serve-one':
    # (re)start a single replica i (e.g. after it crashed)
    i = int(sys.argv[2]); ck = open(D + '/serve.ckpt').read().strip(); gmu = sys.argv[3]
    kill(f'{D}/serve.{i}.pid'); log = open(f'{D}/serve.{i}.log', 'ab')
    p = subprocess.Popen([D + '/serve.sh', f'{D}/ckpt/{ck}', str(PORT + i), gmu], stdout=log, stderr=log, start_new_session=True,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i)))
    open(f'{D}/serve.{i}.pid', 'w').write(str(p.pid)); print(json.dumps(dict(replica=i, pid=p.pid)))
elif cmd == 'stop-serve': stop_all(); print('{"stopped":true}')
elif cmd == 'serve':
    # one vLLM replica per GPU (data parallel), ports PORT..PORT+n-1
    ck, gmu = sys.argv[2], sys.argv[3]; extra = sys.argv[4:]; n = ngpu()
    if served() == ck and replicas() == n: print('{"serving":"%s"}' % ck); sys.exit(0)
    stop_all()
    open(D + '/serve.ckpt', 'w').write(ck); open(D + '/serve.replicas', 'w').write(str(n))
    pids = []
    for i in range(n):
        log = open(f'{D}/serve.{i}.log', 'ab')
        swapcap = os.path.exists(D + '/WARM-SWAP')   # t739u: dev endpoints (loopback) for in-place weight reload
        p = subprocess.Popen([D + '/serve.sh', f'{D}/ckpt/{ck}', str(PORT + i), gmu, *extra], stdout=log, stderr=log, start_new_session=True,
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), **({'WAIT_PORT': str(PORT)} if (i and os.path.exists(D + '/SERIAL-SERVE')) else {}),   # probe-speed 20261030: replicas start in parallel (per-replica vcache); touch SERIAL-SERVE = old serial start
                                      **({'VLLM_SERVER_DEV_MODE': '1'} if swapcap else {})))
        open(f'{D}/serve.{i}.pid', 'w').write(str(p.pid)); pids.append(p.pid)
    open(D + '/serve.gen', 'w').write('0')   # t767u: cold serve resets the swap generation
    if os.path.exists(D + '/WARM-SWAP') and not extra: json.dump(dict(pids=pids, gmu=gmu, at=time.time()), open(D + '/serve.swapcap', 'w'))
    elif os.path.exists(D + '/serve.swapcap'): os.remove(D + '/serve.swapcap')
    print(json.dumps(dict(starting=ck, replicas=n, pids=pids)))
elif cmd == 'swap':
    # t739u: in-place weight swap on the running replicas (see patch_podctl_swap.py docstring)
    ck, gmu = sys.argv[2], sys.argv[3]; n = replicas(); t0 = time.time()
    if served() == ck: print(json.dumps(dict(serving=ck, swapped=False))); sys.exit(0)
    try: cap = json.load(open(D + '/serve.swapcap'))
    except Exception: cap = None
    pids = [pid_alive(f'{D}/serve.{i}.pid') for i in range(n)]
    cdir = f'{D}/ckpt/{ck}'
    why = (None if cap else 'not swap-capable') or (None if cap['pids'] == pids else 'replicas restarted since serve') or \
          (None if str(cap.get('gmu')) == str(gmu) else 'different gmu') or (None if served() else 'not all replicas up') or \
          (None if os.path.isdir(cdir) and not os.path.exists(cdir + '.partial') and os.path.exists(cdir + '/config.json') else 'ckpt incomplete')
    try: g0 = int(open(D + '/serve.gen').read())
    except Exception: g0 = 0
    why = why or (None if g0 < 24 else 'swap generation cap (24) -> cold serve')
    # swapsafe: never swap under another client (its in-flight requests block the prefix-cache reset and would mix weights)
    def metric(i, name):
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT + i}/metrics', timeout=5) as r: txt = r.read().decode()
            return sum(float(l.rsplit(' ', 1)[1]) for l in txt.splitlines() if l.startswith(name + '{'))
        except Exception: return None
    if not why:
        live_runs = sorted(r for r in (os.listdir(D + '/runs') if os.path.isdir(D + '/runs') else []) if pid_alive(f'{D}/runs/{r}/worker.pid'))
        if live_runs: why = 'run in flight: ' + ','.join(live_runs)[:160]
    if not why:
        infl = [None if metric(i, 'vllm:num_requests_running') is None else metric(i, 'vllm:num_requests_running') + (metric(i, 'vllm:num_requests_waiting') or 0) for i in range(n)]
        if any(x is None or x > 0 for x in infl): why = 'requests in flight on replicas: %s' % infl
    if why: print(json.dumps(dict(swapped=False, reason=why))); sys.exit(3)
    CANARY = ' '.join('swapsafe canary %d reset check block filler text.' % j for j in range(24))
    old = open(D + '/serve.ckpt').read().strip()
    open(D + '/serve.ckpt', 'w').write('swapping:' + ck)
    from concurrent.futures import ThreadPoolExecutor
    def post(i, path, body, timeout):
        req = urllib.request.Request(f'http://127.0.0.1:{PORT + i}{path}', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r: return r.status
    def canary(i):
        post(i, '/v1/completions', dict(model='probe', prompt=CANARY, max_tokens=1, temperature=0), 120)
    def one(i):
        canary(i)   # swapsafe: canary blocks are now cached (old weights)
        lp = f'{D}/serve.{i}.log'; off = os.path.getsize(lp) if os.path.exists(lp) else 0
        post(i, '/collective_rpc', dict(method='reload_weights', kwargs=dict(weights_path=cdir)), 900)
        post(i, '/reset_prefix_cache', {}, 60)
        time.sleep(1)
        with open(lp, 'rb') as fh: fh.seek(off); tail = fh.read().decode(errors='replace')
        if 'Failed to reset prefix cache' in tail: raise RuntimeError('replica %d: prefix cache reset failed (vLLM log)' % i)
        h0 = metric(i, 'vllm:prefix_cache_hits_total'); canary(i); h1 = metric(i, 'vllm:prefix_cache_hits_total')
        if h0 is None or h1 is None or h1 - h0 > 0: raise RuntimeError('replica %d: canary still hit the prefix cache after reset (hits %s -> %s)' % (i, h0, h1))
        return True
    try:
        with ThreadPoolExecutor(n) as ex: list(ex.map(one, range(n)))
        if [pid_alive(f'{D}/serve.{i}.pid') for i in range(n)] != pids or not all(up(i) for i in range(n)): raise RuntimeError('replica lost during swap')
    except Exception as e:
        stop_all(); open(D + '/serve.ckpt', 'w').write(ck)
        if os.path.exists(D + '/serve.swapcap'): os.remove(D + '/serve.swapcap')
        print(json.dumps(dict(swapped=False, reason='swap failed: %r' % (e,), stopped=True))); sys.exit(3)
    open(D + '/serve.ckpt', 'w').write(ck)
    try: g = int(open(D + '/serve.gen').read()) + 1
    except Exception: g = 1
    open(D + '/serve.gen', 'w').write(str(g))
    print(json.dumps(dict(swapped=True, previous=old, serving=ck, seconds=round(time.time() - t0, 1), gen=g)))
elif cmd == 'fetch':
    req = json.load(sys.stdin); ck = req['descriptor']['id']
    if pid_alive(D + '/fetch.pid'): print('{"fetching":true}'); sys.exit(0)
    os.makedirs(D + '/ckpt', exist_ok=True)
    json.dump(req['descriptor'], open(f'{D}/ckpt/{ck}.descriptor.json', 'w'))
    uf = f'{D}/ckpt/{ck}.urls.private.json'; fd = os.open(uf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f: json.dump(req.get('urls') or {}, f)
    # keep at most 2 probe checkpoints on disk (current + previous)
    for old in sorted((x for x in os.listdir(D + '/ckpt') if len(x) == 64 and x != ck), key=lambda x: os.path.getmtime(f'{D}/ckpt/{x}'))[:-1]:
        subprocess.run(['rm', '-rf', f'{D}/ckpt/{old}'])
    log = open(f'{D}/ckpt/{ck}.fetch.log', 'ab')
    p = subprocess.Popen(['nice', '-n', '19', 'python3', D + '/fetch_ckpt.py', f'{D}/ckpt/{ck}.descriptor.json', D + '/ckpt', uf], stdout=log, stderr=log, start_new_session=True)
    open(D + '/fetch.pid', 'w').write(str(p.pid)); print(json.dumps(dict(fetch_started=ck)))
elif cmd == 'start-run':
    ck = sys.argv[2]; plan = json.load(sys.stdin); rd = f"{D}/runs/{plan.get('run_name') or ck}"; os.makedirs(rd, exist_ok=True)
    if pid_alive(rd + '/worker.pid'): print('{"already":true}'); sys.exit(0)
    if os.path.exists(rd + '/result.json') and not plan.get('resume'):
        print('{"already":true}'); sys.exit(0)
    for stale in ('FINALIZE', 'result.json', 'worker.pid'):
        if plan.get('resume') and stale == 'result.json' and os.path.exists(rd + '/result.json'):
            try: os.remove(rd + '/result.json')
            except Exception: pass
        elif stale == 'FINALIZE' and os.path.exists(rd + '/' + stale):
            try: os.remove(rd + '/' + stale)
            except Exception: pass
    if 'concurrency_per_replica' in plan: plan['concurrency'] = plan['concurrency_per_replica'] * min(int(plan.get('replicas') or replicas()), replicas())
    nrep = min(int(plan.get('replicas') or replicas()), replicas())
    plan.update(ckpt_dir=f'{D}/ckpt/{ck}', vllm_url=f'http://127.0.0.1:{PORT}', vllm_urls=[f'http://127.0.0.1:{PORT + i}' for i in range(nrep)],
                served_model='probe', pause_file=PAUSE,
                source_root='/root/affine-mimo-public-evaluator-20261013/source-final-v822b')
    envs = json.load(open(D + '/envs.json'))
    for e in plan['envs']: e['spec'] = envs[e['env_id']]['spec']; e['kind'] = envs[e['env_id']]['kind']
    if 'manifest_projection' not in plan:
        live = json.load(open(D + '/live_math_manifest_subset.json'))
        plan['manifest_projection'] = dict(mimo_answer_projection=live['mimo_answer_projection'], sampling_contract=live['sampling_contract'])
        plan['harness'] = live['harness']
    json.dump(plan, open(rd + '/plan.json', 'w'))
    p = subprocess.Popen([EVPY, '-B', D + '/probe_worker.py', rd + '/plan.json', rd + '/result.json', '--progress', rd + '/progress.json'],
                         stdout=open(rd + '/worker.out', 'ab'), stderr=open(rd + '/worker.err', 'ab'), start_new_session=True, cwd=rd,
                         env=dict(os.environ, NLTK_DATA=D + '/nltk_data', HF_HOME=D + '/hf'))
    open(rd + '/worker.pid', 'w').write(str(p.pid)); print(json.dumps(dict(run_started=ck, pid=p.pid)))
elif cmd == 'result':
    print(open(f'{D}/runs/{sys.argv[2]}/result.json').read())
elif cmd == 'abort-run':
    kill(f'{D}/runs/{sys.argv[2]}/worker.pid'); print('{"aborted":true}')
elif cmd == 'finalize-run':
    # continuous-partial: ask worker to stop and write result.json (partial OK). Falls back to SIGTERM.
    ck = sys.argv[2]; rd = f'{D}/runs/{ck}'; wait_s = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    if os.path.exists(rd + '/result.json'):
        print(json.dumps(dict(finalized=True, already=True, partial=bool(json.load(open(rd + '/result.json')).get('partial')))))
        sys.exit(0)
    open(rd + '/FINALIZE', 'w').write(str(time.time()))
    # also nudge with SIGTERM so a stuck sample loop notices promptly
    pid = pid_alive(rd + '/worker.pid')
    if pid:
        try: os.kill(pid, signal.SIGTERM)
        except Exception: pass
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if os.path.exists(rd + '/result.json'):
            res = json.load(open(rd + '/result.json'))
            print(json.dumps(dict(finalized=True, partial=bool(res.get('partial')), samples=res.get('samples'), stop_reason=res.get('stop_reason'))))
            sys.exit(0)
        if not pid_alive(rd + '/worker.pid'):
            break
        time.sleep(1)
    # worker gone without result: assemble from grades.jsonl if any
    gj = rd + '/grades.jsonl'
    if os.path.exists(gj) and os.path.getsize(gj) > 0:
        plan = json.load(open(rd + '/plan.json')) if os.path.exists(rd + '/plan.json') else {}
        stats = {}
        for line in open(gj):
            try: g = json.loads(line)
            except Exception: continue
            e = g['env_id']; st = stats.setdefault(e, dict(tasks=0, samples=0, correct=0, wrong=0, unresolved=0, budget_hits=0, errors=0, out_tokens=0, prompt_tokens=0, per_task={}, unresolved_reasons={}))
            idx = str(g['index']);
            if idx in st['per_task']: continue
            pt = {k: g[k] for k in ('n','c','w','u','b','x','tok','L') if k in g}
            st['per_task'][idx] = pt
            st['tasks'] += 1; st['samples'] += pt.get('n', 0); st['correct'] += pt.get('c', 0); st['wrong'] += pt.get('w', 0)
            st['unresolved'] += pt.get('u', 0); st['budget_hits'] += pt.get('b', 0); st['errors'] += pt.get('x', 0); st['out_tokens'] += pt.get('tok', 0)
        import math
        envs = {}
        for env_id, st in stats.items():
            n = st['samples']; graded = st['correct'] + st['wrong'] + st['unresolved']
            p = st['correct'] / graded if graded else None
            pts = [v for v in st['per_task'].values() if v.get('n')]
            se = None
            if p is not None and len(pts) > 1:
                m = len(pts); rs = [v['c'] / v['n'] for v in pts]; mean = sum(rs) / m
                se = math.sqrt(sum((r - mean) ** 2 for r in rs) / (m - 1) / m)
            envs[env_id] = dict({k: v for k, v in st.items() if k != 'per_task'}, pass_rate=p, se_task_clustered=se,
                                mean_out_tokens=(st['out_tokens'] / n) if n else None, mean_prompt_tokens=None, per_task=st['per_task'])
        samples = sum(st['samples'] for st in stats.values())
        t0 = time.time()
        res = dict(version='sn120-env-probe-worker-v1', checkpoint_id=plan.get('checkpoint_id', ck), seed=plan.get('seed'),
                   started_at=t0, finished_at=t0, wall_seconds=0, gen_tokens=0, samples=samples, tok_per_s=0, rollouts_per_min=0,
                   request_latency_s=dict(p50=None, max=None), concurrency=plan.get('concurrency'), replicas=None,
                   requests_per_replica=[], transient_retries=0, preempted_requests=0,
                   partial=True, stop_reason='assembled_from_grades', planned_samples=sum(len(e.get('indices', [])) * e.get('k', 1) for e in plan.get('envs', [])),
                   envs=envs, engine='vllm')
        json.dump(res, open(rd + '/result.json.tmp', 'w')); os.replace(rd + '/result.json.tmp', rd + '/result.json')
        print(json.dumps(dict(finalized=True, assembled=True, partial=True, samples=samples)))
        sys.exit(0)
    print(json.dumps(dict(finalized=False, error='no result and no grades', worker_alive=bool(pid_alive(rd + '/worker.pid')))))
    sys.exit(2)
