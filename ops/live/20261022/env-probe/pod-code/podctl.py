"""Pod-side control for the SN120 env probe (called over ssh by the Arbos Life daemon). Never touches the evaluator:
reads its checkpoint cache read-only, runs vLLM + the probe worker in /root/affine-env-probe-20261022 at nice 19 / ionice idle.
usage: podctl.py status | pause | resume | serve CKPT_ID GMU | stop-serve | start-run CKPT_ID (plan on stdin) | fetch (descriptor+urls on stdin)"""
import json, os, signal, subprocess, sys, time, urllib.request
D = '/root/affine-env-probe-20261022'; PORT = 18555; PAUSE = D + '/PAUSE'
EVPY = '/root/affine-mimo-20261013/venv/bin/python'
def pid_alive(pf):
    try: pid = int(open(pf).read().split()[0]); os.kill(pid, 0); return pid
    except Exception: return None
def served():
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models', timeout=3) as r: json.load(r); return open(D + '/serve.ckpt').read().strip()
    except Exception: return None
def gpu():
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20).stdout.strip().split(',')
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return dict(util=int(q[0]), mem_used_mib=int(q[1]), mem_total_mib=int(q[2]), apps=[a.replace(' ', '') for a in apps])
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
    return dict(at=time.time(), paused=os.path.exists(PAUSE), serve_pid=pid_alive(D + '/serve.pid'), served=served(), ckpts=ck,
                fetching=bool(pid_alive(D + '/fetch.pid')), gpu=gpu(), runs=runs)
cmd = sys.argv[1]
if cmd == 'status': print(json.dumps(status()))
elif cmd == 'pause': open(PAUSE, 'w').write(str(time.time())); print('{"paused":true}')
elif cmd == 'resume':
    if os.path.exists(PAUSE): os.remove(PAUSE)
    print('{"paused":false}')
elif cmd == 'stop-serve': kill(D + '/serve.pid'); print('{"stopped":true}')
elif cmd == 'serve':
    ck, gmu = sys.argv[2], sys.argv[3]; extra = sys.argv[4:]
    if served() == ck: print('{"serving":"%s"}' % ck); sys.exit(0)
    kill(D + '/serve.pid')
    open(D + '/serve.ckpt', 'w').write(ck)
    log = open(D + '/serve.log', 'ab')
    p = subprocess.Popen([D + '/serve.sh', f'{D}/ckpt/{ck}', str(PORT), gmu, *extra], stdout=log, stderr=log, start_new_session=True)
    open(D + '/serve.pid', 'w').write(str(p.pid)); print(json.dumps(dict(starting=ck, pid=p.pid)))
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
    ck = sys.argv[2]; plan = json.load(sys.stdin); rd = f'{D}/runs/{ck}'; os.makedirs(rd, exist_ok=True)
    if pid_alive(rd + '/worker.pid') or os.path.exists(rd + '/result.json'): print('{"already":true}'); sys.exit(0)
    plan.update(ckpt_dir=f'{D}/ckpt/{ck}', vllm_url=f'http://127.0.0.1:{PORT}', served_model='probe', pause_file=PAUSE,
                source_root='/root/affine-mimo-public-evaluator-20261013/source-final-v822b')
    envs = json.load(open(D + '/envs.json'))
    for e in plan['envs']: e['spec'] = envs[e['env_id']]['spec']; e['kind'] = envs[e['env_id']]['kind']
    if 'manifest_projection' not in plan:
        live = json.load(open(D + '/live_math_manifest_subset.json'))
        plan['manifest_projection'] = dict(mimo_answer_projection=live['mimo_answer_projection'], sampling_contract=live['sampling_contract'])
        plan['harness'] = live['harness']
    json.dump(plan, open(rd + '/plan.json', 'w'))
    p = subprocess.Popen(['nice', '-n', '15', EVPY, '-B', D + '/probe_worker.py', rd + '/plan.json', rd + '/result.json', '--progress', rd + '/progress.json'],
                         stdout=open(rd + '/worker.out', 'ab'), stderr=open(rd + '/worker.err', 'ab'), start_new_session=True, cwd=rd,
                         env=dict(os.environ, NLTK_DATA=D + '/nltk_data', HF_HOME=D + '/hf'))
    open(rd + '/worker.pid', 'w').write(str(p.pid)); print(json.dumps(dict(run_started=ck, pid=p.pid)))
elif cmd == 'result':
    print(open(f'{D}/runs/{sys.argv[2]}/result.json').read())
elif cmd == 'abort-run':
    kill(f'{D}/runs/{sys.argv[2]}/worker.pid'); print('{"aborted":true}')
