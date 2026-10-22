"""SN120 env probe daemon (Arbos Life, systemd user unit affine-env-probe.service).
Watches the lifecycle's published checkpoint (mimo-mainnet-v1/CURRENT.json), derives a public post-publication seed from
a finalized Finney block hash (READ-ONLY chain access), samples fresh random tasks from every probe env on the evaluator
pod (vLLM in a separate venv, graded with the evaluator's own env sessions/graders), and publishes per-env aggregates at
https://affine.io/probe/ (append-only). It stages, but NEVER submits, an on-chain commitment per checkpoint file.
The trainer is not involved. The live evaluator has priority: while an evaluator job is open the probe is paused.
usage: probe_daemon.py --config CONFIG.json [--once]"""
import argparse, hashlib, json, os, random, subprocess, sys, time, traceback, importlib.util
from pathlib import Path
VERSION = 'sn120-env-probe-v1'; SEED_RULE = 'sn120-probe-seed-rule-v1'; MARGIN = 10
LIFE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1')
SITE = Path('/home/const/subnet120/affine/website'); PUB = SITE / 'probe'
R = Path('/home/const/subnet120-rewrite/state/root-audits/20261022-env-probe'); ST = R / 'state'
OWNER = '5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8'
POD = ['ssh', '-n', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
       '-o', 'UserKnownHostsFile=/home/const/.lium/known_hosts/6e469697-f3a7-4a6a-a1ae-40dde7120581', '-p', '20092', 'root@38.255.28.20']
def cr(ts): return time.strftime('%Y-%m-%d %H:%M:%S CR (UTC-6)', time.gmtime(ts - 6 * 3600)) if ts else None
def log(**kw):
    kw['at'] = time.time(); kw['at_cr'] = cr(kw['at']); print(json.dumps(kw, default=str), flush=True)
def atomic(path, obj, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True); t = path.with_name('.' + path.name + '.tmp%d' % os.getpid())
    data = json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    fd = os.open(t, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, 'wb') as f: f.write(data)
    os.chmod(t, mode); os.replace(t, path); return hashlib.sha256(data).hexdigest()
def jl_append(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f: f.write(json.dumps(obj, sort_keys=True, default=str) + '\n')
def jl(path): return [json.loads(l) for l in open(path)] if path.exists() else []
def pod(*args, stdin=None, timeout=120):
    p = subprocess.run(POD + ['python3 /root/affine-env-probe-20261022/podctl.py ' + ' '.join(args)] if stdin is None else
                       [x for x in POD if x != '-n'] + ['python3 /root/affine-env-probe-20261022/podctl.py ' + ' '.join(args)],
                       input=stdin, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0: raise RuntimeError(f'podctl {args[0]} rc={p.returncode}: {p.stderr[-300:]}')
    return json.loads(p.stdout.strip().splitlines()[-1])

# ---------- lifecycle (read-only) ----------
def current():
    d = json.loads((LIFE / 'CURRENT.json').read_text()); c = d['checkpoint']
    a = d.get('checkpoint_authorization', {}); a = a.get('payload', a)
    return dict(id=c['id'], files=c['files'], epoch_index=d.get('epoch_index'), optimizer_step=d.get('optimizer_step'),
                run_id=a.get('run_id'), published_mtime=(LIFE / 'CURRENT.json').stat().st_mtime)
def evaluator_busy(max_age=4 * 3600):
    """An evaluator job is open: newest job file has no receipt yet and is recent/unexpired."""
    ev = LIFE / 'evaluator'; jobs = sorted(ev.glob('mimo-eval-*.job.json'), key=lambda p: p.stat().st_mtime)
    if not jobs: return False, None
    j = jobs[-1]; rec = ev / j.name.replace('.job.json', '.receipt.json')
    if rec.exists(): return False, j.name
    try: pl = json.loads(j.read_text())['payload']; exp = pl.get('expires_at', 0); created = pl.get('created_at', 0)
    except Exception: return True, j.name
    return (time.time() < exp and time.time() - created < max_age), j.name
def presigned_urls(ckpt_id):
    """Presigned R2 GET URLs for this checkpoint from the lifecycle's evaluator job (R2 read only). Never logged."""
    ev = LIFE / 'evaluator'
    for j in sorted(ev.glob('mimo-eval-*.job.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:6]:
        try:
            pl = json.loads(j.read_text())['payload']; m = pl['manifest']['payload']
            if m['checkpoint']['id'] == ckpt_id and time.time() < pl.get('expires_at', 0) - 600: return pl['checkpoint_read_urls']
        except Exception: continue
    return {}

# ---------- public post-publication seed (read-only chain) ----------
def derive_seed(ckpt, first_seen):
    spec = importlib.util.spec_from_file_location('_sb9', '/home/const/subnet120-rewrite/state/root-audits/20261020-measured-impact-v9/seed_block.py')
    sb = importlib.util.module_from_spec(spec); spec.loader.exec_module(sb)       # reuse the v9 read-only Chain client
    ch = sb.Chain()
    try:
        head, _ = ch.finalized(); target = int(first_seen * 1000)
        if ch.ts(head) < target: return None
        lo = max(1, head - int((ch.ts(head) - target) / 12000) - 400); hi = head
        while ch.ts(lo) >= target: lo = max(1, lo - 2000)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ch.ts(mid) >= target: hi = mid
            else: lo = mid
        anchor = hi; s = anchor + MARGIN
        if s > head: return None
        bh = ch.hash(s)
        return dict(rule=SEED_RULE, checkpoint_id=ckpt, anchor_time=first_seen, anchor_block=anchor, margin_blocks=MARGIN,
                    seed_block=s, seed_block_hash=bh, seed_block_ts_ms=ch.ts(s),
                    seed=hashlib.sha256(f'{SEED_RULE}:{ckpt}:{bh}'.encode()).hexdigest(),
                    text='anchor_block = first finalized Finney block with Timestamp.Now >= anchor_time (the probe first saw the '
                         'published checkpoint); seed_block = anchor_block + 10; seed = sha256(rule:checkpoint_id:seed_block_hash)')
    finally: ch.close()
def draw(seed, env, pool, n, exclude=()):
    """Without-replacement uniform draw of n task indices from range(pool), reproducible from the public seed."""
    rng = random.Random(int(hashlib.sha256(f'{seed}:{env}:indices'.encode()).hexdigest(), 16))
    ex = set(exclude); cand = [i for i in range(pool) if i not in ex]
    return sorted(rng.sample(cand, min(n, len(cand))))

# ---------- publication ----------
def publish(cfg, led, res, seed, cur):
    PUB.mkdir(parents=True, exist_ok=True); (PUB / 'c').mkdir(exist_ok=True)
    idx_path = PUB / 'index.json'; idx = json.loads(idx_path.read_text()) if idx_path.exists() else dict(version=VERSION, entries=[])
    prev = idx['entries'][-1] if idx['entries'] else None
    envs = {}
    for e, s in sorted(res['envs'].items()):
        graded = s['correct'] + s['wrong'] + s['unresolved']
        envs[e] = dict(pool=led['plan_pools'][e], tasks=s['tasks'], samples=s['samples'], correct=s['correct'], wrong=s['wrong'],
                       unresolved=s['unresolved'], budget_hits=s['budget_hits'], errors=s['errors'],
                       pass_rate=round(s['correct'] / graded, 6) if graded else None,
                       se=round(s['se_task_clustered'], 6) if s.get('se_task_clustered') is not None else None,
                       mean_out_tokens=round(s['mean_out_tokens'], 1) if s.get('mean_out_tokens') else None,
                       task_indices=led['plan_indices'][e], k=led['plan_k'][e])
    n = (prev['n'] + 1) if prev else 1
    doc = dict(version=VERSION, n=n, prev_sha256=prev['sha256'] if prev else None,
               checkpoint=dict(id=cur['id'], epoch_index=cur['epoch_index'], optimizer_step=cur['optimizer_step'], run_id=cur['run_id'],
                               descriptor_sha256=hashlib.sha256(json.dumps(cur['files'], sort_keys=True).encode()).hexdigest(),
                               published_at=cur['published_mtime'], published_at_cr=cr(cur['published_mtime']),
                               probe_first_seen_at=led['first_seen'], probe_first_seen_at_cr=cr(led['first_seen'])),
               seed=seed, sampling=dict(engine=res.get('engine', 'vllm'), harness=cfg['harness_public'], max_tokens=cfg['max_tokens'],
                                        grading='env verifiers from the pinned evaluator source; affine_math uses the live final-channel '
                                                'grader (completed boxed answer); other envs grade the final channel after </think>; '
                                                'no EOS / malformed thinking / budget = unresolved (not wrong)'),
               run=dict(started_at=res['started_at'], finished_at=res['finished_at'], started_at_cr=cr(res['started_at']), finished_at_cr=cr(res['finished_at']),
                        wall_seconds=round(res['wall_seconds'], 1), gen_tokens=res['gen_tokens'], rollouts=res['samples'],
                        preempted_for_evaluator=res.get('preempted_requests', 0), gpu=cfg['gpu_label']),
               envs=envs)
    name = f"e{cur['epoch_index']:06d}-{cur['id'][:12]}.json"; sha = atomic(PUB / 'c' / name, doc)
    chain = hashlib.sha256(((prev['chain'] if prev else '0' * 64) + sha).encode()).hexdigest()
    entry = dict(n=n, file=f'https://affine.io/probe/c/{name}', sha256=sha, chain=chain, checkpoint_id=cur['id'],
                 epoch_index=cur['epoch_index'], optimizer_step=cur['optimizer_step'], seed_block=seed['seed_block'],
                 finished_at=res['finished_at'], finished_at_cr=cr(res['finished_at']), onchain='staged-not-submitted')
    idx['entries'].append(entry); idx.update(updated_at=time.time(), updated_at_cr=cr(time.time()), latest_series='https://affine.io/probe/series.json',
                                             chain_rule='chain_n = sha256(hex(chain_{n-1}) + hex(sha256(file_n))), chain_0 = 64 zeros')
    ser_path = PUB / 'series.json'; ser = json.loads(ser_path.read_text()) if ser_path.exists() else dict(version=VERSION, points=[])
    ser['points'].append(dict(n=n, epoch_index=cur['epoch_index'], optimizer_step=cur['optimizer_step'], checkpoint_id=cur['id'],
                              seed_block=seed['seed_block'], finished_at=res['finished_at'], file_sha256=sha,
                              envs={e: dict(p=v['pass_rate'], se=v['se'], n=v['samples'], u=v['unresolved']) for e, v in envs.items()}))
    ser.update(updated_at=time.time(), updated_at_cr=cr(time.time()), index='https://affine.io/probe/index.json',
               note='Per-checkpoint per-env pass rates from fresh random tasks drawn with a post-publication block-hash seed. '
                    'p = correct/(correct+wrong+unresolved); se = task-clustered standard error. Aggregates only.')
    atomic(ser_path, ser); atomic(idx_path, idx)
    staged = dict(n=n, checkpoint_id=cur['id'], epoch_index=cur['epoch_index'], file=entry['file'], file_sha256=sha, chain=chain,
                  extrinsic=dict(pallet='Commitments', call='set_commitment', netuid=120,
                                 info=dict(fields=[[{'Sha256': '0x' + sha}], [{'Sha256': '0x' + chain}]])),
                  signer_hotkey=OWNER, status='STAGED-NOT-SUBMITTED (needs Jacob approval)', staged_at=time.time())
    jl_append(ST / 'staged-commitments.jsonl', staged)
    return entry

# ---------- main loop ----------
def tick(cfg):
    ledger = ST / 'ledger.json'; L = json.loads(ledger.read_text()) if ledger.exists() else {}
    cur = current(); c = cur['id']
    if c not in L:
        L[c] = dict(first_seen=time.time(), epoch_index=cur['epoch_index'], optimizer_step=cur['optimizer_step'], state='seen')
        log(event='new_checkpoint', ckpt=c[:12], epoch=cur['epoch_index'])
    for k, v in L.items():   # newest checkpoint wins; unfinished older ones are superseded
        if k != c and v['state'] not in ('published', 'superseded'):
            v['state'] = 'superseded'; log(event='superseded', ckpt=k[:12])
            try: pod('abort-run', k)
            except Exception: pass
    led = L[c]
    atomic(ledger, L, 0o600)
    if led['state'] == 'published': return 'idle'
    busy, job = evaluator_busy()
    st = pod('status')
    if busy and not st['paused']: pod('pause'); log(event='pause_for_evaluator', job=job)
    if not busy and st['paused']: pod('resume'); log(event='resume', job=job)
    if led['state'] == 'seen':
        s = derive_seed(c, led['first_seen'])
        if not s: return 'waiting_seed'
        atomic(ST / 'seeds' / f'{c}.json', s); led['seed'] = s
        envs = cfg['envs']
        led['plan_pools'] = {e: v['pool'] for e, v in envs.items()}
        led['plan_indices'] = {e: draw(s['seed'], e, v['pool'], v.get('n', cfg['n_tasks'])) for e, v in envs.items()}
        led['plan_k'] = {e: v.get('k', cfg['k']) for e in envs for v in [envs[e]]}
        led['state'] = 'seeded'; atomic(ledger, L, 0o600); log(event='seeded', ckpt=c[:12], seed_block=s['seed_block'])
    if led['state'] == 'seeded':
        if c not in st['ckpts']:
            if not st['fetching']:
                pod('fetch', stdin=json.dumps(dict(descriptor=dict(id=c, files=cur['files']), urls=presigned_urls(c))))
                log(event='fetch_started', ckpt=c[:12])
            return 'fetching'
        if busy and not cfg.get('serve_while_evaluator_busy', True): return 'yield'
        if st['served'] != c:
            if st['serve_pid'] is None or st['served'] not in (None, c) or not st.get('serve_pid'):
                pod('serve', c, str(cfg['gpu_memory_utilization'])); log(event='serve_started', ckpt=c[:12])
            return 'starting_vllm'
        plan = dict(checkpoint_id=c, seed=led['seed']['seed'], concurrency=cfg['concurrency'], grade_workers=cfg['grade_workers'],
                    envs=[dict(env_id=e, indices=led['plan_indices'][e], k=led['plan_k'][e], max_tokens=cfg['max_tokens']) for e in sorted(cfg['envs'])])
        pod('start-run', c, stdin=json.dumps(plan)); led['state'] = 'running'; led['run_started'] = time.time()
        atomic(ledger, L, 0o600); log(event='run_started', ckpt=c[:12], envs=len(plan['envs']))
        return 'running'
    if led['state'] == 'running':
        r = st['runs'].get(c, {})
        if r.get('result'):
            p = subprocess.run(POD + ['cat /root/affine-env-probe-20261022/runs/%s/result.json' % c], capture_output=True, text=True, timeout=120)
            res = json.loads(p.stdout); atomic(ST / 'results' / f'{c}.private.json', res, 0o600)
            entry = publish(cfg, led, res, led['seed'], cur)
            led['state'] = 'published'; led['published'] = entry; atomic(ledger, L, 0o600)
            log(event='published', ckpt=c[:12], file=entry['file'], sha256=entry['sha256'][:16], wall_s=round(res['wall_seconds']))
            if not cfg.get('keep_warm'): pod('stop-serve')
            return 'published'
        if not r.get('running'):
            led['state'] = 'seeded'; led['restarts'] = led.get('restarts', 0) + 1; atomic(ledger, L, 0o600)
            log(event='worker_died_restarting', ckpt=c[:12], err=r.get('err_tail', '')[-200:])
            if led['restarts'] > 5: led['state'] = 'failed'; atomic(ledger, L, 0o600)
        return 'running'
    return led['state']
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--once', action='store_true'); a = ap.parse_args()
    os.nice(19); ST.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            cfg = json.loads(Path(a.config).read_text())
            if cfg.get('enabled', True): s = tick(cfg)
            else: s = 'disabled'
            atomic(ST / 'daemon-status.json', dict(at=time.time(), at_cr=cr(time.time()), state=s), 0o600)
        except Exception as e:
            log(event='error', error=f'{type(e).__name__}: {e}', tb=traceback.format_exc()[-800:])
        if a.once: break
        time.sleep(30)
if __name__ == '__main__': main()
