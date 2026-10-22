"""SN120 env probe worker (runs on the evaluator pod with the EVALUATOR's venv + pinned source for env sessions/graders).
Generation goes to a local vLLM server (separate venv/process). Grading reuses subnet.environments sessions and, for
affine_math, the evaluator's exact final-channel grader (subnet.mimo_answer_projection.grade).
usage: python probe_worker.py PLAN.json OUT.json [--progress P.json]
Plan fields: checkpoint_id, ckpt_dir, vllm_url, served_model, source_root, concurrency, grade_workers, harness,
             manifest_projection (manifest subset for the math grader), envs: [{env_id, spec, kind, indices, k, max_tokens}]
Output: per-env aggregates + per-task counts (private) + throughput stats. No rollout text is stored."""
import asyncio, json, math, os, sys, time, multiprocessing as mp, statistics, hashlib, signal, traceback
PLAN = json.load(open(sys.argv[1])); OUT = sys.argv[2]
PROG = sys.argv[sys.argv.index('--progress') + 1] if '--progress' in sys.argv else None
PAUSE = PLAN.get('pause_file')
sys.path.insert(0, PLAN['source_root']); sys.path.append('/root/affine-env-probe-20261022/extra-site')
EOS = (248046, 248044); OPEN_T, CLOSE_T = 248068, 248069
_G = {}
def _init(plan_path):
    os.nice(10)
    p = json.load(open(plan_path)); sys.path.insert(0, p['source_root']); sys.path.append('/root/affine-env-probe-20261022/extra-site')
    from transformers import AutoTokenizer
    _G['tok'] = AutoTokenizer.from_pretrained(p['ckpt_dir']); _G['plan'] = p; _G['sessions'] = {}
    _G['envs'] = {e['env_id']: e for e in p['envs']}
def _session(env_id):
    from subnet.environments import EnvironmentSpec, create_session
    s = _G['sessions'].get(env_id)
    if s is None:
        s = _G['sessions'][env_id] = create_session(EnvironmentSpec.from_dict(_G['envs'][env_id]['spec']))
    return s
def prep(env_id, index):
    """Reset one task -> chat-template token ids (exact evaluator kwargs)."""
    s = _session(env_id); r = s.reset(index, 0)
    if r.get('tools'): raise RuntimeError('probe is single-turn text only')
    ids = _G['tok'].apply_chat_template(r['messages'], tokenize=True, return_dict=False, add_generation_prompt=True,
                                        enable_thinking=_G['plan']['harness']['enable_thinking'])
    if isinstance(ids, dict): ids = ids['input_ids']
    return env_id, index, r['task_hash'], list(ids)
def _final_channel(tok, out, stop):
    if stop != 'eos' or not out or out[-1] not in EOS: return None, 'not-completed-eos'
    body = out[:-1]; special = set(getattr(tok, 'all_special_ids', ()))
    if any(t in special or t in EOS for t in body): return None, 'unexpected-control-token'
    if body.count(OPEN_T) != 1 or body.count(CLOSE_T) != 1: return None, 'thinking-marker-count'
    a, b = body.index(OPEN_T), body.index(CLOSE_T)
    if a >= b or tok.decode(body[:a], skip_special_tokens=False).strip(): return None, 'thinking-marker-order'
    final = tok.decode(body[b + 1:], skip_special_tokens=False).strip()
    if not final: return None, 'empty-final'
    return final, 'ok'
def grade(env_id, index, out, stop):
    try:
        e = _G['envs'][env_id]; s = _session(env_id); s.reset(index, 0); tok = _G['tok']
        if e['kind'] == 'live-math':
            from subnet.mimo_answer_projection import grade as g
            _, kind, reward = g(s, _G['plan']['manifest_projection'], _G['plan']['harness'], tok, out, stop)
            return env_id, index, kind, None
        final, why = _final_channel(tok, out, stop)
        if final is None: return env_id, index, 'neutral', why
        st = s.step({'text': final})
        if not st.get('done'): return env_id, index, 'neutral', 'not-done'
        return env_id, index, ('positive' if st['classification'] == 'positive' else 'negative' if st['classification'] == 'negative' else 'neutral'), None
    except Exception as x:
        return env_id, index, 'indeterminate', f'{type(x).__name__}: {str(x)[:120]}'

async def main():
    import httpx
    P = PLAN; t_start = time.time()
    pool = mp.get_context('spawn').Pool(P.get('grade_workers', 32), initializer=_init, initargs=(sys.argv[1],))
    loop = asyncio.get_running_loop()
    def apool(fn, *a):
        f = loop.create_future()
        pool.apply_async(fn, a, callback=lambda r: loop.call_soon_threadsafe(f.set_result, r),
                         error_callback=lambda x: loop.call_soon_threadsafe(f.set_exception, x))
        return f
    stats = {e['env_id']: dict(tasks=0, samples=0, correct=0, wrong=0, unresolved=0, budget_hits=0, errors=0, out_tokens=0,
                               prompt_tokens=0, per_task={}, unresolved_reasons={}) for e in P['envs']}
    gen_tokens = 0; done_samples = 0; lat = []; preempted = [0]
    sem = asyncio.Semaphore(P['concurrency'])
    client = httpx.AsyncClient(timeout=httpx.Timeout(3600.0), limits=httpx.Limits(max_connections=P['concurrency'] + 8))
    async def one_task(e, index):
        nonlocal gen_tokens, done_samples
        env_id = e['env_id']; st = stats[env_id]
        try:
            _, _, th, ids = await apool(prep, env_id, index)
        except Exception as x:
            st['errors'] += e['k']; st['per_task'][str(index)] = dict(err=f'prep {type(x).__name__}'); return
        max_tok = min(e['max_tokens'], P['harness']['max_context_tokens'] - len(ids))
        if max_tok <= 0: st['errors'] += e['k']; return
        base = dict(model=P['served_model'], prompt=ids, n=1, max_tokens=max_tok, temperature=P['harness']['temperature'],
                    top_p=P['harness']['top_p'], top_k=P['harness']['top_k'], stop_token_ids=list(EOS), skip_special_tokens=False,
                    return_token_ids=True)
        async def sample(j):
            # one request per sample with its own public seed (vLLM n>1 + seed can repeat samples)
            body = dict(base, seed=int(hashlib.sha256(f"{P['seed']}:{env_id}:{index}:{j}".encode()).hexdigest()[:12], 16))
            while True:
                while PAUSE and os.path.exists(PAUSE): await asyncio.sleep(5)
                async with sem:
                    if PAUSE and os.path.exists(PAUSE): continue
                    t0 = time.time(); req = asyncio.create_task(client.post(P['vllm_url'] + '/v1/completions', json=body))
                    while not req.done():
                        await asyncio.wait({req}, timeout=5)
                        if not req.done() and PAUSE and os.path.exists(PAUSE):
                            req.cancel(); preempted[0] += 1; break    # evaluator busy: abort in-flight, retry later
                    if req.cancelled(): continue
                    r = req.result(); r.raise_for_status(); lat.append(time.time() - t0)
                    return r.json()['choices'][0]
        choices = await asyncio.gather(*(sample(j) for j in range(e['k'])))
        j = dict(choices=choices)
        st['tasks'] += 1; st['prompt_tokens'] += len(ids); pt = dict(n=0, c=0, w=0, u=0, b=0, x=0, tok=0)
        futs = []
        for ch in j['choices']:
            out = list(ch.get('token_ids') or [])
            fr = ch.get('finish_reason')
            stop = 'budget' if fr == 'length' else 'eos'
            if stop == 'eos' and (not out or out[-1] not in EOS):
                sr = ch.get('stop_reason'); out.append(sr if isinstance(sr, int) and sr in EOS else EOS[1])
            gen_tokens += len(out); pt['tok'] += len(out); st['out_tokens'] += len(out)
            if stop == 'budget': st['budget_hits'] += 1; pt['b'] += 1
            futs.append(apool(grade, env_id, index, out, stop))
        for env_, idx_, kind, why in await asyncio.gather(*futs):
            st['samples'] += 1; pt['n'] += 1; done_samples += 1
            if kind == 'positive': st['correct'] += 1; pt['c'] += 1
            elif kind == 'negative': st['wrong'] += 1; pt['w'] += 1
            elif kind == 'neutral':
                st['unresolved'] += 1; pt['u'] += 1
                if why: st['unresolved_reasons'][why] = st['unresolved_reasons'].get(why, 0) + 1
            else:
                st['errors'] += 1; pt['x'] += 1; st['unresolved_reasons'][why or 'err'] = st['unresolved_reasons'].get(why or 'err', 0) + 1
        st['per_task'][str(index)] = pt
    async def progress():
        while True:
            await asyncio.sleep(30)
            if PROG:
                el = time.time() - t_start
                json.dump(dict(at=time.time(), elapsed=el, preempted=preempted[0], paused=bool(PAUSE and os.path.exists(PAUSE)), samples=done_samples, gen_tokens=gen_tokens, tok_s=gen_tokens / max(el, 1)), open(PROG + '.tmp', 'w')); os.replace(PROG + '.tmp', PROG)
    pj = asyncio.create_task(progress())
    order = [(e, i) for e in P['envs'] for i in e['indices']]
    # interleave envs so partial runs still cover every env
    order.sort(key=lambda t: (t[0]['indices'].index(t[1]), t[0]['env_id']))
    await asyncio.gather(*(one_task(e, i) for e, i in order))
    pj.cancel(); await client.aclose(); pool.close()
    el = time.time() - t_start
    envs = {}
    for env_id, st in stats.items():
        n = st['samples']; graded = st['correct'] + st['wrong'] + st['unresolved']
        p = st['correct'] / graded if graded else None
        # cluster-robust SE over tasks (samples within a task are correlated)
        pts = [v for v in st['per_task'].values() if isinstance(v, dict) and v.get('n')]
        se = None
        if p is not None and len(pts) > 1:
            m = len(pts); rs = [v['c'] / v['n'] for v in pts]; mean = sum(rs) / m
            se = math.sqrt(sum((r - mean) ** 2 for r in rs) / (m - 1) / m)
        envs[env_id] = dict({k: v for k, v in st.items() if k not in ('per_task',)}, pass_rate=p, se_task_clustered=se,
                            mean_out_tokens=(st['out_tokens'] / n) if n else None, mean_prompt_tokens=(st['prompt_tokens'] / st['tasks']) if st['tasks'] else None,
                            per_task=st['per_task'])
    res = dict(version='sn120-env-probe-worker-v1', checkpoint_id=P['checkpoint_id'], seed=P['seed'], started_at=t_start, finished_at=time.time(),
               wall_seconds=el, gen_tokens=gen_tokens, samples=done_samples, tok_per_s=gen_tokens / el, rollouts_per_min=done_samples / el * 60,
               request_latency_s=dict(p50=statistics.median(lat) if lat else None, max=max(lat) if lat else None),
               concurrency=P['concurrency'], preempted_requests=preempted[0], envs=envs)
    json.dump(res, open(OUT + '.tmp', 'w')); os.replace(OUT + '.tmp', OUT)
    print(json.dumps({k: v for k, v in res.items() if k != 'envs'}))
if __name__ == '__main__':
    asyncio.run(main())
