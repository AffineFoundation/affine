#!/usr/bin/env python3
"""SN120 performance eval worker (pod side; Jacob pt911u 2026-10-31). Scores ONE king on the full 7,496-task affine_math pool and the
private MATH-test holdout with the held-out gate's generation settings, prompt format and grader, fixed seeds, deterministic grading.
Generation: vLLM OpenAI server(s) started with the probe's serve.sh flags; one /v1/completions request per sample, prompt = token ids,
temperature/top_p/top_k/max_tokens from the gate harness, seed(set,task,j) = int(sha256(P:set:task:j)[:12],16) with a fixed P.
Grading (same as probe_worker.py grade() for affine_math): capped reply -> unresolved; else mimo_answer_projection.project (token-level
final channel, exactly one balanced \\boxed) + verify.py semantics (math-verify 0.9.0, gold parsed as \\boxed{gold}); if not positive,
the probe's lenient fallback (allan-lenient-v1-786cdb9, verbatim source shipped from probe_worker.py). pass = correct / graded.
usage: perf_eval.py PLAN.json OUT.json   (PLAN: king, ckpt_dir, urls, tasks_file, k:{full,holdout}, seed, concurrency, grade_workers)"""
import asyncio, hashlib, json, math, os, random, resource, sys, time, multiprocessing as mp
PLAN = json.load(open(sys.argv[1])); OUT = sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
EOS = (248046, 248044); OPEN_T, CLOSE_T = 248068, 248069
SYSTEM = ("Solve the math problem. Reason step by step in plain text, then end your response with the final answer in `\\boxed{}`. "
          "Emit exactly one `\\boxed{}` block and nothing after it.")
H = dict(enable_thinking=True, max_context_tokens=12288, max_output_tokens=8192, temperature=1.0, top_k=20, top_p=0.95)
_G = {}
def _init(ckpt_dir):
    os.nice(5)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt_dir); _G['tok'] = tok
    assert tok.convert_tokens_to_ids('<think>') == OPEN_T and tok.convert_tokens_to_ids('</think>') == CLOSE_T
    _G['special'] = set(getattr(tok, 'all_special_ids', ()))
    ns = {'__name__': 'allan_lenient_v1'}
    exec(compile('import re\nfrom math_verify import parse, verify\n' + open(os.path.join(HERE, 'allan_src.py')).read(), 'allan_lenient_v1', 'exec'), ns)
    _G['allan'] = ns
def _boxed_final(text):   # subnet/mimo_answer_projection._boxed_final (verbatim logic)
    if text.count('\\boxed{') != 1: return None
    begin = text.index('\\boxed{'); start = begin + len('\\boxed{'); depth = 1
    for at in range(start, len(text)):
        if text[at] == '{': depth += 1
        elif text[at] == '}':
            depth -= 1
            if depth == 0:
                if not text[start:at].strip(): return None
                tail = text[at + 1:].strip(); prefix = text[:begin].rstrip()
                if tail:
                    opening = {'\\]': '\\[', '\\)': '\\(', '$$': '$$', '$': '$'}.get(tail)
                    if opening is None or not prefix.endswith(opening): return None
                elif prefix.endswith(('\\[', '\\(', '$')): return None
                return text[begin:at + 1]
    return None
def project(out):   # subnet/mimo_answer_projection.project, enable_thinking=True, stop == 'eos'
    tok = _G['tok']
    if not out or out[-1] not in EOS: return None
    body = list(out[:-1])
    if any(t in _G['special'] or t in EOS for t in body): return None
    if body.count(OPEN_T) != 1 or body.count(CLOSE_T) != 1: return None
    a = body.index(OPEN_T); b = body.index(CLOSE_T)
    if a >= b or tok.decode(body[:a], skip_special_tokens=False).strip(): return None
    decoded = [tok.decode(ids, skip_special_tokens=False) for ids in (body[:a], body[a + 1:b], body[b + 1:])]
    if any('<think>' in t or '</think>' in t for t in decoded): return None
    return _boxed_final(decoded[-1].strip())
def _last_boxed(text):   # verify.py last_boxed
    found = None; O = "\\boxed{"; start = text.find(O)
    while start != -1:
        depth = 0
        for i in range(start + len(O) - 1, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0: found = text[start:i + 1]; break
        start = text.find(O, start + len(O))
    return found
def native(gold, reply):   # verify.py semantics: True/False; raises -> indeterminate (verify.py exit != 0)
    from math_verify import parse, verify
    g = parse("\\boxed{" + gold + "}", raise_on_error=True)
    if not g: raise RuntimeError('grader_reference_unparseable')
    pred = _last_boxed(reply)
    if pred is None: return False
    return bool(verify(g, parse(pred, raise_on_error=True), raise_on_error=True))
def grade(out, stop, gold, prompt):
    """-> (kind, how). kind in positive/negative/neutral/indeterminate (probe_worker._grade for live-math)."""
    try:
        if stop != 'eos': return 'neutral', 'capped'
        box = project(out)
        try: ok = native(gold, box or '')
        except Exception as x: return 'indeterminate', type(x).__name__
        kind = 'neutral' if box is None else ('positive' if ok else 'negative'); how = 'live'
        if kind != 'positive':
            try:
                ns = _G['allan']; text = _G['tok'].decode(out[:-1], skip_special_tokens=False)
                gbox, viol, pfix = ns['lenient_project'](text)
                if gbox is not None:
                    ok2, efix = ns['lenient_equal'](gold, gbox, prompt)
                    if ok2: kind = 'positive'; how = '+'.join(x for x in (pfix, efix) if x) or 'live'
            except Exception: pass
        return kind, how
    except Exception as x:
        return 'indeterminate', 'err ' + type(x).__name__
def render(prompt):
    tok = _G['tok']
    ids = tok.apply_chat_template([dict(role='system', content=SYSTEM), dict(role='user', content=prompt)], tokenize=True,
                                  return_dict=False, add_generation_prompt=True, enable_thinking=True)
    return list(ids['input_ids'] if isinstance(ids, dict) else ids)
def seed_of(P, s, tid, j): return int(hashlib.sha256(f"{P}:{s}:{tid}:{j}".encode()).hexdigest()[:12], 16)
async def main():
    import httpx
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE); resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 1 << 20), hard))
    P = PLAN; T = json.load(open(P['tasks_file'])); tasks = T['tasks']
    _init(P['ckpt_dir'])
    # prompt ids: full set = probe prep-cache ids (evaluator env reset + chat template); check our rendering reproduces them
    mism = 0
    for t in tasks:
        r = render(t['prompt'])
        if t['set'] == 'full':
            if r != t['ids']: mism += 1
        else: t['ids'] = r
    if mism: print('WARNING: %d full-set prompts render differently from the probe prep cache; using the prep-cache ids' % mism, flush=True)
    ctx = mp.get_context('spawn'); pool = ctx.Pool(P.get('grade_workers', 48), initializer=_init, initargs=(P['ckpt_dir'],))
    loop = asyncio.get_running_loop()
    def apool(*a):
        f = loop.create_future()
        pool.apply_async(grade, a, callback=lambda r: loop.call_soon_threadsafe(f.set_result, r),
                         error_callback=lambda x: loop.call_soon_threadsafe(f.set_exception, x))
        return f
    URLS = P['urls']; inflight = [0] * len(URLS); sem = asyncio.Semaphore(P.get('concurrency', 1024))
    client = httpx.AsyncClient(timeout=httpx.Timeout(3600.0), limits=httpx.Limits(max_connections=P.get('concurrency', 1024) + 8))
    GJ = OUT + '.grades.jsonl'; done = {}
    if os.path.exists(GJ):   # resume
        for l in open(GJ):
            try: g = json.loads(l); done[(g['set'], g['id'])] = g
            except Exception: pass
    gl = open(GJ, 'a'); t0 = time.time(); stats = dict(gen_tokens=0, retries=0, samples=0)
    async def sample(t, j):
        mt = min(H['max_output_tokens'], H['max_context_tokens'] - len(t['ids']))
        body = dict(model='perf', prompt=t['ids'], n=1, max_tokens=mt, temperature=H['temperature'], top_p=H['top_p'], top_k=H['top_k'],
                    stop_token_ids=list(EOS), skip_special_tokens=False, return_token_ids=True, seed=seed_of(P['seed'], t['set'], t['id'], j))
        while True:
            async with sem:
                ri = min(range(len(URLS)), key=lambda i: inflight[i]); inflight[ri] += 1
                try:
                    r = await client.post(URLS[ri] + '/v1/completions', json=body); r.raise_for_status(); return r.json()['choices'][0]
                except (httpx.TransportError, httpx.HTTPStatusError):
                    stats['retries'] += 1
                    if stats['retries'] > 2000: raise
                finally: inflight[ri] -= 1
            await asyncio.sleep(2 + random.random() * 3)
    async def one(t):
        k = P['k'][t['set']]
        chs = await asyncio.gather(*(sample(t, j) for j in range(k)))
        rec = dict(set=t['set'], id=t['id'], n=0, c=0, w=0, u=0, x=0, b=0, tok=0, how={})
        futs = []
        for ch in chs:
            out = list(ch.get('token_ids') or []); stop = 'budget' if ch.get('finish_reason') == 'length' else 'eos'
            if stop == 'eos' and (not out or out[-1] not in EOS):
                sr = ch.get('stop_reason'); out.append(sr if isinstance(sr, int) and sr in EOS else EOS[1])
            rec['tok'] += len(out); stats['gen_tokens'] += len(out); rec['b'] += stop == 'budget'
            futs.append(apool(out, stop, t['answer'], t['prompt']))
        for kind, how in await asyncio.gather(*futs):
            key = {'positive': 'c', 'negative': 'w', 'neutral': 'u'}.get(kind, 'x'); rec[key] += 1; rec['n'] += 1; stats['samples'] += 1
            if kind == 'positive': rec['how'][how] = rec['how'].get(how, 0) + 1
        gl.write(json.dumps(rec) + '\n'); gl.flush(); done[(t['set'], t['id'])] = rec
    # deterministic interleaved order (full and holdout progress together)
    todo = [t for t in tasks if (t['set'], t['id']) not in done]
    todo.sort(key=lambda t: hashlib.sha256(f"{P['seed']}:order:{t['set']}:{t['id']}".encode()).hexdigest())
    async def prog():
        while True:
            await asyncio.sleep(15); el = time.time() - t0
            json.dump(dict(at=time.time(), elapsed=el, tasks_done=len(done), tasks_total=len(tasks), samples=stats['samples'],
                           tok_s=stats['gen_tokens'] / max(el, 1), retries=stats['retries']), open(OUT + '.progress', 'w'))
    pj = asyncio.create_task(prog())
    await asyncio.gather(*(one(t) for t in todo))
    pj.cancel(); await client.aclose(); pool.terminate(); pool.join(); gl.close()
    el = time.time() - t0; res = dict(version='sn120-perf-eval-v1', king=P['king'], seed=P['seed'], k=P['k'], wall_seconds=el,
                                     gen_tokens=stats['gen_tokens'], retries=stats['retries'], prompt_render_mismatches=mism,
                                     holdout_sha256=T['holdout_sha256'], harness=H, sets={})
    for s in ('full', 'holdout'):
        recs = [r for (ss, _), r in done.items() if ss == s]
        g = [r for r in recs if r['n'] - r['x'] > 0]
        rates = [r['c'] / (r['n'] - r['x']) for r in g]; m = len(rates)
        p = sum(rates) / m if m else None
        C = sum(r['c'] for r in recs); G = sum(r['n'] - r['x'] for r in recs)
        se = math.sqrt(sum((x - p) ** 2 for x in rates) / (m - 1) / m) if m > 1 else None
        how = {}
        for r in recs:
            for kk, v in r['how'].items(): how[kk] = how.get(kk, 0) + v
        res['sets'][s] = dict(tasks=len(recs), tasks_graded=m, samples=sum(r['n'] for r in recs), graded_samples=G, correct=C,
                              pass_rate=p, pass_rate_samples=(C / G if G else None), se=se,
                              cap_rate=sum(r['b'] for r in recs) / max(1, sum(r['n'] for r in recs)),
                              unresolved=sum(r['u'] for r in recs), indeterminate=sum(r['x'] for r in recs),
                              mean_out_tokens=sum(r['tok'] for r in recs) / max(1, sum(r['n'] for r in recs)), positive_by=how)
    json.dump(res, open(OUT + '.tmp', 'w'), indent=1); os.replace(OUT + '.tmp', OUT)
    print(json.dumps(res['sets']), flush=True)
if __name__ == '__main__':
    asyncio.run(main())
