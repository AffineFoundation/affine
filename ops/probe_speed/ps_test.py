"""probe-speed real-rollout grader identity test (probe pod, idle gap only). EVPY -B ps_test.py gen|grade PLAN [label worker setup_once NP]
gen: k=2 rollouts for the first 150 plan tasks from the served checkpoint (plan harness) -> /tmp/pstest/rollouts.json (saved rollouts)
grade: grade every saved rollout with WORKER (module file) in a fresh spawn pool -> /tmp/pstest/grades-LABEL.json"""
import sys, os, json, time, multiprocessing as mp, importlib.util
T = '/tmp/pstest'; os.makedirs(T, exist_ok=True)
ARGS = json.loads(os.environ['PST_ARGS']) if 'PST_ARGS' in os.environ else sys.argv[1:]   # spawn children re-import main with argv rewritten
MODE, PL = ARGS[0], ARGS[1]
if MODE == 'grade': LABEL, WF, ONCE, NP = ARGS[2], ARGS[3], ARGS[4] == '1', int(ARGS[5])
else: WF = ARGS[2]
os.environ['PST_ARGS'] = json.dumps(ARGS)
sys.argv = ['probe_worker.py', PL, T + '/unused-out.json']
spec = importlib.util.spec_from_file_location('pw_under_test', WF); W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
if hasattr(W, 'SETUP_SWITCH'): W.SETUP_SWITCH = T + '/SETUP-ONCE.json'   # never the live switch
def job(a):
    r = W.grade('affine_math', a[0], a[1], a[2]); return [a[0], a[2], r[2], r[3], W._G.get('setup_once')]
def winit(pl):
    if hasattr(W, 'SETUP_SWITCH'): W.SETUP_SWITCH = T + '/SETUP-ONCE.json'
    W._init(pl)
if __name__ == '__main__':
    P = json.load(open(PL))
    if MODE == 'gen':
        W._init(PL)
        from subnet.environments import EnvironmentSpec, _source_hash
        sp = EnvironmentSpec.from_dict(P['envs'][0]['spec']); ver = {sp.id: _source_hash(sp)}
        rows = json.load(open(os.path.join(W.PREP_DIR, W._prep_cache_key(P, P['envs'][0], ver) + '.json')))['rows']
        idx = [i for i in P['envs'][0]['indices'] if str(i) in rows][:150]
        import httpx, concurrent.futures as cf
        urls = P.get('vllm_urls') or [P['vllm_url']]; H = P['harness']
        def gen(a):
            n, (i, j) = a; ids = rows[str(i)][1]
            body = dict(model=P['served_model'], prompt=ids, n=1, max_tokens=min(8192, H['max_context_tokens'] - len(ids)), temperature=H['temperature'],
                        top_p=H['top_p'], top_k=H['top_k'], stop_token_ids=list(W.EOS), skip_special_tokens=False, return_token_ids=True, seed=7000 * i + j)
            ch = httpx.post(urls[n % len(urls)] + '/v1/completions', json=body, timeout=900).json()['choices'][0]
            out = list(ch.get('token_ids') or []); stop = 'budget' if ch.get('finish_reason') == 'length' else 'eos'
            if stop == 'eos' and (not out or out[-1] not in W.EOS): out.append(W.EOS[1])
            return [i, out, stop]
        t = time.time()
        with cf.ThreadPoolExecutor(400) as ex: items = list(ex.map(gen, enumerate([(i, j) for i in idx for j in range(2)])))
        json.dump(items, open(T + '/rollouts.json', 'w'))
        print(json.dumps(dict(generated=len(items), s=round(time.time() - t, 1), capped=sum(x[2] == 'budget' for x in items))), flush=True)
    else:
        items = json.load(open(T + '/rollouts.json'))
        json.dump(dict(on=ONCE, by='ps test'), open(T + '/SETUP-ONCE.json', 'w'))
        pool = mp.get_context('spawn').Pool(NP, initializer=winit, initargs=(PL,))
        pool.map(job, items[:NP], chunksize=1)   # warm: pool init + first real setup in each worker
        t = time.time(); r = pool.map(job, items, chunksize=1); el = time.time() - t
        pool.terminate(); pool.join()
        json.dump(r, open(T + '/grades-%s.json' % LABEL, 'w'))
        from collections import Counter
        print(json.dumps(dict(label=LABEL, worker=os.path.basename(WF), once=ONCE, n=len(r), s=round(el, 1), per_s=round(len(r) / el, 1),
                              kinds=Counter('%s:%s' % (x[1], x[2]) for x in r), once_active=sorted(set(str(x[4]) for x in r)))), flush=True)
