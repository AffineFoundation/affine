#!/usr/bin/env python3
"""Combined experiment: independent seeds vs CRN vs log-prob vs CRN+log-prob, all on the SAME K tasks of one pair.

Phases (one engine, one GPU):
  load B -> gen B(seeds S) -> lp(Bsamples|B) -> swap A -> gen A(S) [CRN] -> gen A(S') [independent]
  -> lp(Bsamples|A), lp(Asamples|A) -> swap B -> lp(Asamples|B)
Per task t, correctness c in {0,1} (unresolved = 0, indeterminate dropped):
  d_ind  = mean c(A,S')  - mean c(B,S)
  d_crn  = mean c(A,S)   - mean c(B,S)
  d_lp   = SNIS_A[c over B samples, w=exp(logpA-logpB)] - mean c(B)          (log-prob only: 1 generation)
  d_rev  = mean c(A,S) - SNIS_B[c over A samples, w=exp(logpB-logpA)]
  d_comb = min-variance linear combination of (d_crn, d_lp, d_rev) (weights from task-level covariance)
  s_norm = mean dlogp/len (correct) - mean dlogp/len (wrong), B samples, tasks with both   (the original LP score)
Writes per-task values + timings to results-combined/<id>.jsonl.
"""
import argparse, json, math, os, signal, subprocess, socket, sys, time, types
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')


def preflight():
    # The earlier engine-init crash was an orphaned VLLM::EngineCore (from a killed run) still holding the GPU.
    g = os.environ.get('CUDA_VISIBLE_DEVICES')
    if g is None or ',' in g:
        raise SystemExit('set CUDA_VISIBLE_DEVICES to exactly one GPU')
    out = subprocess.check_output(['nvidia-smi', '-i', g, '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits']).decode()
    used, total = (float(x) for x in out.strip().split(','))
    if used > 0.05 * total:
        raise SystemExit(f'GPU {g} already has {used:.0f} MiB in use (another engine or an orphaned EngineCore); refusing to start')


def snis(c, lw):
    m = max(lw); w = [math.exp(x - m) for x in lw]; s = sum(w)
    return sum(wi * ci for wi, ci in zip(w, c)) / s, s * s / sum(wi * wi for wi in w)


def main():
    import sampler, fetch, grader
    ap = argparse.ArgumentParser(); ap.add_argument('--pair', required=True); ap.add_argument('-K', type=int, default=128)
    ap.add_argument('-m', type=int, default=8); ap.add_argument('--id', required=True); a0 = ap.parse_args()
    preflight()
    idx = json.load(open(HERE / 'data/index.json')); tasks = json.load(open(HERE / 'data/original-math7496.tasks.json'))
    b, c = (sampler.find(idx, x) for x in a0.pair.split(':'))
    a = types.SimpleNamespace(tp=1, gpu_mem=0.9, max_num_seqs=1024, max_batched_tokens=16384, enforce_eager=False,
                              engine='swap', max_tokens=8192, K=a0.K, m=a0.m, crn=False)  # crn=False -> draw has both S and S'
    rt = idx['checkpoints'][-1]['mimo_runtime']; vocab = idx['checkpoints'][-1]['vocab_size']
    d = sampler.new_draw(a, idx); S, S2 = d['seeds']['before'], d['seeds']['after']
    T = {}; t0 = time.time()
    eng = sampler.Engine(a); pool = grader.pool(16)
    signal.signal(signal.SIGTERM, lambda *_: (eng.close(), sys.exit(143)))
    pb, _ = fetch.ensure(b, '/root/ckpt-cache'); pc, _ = fetch.ensure(c, '/root/ckpt-cache')
    t = time.time(); eng.load(b, pb); T['load_B'] = time.time() - t
    tok = eng.tokenizer; special = set(getattr(tok, 'all_special_ids', ()))
    P = [list(tok.apply_chat_template([dict(role='system', content=tasks[i]['data']['system_prompt']), dict(role='user', content=tasks[i]['data']['prompt'])],
              tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False)) for i in d['chosen']]
    gold = [tasks[i]['data']['answer'] for i in d['chosen']]

    def gen(seeds, label):
        t = time.time(); outs = eng.generate(P, seeds, rt); T[f'gen_{label}'] = time.time() - t
        t = time.time(); flat = [(ti, o) for ti, ss in enumerate(outs) for o in ss]
        gts = [grader.project(tok, o, special, vocab)[0] for _, o in flat]
        kinds = list(pool.map(grader.classify, gts, [gold[ti] for ti, _ in flat], chunksize=8))
        T[f'verify_{label}'] = time.time() - t
        cc = [None if k == 'indeterminate' else (1.0 if k == 'positive' else 0.0) for k, _ in kinds]
        return outs, [cc[i * a.m:(i + 1) * a.m] for i in range(len(P))]

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    def lp(outs, label):
        idxs = [(ti, j) for ti, ss in enumerate(outs) for j, o in enumerate(ss) if o and len(P[ti]) + len(o) < rt['max_context_tokens']]
        reqs = [TokensPrompt(prompt_token_ids=P[ti] + outs[ti][j]) for ti, j in idxs]
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0, detokenize=False)
        t = time.time(); res = eng.llm.generate(reqs, sp, use_tqdm=False); T[f'lp_{label}'] = time.time() - t
        out = [[None] * a.m for _ in P]
        for (ti, j), r in zip(idxs, res):
            o = outs[ti][j]; pl = r.prompt_logprobs[len(P[ti]):]
            out[ti][j] = sum(x[tt].logprob for x, tt in zip(pl, o))
        return out

    oB, cB = gen(S, 'B')
    lBB = lp(oB, 'Bsamp_on_B')
    t = time.time(); eng.load(c, pc); T['swap_to_A'] = time.time() - t
    oA, cA = gen(S, 'A_crn')
    _, cA2 = gen(S2, 'A_indep')
    lBA = lp(oB, 'Bsamp_on_A'); lAA = lp(oA, 'Asamp_on_A')
    t = time.time(); eng.load(b, pb); T['swap_to_B'] = time.time() - t
    lAB = lp(oA, 'Asamp_on_B')

    per = []
    for ti in range(len(P)):
        def mean(v): v = [x for x in v if x is not None]; return sum(v) / len(v) if v else None
        mB, mA, mA2 = mean(cB[ti]), mean(cA[ti]), mean(cA2[ti])
        fb = [(cB[ti][j], (lBA[ti][j] - lBB[ti][j]) if lBA[ti][j] is not None else 0.0, len(oB[ti][j])) for j in range(a.m) if cB[ti][j] is not None]
        fa = [(cA[ti][j], (lAB[ti][j] - lAA[ti][j]) if lAB[ti][j] is not None else 0.0, len(oA[ti][j])) for j in range(a.m) if cA[ti][j] is not None]
        if None in (mB, mA, mA2) or not fb or not fa:
            continue
        accA_is, essA = snis([x[0] for x in fb], [x[1] for x in fb])
        accB_is, essB = snis([x[0] for x in fa], [x[1] for x in fa])
        nc = [x[1] / max(x[2], 1) for x in fb if x[0] == 1.0]; nw = [x[1] / max(x[2], 1) for x in fb if x[0] == 0.0]
        per.append(dict(task=d['chosen'][ti], cB=mB, cA_crn=mA, cA_ind=mA2, d_ind=mA2 - mB, d_crn=mA - mB,
                        d_lp=accA_is - mB, d_rev=mA - accB_is, ess_fwd=essA, ess_rev=essB,
                        mean_dlogp_seq_B=sum(x[1] for x in fb) / len(fb),
                        s_norm=(sum(nc) / len(nc) - sum(nw) / len(nw)) if nc and nw else None))
    rec = dict(version='trusted-sampler-combined-v1', id=a0.id, host=socket.gethostname(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
               before=b['checkpoint_id'], after=c['checkpoint_id'], step=c['optimizer_step'], K=a.K, m=a.m, rng_seed=d['rng_seed'],
               seed_scheme=sampler.SEED_SCHEME, started_at=t0, timings_s={k: round(v, 1) for k, v in T.items()},
               wall_s=round(time.time() - t0, 1), tasks=per)
    Path(HERE / 'results-combined').mkdir(exist_ok=True)
    with open(HERE / 'results-combined' / f'{a0.id}.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print('DONE', a0.id, len(per), json.dumps(rec['timings_s']), flush=True)
    eng.close()


if __name__ == '__main__':
    main()
