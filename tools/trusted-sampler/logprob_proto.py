#!/usr/bin/env python3
"""Prototype: log-prob scoring of a step instead of re-generating with the after checkpoint.

For a pair (before, after): draw K tasks (own CSPRNG), generate m samples with BEFORE, grade them,
then compute the length-normalized log-prob of each completion under BEFORE and under AFTER with a
prefill-only forward pass (vLLM prompt_logprobs). Per task with >=1 correct and >=1 incorrect sample:
    score_t = mean_correct(dlogp) - mean_incorrect(dlogp),   dlogp = (logp_after - logp_before)/len
(incorrect = completed-wrong + unresolved). Reports mean score, SE, z, and per-phase cost.
Writes one JSON line to results-logprob/<id>.jsonl (separate from the generation results).
"""
import argparse, json, math, os, random, secrets, socket, sys, time, types
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')


def main():
    import sampler, fetch, grader
    ap = argparse.ArgumentParser(); ap.add_argument('--pair', required=True); ap.add_argument('-K', type=int, default=128)
    ap.add_argument('-m', type=int, default=8); ap.add_argument('--id', default=None); a0 = ap.parse_args()
    idx = json.load(open(HERE / 'data/index.json')); tasks = json.load(open(HERE / 'data/original-math7496.tasks.json'))
    b, c = (sampler.find(idx, x) for x in a0.pair.split(':'))
    a = types.SimpleNamespace(tp=1, gpu_mem=0.9, max_num_seqs=1024, max_batched_tokens=16384, enforce_eager=False,
                              engine='swap', max_tokens=8192, K=a0.K, m=a0.m, crn=True)
    rt = idx['checkpoints'][-1]['mimo_runtime']; vocab = idx['checkpoints'][-1]['vocab_size']
    T = {}; t0 = time.time()
    d = sampler.new_draw(a, idx)
    eng = sampler.Engine(a); pool = grader.pool(16)
    pb, _ = fetch.ensure(b, '/root/ckpt-cache'); pc, _ = fetch.ensure(c, '/root/ckpt-cache')
    t = time.time(); eng.load(b, pb); T['load_before'] = time.time() - t
    tok = eng.tokenizer
    P = [list(tok.apply_chat_template([dict(role='system', content=tasks[i]['data']['system_prompt']), dict(role='user', content=tasks[i]['data']['prompt'])],
              tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False)) for i in d['chosen']]
    t = time.time(); outs = eng.generate(P, d['seeds']['before'], rt); T['generate_before'] = time.time() - t
    # grade each sample
    special = set(getattr(tok, 'all_special_ids', ()))
    flat = [(ti, si, o) for ti, ss in enumerate(outs) for si, o in enumerate(ss)]
    t = time.time()
    gts = [grader.project(tok, o, special, vocab)[0] for _, _, o in flat]
    kinds = list(pool.map(grader.classify, gts, [tasks[d['chosen'][ti]]['data']['answer'] for ti, _, _ in flat], chunksize=8))
    T['verify_before'] = time.time() - t
    lab = {(ti, si): k for (ti, si, _), (k, _) in zip(flat, kinds)}
    useful = sorted({ti for ti, si, _ in flat if any(lab[(ti, x)] == 'positive' for x in range(a.m)) and any(lab[(ti, x)] in ('negative', 'neutral') for x in range(a.m))})
    useful_s = set(useful)
    seqs = [(ti, si, P[ti], o) for ti, si, o in flat if ti in useful_s and len(P[ti]) + len(o) < 12288 and lab[(ti, si)] != 'indeterminate' and o]
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    def lp(label):
        reqs = [TokensPrompt(prompt_token_ids=p + o) for _, _, p, o in seqs]
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0, detokenize=False)
        t = time.time(); res = eng.llm.generate(reqs, sp, use_tqdm=False); T[f'logprob_{label}'] = time.time() - t
        out = []
        for (ti, si, p, o), r in zip(seqs, res):
            pl = r.prompt_logprobs[len(p):]
            ids = (p + o)[len(p):]
            s = sum(x[t_].logprob for x, t_ in zip(pl, ids))
            out.append(s / len(o))
        return out
    lpb = lp('before')
    t = time.time(); eng.load(c, pc); T['swap_after'] = time.time() - t
    lpa = lp('after')
    per = {}
    for (ti, si, _, _), x, y in zip(seqs, lpb, lpa):
        per.setdefault(ti, {'c': [], 'w': []})['c' if lab[(ti, si)] == 'positive' else 'w'].append(y - x)
    scores = [sum(v['c']) / len(v['c']) - sum(v['w']) / len(v['w']) for v in per.values() if v['c'] and v['w']]
    if len(scores) < 2:
        raise SystemExit(f'too few informative tasks: {len(scores)}')
    n = len(scores); mu = sum(scores) / n; sd = math.sqrt(sum((x - mu) ** 2 for x in scores) / (n - 1))
    allc = [y - x for (ti, si, _, _), x, y in zip(seqs, lpb, lpa) if lab[(ti, si)] == 'positive']
    allw = [y - x for (ti, si, _, _), x, y in zip(seqs, lpb, lpa) if lab[(ti, si)] != 'positive']
    rec = dict(version='trusted-sampler-logprob-proto-v1', id=a0.id or f'{socket.gethostname()}-g{os.environ.get("CUDA_VISIBLE_DEVICES")}',
               before=b['checkpoint_id'], after=c['checkpoint_id'], step=c['optimizer_step'], K=a.K, m=a.m, rng_seed=d['rng_seed'],
               tasks_informative=n, sequences_scored=len(seqs), mean_score=mu, se=sd / math.sqrt(n), z=mu / (sd / math.sqrt(n)),
               mean_dlogp_correct=sum(allc) / len(allc), mean_dlogp_wrong=sum(allw) / len(allw),
               timings_s={k: round(v, 1) for k, v in T.items()}, wall_s=round(time.time() - t0, 1),
               logprob_tokens=sum(len(o) for *_, o in seqs))
    Path(HERE / 'results-logprob').mkdir(exist_ok=True)
    with open(HERE / 'results-logprob' / f"{rec['id']}.jsonl", 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print('RESULT', json.dumps(rec), flush=True)
    eng.close()


if __name__ == '__main__':
    main()
