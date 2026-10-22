#!/usr/bin/env python3
"""Adversarial generation (vLLM, one GPU).
--mode copies : near-duplicate copies of beyond-policy (bo16) correct solutions on base(E):
                prefix-f (copy the first f of the solution, let the base continue; f=.25/.5/.75) and hint (base solves the task with the
                copied final write-up shown as a hint; the completion is then submitted under the ORIGINAL prompt).
--mode distill: correct solutions to the same hard tasks from a DIFFERENT checkpoint (--teacher, e.g. a later step) = off-policy teacher.
Output: claims/data/adv/e<E>/<mode>.npz (tokens/offsets) + .json (meta: index, variant, src, len, correct, jac5 vs source)."""
import argparse, hashlib, json, os, sys, time, types
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')
def H(*a): return int(hashlib.sha256(':'.join(map(str, a)).encode()).hexdigest()[:8], 16) & 0x7fffffff
def jac(a, b, n=5):
    A = {tuple(a[i:i + n]) for i in range(max(0, len(a) - n + 1))}; B = {tuple(b[i:i + n]) for i in range(max(0, len(b) - n + 1))}
    return len(A & B) / max(1, len(A | B))
def main():
    import sampler, fetch, grader, grad_real
    ap = argparse.ArgumentParser(); ap.add_argument('--epoch', required=True); ap.add_argument('--mode', required=True); ap.add_argument('--teacher')
    ap.add_argument('--samples', type=int, default=16); a0 = ap.parse_args(); E = a0.epoch
    base, meta = grad_real.load_ref(E)
    idx = json.load(open(ROOT / 'data/index.json')); T = json.load(open(ROOT / 'data/original-math7496.tasks.json'))
    rt = idx['checkpoints'][-1]['mimo_runtime']; vocab = idx['checkpoints'][-1]['vocab_size']
    P_ = HERE / 'data' / 'planted' / f'e{E}'; teach = json.load(open(P_ / 'teacher.json')); tz = np.load(P_ / 'teacher.npz'); to = tz['offsets']; tt = tz['tokens']
    src = {}
    for k, c in enumerate(teach['correct']):
        if c['j'] < 16: src.setdefault(c['index'], []).append((c['j'], k))
    srcs = [(i, k) for i, v in sorted(src.items()) for _, k in sorted(v)[:2]]
    cp = sampler.find(idx, (a0.teacher or base)[:8])
    a = types.SimpleNamespace(tp=1, gpu_mem=0.9, max_num_seqs=1024, max_batched_tokens=16384, enforce_eager=False, engine='swap', max_tokens=8192)
    eng = sampler.Engine(a); gp = grader.pool(32); path, _ = fetch.ensure(cp, '/root/ckpt-cache'); eng.load(cp, path); tok = eng.tokenizer
    special = set(getattr(tok, 'all_special_ids', ()))
    def prompt(i, extra=''):
        d = T[i]['data']
        return list(tok.apply_chat_template([dict(role='system', content=d['system_prompt']), dict(role='user', content=d['prompt'] + extra)], tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False))
    jobs = []   # (index, variant, src_k, prompt_tokens, prefix_tokens, seeds)
    if a0.mode == 'copies':
        for i, k in srcs:
            y = list(map(int, tt[to[k]:to[k + 1]]))
            for f in (0.25, 0.5, 0.75):
                pre = y[:int(len(y) * f)]; jobs.append((i, f'prefix{int(f * 100)}', k, prompt(i) + pre, pre, [H('pre', E, i, k, f, s) for s in range(2)]))
            txt = tok.decode(y, skip_special_tokens=True); txt = txt.split('</think>')[-1].strip()[-3000:]
            jobs.append((i, 'hint', k, prompt(i, '\n\nHint: a correct worked solution written by another solver is shown below. Write your own complete solution.\n\n' + txt), [], [H('hint', E, i, k, s) for s in range(2)]))
    else:
        by = {}
        for x in meta:
            if x['kind'] == 'ref': by.setdefault(x['index'], []).append(x)
        hard = sorted({c['index'] for c in teach['correct']} | set(sorted((i for i, xs in by.items() if sum(x['outcome'] == 'positive' for x in xs) <= 1), key=lambda i: H('pick', E, i))[:teach['tasks']]))
        for i in hard: jobs.append((i, 'distill', -1, prompt(i), [], [H('dist', E, i, s) for s in range(a0.samples)]))
    t = time.time(); outs = eng.generate([j[3] for j in jobs], [j[5] for j in jobs], rt)
    flat = []
    for (i, var, k, _, pre, _), ss in zip(jobs, outs):
        for o in ss: flat.append((i, var, k, (list(pre) + list(o))[:8192]))
    gts = [grader.project(tok, y, special, vocab)[0] for _, _, _, y in flat]
    kinds = list(gp.map(grader.classify, gts, [T[i]['data']['answer'] for i, _, _, _ in flat], chunksize=8))
    toks, offs, rec = [], [0], []
    for (i, var, k, y), (kd, c) in zip(flat, kinds):
        s = list(map(int, tt[to[k]:to[k + 1]])) if k >= 0 else []
        toks.append(np.asarray(y, dtype=np.int32)); offs.append(offs[-1] + len(y))
        rec.append(dict(index=i, variant=var, src=k, len=len(y), outcome=kd, correct=float(c or 0.0), jac5=jac(y, s) if s else None))
    out = HERE / 'data' / 'adv' / f'e{E}'; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f'{a0.mode}.npz', tokens=np.concatenate(toks), offsets=np.asarray(offs, dtype=np.int64))
    json.dump(dict(epoch=E, base=base, gen_ckpt=cp['checkpoint_id'] if isinstance(cp, dict) and 'checkpoint_id' in cp else str(a0.teacher or base), meta=rec, wall=round(time.time() - t)), open(out / f'{a0.mode}.json', 'w'))
    import collections
    print('DONE', a0.mode, E, len(rec), collections.Counter((r['variant'], r['outcome']) for r in rec), round(time.time() - t), flush=True)
    eng.close()
if __name__ == '__main__':
    main()
