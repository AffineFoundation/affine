#!/usr/bin/env python3
"""Planted type 8 ('beyond-policy teacher'): for reference tasks the base mostly fails (<=1/8 correct in the
reference draw; such tasks are never in the reference chunks), draw S extra samples from the SAME base model
(best-of-(8+S) search) and keep the correct ones. Output claims/data/planted/e<E>/teacher.npz/.json"""
import argparse, hashlib, json, os, sys, time, types
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')
def H(*a): return int(hashlib.sha256(':'.join(map(str, a)).encode()).hexdigest()[:8], 16) & 0x7fffffff
def main():
    import sampler, fetch, grader, grad_real
    ap = argparse.ArgumentParser(); ap.add_argument('--epoch', required=True); ap.add_argument('--samples', type=int, default=40)
    ap.add_argument('--max-tasks', type=int, default=160); a0 = ap.parse_args(); E = a0.epoch
    base, meta = grad_real.load_ref(E)
    idx = json.load(open(ROOT / 'data/index.json')); T = json.load(open(ROOT / 'data/original-math7496.tasks.json'))
    cp = sampler.find(idx, base[:8]); rt = idx['checkpoints'][-1]['mimo_runtime']; vocab = idx['checkpoints'][-1]['vocab_size']
    by = {}
    for x in meta:
        if x['kind'] == 'ref': by.setdefault(x['index'], []).append(x)
    hard = sorted(i for i, xs in by.items() if sum(x['outcome'] == 'positive' for x in xs) <= 1)
    hard = sorted(hard, key=lambda i: H('pick', E, i))[:a0.max_tasks]
    a = types.SimpleNamespace(tp=1, gpu_mem=0.9, max_num_seqs=1024, max_batched_tokens=16384, enforce_eager=False, engine='swap', max_tokens=8192)
    eng = sampler.Engine(a); gp = grader.pool(32); path, _ = fetch.ensure(cp, '/root/ckpt-cache'); eng.load(cp, path); tok = eng.tokenizer
    special = set(getattr(tok, 'all_special_ids', ()))
    def prompt(i):
        d = T[i]['data']
        return list(tok.apply_chat_template([dict(role='system', content=d['system_prompt']), dict(role='user', content=d['prompt'])], tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False))
    t = time.time(); P = [prompt(i) for i in hard]; S = [[H('teach', E, i, j) for j in range(a0.samples)] for i in hard]
    outs = eng.generate(P, S, rt)
    flat = [(n, j, o) for n, ss in enumerate(outs) for j, o in enumerate(ss)]
    gts = [grader.project(tok, o, special, vocab)[0] for _, _, o in flat]
    kinds = list(gp.map(grader.classify, gts, [T[hard[n]]['data']['answer'] for n, _, _ in flat], chunksize=8))
    toks, offs, rec = [], [0], []
    for (n, j, o), (k, c) in zip(flat, kinds):
        if k == 'positive':
            toks.append(np.asarray(o, dtype=np.int32)); offs.append(offs[-1] + len(o)); rec.append(dict(index=hard[n], j=j, len=len(o)))
    out = HERE / 'data' / 'planted' / f'e{E}'; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'teacher.npz', tokens=np.concatenate(toks) if toks else np.zeros(0, np.int32), offsets=np.asarray(offs, dtype=np.int64))
    json.dump(dict(epoch=E, base=base, tasks=len(hard), samples=a0.samples, wall=round(time.time() - t, 1), correct=rec,
                   tasks_with_correct=len({r['index'] for r in rec})), open(out / 'teacher.json', 'w'))
    print('DONE teacher', E, len(hard), 'tasks', len(rec), 'correct on', len({r['index'] for r in rec}), 'tasks', round(time.time() - t), 's', flush=True)
if __name__ == '__main__':
    main()
