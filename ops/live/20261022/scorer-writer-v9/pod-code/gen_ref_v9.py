#!/usr/bin/env python3
"""SN120 v9 reference generation, one GPU shard (vLLM).  Derived from the v7c scorer's gen_ref.py (ref-only path).
The ONLY randomness is the public post-commit seed in data/e<E>/seed.json (rule sn120-v9-seed-rule-v1):
  draw   : 200 tasks per subject from the full offered pool, random.Random(HS(seed,'draw'))
  samples: 8 per task, sampler seed HS(seed,'sample',task,j)   (network harness: T=1, top-p .95, top-k 20, thinking, 8192)
No H('ref',E), no private reference, no salt.  Output data/e<E>/ref/gen-<shard>.{npz,json}."""
import argparse, json, os, sys, time, types
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = Path('/root/trusted-sampler'); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')
from v9common import HS, load_seed, draw_reference

def main():
    import sampler, fetch, grader
    ap = argparse.ArgumentParser(); ap.add_argument('--epoch', required=True); ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, default=1); ap.add_argument('--ref-per-subject', type=int, default=200); ap.add_argument('--base', required=True)
    a0 = ap.parse_args()
    idx = json.load(open(ROOT / 'data/index.json')); T = json.load(open(ROOT / 'data/original-math7496.tasks.json'))
    cp = sampler.find(idx, a0.base[:8]); rt = idx['checkpoints'][-1]['mimo_runtime']; vocab = idx['checkpoints'][-1]['vocab_size']
    SEED = load_seed(a0.epoch)['seed']
    ref = draw_reference(SEED, idx['pool']['indices'], T, a0.ref_per_subject)
    mine = [i for k, i in enumerate(ref) if k % a0.nshards == a0.shard]
    a = types.SimpleNamespace(tp=1, gpu_mem=0.9, max_num_seqs=1024, max_batched_tokens=16384, enforce_eager=False, engine='swap', max_tokens=8192)
    eng = sampler.Engine(a); gp = grader.pool(16)
    path, _ = fetch.ensure(cp, '/root/ckpt-cache'); t0 = time.time(); eng.load(cp, path); tok = eng.tokenizer
    special = set(getattr(tok, 'all_special_ids', ()))
    def prompt(i):
        d = T[i]['data']
        return list(tok.apply_chat_template([dict(role='system', content=d['system_prompt']), dict(role='user', content=d['prompt'])],
                                            tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False))
    t = time.time()
    outs = eng.generate([prompt(i) for i in mine], [[HS(SEED, 'sample', i, j) for j in range(8)] for i in mine], rt)
    flat = [(n, j, o) for n, ss in enumerate(outs) for j, o in enumerate(ss)]
    gts = [grader.project(tok, o, special, vocab)[0] for _, _, o in flat]
    kinds = list(gp.map(grader.classify, gts, [T[mine[n]]['data']['answer'] for n, _, _ in flat], chunksize=8))
    gen_s = round(time.time() - t, 1)
    toks, offs, meta = [], [0], []
    for (n, j, o), (k, c) in zip(flat, kinds):
        i = mine[n]; toks.append(np.asarray(o, dtype=np.int32)); offs.append(offs[-1] + len(o))
        meta.append(dict(kind='ref', uid=None, index=i, subject=T[i]['data']['subject'], attempt=0, j=j, outcome=k, correct=float(c or 0.0),
                         reward=grader.overlong_reward(float(c or 0.0), len(o)) if k != 'indeterminate' else None, len=len(o)))
    out = HERE / 'data' / f'e{a0.epoch}' / 'ref'; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f'gen-{a0.shard}.npz', tokens=np.concatenate(toks), offsets=np.asarray(offs, dtype=np.int64))
    json.dump(dict(epoch=a0.epoch, base=a0.base, seed=SEED, shard=a0.shard, ref_tasks_total=len(ref), timings=dict(gen=gen_s),
                   wall=round(time.time() - t0, 1), meta=meta), open(out / f'gen-{a0.shard}.json', 'w'))
    print('DONE', a0.epoch, a0.shard, len(mine), gen_s, flush=True); eng.close()

if __name__ == '__main__':
    main()
