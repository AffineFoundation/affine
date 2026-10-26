#!/usr/bin/env python3
"""Phase 1 (vLLM): regenerate miner batches + build the fixed reference set, for one epoch, one GPU shard.

Batch rollouts: real miner rollout tokens are not available (presigned URLs expired; R2 RW key forbidden), so each
admitted group (miner, task index) is re-sampled from the epoch's base checkpoint with the network harness
(T=1, top-p 0.95, top-k 20, thinking, 8192), 8 rollouts, with the group contract (>=1 correct, >=1 non-correct)
enforced by up to 2 re-draws, exactly as an honest miner builds groups. Seeds are per (uid, index, attempt, j).
Reference: R tasks stratified over the 7 MATH subjects, 8 samples each, seed = hash(ref_seed, task, j).
Output: claims/data/e<E>/gen-<shard>.npz (tokens + offsets) and .json (metadata)."""
import argparse, hashlib, json, os, random, sys, time, types
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')

def H(*a): return int(hashlib.sha256(':'.join(map(str, a)).encode()).hexdigest()[:8], 16) & 0x7fffffff

def main():
    import sampler, fetch, grader
    ap = argparse.ArgumentParser(); ap.add_argument('--epoch', required=True); ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, default=8); ap.add_argument('--ref-per-subject', type=int, default=200)
    ap.add_argument('--min-groups', type=int, default=5); a0 = ap.parse_args()
    B = json.load(open(ROOT / 'data/batches.json'))[a0.epoch]
    idx = json.load(open(ROOT / 'data/index.json')); T = json.load(open(ROOT / 'data/original-math7496.tasks.json'))
    cp = sampler.find(idx, B['base'][:8]); rt = idx['checkpoints'][-1]['mimo_runtime']; vocab = idx['checkpoints'][-1]['vocab_size']
    # batches = one per uid (each uid has exactly one commitment per epoch)
    by = {}
    for g in B['groups']:
        by.setdefault(g['uid'], []).append(g)
    groups = [(u, g['index']) for u, gs in sorted(by.items(), key=lambda kv: str(kv[0])) if len(gs) >= a0.min_groups for g in gs]
    # reference draw (same on all shards): stratified by subject
    rng = random.Random(H('ref', a0.epoch)); pool = idx['pool']['indices']
    subj = {}
    for i in pool:
        subj.setdefault(T[i]['data']['subject'], []).append(i)
    ref = sorted(t for s in sorted(subj) for t in rng.sample(subj[s], min(a0.ref_per_subject, len(subj[s]))))
    items = [('batch', u, i) for u, i in groups] + [('ref', None, i) for i in ref]
    mine = [it for k, it in enumerate(items) if k % a0.nshards == a0.shard]
    a = types.SimpleNamespace(tp=1, gpu_mem=0.9, max_num_seqs=1024, max_batched_tokens=16384, enforce_eager=False,
                              engine='swap', max_tokens=8192)
    eng = sampler.Engine(a); gp = grader.pool(16)
    path, _ = fetch.ensure(cp, '/root/ckpt-cache'); t0 = time.time(); eng.load(cp, path); tok = eng.tokenizer
    special = set(getattr(tok, 'all_special_ids', ()))
    def prompt(i):
        d = T[i]['data']
        return list(tok.apply_chat_template([dict(role='system', content=d['system_prompt']), dict(role='user', content=d['prompt'])],
                                            tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False))
    def run(its, attempt):
        P = [prompt(i) for _, _, i in its]
        S = [[H('ref', a0.epoch, i, j) if k == 'ref' else H('batch', a0.epoch, u, i, attempt, j) for j in range(8)] for k, u, i in its]
        outs = eng.generate(P, S, rt)
        flat = [(n, j, o) for n, ss in enumerate(outs) for j, o in enumerate(ss)]
        gts = [grader.project(tok, o, special, vocab)[0] for _, _, o in flat]
        kinds = list(gp.map(grader.classify, gts, [T[its[n][2]]['data']['answer'] for n, _, _ in flat], chunksize=8))
        res = [[None] * 8 for _ in its]
        for (n, j, o), (k, c) in zip(flat, kinds):
            res[n][j] = dict(tokens=o, kind=k, correct=float(c or 0.0), reward=grader.overlong_reward(float(c or 0.0), len(o)) if k != 'indeterminate' else None)
        return res
    final = {}; todo = list(range(len(mine))); Tm = {}
    for attempt in range(3):
        if not todo: break
        t = time.time(); res = run([mine[k] for k in todo], attempt); Tm[f'gen_attempt{attempt}'] = round(time.time() - t, 1)
        nxt = []
        for k, r in zip(todo, res):
            final[k] = (attempt, r)
            if mine[k][0] == 'batch':
                nc = sum(x['kind'] == 'positive' for x in r); nn = sum(x['kind'] in ('negative', 'neutral') for x in r)
                if (nc == 0 or nn == 0) and attempt < 2: nxt.append(k)
        todo = nxt
    toks, offs, meta = [], [0], []
    for k, (kind, u, i) in enumerate(mine):
        att, r = final[k]
        for j, x in enumerate(r):
            toks.append(np.asarray(x['tokens'], dtype=np.int32)); offs.append(offs[-1] + len(x['tokens']))
            meta.append(dict(kind=kind, uid=u, index=i, subject=T[i]['data']['subject'], attempt=att, j=j,
                             outcome=x['kind'], correct=x['correct'], reward=x['reward'], len=len(x['tokens'])))
    out = HERE / 'data' / f'e{a0.epoch}'; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f'gen-{a0.shard}.npz', tokens=np.concatenate(toks), offsets=np.asarray(offs, dtype=np.int64))
    json.dump(dict(epoch=a0.epoch, base=B['base'], shard=a0.shard, timings=Tm, wall=round(time.time() - t0, 1), meta=meta),
              open(out / f'gen-{a0.shard}.json', 'w'))
    print('DONE', a0.epoch, a0.shard, len(mine), Tm, flush=True); eng.close()

if __name__ == '__main__':
    main()
