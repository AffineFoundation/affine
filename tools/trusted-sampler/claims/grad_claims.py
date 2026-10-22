#!/usr/bin/env python3
"""Phase 2 (torch, one GPU shard): first-order per-batch scores.

Subspace theta_s = MLP weights (gate/up/down) of the last NL decoder layers (default 4 of 32, ~604M params);
all other weights frozen, so no autograd graph is built below layer 28.
Reference objective per chunk k (4 informative tasks of one subject or of the pool, one split half):
    J_k = mean_tasks [ mean_correct logp(y)/|y|  -  mean_wrong logp(y)/|y| ]          (full softmax, base model)
    G_k = dJ_k/dtheta_s                                                              (saved, bf16)
Per batch rollout i:  d_i[k] = < d logp(y_i)/dtheta_s , G_k >                       (saved)
A GRPO step on batch b with SGD and lr eta changes J_k by  eta/(8192 N) * sum_i A_i d_i[k]   (first order),
A_i = r_i - mean_group(r), seq-mean-token-sum-norm with loss_scale_factor 8192 as in the live trainer
(KL term, clipping and Adam preconditioning omitted; at ratio 1 clipping is inactive)."""
import argparse, glob, json, math, os, random, sys, time, hashlib
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
SHM = Path('/dev/shm/claims')

def H(*a): return int(hashlib.sha256(':'.join(map(str, a)).encode()).hexdigest()[:8], 16)

def load_gen(E):
    d = HERE / 'data' / f'e{E}'; meta, toks = [], []
    for s in range(8):
        m = json.load(open(d / f'gen-{s}.json')); z = np.load(d / f'gen-{s}.npz')
        o = z['offsets']; t = z['tokens']
        for k, x in enumerate(m['meta']):
            x = dict(x); x['tokens'] = t[o[k]:o[k + 1]]; meta.append(x)
    return m['base'], meta

def build_chunks(meta, E, per=4, nch=6):
    by = {}
    for x in meta:
        if x['kind'] == 'ref': by.setdefault(x['index'], []).append(x)
    info = {}
    for i, xs in by.items():
        nc = sum(x['outcome'] == 'positive' for x in xs); nw = sum(x['outcome'] in ('negative', 'neutral') for x in xs)
        if 2 <= nc <= 6 and nw >= 1: info[i] = xs
    subj = {}
    for i in sorted(info):
        subj.setdefault(info[i][0]['subject'], []).append(i)
    rng = random.Random(H('chunks', E)); chunks = []; halves = {0: [], 1: []}
    for s in sorted(subj):
        ts = subj[s][:]; rng.shuffle(ts); hs = [ts[0::2], ts[1::2]]
        for h in (0, 1):
            halves[h] += hs[h]
            for c in range(min(nch, len(hs[h]) // per)):
                chunks.append(dict(kind='subject', subject=s, half=h, c=c, tasks=hs[h][c * per:(c + 1) * per]))
    for h in (0, 1):
        ts = halves[h][:]; rng.shuffle(ts)
        for c in range(nch):
            chunks.append(dict(kind='pool', subject=None, half=h, c=c, tasks=ts[c * per:(c + 1) * per]))
    return chunks, info, {s: len(v) for s, v in subj.items()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epoch', required=True); ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, default=8); ap.add_argument('--layers', type=int, default=4); a = ap.parse_args()
    E = a.epoch; T = json.load(open(ROOT / 'data/original-math7496.tasks.json'))
    base, meta = load_gen(E); path = glob.glob(f'/root/ckpt-cache/{base}')[0]
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(path)
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model.eval(); model.requires_grad_(False)
    lm = model.model.language_model; L = len(lm.layers)
    params = [p for li in range(L - a.layers, L) for n, p in lm.layers[li].mlp.named_parameters()]
    for p in params: p.requires_grad_(True)
    P = sum(p.numel() for p in params); Tm = dict(load=time.time() - t0)
    prompts = {}; PAD = tok.pad_token_id if tok.pad_token_id is not None else 248044
    def prompt(i):
        if i not in prompts:
            d = T[i]['data']
            prompts[i] = list(tok.apply_chat_template([dict(role='system', content=d['system_prompt']), dict(role='user', content=d['prompt'])],
                                                      tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False))
        return prompts[i]
    def logp(i, y):
        p = prompt(i); seq = p + list(map(int, y)); n = len(seq); padn = -(-n // 512) * 512
        ids = torch.tensor([seq + [PAD] * (padn - n)], device='cuda')   # right padding: causal, so positions < n are unaffected
        h = model.model(input_ids=ids).last_hidden_state[0, len(p) - 1:n - 1]
        tgt = ids[0, len(p):n]; tot = 0.0
        for s in range(0, h.shape[0], 1024):
            lg = model.lm_head(h[s:s + 1024]).float()
            tot = tot + (lg.gather(1, tgt[s:s + 1024, None])[:, 0] - torch.logsumexp(lg, 1)).sum()
        return tot
    def flat_grad():
        g = torch.cat([p.grad.reshape(-1) for p in params]); 
        for p in params: p.grad = None
        return g
    chunks, info, nsub = build_chunks(meta, E)
    out = SHM / f'e{E}'; out.mkdir(parents=True, exist_ok=True)
    # ---- phase A: reference chunk gradients
    t = time.time()
    for k, ch in enumerate(chunks):
        if k % a.nshards != a.shard or (out / f'ref-{k}.pt').exists(): continue
        for p in params: p.grad = None
        for i in ch['tasks']:
            xs = [x for x in info[i] if x['outcome'] != 'indeterminate' and x['len'] > 0]
            cor = [x for x in xs if x['outcome'] == 'positive']; wr = [x for x in xs if x['outcome'] != 'positive']
            for x in xs:
                c = (1 / len(cor) if x['outcome'] == 'positive' else -1 / len(wr)) / x['len'] / len(ch['tasks'])
                (c * logp(i, x['tokens'])).backward()
        g = flat_grad().to(torch.bfloat16).cpu(); torch.save(g, out / f'ref-{k}.tmp'); os.replace(out / f'ref-{k}.tmp', out / f'ref-{k}.pt')
    Tm['ref_grads'] = time.time() - t
    while len(glob.glob(str(out / 'ref-*.pt'))) < len(chunks): time.sleep(5)
    time.sleep(3)
    V = torch.stack([torch.load(out / f'ref-{k}.pt') for k in range(len(chunks))]).cuda()
    # ---- phase B: per-rollout dot products
    t = time.time(); res = []
    groups = {}
    for n, x in enumerate(meta):
        if x['kind'] == 'batch': groups.setdefault((x['uid'], x['index']), []).append(n)
    keys = sorted(groups, key=str); mine = [k for k in keys if H('g', *k) % a.nshards == a.shard]
    for (u, i) in mine:
        xs = [meta[n] for n in groups[(u, i)]]
        rw = [x['reward'] for x in xs]
        valid = None not in rw and any(x['outcome'] == 'positive' for x in xs) and any(x['outcome'] != 'positive' for x in xs)
        rec = dict(uid=u, index=i, subject=xs[0]['subject'], valid=valid, rewards=rw, lens=[x['len'] for x in xs], dots=None)
        if valid:
            D = []
            for x in xs:
                logp(i, x['tokens']).backward(); g = flat_grad()
                D.append((V @ g.to(torch.bfloat16)).float().cpu().tolist())
            rec['dots'] = D
        res.append(rec)
    Tm['batch_dots'] = time.time() - t
    json.dump(dict(epoch=E, base=base, shard=a.shard, subspace=dict(layers=list(range(L - a.layers, L)), modules='mlp.gate/up/down', params=P),
                   chunks=[{k: v for k, v in c.items()} for c in chunks], informative_by_subject=nsub, timings=Tm, groups=res),
              open(HERE / 'data' / f'e{E}' / f'dots-{a.shard}.json', 'w'))
    print('DONE', E, a.shard, len(mine), {k: round(v, 1) for k, v in Tm.items()}, flush=True)

if __name__ == '__main__':
    main()
