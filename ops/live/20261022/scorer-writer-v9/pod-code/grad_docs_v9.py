#!/usr/bin/env python3
"""SN120 v9 scorer phase B (one GPU shard): per-rollout gradient-dot vs the epoch's PUBLIC POST-COMMIT reference
(chunks from v9common.build_chunks(seed); chunk gradients stored on local disk, not /dev/shm).
Same estimator as claims/grad_real.py (layers 28-31 MLP subspace, reference chunk gradients G_k of
J_k = mean_tasks[mean_correct logp/len - mean_wrong logp/len] at the epoch base checkpoint).
d_i[k] = < d logp(y_i | miner prompt)/d theta_s , G_k >.  Input docs: data/e<E>/docs.jsonl.gz; only docs listed in
data/e<E>/todo.json are computed. Output data/e<E>/dots-<shard>.json"""
import argparse, glob, gzip, json, os, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; TS = Path('/root/trusted-sampler'); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(TS))
from v9common import build_chunks, load_seed, HS

def load_ref(E):
    d = HERE / 'data' / f'e{E}' / 'ref'; meta = []; base = None
    for f in sorted(glob.glob(str(d / 'gen-[0-9]*.json'))):
        m = json.load(open(f)); z = np.load(f[:-5] + '.npz'); o = z['offsets']; t = z['tokens']; base = m['base']
        for k, x in enumerate(m['meta']):
            if x['kind'] != 'ref': continue
            x = dict(x); x['tokens'] = t[o[k]:o[k + 1]]; meta.append(x)
    return base, meta

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epoch', required=True); ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, required=True); ap.add_argument('--layers', type=int, default=4); a = ap.parse_args()
    E = a.epoch; T = json.load(open(TS / 'data/original-math7496.tasks.json')); ED = HERE / 'data' / f'e{E}'
    base, meta = load_ref(E); path = glob.glob(f'/root/ckpt-cache/{base}')[0]
    todo = {(m, s) for m, s in json.load(open(ED / 'todo.json'))}
    R = [r for r in (json.loads(l) for l in gzip.open(ED / 'docs.jsonl.gz', 'rt')) if (r['miner'], r['slot']) in todo]
    SEED = load_seed(E)['seed']
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(path); t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model.eval(); model.requires_grad_(False)
    lm = model.model.language_model; L = len(lm.layers)
    params = [p for li in range(L - a.layers, L) for n, p in lm.layers[li].mlp.named_parameters()]
    for p in params: p.requires_grad_(True)
    Tm = dict(load=time.time() - t0); PAD = tok.pad_token_id if tok.pad_token_id is not None else 248044
    prompts = {}
    def our_prompt(i):
        if i not in prompts:
            d = T[i]['data']
            prompts[i] = list(tok.apply_chat_template([dict(role='system', content=d['system_prompt']), dict(role='user', content=d['prompt'])],
                                                      tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False))
        return prompts[i]
    def logp(p, y):
        seq = list(p) + list(map(int, y)); n = len(seq); padn = -(-n // 512) * 512
        ids = torch.tensor([seq + [PAD] * (padn - n)], device='cuda')
        h = model.model(input_ids=ids).last_hidden_state[0, len(p) - 1:n - 1]
        tgt = ids[0, len(p):n]; tot = 0.0
        for s in range(0, h.shape[0], 1024):
            lg = model.lm_head(h[s:s + 1024]).float()
            tot = tot + (lg.gather(1, tgt[s:s + 1024, None])[:, 0] - torch.logsumexp(lg, 1)).sum()
        return tot
    def flat_grad():
        g = torch.cat([p.grad.reshape(-1) for p in params])
        for p in params: p.grad = None
        return g
    chunks, info, nsub = build_chunks(meta, SEED)
    out = ED / 'refgrad'; out.mkdir(parents=True, exist_ok=True); t = time.time()
    for k, ch in enumerate(chunks):
        if k % a.nshards != a.shard or (out / f'ref-{k}.pt').exists(): continue
        for p in params: p.grad = None
        for i in ch['tasks']:
            xs = [x for x in info[i] if x['outcome'] != 'indeterminate' and x['len'] > 0]
            cor = [x for x in xs if x['outcome'] == 'positive']; wr = [x for x in xs if x['outcome'] != 'positive']
            for x in xs:
                c = (1 / len(cor) if x['outcome'] == 'positive' else -1 / len(wr)) / x['len'] / len(ch['tasks'])
                (c * logp(our_prompt(i), x['tokens'])).backward()
        g = flat_grad().to(torch.bfloat16).cpu(); torch.save(g, out / f'ref-{k}.tmp'); os.replace(out / f'ref-{k}.tmp', out / f'ref-{k}.pt')
    Tm['ref_grads'] = time.time() - t
    while len(glob.glob(str(out / 'ref-*.pt'))) < len(chunks): time.sleep(5)
    time.sleep(3)
    V = torch.stack([torch.load(out / f'ref-{k}.pt') for k in range(len(chunks))]).cuda()
    t = time.time(); res = []
    mine = [r for r in R if HS(SEED, 'g', r['miner'], r['slot']) % a.nshards == a.shard]
    for r in mine:
        D = []
        for x in r['rollouts']:
            if len(x['out']) == 0: D.append([0.0] * len(chunks)); continue
            logp(r['prompt'], x['out']).backward(); g = flat_grad()
            D.append((V @ g.to(torch.bfloat16)).float().cpu().tolist())
        res.append(dict(miner=r['miner'], slot=r['slot'], lens=[len(x['out']) for x in r['rollouts']], dots=D))
    Tm['batch_dots'] = time.time() - t
    tmp = ED / f'dots-{a.shard}.json.tmp'
    json.dump(dict(epoch=E, base=base, seed=SEED, shard=a.shard, chunks=chunks, timings=Tm, docs=res), open(tmp, 'w')); os.replace(tmp, ED / f'dots-{a.shard}.json')
    print('DONE', E, a.shard, len(mine), {k: round(v, 1) for k, v in Tm.items()}, flush=True)

if __name__ == '__main__':
    main()
