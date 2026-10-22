#!/usr/bin/env python3
"""Phase 2 on REAL miner rollouts (one GPU shard).

Same subspace / reference chunks / estimator as grad_claims.py; only the batch rollouts change:
the miners' own training documents (prompt tokens + completion tokens + claimed reward), read read-only
from R2 into claims/data/real/real-<E>.jsonl.gz. Phase A (reference gradients G_k) is reused from
/dev/shm/claims/e<E>/ref-k.pt when present (e62, e64) and built otherwise from the ref-only gen files.
Per rollout: d_i[k] = < d logp(y_i | miner prompt) / d theta_s , G_k >.  Rewards are attached later
(claimed vs regraded), the dots do not depend on them.
Output: claims/data/e<E>/dots-real-<shard>.json"""
import argparse, glob, gzip, json, os, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
from grad_claims import build_chunks, H, SHM

def load_ref(E):
    d = HERE / 'data' / f'e{E}'; meta = []
    fs = sorted(glob.glob(str(d / 'gen-[0-9]*.json')))
    for f in fs:
        m = json.load(open(f)); z = np.load(f[:-5] + '.npz'); o = z['offsets']; t = z['tokens']
        for k, x in enumerate(m['meta']):
            x = dict(x); x['tokens'] = t[o[k]:o[k + 1]]; meta.append(x)
    return m['base'], meta

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epoch', required=True); ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, default=4); ap.add_argument('--layers', type=int, default=4); a = ap.parse_args()
    E = a.epoch; T = json.load(open(ROOT / 'data/original-math7496.tasks.json'))
    base, meta = load_ref(E); path = glob.glob(f'/root/ckpt-cache/{base}')[0]
    R = [json.loads(l) for l in gzip.open(HERE / 'data' / 'real' / f'real-{E}.jsonl.gz', 'rt')]
    assert all(r['built_on'] == base[:16] for r in R), 'batch built on a different checkpoint than the reference base'
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(path); t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model.eval(); model.requires_grad_(False)
    lm = model.model.language_model; L = len(lm.layers)
    params = [p for li in range(L - a.layers, L) for n, p in lm.layers[li].mlp.named_parameters()]
    for p in params: p.requires_grad_(True)
    P = sum(p.numel() for p in params); Tm = dict(load=time.time() - t0)
    PAD = tok.pad_token_id if tok.pad_token_id is not None else 248044
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
    chunks, info, nsub = build_chunks(meta, E)
    out = SHM / f'e{E}'; out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    for k, ch in enumerate(chunks):   # phase A only for missing chunks (new epochs)
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
    mine = [r for r in R if H('g', r['uid'], r['miner'], r['slot']) % a.nshards == a.shard]
    for r in mine:
        i = r['index']; D = []
        for x in r['rollouts']:
            logp(r['prompt'], x['out']).backward(); g = flat_grad()
            D.append((V @ g.to(torch.bfloat16)).float().cpu().tolist())
        res.append(dict(uid=r['uid'], miner=r['miner'], slot=r['slot'], index=i, subject=T[i]['data']['subject'], trained=r['trained'],
                        prompt_matches_ours=list(r['prompt']) == our_prompt(i), claimed_cls=[x['cls'] for x in r['rollouts']],
                        claimed_reward=[x['reward'] for x in r['rollouts']], lens=[len(x['out']) for x in r['rollouts']], dots=D))
    Tm['batch_dots'] = time.time() - t
    json.dump(dict(epoch=E, base=base, shard=a.shard, source='real-miner-documents',
                   subspace=dict(layers=list(range(L - a.layers, L)), modules='mlp.gate/up/down', params=P),
                   chunks=chunks, informative_by_subject=nsub, timings=Tm, groups=res),
              open(HERE / 'data' / f'e{E}' / f'dots-real-{a.shard}.json', 'w'))
    print('DONE', E, a.shard, len(mine), {k: round(v, 1) for k, v in Tm.items()}, flush=True)

if __name__ == '__main__':
    main()
