#!/usr/bin/env python3
"""Pass-band filter test (GPU, one epoch). Current filter keeps reference tasks with 2..6 of 8 correct. 'No band' = also use 1/8 and 7/8 tasks
(0/8 and 8/8 carry no correct-vs-wrong term). Builds EXTRA chunks (4 tasks, per subject and half, seeded) from the 1/8 and 7/8 tasks of the same
reference draw, computes their J-gradients G_k (same code path as grad_claims), and dots for a random subset of honest real groups and the
bo16 beyond-policy groups against the extra chunks. Output data/adv/e<E>/band.json."""
import argparse, glob, gzip, json, random, sys, time, collections
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import grad_claims as gc, grad_real, grader
from plant_dots import key
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epoch', required=True); ap.add_argument('--groups', type=int, default=600); a = ap.parse_args(); E = a.epoch
    T = json.load(open(gc.ROOT / 'data/original-math7496.tasks.json')); base, meta = grad_real.load_ref(E)
    by = collections.defaultdict(list)
    for x in meta:
        if x['kind'] == 'ref': by[x['index']].append(x)
    ext = {}
    for i, xs in by.items():
        nc = sum(x['outcome'] == 'positive' for x in xs); nw = sum(x['outcome'] in ('negative', 'neutral') for x in xs)
        if nc in (1, 7) and nw >= 1: ext[i] = xs
    subj = collections.defaultdict(list)
    for i in sorted(ext): subj[ext[i][0]['subject']].append(i)
    rng = random.Random(gc.H('chunks-band', E)); chunks = []
    for s in sorted(subj):
        ts = subj[s][:]; rng.shuffle(ts)
        for c in range(len(ts) // 4): chunks.append(dict(kind='subject', subject=s, half=c % 2, c=c, tasks=ts[c * 4:(c + 1) * 4], band=[sum(x['outcome'] == 'positive' for x in ext[t]) for t in ts[c * 4:(c + 1) * 4]]))
    print('extra band tasks', len(ext), 'chunks', len(chunks), {s: len(v) for s, v in subj.items()}, flush=True)
    path = glob.glob(f'/root/ckpt-cache/{base}')[0]
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(path); PAD = tok.pad_token_id if tok.pad_token_id is not None else 248044
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa').cuda(); model.eval(); model.requires_grad_(False)
    lm = model.model.language_model; L = len(lm.layers); params = [p for li in range(L - 4, L) for n, p in lm.layers[li].mlp.named_parameters()]
    for p in params: p.requires_grad_(True)
    prompts = {}
    def prompt(i):
        if i not in prompts:
            d = T[i]['data']; prompts[i] = list(tok.apply_chat_template([dict(role='system', content=d['system_prompt']), dict(role='user', content=d['prompt'])], tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False))
        return prompts[i]
    def logp(i, y, p=None):
        p = p if p is not None else prompt(i); seq = p + list(map(int, y)); n = len(seq); padn = -(-n // 512) * 512
        ids = torch.tensor([seq + [PAD] * (padn - n)], device='cuda'); h = model.model(input_ids=ids).last_hidden_state[0, len(p) - 1:n - 1]; tgt = ids[0, len(p):n]; tot = 0.0
        for s in range(0, h.shape[0], 1024):
            lg = model.lm_head(h[s:s + 1024]).float(); tot = tot + (lg.gather(1, tgt[s:s + 1024, None])[:, 0] - torch.logsumexp(lg, 1)).sum()
        return tot
    def fg():
        g = torch.cat([p.grad.reshape(-1) for p in params]); [setattr(p, 'grad', None) for p in params]; return g
    t0 = time.time(); NP = sum(p.numel() for p in params); V = torch.empty((len(chunks), NP), dtype=torch.bfloat16, device='cuda')
    for kc, ch in enumerate(chunks):
        for p in params: p.grad = None
        for i in ch['tasks']:
            xs = [x for x in ext[i] if x['outcome'] != 'indeterminate' and x['len'] > 0]; cor = [x for x in xs if x['outcome'] == 'positive']; wr = [x for x in xs if x['outcome'] != 'positive']
            for x in xs: ((1 / len(cor) if x['outcome'] == 'positive' else -1 / len(wr)) / x['len'] / len(ch['tasks']) * logp(i, x['tokens'])).backward()
        V[kc] = fg().to(torch.bfloat16)
    tG = time.time() - t0; print('G done', len(chunks), round(tG), flush=True)
    # items: random honest real groups (seeded) + bo16 beyond-policy groups
    D = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-real-*.json')))]
    R = {(r['miner'], r['slot']): r for r in (json.loads(l) for l in gzip.open(HERE / 'data/real' / f'real-{E}.jsonl.gz', 'rt'))}
    allg = [(g['miner'], g['slot']) for d in D for g in d['groups'] if g.get('dots')]
    pick = sorted(allg, key=lambda k: gc.H('band-pick', E, *k))[:a.groups]
    items = []
    for mk in pick:
        r = R[mk]
        for j, x in enumerate(r['rollouts']): items.append(('real', [mk[0], mk[1], j], r['index'], x['out'], list(r['prompt'])))
    P = HERE / 'data' / 'planted' / f'e{E}'; teach = json.load(open(P / 'teacher.json')); tz = np.load(P / 'teacher.npz'); to = tz['offsets']; tt = tz['tokens']
    cor = collections.defaultdict(list)
    for k, c in enumerate(teach['correct']):
        if c['j'] < 16: cor[c['index']].append((c['j'], k))
    for i, cs in sorted(cor.items()):
        for j, k in sorted(cs)[:2]: items.append(('bp', [i, 'c', k], i, tt[to[k]:to[k + 1]], None))
        for j, x in enumerate([x for x in by[i] if x['outcome'] != 'positive'][:6]): items.append(('bp', [i, 'w', j], i, x['tokens'], None))
    out = {}; t1 = time.time()
    for n, (src, kk, i, y, pr) in enumerate(items):
        logp(i, y, pr).backward(); out[key(i, y)] = (V @ fg().to(torch.bfloat16)).float().cpu().tolist()
        if n % 1000 == 999: print(n + 1, round(time.time() - t1), flush=True)
    json.dump(dict(epoch=E, chunks=chunks, n_extra_tasks=len(ext), groups=pick, dots=out, t_G=tG, t_dots=time.time() - t1), open(HERE / 'data' / 'adv' / f'e{E}' / 'band.json', 'w'))
    print('DONE band', E, len(items), round(time.time() - t0), flush=True)
if __name__ == '__main__':
    main()
