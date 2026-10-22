#!/usr/bin/env python3
"""Gradient-dots for arbitrary sequence sets: data/adv/e<E>/<name>.npz/.json (meta[i]['index'] = task), against base(E) and V(E),
identical computation to plant_dots.py. Cache: data/adv/e<E>/dots-<name>-<shard>.json keyed by plant_dots.key(index, tokens)."""
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
from grad_claims import SHM
import grad_real
from plant_dots import key
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epoch', required=True); ap.add_argument('--name', required=True)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1); a = ap.parse_args(); E = a.epoch
    D = HERE / 'data' / 'adv' / f'e{E}'; T = json.load(open(ROOT / 'data/original-math7496.tasks.json'))
    m = json.load(open(D / f'{a.name}.json')); z = np.load(D / f'{a.name}.npz'); o = z['offsets']; tk = z['tokens']
    done = {}
    for f in glob.glob(str(D / f'dots-{a.name}-*.json')): done.update(json.load(open(f)))
    todo = {}
    for n, x in enumerate(m['meta']):
        t = tk[o[n]:o[n + 1]]; k = key(x['index'], t)
        if k not in done and len(t) > 0 and int(k, 16) % a.nshards == a.shard: todo[k] = (k, x['index'], t)
    todo = list(todo.values()); print('todo', len(todo), flush=True)
    if not todo: return
    base, _ = grad_real.load_ref(E); path = glob.glob(f"/root/ckpt-cache/{base}")[0]
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa').cuda(); model.eval(); model.requires_grad_(False)
    lm = model.model.language_model; L = len(lm.layers)
    params = [p for li in range(L - 4, L) for n, p in lm.layers[li].mlp.named_parameters()]
    for p in params: p.requires_grad_(True)
    PAD = tok.pad_token_id if tok.pad_token_id is not None else 248044
    nch = len(glob.glob(str(SHM / f'e{E}' / 'ref-*.pt'))); V = torch.stack([torch.load(SHM / f'e{E}' / f'ref-{k}.pt') for k in range(nch)]).cuda()
    prompts = {}
    def prompt(i):
        if i not in prompts:
            d = T[i]['data']; prompts[i] = list(tok.apply_chat_template([dict(role='system', content=d['system_prompt']), dict(role='user', content=d['prompt'])], tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False))
        return prompts[i]
    out = {}; t0 = time.time(); fn = D / f'dots-{a.name}-{a.shard}-{os.getpid()}.json'
    for n, (k, i, y) in enumerate(todo):
        p = prompt(i); seq = p + list(map(int, y)); L_ = len(seq); padn = -(-L_ // 512) * 512
        ids = torch.tensor([seq + [PAD] * (padn - L_)], device='cuda')
        h = model.model(input_ids=ids).last_hidden_state[0, len(p) - 1:L_ - 1]; tgt = ids[0, len(p):L_]; tot = 0.0
        for s in range(0, h.shape[0], 1024):
            lg = model.lm_head(h[s:s + 1024]).float(); tot = tot + (lg.gather(1, tgt[s:s + 1024, None])[:, 0] - torch.logsumexp(lg, 1)).sum()
        tot.backward(); g = torch.cat([q.grad.reshape(-1) for q in params])
        for q in params: q.grad = None
        out[k] = (V @ g.to(torch.bfloat16)).float().cpu().tolist()
        if n % 200 == 199: json.dump(out, open(fn, 'w')); print(n + 1, round(time.time() - t0), flush=True)
    json.dump(out, open(fn, 'w')); print('DONE', E, a.name, a.shard, len(todo), round(time.time() - t0), flush=True)
if __name__ == '__main__':
    main()
