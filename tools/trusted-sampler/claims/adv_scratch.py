#!/usr/bin/env python3
"""Goodhart check with REAL scratch steps (subspace = MLP layers 28-31 in fp32, throwaway copy, restored per batch).
Spec: data/adv/e<E>/scratch-spec.json  batches[{id,type,groups[{index,rollouts[{src,key,reward}]}]}]
  src: real=[miner,slot,j] | new=planted new.npz idx | adv=[name, idx] (data/adv/e<E>/<name>.npz) | ref=[index, j] base reference sample.
Per batch: K steps; each step g = sum_i A_i dlogp(y_i)/(8192N) at the CURRENT params, update = U * g/max|g| (sgd) or U * sign(g) (sign,
  = Adam's first step / Adam with a flat preconditioner). Measure J_k (mean_tasks[correct logp/len - wrong logp/len]) on half-0 chunks
  (the scoring reference) and half-1 chunks (held-out tasks) after each step. First-order prediction eta*<g, G_k> recorded for step 1.
Forwards batched by length for speed."""
import argparse, glob, gzip, json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import grad_claims as gc, grad_real as gr
sys.path.insert(0, str(gc.ROOT))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epoch', required=True); ap.add_argument('--ids', default=''); ap.add_argument('--tag', default='')
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--steps', type=int, default=3); ap.add_argument('--U', type=float, default=1e-3); ap.add_argument('--modes', default='sgd'); ap.add_argument('--tasks-per-chunk', type=int, default=2); ap.add_argument('--batched', type=int, default=0)
    a = ap.parse_args(); E = a.epoch; T = json.load(open(gc.ROOT / 'data/original-math7496.tasks.json'))
    D = HERE / 'data' / 'adv' / f'e{E}'; spec = json.load(open(D / 'scratch-spec.json'))
    bs = [b for n, b in enumerate(spec['batches']) if (not a.ids or b['id'] in a.ids.split(',')) and n % a.nshards == a.shard]
    base, meta = gr.load_ref(E); path = glob.glob(f'/root/ckpt-cache/{base}')[0]
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(path); PAD = tok.pad_token_id if tok.pad_token_id is not None else 248044
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model.eval(); model.requires_grad_(False); lm = model.model.language_model; L = len(lm.layers)
    for li in range(L - 4, L):
        mlp = lm.layers[li].mlp; mlp.float()
        mlp.register_forward_pre_hook(lambda m, args: tuple(x.float() if torch.is_tensor(x) else x for x in args))
        mlp.register_forward_hook(lambda m, i, o: o.to(torch.bfloat16))
    params = [p for li in range(L - 4, L) for n, p in lm.layers[li].mlp.named_parameters()]
    for p in params: p.requires_grad_(True)
    orig = [p.detach().clone() for p in params]
    prompts = {}
    def prompt(i):
        if i not in prompts:
            d = T[i]['data']; prompts[i] = list(tok.apply_chat_template([dict(role='system', content=d['system_prompt']), dict(role='user', content=d['prompt'])], tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False))
        return prompts[i]
    def logp1(i, y):
        p = prompt(i); seq = p + list(map(int, y)); n = len(seq); padn = -(-n // 512) * 512
        ids = torch.tensor([seq + [PAD] * (padn - n)], device='cuda')
        h = model.model(input_ids=ids).last_hidden_state[0, len(p) - 1:n - 1]; tgt = ids[0, len(p):n]; tot = 0.0
        for s in range(0, h.shape[0], 1024):
            lg = model.lm_head(h[s:s + 1024]).float(); tot = tot + (lg.gather(1, tgt[s:s + 1024, None])[:, 0] - torch.logsumexp(lg, 1)).sum()
        return tot
    def logp_batch(items, budget=65536):   # items: [(i, y)] -> per-token-mean logp list (no grad); right-padded, causal => exact
        order = sorted(range(len(items)), key=lambda n: len(prompt(items[n][0])) + len(items[n][1])); res = [None] * len(items); k = 0
        with torch.no_grad():
            while k < len(order):
                Lm = len(prompt(items[order[k]][0])) + len(items[order[k]][1]); grp = [order[k]]; k += 1
                while k < len(order):
                    Ln = len(prompt(items[order[k]][0])) + len(items[order[k]][1])
                    if Ln * (len(grp) + 1) > budget: break
                    grp.append(order[k]); Lm = Ln; k += 1
                padn = -(-Lm // 256) * 256; seqs = []
                for n in grp:
                    i, y = items[n]; s = prompt(i) + list(map(int, y)); seqs.append(s + [PAD] * (padn - len(s)))
                ids = torch.tensor(seqs, device='cuda'); am = (ids != PAD).long()
                for r, n in enumerate(grp): am[r, :len(prompt(items[n][0])) + len(items[n][1])] = 1
                h = model.model(input_ids=ids, attention_mask=am).last_hidden_state
                for r, n in enumerate(grp):
                    i, y = items[n]; lp = len(prompt(i)); ln = len(y); hh = h[r, lp - 1:lp - 1 + ln]; tgt = ids[r, lp:lp + ln]; tot = 0.0
                    for s in range(0, ln, 2048):
                        lg = model.lm_head(hh[s:s + 2048]).float(); tot += float((lg.gather(1, tgt[s:s + 2048, None])[:, 0] - torch.logsumexp(lg, 1)).sum())
                    res[n] = tot / ln
        return res
    chunks, info, _ = gc.build_chunks(meta, E)
    ev = {h: [k for k, c in enumerate(chunks) if c['half'] == h and c['kind'] == 'subject'] for h in (0, 1)}
    allk = ev[0] + ev[1]; G = torch.stack([torch.load(gc.SHM / f'e{E}' / f'ref-{k}.pt') for k in allk]).cuda()
    items, where = [], []
    for k in allk:
        for i in chunks[k]['tasks'][:a.tasks_per_chunk]:
            for x in info[i]:
                if x['outcome'] != 'indeterminate' and x['len'] > 0: items.append((i, x['tokens'])); where.append((k, i, x['outcome'] == 'positive'))
    def J():
        if a.batched: lp = logp_batch(items)
        else:
            with torch.no_grad(): lp = [float(logp1(i, y)) / len(y) for i, y in items]
        acc = {}
        for (k, i, pos), v in zip(where, lp): acc.setdefault(k, {}).setdefault(i, ([], []))[0 if pos else 1].append(v)
        return np.array([np.mean([np.mean(c) - np.mean(w) for c, w in acc[k].values() if c and w]) for k in allk])
    chk = [0, len(items) // 2, len(items) - 1]; bv = logp_batch([items[c] for c in chk])
    with torch.no_grad(): sv = [float(logp1(*items[c])) / len(items[c][1]) for c in chk]
    print('batched-vs-single logp check', [round(x, 5) for x in bv], [round(x, 5) for x in sv], flush=True)
    t = time.time(); J0 = J(); tJ = time.time() - t; print('J0', round(tJ), 's', len(items), 'seqs', flush=True)
    RT = {(r['miner'], r['slot']): r for r in (json.loads(l) for l in gzip.open(HERE / 'data' / 'real' / f'real-{E}.jsonl.gz', 'rt'))}
    PZ = HERE / 'data' / 'planted' / f'e{E}' / 'new.npz'; nz = np.load(PZ) if PZ.exists() else None
    AZ = {}; R = {}
    for x in meta:
        if x['kind'] == 'ref': R.setdefault(x['index'], []).append(x['tokens'])
    def toks(r):
        if r['src'] == 'real': return RT[(r['key'][0], r['key'][1])]['rollouts'][r['key'][2]]['out']
        if r['src'] == 'new': return nz['tokens'][nz['offsets'][r['key']]:nz['offsets'][r['key'] + 1]]
        if r['src'] == 'ref': return R[r['key'][0]][r['key'][1]]
        nm, j = r['key']
        if nm not in AZ: AZ[nm] = np.load(D / f'{nm}.npz')
        z = AZ[nm]; return z['tokens'][z['offsets'][j]:z['offsets'][j + 1]]
    n0 = len(ev[0]); out = []
    fn = D / f'scratch-{a.tag}{a.shard}.jsonl'
    for b in bs:
        N = sum(len(g['rollouts']) for g in b['groups'])
        for mode in a.modes.split(','):
            for p, o in zip(params, orig):
                with torch.no_grad(): p.copy_(o)
            rec = dict(id=b['id'], type=b['type'], mode=mode, U=a.U, steps=[]); t1 = time.time()
            for st in range(a.steps):
                for p in params: p.grad = None
                for g in b['groups']:
                    rw = np.array([r['reward'] for r in g['rollouts']]); A = rw - rw.mean()
                    for r, ai in zip(g['rollouts'], A):
                        if ai != 0: (float(ai) / (8192 * N) * logp1(g['index'], toks(r))).backward()
                gv = torch.cat([p.grad.reshape(-1) for p in params]).float(); gn = float(gv.norm())
                for p in params: p.grad = None
                upd = a.U * (gv / gv.abs().max() if mode == 'sgd' else torch.sign(gv))
                pred = (G @ upd.to(torch.bfloat16)).float().cpu().numpy()
                off = 0
                with torch.no_grad():
                    for p in params:
                        n = p.numel(); p.add_(upd[off:off + n].view_as(p).to(p.dtype)); off += n
                J1 = J(); d = J1 - J0
                rec['steps'].append(dict(step=st + 1, gnorm=gn, d_ref=float(d[:n0].mean()), d_heldout=float(d[n0:].mean()), pred_ref=float(pred[:n0].mean()), pred_heldout=float(pred[n0:].mean()),
                                         corr_pred_act=float(np.corrcoef(pred, d)[0, 1]) if st == 0 else None))
            rec['sec'] = round(time.time() - t1); out.append(rec); print(json.dumps(rec), flush=True)
            with open(fn, 'a') as f: f.write(json.dumps(rec) + '\n')
    for p, o in zip(params, orig):
        with torch.no_grad(): p.copy_(o)
    print('DONE', E, a.shard, len(out), flush=True)
if __name__ == '__main__':
    main()
