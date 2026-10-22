#!/usr/bin/env python3
"""Real scratch step vs first-order prediction, out of sample.
For each given uid: g_b = sum_i A_i dlogp(y_i)/(8192 N) on the subspace (the GRPO/SGD ascent direction);
theta_s <- theta_s + eta*g_b on a throwaway copy (restored after), with eta set so max|eta*g_b| = U (per-param update);
measure J_k on held-out half-1 chunks (forward only) before/after; compare with eta*<g_b, G_k> (first order)."""
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import grad_claims as gc, grad_real as gr, gzip
sys.path.insert(0, str(gc.ROOT)); import grader

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epoch', required=True); ap.add_argument('--uids', required=True)
    ap.add_argument('--updates', default="1e-3"); ap.add_argument('--half', type=int, default=1); a = ap.parse_args()
    E = a.epoch; T = json.load(open(gc.ROOT / 'data/original-math7496.tasks.json'))
    base, meta = gr.load_ref(E); path = glob.glob(f'/root/ckpt-cache/{base}')[0]
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(path); PAD = tok.pad_token_id if tok.pad_token_id is not None else 248044
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model.eval(); model.requires_grad_(False); lm = model.model.language_model; L = len(lm.layers)
    # subspace MLPs in fp32 so that small updates are representable (inputs cast up, outputs cast back to bf16)
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
    def logp(i, y, p=None):
        p = prompt(i) if p is None else list(p); seq = p + list(map(int, y)); n = len(seq); padn = -(-n // 512) * 512
        ids = torch.tensor([seq + [PAD] * (padn - n)], device='cuda')
        h = model.model(input_ids=ids).last_hidden_state[0, len(p) - 1:n - 1]; tgt = ids[0, len(p):n]; tot = 0.0
        for s in range(0, h.shape[0], 1024):
            lg = model.lm_head(h[s:s + 1024]).float(); tot = tot + (lg.gather(1, tgt[s:s + 1024, None])[:, 0] - torch.logsumexp(lg, 1)).sum()
        return tot
    chunks, info, _ = gc.build_chunks(meta, E)
    ev = [k for k, c in enumerate(chunks) if c['half'] == a.half]
    G = torch.stack([torch.load(gc.SHM / f'e{E}' / f'ref-{k}.pt') for k in ev]).cuda()
    def J():
        out = []
        with torch.no_grad():
            for k in ev:
                v = 0.0
                for i in chunks[k]['tasks']:
                    xs = [x for x in info[i] if x['outcome'] != 'indeterminate' and x['len'] > 0]
                    c = [float(logp(i, x['tokens'])) / x['len'] for x in xs if x['outcome'] == 'positive']
                    w = [float(logp(i, x['tokens'])) / x['len'] for x in xs if x['outcome'] != 'positive']
                    v += (np.mean(c) - np.mean(w)) / len(chunks[k]['tasks'])
                out.append(v)
        return np.array(out)
    t = time.time(); J0 = J(); tJ = time.time() - t
    R = [json.loads(l) for l in gzip.open(HERE / 'data' / 'real' / f'real-{E}.jsonl.gz', 'rt')]
    res = []
    for u in a.uids.split(','):
        u = int(u); gs = [r for r in R if r['uid'] == u]; N = sum(len(r['rollouts']) for r in gs)
        for p in params: p.grad = None
        for r in gs:
            rw = np.array([grader.overlong_reward(1.0 if x['cls'] == 'positive' else 0.0, len(x['out'])) for x in r['rollouts']])
            A = rw - rw.mean()
            for x, ai in zip(r['rollouts'], A):
                if ai != 0: (float(ai) / (8192 * N) * logp(r['index'], x['out'], r['prompt'])).backward()
        g = torch.cat([p.grad.reshape(-1) for p in params]).float()
        for p in params: p.grad = None
        pred_unit = (G @ g.to(torch.bfloat16)).float().cpu().numpy()   # dJ_k per unit eta (first order)
        for U in map(float, a.updates.split(',')):
            eta = U / float(g.abs().max())
            off = 0
            with torch.no_grad():
                for p in params:
                    n = p.numel(); p.add_((eta * g[off:off + n]).view_as(p).to(p.dtype)); off += n
            J1 = J()
            with torch.no_grad():
                for p, o in zip(params, orig): p.copy_(o)
            act = J1 - J0; pred = eta * pred_unit
            res.append(dict(uid=u, max_update=U, eta=eta, actual_mean=float(act.mean()), predicted_mean=float(pred.mean()),
                            corr_over_chunks=float(np.corrcoef(act, pred)[0, 1]), slope=float(np.polyfit(pred, act, 1)[0]),
                            actual_by_chunk=act.tolist(), predicted_by_chunk=pred.tolist()))
            print(json.dumps({k: v for k, v in res[-1].items() if 'by_chunk' not in k}), flush=True)
    json.dump(dict(epoch=E, half=a.half, eval_chunks=len(ev), eval_seconds=tJ, results=res),
              open(HERE / 'data' / f'e{E}' / f'fd-real-{a.uids.replace(",", "_")}.json', 'w'))

if __name__ == '__main__':
    main()
