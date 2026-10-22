#!/usr/bin/env python3
"""Per-rollout CountSketch of d logp(y|prompt)/d theta_s (same subspace: MLP layers 28-31, 604M params), so the norm of ANY
batch gradient g_b = sum_i A_i grad_i / (8192 N) (any rewards, any mixture of groups) can be estimated as ||sum_i A_i sk_i||/(8192N).
Sketch: bucket h(p) in [0,D), sign s(p) in {+-1} from a fixed seed; E||sk||^2 = ||g||^2, rel. error ~ sqrt(2/D).
Items: real miner rollouts (real-<E>.jsonl.gz), planted new sequences (planted/e<E>/new.npz), regenerated groups used by type 7.
--exact-check N: also accumulate the exact gradient of the first N planted batches' g_b to validate the sketch."""
import argparse, glob, gzip, json, os, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
from plant_dots import key
import grad_claims as gc
DS = 16384
def items(E):
    T = []
    for r in (json.loads(l) for l in gzip.open(HERE / 'data/real' / f'real-{E}.jsonl.gz', 'rt')):
        for x in r['rollouts']: T.append((r['index'], np.asarray(x['out'], dtype=np.int32)))
    P = HERE / 'data' / 'planted' / f'e{E}'
    if (P / 'new.json').exists():
        m = json.load(open(P / 'new.json')); z = np.load(P / 'new.npz'); o = z['offsets']; t = z['tokens']
        for n, x in enumerate(m['meta']): T.append((x['index'], t[o[n]:o[n + 1]]))
        spec = json.load(open(P / 'spec.json')); need = {(r['key'][0], r['key'][1]) for b in spec['batches'] for g in b['groups'] for r in g['rollouts'] if r['src'] == 'regen'}
        if need:
            _, gm = gc.load_gen(E)
            for x in gm:
                if x['kind'] == 'batch' and (x['uid'], x['index']) in need: T.append((x['index'], np.asarray(x['tokens'], dtype=np.int32)))
    return T
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epoch', required=True); ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args(); E = a.epoch; out = HERE / 'data' / 'gnorm' / f'e{E}'; out.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in glob.glob(str(out / 'sk-*.npz')): done.update(np.load(f)['keys'].tolist())
    todo = {}
    for i, t in items(E):
        k = key(i, t)
        if k not in done and int(k, 16) % a.nshards == a.shard: todo[k] = (i, t)
    print('todo', len(todo), flush=True)
    if not todo: return
    T = json.load(open(ROOT / 'data/original-math7496.tasks.json'))
    import grad_real
    base, _ = grad_real.load_ref(E); path = glob.glob(f'/root/ckpt-cache/{base}')[0]
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(path); PAD = tok.pad_token_id if tok.pad_token_id is not None else 248044
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa').cuda(); model.eval(); model.requires_grad_(False)
    lm = model.model.language_model; L = len(lm.layers)
    params = [p for li in range(L - 4, L) for n, p in lm.layers[li].mlp.named_parameters()]
    for p in params: p.requires_grad_(True)
    P = sum(p.numel() for p in params)
    g = torch.Generator(device='cuda'); g.manual_seed(20261019)
    H = torch.randint(0, DS, (P,), device='cuda', generator=g, dtype=torch.int64).to(torch.int32)
    S = (torch.randint(0, 2, (P,), device='cuda', generator=g, dtype=torch.int8) * 2 - 1)
    prompts = {}
    def prompt(i):
        if i not in prompts:
            d = T[i]['data']; prompts[i] = list(tok.apply_chat_template([dict(role='system', content=d['system_prompt']), dict(role='user', content=d['prompt'])], tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False))
        return prompts[i]
    keys, sks, sq = [], [], []; t0 = time.time()
    def flush():
        if keys: np.savez(out / f'sk-{a.shard}-{os.getpid()}.npz', keys=np.array(keys), sk=np.stack(sks), sq=np.array(sq))
    for n, (k, (i, y)) in enumerate(todo.items()):
        p = prompt(i); seq = p + list(map(int, y)); Ln = len(seq); padn = -(-Ln // 512) * 512
        ids = torch.tensor([seq + [PAD] * (padn - Ln)], device='cuda')
        h = model.model(input_ids=ids).last_hidden_state[0, len(p) - 1:Ln - 1]; tgt = ids[0, len(p):Ln]; tot = 0.0
        for s in range(0, h.shape[0], 1024):
            lg = model.lm_head(h[s:s + 1024]).float(); tot = tot + (lg.gather(1, tgt[s:s + 1024, None])[:, 0] - torch.logsumexp(lg, 1)).sum()
        tot.backward(); gr = torch.cat([q.grad.reshape(-1) for q in params]).float()
        for q in params: q.grad = None
        sk = torch.zeros(DS, device='cuda').index_add_(0, H, gr * S); keys.append(k); sks.append(sk.cpu().numpy()); sq.append(float((gr * gr).sum()))
        if n % 1000 == 999: flush(); print(n + 1, round(time.time() - t0), flush=True)
    flush(); print('DONE', E, a.shard, len(todo), round(time.time() - t0), flush=True)
if __name__ == '__main__':
    main()
