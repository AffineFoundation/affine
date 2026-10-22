import time, torch, glob, json
from transformers import AutoTokenizer, AutoModelForImageTextToText
path = glob.glob('/root/ckpt-cache/cf791667*')[0]
t = time.time()
model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
model.eval(); model.requires_grad_(False); print('load', time.time() - t, type(model).__name__)
lm = model.model.language_model; L = len(lm.layers)
params = [p for li in range(L - 4, L) for n, p in lm.layers[li].mlp.named_parameters()]
print([n for n, _ in lm.layers[L-1].mlp.named_parameters()], sum(p.numel() for p in params))
for p in params: p.requires_grad_(True)
for n in (3072, 3072, 4096, 4096, 3072, 8192):
    ids = torch.randint(0, 200000, (1, n), device='cuda'); torch.cuda.synchronize(); t = time.time()
    h = model.model(input_ids=ids).last_hidden_state[0, :-1]; tgt = ids[0, 1:]; tot = 0
    for s in range(0, h.shape[0], 1024):
        lg = model.lm_head(h[s:s+1024]).float(); tot = tot + (lg.gather(1, tgt[s:s+1024, None])[:, 0] - torch.logsumexp(lg, 1)).sum()
    tot.backward(); torch.cuda.synchronize()
    g = torch.cat([p.grad.reshape(-1) for p in params]); [setattr(p, 'grad', None) for p in params]
    print(n, 'fwd+bwd s', round(time.time() - t, 3), 'mem GB', round(torch.cuda.max_memory_allocated() / 1e9, 1), float(tot) / n, float(g.norm()))
