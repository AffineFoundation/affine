# quick CPU sanity: v833f vs v833g fresh-adam fp32-master step give bit-identical params/masters/digests
import sys, importlib.util, types, json, threading, torch
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
F = load('lf', sys.argv[1]); G = load('lg', sys.argv[2])
F.FP32_STORE = G.FP32_STORE = '/tmp/ts-sanity-store-%d' % id(F)
def make(seed):
    g = torch.Generator().manual_seed(seed)
    names = ['a.weight', 'b.bias', 'c.weight', 'norm.weight']
    shapes = [(64, 32), (32,), (128, 64), (64,)]
    params = [(n, torch.nn.Parameter((torch.randn(*s, generator=g) * 0.05).to(torch.bfloat16))) for n, s in zip(names, shapes)]
    grads = {n: torch.randn(*s, generator=g) for n, s in zip(names, shapes)}
    opt = types.SimpleNamespace(parameters=tuple(params), masters={n: torch.nn.Parameter(p.detach().float().clone()) for n, p in params},
                                failed=False, global_step=5, zero_grad=lambda: None)
    return opt, grads
out = []
for M in (F, G):
    opt, grads = make(1)
    M._fp32_save = lambda *a, **k: None
    rows = []
    for step in range(3):
        r = M.fresh_adam_step(opt, grads, list(opt.parameters), 5e-7, 1.0, (), lambda row: rows.append(row), 0.0, 2e-5, True)
        if hasattr(M, '_PUBLISHED'): M._PUBLISHED.set()
        opt._v833f_lock.acquire(); opt._v833f_lock.release()
    out.append(({n: p.detach().view(torch.int16).clone() for n, p in opt.parameters},
                {n: m.detach().view(torch.int32).clone() for n, m in opt.masters.items()},
                [(x['fp32_master']['input_master_digest'], x['fp32_master']['output_master_digest'], x['fp32_master']['source'], x['update_norm_bf16']) for x in rows]))
a, b = out
assert all(torch.equal(a[0][n], b[0][n]) for n in a[0]), 'params differ'
assert all(torch.equal(a[1][n], b[1][n]) for n in a[1]), 'masters differ'
assert a[2] == b[2], (a[2], b[2])
print(json.dumps(dict(ok=True, steps=len(a[2]), sources=[x[2] for x in a[2]])))
