"""CPU tests for trainer host v8.3.3i (real launcher code, toy tensors): normgrad direction step + support renorm helpers."""
import importlib.util, sys, json, math, shutil, tempfile, threading
from pathlib import Path
import torch
RA = Path('/home/const/subnet120-rewrite/state/root-audits')
OLDP = RA / '20261030-free-file/release-v833h-signed/trainer-host-v833h'
NEWP = Path(sys.argv[1]) if len(sys.argv) > 1 else RA / '20261031-renorm-step-live/stage/host-v833i'
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
OLD = load(OLDP / 'mimo_host_trainer_v83.py', 'old_l'); NEW = load(NEWP / 'mimo_host_trainer_v83.py', 'new_l')
tmp = Path(tempfile.mkdtemp()); OLD.FP32_STORE = str(tmp / 'old'); NEW.FP32_STORE = str(tmp / 'new'); OLD.FP32_SHARD_BYTES = NEW.FP32_SHARD_BYTES = 4096; OLD.FP32_SAVE_DEFER_MAX = NEW.FP32_SAVE_DEFER_MAX = 0
EV = []; emit = EV.append
class Opt:
    def __init__(self, params):
        self.parameters = tuple(params)
        self.masters = {n: torch.nn.Parameter(p.detach().float().clone()) for n, p in self.parameters}
        self.global_step = 0; self.failed = False
    def zero_grad(self): pass
def model(seed):
    g = torch.Generator().manual_seed(seed)
    return [('a.weight', torch.randn(64, 32, generator=g).bfloat16()), ('b.bias', torch.randn(32, generator=g).bfloat16()),
            ('c.weight', (torch.randn(128, 16, generator=g) * 3).bfloat16()), ('z.weight', torch.randn(8, 8, generator=g).bfloat16())]
def grads(seed, params, zero=()):
    g = torch.Generator().manual_seed(seed)
    out = {n: torch.randn(p.shape, generator=g) * (0.01 if n.startswith('c') else 1.0) for n, p in params}
    for n in zero: out[n] = torch.zeros_like(out[n])
    return out
LR, CAP, TRC = 5e-4, 1.0, 2e-3
def wait():
    for t in threading.enumerate():
        if t.name == 'v833f-fp32-save': t.join()
ok = lambda c, m: print(('PASS ' if c else 'FAIL ') + m, flush=True) or (c or sys.exit(1))
G = grads(7, model(1), zero=('z.weight',))
# 1. sgd_direction absent/sign == v8.3.3h bit for bit (fp32 master on)
pa, pb = model(1), model(1); oa, ob = Opt(pa), Opt(pb)
ra = OLD.sgd_step(oa, {k: v.clone() for k, v in G.items()}, LR, CAP, 'fresh-adam', (), emit, TRC, fp32_master=True); wait()
rb = NEW.sgd_step(ob, {k: v.clone() for k, v in G.items()}, LR, CAP, 'fresh-adam', (), emit, TRC, fp32_master=True); wait()
ok(all(torch.equal(x, y) for (_, x), (_, y) in zip(pa, pb)) and all(torch.equal(oa.masters[n], ob.masters[n]) for n in oa.masters)
   and ra['sgd']['raw_update_norm'] == rb['sgd']['raw_update_norm'] and ra['implementation'] == rb['implementation'], 'direction sign: identical to v8.3.3h')
# 2. normgrad == Allan bf16_exp Env.scales/new_weights (independent re-implementation, python-float gscale * g)
pc = model(1); oc = Opt(pc); M0 = {n: oc.masters[n].detach().clone() for n, _ in pc}
rc = NEW.sgd_step(oc, {k: v.clone() for k, v in G.items()}, LR, CAP, 'fresh-adam', (), emit, TRC, fp32_master=True, direction='normgrad'); wait()
def allan(M, G):
    rr, qq, gn2, dn2 = {}, {}, {}, []
    for n, m in M.items():
        d = G[n] / (G[n].abs() + torch.tensor(1e-8, dtype=torch.float32))
        if m.dim() >= 2 and not n.endswith('bias'): d = d + torch.tensor(0.01, dtype=torch.float32) * m
        d = torch.tensor(LR, dtype=torch.float32) * d
        q = float(d.double().pow(2).sum()); qq[n] = q; gn2[n] = float(G[n].double().pow(2).sum())
        wn = math.sqrt(float(m.double().pow(2).sum())); rt = min(1.0, TRC * (1 - 1e-6) * wn / math.sqrt(q)) if q > 0 and wn > 0 else 1.0
        rr[n] = rt; dn2.append(q * rt * rt)
    raw = math.sqrt(math.fsum(dn2)); cs = min(1.0, CAP / raw) if raw > 0 else 1.0
    gs = {n: (cs * rr[n] * math.sqrt(qq[n]) / math.sqrt(gn2[n])) if gn2[n] > 0 else 0.0 for n in rr}
    return {n: M[n] - gs[n] * G[n] for n in M}, raw, {n: cs * rr[n] * math.sqrt(qq[n]) for n in M}, rr
R, raw, tnorm, rr = allan(M0, G)
mx = max(float((oc.masters[n].detach() - R[n]).abs().max()) for n in R)
ok(mx == 0.0, 'normgrad master == Allan bf16_exp normgrad (max |diff| %g)' % mx)
ok(all(torch.equal(p, R[n].bfloat16()) for n, p in pc), 'normgrad bf16 candidate == bf16_rne(Allan master)')
dn = {n: math.sqrt(float((oc.masters[n].detach().double() - M0[n].double()).pow(2).sum())) for n in M0}
ok(all(abs(dn[n] - tnorm[n]) <= 1e-5 * max(tnorm[n], 1e-30) + 1e-12 for n in M0 if n != 'z.weight'), 'per-tensor ||Delta_t|| == s * r_t * ||d_t||')
s = rc['sgd']; um = s['fp32_master']['update_norm_master']
ok(abs(um - s['raw_update_norm']) / s['raw_update_norm'] < 1e-5 and abs(raw - s['raw_update_norm']) < 1e-12, 'update_norm_master == raw_update_norm (%.6g vs %.6g)' % (um, s['raw_update_norm']))
ok(torch.equal(oc.masters['z.weight'].detach(), M0['z.weight']) and s['zero_gradient_tensors'] == 1, 'zero-gradient tensor not moved')
ok(s['sgd_direction'] == 'normgrad' and s['version'] == 'v833i-normgrad-fp32-master-v1' and sum(v < 1 for v in rr.values()) == s['tensors_rel_capped'], 'report fields')
ok(not all(torch.equal(x, y) for (_, x), (_, y) in zip(pa, pc)), 'normgrad differs from the sign step')
# direction: Delta_t parallel to g_t
for n in ('a.weight', 'c.weight'):
    dlt = (M0[n] - oc.masters[n].detach()).double().flatten(); g = G[n].double().flatten()
    cos = float(dlt @ g / (dlt.norm() * g.norm())); ok(cos > 0.999999, '%s Delta parallel to g (cos %.9f)' % (n, cos))
# 3. second normgrad step from memory master
G2 = grads(8, pc); M1 = {n: oc.masters[n].detach().clone() for n, _ in pc}
rc2 = NEW.sgd_step(oc, {k: v.clone() for k, v in G2.items()}, LR, CAP, 'fresh-adam', (), emit, TRC, fp32_master=True, direction='normgrad'); wait()
R2 = allan(M1, G2)[0]
ok(rc2['sgd']['fp32_master']['source'] == 'memory' and all(torch.equal(oc.masters[n].detach(), R2[n]) for n in R2), 'GREEN continuation: memory master, == Allan math')
# 4. settings validation
E = dict(sgd_mode='fresh-adam', sgd_fp32_master=True)
ok(NEW.direction_settings({}) == 'sign' and NEW.direction_settings(dict(E, sgd_direction='normgrad')) == 'normgrad', 'direction_settings')
for bad in (dict(E, sgd_direction='x'), dict(sgd_mode='fresh-adam', sgd_direction='normgrad'), dict(sgd_mode='plain', sgd_fp32_master=True, sgd_direction='normgrad')):
    try: NEW.direction_settings(bad); ok(False, 'bad direction accepted %r' % bad)
    except ValueError: pass
ok(True, 'bad direction settings rejected')
sup = dict(top_k=20, top_p=0.95, temperature=1.0)
ok(NEW.support_settings({}) is None and NEW.support_settings(dict(support_renorm=sup)) == sup, 'support_settings')
for bad in (dict(top_k=20, top_p=0.95), dict(top_k=20.0, top_p=0.95, temperature=1.0), dict(top_k=20, top_p=1, temperature=1.0), dict(top_k=20, top_p=0.95, temperature=1.0, x=1)):
    try: NEW.support_settings(dict(support_renorm=bad)); ok(False, 'bad support accepted %r' % bad)
    except ValueError: pass
ok(True, 'bad support settings rejected')
# 5. support_logq (patched v8211_batched and v83_fast_kernels) vs a direct reference
B = load(NEWP / 'v8211_batched.py', 'b_new') if False else None
src = {}
for f in ('v8211_batched.py', 'v83_fast_kernels.py'):
    txt = (NEWP / f).read_text(); i = txt.index('def support_logq'); j = txt.index('\ndef ', i + 10)
    ns = {'torch': torch}; exec(txt[i:j], ns); src[f] = ns['support_logq']
g = torch.Generator().manual_seed(3); logits = torch.randn(200, 1000, generator=g) * 3; logits.requires_grad_(True)
t = torch.randint(0, 1000, (200,), generator=g); t[:100] = logits[:100].argmax(-1)
for f, fn in src.items():
    lq, inside = fn(logits, t, 20, 0.95)
    exp_in, exp_lq = [], []
    for i in range(200):
        v, ids = logits[i].detach().topk(20); p = torch.softmax(v, -1); keep = (p.cumsum(-1) - p) < 0.95
        k_ids = ids[keep]; hit = (k_ids == t[i]).any().item(); exp_in.append(hit)
        exp_lq.append(float(torch.log_softmax(v[keep], -1)[(k_ids == t[i]).nonzero()[0, 0]]) if hit else 0.0)
    ok(inside.tolist() == exp_in and max(abs(a - b) for a, b in zip(lq.tolist(), exp_lq)) < 1e-5, '%s support_logq == reference (%d/200 inside)' % (f, sum(exp_in)))
    lq.sum().backward(); ok(logits.grad is not None and torch.isfinite(logits.grad).all(), '%s log q differentiable' % f); logits.grad = None
# 6. support_summary
tr = [dict(support_outside_tokens=1, loss_tokens=100, support_logq_mean=-0.5), dict(support_outside_tokens=0, loss_tokens=50, support_logq_mean=-0.2),
      dict(support_outside_tokens=None, loss_tokens=10)]
sm = NEW.support_summary(tr, sup)
ok(sm['outside_tokens'] == 1 and sm['loss_tokens'] == 150 and abs(sm['outside_fraction'] - 1 / 150) < 1e-12 and sm['rollouts_outside_over_0p5pct'] == 1
   and abs(sm['inside_logq_mean'] - (-0.5 * 99 - 0.2 * 50) / 149) < 1e-12, 'support_summary')
json.dumps(rc2); json.dumps(sm); print('ALL PASS'); shutil.rmtree(tmp)
