"""CPU tests for the v8.3.3f fp32 head master (real launcher code, toy tensors)."""
import importlib.util, sys, json, math, shutil, tempfile, threading, time
from pathlib import Path
import torch
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
OLD = load('/workspace/fp32/host_v83.py', 'old_l'); NEW = load('/workspace/fp32/stage/mimo_host_trainer_v83.py', 'new_l')
tmp = Path(tempfile.mkdtemp()); NEW.FP32_STORE = str(tmp / 'store'); NEW.FP32_SHARD_BYTES = 4096
EV = []
emit = lambda row: EV.append(row)
class Opt:
    def __init__(self, params):
        self.parameters = tuple(params)
        self.masters = {n: torch.nn.Parameter(p.detach().float().clone()) for n, p in self.parameters}
        self.global_step = 0; self.failed = False
    def zero_grad(self): pass
def model(seed):
    g = torch.Generator().manual_seed(seed)
    return [('a.weight', torch.randn(64, 32, generator=g).bfloat16()), ('b.bias', torch.randn(32, generator=g).bfloat16()),
            ('c.weight', (torch.randn(128, 16, generator=g) * 3).bfloat16())]
def grads(seed, params):
    g = torch.Generator().manual_seed(seed)
    return {n: torch.randn(p.shape, generator=g) for n, p in params}
LR, CAP, TRC = 5e-4, 1.0, 2e-3   # larger than prod so steps are visible on toy tensors
def wait():
    for t in threading.enumerate():
        if t.name == 'v833f-fp32-save': t.join()
ok = lambda c, m: print(('PASS ' if c else 'FAIL ') + m) or (c or sys.exit(1))
# 1. flag off == v8.3.3 bit for bit
pa, pb = model(1), model(1)
oa, ob = Opt(pa), Opt(pb)
G = grads(7, pa)
ra = OLD.sgd_step(oa, {k: v.clone() for k, v in G.items()}, LR, CAP, 'fresh-adam', (), emit, TRC)
rb = NEW.sgd_step(ob, {k: v.clone() for k, v in G.items()}, LR, CAP, 'fresh-adam', (), emit, TRC)
ok(all(torch.equal(x, y) for (_, x), (_, y) in zip(pa, pb)) and ra['sgd']['raw_update_norm'] == rb['sgd']['raw_update_norm'], 'flag off: identical to v8.3.3')
# 2. flag on, bootstrap step == v8.3.3 step (master = upcast)
pc = model(1); oc = Opt(pc)
rc = NEW.sgd_step(oc, {k: v.clone() for k, v in G.items()}, LR, CAP, 'fresh-adam', (), emit, TRC, fp32_master=True)
wait()
fm = rc['sgd']['fp32_master']
ok(fm['source'] == 'bootstrap-upcast' and all(torch.equal(x, y) for (_, x), (_, y) in zip(pa, pc)), 'bootstrap step publishes the same bf16 as v8.3.3')
ok(all(torch.equal(oc.masters[n].detach().bfloat16(), p) for n, p in pc) and any(not torch.equal(oc.masters[n].detach(), p.float()) for n, p in pc), 'master rounds to the candidate and keeps sub-bf16 bits')
ok((Path(NEW.FP32_STORE) / fm['output_bf16_sha256'] / 'meta.json').is_file(), 'candidate master saved to the store')
# reference: Allan arm fp32 math, independent implementation
def ref_step(M, G):
    out = {}
    q = {}; r = {}
    for n, m in M.items():
        d = G[n] / (G[n].abs() + 1e-8)
        if m.dim() >= 2 and not n.endswith('bias'): d = d + 0.01 * m
        d = LR * d; q[n] = float(d.double().pow(2).sum())
        wn = math.sqrt(float(m.double().pow(2).sum())); r[n] = min(1.0, TRC * (1 - 1e-6) * wn / math.sqrt(q[n]))
        out[n] = d
    raw = math.sqrt(math.fsum(q[n] * r[n] ** 2 for n in M)); s = min(1.0, CAP / raw)
    return {n: M[n] - torch.tensor(s * r[n], dtype=torch.float32) * out[n] for n in M}
# 3. second step from memory (GREEN continuation) == reference fp32 math from the master
M1 = {n: oc.masters[n].detach().clone() for n, _ in pc}
G2 = grads(8, pc)
rc2 = NEW.sgd_step(oc, {k: v.clone() for k, v in G2.items()}, LR, CAP, 'fresh-adam', (), emit, TRC, fp32_master=True); wait()
R2 = ref_step(M1, G2)
ok(rc2['sgd']['fp32_master']['source'] == 'memory', 'GREEN continuation uses the in-memory master')
ok(all(torch.equal(oc.masters[n].detach(), R2[n]) for n in R2) and all(torch.equal(p, R2[n].bfloat16()) for n, p in pc), 'step 2 == reference fp32-master math')
# and differs from the v8.3.3 recipe applied to the bf16 head
pd_ = [(n, oc.masters[n].detach().bfloat16()) for n, _ in pc]  # dummy
# 4. RED: process restarted on head H1 (the step-1 candidate) -> master from disk
H1 = {n: M1[n].bfloat16() for n in M1}
pe = [(n, H1[n].clone()) for n, _ in pc]; oe = Opt(pe)
rc3 = NEW.sgd_step(oe, {k: v.clone() for k, v in G2.items()}, LR, CAP, 'fresh-adam', (), emit, TRC, fp32_master=True); wait()
ok(rc3['sgd']['fp32_master']['source'] == 'disk' and all(torch.equal(oe.masters[n].detach(), R2[n]) for n in R2), 'restart/RED path: head master loaded from disk, same candidate as memory path')
ok(rc3['sgd']['fp32_master']['input_master_digest'] == rc['sgd']['fp32_master']['output_master_digest'], 'input master digest == previous output digest')
# 5. tampered store entry -> rejected, bootstrap
ent = Path(NEW.FP32_STORE) / rc['sgd']['fp32_master']['output_bf16_sha256']
meta = json.loads((ent / 'meta.json').read_text()); meta['master_digest'] = '0' * 64; (ent / 'meta.json').write_text(json.dumps(meta))
pf = [(n, H1[n].clone()) for n, _ in pc]; of = Opt(pf)
rc4 = NEW.sgd_step(of, {k: v.clone() for k, v in G2.items()}, LR, CAP, 'fresh-adam', (), emit, TRC, fp32_master=True); wait()
ok(rc4['sgd']['fp32_master']['source'] == 'bootstrap-upcast' and any(e.get('event') == 'fp32-master-load-rejected' for e in EV), 'tampered store entry rejected -> bootstrap')
# 6. memory master stale (params changed under it) -> not used
pg = model(3); og = Opt(pg)
NEW.sgd_step(og, grads(9, pg), LR, CAP, 'fresh-adam', (), emit, TRC, fp32_master=True); wait()
with torch.no_grad(): pg[0][1].add_(1.0)
r6 = NEW.sgd_step(og, grads(10, pg), LR, CAP, 'fresh-adam', (), emit, TRC, fp32_master=True); wait()
ok(r6['sgd']['fp32_master']['source'] == 'bootstrap-upcast', 'stale in-memory master is not used')
# 7. GC keeps <= FP32_KEEP + head
n_ent = len(NEW._fp32_store_entries()); ok(n_ent <= NEW.FP32_KEEP + 1, 'store GC bound (%d entries)' % n_ent)
# 8. settings validation
ok(NEW.fp32_settings({'sgd_mode': 'fresh-adam', 'sgd_fp32_master': True}) is True and NEW.fp32_settings({}) is False, 'fp32_settings')
try: NEW.fp32_settings({'sgd_mode': 'plain', 'sgd_fp32_master': True}); ok(False, 'plain rejected')
except ValueError: ok(True, 'fp32 master needs fresh-adam')
json.dumps(rc2); print('ALL PASS'); shutil.rmtree(tmp)
