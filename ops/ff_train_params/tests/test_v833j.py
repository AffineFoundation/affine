"""CPU tests for trainer host v8.3.3j (real launcher code, toy tensors): per-turn lr multiplier on the v8.3.3i normgrad step,
ff_trainer.train_params, and the train_update_v833j wrapper (exact source, stubbed inner update). RA/20261031-ff-train-params."""
import ast, importlib.util, sys, json, math, shutil, tempfile, threading, types, dataclasses
from pathlib import Path
import torch
RA = Path('/home/const/subnet120-rewrite/state/root-audits')
OLDP = Path(sys.argv[1]); NEWP = Path(sys.argv[2])
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
OLD = load(OLDP / 'mimo_host_trainer_v83.py', 'old_i'); NEW = load(NEWP / 'mimo_host_trainer_v83.py', 'new_j')
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
            ('c.weight', (torch.randn(128, 16, generator=g) * 3).bfloat16()), ('n.weight', (torch.ones(48) + 0.01 * torch.randn(48, generator=g)).bfloat16()),
            ('z.weight', torch.randn(8, 8, generator=g).bfloat16())]
def grads(seed, params, zero=()):
    g = torch.Generator().manual_seed(seed)
    out = {n: torch.randn(p.shape, generator=g) * (0.01 if n.startswith('c') else 1.0) for n, p in params}
    for n in zero: out[n] = torch.zeros_like(out[n])
    return out
# per-tensor ceiling 1e-3 * ||W||: 'a','c','z' (rms ~1-3) are far below lr*sqrt(n) -> capped? lr 2e-3 -> ||d|| ~ 2e-3 sqrt(n) vs 1e-3 * rms * sqrt(n)
LR, CAP, TRC = 2e-3, 1.0, 1e-3
def wait():
    for t in threading.enumerate():
        if t.name == 'v833f-fp32-save': t.join()
ok = lambda c, m: print(('PASS ' if c else 'FAIL ') + m, flush=True) or (c or sys.exit(1))
G = grads(7, model(1), zero=('z.weight',))
def step(mod, mult, cap=CAP, seed=1):
    p = model(seed); o = Opt(p); M0 = {n: o.masters[n].detach().clone() for n, _ in p}
    if mod is NEW: NEW.TURN['lr_multiplier'] = mult
    try:
        r = mod.sgd_step(o, {k: v.clone() for k, v in G.items()}, LR, cap, 'fresh-adam', (), emit, TRC, fp32_master=True, direction='normgrad'); wait()
    finally:
        if mod is NEW: NEW.TURN['lr_multiplier'] = 1.0
    return p, o, M0, r, (dict(NEW.TURN['last']) if mod is NEW else None)
# 1. defaults (m = 1.0) == v8.3.3i bit for bit
pa, oa, M0, ra, _ = step(OLD, 1.0); pb, ob, _, rb, la = step(NEW, 1.0)
ok(all(torch.equal(x, y) for (_, x), (_, y) in zip(pa, pb)) and all(torch.equal(oa.masters[n], ob.masters[n]) for n in oa.masters)
   and ra['sgd']['raw_update_norm'] == rb['sgd']['raw_update_norm'] and ra['implementation'] == rb['implementation']
   and ra['sgd']['tensors_rel_capped'] == rb['sgd']['tensors_rel_capped'], 'm=1: bf16 + fp32 master identical to v8.3.3i')
ok(la['path'] == 'v833i-exact' and la['raw_update_norm'] == ra['sgd']['raw_update_norm'], 'm=1: path v833i-exact')
capped = ra['sgd']['tensors_rel_capped']; ok(0 < capped < 5, 'toy model has capped (%d) and uncapped tensors' % capped)
# independent reference for the scaled step
def ref(M, mult, cap=CAP):
    rr, qq, gn2, wn = {}, {}, {}, {}
    for n, m in M.items():
        d = G[n] / (G[n].abs() + torch.tensor(1e-8, dtype=torch.float32))
        if m.dim() >= 2 and not n.endswith('bias'): d = d + torch.tensor(0.01, dtype=torch.float32) * m
        d = torch.tensor(LR, dtype=torch.float32) * d
        q = float(d.double().pow(2).sum()); qq[n] = q; gn2[n] = float(G[n].double().pow(2).sum())
        wn[n] = math.sqrt(float(m.double().pow(2).sum())); rr[n] = min(1.0, TRC * (1 - 1e-6) * wn[n] / math.sqrt(q)) if q > 0 and wn[n] > 0 else 1.0
    n0 = {n: rr[n] * math.sqrt(qq[n]) for n in M}
    nt = {n: min(mult * n0[n], TRC * (1 - 1e-6) * wn[n]) for n in M}
    raw = math.sqrt(math.fsum(v * v for v in nt.values())); s = min(1.0, cap / raw) if raw > 0 else 1.0
    return {n: M[n] - ((s * nt[n] / math.sqrt(gn2[n])) if gn2[n] > 0 else 0.0) * G[n] for n in M}, {n: s * nt[n] for n in M}, n0, rr, s
def dnorm(o, M0):
    return {n: math.sqrt(float((o.masters[n].detach().double() - M0[n].double()).pow(2).sum())) for n in M0}
for mult in (0.25, 0.5, 2.0):
    p, o, M0, r, last = step(NEW, mult)
    R, tn, n0, rr, s = ref(M0, mult)
    mx = max(float((o.masters[n].detach() - R[n]).abs().max()) for n in R)
    ok(mx == 0.0, 'm=%g: master == independent reference (max |diff| %g)' % (mult, mx))
    ok(all(torch.equal(x, R[n].bfloat16()) for n, x in p), 'm=%g: bf16 candidate == bf16_rne(reference master)' % mult)
    dn = dnorm(o, M0)
    ok(all(abs(dn[n] - tn[n]) <= 1e-3 * max(tn[n], 1e-30) + 1e-12 for n in M0 if n != 'z.weight'), 'm=%g: per-tensor ||Delta_t|| == s * n_t' % mult)
    if mult < 1:
        ok(all(abs(dn[n] - mult * n0[n]) <= 1e-3 * n0[n] + 1e-12 for n in M0 if n != 'z.weight') and last['tensors_at_ceiling'] == 0,
           'm=%g: every tensor step = m x the default step (lr matters even where the cap binds)' % mult)
    else:
        grow = [n for n in M0 if rr[n] == 1.0 and n != 'z.weight']; stay = [n for n in M0 if rr[n] < 1.0]
        ok(grow and stay and all(dn[n] <= TRC * math.sqrt(float(M0[n].double().pow(2).sum())) * (1 + 1e-3) for n in M0)
           and all(abs(dn[n] - n0[n]) <= 1e-3 * n0[n] for n in stay) and all(dn[n] > 1.0001 * n0[n] for n in grow),
           'm=2: uncapped tensors grow (%s), capped tensors stay at the ceiling (%s), no tensor above the ceiling' % (grow, stay))
    for n in ('a.weight', 'c.weight'):
        dlt = (M0[n] - o.masters[n].detach()).double().flatten(); g = G[n].double().flatten()
        ok(float(dlt @ g / (dlt.norm() * g.norm())) > 0.999999, 'm=%g: %s Delta parallel to g' % (mult, n))
    ok(torch.equal(o.masters['z.weight'].detach(), M0['z.weight']), 'm=%g: zero-gradient tensor not moved' % mult)
    ok(last['path'] == 'scaled' and last['lr_multiplier'] == mult and abs(last["update_norm_master"] - last["raw_update_norm"]) / last["raw_update_norm"] < 1e-3
       and last['raw_update_norm_default'] == r['sgd']['raw_update_norm'], 'm=%g: TURN.last stats (raw %.6g, default %.6g, at ceiling %d)'
       % (mult, last['raw_update_norm'], last['raw_update_norm_default'], last['tensors_at_ceiling']))
# global cap still a ceiling
p, o, M0, r, last = step(NEW, 0.5, cap=1e-3)
R, tn, n0, rr, s = ref(M0, 0.5, cap=1e-3)
ok(s < 1.0 and last['cap_fired'] and float(max((o.masters[n].detach() - R[n]).abs().max() for n in R)) == 0.0 and last["update_norm_master"] <= 1e-3 * (1 + 1e-3),
   'global update-norm cap applies to the scaled step (s %.4g, norm %.6g)' % (s, last['update_norm_master']))
# m != 1 refused without normgrad
NEW.TURN['lr_multiplier'] = 0.5
try:
    p = model(1); o = Opt(p); NEW.sgd_step(o, {k: v.clone() for k, v in G.items()}, LR, CAP, 'fresh-adam', (), emit, TRC, fp32_master=True); ok(False, 'sign + m accepted')
except ValueError: ok(True, 'm != 1 refused for the sign step')
finally: NEW.TURN['lr_multiplier'] = 1.0; wait()
# 2. ff_trainer.train_params
FT_SRC = (NEWP / 'ff_trainer.py').read_text(); i = FT_SRC.index('def need'); j = FT_SRC.index('\n@dataclass')
ns = {}; exec("TRAIN_BOUNDS = {'lr': (1.25e-07, 1e-06), 'kl': (0.005, 0.1)}\n" + FT_SRC[i:j], ns)
k = FT_SRC.index('def train_params'); l = FT_SRC.index('\ndef prepare'); exec(FT_SRC[k:l], ns); tp = ns['train_params']
B = types.SimpleNamespace(epoch='e1', checkpoint='c' * 64)
D = dict(lr=5e-07, kl=0.02)
ok(tp({}, B) is None, 'train_params: absent -> None')
got = tp(dict(train=dict(lr=1e-06, kl=0.005, given=['kl', 'lr'], defaults=D)), B)
ok(got == dict(lr=1e-06, kl=0.005, given=['kl', 'lr'], defaults=D, epoch='e1', checkpoint='c' * 64), 'train_params: valid')
for bad in (dict(lr=1.1e-06, kl=0.02, given=[], defaults=D), dict(lr=5e-07, kl=0.2, given=[], defaults=D), dict(lr=5e-07, kl=1, given=[], defaults=D),
            dict(lr=5e-07, kl=0.02, given=['x'], defaults=D), dict(lr=5e-07, kl=0.02, given=[]), dict(lr=5e-07, kl=0.02, given=[], defaults=D, x=1)):
    try: tp(dict(train=bad), B); ok(False, 'bad train accepted %r' % bad)
    except ValueError: pass
ok(True, 'train_params: out-of-range / malformed refused')
# 3. train_update_v833j wrapper (exact source from the launcher, stubbed inner update)
src = (NEWP / 'mimo_host_trainer_v83.py').read_text(); tree = ast.parse(src)
inst = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install'][0]
fn = [n for n in ast.walk(inst) if isinstance(n, ast.FunctionDef) and n.name == 'train_update_v833j'][0]
code = ast.get_source_segment(src, fn)
import textwrap; code = textwrap.dedent(code)
@dataclasses.dataclass(frozen=True)
class Cfg:
    kl_coefficient: float = 0.02
    top_k: int = 20
    def validate(self):
        if not 0 <= self.kl_coefficient <= 1: raise ValueError('kl')
        return self
    def document(self): return dict(settings=dict(kl_coefficient=self.kl_coefficient))
calls = []
def inner(runtime, optimizer, groups, binding, *, config, **kw):
    calls.append(dict(cfg=config, opt_cfg=optimizer.config, mult=W['TURN']['lr_multiplier']))
    W['TURN']['last'] = dict(path='scaled' if W['TURN']['lr_multiplier'] != 1.0 else 'v833i-exact', lr_multiplier=W['TURN']['lr_multiplier'])
    return dict(status='updated', config=config.document())
W = dict(TURN=dict(lr_multiplier=1.0, last=None), TURN_PARAMS=dict(params=None), train_update_v833i=inner, _dc=dataclasses,
         sgd_lr=5e-07, sgd_mode='fresh-adam', sgd_fp32=True, sgd_dir='normgrad', sgd_trc=2e-05)
exec(code, W); wrap = W['train_update_v833j']
cfg = Cfg(); opt = types.SimpleNamespace(config=cfg); bind = types.SimpleNamespace(epoch='e1', checkpoint='c' * 64)
out = wrap(None, opt, (), bind, config=cfg)
ok(calls[-1]['cfg'] is cfg and calls[-1]['mult'] == 1.0 and 'train_params' not in out, 'wrapper: no train -> v8.3.3i call unchanged')
W['TURN_PARAMS']['params'] = dict(lr=5e-07, kl=0.02, given=[], defaults=D, epoch='e1', checkpoint='c' * 64)
out = wrap(None, opt, (), bind, config=cfg)
ok(calls[-1]['cfg'] is cfg and calls[-1]['mult'] == 1.0 and out['train_params']['defaults_used'] is True and out['train_params']['step']['path'] == 'v833i-exact',
   'wrapper: default params -> same config object, m = 1 (exact v8.3.3i)')
W['TURN_PARAMS']['params'] = dict(lr=1e-06, kl=0.05, given=['kl', 'lr'], defaults=D, epoch='e1', checkpoint='c' * 64)
out = wrap(None, opt, (), bind, config=cfg)
c = calls[-1]
ok(c['cfg'].kl_coefficient == 0.05 and c['opt_cfg'] is c['cfg'] and c['mult'] == 2.0 and opt.config is cfg and W['TURN']['lr_multiplier'] == 1.0,
   'wrapper: kl 0.05 applied to the update config (optimizer config swapped for the update only), lr multiplier 2.0')
ok(out['config'] == cfg.document() and out['train_params']['kl_applied'] == 0.05 and out['train_params']['lr'] == 1e-06 and out['train_params']['lr_multiplier'] == 2.0
   and out['train_params']['defaults_used'] is False, 'wrapper: signed config reported unchanged, train_params recorded')
W['TURN_PARAMS']['params'] = dict(lr=2.5e-07, kl=0.02, given=['lr'], defaults=D, epoch='e1', checkpoint='c' * 64)
out = wrap(None, opt, (), bind, config=cfg); ok(calls[-1]['cfg'] is cfg and calls[-1]['mult'] == 0.5, 'wrapper: lr only -> m 0.5, config untouched')
W['TURN_PARAMS']['params'] = dict(lr=1e-06, kl=0.05, given=[], defaults=D, epoch='e2', checkpoint='c' * 64)
try: wrap(None, opt, (), bind, config=cfg); ok(False, 'foreign params accepted')
except ValueError: ok(True, 'wrapper: params of another job refused')
W['TURN_PARAMS']['params'] = dict(lr=1e-06, kl=0.05, given=[], defaults=dict(lr=1e-06, kl=0.02), epoch='e1', checkpoint='c' * 64)
try: wrap(None, opt, (), bind, config=cfg); ok(False, 'mismatched defaults accepted')
except ValueError: ok(True, 'wrapper: dispatch defaults != trainer defaults refused')
def boom(*a, **k): raise RuntimeError('x')
W['train_update_v833i'] = boom; W['TURN_PARAMS']['params'] = dict(lr=1e-06, kl=0.05, given=[], defaults=D, epoch='e1', checkpoint='c' * 64)
try: wrap(None, opt, (), bind, config=cfg)
except RuntimeError: pass
ok(opt.config is cfg and W['TURN']['lr_multiplier'] == 1.0, 'wrapper: failure restores optimizer config and multiplier')
print('ALL PASS'); shutil.rmtree(tmp)
