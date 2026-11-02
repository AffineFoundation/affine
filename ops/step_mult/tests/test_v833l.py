"""CPU tests for trainer host v8.3.3l (real launcher code, toy tensors named like the MiMo tensors): optional per-turn knobs.
1. no knobs / all-default knobs -> bit-identical to v8.3.3k (bf16 params, fp32 masters, stats) for lr x step_mult combos
2. knob semantics (group lr scales, freeze, early/late, clamp modes, eps / wd) 3. loss knobs (v8211_batched._apply_knobs, gradient
weight) 4. ff_trainer.train_params 5. wrapper (exact source, stubbed inner update) 6. DP broadcast. RA/20261102-step-mult.
usage: test_v833l.py <v833k host dir> <v833l host dir>"""
import ast, importlib.util, sys, math, shutil, tempfile, threading, types, dataclasses, textwrap
from pathlib import Path
import torch
OLDP = Path(sys.argv[1]); NEWP = Path(sys.argv[2])
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
OLD = load(OLDP / 'mimo_host_trainer_v83.py', 'old_k'); NEW = load(NEWP / 'mimo_host_trainer_v83.py', 'new_l')
tmp = Path(tempfile.mkdtemp()); OLD.FP32_STORE = str(tmp / 'old'); NEW.FP32_STORE = str(tmp / 'new'); OLD.FP32_SHARD_BYTES = NEW.FP32_SHARD_BYTES = 4096; OLD.FP32_SAVE_DEFER_MAX = NEW.FP32_SAVE_DEFER_MAX = 0
EV = []; emit = EV.append
ok = lambda c, m: print(('PASS ' if c else 'FAIL ') + m, flush=True) or (c or sys.exit(1))
class Opt:
    def __init__(self, params):
        self.parameters = tuple(params)
        self.masters = {n: torch.nn.Parameter(p.detach().float().clone()) for n, p in self.parameters}
        self.global_step = 0; self.failed = False
    def zero_grad(self): pass
P = 'model.language_model.layers.'
NAMES = [(P + '0.self_attn.q_proj.weight', (64, 32), 1.0), (P + '0.linear_attn.in_proj_qkv.weight', (48, 32), 1.0), (P + '0.mlp.up_proj.weight', (96, 32), 0.01),
         (P + '1.mlp.down_proj.weight', (32, 96), 1.0), (P + '1.self_attn.o_proj.weight', (32, 64), 0.01), (P + '1.input_layernorm.weight', (32,), 1.0),
         ('model.language_model.embed_tokens.weight', (128, 32), 1.0), ('lm_head.weight', (128, 32), 0.01), (P + '0.linear_attn.A_log', (8,), 0.0)]
def model(seed):
    g = torch.Generator().manual_seed(seed)
    big = ('1.mlp.down_proj', '1.self_attn.o_proj', 'lm_head', 'linear_attn.in_proj_qkv')
    return [(n, ((torch.ones(s) + 0.01 * torch.randn(s, generator=g)) if len(s) == 1 else torch.randn(s, generator=g) * (50.0 if any(b in n for b in big) else 0.5)).bfloat16()) for n, s, _ in NAMES]
def grads(seed, params):
    g = torch.Generator().manual_seed(seed); sc = {n: x for n, _, x in NAMES}
    return {n: torch.randn(p.shape, generator=g) * sc[n] for n, p in params}
G = grads(7, model(1))
LR, CAP, TRC = 2e-3, 1.0, 1e-3
ZERO = P + '0.linear_attn.A_log'
def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items() if 'seconds' not in k and k != 'store'}
    return x
def wait():
    for t in threading.enumerate():
        if t.name == 'v833f-fp32-save': t.join()
def step(mod, mult=1.0, k=1, opt=None, cap=CAP):
    p = model(1); o = Opt(p); M0 = {n: o.masters[n].detach().clone() for n, _ in p}
    mod.TURN['lr_multiplier'] = mult; mod.TURN['step_mult'] = k
    if mod is NEW: NEW.TURN['opt'] = opt
    try:
        r = mod.sgd_step(o, {kk: v.clone() for kk, v in G.items()}, LR, cap, 'fresh-adam', (), emit, TRC, fp32_master=True, direction='normgrad'); wait()
    finally:
        mod.TURN['lr_multiplier'] = 1.0; mod.TURN['step_mult'] = 1
        if mod is NEW: NEW.TURN['opt'] = None
    return p, o, M0, r, dict(mod.TURN['last'])
def D(o, M0): return {n: (M0[n].double() - o.masters[n].detach().double()) for n in M0}
# 1. bit identity
for mult, k in ((1.0, 1), (0.5, 1), (2.0, 1), (1.0, 2), (2.0, 4), (0.5, 8)):
    pa, oa, _, ra, la = step(OLD, mult, k); pb, ob, _, rb, lb = step(NEW, mult, k)
    ok(all(torch.equal(x, y) for (_, x), (_, y) in zip(pa, pb)) and all(torch.equal(oa.masters[n], ob.masters[n]) for n in oa.masters)
       and clean(ra['sgd']) == clean(rb['sgd']) and la == lb, 'no knobs, m=%g k=%d: bf16 + fp32 master + stats bit-identical to v8.3.3k' % (mult, k))
DEF = {k: v for k, v in NEW.KNOB_DEFAULTS.items() if k in ('lr_scale_attn', 'lr_scale_mlp', 'lr_scale_embed', 'lr_scale_lm_head', 'lr_scale_early', 'lr_scale_late', 'adam_eps', 'weight_decay', 'clamp_mode', 'global_norm_cap')}
# 2. knob semantics (TURN['opt'] set directly; the wrapper only sets it when a knob is not default)
_, o0, M0, r0, l0 = step(NEW, opt=None); D0 = D(o0, M0)
wn = {n: math.sqrt(float(M0[n].double().pow(2).sum())) for n in M0}
atcap = [n for n in M0 if n != ZERO and abs(float(D0[n].norm()) - TRC * wn[n]) <= 1e-3 * TRC * wn[n]]
uncap = [n for n in M0 if n != ZERO and n not in atcap]
ok(atcap and uncap, 'toy model has capped %d and uncapped %d tensors' % (len(atcap), len(uncap)))
_, o, _, r, l = step(NEW, opt=dict(DEF)); Dd = D(o, M0)
ok(max(float((Dd[n] - D0[n]).abs().max()) / max(float(D0[n].abs().max()), 1e-30) for n in M0 if n != ZERO) < 1e-5 and l['path'].endswith('+knobs'),
   'knob path forced with default values == default step (fp32 rounding only)')
_, o, _, r, l = step(NEW, opt=dict(DEF, lr_scale_attn=0.0)); Dk = D(o, M0)
attn = [n for n in M0 if '.self_attn.' in n or '.linear_attn.' in n]
ok(all(torch.equal(o.masters[n].detach(), M0[n]) for n in attn) and all(float((Dk[n] - D0[n]).abs().max()) <= 1e-5 * float(D0[n].abs().max()) for n in M0 if n not in attn and n != ZERO)
   and l['frozen_tensors'] >= len(attn), 'lr_scale_attn 0: attention tensors (self_attn + linear_attn) frozen exactly, others unchanged')
_, o, _, r, l = step(NEW, opt=dict(DEF, lr_scale_mlp=0.5)); Dk = D(o, M0)
mlp = [n for n in M0 if '.mlp.' in n]
ok(all(float((Dk[n] - 0.5 * D0[n]).abs().max()) <= 5e-3 * float(D0[n].abs().max()) for n in mlp), 'lr_scale_mlp 0.5: mlp steps halve (capped and uncapped)')
_, o, _, r, l = step(NEW, opt=dict(DEF, lr_scale_mlp=2.0)); Dk = D(o, M0)
ok(all(float(Dk[n].norm()) <= TRC * wn[n] * (1 + 1e-3) for n in mlp) and all(float((Dk[n] - 2 * D0[n]).abs().max()) <= 5e-3 * float(D0[n].abs().max()) for n in mlp if n in uncap and 2 * float(D0[n].norm()) < TRC * wn[n]),
   'lr_scale_mlp 2: mlp grows only under the per-tensor cap (like lr)')
_, o, _, r, l = step(NEW, opt=dict(DEF, lr_scale_early=0.0)); 
ok(all(torch.equal(o.masters[n].detach(), M0[n]) for n in M0 if '.layers.0.' in n) and not torch.equal(o.masters[P + '1.mlp.down_proj.weight'].detach(), M0[P + '1.mlp.down_proj.weight']),
   'lr_scale_early 0: layers 0..half-1 frozen, later layers move')
_, o, _, r, l = step(NEW, opt=dict(DEF, lr_scale_embed=0.0, lr_scale_lm_head=0.0))
ok(torch.equal(o.masters['lm_head.weight'].detach(), M0['lm_head.weight']) and torch.equal(o.masters['model.language_model.embed_tokens.weight'].detach(), M0['model.language_model.embed_tokens.weight']), 'lr_scale_embed / lm_head 0: frozen')
_, o, _, r, l = step(NEW, opt=dict(DEF, clamp_mode='global_norm', global_norm_cap=1.0)); Dk = D(o, M0)
ok(any(float(Dk[n].norm()) > 1.5 * TRC * wn[n] for n in atcap) and not l['cap_fired'], 'clamp global_norm: no per-tensor cap (capped tensors move past c_t), global cap not fired at 1.0')
G2 = 0.5 * l['raw_update_norm']
_, o, _, r, l = step(NEW, opt=dict(DEF, clamp_mode='global_norm', global_norm_cap=G2))
ok(l['cap_fired'] and abs(l['update_norm_master'] - G2) / G2 < 1e-3 and l['global_cap_effective'] == G2, 'clamp global_norm cap %.4g: fires, step norm == cap' % G2)
G3 = 0.5 * l0['raw_update_norm']
_, o, _, r, l = step(NEW, opt=dict(DEF, clamp_mode='both', global_norm_cap=G3)); Dk = D(o, M0)
ok(l['cap_fired'] and abs(l['update_norm_master'] - G3) / G3 < 1e-3 and all(float((Dk[n] - 0.5 * D0[n]).abs().max()) <= 5e-3 * float(D0[n].abs().max()) for n in M0 if n != ZERO),
   'clamp both: per-tensor caps then global cap (half the default norm -> every tensor exactly halves)')
_, o, _, r, l = step(NEW, opt=dict(DEF, clamp_mode='per_tensor', global_norm_cap=0.01)); Dk = D(o, M0)
ok(not l['cap_fired'] or l['global_cap_effective'] == CAP, 'clamp per_tensor ignores global_norm_cap (fixed safety cap %g)' % CAP)
_, o, _, r, l = step(NEW, opt=dict(DEF, adam_eps=1e-3, weight_decay=0.0)); Dk = D(o, M0)
def expect_norm(n, eps, wd):
    g = G[n]; w = M0[n].float(); e = g / (g.abs() + eps)
    if w.dim() >= 2 and not n.endswith('bias'): e = e + wd * w
    return min(LR * float(e.double().norm()), TRC * (1 - 1e-6) * wn[n])
ok(all(abs(float(Dk[n].norm()) - expect_norm(n, 1e-3, 0.0)) <= 5e-3 * expect_norm(n, 1e-3, 0.0) for n in M0 if n != ZERO)
   and all(float((Dk[n] * D0[n]).sum() / (Dk[n].norm() * D0[n].norm())) > 1 - 1e-6 for n in M0 if n != ZERO)
   and any(abs(float(Dk[n].norm()) - float(D0[n].norm())) > 1e-2 * float(D0[n].norm()) for n in M0 if n != ZERO),
   'adam_eps 1e-3 / weight_decay 0: per-tensor norm = min(||lr (g/(|g|+eps) + wd W)||, c_t); direction g/||g|| unchanged')
_, o, _, r, l = step(NEW, k=2, opt=dict(DEF, lr_scale_mlp=0.5)); Dk = D(o, M0)
ok(all(float((Dk[n] - D0[n]).abs().max()) <= 5e-3 * float(D0[n].abs().max()) for n in mlp) and all(float((Dk[n] - 2 * D0[n]).abs().max()) <= 5e-3 * float(D0[n].abs().max()) for n in M0 if n not in mlp and n != ZERO),
   'step_mult 2 x lr_scale_mlp 0.5: mlp unchanged, others 2x')
pq, oq, _, _, _ = step(NEW, opt=dict(DEF, lr_scale_attn=1.7, adam_eps=1e-6))
ok(all(torch.equal(x, oq.masters[n].detach().bfloat16()) for n, x in pq), 'knobs: bf16 candidate == bf16_rne(master)')
# 3. loss knobs
B_SRC = (NEWP / 'v8211_batched.py').read_text(); i = B_SRC.index('def _apply_knobs'); j = B_SRC.index('\nDROP_UNRESOLVED')
OLD_B = (OLDP / 'v8211_batched.py').read_text()
removed = OLD_B.replace(B_SRC[B_SRC.index('    if KNOBS is not None:          # v8.3.3l'):B_SRC.index('    _prompt_weights(rows)\n    return rows\n\n\nKNOBS')], '')
ok(B_SRC.replace(B_SRC[B_SRC.index('    if KNOBS is not None:          # v8.3.3l'):B_SRC.index('    _prompt_weights(rows)\n    return rows\n\n\nKNOBS')], '', 1)
   .replace(B_SRC[B_SRC.index('\n\nKNOBS = None'):j], '', 1).replace("    if KNOBS is not None:\n        raise ValueError('v8.3.3l loss knobs need drop_unresolved off')\n", '', 1)
   .replace("            if row.get('agg_w') is not None:           # v8.3.3l loss_agg token_mean (absent -> unchanged)\n                loss = loss * row['agg_w']\n", '', 1) == OLD_B,
   'v8211_batched: v8.3.3k source + only the KNOBS-guarded additions (KNOBS None and no agg_w -> the same statements run)')
import numpy as np
def ref_adv(rewards, mask, index, norm_adv_by_std_in_grpo=True, epsilon=1e-6):   # pinned verl semantics (1-rollout group: mean 0, std 1)
    s = rewards.sum(-1); out = s.clone()
    if len(s) == 1: return out.unsqueeze(-1), out.unsqueeze(-1)
    m, sd = s.mean(), s.std()
    out = (s - m) / (sd + epsilon) if norm_adv_by_std_in_grpo else s - m
    return out.unsqueeze(-1), out.unsqueeze(-1)
ns = dict(torch=torch, np=np, reference=lambda: types.SimpleNamespace(compute_grpo_outcome_advantage=ref_adv)); exec(B_SRC[i:j], ns)
def rows():
    g1 = types.SimpleNamespace(task_id='a'); g2 = types.SimpleNamespace(task_id='b')
    R = [(g1, 1.0, 10), (g1, 0.0, 30), (g1, 0.0, 20), (g2, 1.0, 40)]
    m1 = (1.0 + 0 + 0) / 3
    return [dict(group=g, rollout=types.SimpleNamespace(tokens=[0] * n), reward=r, advantage=(r - m1) if g is g1 else r, active=True) for g, r, n in R]
ns['KNOBS'] = dict(loss_agg='seq_mean', adv_center='none', adv_clip=None); x = rows(); ns['_apply_knobs'](x)
ok([r['advantage'] for r in x] == [1.0, 0.0, 0.0, 1.0] and all('agg_w' not in r for r in x), 'adv_center none: A = r')
ns['KNOBS'] = dict(loss_agg='seq_mean', adv_center='group_norm', adv_clip=None); x = rows(); ns['_apply_knobs'](x)
sd = float(torch.tensor([1.0, 0, 0]).std())
ok(abs(x[0]['advantage'] - (2 / 3) / (sd + 1e-6)) < 1e-6 and x[3]['advantage'] == 1.0, 'adv_center group_norm: (r - mean) / (std + 1e-6), 1-rollout group A = r')
ns['KNOBS'] = dict(loss_agg='token_mean', adv_center='mean', adv_clip=0.5); x = rows(); a0 = [r['advantage'] for r in rows()]; ns['_apply_knobs'](x)
ok([r['advantage'] for r in x] == [max(-0.5, min(0.5, a)) for a in a0] and abs(sum(r['agg_w'] for r in x) - 4) < 1e-12 and abs(x[3]['agg_w'] - 4 * 40 / 100) < 1e-12,
   'adv_clip 0.5 clamps; loss_agg token_mean: weight N * len_i / sum len (sum = N)')
# 4. ff_trainer.train_params
FT_SRC = (NEWP / 'ff_trainer.py').read_text(); i = FT_SRC.index('def need'); j = FT_SRC.index('\n@dataclass')
ns = {}; exec("TRAIN_BOUNDS = {'lr': (1.25e-07, 1e-06), 'kl': (0.005, 0.1)}\nSTEP_MULT_ALLOWED = (1, 2, 3, 4, 5, 6, 7, 8)\n" + FT_SRC[i:j], ns)
kk = FT_SRC.index('KNOB_SPEC = {'); l = FT_SRC.index('\ndef prepare'); exec(FT_SRC[kk:l], ns); tp = ns['train_params']; KD = ns['KNOB_DEFAULTS']
Bd = types.SimpleNamespace(epoch='e1', checkpoint='c' * 64); Dd = dict(lr=5e-07, kl=0.02)
got = tp(dict(train=dict(lr=5e-07, kl=0.02, step_mult=1, given=[], defaults=dict(Dd, step_mult=1))), Bd)
ok(got['knobs'] == KD, 'train_params: dispatch without knobs -> all defaults')
got = tp(dict(train=dict(lr=5e-07, kl=0.02, step_mult=1, knobs=dict(KD, lr_scale_attn=0.0, clamp_mode='both', adv_clip=2.0), given=['lr_scale_attn'], defaults=dict(Dd, step_mult=1))), Bd)
ok(got['knobs']['clamp_mode'] == 'both' and got['knobs']['adv_clip'] == 2.0, 'train_params: knobs accepted')
for bad in (dict(KD, lr_scale_attn=3.0), dict(KD, clamp_mode='x'), dict(KD, adam_eps=1), dict(KD, adv_clip=0.0), {k: v for k, v in KD.items() if k != 'loss_agg'}, dict(KD, foo=1.0)):
    try: tp(dict(train=dict(lr=5e-07, kl=0.02, knobs=bad, given=[], defaults=Dd)), Bd); ok(False, 'bad knobs accepted %r' % bad)
    except ValueError: pass
ok(True, 'train_params: out-of-range / unknown / missing / int-typed knobs refused')
# 5. wrapper
src = (NEWP / 'mimo_host_trainer_v83.py').read_text(); tree = ast.parse(src)
inst = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install'][0]
fn = [n for n in ast.walk(inst) if isinstance(n, ast.FunctionDef) and n.name == 'train_update_v833j'][0]
code = textwrap.dedent(ast.get_source_segment(src, fn))
@dataclasses.dataclass(frozen=True)
class Cfg:
    kl_coefficient: float = 0.02
    def validate(self): return self
    def document(self): return dict(settings=dict(kl_coefficient=self.kl_coefficient))
calls = []; BT = types.SimpleNamespace(KNOBS=None, DROP_UNRESOLVED=False)
def inner(runtime, optimizer, groups, binding, *, config, **kw):
    calls.append(dict(opt=W['TURN']['opt'], knobs=BT.KNOBS, k=W['TURN']['step_mult']))
    W['TURN']['last'] = dict(path='x'); return dict(status='updated', config=config.document())
W = dict(TURN=dict(lr_multiplier=1.0, step_mult=1, opt=None, last=None), TURN_PARAMS=dict(params=None), train_update_v833i=inner, _dc=dataclasses,
         sgd_lr=5e-07, sgd_mode='fresh-adam', sgd_fp32=True, sgd_dir='normgrad', sgd_trc=2e-05, batched=BT,
         KNOB_SPEC=NEW.KNOB_SPEC, KNOB_DEFAULTS=NEW.KNOB_DEFAULTS, knob_value=NEW.knob_value)
exec(code, W); wrap = W['train_update_v833j']
cfg = Cfg(); opt = types.SimpleNamespace(config=cfg); bind = types.SimpleNamespace(epoch='e1', checkpoint='c' * 64)
base = dict(lr=5e-07, kl=0.02, step_mult=1, given=[], defaults=Dd, epoch='e1', checkpoint='c' * 64)
W['TURN_PARAMS']['params'] = dict(base); out = wrap(None, opt, (), bind, config=cfg)
ok(calls[-1]['opt'] is None and calls[-1]['knobs'] is None and out['train_params']['defaults_used'] is True and out['train_params']['knobs'] == NEW.KNOB_DEFAULTS, 'wrapper: no knobs -> TURN.opt None, KNOBS None, defaults_used')
W['TURN_PARAMS']['params'] = dict(base, knobs=dict(NEW.KNOB_DEFAULTS), given=list(NEW.KNOB_SPEC)); out = wrap(None, opt, (), bind, config=cfg)
ok(calls[-1]['opt'] is None and calls[-1]['knobs'] is None and out['train_params']['defaults_used'] is True, 'wrapper: every knob given at its default -> same path (opt None, KNOBS None)')
W['TURN_PARAMS']['params'] = dict(base, knobs=dict(NEW.KNOB_DEFAULTS, global_norm_cap=0.01)); out = wrap(None, opt, (), bind, config=cfg)
ok(calls[-1]['opt'] is None, 'wrapper: global_norm_cap with clamp_mode per_tensor has no effect -> default path')
W['TURN_PARAMS']['params'] = dict(base, knobs=dict(NEW.KNOB_DEFAULTS, lr_scale_mlp=0.5, adv_center='none')); out = wrap(None, opt, (), bind, config=cfg)
ok(calls[-1]['opt']['lr_scale_mlp'] == 0.5 and calls[-1]['knobs'] == dict(loss_agg='seq_mean', adv_center='none', adv_clip=None) and W['TURN']['opt'] is None and BT.KNOBS is None
   and out['train_params']['knobs_applied'] == dict(step=True, loss=True) and out['train_params']['defaults_used'] is False, 'wrapper: knobs applied for the update only, recorded, reset after')
for badkn in (dict(NEW.KNOB_DEFAULTS, lr_scale_mlp=5.0), dict(NEW.KNOB_DEFAULTS, loss_agg='x'), {'adam_eps': 1e-8}):
    W['TURN_PARAMS']['params'] = dict(base, knobs=badkn)
    try: wrap(None, opt, (), bind, config=cfg); ok(False, 'bad knobs accepted')
    except ValueError: pass
ok(True, 'wrapper: bad knobs refused')
BT.DROP_UNRESOLVED = True; W['TURN_PARAMS']['params'] = dict(base, knobs=dict(NEW.KNOB_DEFAULTS, adv_clip=1.0))
try: wrap(None, opt, (), bind, config=cfg); ok(False, 'loss knobs with drop_unresolved accepted')
except ValueError: ok(True, 'wrapper: loss knobs refused under drop_unresolved')
BT.DROP_UNRESOLVED = False
def boom(*a, **k): raise RuntimeError('x')
W['train_update_v833i'] = boom; W['TURN_PARAMS']['params'] = dict(base, knobs=dict(NEW.KNOB_DEFAULTS, lr_scale_attn=0.0, adv_clip=1.0))
try: wrap(None, opt, (), bind, config=cfg)
except RuntimeError: pass
ok(W['TURN']['opt'] is None and BT.KNOBS is None, 'wrapper: failure resets TURN.opt and KNOBS')
# 6. DP broadcast
DP = (NEWP / 'v83_dp.py').read_text()
ok("knobs=getattr(B, 'KNOBS', None)" in DP and "B.KNOBS = job.get('knobs')" in DP, 'v83_dp: KNOBS broadcast to helpers and set before their _layout')
print('ALL PASS'); shutil.rmtree(tmp)
