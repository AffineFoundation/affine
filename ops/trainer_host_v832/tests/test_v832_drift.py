"""CPU sanity test for v8.3.2 drift control (synthetic tensors, real CPUAdamW + real v8.3.1 PipelinedCPUAdamW step).
usage: python test_v832_drift.py <source_root> <stage_dir> <settings.json>   (+ rank-agreement test via gloo world 8)"""
import sys, json, hashlib, math, os, importlib.util, copy
SRC, STAGE, SETTINGS = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, SRC); sys.path.insert(0, STAGE)
import torch
torch.manual_seed(0)
from subnet.mimo_grpo.config import TrainerConfig
from subnet.mimo_grpo.optimizer import CPUAdamW
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
L = load('launcher832', os.path.join(STAGE, 'mimo_host_trainer_v83.py'))
FO = load('fast_opt', os.path.join(STAGE, 'v824_optimizer_fast.py'))
cfg = TrainerConfig(**json.load(open(SETTINGS)))
CK = 'a' * 64
SHAPES = {'layers.0.mlp.weight': (256, 512), 'layers.0.gate.weight': (64, 256), 'layers.0.norm.weight': (512,), 'layers.0.tiny.bias': (32,), 'embed.weight': (1000, 64)}
SCALES = {'layers.0.mlp.weight': 0.02, 'layers.0.gate.weight': 3e-4, 'layers.0.norm.weight': 1.0, 'layers.0.tiny.bias': 1e-3, 'embed.weight': 0.05}
def model(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(n, torch.nn.Parameter((torch.randn(s, generator=g) * SCALES[n]).to(torch.bfloat16))) for n, s in SHAPES.items()]
def grads(step, seed=100):
    g = torch.Generator().manual_seed(seed + step)
    return {n: torch.randn(s, generator=g) * 1e-3 for n, s in SHAPES.items()}
def make(params):
    o = CPUAdamW(params, cfg, input_checkpoint=CK); o.step_threads = 4
    peak = max(p.numel() for _, p in o.parameters)
    o._pinned = [torch.empty(peak, dtype=torch.float32) for _ in range(2)]; o._copy_stream = None
    return o
inner = FO.PipelinedCPUAdamW.step
H = lambda o: hashlib.sha256(b''.join(o.masters[n].detach().numpy().tobytes() for n, _ in o.parameters)).hexdigest()
rel = lambda a, b: math.sqrt(sum(float(((a[n] - b[n]).double() ** 2).sum()) for n in a)) / math.sqrt(sum(float((b[n].double() ** 2).sum()) for n in b))
snap = lambda o: {n: o.masters[n].detach().clone() for n, _ in o.parameters}
res = {}; rows = []
emit = rows.append
# 1. flags OFF path == v8.3.1 bitwise (the launcher calls `pipelined` directly when OFF; controlled_step with cap None, scale 1.0 must also be identical)
a, b = make(model()), make(model())
for k in range(3):
    inner(a, grads(k)); controlled_step_out = L.controlled_step(b, grads(k), inner, None, 3, 1.0, emit=emit)
res['off_bitwise_equal_to_v831'] = H(a) == H(b)
# 2. restart LR: fresh optimizer -> factor .5 for steps 1-3, 1.0 at step 4; first update exactly half of the reference sign-like step
a, b = make(model()), make(model()); rows.clear()
a0 = snap(a); inner(a, grads(0)); da = {n: a.masters[n].detach() - a0[n] for n in a0}
b0 = snap(b); L.controlled_step(b, grads(0), inner, None, 3, 0.5, emit=emit); db = {n: b.masters[n].detach() - b0[n] for n in b0}
ratio = math.sqrt(sum(float((db[n].double() ** 2).sum()) for n in db)) / math.sqrt(sum(float((da[n].double() ** 2).sum()) for n in da))
for k in range(1, 5): L.controlled_step(b, grads(k), inner, None, 3, 0.5, emit=emit)
res['restart_lr'] = dict(first_step_update_ratio=ratio, factors=[r['lr_factor'] for r in rows], fresh_detected=rows[0]['fresh_moments_at_start'],
                         lr_restored=all(g['lr'] == cfg.learning_rate for g in b.optimizer.param_groups))
# 3. warm optimizer (non-zero moments) -> no LR scaling
w = make(model()); inner(w, grads(0)); rows.clear(); w2 = w   # moments now non-zero; new 'process' view = fresh check on this object
w2._v832_checked = False
L.controlled_step(w2, grads(1), inner, None, 3, 0.5, emit=emit)
res['warm_not_scaled'] = dict(fresh_detected=rows[0]['fresh_moments_at_start'], lr_factor=rows[0]['lr_factor'])
# 4. drift cap 2e-5: per tensor and global bound, bf16 params == masters projection, Adam moments untouched by the cap
cap = 2e-5
for kind, seq in (('cold', [0]), ('warm3', [0, 1, 2])):
    c = make(model()); u = make(model()); rows.clear()
    for k in seq[:-1]:
        L.controlled_step(c, grads(k), inner, cap, 3, 0.5, emit=emit); L.controlled_step(u, grads(k), inner, None, 3, 0.5, emit=lambda r: None)
    c0 = snap(c)
    L.controlled_step(c, grads(seq[-1]), inner, cap, 3, 0.5, emit=emit); L.controlled_step(u, grads(seq[-1]), inner, None, 3, 0.5, emit=lambda r: None)
    r = rows[-1]; c1 = snap(c)
    per = {n: math.sqrt(float(((c1[n] - c0[n]).double() ** 2).sum())) / math.sqrt(float((c0[n].double() ** 2).sum())) for n in c0}
    proj_ok = all(torch.equal(p.detach(), c.masters[n].detach().to(torch.bfloat16)) for n, p in c.parameters)
    mom_ok = all(torch.equal(c.optimizer.state[c.masters[n]]['exp_avg'], u.optimizer.state[u.masters[n]]['exp_avg']) for n, _ in c.parameters) if kind == 'cold' else None
    res['cap_' + kind] = dict(pre_cap_rel=r['pre_cap_rel'], post_cap_rel=r['post_cap_rel'], measured_global_rel=rel(c1, c0), max_tensor_rel=max(per.values()),
                             tensors_capped=r['tensors_capped'], global_scale=r['global_scale'], cap_fired=r['cap_fired'], bf16_projection_ok=proj_ok,
                             moments_equal_uncapped_twin=mom_ok, lr_factor=r['lr_factor'],
                             bound_ok=max(per.values()) <= cap and rel(c1, c0) <= cap)
# 4b. global cap binds (cap 3e-6 < global pre-cap rel)
c = make(model()); rows.clear(); c0 = snap(c)
L.controlled_step(c, grads(0), inner, 3e-6, 3, 0.5, emit=emit); r = rows[-1]; c1 = snap(c)
per = {n: math.sqrt(float(((c1[n] - c0[n]).double() ** 2).sum())) / math.sqrt(float((c0[n].double() ** 2).sum())) for n in c0}
res['cap_global'] = dict(pre_cap_rel=r['pre_cap_rel'], post_cap_rel=r['post_cap_rel'], measured_global_rel=rel(c1, c0), max_tensor_rel=max(per.values()),
                         global_scale=r['global_scale'], tensors_capped=r['tensors_capped'], bound_ok=rel(c1, c0) <= 3e-6 and max(per.values()) <= 3e-6)
print(json.dumps(dict(sample_log_row=r), default=str))
# 5. determinism: two identical runs -> identical master bytes
h = []
for _ in range(2):
    o = make(model())
    for k in range(4): L.controlled_step(o, grads(k), inner, cap, 3, 0.5, emit=lambda r: None)
    h.append(H(o))
res['deterministic'] = h[0] == h[1]
# 6. a loose cap (1.0) never fires and equals the uncapped run bitwise
a, b = make(model()), make(model()); rows.clear()
L.controlled_step(a, grads(0), inner, 1.0, 3, 0.5, emit=emit); L.controlled_step(b, grads(0), inner, None, 3, 0.5, emit=lambda r: None)
res['loose_cap_noop'] = dict(cap_fired=rows[-1]['cap_fired'], bitwise=H(a) == H(b))
# 7. drift_settings validation
bad = []
for e in ({'drift_cap_rel': 0.0}, {'drift_cap_rel': 2}, {'restart_lr_warm_steps': -1}, {'restart_lr_scale': 0.0}, {'restart_lr_scale': 1}):
    try: L.drift_settings(e); bad.append(e)
    except ValueError: pass
res['settings_validation'] = dict(off_default=L.drift_settings({}), prod=L.drift_settings(dict(drift_cap_rel=2e-5, restart_lr_warm_steps=3, restart_lr_scale=0.5)), accepted_bad=bad)
print(json.dumps(res, indent=1, default=str))
ok = (res['off_bitwise_equal_to_v831'] and abs(res['restart_lr']['first_step_update_ratio'] - 0.5) < 0.02 and res['restart_lr']['factors'] == [0.5, 0.5, 0.5, 1.0, 1.0]
      and res['restart_lr']['fresh_detected'] and res['restart_lr']['lr_restored'] and res['warm_not_scaled']['lr_factor'] == 1.0
      and all(res['cap_' + k]['bound_ok'] and res['cap_' + k]['cap_fired'] and res['cap_' + k]['bf16_projection_ok'] for k in ('cold', 'warm3'))
      and res['cap_cold']['moments_equal_uncapped_twin'] and res['cap_global']['bound_ok'] and res['deterministic'] and not res['loose_cap_noop']['cap_fired'] and res['loose_cap_noop']['bitwise']
      and not res['settings_validation']['accepted_bad'])
print('ALL PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
