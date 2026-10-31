#!/usr/bin/env python3
"""RA/20261031-renorm-step-live: mimo_host_trainer_v83.py v8.3.3h -> v8.3.3i (normgrad direction + support renorm wiring)."""
import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text()
def rep(old, new, n=1):
    global s
    c = s.count(old)
    if c != n: raise SystemExit('anchor count %d != %d: %r' % (c, n, old[:80]))
    s = s.replace(old, new)

# A. settings block after fp32_settings()
rep("""def _sha_tensor(t):""", '''# v8.3.3i (RA/20261031-renorm-step-live; Jacob 2026-10-31 07:00 CR; Allan catoneone/sn120-k1-fixes@230b111). OPTIONAL engine keys;
# absent = exact v8.3.3h behaviour.
#  sgd_direction 'normgrad' (fresh-adam + sgd_fp32_master only): every tensor keeps the norm of today's capped fresh-AdamW step
#     (d_t, r_t, s exactly as fresh-adam, computed from the fp32 head master M_H), only the direction changes from ~sign(g_t) to g_t/||g_t||:
#     Delta_t = fp32(s * r_t * ||d_t|| / ||g_t||) * g_t   (fp64 norms; Delta_t = 0 if ||g_t|| = 0);  M_C = M_H - Delta_t;  W_C = bf16_rne(M_C)
#     so ||Delta_t|| = s * r_t * ||d_t|| and sqrt(sum_t ||Delta_t||^2) = raw_update_norm (multiplier 1x, lr and caps unchanged).
#  support_renorm {top_k, top_p, temperature}: sets v8211_batched.SUPPORT_RENORM (Allan patch/support_renorm.diff, applied as written): the PG
#     term uses log q over the trainer's own top-k -> top-p support; sampled tokens outside it leave the PG token mean but stay in KL and
#     derail. Asserted equal to the job's signed sampling settings (temperature/top_k/top_p) at every update. Per-turn summary:
#     training.support_renorm (outside-support token fraction) in the trainer report.
DIR_KEYS = {'sgd_direction', 'support_renorm'}
SGD_DIRECTIONS = ('sign', 'normgrad')
SUPPORT_KEYS = ('top_k', 'top_p', 'temperature')


def direction_settings(engine):
    d = engine.get('sgd_direction', 'sign')
    if d not in SGD_DIRECTIONS:
        raise ValueError('sgd_direction must be sign or normgrad')
    if d == 'normgrad' and not (engine.get('sgd_mode') == 'fresh-adam' and engine.get('sgd_fp32_master') is True):
        raise ValueError('sgd_direction normgrad needs sgd_mode fresh-adam and sgd_fp32_master true')
    return d


def support_settings(engine):
    s = engine.get('support_renorm')
    if s is None:
        return None
    if not (type(s) is dict and set(s) == set(SUPPORT_KEYS) and type(s['top_k']) is int and 1 <= s['top_k'] <= 1024
            and type(s['top_p']) is float and 0 < s['top_p'] <= 1.0 and type(s['temperature']) is float and s['temperature'] > 0):
        raise ValueError('support_renorm must be {top_k: int, top_p: float, temperature: float}')
    return dict(top_k=s['top_k'], top_p=s['top_p'], temperature=s['temperature'])


def support_summary(trajectories, support):
    """Per-turn outside-support statistics from the per-rollout trainer details (support_outside_tokens over kept loss tokens)."""
    rows = [t for t in trajectories if t.get('support_outside_tokens') is not None]
    out = sum(int(t['support_outside_tokens']) for t in rows); tok = sum(int(t.get('loss_tokens') or 0) for t in rows)
    fr = [(t['support_outside_tokens'] / t['loss_tokens']) if t.get('loss_tokens') else 0.0 for t in rows]
    inside = tok - out
    lq = [t['support_logq_mean'] * (t['loss_tokens'] - t['support_outside_tokens']) for t in rows
          if t.get('support_logq_mean') is not None and t.get('loss_tokens')]
    return dict(version='v833i-support-renorm-v1', settings=dict(support), rollouts=len(rows), loss_tokens=tok, outside_tokens=out,
                outside_fraction=(out / tok) if tok else 0.0, rollouts_with_outside=sum(1 for t in rows if t['support_outside_tokens']),
                max_rollout_outside_fraction=max(fr) if fr else 0.0,
                rollouts_outside_over_0p5pct=sum(1 for f in fr if f > 0.005),
                inside_logq_mean=(sum(lq) / inside) if inside > 0 else None,
                rule='sampled tokens outside the replayed top_k -> top_p support are dropped from the PG token mean only (still in KL and derail); files are never rejected for it')


def _sha_tensor(t):''')

# B. checked(): allow + validate keys
rep("ENGINE_KEYS <= set(config['engine']) <= ENGINE_KEYS | DRIFT_KEYS | SGD_KEYS | FP32_KEYS and",
    "ENGINE_KEYS <= set(config['engine']) <= ENGINE_KEYS | DRIFT_KEYS | SGD_KEYS | FP32_KEYS | DIR_KEYS and")
rep("    drift_settings(config['engine']); sgd_settings(config['engine']); fp32_settings(config['engine'])\n",
    "    drift_settings(config['engine']); sgd_settings(config['engine']); fp32_settings(config['engine'])\n"
    "    direction_settings(config['engine']); support_settings(config['engine'])\n")

# C. signatures
rep("def fresh_adam_step(opt, gradients, live, lr, cap, freeze, emit, t0, tensor_rel_cap=None, fp32_master=False):",
    "def fresh_adam_step(opt, gradients, live, lr, cap, freeze, emit, t0, tensor_rel_cap=None, fp32_master=False, direction='sign'):")
rep("        return _fresh_adam_step(opt, gradients, live, lr, cap, freeze, emit_, t0, tensor_rel_cap, fp32_master)\n",
    "        return _fresh_adam_step(opt, gradients, live, lr, cap, freeze, emit_, t0, tensor_rel_cap, fp32_master, direction)\n")
rep("def _fresh_adam_step(opt, gradients, live, lr, cap, freeze, emit_, t0, tensor_rel_cap, fp32_master):\n    import math, time, torch\n",
    "def _fresh_adam_step(opt, gradients, live, lr, cap, freeze, emit_, t0, tensor_rel_cap, fp32_master, direction='sign'):\n    import math, time, torch\n"
    "    normgrad = direction == 'normgrad'\n    if normgrad and not fp32_master:\n        raise ValueError('normgrad direction needs the fp32 master')\n")
# D. first loop: keep q and ||g||^2
rep("""        dn2 = []; r = {}
        for name, parameter in live:
            g = gradients[name].detach().to(device=parameter.device); w32 = W32(name, parameter)
            q = float(_fresh_adam_delta(name, w32, g, lr).double().pow(2).sum())
""", """        dn2 = []; r = {}; qd = {}; gn2 = {}
        for name, parameter in live:
            g = gradients[name].detach().to(device=parameter.device); w32 = W32(name, parameter)
            q = float(_fresh_adam_delta(name, w32, g, lr).double().pow(2).sum())
            if normgrad:
                qd[name] = q; gn2[name] = float(g.double().pow(2).sum())
""")
# second loop, fp32 master branch
rep("""                    d = _fresh_adam_delta(name, w32, g, lr)
                    d.mul_(torch.tensor(scale * r[name], dtype=torch.float32, device=parameter.device))
""", """                    if normgrad:         # v8.3.3i: same per-tensor norm s * r_t * ||d_t||, direction g_t / ||g_t||
                        c = (scale * r[name] * math.sqrt(qd[name]) / math.sqrt(gn2[name])) if gn2[name] > 0 else 0.0
                        d = g * torch.tensor(c, dtype=torch.float32, device=parameter.device)
                    else:
                        d = _fresh_adam_delta(name, w32, g, lr)
                        d.mul_(torch.tensor(scale * r[name], dtype=torch.float32, device=parameter.device))
""")
# F. info
rep("""                tensors_rel_capped=sum(v < 1.0 for v in r.values()), sgd_freeze=list(freeze), frozen_tensors=len(opt.parameters) - len(live),
                raw_update_norm=raw,""", """                tensors_rel_capped=sum(v < 1.0 for v in r.values()), sgd_freeze=list(freeze), frozen_tensors=len(opt.parameters) - len(live),
                sgd_direction=direction, zero_gradient_tensors=(sum(1 for v in gn2.values() if v == 0) if normgrad else None),
                raw_update_norm=raw,""")
if normgrad_impl := True:
    rep("""    info = dict(version='v833-fresh-adam-v1' if not fp32_master else 'v833f-fresh-adam-fp32-master-v1',""",
        """    info = dict(version='v833-fresh-adam-v1' if not fp32_master else ('v833f-fresh-adam-fp32-master-v1' if not normgrad else 'v833i-normgrad-fp32-master-v1'),""")
rep("""    return dict(optimizer_step=opt.global_step, implementation='adamw-fresh-state-bf16-replayable-v833' if not fp32_master else 'adamw-fresh-state-fp32-head-master-v833f',""",
    """    return dict(optimizer_step=opt.global_step, implementation='adamw-fresh-state-bf16-replayable-v833' if not fp32_master else ('adamw-fresh-state-fp32-head-master-v833f' if not normgrad else 'fresh-adamw-norm-gradient-direction-fp32-head-master-v833i'),""")
# sgd_step plumbing
rep("def sgd_step(opt, gradients, lr, cap, mode='plain', freeze=(), emit=None, tensor_rel_cap=None, fp32_master=False):",
    "def sgd_step(opt, gradients, lr, cap, mode='plain', freeze=(), emit=None, tensor_rel_cap=None, fp32_master=False, direction='sign'):")
rep("        return fresh_adam_step(opt, gradients, live, lr, cap, freeze, emit, t0, tensor_rel_cap, fp32_master)\n",
    "        return fresh_adam_step(opt, gradients, live, lr, cap, freeze, emit, t0, tensor_rel_cap, fp32_master, direction)\n")
# E. install
rep("""    batched.LOSS_SCALE = engine['loss_scale']
""", """    batched.LOSS_SCALE = engine['loss_scale']
    support = support_settings(engine)                 # v8.3.3i
    batched.SUPPORT_RENORM = support
""")
rep("""    sgd_fp32 = fp32_settings(engine)
""", """    sgd_fp32 = fp32_settings(engine)
    sgd_dir = direction_settings(engine)               # v8.3.3i
""")
rep("return sgd_step(self, gradients, sgd_lr, sgd_cap, sgd_mode, sgd_freeze, tensor_rel_cap=sgd_trc, fp32_master=sgd_fp32)",
    "return sgd_step(self, gradients, sgd_lr, sgd_cap, sgd_mode, sgd_freeze, tensor_rel_cap=sgd_trc, fp32_master=sgd_fp32, direction=sgd_dir)")
rep("""    batched.train_update_batched = dp.make_train_update(batched, fast_kernels)
""", """    dp_update = dp.make_train_update(batched, fast_kernels)

    def train_update_v833i(runtime, optimizer, groups, binding, *, config, **kwargs):
        if support is not None:     # Allan: the support settings must equal the job's (manifest) sampling settings
            got = dict(top_k=getattr(config, 'top_k', None), top_p=getattr(config, 'top_p', None), temperature=getattr(config, 'temperature', None))
            if got != support:
                raise ValueError('support_renorm %r != job sampling settings %r' % (support, got))
            if batched.SUPPORT_RENORM != support:
                raise ValueError('v8211_batched.SUPPORT_RENORM changed')
        out = dp_update(runtime, optimizer, groups, binding, config=config, **kwargs)
        if support is not None and isinstance(out, dict) and out.get('status') == 'updated':
            out['support_renorm'] = support_summary(out.get('trajectories') or [], support)
        return out
    train_update_v833i._original = dp_update._original
    batched.train_update_batched = train_update_v833i
""")
rep("""sgd_tensor_rel_cap=sgd_trc, fp32_master=sgd_fp32, on=sgd_lr is not None),""",
    """sgd_tensor_rel_cap=sgd_trc, fp32_master=sgd_fp32, direction=sgd_dir, on=sgd_lr is not None),
                support_renorm=support,""")
# H. header note
rep('v8.3.3f = v8.3.3 + optional engine.sgd_fp32_master', 'v8.3.3i = v8.3.3h + optional engine.sgd_direction normgrad + engine.support_renorm (Allan; Jacob 2026-10-31 07:00 CR).\nv8.3.3f = v8.3.3 + optional engine.sgd_fp32_master')
p.write_text(s); print('patched', p)
