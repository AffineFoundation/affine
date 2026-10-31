#!/usr/bin/env python3
"""RA/20261031-ff-train-params: trainer host v8.3.3i -> v8.3.3j (per-turn lr / kl from a free-file v2 dispatch).
Usage: patch_v833j.py <host dir>   (edits mimo_host_trainer_v83.py and ff_trainer.py in place; every anchor must match exactly once)."""
import sys
from pathlib import Path
H = Path(sys.argv[1])


def patch(name, edits):
    p = H / name; s = p.read_text()
    for old, new in edits:
        c = s.count(old)
        if c != 1:
            raise SystemExit('%s: anchor count %d != 1: %r' % (name, c, old[:90]))
        s = s.replace(old, new)
    p.write_text(s); print('patched', p)


patch('mimo_host_trainer_v83.py', [
# header
('v8.3.3i = v8.3.3h + optional engine.sgd_direction normgrad',
 'v8.3.3j = v8.3.3i + per-turn training params (lr, kl) from a free-file affine-batch-v2 dispatch (Jacob 2026-10-31 pt902u,\n'
 '  RA/20261031-ff-train-params). No engine key; a dispatch without \'train\' or with the defaults runs the exact v8.3.3i step.\n'
 'v8.3.3i = v8.3.3h + optional engine.sgd_direction normgrad'),
# module state + documentation
('''def _sha_tensor(t):''', '''# v8.3.3j (RA/20261031-ff-train-params; Jacob 2026-10-31 07:08 CR pt902u): per-turn training params from a signed free-file dispatch
# ('train': {lr, kl, given, defaults}; ff_trainer.train_params). Defaults (lr == engine.sgd_lr, kl == config.kl_coefficient) -> the exact
# v8.3.3i code path (same config object, lr multiplier 1.0, same expressions).
#  kl: config.kl_coefficient := the turn's kl for this one update (all DP ranks: the job header carries the config); the loss is
#      L = loss_scale * (1/N) * sum_i (1/K_i) * sum_t [ pg_it + kl * k3_it ]; the signed config document is reported unchanged and
#      the applied value is in training.train_params.
#  lr: m = lr / engine.sgd_lr. With sgd_direction normgrad + sgd_tensor_rel_cap (required when m != 1):
#      n0_t = r_t * ||d_t||                                   the v8.3.3i per-tensor step norm (d_t, r_t at sgd_lr, before the global cap)
#      c_t  = sgd_tensor_rel_cap * (1 - 1e-6) * ||M_H,t||     the unchanged per-tensor ceiling (none if ||M_H,t|| = 0)
#      n_t  = min(m * n0_t, c_t)                              m <= 1: exactly m * n0_t; m > 1: grows only below the ceiling
#      s    = min(1, sgd_update_norm_cap / sqrt(sum_t n_t^2)) the unchanged global cap
#      Delta_t = fp32(s * n_t / ||g_t||) * g_t (fp64 scalars);  M_C = M_H - Delta_t;  W_C = bf16_rne(M_C)
#  training.optimizer.sgd keeps the default-lr (v8.3.3i) statistics; the turn's own step norms are in training.train_params.step.
TURN = dict(lr_multiplier=1.0, last=None)      # set only by train_update_v833j around one update
TURN_PARAMS = dict(params=None)                # set by prepare_signed_dispatch for the job being prepared


def _sha_tensor(t):'''),
# first loop: keep ||M_H,t||
('''        dn2 = []; r = {}; qd = {}; gn2 = {}
''', '''        dn2 = []; r = {}; qd = {}; gn2 = {}; wnd = {}
'''),
('''                wn = math.sqrt(float(w32.double().pow(2).sum()))
                rt = min(1.0, tensor_rel_cap * (1 - 1e-6) * wn / math.sqrt(q)) if wn > 0 else 1.0
            r[name] = rt; dn2.append(q * rt * rt)''', '''                wn = math.sqrt(float(w32.double().pow(2).sum())); wnd[name] = wn
                rt = min(1.0, tensor_rel_cap * (1 - 1e-6) * wn / math.sqrt(q)) if wn > 0 else 1.0
            r[name] = rt; dn2.append(q * rt * rt)'''),
# scaled step coefficients
('''    raw = math.sqrt(math.fsum(dn2)); scale = min(1.0, cap / raw) if raw > 0 else 1.0
    changed = 0; dsq = []; wsq = []; msq = []
''', '''    raw = math.sqrt(math.fsum(dn2)); scale = min(1.0, cap / raw) if raw > 0 else 1.0
    lr_mult = TURN['lr_multiplier']; coef_j = None          # v8.3.3j
    n0 = {name: r[name] * math.sqrt(qd[name]) for name in r} if normgrad else {}
    capped_frac = (math.sqrt(math.fsum(n0[k] ** 2 for k in n0 if r[k] < 1.0)) / raw) if (normgrad and raw > 0) else None
    if lr_mult != 1.0:
        if not (normgrad and tensor_rel_cap is not None and fp32_master):
            raise ValueError('v8.3.3j: a turn lr needs the normgrad fp32-master step with the per-tensor cap')
        nj = {}
        for name in r:
            mn = lr_mult * n0[name]
            nj[name] = min(mn, tensor_rel_cap * (1 - 1e-6) * wnd[name]) if wnd.get(name, 0.0) > 0 else mn
        raw_j = math.sqrt(math.fsum(v * v for v in nj.values())); scale_j = min(1.0, cap / raw_j) if raw_j > 0 else 1.0
        coef_j = {name: (scale_j * nj[name] / math.sqrt(gn2[name])) if gn2[name] > 0 else 0.0 for name in r}
        TURN['last'] = dict(path='scaled', lr_multiplier=lr_mult, raw_update_norm=raw_j, cap_scale=scale_j, cap_fired=scale_j < 1.0,
                            tensors_at_ceiling=sum(1 for k in nj if nj[k] < lr_mult * n0[k]), tensors=len(nj),
                            raw_update_norm_default=raw, cap_scale_default=scale, tensors_rel_capped_default=sum(v < 1.0 for v in r.values()),
                            capped_norm_fraction_default=capped_frac)
    else:
        TURN['last'] = dict(path='v833i-exact', lr_multiplier=1.0, raw_update_norm=raw, cap_scale=scale, cap_fired=scale < 1.0,
                            tensors=len(r), raw_update_norm_default=raw, cap_scale_default=scale,
                            tensors_rel_capped_default=sum(v < 1.0 for v in r.values()), capped_norm_fraction_default=capped_frac)
    changed = 0; dsq = []; wsq = []; msq = []
'''),
('''                        c = (scale * r[name] * math.sqrt(qd[name]) / math.sqrt(gn2[name])) if gn2[name] > 0 else 0.0
''', '''                        c = ((scale * r[name] * math.sqrt(qd[name]) / math.sqrt(gn2[name])) if gn2[name] > 0 else 0.0) if coef_j is None else coef_j[name]
'''),
('''        info['fp32_master'] = dict(fpi, update_norm_master=dM, update_rel_master=dM / W if W else None)
''', '''        info['fp32_master'] = dict(fpi, update_norm_master=dM, update_rel_master=dM / W if W else None)
        if TURN['last'] is not None:
            TURN['last']['update_norm_master'] = dM
'''),
# prepare: capture the turn params of the dispatch being prepared
('''        if ff.is_ff(dispatch):        # v8.3.3h: signed sn120-ff-dispatch-v1 -> miner rewards as given
            batched.TASK_INDEX.clear()
            return ff.prepare(*args, **kwargs)
        prepared_ = original_prepare(*args, **kwargs)''', '''        TURN_PARAMS['params'] = None
        if ff.is_ff(dispatch):        # v8.3.3h: signed sn120-ff-dispatch-v1 -> miner rewards as given
            batched.TASK_INDEX.clear()
            out = ff.prepare(*args, **kwargs)
            TURN_PARAMS['params'] = ff.train_params(out['dispatch'], out['binding'])     # v8.3.3j (None: no 'train')
            return out
        prepared_ = original_prepare(*args, **kwargs)'''),
# train update wrapper
('''    train_update_v833i._original = dp_update._original
    batched.train_update_batched = train_update_v833i
''', '''    train_update_v833i._original = dp_update._original
    import dataclasses as _dc

    def train_update_v833j(runtime, optimizer, groups, binding, *, config, **kwargs):
        """v8.3.3j: apply the prepared free-file turn's lr / kl to this one update (defaults -> exact v8.3.3i)."""
        tp = TURN_PARAMS['params']
        TURN['lr_multiplier'] = 1.0; TURN['last'] = None
        if tp is None:
            return train_update_v833i(runtime, optimizer, groups, binding, config=config, **kwargs)
        if (binding.epoch, binding.checkpoint) != (tp['epoch'], tp['checkpoint']):
            raise ValueError('v8.3.3j: turn params belong to another job')
        if tp['defaults'] != dict(lr=sgd_lr, kl=float(config.kl_coefficient)):
            raise ValueError('v8.3.3j: dispatch defaults %r != trainer defaults lr %r kl %r' % (tp['defaults'], sgd_lr, config.kl_coefficient))
        lr, kl = tp['lr'], tp['kl']
        m = 1.0 if lr == sgd_lr else lr / sgd_lr
        if m != 1.0 and not (sgd_mode == 'fresh-adam' and sgd_fp32 and sgd_dir == 'normgrad' and sgd_trc is not None):
            raise ValueError('v8.3.3j: a turn lr needs fresh-adam + fp32 master + normgrad + sgd_tensor_rel_cap')
        cfg = config if kl == config.kl_coefficient else _dc.replace(config, kl_coefficient=kl).validate()
        saved = optimizer.config
        TURN['lr_multiplier'] = m
        try:
            if cfg is not config:
                optimizer.config = cfg
            out = train_update_v833i(runtime, optimizer, groups, binding, config=cfg, **kwargs)
        finally:
            optimizer.config = saved; TURN['lr_multiplier'] = 1.0
        if isinstance(out, dict):
            if cfg is not config and 'config' in out:
                out['config'] = config.document()           # the signed config document, unchanged (validate_result)
            if out.get('status') == 'updated':
                out['train_params'] = dict(
                    version='v833j-turn-train-params-v1', lr=lr, kl=kl, given=list(tp['given']), defaults=dict(tp['defaults']),
                    defaults_used=(m == 1.0 and cfg is config), lr_multiplier=m, kl_applied=float(cfg.kl_coefficient),
                    step=TURN['last'],
                    rule=('kl replaces kl_coefficient in L = loss_scale * (1/N) * sum_i (1/K_i) * sum_t [pg + kl * k3]; '
                          'lr: m = lr / sgd_lr, n_t = min(m * r_t * ||d_t||, sgd_tensor_rel_cap * (1 - 1e-6) * ||M_H,t||), '
                          's = min(1, sgd_update_norm_cap / ||n||), Delta_t = s * n_t * g_t / ||g_t||; m = 1 -> the v8.3.3i step exactly'))
        TURN['last'] = None
        return out
    train_update_v833j._original = dp_update._original
    batched.train_update_batched = train_update_v833j
'''),
('''                support_renorm=support,
                sites=sorted(SITES))''', '''                support_renorm=support, turn_train_params='v833j: free-file dispatch train {lr, kl}',
                sites=sorted(SITES))'''),
])

patch('ff_trainer.py', [
('''VERSION = 'sn120-ff-dispatch-v1'
''', '''VERSION = 'sn120-ff-dispatch-v1'
# v8.3.3j (RA/20261031-ff-train-params, Jacob pt902u): optional selection key 'train' = {lr, kl, given, defaults} (lifecycle fills the
# defaults for keys the file omitted). Same bounds as the lifecycle format check (ff_format.TRAIN_BOUNDS); checked again here.
TRAIN_BOUNDS = {'lr': (1.25e-07, 1e-06), 'kl': (0.005, 0.1)}
'''),
('''def prepare(opening, dispatch, *, authority, documents_root, runtime, optimizer_step=0, max_groups=None):''',
'''def train_params(selection, binding):
    """The signed selection's 'train' -> dict(lr, kl, given, defaults, epoch, checkpoint), or None when absent."""
    t = selection.get('train')
    if t is None:
        return None
    need(type(t) is dict and set(t) == {'lr', 'kl', 'given', 'defaults'}, 'train keys')
    for k, (lo, hi) in TRAIN_BOUNDS.items():
        need(type(t[k]) is float and lo <= t[k] <= hi, 'train.%s in [%r, %r]' % (k, lo, hi))
    need(type(t['given']) is list and set(t['given']) <= set(TRAIN_BOUNDS) and len(set(t['given'])) == len(t['given']), 'train.given')
    d = t['defaults']
    need(type(d) is dict and set(d) == set(TRAIN_BOUNDS) and all(type(d[k]) is float for k in d), 'train.defaults')
    return dict(lr=t['lr'], kl=t['kl'], given=list(t['given']), defaults=dict(lr=d['lr'], kl=d['kl']),
                epoch=binding.epoch, checkpoint=binding.checkpoint)


def prepare(opening, dispatch, *, authority, documents_root, runtime, optimizer_step=0, max_groups=None):'''),
])
