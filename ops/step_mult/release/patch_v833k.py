#!/usr/bin/env python3
"""RA/20261102-step-mult: trainer host v8.3.3j -> v8.3.3k (optional per-turn step multiplier train.step_mult from a free-file dispatch).
duongna (#120, 2026-11-02 09:42 CR): "a step multiplier m (e.g. 1-8, default 1) that scales both the lr and the per-tensor norm cap";
Jacob 2026-11-02 10:19 CR: "lets do this exactly like dou suggests and add a new hparam to the submission format".
Usage: patch_v833k.py <host dir>   (edits mimo_host_trainer_v83.py and ff_trainer.py in place; every anchor must match exactly once)."""
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
('v8.3.3j = v8.3.3i + per-turn training params (lr, kl)',
 'v8.3.3k = v8.3.3j + optional per-turn step multiplier k = train.step_mult (integer 1..8, default 1) from a free-file dispatch\n'
 '  (duongna #120 proposal, Jacob 2026-11-02 10:19 CR, RA/20261102-step-mult): the whole per-tensor step is multiplied by k, i.e. lr and the\n'
 '  per-tensor cap are scaled together; one gradient, one step, same trainer time. k = 1 (or no step_mult) runs the exact v8.3.3j step.\n'
 'v8.3.3j = v8.3.3i + per-turn training params (lr, kl)'),
('''TURN = dict(lr_multiplier=1.0, last=None)      # set only by train_update_v833j around one update''',
 '''#  step_mult (v8.3.3k, RA/20261102-step-mult): k = the dispatch's train.step_mult (int, 1..8, default 1). With n_t the v8.3.3j per-tensor
#      step norm above (n_t = n0_t when m = 1):  n_t(k) = k * n_t  ==  min(k*m*n0_t, k*c_t) for m >= 1: lr and the per-tensor cap scaled
#      together;  s = min(1, sgd_update_norm_cap / sqrt(sum_t n_t(k)^2)) (global cap unchanged, ~0.023*k << 1 so it does not fire);
#      Delta_t = fp32(s * n_t(k) / ||g_t||) * g_t.  k = 1 -> no change at all (same code path as v8.3.3j).
TURN = dict(lr_multiplier=1.0, step_mult=1, last=None)      # set only by train_update_v833j around one update'''),
('''                            tensors_rel_capped_default=sum(v < 1.0 for v in r.values()), capped_norm_fraction_default=capped_frac)
    changed = 0; dsq = []; wsq = []; msq = []
''', '''                            tensors_rel_capped_default=sum(v < 1.0 for v in r.values()), capped_norm_fraction_default=capped_frac)
    k_mult = TURN.get('step_mult', 1)                      # v8.3.3k
    if k_mult != 1:
        if not (normgrad and tensor_rel_cap is not None and fp32_master):
            raise ValueError('v8.3.3k: a turn step_mult needs the normgrad fp32-master step with the per-tensor cap')
        base = nj if lr_mult != 1.0 else n0
        nk = {name: k_mult * base[name] for name in r}
        raw_k = math.sqrt(math.fsum(v * v for v in nk.values())); scale_k = min(1.0, cap / raw_k) if raw_k > 0 else 1.0
        coef_j = {name: (scale_k * nk[name] / math.sqrt(gn2[name])) if gn2[name] > 0 else 0.0 for name in r}
        TURN['last'].update(path=TURN['last']['path'] + '+step_mult', step_mult=k_mult, raw_update_norm_k1=TURN['last']['raw_update_norm'],
                            raw_update_norm=raw_k, cap_scale=scale_k, cap_fired=scale_k < 1.0, tensor_rel_cap_effective=tensor_rel_cap * k_mult)
    changed = 0; dsq = []; wsq = []; msq = []
'''),
('''        tp = TURN_PARAMS['params']
        TURN['lr_multiplier'] = 1.0; TURN['last'] = None
''', '''        tp = TURN_PARAMS['params']
        TURN['lr_multiplier'] = 1.0; TURN['step_mult'] = 1; TURN['last'] = None
'''),
('''        if tp['defaults'] != dict(lr=sgd_lr, kl=float(config.kl_coefficient)):''',
 '''        if {k: v for k, v in tp['defaults'].items() if k != 'step_mult'} != dict(lr=sgd_lr, kl=float(config.kl_coefficient)) or tp['defaults'].get('step_mult', 1) != 1:'''),
('''        m = 1.0 if lr == sgd_lr else lr / sgd_lr
        if m != 1.0 and not (sgd_mode == 'fresh-adam' and sgd_fp32 and sgd_dir == 'normgrad' and sgd_trc is not None):
            raise ValueError('v8.3.3j: a turn lr needs fresh-adam + fp32 master + normgrad + sgd_tensor_rel_cap')''',
 '''        m = 1.0 if lr == sgd_lr else lr / sgd_lr
        k_mult = tp.get('step_mult', 1)                     # v8.3.3k
        if type(k_mult) is not int or not 1 <= k_mult <= 8:
            raise ValueError('v8.3.3k: step_mult %r not an integer in 1..8' % (k_mult,))
        if (m != 1.0 or k_mult != 1) and not (sgd_mode == 'fresh-adam' and sgd_fp32 and sgd_dir == 'normgrad' and sgd_trc is not None):
            raise ValueError('v8.3.3j: a turn lr needs fresh-adam + fp32 master + normgrad + sgd_tensor_rel_cap')'''),
('''        TURN['lr_multiplier'] = m
        try:''', '''        TURN['lr_multiplier'] = m; TURN['step_mult'] = k_mult
        try:'''),
('''            optimizer.config = saved; TURN['lr_multiplier'] = 1.0
''', '''            optimizer.config = saved; TURN['lr_multiplier'] = 1.0; TURN['step_mult'] = 1
'''),
('''                    defaults_used=(m == 1.0 and cfg is config), lr_multiplier=m, kl_applied=float(cfg.kl_coefficient),''',
 '''                    defaults_used=(m == 1.0 and cfg is config and k_mult == 1), lr_multiplier=m, kl_applied=float(cfg.kl_coefficient),
                    step_mult=k_mult,'''),
('''                          's = min(1, sgd_update_norm_cap / ||n||), Delta_t = s * n_t * g_t / ||g_t||; m = 1 -> the v8.3.3i step exactly'))''',
 '''                          's = min(1, sgd_update_norm_cap / ||n||), Delta_t = s * n_t * g_t / ||g_t||; m = 1 -> the v8.3.3i step exactly; '
                          'step_mult k (v8.3.3k): n_t := k * n_t before the global cap (lr and per-tensor cap scaled together), k = 1 -> unchanged'))'''),
('''turn_train_params='v833j: free-file dispatch train {lr, kl}',''',
 '''turn_train_params='v833k: free-file dispatch train {lr, kl, step_mult}','''),
])

patch('ff_trainer.py', [
('''TRAIN_BOUNDS = {'lr': (1.25e-07, 1e-06), 'kl': (0.005, 0.1)}
''', '''TRAIN_BOUNDS = {'lr': (1.25e-07, 1e-06), 'kl': (0.005, 0.1)}
# v8.3.3k (RA/20261102-step-mult): optional 'step_mult' (int 1..8; absent = 1) in 'train' and in 'given'; defaults may carry step_mult 1.
STEP_MULT_ALLOWED = (1, 2, 3, 4, 5, 6, 7, 8)
'''),
('''    need(type(t) is dict and set(t) == {'lr', 'kl', 'given', 'defaults'}, 'train keys')''',
 '''    need(type(t) is dict and set(t) - {'step_mult'} == {'lr', 'kl', 'given', 'defaults'}, 'train keys')
    km = t.get('step_mult', 1)
    need(type(km) is int and km in STEP_MULT_ALLOWED, 'train.step_mult in %r' % (STEP_MULT_ALLOWED,))'''),
('''    need(type(t['given']) is list and set(t['given']) <= set(TRAIN_BOUNDS) and len(set(t['given'])) == len(t['given']), 'train.given')''',
 '''    need(type(t['given']) is list and set(t['given']) <= set(TRAIN_BOUNDS) | {'step_mult'} and len(set(t['given'])) == len(t['given']), 'train.given')'''),
('''    need(type(d) is dict and set(d) == set(TRAIN_BOUNDS) and all(type(d[k]) is float for k in d), 'train.defaults')
    return dict(lr=t['lr'], kl=t['kl'], given=list(t['given']), defaults=dict(lr=d['lr'], kl=d['kl']),
                epoch=binding.epoch, checkpoint=binding.checkpoint)''',
 '''    need(type(d) is dict and set(d) - {'step_mult'} == set(TRAIN_BOUNDS) and all(type(d[x]) is float for x in TRAIN_BOUNDS)
         and type(d.get('step_mult', 1)) is int and d.get('step_mult', 1) == 1, 'train.defaults')
    return dict(lr=t['lr'], kl=t['kl'], step_mult=km, given=list(t['given']), defaults=dict(lr=d['lr'], kl=d['kl']),
                epoch=binding.epoch, checkpoint=binding.checkpoint)'''),
])
