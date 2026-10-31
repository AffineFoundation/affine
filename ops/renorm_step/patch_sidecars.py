#!/usr/bin/env python3
"""RA/20261031-renorm-step-live: public texts/fields for trainer v8.3.3i. usage: patch_sidecars.py training_params.py k1_transparency.py"""
import sys
from pathlib import Path
def patch(path, pairs):
    p = Path(path); s = p.read_text()
    for old, new in pairs:
        if s.count(old) != 1: raise SystemExit('%s anchor: %r' % (path, old[:70]))
        s = s.replace(old, new)
    p.write_text(s); print('patched', p)
patch(sys.argv[1], [
("SGD_KEYS = ('sgd_lr', 'sgd_update_norm_cap', 'sgd_mode', 'sgd_freeze', 'sgd_tensor_rel_cap')",
 "SGD_KEYS = ('sgd_lr', 'sgd_update_norm_cap', 'sgd_mode', 'sgd_freeze', 'sgd_tensor_rel_cap', 'sgd_fp32_master', 'sgd_direction', 'support_renorm')   # + v8.3.3i"),
("            if k in keys and not isinstance(v, dict): found.setdefault(k, v)",
 "            if k in keys and (not isinstance(v, dict) or k == 'support_renorm'): found.setdefault(k, v)"),
("""    return dict(epoch=int(best.name), loss_aggregation=t.get('loss_aggregation'), microbatch=t.get('microbatch'),
                runtime_versions=(r.get('report') or {}).get('runtime_versions'))""",
 """    sgd = ((t.get('optimizer') or {}).get('sgd')) or {}
    return dict(epoch=int(best.name), loss_aggregation=t.get('loss_aggregation'), microbatch=t.get('microbatch'),
                runtime_versions=(r.get('report') or {}).get('runtime_versions'), support_renorm=t.get('support_renorm'),
                sgd_direction=sgd.get('sgd_direction'), raw_update_norm=sgd.get('raw_update_norm'),
                update_norm_master=(sgd.get('fp32_master') or {}).get('update_norm_master'))"""),
("""        sched = dict(kind='constant', learning_rate=hp['sgd_lr'], restart_rule='none (every turn is one step from zero optimizer state)',
                     drift_cap_rule='per-tensor relative cap tensor_relative_cap (same 2e-5 as the pooled AdamW drift cap)')
""", """        sched = dict(kind='constant', learning_rate=hp['sgd_lr'], restart_rule='none (every turn is one step from zero optimizer state)',
                     drift_cap_rule='per-tensor relative cap tensor_relative_cap (same 2e-5 as the pooled AdamW drift cap)')
        if hp.get('sgd_direction') == 'normgrad':   # v8.3.3i (RA/20261031-renorm-step-live): direction step + support renorm, fp32 master kept
            opt.update(name='direction step (norm of the capped fresh-AdamW step, direction of the gradient)',
                       implementation='fresh-adamw-norm-gradient-direction-fp32-head-master-v833i (trainer host v8.3.3i)',
                       direction='normgrad', fp32_master=hp.get('sgd_fp32_master') is True, step_multiplier=1.0,
                       replay=('g = the fp32 gradient of the loss below over the turn batch (packed varlen micro-batches on 8 GPUs, fp32 per-rank '
                               'accumulators SUM-reduced over NCCL), clipped to global L2 norm max_gradient_norm. M_t = the head fp32 master of tensor t '
                               '(published on every GREEN, https://affine.io/k1/fp32-master/index.json). The step NORM is exactly the old capped fresh-AdamW '
                               'step: d_t = learning_rate * (g_t / (|g_t| + eps) + weight_decay * M_t * [M_t >= 2-D and not a bias]); '
                               'r_t = min(1, tensor_relative_cap * (1 - 1e-6) * ||M_t||_2 / ||d_t||_2) (fp64 norms); s = min(1, global_update_norm_cap / '
                               'sqrt(sum_t r_t^2 ||d_t||^2)) (not binding). Only the DIRECTION changes, from ~sign(g_t) to g_t / ||g_t|| per tensor: '
                               'Delta_t = fp32(s * r_t * ||d_t||_2 / ||g_t||_2) * g_t (0 if ||g_t|| = 0); M_C = M_t - Delta_t; candidate bf16 = '
                               'bf16_round_nearest_even(M_C). So each tensor moves by s * r_t * ||d_t|| (about 2e-5 of its norm on the ~242 rel-capped '
                               'tensors), sqrt(sum ||Delta_t||^2) = raw_update_norm (~0.022) and weight decay only sets the norm budget. Exactly one '
                               'step per turn, no moments carried. Replay: https://affine.io/k1/replay/README.md'))
            sup = hp.get('support_renorm') if isinstance(hp.get('support_renorm'), dict) else None
            last = lt.get('support_renorm') if lt.get('sgd_direction') == 'normgrad' else None
""" ),
("""    if sgd and hp.get('sgd_mode') == 'fresh-adam':   # t751u""", """    sup = last = None                                 # v8.3.3i support renorm (set below when live)
    if sgd and hp.get('sgd_mode') == 'fresh-adam':   # t751u"""),
("""        optimizer=opt,
        schedule=sched,""", """        optimizer=opt,
        support_renorm=(dict(settings=sup, trainer='v8.3.3i', last_turn=last,
                             rule=('PG term uses log q of the sampled token renormalized over the trainer own sampler support (top_k candidates of '
                                   'its fp32 logits / temperature, then top_p on their softmax with cumsum(p) - p < top_p; the first is always kept; '
                                   'support computed without gradient). A sampled token outside that support is dropped from the PG term only (it '
                                   'leaves both numerator and denominator of that rollout PG token mean) and still counts in the k3 KL term and '
                                   'derail detection (full-vocab log p). Files are never rejected for it (format-only validation): the per-turn '
                                   'outside-support token fraction is published here (last_turn) and in turn.json training.support_renorm.'),
                             code='https://affine.io/k1/replay/host/v8211_batched.py support_logq (Allan catoneone/sn120-k1-fixes@230b111 patch/support_renorm.diff)')
                        if sup else None),
        schedule=sched,"""),
])
patch(sys.argv[2], [
("               'optimizer_persistence', 'training_kernels')",
 "               'optimizer_persistence', 'training_kernels', 'sgd_fp32_master', 'sgd_direction', 'support_renorm')"),
("              'maximum_old_actor_logprob_drift')",
 "              'maximum_old_actor_logprob_drift', 'support_renorm')"),
])
