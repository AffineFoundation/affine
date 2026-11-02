#!/usr/bin/env python3
"""RA/20261102-step-mult: training_params.py (training.json / mining.json training.miner_train_params) + step_mult + v8.3.3l knobs.
usage: SM_ALLOWED=1,2,4 patch_tp_l.py SRC OUT"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knobdoc as K
s = open(sys.argv[1]).read()
smr, smw = K.sm_text()
kf = dict(step_mult=K.sm_field(meaning=smw)); kf.update(K.fields_json())
E = [
("TP_FIELDS = dict(lr=dict(default=5e-07, min=1.25e-07, max=1e-06), kl=dict(default=0.02, min=0.005, max=0.1))",
 "TP_FIELDS = dict(lr=dict(default=5e-07, min=1.25e-07, max=1e-06), kl=dict(default=0.02, min=0.005, max=0.1))\n"
 "TP_FIELDS.update(%r)   # RA/20261102-step-mult (trainer v8.3.3l)" % kf),
("           'kl replaces kl_coefficient in L = loss_scale * (1/N) * sum_i (1/K_i) * sum_t [pg + kl * k3] for that turn only.')",
 "           'kl replaces kl_coefficient in L = loss_scale * (1/N) * sum_i (1/K_i) * sum_t [pg + kl * k3] for that turn only. '\n"
 "           'step_mult k (trainer v8.3.3k/l): n_t(k) = k * n_t before the global cap (lr and the per-tensor cap scaled together; one gradient, one step). '\n"
 "           'Knobs (v8.3.3l): eps = adam_eps and 0.01 = weight_decay in d_t; n_t is multiplied by the tensor group scale (lr_scale_attn / mlp / embed / lm_head) '\n"
 "           'x lr_scale_early (layers 0..15) or lr_scale_late (16..31); clamp_mode global_norm drops c_t and uses G = global_norm_cap instead of 1.0 '\n"
 "           '(both: c_t and G); adv_center / adv_clip set the advantages; loss_agg token_mean weighs tokens instead of rollouts. All defaults = the step above.')"),
("        out.append(dict(epoch=int(d.name), lr=tp.get('lr'), kl=tp.get('kl'), given=tp.get('given'), defaults_used=tp.get('defaults_used'),",
 "        out.append(dict(epoch=int(d.name), lr=tp.get('lr'), kl=tp.get('kl'), step_mult=tp.get('step_mult', 1), knobs=tp.get('knobs'), given=tp.get('given'), defaults_used=tp.get('defaults_used'),"),
("        rule=('An affine-batch-v2 file may carry \"train\": {\"lr\": x, \"kl\": y} (both optional; omitted = default). Bounds inclusive; a value '",
 "        rule=('An affine-batch-v2 file may carry \"train\": {\"lr\": x, \"kl\": y, \"step_mult\": k, plus any knob in fields} (all optional; omitted = default = '\n"
 "              'today\\'s trainer; a file without them trains exactly as before). Bounds inclusive; a choice not listed, a value '"),
("'with the defaults. Everything else (per-tensor cap, global cap,", "'with the defaults. Everything else (per-tensor cap and global cap except through step_mult / the knobs,"),
("'train': {'lr': 8e-07, 'kl': 0.01}, 'groups': ['...']},",
 "'train': {'lr': 8e-07, 'kl': 0.01, 'step_mult': %d, 'lr_scale_embed': 0.0, 'adv_center': 'group_norm'}, 'groups': ['...']}," % (2 if K.SM_MAX > 1 else 1)),
]
for o, n in E:
    assert s.count(o) == 1, o[:80]; s = s.replace(o, n)
open(sys.argv[2], 'w').write(s); print('ok')
