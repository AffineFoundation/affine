#!/usr/bin/env python3
"""Patch miner/kit/train_step.py (already step_mult-patched by gh_apply_sm.py) for the v8.3.3l knobs, exactly as the host wrapper applies them.
usage: kit_train_step_knobs.py PATH"""
import sys
p = sys.argv[1]; s = open(p).read()
E = [
("    lr, kl = float(train['lr']), float(train['kl']); k_mult = int(train.get('step_mult', 1))\n",
 "    lr, kl = float(train['lr']), float(train['kl']); k_mult = int(train.get('step_mult', 1))\n"
 "    kn = {x: train.get(x, H.KNOB_DEFAULTS[x]) for x in H.KNOB_SPEC}          # v8.3.3l knobs, as train_update_v833j applies them\n"
 "    OPT_KEYS = ('lr_scale_attn', 'lr_scale_mlp', 'lr_scale_embed', 'lr_scale_lm_head', 'lr_scale_early', 'lr_scale_late',\n"
 "                'adam_eps', 'weight_decay', 'clamp_mode', 'global_norm_cap')\n"
 "    LOSS_KEYS = ('loss_agg', 'adv_center', 'adv_clip')\n"
 "    opt_on = kn['clamp_mode'] != 'per_tensor' or any(kn[x] != H.KNOB_DEFAULTS[x] for x in OPT_KEYS if x != 'global_norm_cap')\n"
 "    loss_on = any(kn[x] != H.KNOB_DEFAULTS[x] for x in LOSS_KEYS)\n"),
("    H.TURN['lr_multiplier'] = m; H.TURN['step_mult'] = k_mult; H.TURN['last'] = None\n",
 "    H.TURN['lr_multiplier'] = m; H.TURN['step_mult'] = k_mult; H.TURN['last'] = None\n"
 "    H.TURN['opt'] = {x: kn[x] for x in OPT_KEYS} if opt_on else None\n"
 "    B.KNOBS = {x: kn[x] for x in LOSS_KEYS} if loss_on else None\n"),
("    out['train_params'] = dict(lr=lr, kl=kl, step_mult=k_mult, lr_multiplier=m,",
 "    out['train_params'] = dict(lr=lr, kl=kl, step_mult=k_mult, knobs=kn, knobs_applied=dict(step=opt_on, loss=loss_on), lr_multiplier=m,"),
("    H.TURN['lr_multiplier'] = 1.0; H.TURN['step_mult'] = 1\n", "    H.TURN['lr_multiplier'] = 1.0; H.TURN['step_mult'] = 1; H.TURN['opt'] = None; B.KNOBS = None\n"),
("+ the v8.3.3j lr multiplier / kl + the v8.3.3k step_mult.", "+ the v8.3.3j lr multiplier / kl + the v8.3.3k step_mult + the v8.3.3l knobs."),
]
for o, n in E:
    c = s.count(o)
    if c != 1: raise SystemExit('anchor count %d: %r' % (c, o[:80]))
    s = s.replace(o, n)
open(p, 'w').write(s); print('ok')
