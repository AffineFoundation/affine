#!/usr/bin/env python3
"""RA/20261102-step-mult: apply the GitHub (AffineFoundation/affine) changes for trainer v8.3.3l (step_mult + optional knobs) in REPO
from ART (artifacts copied from the host: host/<4 changed trainer files>, release/{ff_format,affine_batch}.py with the live step_mult range,
ops/ -> ops/step_mult). usage: SM_ALLOWED=1,2,4 gh_apply_l.py REPO ART FROM_EPOCH"""
import hashlib, os, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knobdoc as K
R, A, FE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
smr, smw = K.sm_text()
CHANGED = ('mimo_host_trainer_v83.py', 'ff_trainer.py', 'v8211_batched.py', 'v83_dp.py')
def rep(p, o, n, cnt=1):
    s = (R / p).read_text(); c = s.count(o)
    if c != cnt: raise SystemExit('%s: anchor count %d != %d: %r' % (p, c, cnt, o[:90]))
    (R / p).write_text(s.replace(o, n))
def sh(p): return hashlib.sha256((R / p).read_bytes()).hexdigest()
git = lambda *a: subprocess.run(['git', '-C', str(R)] + list(a), check=True)
# 1. verbatim validator code: trainer-v833j -> trainer-v833l (4 changed files), ff_format, client
if (R / 'miner/validator/trainer-v833j').exists(): git('mv', 'miner/validator/trainer-v833j', 'miner/validator/trainer-v833l')
for n in CHANGED: shutil.copy(A / 'host' / n, R / 'miner/validator/trainer-v833l' / n)
shutil.copy(A / 'release/ff_format.py', R / 'miner/validator/batch/ff_format.py')
shutil.copy(A / 'release/affine_batch.py', R / 'miner/affine_batch.py')
# 2. MANIFEST
m = (R / 'miner/validator/MANIFEST.sha256').read_text().split('\n'); out = []
for l in m:
    if l.startswith('#') or not l.strip(): out.append(l); continue
    f = l.split('  ')
    if 'validator/trainer-v833j/' in f[1]:
        f[1] = f[1].replace('trainer-v833j', 'trainer-v833l')
        if len(f) > 3: f[3] = f[3].replace('20261031-ff-train-params/release-v833j-signed/trainer-host-v833j', '20261102-step-mult/release-v833l-signed/trainer-host-v833l')
        if f[2].startswith('public-config'): f[2] = f[2].replace('v8.3.3j', 'v8.3.3l (engine block unchanged from v8.3.3j)')
    if 'validator/trainer-v833l/' in f[1] or f[1] == 'validator/batch/ff_format.py': f[0] = sh('miner/' + f[1])   # only the files this change touches
    out.append('  '.join(f))
(R / 'miner/validator/MANIFEST.sha256').write_text('\n'.join(out))
# 3. kit: v833j -> v833l references, train_step step_mult + knobs (applied exactly as the host wrapper train_update_v833j does)
for p in ('miner/kit/kitpath.py', 'miner/kit/simulate_turn.sh', 'miner/kit/generate.py', 'miner/kit/tests/test_kit_cpu.py', 'miner/requirements-trainer.txt'):
    s = (R / p).read_text(); s = s.replace('trainer-v833j', 'trainer-v833l').replace('v8.3.3j', 'v8.3.3l'); (R / p).write_text(s)
p = 'miner/kit/train_step.py'
rep(p, 'with the live trainer code (v8.3.3j).', 'with the live trainer code (v8.3.3l).')
rep(p, '+ the v8.3.3j lr multiplier / kl.', '+ the v8.3.3j lr multiplier / kl + the v8.3.3l step_mult and knobs.')
rep(p, "    lr, kl = float(train['lr']), float(train['kl'])\n",
    "    lr, kl = float(train['lr']), float(train['kl']); k_mult = int(train.get('step_mult', 1))\n"
    "    kn = {x: train.get(x, H.KNOB_DEFAULTS[x]) for x in H.KNOB_SPEC}          # v8.3.3l knobs, as train_update_v833j applies them\n"
    "    OPT_KEYS = ('lr_scale_attn', 'lr_scale_mlp', 'lr_scale_embed', 'lr_scale_lm_head', 'lr_scale_early', 'lr_scale_late',\n"
    "                'adam_eps', 'weight_decay', 'clamp_mode', 'global_norm_cap')\n"
    "    LOSS_KEYS = ('loss_agg', 'adv_center', 'adv_clip')\n"
    "    opt_on = kn['clamp_mode'] != 'per_tensor' or any(kn[x] != H.KNOB_DEFAULTS[x] for x in OPT_KEYS if x != 'global_norm_cap')\n"
    "    loss_on = any(kn[x] != H.KNOB_DEFAULTS[x] for x in LOSS_KEYS)\n")
rep(p, "    H.TURN['lr_multiplier'] = m; H.TURN['last'] = None\n",
    "    H.TURN['lr_multiplier'] = m; H.TURN['step_mult'] = k_mult; H.TURN['last'] = None\n"
    "    H.TURN['opt'] = {x: kn[x] for x in OPT_KEYS} if opt_on else None\n"
    "    B.KNOBS = {x: kn[x] for x in LOSS_KEYS} if loss_on else None\n")
rep(p, "    out['train_params'] = dict(lr=lr, kl=kl, lr_multiplier=m,",
    "    out['train_params'] = dict(lr=lr, kl=kl, step_mult=k_mult, knobs=kn, knobs_applied=dict(step=opt_on, loss=loss_on), lr_multiplier=m,")
rep(p, "    H.TURN['lr_multiplier'] = 1.0\n    Path(a.out)", "    H.TURN['lr_multiplier'] = 1.0; H.TURN['step_mult'] = 1; H.TURN['opt'] = None; B.KNOBS = None\n    Path(a.out)")
rep(p, "tensors_rel_capped=sgd.get('tensors_rel_capped'), lr=lr, kl=kl, out=a.out)",
    "tensors_rel_capped=sgd.get('tensors_rel_capped'), lr=lr, kl=kl, step_mult=k_mult, knobs={x: v for x, v in kn.items() if v != H.KNOB_DEFAULTS[x]}, out=a.out)")
# 4. docs
TABLE = K.md_table()
p = 'miner/README.md'
s = (R / p).read_text(); s = s.replace('trainer v8.3.3j', 'trainer v8.3.3l').replace('ONE v8.3.3j step', 'ONE v8.3.3l step').replace('trainer-v833j/ ', 'trainer-v833l/ ')
s = s.replace('## 3. The training step (v8.3.3j)', '## 3. The training step (v8.3.3l)').replace('`validator/trainer-v833j`', '`validator/trainer-v833l`'); (R / p).write_text(s)
rep(p, '  (default 5e-7) and `kl` in [0.005, 0.1] (default 0.02). The bounds are inclusive and values are never clamped.\n',
    '  (default 5e-7) and `kl` in [0.005, 0.1] (default 0.02). From epoch %s (trainer v8.3.3l) also `step_mult` and the optional\n'
    '  knobs below; every one defaults to the previous trainer, so a file without them trains bit-identically to before.\n'
    '  The bounds are inclusive and values are never clamped (a non-integer step_mult such as 2.0 is invalid).\n\n' % FE + TABLE + '\n')
rep(p, '  (at e368, 242 of 427 tensors were at the cap; turn.json `training.train_params.step`).\n',
    '  (at e368, 242 of 427 tensors were at the cap; turn.json `training.train_params.step`).\n'
    '  **step_mult k** (v8.3.3l, from epoch %s; proposed in #120 by duongna): n_t := k·n_t before the global cap, i.e. lr and the\n'
    '  per-tensor cap scale together and every tensor moves k× (the capped ones too). One gradient, one step, no extra trainer\n'
    '  time. The default step norm is ≈0.023, so the global cap (1.0) does not fire for any allowed k. k = 1 is exactly the v8.3.3j step.\n'
    '  **Knobs** (v8.3.3l, only when one differs from its default): eps = adam_eps and 0.01 = weight_decay in d_t; n_t is multiplied\n'
    '  by the tensor\'s group scale (lr_scale_attn / mlp / embed / lm_head; norms 1) × lr_scale_early (layers 0–15) or lr_scale_late\n'
    '  (16–31); clamp_mode global_norm drops c_t and caps the whole step at global_norm_cap (both: c_t and then global_norm_cap);\n'
    '  adv_center / adv_clip set the GRPO advantages; loss_agg token_mean weighs tokens instead of rollouts. The direction is\n'
    '  always g_t/‖g_t‖. Code: `validator/trainer-v833l/mimo_host_trainer_v83.py` (`knob_scale`, `_fresh_adam_step`) and\n'
    '  `v8211_batched.py` (`_apply_knobs`); `kit/train_step.py` applies them exactly like the live wrapper.\n' % FE)
p = 'docs/MINER_QUICKSTART.md'
rep(p, '## 2. The file: `affine-batch-v1` (or `affine-batch-v2` with optional lr / kl)', '## 2. The file: `affine-batch-v1` (or `affine-batch-v2` with optional lr / kl / step_mult / knobs)')
rep(p, '  "train": {"lr": 8e-7, "kl": 0.01},\n  "groups": [ ...same as v1... ]',
    '  "train": {"lr": 8e-7, "kl": 0.01, "step_mult": %d, "lr_scale_embed": 0, "adv_center": "group_norm"},\n  "groups": [ ...same as v1... ]' % (2 if K.SM_MAX > 1 else 1))
rep(p, "| `train.kl` | 0.02 | 0.005 .. 0.1 | kl_coefficient of your turn's loss |\n",
    "| `train.kl` | 0.02 | 0.005 .. 0.1 | kl_coefficient of your turn's loss |\n" + K.md_table(lead=''))
rep(p, '- Both fields are optional;', '- All fields are optional (step_mult and the knobs from epoch %s, trainer v8.3.3l); the defaults are exactly the previous trainer, so a file without them, or with every one at its default, trains bit-identically to before;' % FE)
rep(p, '- Out of range, not a number (string, true/false, null) or any other key',
    '- Out of range, not a number (string, true/false, null), a step_mult that is not an integer (2.0, "2"), a choice that is not listed, or any other key')
rep(p, '- How kl enters: it replaces kl_coefficient in\n',
    '- How step_mult enters (trainer v8.3.3l, from epoch %s; proposed in #120 by duongna): your step norm n_t (above)\n'
    '  is multiplied by k before the global cap, n_t := k x n_t. That is lr and the per-tensor cap scaled together,\n'
    '  so the capped tensors move k x too. Still one gradient and one step (no extra trainer time); the default\n'
    '  step norm is about 0.023, so the global cap 1.0 does not fire for any allowed k. k = 1 is exactly the old step.\n'
    '- How the step knobs enter: eps = adam_eps and 0.01 = weight_decay in d_t; n_t x (group scale x early/late scale); clamp_mode\n'
    '  global_norm: no per-tensor cap, the whole step is scaled to at most global_norm_cap (both: per-tensor cap, then global_norm_cap).\n'
    '  The direction is always g_t / ||g_t||. lr_scale_* = 0 freezes those tensors for your turn.\n'
    '- How the loss knobs enter: adv_center none uses the raw reward, group_norm divides by the group std (+1e-6); adv_clip clips\n'
    '  the advantage; loss_agg token_mean weighs every token equally (default seq_mean: every rollout equally).\n'
    '- How kl enters: it replaces kl_coefficient in\n' % FE)
rep(p, '- Everything else is fixed: per-tensor cap, global cap', '- Everything else is fixed: per-tensor cap and global cap (except through step_mult / the knobs)')
rep(p, '- Your effective lr / kl are published', '- Your effective lr / kl / step_mult / knobs are published')
rep(p, '(reports format, lr, kl).', '(reports format, lr, kl, step_mult and the knobs you set).')
p = 'README.md'
s = (R / p).read_text(); s = s.replace('(trainer v8.3.3j)', '(trainer v8.3.3l)'); (R / p).write_text(s)
a = [l for l in s.split('\n') if l.startswith('- Optional per-turn `lr` (1.25e-7 to 1e-6')][0]
rep(p, a + '\n', a + '\n- Optional per-turn `step_mult` (%s, default 1; duongna\'s step multiplier) and optimizer / loss knobs (per-group lr scales, '
    'adam_eps, weight_decay, clamp_mode, loss_agg, adv_center, adv_clip; all default to the previous trainer) in an `affine-batch-v2` file, '
    'from e%s: [fields and math](docs/MINER_QUICKSTART.md) · [validator side (trainer v8.3.3l, replay, activation/revert)](ops/step_mult/README.md)\n' % (smr, FE))
# 5. ops/step_mult
O = R / 'ops/step_mult'
if O.exists(): shutil.rmtree(O)
shutil.copytree(A / 'ops', O)
print('applied')
