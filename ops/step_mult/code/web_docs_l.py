#!/usr/bin/env python3
"""RA/20261102-step-mult public docs on affine.io (marker step-mult-20261102), trainer v8.3.3l (step_mult + optional knobs):
llms.txt, k1/spec/step-recipe.json, k1/replay/README.md, k1/replay/host/v833l/step_v833l.py (verbatim excerpts of the signed v8.3.3l host).
  build FROM_EPOCH OUTDIR    patched copies in OUTDIR (nothing live touched)
  apply FROM_EPOCH           build from the CURRENT live files, back them up to RA/20261102-step-mult/web/backup, install
  revert                     restore the backup (only files still equal to what apply installed; otherwise left as is + reported)
SM_ALLOWED=1,2,4 (env): accepted step_mult values."""
import json, os, re, shutil, sys, hashlib, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knobdoc as K
W = Path('/home/const/subnet120/affine/website'); D = Path('/home/const/subnet120-rewrite/state/root-audits/20261102-step-mult')
HD = Path(os.environ.get('SM_HOSTDIR', str(D / 'release-v833l-signed/trainer-host-v833l'))); WB = D / 'web'
FILES = {'llms.txt': 'llms.txt', 'step-recipe.json': 'k1/spec/step-recipe.json', 'README.md': 'k1/replay/README.md',
         'step_v833l.py': 'k1/replay/host/v833l/step_v833l.py'}
def rep(s, o, n, where):
    c = s.count(o)
    if c != 1: raise SystemExit('%s: anchor count %d: %r' % (where, c, o[:90]))
    return s.replace(o, n)
def llms(s, fe):
    smr, _ = K.sm_text()
    s = rep(s, 'Trainer v8.3.3j, gate', 'Trainer v8.3.3l, gate', 'llms')
    s = rep(s, 'v2 `train` (optional; either field may be omitted and takes its default):',
            'v2 `train` (optional; every field may be omitted and takes its default; the defaults are exactly today\'s trainer):', 'llms')
    s = rep(s, '| `train.kl` | 0.02 | 0.005 .. 0.1 | kl_coefficient of your turn\'s loss |\n',
            '| `train.kl` | 0.02 | 0.005 .. 0.1 | kl_coefficient of your turn\'s loss |\n'
            + K.md_table(lead='').replace(' |\n', ' |\n', 1), 'llms')
    s = rep(s, '- Out of range, not a number (string, true/false, null), or any other key under `train` makes the whole file invalid.',
            '- New from epoch %s (trainer v8.3.3l): step_mult and the knobs above. A file without them, or with every one at its default,\n'
            '  trains bit-identically to before. 0 in an lr_scale_* freezes that part of the model for your turn.\n'
            '- Out of range, not a number (string, true/false, null), a step_mult that is not an integer (2.0, "2"), a string that is\n'
            '  not one of the listed choices, or any other key under `train` makes the whole file invalid.' % fe, 'llms')
    s = rep(s, '- Your effective lr / kl are published only', '- Your effective lr / kl / step_mult / knobs are published only', 'llms')
    s = rep(s, '## 4. Training: the exact v8.3.3j step', '## 4. Training: the exact v8.3.3l step', 'llms')
    s = rep(s, 'miner/validator/trainer-v833j (+ engine/) and the live settings are in miner/validator/trainer-v833j/trainer-config.public.json.',
            'miner/validator/trainer-v833l (+ engine/) and the live settings are in miner/validator/trainer-v833l/trainer-config.public.json.', 'llms')
    s = rep(s, '    m    = lr / 5e-7 ;  n_t = min(m x n0_t, c_t)\n',
            '    m    = lr / 5e-7 ;  n_t = min(m x n0_t, c_t)\n'
            '    k    = step_mult ;  n_t = k x n_t                     <!-- step-mult-20261102 --> (from epoch %s; the cap scales with k)\n'
            '    knobs (all at default = the two lines above): eps = adam_eps and 0.01 = weight_decay in d_t;\n'
            '         a_t = lr_scale of t\'s group (attn / mlp / embed / lm_head; norms 1) x lr_scale_early (layers 0..15) or lr_scale_late (16..31)\n'
            '         per_tensor, both: n_t = k x min(m x a_t x n0_t, c_t)    global_norm: n_t = k x m x a_t x ||d_t|| (no c_t)\n'
            '         s = min(1, G / ||n||), G = 1.0 (per_tensor) or min(1.0, global_norm_cap) (global_norm, both)\n' % fe, 'llms')
    s = rep(s, '  At e368, 242 of 427 tensors were at the cap and carried 98.8% of the step norm, so lr 1e-6 barely moves the model more.\n',
            '  At e368, 242 of 427 tensors were at the cap and carried 98.8% of the step norm, so lr 1e-6 barely moves the model more.\n'
            '  step_mult k (proposed in #120 by duongna) multiplies every tensor\'s step, the capped ones included: it is the same as\n'
            '  scaling lr and the per-tensor cap by k together. Still one gradient and one step from the head (no extra trainer time).\n'
            '  The default step norm is about 0.023, so the global cap (1.0) does not fire for any allowed k. k = 1 is exactly the old step.\n'
            '  The loss knobs change the advantages (adv_center, adv_clip) or the weighting (loss_agg token_mean: every token\n'
            '  counts the same instead of every rollout). The step direction is always g_t / ||g_t||.\n', 'llms')
    s = s.replace('- the trainer v8.3.3j host + engine;', '- the trainer v8.3.3l host + engine;')
    s = s.replace('miner/validator/trainer-v833j + engine/. kit/train_step.py', 'miner/validator/trainer-v833l + engine/. kit/train_step.py')
    s, n = re.subn(r'Last updated: [0-9-]+ [0-9:]+ CR \(UTC-6\)\.', 'Last updated: %s CR (UTC-6).' % time.strftime('%Y-%m-%d %H:%M', time.gmtime(time.time() - 6 * 3600)), s, count=1)
    if n != 1: raise SystemExit('llms: Last updated anchor')
    return s
def recipe(s, fe):
    d = json.loads(s); o = d['optimizer']
    f = dict(step_mult=K.sm_field()); f.update(K.fields_json())
    o['train_params_v833l'] = dict(applies_from_epoch=int(fe), trainer='v8.3.3l = v8.3.3j + per-turn train.step_mult + optional step / loss knobs from an affine-batch-v2 file',
        fields=f, defaults_rule=K.DEFAULTS_RULE,
        step_mult_rule=('k = step_mult. With n_t the v8.3.3j per-tensor norm (n_t = min(m * n0_t, c_t), or n0_t when m = 1): n_t := k * n_t; '
                        's = min(1, sgd_update_norm_cap / sqrt(sum_t n_t^2)); Delta_t = fp32(s * n_t / ||g_t||_2) * g_t; M_C = M_H - Delta_t; '
                        'W_C = bf16_rne(M_C). Equivalent to scaling lr and sgd_tensor_rel_cap by k together.'),
        knob_rule=('Only when a step knob differs from its default (else the code below does not run): e_t = lr * (g_t / (|g_t| + adam_eps) + '
                   'weight_decay * W_t [2-D, not bias]); a_t = lr_scale_<group>(t) x (lr_scale_early if layer < 16 else lr_scale_late) (no layer: 1; '
                   'group other: 1); per_tensor / both: n_t = k * min(m * a_t * min(||e_t||, c_t), c_t); global_norm: n_t = k * m * a_t * ||e_t||; '
                   'G = sgd_update_norm_cap (per_tensor) else min(sgd_update_norm_cap, global_norm_cap); s = min(1, G / ||n||); Delta_t = fp32(s * n_t / ||g_t||) * g_t. '
                   'Loss knobs (when not default): adv_center none: A = r; group_norm: (r - mean) / (std + 1e-6) (upstream GRPO); adv_clip c: A = clamp(A, -c, c); '
                   'loss_agg token_mean: each active rollout loss x N * len_i / sum_j len_j.'),
        check=('turn.json training.train_params: step_mult, knobs, knobs_applied {step, loss}; step.path ends "+step_mult" (k > 1) and/or "+knobs" (step knobs); '
               'step.raw_update_norm_pre_knobs, global_cap_effective, frozen_tensors, group_step_norms'),
        code='/k1/replay/host/v833l/step_v833l.py', step_mult_proposed_by='duongna (#120)', published='GREEN turns only')
    return json.dumps(d, indent=1) + '\n'
def readme(s, fe):
    a = '## From epoch 362: trainer v8.3.3j (optional per-turn lr / kl) <!-- ff-train-params-20261031 -->'
    return rep(s, a, ('## From epoch %s: trainer v8.3.3l (optional per-turn step_mult + step / loss knobs) <!-- step-mult-20261102 -->\n\n'
        'An affine-batch-v2 file may also set `train.step_mult` and the knobs below. All default to the v8.3.3j step; a file without\n'
        'them (or with every one at its default) trains bit-identically to before. With n_t the v8.3.3j per-tensor step norm (below),\n'
        'step_mult k uses n_t := k * n_t before the global cap (lr and the per-tensor cap scale together, one gradient, one step).\n\n'
        % fe) + K.md_table() + ('\nThe effective values are in turn.json `training.train_params` (GREEN turns only). Code: `host/v833l/step_v833l.py` (verbatim\n'
        'from the live host). Recipe: `/k1/spec/step-recipe.json` -> `optimizer.train_params_v833l`.\n\n') + a, 'README')
def block(L, start, end_pred, title):
    i = next(n for n, l in enumerate(L) if start in l)
    j = next(n for n in range(i + 1, len(L)) if end_pred(L[n]))
    return '# --- %s (lines %d-%d) ---\n' % (title, i + 1, j) + '\n'.join(L[i:j]) + '\n\n'
def excerpt():
    L = (HD / 'mimo_host_trainer_v83.py').read_text().split('\n'); B = (HD / 'v8211_batched.py').read_text().split('\n')
    return ('"""Excerpts of the live trainer host v8.3.3l (verbatim lines of mimo_host_trainer_v83.py and v8211_batched.py).\n'
            'v8.3.3l = v8.3.3j (/k1/replay/host/v833j/step_v833j.py) + step_mult + the optional knobs. With every knob at its default\n'
            'TURN[\'opt\'] and KNOBS are None and none of the knob code runs."""\n\n'
            + block(L, '# v8.3.3l knobs (RA/20261102-step-mult)', lambda l: l.startswith('TURN_PARAMS'), 'knob spec + tensor groups')
            + block(L, "k_mult = TURN.get('step_mult', 1)", lambda l: l.strip().startswith('changed = 0;'), '_fresh_adam_step, after the v8.3.3j lr block')
            + block(L, "k_mult = tp.get('step_mult', 1)", lambda l: 'if isinstance(out, dict):' in l, 'train_update_v833j, turn params')
            + block(B, 'KNOBS = None', lambda l: l.startswith('DROP_UNRESOLVED'), 'v8211_batched loss knobs'))
def build(fe, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / 'llms.txt').write_text(llms((W / 'llms.txt').read_text(), fe))
    (out / 'step-recipe.json').write_text(recipe((W / 'k1/spec/step-recipe.json').read_text(), fe))
    (out / 'README.md').write_text(readme((W / 'k1/replay/README.md').read_text(), fe))
    (out / 'step_v833l.py').write_text(excerpt())
    return out
def h(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'build': build(sys.argv[2], sys.argv[3]); print('built')
    elif cmd == 'apply':
        if (WB / 'APPLIED.json').exists(): raise SystemExit('already applied')
        nx = build(sys.argv[2], WB / 'next'); bk = WB / 'backup'
        if bk.exists(): shutil.move(str(bk), str(WB / ('backup-old-%d' % time.time())))
        bk.mkdir(parents=True); inst = {}
        for n, rel in FILES.items():
            dst = W / rel
            if dst.exists(): shutil.copy2(dst, bk / n)
            dst.parent.mkdir(parents=True, exist_ok=True); tmp = dst.with_name(dst.name + '.sm-new'); shutil.copy2(nx / n, tmp); tmp.chmod(0o644); tmp.replace(dst)
            inst[rel] = h(dst)
        (WB / 'APPLIED.json').write_text(json.dumps(dict(fe=int(sys.argv[2]), installed=inst, sm_allowed=list(K.SM_ALLOWED)), indent=1)); print('applied')
    elif cmd == 'revert':
        a = json.loads((WB / 'APPLIED.json').read_text()); bk = WB / 'backup'
        for n, rel in FILES.items():
            dst = W / rel
            if not dst.exists() or h(dst) != a['installed'].get(rel): print('left as is (changed since apply):', rel); continue
            if (bk / n).exists(): tmp = dst.with_name(dst.name + '.sm-new'); shutil.copy2(bk / n, tmp); tmp.replace(dst)
            else: dst.unlink()
        (WB / 'APPLIED.json').rename(WB / ('REVERTED-%d.json' % time.time())); print('reverted')
