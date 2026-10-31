#!/usr/bin/env python3
"""RA/20261031-ff-train-params public docs (anchor edits on the LIVE website files, marker ff-train-params-20261031):
llms.txt, k1/spec/step-recipe.json, k1/replay/README.md, k1/replay/replay_k1.py (kl from turn.json training.train_params),
k1/replay/host/v833j/step_v833j.py (verbatim excerpt of the live v8.3.3j step).
  build <website> <host-v833j dir> <outdir>   -> patched copies in outdir (nothing live touched)
"""
import ast, json, sys
from pathlib import Path
MK = '<!-- ff-train-params-20261031 -->'

def rep(s, old, new, name):
    if s.count(old) != 1: raise SystemExit('REFUSED: anchor %r in %s (count %d)' % (old[:70], name, s.count(old)))
    return s.replace(old, new)

V2 = '''
### affine-batch-v2: optional lr and kl (from 2026-10-31) ''' + MK + '''

A v2 file is a v1 file with `"version": "affine-batch-v2"` and an optional top-level `train`:

```json
{
  "version": "affine-batch-v2",
  "hotkey": "5YourHotkeySs58Address",
  "train": {"lr": 8e-7, "kl": 0.01},
  "groups": [ ...same as v1... ]
}
```

| field | default | allowed (inclusive) | what it does |
| --- | --- | --- | --- |
| `train.lr` | 5e-7 | 1.25e-7 .. 1e-6 | step size of your turn's one step |
| `train.kl` | 0.02 | 0.005 .. 0.1 | kl_coefficient of your turn's loss |

- Both fields are optional; an omitted field takes its default. `train` itself is optional in v2.
  v1 files keep working unchanged (they train with the defaults). `train` in a v1 file is invalid.
- Out of range, not a number (string, true/false, null) or any other key under `train` = the whole
  file is invalid (skipped, not charged, you keep your place). Values are never clamped.
- How lr enters (trainer v8.3.3j): m = lr / 5e-7. For each tensor, n0_t = the default step norm
  (min(||d_t||, c_t), c_t = 2e-5 x ||W_t|| the per-tensor cap); your step norm is n_t = min(m x n0_t, c_t),
  then the global cap 1.0 and the direction g_t/||g_t|| as before: Delta_t = s x n_t x g_t / ||g_t||.
  Below the default every tensor's step shrinks by exactly m. Above it only tensors still under the
  per-tensor cap grow, up to the cap (the cap stays a ceiling; with the default many tensors already
  sit at it, so lr 1e-6 moves the model less than 2x).
- How kl enters: it replaces kl_coefficient in
  L = loss_scale x (1/N) x sum_i (1/K_i) x sum_t [pg + kl x k3] for your turn only.
- Everything else is fixed: per-tensor cap, global cap and gradient clip, overlong penalty, starting
  head, one step per turn, batch cap (125 x 8), queue and gate.
- Your effective lr / kl are published with your turn only if it is GREEN (turn.json
  training.train_params, mining.json training.miner_train_params). RED: private like the file.
- Check locally: python3 affine_batch.py check --hotkey <ss58> batch.json (reports format, lr, kl).
'''

def llms(s):
    s = rep(s, '## 2. The batch file: affine-batch-v1\n', '## 2. The batch file: affine-batch-v1 (or v2 with optional lr / kl)\n', 'llms.txt')
    s = rep(s, '\n## 3. Upload\n', V2 + '\n## 3. Upload\n', 'llms.txt')
    s = rep(s, '(lr 5e-7 fixed by the validator,\n  see training.optimizer)',
               '(lr 5e-7 and kl 0.02 by default; an affine-batch-v2\n  file may set both within bounds, section 2; see training.optimizer)', 'llms.txt')
    s = rep(s, '- RED: decision, counts and the file\'s sha256 only; the file stays private.\n',
               '- RED: decision, counts and the file\'s sha256 only; the file stays private.\n'
               '- Your v2 train values (lr, kl) follow the file: public with a GREEN turn, private on RED.\n', 'llms.txt')
    return s

RECIPE = dict(
    applies_from_epoch=None, trainer='v8.3.3j = v8.3.3i + per-turn lr / kl from an affine-batch-v2 file',
    fields=dict(lr=dict(default=5e-07, min=1.25e-07, max=1e-06), kl=dict(default=0.02, min=0.005, max=0.1)),
    lr_rule=('m = lr / sgd_lr (sgd_lr = 5e-7). n0_t = r_t * ||d_t||_2 (the v8.3.3i per-tensor norm at sgd_lr, = min(||d_t||, c_t)); '
             'c_t = sgd_tensor_rel_cap * (1 - 1e-6) * ||M_H,t||_2; n_t = min(m * n0_t, c_t); s = min(1, sgd_update_norm_cap / sqrt(sum_t n_t^2)); '
             'Delta_t = fp32(s * n_t / ||g_t||_2) * g_t; M_C = M_H - Delta_t; W_C = bf16_rne(M_C). m == 1 runs the v8.3.3i code path unchanged.'),
    kl_rule='kl replaces config.kl_coefficient for that one update (L = loss_scale * (1/N) * sum_i (1/K_i) * sum_t [pg + kl * k3]); the reported config is the signed one',
    check=('turn.json training.train_params: lr, kl, lr_multiplier, kl_applied, step.raw_update_norm (== sqrt(sum n_t^2) == '
           'fp32_master.update_norm_master), step.raw_update_norm_default (the m = 1 norm), step.tensors_at_ceiling'),
    code='/k1/replay/host/v833j/step_v833j.py', published='GREEN turns only')

def recipe(s, from_epoch):
    d = json.loads(s)
    if 'train_params_v833j' in d['optimizer']: raise SystemExit('REFUSED: recipe already has train_params_v833j')
    r = dict(RECIPE, applies_from_epoch=from_epoch)
    d['optimizer']['train_params_v833j'] = r
    d['loss']['kl_coefficient_v833j'] = 'per turn: training.train_params.kl_applied (default 0.02; affine-batch-v2 train.kl in [0.005, 0.1])'
    return json.dumps(d, indent=1, ensure_ascii=False) + '\n'

README = '''## From epoch {E}: trainer v8.3.3j (optional per-turn lr / kl) ''' + MK + '''

An affine-batch-v2 file may set `train.lr` (1.25e-7 .. 1e-6, default 5e-7) and `train.kl` (0.005 .. 0.1, default 0.02). The effective
values are in turn.json `training.train_params` (GREEN turns only). To replay such a turn: use `kl_applied` as kl_coefficient in the loss,
and the step in `host/v833j/step_v833j.py`: m = lr / 5e-7, n_t = min(m * r_t * ||d_t||, 2e-5 * (1 - 1e-6) * ||M_H,t||),
s = min(1, 1.0 / ||n||), Delta_t = s * n_t * g_t / ||g_t||. With the defaults (m = 1, kl 0.02) the step is exactly v8.3.3i.
Recipe: `/k1/spec/step-recipe.json` -> `optimizer.train_params_v833j`. `replay_k1.py` reads `train_params.kl_applied` for the loss.

'''

def readme(s, from_epoch):
    return rep(s, '## From epoch 354: trainer v8.3.3i', README.replace('{E}', str(from_epoch)) + '## From epoch 354: trainer v8.3.3i', 'README.md')

def replay(s):
    return rep(s, "    cfg = TrainerConfig(**tr['config']['settings'])\n",
               "    cfg = TrainerConfig(**tr['config']['settings'])\n"
               "    if (tr.get('train_params') or {}).get('kl_applied') is not None:   # v8.3.3j: the turn's kl (affine-batch-v2 train.kl)\n"
               "        import dataclasses; cfg = dataclasses.replace(cfg, kl_coefficient=float(tr['train_params']['kl_applied'])).validate()\n"
               "        print(json.dumps(dict(stage='train-params', kl=cfg.kl_coefficient, lr_multiplier=tr['train_params'].get('lr_multiplier'),\n"
               "              note='lr: see host/v833j/step_v833j.py (this script implements the v8.3.3 sign step)')), flush=True)\n", 'replay_k1.py')

def excerpt(host):
    src = (Path(host) / 'mimo_host_trainer_v83.py').read_text(); tree = ast.parse(src); lines = src.splitlines(keepends=True)
    want = {'_fresh_adam_delta', '_fresh_adam_step', 'train_update_v833j'}; got = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in want and node.name not in got:
            got[node.name] = ''.join(lines[node.lineno - 1 - len(node.decorator_list):node.end_lineno])
    head = [l for l in lines if l.startswith('TURN = dict(') or l.startswith('TURN_PARAMS = dict(')]
    if set(got) != want or len(head) != 2: raise SystemExit('REFUSED: excerpt %s %d' % (sorted(got), len(head)))
    import textwrap
    return ('"""Excerpt of the live trainer host v8.3.3j (mimo_host_trainer_v83.py, verbatim): the fp32-master normgrad step with the\n'
            'per-turn lr multiplier (TURN[\'lr_multiplier\'] = lr / sgd_lr), and train_update_v833j, which applies a free-file turn\'s lr / kl\n'
            'to one update. Helpers (_fp32_prepare/_fp32_finish, train_update_v833i) are as in v8.3.3i (/k1/replay/host/v833i/)."""\n'
            'FRESH_ADAM_EPS, FRESH_ADAM_WD = 1e-8, 0.01\n' + ''.join(head) + '\n\n' + got['_fresh_adam_delta'] + '\n\n' + got['_fresh_adam_step']
            + '\n\n# inside install(): wraps train_update_v833i\n' + textwrap.dedent(got['train_update_v833j']))

if __name__ == '__main__':
    cmd, web, host, out = sys.argv[1:5]; fe = int(sys.argv[5]) if len(sys.argv) > 5 else None
    W, O = Path(web), Path(out); O.mkdir(parents=True, exist_ok=True)
    (O / 'llms.txt').write_text(llms((W / 'llms.txt').read_text()))
    (O / 'step-recipe.json').write_text(recipe((W / 'k1/spec/step-recipe.json').read_text(), fe))
    (O / 'README.md').write_text(readme((W / 'k1/replay/README.md').read_text(), fe if fe is not None else '?'))
    (O / 'replay_k1.py').write_text(replay((W / 'k1/replay/replay_k1.py').read_text()))
    (O / 'step_v833j.py').write_text(excerpt(host))
    print('built', sorted(p.name for p in O.iterdir()))
