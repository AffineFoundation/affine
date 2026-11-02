#!/usr/bin/env python3
"""RA/20261102-step-mult: host free-file files + client: optional train knobs (v8.3.3l). usage: patch_host_knobs.py SRC_DIR OUT_DIR KNOBSPEC.py
SRC_DIR = stage/release (step_mult already applied). Patches ff_format.py, ff_lifecycle.py, affine_batch.py; copies the rest."""
import shutil, sys
from pathlib import Path
S, O, KS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]).read_text()
O.mkdir(parents=True, exist_ok=True)
for f in S.iterdir():
    if f.is_file(): shutil.copy2(f, O / f.name)
SPEC = 'KNOB_SPEC = {' + KS.split('KNOB_SPEC = {', 1)[1]
def edit(n, pairs):
    p = O / n; s = p.read_text()
    for o, nw in pairs:
        c = s.count(o)
        if c != 1: raise SystemExit('%s: anchor count %d: %r' % (n, c, o[:100]))
        s = s.replace(o, nw)
    p.write_text(s)
DOC = ('  * knobs (trainer v8.3.3l, Jacob 2026-11-02 11:19 CR): optional "train" keys, each defaulting to today\'s trainer: lr_scale_attn /\n'
       '    lr_scale_mlp / lr_scale_embed / lr_scale_lm_head / lr_scale_early / lr_scale_late [0, 2] (1), adam_eps [1e-10, 1e-3] (1e-8),\n'
       '    weight_decay [0, 0.05] (0.01), clamp_mode per_tensor|global_norm|both (per_tensor), global_norm_cap [0.005, 1] (1),\n'
       '    loss_agg seq_mean|token_mean (seq_mean), adv_center mean|none|group_norm (mean), adv_clip [0.1, 100] (none). Out of range,\n'
       '    wrong type or unknown key = invalid file (never clamped), like lr.\n')
edit('ff_format.py', [
  ('  * "step_mult" (RA/20261102-step-mult; duongna\'s #120 proposal, Jacob 2026-11-02): optional INTEGER 1..8 (default 1) inside "train".\n',
   '  * "step_mult" (RA/20261102-step-mult; duongna\'s #120 proposal, Jacob 2026-11-02): optional INTEGER 1..8 (default 1) inside "train".\n' + DOC),
  ("TRAIN_KEYS = ('lr', 'kl', 'step_mult')\n", SPEC + "\nTRAIN_KEYS = ('lr', 'kl', 'step_mult') + tuple(KNOB_SPEC)\n"),
  ("    need(not extra, 'train keys must be among lr, kl, step_mult (got %s)' % ', '.join(map(repr, extra))[:100])\n",
   "    need(not extra, 'train keys must be among %s (got %s)' % (', '.join(TRAIN_KEYS), ', '.join(map(repr, extra))[:100]))\n"),
  ("        out['step_mult'] = v\n    return out, sorted(k for k in TRAIN_KEYS if k in value)\n",
   "        out['step_mult'] = v\n"
   "    for k in KNOB_SPEC:                                # v8.3.3l knobs: effective value (default when omitted)\n"
   "        if k in value:\n"
   "            try: out[k] = knob_value(k, value[k])\n"
   "            except ValueError as e: raise Invalid(str(e))\n"
   "        else:\n"
   "            out[k] = KNOB_DEFAULTS[k]\n"
   "    return out, sorted(k for k in TRAIN_KEYS if k in value)\n")])
edit('ff_lifecycle.py', [
  ("        payload['train']['step_mult'] = int(turn['train'].get('step_mult', 1))     # RA/20261102-step-mult: trainer v8.3.3k\n",
   "        payload['train']['step_mult'] = int(turn['train'].get('step_mult', 1))     # RA/20261102-step-mult: trainer v8.3.3k\n"
   "        kd = getattr(ff_format, 'KNOB_SPEC', None)                                 # trainer v8.3.3l knobs (all defaults = today)\n"
   "        if kd is not None:\n"
   "            payload['train']['knobs'] = {k: (None if turn['train'].get(k, ff_format.KNOB_DEFAULTS[k]) is None else\n"
   "                                             float(turn['train'].get(k, ff_format.KNOB_DEFAULTS[k])) if kd[k][0] == 'number' else\n"
   "                                             str(turn['train'].get(k, ff_format.KNOB_DEFAULTS[k]))) for k in kd}\n")])
edit('affine_batch.py', [
  ("STEP_MULT_ALLOWED = (1, 2, 3, 4, 5, 6, 7, 8)\n", "STEP_MULT_ALLOWED = (1, 2, 3, 4, 5, 6, 7, 8)\n" + SPEC + "\n"),
  ("        if set(t) - set(TRAIN_BOUNDS) - {'step_mult'}: return False, 'train keys must be among lr, kl, step_mult'\n",
   "        if set(t) - set(TRAIN_BOUNDS) - {'step_mult'} - set(KNOB_SPEC): return False, 'train keys must be among lr, kl, step_mult, ' + ', '.join(KNOB_SPEC)\n"),
  ("            train['step_mult'] = t['step_mult']\n",
   "            train['step_mult'] = t['step_mult']\n"
   "        for k in KNOB_SPEC:\n"
   "            if k in t:\n"
   "                try: train[k] = knob_value(k, t[k])\n"
   "                except ValueError as e: return False, str(e)\n"),
  ("        v['version'], kept, n, caps['max_tasks_per_miner'], caps['max_rollouts_per_miner'], train['lr'], train['kl'], train['step_mult'],\n",
   "        v['version'], kept, n, caps['max_tasks_per_miner'], caps['max_rollouts_per_miner'], train['lr'], train['kl'], train['step_mult'],\n"
   "        ''.join(', %s %s' % (k, train[k]) for k in KNOB_SPEC if k in train) + (\n"),
  ("        '' if kept == len(gs) else '; %d prompts beyond the cap are dropped' % (len(gs) - kept))\n",
   "        '' if kept == len(gs) else '; %d prompts beyond the cap are dropped' % (len(gs) - kept)))\n"),
  ("    return True, 'ok (%s): %d prompts, %d rollouts will be trained (cap %d prompts / %d rollouts), lr %g, kl %g, step_mult %d%s' % (",
   "    return True, 'ok (%s): %d prompts, %d rollouts will be trained (cap %d prompts / %d rollouts), lr %g, kl %g, step_mult %d%s' % (")])
s = (O / 'affine_batch.py').read_text()
s = s.replace("Formats: affine-batch-v1, and affine-batch-v2 = v1 + an optional top-level \"train\"",
              "Formats: affine-batch-v1, and affine-batch-v2 = v1 + an optional top-level \"train\" (also the v8.3.3l knobs, KNOB_SPEC below)", 1)
(O / 'affine_batch.py').write_text(s)
print('ok')
