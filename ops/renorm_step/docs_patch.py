#!/usr/bin/env python3
"""RA/20261031-renorm-step-live: llms.txt + k1/spec/step-recipe.json + k1/replay (README, host/v833i/) for trainer v8.3.3i.
usage: docs_patch.py FROM_EPOCH   (idempotent: marker / keys; backups *.bak-pre-renorm-step-<ts>)"""
import json, os, shutil, sys, time
from pathlib import Path
W = Path('/home/const/subnet120/affine/website'); F = int(sys.argv[1]); ts = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
REL = Path('/home/const/subnet120-rewrite/state/root-audits/20261031-renorm-step-live/release-v833i-signed/trainer-host-v833i')
MARK = '<!-- renorm-step-20261031 -->'
def wr(p, s):
    t = Path(str(p) + '.tmp'); t.write_text(s); os.chmod(t, 0o644); os.replace(t, p)
L = W / 'llms.txt'; s = L.read_text()
if MARK not in s:
    shutil.copy2(L, str(L) + '.bak-pre-renorm-step-' + ts)
    old = '''- One AdamW step from zero optimizer state from the published head (lr fixed by the validator,
  see training.optimizer), GRPO loss with one group per prompt (training.loss), the trainer recomputes
  every log-prob from the head. No continuous training: every turn starts from the published head.'''
    new = '''- One stateless step from the published head and its fp32 master (lr 5e-7 fixed by the validator,
  see training.optimizer), GRPO loss with one group per prompt (training.loss), the trainer recomputes
  every log-prob from the head. No continuous training: every turn starts from the published head.
- Direction step (from epoch %d, trainer v8.3.3i) %s: every tensor moves by exactly the norm of the old capped
  fresh-AdamW step (same lr 5e-7, same 2e-5 per-tensor relative cap, same global cap, computed from the fp32
  master), but in the direction of its gradient g_t/||g_t|| instead of sign(g):
  `Delta_t = fp32(s * r_t * ||d_t|| / ||g_t||) * g_t`, `M_C = M_H - Delta_t`, candidate = bf16_rne(M_C).
  Recipe: https://affine.io/k1/spec/step-recipe.json -> optimizer.direction_v833i.
- Support renorm (from epoch %d): the policy-gradient term uses log q of your sampled token renormalized over
  the trainer's own top-20 then top-p 0.95 support (the manifest sampling settings, recomputed from the head's
  logits). A sampled token outside that support is dropped from the PG term only; it still counts in the KL
  term and the derail mask. Files are never rejected for it (format-only validation). The per-turn
  outside-support token fraction is published: https://affine.io/training.json -> support_renorm.last_turn
  (also mining.json training.support_renorm) and turn.json training.support_renorm. Sampling your rollouts
  with top_k 20 / top_p 0.95 / temperature 1.0 from the head keeps every token inside the support.''' % (F, MARK, F)
    assert s.count(old) == 1; s = s.replace(old, new, 1)
    wr(L, s)
R = W / 'k1' / 'spec' / 'step-recipe.json'; d = json.loads(R.read_text())
if 'direction_v833i' not in d['optimizer']:
    shutil.copy2(R, str(R) + '.bak-pre-renorm-step-' + ts)
    d['optimizer']['direction_v833i'] = dict(
        applies_from_epoch=F, trainer='v8.3.3i (engine sgd_direction: normgrad, sgd_fp32_master: true)', step_multiplier=1.0,
        norm='unchanged: d_t, r_t and s exactly as above, computed from the head fp32 master M_H; ||Delta_t|| = s * r_t * ||d_t||',
        direction='g_t / ||g_t||_2 per tensor (g = the clipped fp32 gradient) instead of g_t / (|g_t| + eps) ~ sign(g_t); no global normalization',
        update='Delta_t = fp32(s * r_t * ||d_t||_2 / ||g_t||_2) * g_t (fp64 norms; Delta_t = 0 if ||g_t|| = 0); M_C = M_H - Delta_t; W_C = bf16_round_nearest_even(M_C)',
        check='sqrt(sum_t ||Delta_t||^2) == turn.json training.optimizer.sgd.raw_update_norm == fp32_master.update_norm_master',
        code='/k1/replay/host/v833i/normgrad_v833i.py', origin='Allan, catoneone/sn120-k1-fixes@230b111 code/bf16_exp.py Env.scales/new_weights (opt normgrad)',
        per_turn_stats='turn.json training.optimizer.sgd (sgd_direction, raw_update_norm, tensors_rel_capped, zero_gradient_tensors)')
    d['loss']['support_renorm_v833i'] = dict(
        applies_from_epoch=F, settings=dict(top_k=20, top_p=0.95, temperature=1.0), settings_rule='must equal the signed sampling settings (asserted by the trainer every turn)',
        support='top_k candidates of the trainer fp32 logits / temperature, softmax over them, keep cumsum(p) - p < top_p (the first always kept); no gradient; ties in torch.topk order',
        pg='log q = log_softmax(kept logits)[sampled token], differentiable through the kept logits; OLD = log q detached (ratio 1)',
        outside='a sampled token outside the support leaves the PG token mean of its rollout (numerator and denominator); it still counts in the k3 KL term and derail detection (full-vocab log p)',
        published='turn.json training.support_renorm (outside_tokens, loss_tokens, outside_fraction, max_rollout_outside_fraction) and per rollout trajectories[].support_outside_tokens / support_logq_mean; https://affine.io/training.json support_renorm.last_turn',
        code='/k1/replay/host/v833i/v8211_batched.py support_logq + _gradient_pass, /k1/replay/host/v833i/v83_fast_kernels.py, /k1/replay/host/v833i/v83_dp.py (Allan patch/support_renorm.diff, catoneone/sn120-k1-fixes@230b111)')
    wr(R, json.dumps(d, indent=1) + '\n')
H = W / 'k1' / 'replay' / 'host' / 'v833i'; H.mkdir(mode=0o755, exist_ok=True)
for f in ('v8211_batched.py', 'v83_fast_kernels.py', 'v83_dp.py'):
    shutil.copyfile(REL / f, H / f); os.chmod(H / f, 0o644)
src = (REL / 'mimo_host_trainer_v83.py').read_text()
i = src.index('def _fresh_adam_delta'); j = src.index('\n# v8.3.3f fp32 head master', i)
k = src.index('def _fresh_adam_step'); m = src.index('\ndef sgd_step', k)
wr(H / 'normgrad_v833i.py', '"""Excerpt of the live trainer host v8.3.3i (mimo_host_trainer_v83.py, verbatim): fresh-AdamW delta and the fp32-master step with\n'
   'sgd_direction normgrad (Delta_t = fp32(s * r_t * ||d_t|| / ||g_t||) * g_t). opt.masters = the head fp32 master; helper functions\n_fp32_prepare/_fp32_finish only load/verify/save the master."""\n'
   'FRESH_ADAM_EPS, FRESH_ADAM_WD = 1e-8, 0.01\n\n\n' + src[i:j] + '\n\n' + src[k:m] + '\n')
RM = W / 'k1' / 'replay' / 'README.md'; t = RM.read_text()
if MARK not in t:
    shutil.copy2(RM, str(RM) + '.bak-pre-renorm-step-' + ts)
    t = t.replace('## Files\n', '''## From epoch %d: trainer v8.3.3i (direction step + support renorm) %s

The step keeps each tensor's capped fresh-AdamW step norm but moves along g_t/||g_t||, from the head fp32 master, and the PG term
uses the support-renormalized log q (top-20, top-p 0.95). The exact live code is in `host/v833i/` (`normgrad_v833i.py`, `v8211_batched.py`,
`v83_fast_kernels.py`, `v83_dp.py`); the recipe is `/k1/spec/step-recipe.json` (`optimizer.direction_v833i`, `loss.support_renorm_v833i`).
`replay_k1.py` below still implements the sign step from the bf16 head (turns before epoch 274/%d). An independent 1-GPU replay of the
v8.3.3i recipe is Allan's `code/bf16_exp.py stream --arm fp32_renorm --opt normgrad --ff <file>` (catoneone/sn120-k1-fixes@230b111);
our 8-GPU vs 1-GPU check of the first v8.3.3i turn is reported in `results.json` (`v833i_8v1`) when it finishes.

## Files
''' % (F, MARK, F), 1)
    wr(RM, t)
print('ok', L.stat().st_size, R.stat().st_size, sorted(p.name for p in H.iterdir()))
