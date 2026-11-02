#!/usr/bin/env python3
"""RA/20261102-step-mult: FF-PUBLIC.json (mining.json batch_file) + train.step_mult + the v8.3.3l optional knobs.
usage: SM_ALLOWED=1,2,4 make_ff_public_l.py IN OUT"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knobdoc as K
d = json.load(open(sys.argv[1])); b = d['batch_file']; t = b['train']
def rep(obj, key, old, new):
    s = obj[key]; assert s.count(old) == 1, (key, old); obj[key] = s.replace(old, new)
smr, smw = K.sm_text(); names = ', '.join(k[0] for k in K.KNOBS)
rep(d, 'protocol', '(v2 from 2026-10-31: optional per-turn lr and kl)',
    '(v2 from 2026-10-31: optional per-turn lr and kl; from 2026-11-02 also step_mult and the optional optimizer / loss knobs)')
d['notice_after_cutoff'] += (' New (2026-11-02, trainer v8.3.3l): optional train fields step_mult (%s) and %s. All default to today\'s '
                             'trainer, so a file without them trains exactly as before (batch_file.train.fields).' % (smr, names))
b['schema']['train'] = ('v2 only, optional: {"lr": number, "kl": number, "step_mult": integer, plus any of the knobs in batch_file.train.fields}, '
                        'all optional. Omitted field = default (= today\'s trainer). Not allowed in a v1 file.')
rep(b['rules'], 7, 'kl in [0.005, 0.1] (default 0.02); both optional;',
    'kl in [0.005, 0.1] (default 0.02), step_mult %s (default 1), and the knobs in batch_file.train.fields (each with its range and default); all optional;' % smr)
rep(b['steps'], 1, 'to also set lr / kl', 'to also set lr / kl / step_mult / the knobs')
t['fields']['step_mult'] = K.sm_field(
    meaning=('step multiplier k (proposed in #120 by duongna): the whole per-tensor step is multiplied by k, i.e. lr and the per-tensor cap c_t '
             'are scaled together. Still one gradient and one step from the head, so no extra trainer time. k = 1 is the previous step exactly.'
             + ('' if K.SM_MAX > 1 else ' Only 1 is accepted for now.')))
t['fields'].update(K.fields_json())
t['defaults_rule'] = K.DEFAULTS_RULE
t['bounds'] = ('inclusive. A value outside its range, a non-number (string, bool, null) where a number is expected, a step_mult that is not an '
               'integer (2.0, "2", true), a string that is not one of the listed choices, or any other key under "train" makes the whole file '
               'invalid: you are skipped for that turn (not charged, you keep your place). Values are never clamped.')
t['math'] += (' step_mult k (trainer v8.3.3k/l): n_t is multiplied by k before the global cap, n_t(k) = k * n_t, so a tensor at the per-tensor cap moves '
              'k * c_t (the cap scales with k); s = min(1, 1.0 / ||n(k)||) is unchanged (the default step norm is about 0.023). k = 1 is exactly the previous step. '
              'Knobs (trainer v8.3.3l): e_t = lr * (g/(|g| + adam_eps) + weight_decay * W [2-D, not bias]); a_t = group scale (lr_scale_attn / mlp / embed / '
              'lm_head, other tensors 1) x lr_scale_early (layers 0..15) or lr_scale_late (16..31); per_tensor / both: n_t = k * min(m * a_t * min(||e_t||, c_t), c_t); '
              'global_norm: n_t = k * m * a_t * ||e_t||; G = 1.0 (per_tensor) or min(1.0, global_norm_cap); s = min(1, G / ||n||); Delta_t = s * n_t * g_t / ||g_t||. '
              'Loss: adv_center / adv_clip set the GRPO advantage; loss_agg token_mean weighs every token equally instead of every rollout. '
              'With every knob at its default the v8.3.3j/k code path runs unchanged.')
rep(t, 'unchanged', 'per-tensor cap, global norm cap and gradient clip',
    'per-tensor cap and global norm cap (except through step_mult / the knobs), gradient clip')
rep(t, 'published', 'The effective lr and kl of a turn', 'The effective lr, kl, step_mult and knobs of a turn')
b['example_v2']['train'] = {'lr': 8e-07, 'kl': 0.01, 'step_mult': 2 if K.SM_MAX > 1 else 1, 'lr_scale_embed': 0.0, 'adv_center': 'group_norm'}
rep(b['kit'], 'about', 'the live trainer (v8.3.3j host + engine)', 'the live trainer (v8.3.3l host + engine)')
open(sys.argv[2], 'w').write(json.dumps(d, indent=1) + '\n')
print('ok')
