"""RA/20261102-step-mult: one public description of the optional affine-batch-v2 train fields (v8.3.3l), used by every doc
generator (mining.json, training.json, llms.txt, step-recipe, replay README, GitHub miner kit). Ranges/defaults = knobspec.KNOB_SPEC."""
import os
# step_mult accepted values (env SM_ALLOWED, e.g. "1,2,4"; Jacob 2026-11-02 11:48 CR: m in {1,2,4}). "1" -> field reserved.
SM_ALLOWED = tuple(sorted({int(x) for x in os.environ.get('SM_ALLOWED', '1').split(',')}))
assert SM_ALLOWED[0] == 1 and set(SM_ALLOWED) <= set(range(1, 9)), SM_ALLOWED
SM_MAX = max(SM_ALLOWED)
SM_LIST = ', '.join(map(str, SM_ALLOWED))
def sm_field(**extra):
    return dict(default=1, type='integer', choices=list(SM_ALLOWED), **extra)
KNOBS = [  # name, json type, range text, default text, min, max, choices, default, what it does
 ('lr_scale_attn', 'number', '0 .. 2', '1', 0.0, 2.0, None, 1.0,
  "multiplies the step of every attention tensor (self_attn.*, linear_attn.*); 0 freezes them"),
 ('lr_scale_mlp', 'number', '0 .. 2', '1', 0.0, 2.0, None, 1.0, "multiplies the step of every mlp.* tensor; 0 freezes them"),
 ('lr_scale_embed', 'number', '0 .. 2', '1', 0.0, 2.0, None, 1.0, "multiplies the step of embed_tokens; 0 freezes it"),
 ('lr_scale_lm_head', 'number', '0 .. 2', '1', 0.0, 2.0, None, 1.0, "multiplies the step of lm_head; 0 freezes it"),
 ('lr_scale_early', 'number', '0 .. 2', '1', 0.0, 2.0, None, 1.0,
  "multiplies the step of every tensor in decoder layers 0..15 (on top of the group scale); 0 freezes them"),
 ('lr_scale_late', 'number', '0 .. 2', '1', 0.0, 2.0, None, 1.0,
  "multiplies the step of every tensor in decoder layers 16..31 (on top of the group scale); 0 freezes them"),
 ('adam_eps', 'number', '1e-10 .. 1e-3', '1e-8', 1e-10, 1e-3, None, 1e-08,
  "eps in the normalized-gradient term g / (abs(g) + eps); larger eps shrinks the per-tensor step of tensors with small gradients "
  "(the step direction stays g / norm(g))"),
 ('weight_decay', 'number', '0 .. 0.05', '0.01', 0.0, 0.05, None, 0.01,
  "decoupled weight decay inside the per-tensor step norm (2-D non-bias tensors); it changes step sizes, not the direction"),
 ('clamp_mode', 'string', 'per_tensor, global_norm or both', 'per_tensor', None, None, ('per_tensor', 'global_norm', 'both'), 'per_tensor',
  "per_tensor: each tensor's step is capped at c_t = 2e-5 x norm(W_t) (today). global_norm: no per-tensor cap; the whole step "
  "is scaled down to global_norm_cap if larger. both: per-tensor cap, then the global cap at global_norm_cap"),
 ('global_norm_cap', 'number', '0.005 .. 1', '1', 0.005, 1.0, None, 1.0,
  "the global step-norm cap used by clamp_mode global_norm / both (ignored with per_tensor, which keeps the fixed cap 1.0)"),
 ('loss_agg', 'string', 'seq_mean or token_mean', 'seq_mean', None, None, ('seq_mean', 'token_mean'), 'seq_mean',
  "seq_mean: every rollout weighs the same (mean over its tokens, then over rollouts; today). token_mean: every token "
  "weighs the same (long rollouts weigh more)"),
 ('adv_center', 'string', 'mean, none or group_norm', 'mean', None, None, ('mean', 'none', 'group_norm'), 'mean',
  "advantage of a rollout: mean = reward - mean reward of its prompt group (today); none = the raw reward; "
  "group_norm = (reward - mean) / (std + 1e-6) within the group"),
 ('adv_clip', 'number', '0.1 .. 100', 'none (no clip)', 0.1, 100.0, None, None,
  "if set, advantages are clipped to [-adv_clip, +adv_clip] after centering; omit it for no clip (today)"),
]
def sm_text():
    if SM_ALLOWED == (1,):
        return ('integer, only 1 is accepted for now', 'reserved: multiplies the whole one-step update (lr and the per-tensor cap together); '
                'only 1 (the default) is accepted until a replay shows gains growing with it')
    return ('integer: %s' % SM_LIST, "multiplies the whole one-step update: lr and the per-tensor cap scale together, one gradient, no extra trainer time")
def md_table(lead='| field | default | range | what it does |\n|---|---|---|---|\n'):
    r, w = sm_text()
    s = lead + '| `train.step_mult` | 1 | %s | %s |\n' % (r, w)
    for n, t, rg, dt, *_x, desc in KNOBS:
        s += '| `train.%s` | %s | %s | %s |\n' % (n, dt, rg, desc)
    return s
def fields_json():
    out = {}
    for n, t, rg, dt, lo, hi, ch, d, desc in KNOBS:
        f = dict(type=t, default=d, meaning=desc)
        if ch: f['choices'] = list(ch)
        else: f.update(min=lo, max=hi)
        if n == 'adv_clip': f['default_meaning'] = 'omitted = no clip (null is not accepted)'
        out[n] = f
    return out
DEFAULTS_RULE = ('Every field is optional and defaults to today\'s trainer; a file without them, or with every one at its default, trains '
                 'bit-identically to before (checked on a real GREEN file). Bounds are inclusive; out of range, the wrong type, an unknown '
                 'choice or an unknown key makes the whole file invalid (never clamped).')
