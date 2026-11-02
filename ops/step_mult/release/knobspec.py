# RA/20261102-step-mult v8.3.3l: optional per-turn trainer knobs (Jacob 2026-11-02 11:19 CR). Shared text; pasted verbatim into
# ff_format.py, affine_batch.py and ff_trainer.py by patch scripts. kind, bounds/choices (inclusive), default (= today's trainer).
KNOB_SPEC = {
    'lr_scale_attn':    ('number', (0.0, 2.0), 1.0),     # self_attn.* and linear_attn.* tensors
    'lr_scale_mlp':     ('number', (0.0, 2.0), 1.0),     # mlp.* tensors
    'lr_scale_embed':   ('number', (0.0, 2.0), 1.0),     # embed_tokens
    'lr_scale_lm_head': ('number', (0.0, 2.0), 1.0),     # lm_head
    'lr_scale_early':   ('number', (0.0, 2.0), 1.0),     # every tensor of layers 0..15 (x the group scale)
    'lr_scale_late':    ('number', (0.0, 2.0), 1.0),     # every tensor of layers 16..31 (x the group scale)
    'adam_eps':         ('number', (1e-10, 1e-3), 1e-08),
    'weight_decay':     ('number', (0.0, 0.05), 0.01),
    'clamp_mode':       ('choice', ('per_tensor', 'global_norm', 'both'), 'per_tensor'),
    'global_norm_cap':  ('number', (0.005, 1.0), 1.0),   # used by clamp_mode global_norm / both
    'loss_agg':         ('choice', ('seq_mean', 'token_mean'), 'seq_mean'),
    'adv_center':       ('choice', ('mean', 'none', 'group_norm'), 'mean'),
    'adv_clip':         ('number', (0.1, 100.0), None),  # omitted = no clip
}
KNOB_DEFAULTS = {k: v[2] for k, v in KNOB_SPEC.items()}


def knob_value(k, v):
    """-> the effective value of one knob, or raise ValueError(reason). Never clamps (same policy as lr / kl)."""
    import json, math
    kind, b, _ = KNOB_SPEC[k]
    if kind == 'number':
        if type(v) not in (int, float) or not math.isfinite(v) or not b[0] <= v <= b[1]:
            raise ValueError('train.%s must be a number in [%r, %r] (got %s)' % (k, b[0], b[1], json.dumps(v)[:40]))
        return float(v)
    if type(v) is not str or v not in b:
        raise ValueError('train.%s must be one of %s (got %s)' % (k, ', '.join(b), json.dumps(v)[:40]))
    return v
