import json, sys, copy
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src)); o = copy.deepcopy(d)
b = o['batch_file']
o['protocol'] = 'free-file affine-batch-v1 / affine-batch-v2 (v2 from 2026-10-31: optional per-turn lr and kl)'
o['notice_after_cutoff'] = d['notice_after_cutoff'] + ' New (2026-10-31): an affine-batch-v2 file may set the turn\'s lr and kl (batch_file.train).'
b['format'] = 'affine-batch-v1 or affine-batch-v2'
b['formats'] = ['affine-batch-v1', 'affine-batch-v2']
b['schema'] = dict(b['schema'])
b['schema']['version'] = '"affine-batch-v1" or "affine-batch-v2"'
b['schema']['train'] = ('v2 only, optional: {"lr": number, "kl": number}, both optional. Omitted field = default. '
                        'Not allowed in a v1 file.')
b['train'] = dict(
    fields=dict(lr=dict(default=5e-07, min=1.25e-07, max=1e-06, meaning='step size of the one training step (see math)'),
                kl=dict(default=0.02, min=0.005, max=0.1, meaning='kl_coefficient of the turn loss (weight of the k3 KL term to the reference)')),
    bounds='inclusive. A value outside its range, a non-number (string, bool, null) or any other key under "train" makes the whole file '
           'invalid: you are skipped for that turn (not charged, you keep your place). Values are never clamped.',
    math=('m = lr / 5e-7. Per tensor t (fp32 master W_t): d_t = 5e-7 * (g_t / (|g_t| + 1e-8) + 0.01 * W_t * [W_t >= 2-D and not a bias]); '
          'c_t = 2e-5 * (1 - 1e-6) * ||W_t||; n0_t = min(||d_t||, c_t) (the default-lr step norm); n_t = min(m * n0_t, c_t); '
          's = min(1, 1.0 / ||n||) over all tensors; W_t_new = W_t - s * n_t * g_t / ||g_t||. lr below the default shrinks every '
          'tensor\'s step by exactly m; lr above the default only grows tensors that are under the per-tensor cap c_t, up to it '
          '(the cap is a hard ceiling; many tensors already sit at it with the default). The default (and any v1 file) is exactly the '
          'previous step. kl replaces kl_coefficient in L = loss_scale * (1/N) * sum_i (1/K_i) * sum_t [pg + kl * k3] for that turn.'),
    unchanged='Everything else is fixed by the validator: per-tensor cap, global norm cap and gradient clip, overlong penalty, starting '
              'head, one step per turn, batch cap, queue and gate.',
    published='The effective lr and kl of a turn are published with the turn only if it is GREEN (k1/turns/<epoch>/turn.json '
              'training.train_params, mining.json training.miner_train_params.recent_green_turns). RED: private like the file.')
b['example_v2'] = dict(version='affine-batch-v2', hotkey=b['example']['hotkey'], train=dict(lr=8e-07, kl=0.01), groups=b['example']['groups'])
r = b['rules']
r[0] = ('UTF-8 JSON, at most 64 MiB, exactly the keys shown (v1: version, hotkey, groups; v2: the same plus the optional train). '
        'Prompts non-empty and distinct; at least one rollout per prompt, any number per prompt.')
r.insert(len(r) - 1, 'train (v2 only): lr in [1.25e-7, 1e-6] (default 5e-7), kl in [0.005, 0.1] (default 0.02); both optional; '
                     'out of range or not a number = invalid file (batch_file.train).')
s = b['steps']
s[1] = ('2. Write your file in the affine-batch-v1 format (batch_file.schema, batch_file.example, batch_file.rules), or affine-batch-v2 '
        'to also set lr / kl (batch_file.train, batch_file.example_v2): plain-text prompts, each with your rollouts (output text + reward).')
b['status'] = b['status'].replace('(valid, reason, prompts, rollouts; never content)', '(valid, reason, format, prompts, rollouts; never content or train values)')
json.dump(o, open(dst, 'w'), indent=1, ensure_ascii=False); open(dst, 'a').write('\n')
