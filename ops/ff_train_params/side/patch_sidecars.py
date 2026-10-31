#!/usr/bin/env python3
"""RA/20261031-ff-train-params sidecar patches (anchor replacement, refuses on any anchor miss):
  training_params.py: + training.miner_train_params (fields, bounds, defaults, math, recent GREEN turns' lr/kl) while FF-TRAIN.json
                      is enabled; learning_rate_rule / learning_rate_choice / loss.kl_coefficient_rule texts.
  k1_transparency.py: free-file turn.json gets training.train_params, turn.train / train_given and dispatch.train for GREEN only."""
import sys
from pathlib import Path

def rep(s, old, new, name):
    if s.count(old) != 1: raise SystemExit('REFUSED: anchor %r in %s (count %d)' % (old[:60], name, s.count(old)))
    return s.replace(old, new)

TP = r'''

TP_DIR = RA / '20261031-ff-train-params'
TP_FLAG = TP_DIR / 'state/FF-TRAIN.json'
TP_FIELDS = dict(lr=dict(default=5e-07, min=1.25e-07, max=1e-06), kl=dict(default=0.02, min=0.005, max=0.1))
TP_MATH = ('m = lr / 5e-7. Per tensor t (fp32 master W_t): d_t = 5e-7 * (g_t / (|g_t| + 1e-8) + 0.01 * W_t * [W_t >= 2-D and not a bias]); '
           'c_t = 2e-5 * (1 - 1e-6) * ||W_t||; n0_t = min(||d_t||, c_t); n_t = min(m * n0_t, c_t); s = min(1, 1.0 / ||n||); '
           'W_t_new = W_t - s * n_t * g_t / ||g_t|| (normgrad direction, trainer v8.3.3j). m = 1 is exactly the v8.3.3i step. lr below the '
           'default scales every tensor by m; above it only tensors under the per-tensor cap grow, up to c_t (hard ceiling). '
           'kl replaces kl_coefficient in L = loss_scale * (1/N) * sum_i (1/K_i) * sum_t [pg + kl * k3] for that turn only.')


def tp_recent(start, n=20):
    """GREEN turns only: the effective lr / kl the trainer reported (TRAINING-RESULT training.train_params)."""
    out = []
    if type(start) is not int: return out
    for d in sorted((p for p in (STATE_ROOT / 'epochs').iterdir() if p.name.isdigit() and int(p.name) >= start), key=lambda p: -int(p.name)):
        dec = rj(GATE_STATE / 'epochs' / d.name / 'DECISION.json') if (GATE_STATE / 'epochs' / d.name / 'DECISION.json').exists() else {}
        if dec.get('decision') != 'green' or not (d / 'TRAINING-RESULT.json').exists(): continue
        r = rj(d / 'TRAINING-RESULT.json'); r = r.get('payload', r)
        tp = ((r.get('report') or {}).get('training') or {}).get('train_params') or {}
        if 'lr' not in tp: continue
        out.append(dict(epoch=int(d.name), lr=tp.get('lr'), kl=tp.get('kl'), given=tp.get('given'), defaults_used=tp.get('defaults_used'),
                        turn='https://affine.io/k1/turns/%s/turn.json' % d.name))
        if len(out) >= n: break
    return out


def train_params_public(d):
    """Jacob 2026-10-31 07:08 CR (pt902u), RA/20261031-ff-train-params: optional miner lr / kl in affine-batch-v2 files."""
    f = rj(TP_FLAG) if TP_FLAG.exists() else {}
    if f.get('enabled') is not True:
        return d
    d['miner_train_params'] = dict(
        format='affine-batch-v2', field='train', optional=True, fields=TP_FIELDS, from_epoch=f.get('from_epoch'),
        rule=('An affine-batch-v2 file may carry "train": {"lr": x, "kl": y} (both optional; omitted = default). Bounds inclusive; a value '
              'outside them or not a number makes the file invalid (skipped, not charged). Never clamped. affine-batch-v1 files train '
              'with the defaults. Everything else (per-tensor cap, global cap, gradient clip, overlong penalty, head, one step, batch cap, '
              'queue, gate) is fixed.'),
        math=TP_MATH, example={'version': 'affine-batch-v2', 'hotkey': '5YourHotkeySs58Address', 'train': {'lr': 8e-07, 'kl': 0.01}, 'groups': ['...']},
        published='per turn, GREEN only: k1/turns/<epoch>/turn.json training.train_params and recent_green_turns below; RED stays private',
        recent_green_turns=tp_recent(f.get('from_epoch')))
    o = d.get('optimizer')
    if isinstance(o, dict):
        o['learning_rate_rule'] = ('default %s; the turn miner may set lr in [1.25e-7, 1e-6] in an affine-batch-v2 file '
                                   '(training.miner_train_params)' % o.get('learning_rate'))
    k = d.get('k1')
    if isinstance(k, dict):
        k['learning_rate_choice'] = 'miner-chosen per turn via affine-batch-v2 train.lr / train.kl (training.miner_train_params)'
    lo = d.get('loss')
    if isinstance(lo, dict):
        lo['kl_coefficient_rule'] = ('default %s; the turn miner may set kl in [0.005, 0.1] in an affine-batch-v2 file '
                                     '(training.miner_train_params)' % lo.get('kl_coefficient'))
    return d
'''

def patch_tp(s):
    s = rep(s, "def build():\n    return free_file(_build())\n", "def build():\n    return train_params_public(free_file(_build()))\n", 'training_params')
    s = rep(s, "\n\ndef write():\n", TP + "\n\ndef write():\n", 'training_params')
    return s

def patch_k1t(s):
    s = rep(s, "    private = ('trajectories', 'admitted_groups', 'input_documents_sha256', 'data_sha256')\n",
               "    private = ('trajectories', 'admitted_groups', 'input_documents_sha256', 'data_sha256', 'train_params')   # ff-train-params: lr/kl GREEN only\n", 'k1_transparency')
    s = rep(s, "'capped_rollouts', 'file_sha256', 'file_bytes', 'head', 'at', 'skipped_not_ready_count')},\n",
               "'capped_rollouts', 'file_sha256', 'file_bytes', 'head', 'at', 'skipped_not_ready_count')}\n"
               "             | ({k: turn[k] for k in ('train', 'train_given') if k in turn} if green else {}),\n", 'k1_transparency')
    s = rep(s, "                      **({'selected': pay.get('selected'), 'signature': env.get('signature')} if green else {})),\n",
               "                      **({'selected': pay.get('selected'), 'signature': env.get('signature')} if green else {}),\n"
               "                      **({'train': pay.get('train')} if green and 'train' in pay else {})),\n", 'k1_transparency')
    s = rep(s, "        training={k: tr.get(k) for k in TRAIN_KEYS if k in tr and (green or k not in private)},\n",
               "        training={k: tr.get(k) for k in TRAIN_KEYS + ('train_params',) if k in tr and (green or k not in private)},\n", 'k1_transparency')
    return s

if __name__ == '__main__':
    which, src, dst = sys.argv[1:4]
    s = Path(src).read_text()
    out = patch_tp(s) if which == 'training_params' else patch_k1t(s) if which == 'k1_transparency' else None
    if out is None: raise SystemExit('which?')
    Path(dst).write_text(out); print('patched', which, '->', dst)
