"""v8.3.3h: trainer-side reader for free-file (affine-batch-v1) turns. Jacob 2026-10-30 pt898u, RA/20261030-free-file.

The lifecycle signs an 'sn120-ff-dispatch-v1' selection (same authority, same opening/manifest binding as the
legacy dispatch). Each selected row names one group document (canonical JSON, sha256-addressed, staged by the
unchanged worker fetch) holding the rendered prompt tokens and the miner's rollouts: tokens (EOS already applied
by the lifecycle's format check), stop ('eos' | 'max_tokens') and the miner's reward, used as given.
Groups may hold any number (>= 1) of rollouts. Training math is unchanged; the reward of a rollout is the miner's
reward plus the same soft overlong penalty term config.reward applies (v8211_batched.rollout_reward).
"""
from dataclasses import dataclass
import json
from pathlib import Path

from subnet.backend_jobs import signed
from subnet.mimo_grpo.data import InputBinding, TaskGroup, Trajectory, digest, sha

VERSION = 'sn120-ff-dispatch-v1'
# v8.3.3j (RA/20261031-ff-train-params, Jacob pt902u): optional selection key 'train' = {lr, kl, given, defaults} (lifecycle fills the
# defaults for keys the file omitted). Same bounds as the lifecycle format check (ff_format.TRAIN_BOUNDS); checked again here.
TRAIN_BOUNDS = {'lr': (1.25e-07, 1e-06), 'kl': (0.005, 0.1)}
# v8.3.3k (RA/20261102-step-mult): optional 'step_mult' (int 1..8; absent = 1) in 'train' and in 'given'; defaults may carry step_mult 1.
STEP_MULT_ALLOWED = (1, 2, 3, 4, 5, 6, 7, 8)
DOCUMENT = 'sn120-ff-group-v1'
MAX_DOCUMENT_BYTES = 128 * 1024 * 1024


def need(ok, why):
    if not ok:
        raise ValueError('ff: ' + why)


@dataclass(frozen=True)
class RewardedTrajectory(Trajectory):
    reward: float = 0.0


@dataclass(frozen=True)
class FreeGroup(TaskGroup):
    def validate(self, size=None, *, vocab_size, eos_token_ids, unresolved_failures=False):
        """Format-only: any number of rollouts (size ignored), no outcome mix, finite rewards, EOS semantics."""
        need(type(self.task_id) is str and self.task_id and sha(self.admission_sha256) and self.prompt, 'task, prompt, receipt')
        need(self.trajectories and [r.nonce for r in self.trajectories] == list(range(len(self.trajectories))), 'nonces 0..N-1')
        need(all(type(t) is int and 0 <= t < vocab_size for t in self.prompt), 'prompt tokens')
        for r in self.trajectories:
            need(isinstance(r, RewardedTrajectory) and type(r.reward) is float and r.reward == r.reward
                 and abs(r.reward) != float('inf'), 'finite reward')
            need(r.tokens and all(type(t) is int and 0 <= t < vocab_size for t in r.tokens), 'output tokens')
            need(r.outcome in ('correct', 'wrong', 'unresolved'), 'label')
            if r.stop == 'max_tokens':
                need(r.outcome == 'unresolved' and not any(t in eos_token_ids for t in r.tokens), 'capped rollout has no EOS')
            else:
                need(r.stop == 'eos' and r.tokens[-1] in eos_token_ids and not any(t in eos_token_ids for t in r.tokens[:-1]),
                     'output ends at its first EOS')


def is_ff(dispatch):
    try:
        return dispatch['payload']['version'] == VERSION
    except Exception:
        return False


def label(stop, reward):
    """Diagnostic label only (reports, cap policy): capped -> unresolved; else reward > 0 -> correct."""
    return 'unresolved' if stop == 'max_tokens' else ('correct' if reward > 0 else 'wrong')


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


def train_params(selection, binding):
    """The signed selection's 'train' -> dict(lr, kl, given, defaults, epoch, checkpoint), or None when absent."""
    t = selection.get('train')
    if t is None:
        return None
    need(type(t) is dict and set(t) - {'step_mult', 'knobs'} == {'lr', 'kl', 'given', 'defaults'}, 'train keys')
    kn = t.get('knobs', dict(KNOB_DEFAULTS))                # v8.3.3l
    need(type(kn) is dict and set(kn) == set(KNOB_SPEC), 'train.knobs keys')
    for kk, vv in kn.items():
        if vv is None and KNOB_SPEC[kk][2] is None: continue
        try: ok = knob_value(kk, vv) == vv and type(vv) is (float if KNOB_SPEC[kk][0] == 'number' else str)
        except ValueError: ok = False
        need(ok, 'train.knobs.%s' % kk)
    km = t.get('step_mult', 1)
    need(type(km) is int and km in STEP_MULT_ALLOWED, 'train.step_mult in %r' % (STEP_MULT_ALLOWED,))
    for k, (lo, hi) in TRAIN_BOUNDS.items():
        need(type(t[k]) is float and lo <= t[k] <= hi, 'train.%s in [%r, %r]' % (k, lo, hi))
    need(type(t['given']) is list and set(t['given']) <= set(TRAIN_BOUNDS) | {'step_mult'} | set(KNOB_SPEC) and len(set(t['given'])) == len(t['given']), 'train.given')
    d = t['defaults']
    need(type(d) is dict and set(d) - {'step_mult'} == set(TRAIN_BOUNDS) and all(type(d[x]) is float for x in TRAIN_BOUNDS)
         and type(d.get('step_mult', 1)) is int and d.get('step_mult', 1) == 1, 'train.defaults')
    return dict(lr=t['lr'], kl=t['kl'], step_mult=km, knobs=dict(kn), given=list(t['given']), defaults=dict(lr=d['lr'], kl=d['kl']),
                epoch=binding.epoch, checkpoint=binding.checkpoint)


def prepare(opening, dispatch, *, authority, documents_root, runtime, optimizer_step=0, max_groups=None):
    """Same contract and return shape as subnet.mimo_grpo.prepared.prepare_signed_dispatch."""
    manifest, selection = signed(opening, authority), signed(dispatch, authority)
    need(selection.get('version') == VERSION and selection.get('epoch') == manifest['epoch']
         and selection.get('run_id') == manifest['run_id'] and selection.get('checkpoint') == runtime.checkpoint_id
         and manifest['checkpoint']['id'] == runtime.checkpoint_id and selection.get('manifest_sha256') == digest(manifest),
         'dispatch must bind the signed opening and the executed checkpoint')
    need(manifest.get('mimo_runtime') == runtime.spec.to_dict(), 'opening must pin the executed runtime')
    need(max_groups is None, 'no diagnostic narrowing')
    rows = selection['selected']
    need(type(rows) is list and rows and selection['selected_task_count'] == len(rows)
         and selection['selected_rollout_count'] == sum(r['rollouts'] for r in rows), 'selection counts')
    root = Path(documents_root).resolve()
    binding = InputBinding(manifest['run_id'], manifest['epoch'], runtime.checkpoint_id, digest(manifest['group_contract']), optimizer_step)
    eos = set(runtime.eos_token_ids)
    groups, receipts = [], []
    for i, row in enumerate(rows):
        checksum = row['document']['sha256']
        need(sha(checksum) and row['index'] == i, 'row %d' % i)
        path = root / (checksum + '.json')
        need(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_DOCUMENT_BYTES, 'staged document %d' % i)
        doc = json.loads(path.read_bytes())
        need(digest(doc) == checksum, 'document digest %d' % i)
        need(doc.get('version') == DOCUMENT and doc['epoch'] == manifest['epoch'] and doc['checkpoint'] == runtime.checkpoint_id
             and doc['hotkey'] == selection['hotkey'] and doc['group_index'] == i and len(doc['rollouts']) == row['rollouts']
             and doc['prompt_sha256'] == row['prompt_sha256'], 'document %d binding' % i)
        trajectories = tuple(RewardedTrajectory(nonce=n, tokens=tuple(r['tokens']), outcome=label(r['stop'], r['reward']),
                                                stop=r['stop'], reward=float(r['reward']))
                             for n, r in enumerate(doc['rollouts']))
        group = FreeGroup(task_id=digest(dict(ff=VERSION, epoch=manifest['epoch'], hotkey=doc['hotkey'], group_index=i,
                                              prompt_sha256=doc['prompt_sha256'])),
                          prompt=tuple(doc['prompt_tokens']), trajectories=trajectories, admission_sha256=checksum)
        group.validate(vocab_size=runtime.vocab_size, eos_token_ids=eos)
        groups.append(group)
        receipts.append(dict(version=DOCUMENT, document_sha256=checksum, group_index=i, hotkey=doc['hotkey']))
    need(len({g.task_id for g in groups}) == len(groups), 'distinct groups')
    return dict(manifest=manifest, dispatch=selection, binding=binding, groups=tuple(groups), receipts=receipts,
                diagnostic_omitted_groups=0)
