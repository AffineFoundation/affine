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
