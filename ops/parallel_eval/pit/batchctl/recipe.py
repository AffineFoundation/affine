"""What one lease runs on its box, as documents. This module does no I/O.

A lease has a directory of its own on the box. Three documents go there:

``job.json``     for ``box/job.py``: what the box must be, where the files of
                 the base state go, the steps, what leaves the box and what
                 is removed at the end. It holds the signed URLs.
``steps.json``   for ``box/steps.py``: where the trainer and the probe keep
                 their files, and what the trainer must report.
``private/probe-args.json``
                 which tasks the probe runs, the seed of the plan, and the
                 key that names the tasks in the result. Host-private: it is
                 in no object store and in no journal, and the job removes it.

The steps of a lease:

train     the train command of the live lane, unchanged, in the environment
          of the box class
collect   check what the trainer wrote and name the files of the new state;
          the upload of the new state starts here and runs beside the probe
plan      write the probe plan
probe     serve the new checkpoint with the probe layouts, probe it, stop
          the servers
headplan, head-<n>
          only on a lease that runs HEAD blocks (``baseline.py``): serve
          the base checkpoint and probe it, once for each block
outcome   write the result that leaves the box

The train command and the probe command are text from the helpers of the
live lane and of the probe tool. This module does not write them again; it
refuses a train command that names another fp32 master than the lease
expects, or that sets the bootstrap override.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .files import Index
from .interfaces import JobSpec
from .layout import is_hex64

# The formats of the box side (``box/job.py``, ``box/steps.py``). A test compares them.
JOB_SCHEMA = 'affine-batchctl-job-v1'
STEPS_SCHEMA = 'affine-batchctl-steps-v1'
PROBE_ARGS_SCHEMA = 'affine-batchctl-probe-args-v1'
EXPECTED_MASTER_ENV = 'AFFINE_FP32_EXPECTED_MASTER_HASH'
MIN_MASK_KEY = 32
# ``box/job.py`` puts the id of the new checkpoint in the place of this text when the probe step starts.
CANDIDATE_TOKEN = '@@CANDIDATE_ID@@'
COLLECT_TIMEOUT_S = 3600
PLAN_TIMEOUT_S = 300
OUTCOME_TIMEOUT_S = 600
# A HEAD block ends itself with ``timeout``. Its step waits this much longer, so the step never fails the lease.
HEAD_STEP_MARGIN_S = 300
# A HEAD block does not start with less time than this left in the lease.
HEAD_MIN_S = 1800

SETS_EXPECTED_MASTER = re.compile(re.escape(EXPECTED_MASTER_ENV) + r'\s*=\s*["\']?([^\s"\';]*)')
SETS_OVERRIDE = re.compile(r'BOOTSTRAP_OVERRIDE\s*=')
SAFE_PATH = re.compile(r'(/[A-Za-z0-9][A-Za-z0-9._-]*)+')


class RecipeError(ValueError):
    """The lease cannot be written as a job. Nothing was sent to a box."""


def need(cond: bool, why: str) -> None:
    if not cond:
        raise RecipeError(why)


@dataclass(frozen=True)
class BoxPaths:
    """Where the trainer and the probe keep their files on a box. The defaults
    are the paths of the live lane; a test compares them with ``pe_box``."""

    pst: str = '/workspace/pst'
    fp32: str = '/workspace/affine-mimo-trainer-prod/fp32-master'
    ckpt: str = '/opt/plain/probe/affine-env-probe/ckpt'
    probe_root: str = '/root/affine-env-probe'
    work_root: str = '/workspace/batchctl'

    def __post_init__(self) -> None:
        for name in ('pst', 'fp32', 'ckpt', 'probe_root', 'work_root'):
            value = getattr(self, name)
            need(isinstance(value, str) and SAFE_PATH.fullmatch(value) is not None and value.count('/') >= 2,
                 'box path %s is not a plain absolute path of two parts or more' % name)

    @property
    def train_in(self) -> str:
        return self.pst + '/in'

    @property
    def train_out(self) -> str:
        return self.pst + '/out'

    @property
    def train_log(self) -> str:
        return self.pst + '/train.log'

    @property
    def probe_runs(self) -> str:
        return self.probe_root + '/runs'


@dataclass(frozen=True)
class BoxClass:
    """What a box must be, and how it trains."""

    name: str
    gpus: int
    gpu_name: str
    # Host RAM as the box itself reports it (MemTotal, GiB).
    ram_gb: int
    disk_free_gb: int
    # The trainer package that must have run, as the trainer reports it.
    trainer_host: str
    # True: the update must have run sharded over the GPUs.
    sharded: bool
    # The named set of sharded settings that must have run. None: the trainer has one set only.
    zero3_profile: Optional[str]
    train_env: Mapping[str, str] = field(default_factory=dict)
    train_timeout_s: int = 7200

    def __post_init__(self) -> None:
        need(self.gpus >= 1 and self.ram_gb >= 0 and self.disk_free_gb >= 0 and self.train_timeout_s > 0,
             'box class %s has a number that is not valid' % self.name)
        for key, value in self.train_env.items():
            need(isinstance(key, str) and isinstance(value, str) and 'BOOTSTRAP' not in key.upper()
                 and key != EXPECTED_MASTER_ENV, 'box class %s sets %s' % (self.name, key))


def rtx5090x8(zero3_profile: Optional[str] = 'rtx5090-u6') -> BoxClass:
    """Eight RTX 5090: the update runs sharded (host-v834f), with the settings
    measured on this class. The rental asks for 300 GB of host RAM; such a
    host reports about 279 GiB, less what its kernel keeps."""
    env = {'PIT_ZERO3': '1', 'PIT_ALLOW_SHARDED_GPUS': '1'}
    if zero3_profile:
        env['PIT_ZERO3_PROFILE'] = zero3_profile
    return BoxClass(
        name='rtx5090x8', gpus=8, gpu_name='RTX 5090', ram_gb=270, disk_free_gb=160, trainer_host='host-v834f',
        sharded=True, zero3_profile=zero3_profile or None, train_env=env,
    )


@dataclass(frozen=True)
class Transfer:
    """How the box moves bytes through the signed URLs."""

    content_type: str
    chunk_bytes: int
    workers: int = 8
    http_tries: int = 5
    http_pause_s: float = 2.0
    http_timeout_s: float = 120.0


@dataclass(frozen=True)
class LeaseFiles:
    """The files of one lease on its box, all under ``work``."""

    work: str

    def at(self, *parts: str) -> str:
        return '/'.join((self.work,) + parts)

    job = property(lambda self: self.at('job.json'))
    job_code = property(lambda self: self.at('job.py'))
    job_log = property(lambda self: self.at('job.log'))
    steps_config = property(lambda self: self.at('steps.json'))
    steps_code = property(lambda self: self.at('steps.py'))
    plan_builder = property(lambda self: self.at('pe_probe_plan.py'))
    private = property(lambda self: self.at('private'))
    probe_args = property(lambda self: self.at('private', 'probe-args.json'))
    plan = property(lambda self: self.at('private', 'eval-plan.json'))
    probe_dir = property(lambda self: self.at('probe'))
    probe_tool = property(lambda self: self.at('probe', 'probe_serve.py'))
    probe_result = property(lambda self: self.at('probe', 'RESULT.json'))
    base_aux = property(lambda self: self.at('base'))
    logs = property(lambda self: self.at('logs'))
    state = property(lambda self: self.at('state.json'))
    candidate = property(lambda self: self.at('candidate.json'))
    outcome = property(lambda self: self.at('outcome.json'))
    clock = property(lambda self: self.at('t0'))

    def head_dir(self, block: int) -> str:
        return self.at('head-%d' % block)

    def head_plan(self, block: int) -> str:
        return self.at('private', 'head-plan-%d.json' % block)

    def head_result(self, block: int) -> str:
        return self.at('head-%d' % block, 'RESULT.json')


@dataclass(frozen=True)
class HeadBlock:
    """One HEAD block that a lease runs after the probe of its batch: one
    probe run of the base checkpoint on the tasks of the round (``baseline.py``)."""

    block: int
    probe_cmd: str
    timeout_s: int

    def __post_init__(self) -> None:
        need(isinstance(self.block, int) and not isinstance(self.block, bool) and 0 <= self.block < 10000,
             'a HEAD block has a number from zero up')
        need(isinstance(self.probe_cmd, str) and self.probe_cmd.strip() != '' and CANDIDATE_TOKEN not in self.probe_cmd,
             'a HEAD block has a probe command of its own')
        need(isinstance(self.timeout_s, int) and self.timeout_s > 0, 'a HEAD block has no time')


def lease_files(paths: BoxPaths, spec: JobSpec) -> LeaseFiles:
    return LeaseFiles('%s/%s-f%d' % (paths.work_root, spec.batch_id, spec.fence))


def check_train_cmd(train_cmd: str, expected: str) -> None:
    """The train command gives the trainer the fp32 master digest of this
    lease and no other, and never sets the bootstrap override."""
    need(isinstance(train_cmd, str) and train_cmd.strip() != '', 'no train command')
    named = SETS_EXPECTED_MASTER.findall(train_cmd)
    need(bool(named) and all(value == expected for value in named),
         'the train command does not name the fp32 master digest that the lease expects')
    need(SETS_OVERRIDE.search(train_cmd) is None, 'the train command sets the bootstrap override')


def probe_args(*, indices: Sequence[Any], plan_seed: str, mask_key: str, env_id: str, k: int, max_tokens: int,
               max_num_seqs: int, harness: Optional[Mapping[str, Any]] = None,
               head: Sequence[Mapping[str, Any]] = ()) -> Dict[str, Any]:
    """The host-private arguments of the probe. Never stored off the box.

    ``head`` names the HEAD blocks of the lease: ``block``, ``plan_seed`` and
    ``k`` (rollouts for each task) of each."""
    for block in head:
        need(isinstance(block.get('block'), int) and not isinstance(block['block'], bool) and block['block'] >= 0
             and isinstance(block.get('plan_seed'), str) and block['plan_seed'] != ''
             and isinstance(block.get('k'), int) and not isinstance(block['k'], bool) and block['k'] > 0,
             'a HEAD block of the probe is not valid')
    need(len({block['block'] for block in head}) == len(head), 'a HEAD block is named two times')
    need(len(indices) > 0, 'the probe has no tasks')
    need(isinstance(plan_seed, str) and plan_seed != '', 'the probe has no seed')
    need(isinstance(mask_key, str) and len(mask_key) >= MIN_MASK_KEY, 'the probe has no mask key')
    need(isinstance(env_id, str) and env_id != '', 'the probe has no environment')
    need(all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in (k, max_tokens, max_num_seqs)),
         'the probe has a number that is not valid')
    return {
        'schema': PROBE_ARGS_SCHEMA, 'indices': list(indices), 'plan_seed': plan_seed, 'mask_key': mask_key,
        'env_id': env_id, 'k': k, 'max_tokens': max_tokens, 'max_num_seqs': max_num_seqs,
        'harness': dict(harness) if harness is not None else None,
        'head': [{'block': block['block'], 'plan_seed': block['plan_seed'], 'k': block['k']} for block in head],
    }


def steps_config(spec: JobSpec, *, paths: BoxPaths, klass: BoxClass, base: Index,
                 head: Sequence[HeadBlock] = ()) -> Dict[str, Any]:
    files = lease_files(paths, spec)
    need(is_hex64(base.checkpoint_id), 'the base state names no checkpoint')
    head_section = None
    if head:
        head_section = {
            'weights': '%s/actor/%s' % (paths.train_in, base.checkpoint_id),
            'descriptor': files.base_aux + '/descriptor.json',
            'blocks': {str(one.block): {'plan': files.head_plan(one.block), 'result': files.head_result(one.block)}
                       for one in head},
        }
    return {
        'head': head_section,
        'schema': STEPS_SCHEMA, 'expected_master': spec.expected_fp32_master_digest,
        'base_checkpoint': base.checkpoint_id, 'trainer_host': klass.trainer_host, 'sharded': klass.sharded,
        'zero3_profile': klass.zero3_profile,
        'train_log': paths.train_log, 'out': paths.train_out, 'fp32': paths.fp32, 'ckpt': paths.ckpt,
        'state': files.state, 'candidate': files.candidate, 'probe_args': files.probe_args,
        'plan_builder': files.plan_builder, 'plan': files.plan, 'probe_result': files.probe_result,
        'outcome': files.outcome,
    }


def _head_steps(*, files: LeaseFiles, step: str, stop: str, head: Sequence[HeadBlock],
                tool_files: Sequence[str], budget_s: int) -> List[Dict[str, Any]]:
    """The steps of the HEAD blocks. None of them can fail the lease: a block
    that does not finish is named in the result, and another box runs it.

    A block starts only while the lease has time for it. ``budget_s`` is the
    time from the start of the job in which every block must end. A block
    that has less than ``HEAD_MIN_S`` left does not start, and a block that
    starts is ended at the budget.
    """
    quote = shlex.quote
    if not head:
        return []
    need(len({one.block for one in head}) == len(head), 'a HEAD block is named two times')
    need(len(tool_files) > 0 and all(SAFE_PATH.fullmatch(path) is not None for path in tool_files),
         'the HEAD blocks have no probe tool')
    need(isinstance(budget_s, int) and not isinstance(budget_s, bool) and budget_s > 0, 'the HEAD blocks have no time')
    out = [{'name': 'headplan', 'script': (step % 'headplan').rstrip('\n') + ' || true\n', 'timeout_s': PLAN_TIMEOUT_S}]
    for one in head:
        folder = files.head_dir(one.block)
        script = (
            'start=$(cat %s 2>/dev/null || echo 0)\n'
            'left=$(( start + %d - $(date +%%s) ))\n'
            'if [ "$left" -lt %d ]; then echo "HEAD block %d: the lease has no time left for it"; exit 0; fi\n'
            'if [ "$left" -gt %d ]; then left=%d; fi\n'
        ) % (quote(files.clock), budget_s, HEAD_MIN_S, one.block, one.timeout_s, one.timeout_s)
        script += 'mkdir -p %s\ncp -f %s %s/\n' % (quote(folder), ' '.join(quote(path) for path in tool_files),
                                                quote(folder))
        script += 'timeout -k 60 "$left" bash -c %s\n' % quote(one.probe_cmd)
        out.append({'name': 'head-%d' % one.block, 'script': script + stop + 'exit 0\n',
                    'timeout_s': one.timeout_s + HEAD_STEP_MARGIN_S})
    return out


def _steps(*, paths: BoxPaths, klass: BoxClass, files: LeaseFiles, train_cmd: str, probe_cmd: str,
           probe_timeout_s: int, head: Sequence[HeadBlock] = (), tool_files: Sequence[str] = (),
           head_budget_s: int = 0) -> List[Dict[str, Any]]:
    quote = shlex.quote
    step = 'python3 %s %%s --config %s\n' % (quote(files.steps_code), quote(files.steps_config))
    stop = 'python3 %s --root %s stop >/dev/null 2>&1\n' % (quote(files.probe_tool), quote(paths.probe_root))
    clock = [{'name': 'clock', 'script': 'date +%%s > %s\n' % quote(files.clock), 'timeout_s': 60}] if head else []
    return [
        *clock,
        {'name': 'train', 'script': train_cmd, 'timeout_s': klass.train_timeout_s, 'env': dict(klass.train_env)},
        {'name': 'collect', 'script': step % 'collect', 'timeout_s': COLLECT_TIMEOUT_S, 'state': files.state},
        {'name': 'plan', 'script': step % 'plan', 'timeout_s': PLAN_TIMEOUT_S},
        {
            'name': 'probe', 'script': probe_cmd.rstrip('\n') + '\ncode=$?\n' + stop + 'exit $code\n',
            'timeout_s': probe_timeout_s,
            'subst': {CANDIDATE_TOKEN: {'file': files.candidate, 'key': ['candidate_id'], 'match': '[0-9a-f]{64}'}},
        },
        *_head_steps(files=files, step=step, stop=stop, head=head, tool_files=tool_files, budget_s=head_budget_s),
        {'name': 'outcome', 'script': step % 'outcome', 'timeout_s': OUTCOME_TIMEOUT_S},
    ]


def job_document(spec: JobSpec, *, paths: BoxPaths, klass: BoxClass, base: Index, binding: Mapping[str, Any],
                 train_cmd: str, probe_cmd: str, probe_timeout_s: int, urls: Mapping[str, Any],
                 transfer: Transfer, head: Sequence[HeadBlock] = (), tool_files: Sequence[str] = (),
                 head_budget_s: int = 0) -> Dict[str, Any]:
    """``job.json`` of one lease.

    ``base`` is the file index of the base state. ``binding`` says which
    checkpoint the batch was made for (``checkpoint_id``); the box trains only
    if that is the checkpoint of the base state. ``urls`` holds the signed
    URLs: ``base`` (``manifest``, ``objects``) and ``put`` (``objects``,
    ``heartbeat``, ``result``, ``complete``). ``head`` are the HEAD blocks
    that the lease runs after its own probe, ``tool_files`` the files of the
    probe tool on the box, which each block gets a copy of, and
    ``head_budget_s`` the time from the start of the job in which the blocks
    must end.
    """
    expected = spec.expected_fp32_master_digest
    need(base.fp32_master_digest == expected, 'the base state has another fp32 master than the lease expects')
    need(is_hex64(base.checkpoint_id) and is_hex64(base.fp32_entry),
         'the base state names no checkpoint or no fp32 master entry')
    need(isinstance(binding.get('checkpoint_id'), str) and binding['checkpoint_id'] != '',
         'the batch names no checkpoint that it was made for')
    check_train_cmd(train_cmd, expected)
    need(isinstance(probe_cmd, str) and probe_cmd.count(CANDIDATE_TOKEN) == 1,
         'the probe command does not name the new checkpoint once')
    need(probe_timeout_s > 0, 'the probe has no time')
    files = lease_files(paths, spec)
    master_dir = '%s/%s' % (paths.fp32, base.fp32_entry)
    return {
        'schema': JOB_SCHEMA, 'run_id': spec.run_id, 'batch_id': spec.batch_id, 'fence': spec.fence,
        'attempt': spec.attempt, 'base_seq': spec.base_seq, 'base_state': spec.base_state,
        'expected_fp32_master_digest': expected,
        'heartbeat_every_s': spec.heartbeat_every_s, 'job_ttl_s': spec.job_ttl_s,
        'content_type': transfer.content_type, 'chunk_bytes': transfer.chunk_bytes, 'workers': transfer.workers,
        'http': {'tries': transfer.http_tries, 'pause_s': transfer.http_pause_s,
                 'timeout_s': transfer.http_timeout_s},
        'base': {'manifest': urls['base']['manifest'], 'objects': dict(urls['base']['objects'])},
        'put': {'objects': dict(urls['put']['objects']), 'heartbeat': urls['put']['heartbeat'],
                'result': urls['put']['result'], 'complete': urls['put']['complete']},
        'require': {
            'gpus': klass.gpus, 'gpu_name': klass.gpu_name, 'ram_gb': klass.ram_gb,
            'disk_free_gb': klass.disk_free_gb, 'disk_path': files.work,
        },
        'binding': dict(binding),
        'place': [
            {'prefix': 'bf16/', 'dir': '%s/actor/%s' % (paths.train_in, base.checkpoint_id)},
            {'prefix': 'fp32/', 'dir': master_dir},
            {'prefix': 'aux/', 'dir': files.base_aux},
        ],
        'expect': [
            {'file': master_dir + '/meta.json', 'key': ['master_digest'], 'equals': expected},
            {'file': master_dir + '/meta.json', 'key': ['bf16_sha256'], 'equals': base.fp32_entry},
            {'file': paths.train_in + '/head.descriptor.json', 'key': ['id'], 'equals': base.checkpoint_id},
        ],
        'steps': _steps(paths=paths, klass=klass, files=files, train_cmd=train_cmd, probe_cmd=probe_cmd,
                        probe_timeout_s=probe_timeout_s, head=head, tool_files=tool_files,
                        head_budget_s=head_budget_s),
        'outcome': files.outcome,
        'wipe': [
            paths.train_in, paths.train_out, paths.fp32, paths.ckpt, paths.probe_runs,
            files.private, files.probe_dir, files.base_aux, files.logs,
            *[files.head_dir(one.block) for one in head],
        ],
    }


def lease_seconds(klass: BoxClass, probe_timeout_s: int, *, transfer_s: int = 3600) -> int:
    """The longest time that a lease of this class can need for its batch: the
    download of the base, every step at its limit, and the rest of the upload.
    HEAD blocks are not in it: they run only in the time that is left."""
    steps = klass.train_timeout_s + COLLECT_TIMEOUT_S + PLAN_TIMEOUT_S + probe_timeout_s + OUTCOME_TIMEOUT_S
    return transfer_s + steps + transfer_s
