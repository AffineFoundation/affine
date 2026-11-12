#!/usr/bin/env python3
"""File work of one lease between the trainer and the probe. One file,
standard library only. ``job.py`` runs each command as one step.

    python3 steps.py collect --config steps.json
    python3 steps.py plan    --config steps.json
    python3 steps.py outcome --config steps.json

collect   After the train command. Training is done only if its log ends
          with ``rc=0`` and holds the ``done`` event. The fp32 master that
          the trainer read must be the master that the lease expects, and
          it must not come from a bootstrap. The new weights are checked
          against the descriptor that the trainer wrote, and the new fp32
          master against the ``meta.json`` that the trainer wrote last.
          Then the weights are put where the probe server reads them, and
          the state file for ``job.py`` is written: every file of the new
          state with the sha256 that it must have when it is uploaded.
plan      Writes the probe plan for the new checkpoint. The plan is built
          by the plan builder of the live lane, from arguments that the
          host put into a private directory of the lease.
outcome   After the probe. Writes what leaves the box: the counters of
          every probed task and a summary of the training. A task is named
          by a keyed hash, not by its id: which tasks a round holds stays
          on the validator host. The key is in the private directory.

A command that fails exits with 1. ``job.py`` then writes no decision.
"""
import argparse
import hashlib
import hmac
import importlib.util
import json
import math
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

STEPS_SCHEMA = 'affine-batchctl-steps-v1'
CANDIDATE_SCHEMA = 'affine-batchctl-candidate-v1'
PROBE_ARGS_SCHEMA = 'affine-batchctl-probe-args-v1'
# The formats of ``job.py``. A test compares them.
BOX_STATE_SCHEMA = 'affine-batchctl-box-state-v1'
BOX_OUTCOME_SCHEMA = 'affine-batchctl-box-outcome-v1'
OUTCOME_PROBED = 'probed'
ROLE_BF16, ROLE_FP32, ROLE_AUX = 'bf16_weights', 'fp32_master', 'aux'
EXPECTED_MASTER_ENV = 'AFFINE_FP32_EXPECTED_MASTER_HASH'

HEX64 = re.compile(r'[0-9a-f]{64}')
SAFE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,199}')
FIRST_SHARD = re.compile(r'model-00001-of-[0-9]+\.safetensors')
EXIT_LINE = re.compile(r'rc=([0-9]+)')
# The trainer reads its fp32 master from the disk or keeps it in memory. Every other source is a bootstrap.
FP32_SOURCES = ('disk', 'memory')
ADMISSION_EVENT = 'root-admission-checked'
PROBE_DONE = ('ok', 'finalized')
MASK_HEX = 32
MIN_MASK_KEY = 32
MAX_SMALL_BYTES = 64 << 20
MAX_SUMMARY_BYTES = 256 << 10
TRAIN_EVENTS = (
    'zero3-policy', 'zero3-installed', 'vram-preflight-ok', 'signed-dispatch-prepared', 'native-grpo-complete',
    'zero3-microbatch', 'done',
)
REPORT_KEYS = (
    'status', 'optimizer_step', 'input_checkpoint', 'trained_rollouts', 'trained_tokens', 'preparation_seconds',
    'optimizer_seconds', 'preclip_gradient_norm', 'objective_loss', 'microbatch',
)
CONFIG_PATHS = (
    'train_log', 'out', 'fp32', 'ckpt', 'state', 'candidate', 'probe_args', 'plan_builder', 'plan',
    'probe_result', 'outcome',
)


class StepError(Exception):
    """The step cannot be done. No decision is written for this lease."""


def need(cond, why):
    if not cond:
        raise StepError(why)


def is_hex64(value):
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def is_name(value):
    return isinstance(value, str) and SAFE_NAME.fullmatch(value) is not None and '..' not in value


def is_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def read_json(path, what):
    try:
        if os.path.getsize(path) > MAX_SMALL_BYTES:
            raise ValueError('too large')
        with open(path, 'rb') as handle:
            return json.loads(handle.read())
    except (OSError, ValueError, RecursionError):
        raise StepError('cannot read %s' % what) from None


def write_json(path, doc, mode=0o644):
    """Write a file whole or not at all."""
    tmp = '%s.tmp.%d' % (path, os.getpid())
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode), 'w') as handle:
        handle.write(json.dumps(doc, sort_keys=True, allow_nan=False))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256_file(path):
    sha = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            sha.update(block)
    return sha.hexdigest()


def brief(value, depth=0):
    """A copy of ``value`` with numbers, flags and short words only. No path
    and no long text leaves the box this way."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value if len(value) <= 200 and '/' not in value else None
    if depth < 5 and isinstance(value, dict):
        return {key: brief(item, depth + 1) for key, item in value.items()
                if isinstance(key, str) and len(key) <= 80 and '/' not in key}
    if depth < 5 and isinstance(value, list):
        return [brief(item, depth + 1) for item in value]
    return None


def fits(doc):
    """``doc`` if it is small enough for a result, else nothing."""
    return doc if len(json.dumps(doc, allow_nan=False)) <= MAX_SUMMARY_BYTES else None


def load_config(path):
    cfg = read_json(path, 'the steps config')
    need(isinstance(cfg, dict) and cfg.get('schema') == STEPS_SCHEMA, 'not a steps config of this version')
    need(is_hex64(cfg.get('expected_master')), 'the steps config names no expected fp32 master digest')
    need(is_hex64(cfg.get('base_checkpoint')), 'the steps config names no base checkpoint')
    need(is_name(cfg.get('trainer_host')), 'the steps config names no trainer host')
    need(isinstance(cfg.get('sharded'), bool), 'the steps config does not say if the update runs sharded')
    need(cfg.get('zero3_profile') is None or is_name(cfg['zero3_profile']), 'the steps config names no valid settings')
    for key in CONFIG_PATHS:
        value = cfg.get(key)
        need(isinstance(value, str) and value.startswith('/') and os.path.normpath(value) == value,
             'the steps config has no absolute path for %s' % key)
    return cfg


# ----------------------------------------------------------------------------
# collect
# ----------------------------------------------------------------------------

def read_train_log(path):
    """-> (events by name, exit code of the trainer or None, end of the log)."""
    try:
        with open(path, 'rb') as handle:
            handle.seek(max(0, os.path.getsize(path) - MAX_SMALL_BYTES))
            text = handle.read().decode('utf-8', 'replace')
    except OSError:
        raise StepError('the train log is missing') from None
    events, code = {}, None
    for line in text.splitlines():
        line = line.strip()
        found = EXIT_LINE.fullmatch(line)
        if found is not None:
            code = int(found.group(1))
        if not line.startswith('{'):
            continue
        try:
            doc = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(doc, dict) and isinstance(doc.get('event'), str):
            events[doc['event']] = doc
    return events, code, text[-4000:]


def descriptor_files(desc):
    """The weight files of a checkpoint descriptor: ``{name: sha256}``."""
    need(isinstance(desc, dict) and is_hex64(desc.get('id')), 'the descriptor names no checkpoint id')
    files = desc.get('files')
    need(isinstance(files, dict) and files, 'the descriptor names no files')
    out = {}
    for name, value in files.items():
        sha = value.get('sha256') if isinstance(value, dict) else value
        need(is_name(name) and is_hex64(sha), 'the descriptor has a file entry that is not valid')
        out[name] = sha
    return out


def model_dir_of(out_dir):
    for root, dirs, names in os.walk(out_dir):
        dirs.sort()
        if any(FIRST_SHARD.fullmatch(name) for name in names):
            return root
    raise StepError('the trainer wrote no model files')


def fp32_block(report, done, expected):
    """The fp32 facts of the training, from the report and from the ``done``
    event. Both must say the same, and the input must be the expected master."""
    need(not report.get('used_fp32_bootstrap_override'), 'the trainer used a bootstrap override')
    optimizer = report.get('optimizer')
    sgd = optimizer.get('sgd') if isinstance(optimizer, dict) else None
    block = sgd.get('fp32_master') if isinstance(sgd, dict) else None
    need(isinstance(block, dict), 'the training report has no fp32 master block')
    need('finish_error' not in block, 'the trainer did not finish its fp32 master')
    said = done.get('fp32')
    need(isinstance(said, dict), 'the done event has no fp32 block')
    # Where the report says once more where the master came from, it must say the same.
    again = [sgd.get('fp32_master_source'), block.get('fp32_master_source')]
    again += [report[key].get('source') for key in ('fp32_master', 'fp32') if isinstance(report.get(key), dict)]
    for source in [block.get('source'), said.get('source')] + [source for source in again if source is not None]:
        need(isinstance(source, str) and source in FP32_SOURCES,
             'the fp32 master of the trainer came from %s' % str(source)[:80])
    need(block.get('input_master_digest') == expected and said.get('input_master_digest') == expected,
         'the trainer did not start from the fp32 master that the lease expects')
    digest = block.get('output_master_digest')
    need(is_hex64(digest) and said.get('output_master_digest') == digest,
         'the trainer names no fp32 master digest for the new state')
    # The trainer stores a master under the digest of the bf16 weights that belong to it.
    entry = block.get('output_bf16_sha256')
    need(is_hex64(entry) and block.get('output_store_entry') == entry and said.get('output_bf16') == entry,
         'the trainer names no store entry for the new fp32 master')
    return {'source': block['source'], 'input_master_digest': expected, 'output_master_digest': digest, 'entry': entry}


def check_settings(cfg, events):
    """The update ran the way the lease names: sharded over the GPUs or not,
    and with the named set of sharded settings."""
    policy = events.get('zero3-policy')
    policy = policy if isinstance(policy, dict) else {}
    need((policy.get('zero3') is True) == cfg['sharded'],
         'the update ran %s, the lease needs it %s' % (
             'sharded' if policy.get('zero3') is True else 'on one GPU', 'sharded' if cfg['sharded'] else 'on one GPU'))
    if cfg.get('zero3_profile') is not None:
        profile = policy.get('profile')
        name = profile.get('name') if isinstance(profile, dict) else None
        need(name == cfg['zero3_profile'],
             'the update ran the settings %s, the lease names %s' % (str(name)[:80], cfg['zero3_profile']))


def master_files(master_dir, entry, digest):
    """The files of the new fp32 master. ``meta.json`` is written last by the
    trainer, after every shard: without it the master is not complete."""
    meta_path = os.path.join(master_dir, 'meta.json')
    need(os.path.isfile(meta_path), 'the new fp32 master is not in the store (no meta.json)')
    meta = read_json(meta_path, 'meta.json of the new fp32 master')
    need(isinstance(meta, dict) and meta.get('master_digest') == digest,
         'meta.json of the new fp32 master names another master digest')
    need(meta.get('bf16_sha256') == entry, 'meta.json of the new fp32 master names another store entry')
    shards = meta.get('files')
    need(isinstance(shards, dict) and shards, 'meta.json of the new fp32 master names no files')
    out = []
    for name in sorted(shards):
        info = shards[name]
        need(is_name(name) and name != 'meta.json' and isinstance(info, dict) and is_hex64(info.get('sha256')),
             'meta.json of the new fp32 master has a file entry that is not valid')
        path = os.path.join(master_dir, name)
        need(os.path.isfile(path) and os.path.getsize(path) == info.get('bytes'),
             'file %s of the new fp32 master is missing or has another size' % name)
        out.append({'path': 'fp32/' + name, 'role': ROLE_FP32, 'source': path, 'sha256': info['sha256']})
    out.append({'path': 'fp32/meta.json', 'role': ROLE_FP32, 'source': meta_path, 'sha256': sha256_file(meta_path)})
    return out


def place(model_dir, ckpt_root, checkpoint, descriptor_path):
    """Put the weights where the probe server reads them, as the live lane does."""
    target = os.path.join(ckpt_root, checkpoint)
    os.makedirs(target, exist_ok=True)
    for name in sorted(os.listdir(model_dir)):
        source, dest = os.path.join(model_dir, name), os.path.join(target, name)
        if not os.path.isfile(source):
            continue
        if os.path.lexists(dest):
            os.remove(dest)
        try:
            os.link(source, dest)
        except OSError:
            shutil.copyfile(source, dest)
    shutil.copyfile(descriptor_path, os.path.join(ckpt_root, checkpoint + '.descriptor.json'))
    with open(os.path.join(target, '.verified'), 'w'):
        pass


def collect(cfg):
    expected = cfg['expected_master']
    need(os.environ.get(EXPECTED_MASTER_ENV) == expected, 'this step does not run under the lease of its config')
    events, code, end = read_train_log(cfg['train_log'])
    done = events.get('done')
    if code != 0 or not isinstance(done, dict):
        # The end of the train log goes to the log of this step: ``job.py`` looks there for the cause.
        sys.stderr.write(end + '\n')
        raise StepError('training did not finish (exit code %s, done event %s)' % (
            code, 'present' if isinstance(done, dict) else 'missing'))
    need(ADMISSION_EVENT in events, 'the trainer did not report its ROOT admission check')
    need(done.get('host') == cfg['trainer_host'],
         'the trainer host was %s, the lease needs %s' % (str(done.get('host'))[:80], cfg['trainer_host']))
    check_settings(cfg, events)

    descriptor_path = os.path.join(cfg['out'], 'descriptor.json')
    desc = read_json(descriptor_path, 'the descriptor of the new checkpoint')
    files = descriptor_files(desc)
    checkpoint = desc['id']
    need(done.get('checkpoint') == checkpoint, 'the done event and the descriptor name two checkpoints')
    report = read_json(os.path.join(cfg['out'], 'report.private.json'), 'the training report')
    need(isinstance(report, dict), 'the training report is not valid')
    need(report.get('input_checkpoint') == cfg['base_checkpoint'],
         'the trainer did not start from the checkpoint of the base state')
    fp32 = fp32_block(report, done, expected)

    model_dir = model_dir_of(cfg['out'])
    for name in files:
        need(os.path.isfile(os.path.join(model_dir, name)), 'weight file %s is missing' % name)
    with ThreadPoolExecutor(max_workers=8) as pool:
        measured = dict(zip(files, pool.map(lambda name: sha256_file(os.path.join(model_dir, name)), files)))
    bad = sorted(name for name in files if measured[name] != files[name])
    need(not bad, 'weight files are not the files of the descriptor: %s' % ', '.join(bad[:5]))

    state_files = [
        {'path': 'bf16/' + name, 'role': ROLE_BF16, 'source': os.path.join(model_dir, name), 'sha256': files[name]}
        for name in sorted(files)
    ]
    state_files += master_files(os.path.join(cfg['fp32'], fp32['entry']), fp32['entry'], fp32['output_master_digest'])
    state_files.append({
        'path': 'aux/descriptor.json', 'role': ROLE_AUX, 'source': descriptor_path,
        'sha256': sha256_file(descriptor_path),
    })
    place(model_dir, cfg['ckpt'], checkpoint, descriptor_path)

    write_json(cfg['candidate'], {
        'schema': CANDIDATE_SCHEMA, 'candidate_id': checkpoint,
        'train': {
            'checkpoint': checkpoint, 'input_checkpoint': cfg['base_checkpoint'], 'host': cfg['trainer_host'],
            'fp32': fp32,
            'events': fits({name: brief(events[name]) for name in TRAIN_EVENTS if name in events}),
            'report': fits(brief({key: report[key] for key in REPORT_KEYS if key in report})),
        },
    })
    # Last: ``job.py`` starts the upload from this file.
    write_json(cfg['state'], {
        'schema': BOX_STATE_SCHEMA,
        'trainer': {
            'input_master_digest': fp32['input_master_digest'], 'output_master_digest': fp32['output_master_digest'],
        },
        'meta': {
            'fp32_master_digest': fp32['output_master_digest'], 'checkpoint_id': checkpoint,
            'fp32_entry': fp32['entry'],
        },
        'files': state_files,
    })
    return 'collected %s: %d weight files, %d fp32 master files' % (
        checkpoint[:12], len(files), len(state_files) - len(files) - 1)


# ----------------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------------

def load_candidate(cfg):
    doc = read_json(cfg['candidate'], 'the candidate file')
    need(isinstance(doc, dict) and doc.get('schema') == CANDIDATE_SCHEMA and is_hex64(doc.get('candidate_id'))
         and isinstance(doc.get('train'), dict), 'the candidate file is not valid')
    return doc


def load_probe_args(cfg):
    args = read_json(cfg['probe_args'], 'the probe arguments')
    need(isinstance(args, dict) and args.get('schema') == PROBE_ARGS_SCHEMA, 'the probe arguments are not valid')
    need(isinstance(args.get('indices'), list) and args['indices'], 'the probe arguments name no tasks')
    need(isinstance(args.get('plan_seed'), str) and args['plan_seed'], 'the probe arguments name no seed')
    need(isinstance(args.get('env_id'), str) and args['env_id'], 'the probe arguments name no environment')
    for key in ('k', 'max_tokens', 'max_num_seqs'):
        need(is_count(args.get(key)), 'the probe arguments have no %s' % key)
    need(isinstance(args.get('mask_key'), str) and len(args['mask_key']) >= MIN_MASK_KEY,
         'the probe arguments name no mask key')
    need(args.get('harness') is None or isinstance(args['harness'], dict), 'the probe harness is not valid')
    return args


def load_builder(path):
    """The plan builder of the live lane: a file that the host shipped."""
    try:
        spec = importlib.util.spec_from_file_location('batchctl_plan_builder', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build_probe_plan
    except Exception as exc:
        raise StepError('cannot load the plan builder: %s' % type(exc).__name__) from None


def plan(cfg):
    candidate = load_candidate(cfg)['candidate_id']
    args = load_probe_args(cfg)
    built = load_builder(cfg['plan_builder'])(
        candidate, args['indices'], args['plan_seed'], k=args['k'], max_tokens=args['max_tokens'],
        env_id=args['env_id'], max_num_seqs=args['max_num_seqs'], harness=args.get('harness'),
    )
    need(isinstance(built, dict) and built.get('checkpoint_id') == candidate, 'the plan is for another checkpoint')
    write_json(cfg['plan'], built, 0o600)
    return 'plan for %s: %d tasks' % (candidate[:12], len(args['indices']))


# ----------------------------------------------------------------------------
# outcome
# ----------------------------------------------------------------------------

def mask(key, env_id, index):
    """The name of one task outside the box and the validator host."""
    return hmac.new(key.encode(), ('%s:%s' % (env_id, index)).encode(), hashlib.sha256).hexdigest()[:MASK_HEX]


def plain(doc):
    """The finite numbers and the flags of one level of a probe result."""
    return {name: value for name, value in doc.items() if isinstance(name, str) and (
        isinstance(value, (bool, int)) or (isinstance(value, float) and math.isfinite(value)))}


def public_probe(result, key):
    """What of a probe result leaves the box: numbers and flags, the layout
    that served, and the counters of every task under its masked name."""
    need(isinstance(result, dict) and isinstance(result.get('envs'), dict) and result['envs'],
         'the probe result has no environments')
    out = plain(result)
    out['envs'] = {}
    for env_id, env in result['envs'].items():
        need(isinstance(env, dict) and isinstance(env.get('per_task'), dict) and env['per_task'],
             'the probe result has no counters of tasks')
        cells = {}
        for index, cell in env['per_task'].items():
            need(isinstance(cell, dict), 'the probe result has a task without counters')
            cells[mask(key, env_id, index)] = plain(cell)
        need(len(cells) == len(env['per_task']), 'two tasks have one masked name')
        out['envs'][env_id] = dict(plain(env), per_task=cells)
    served = result.get('requests_per_replica')
    if isinstance(served, list):
        out['requests_per_replica'] = [count if is_count(count) else 0 for count in served]
    if isinstance(result.get('serve_layout'), dict):
        out['serve_layout'] = fits(brief(result['serve_layout']))
    return out


def outcome(cfg):
    candidate = load_candidate(cfg)
    args = load_probe_args(cfg)
    result = read_json(cfg['probe_result'], 'the result of the probe')
    need(isinstance(result, dict), 'the result of the probe is not valid')
    need(result.get('status') in PROBE_DONE, 'the probe did not finish: %s' % str(result.get('error'))[:300])
    need(result.get('candidate_id') == candidate['candidate_id'], 'the probe ran on another checkpoint')
    write_json(cfg['outcome'], {
        'schema': BOX_OUTCOME_SCHEMA, 'outcome': OUTCOME_PROBED,
        'payload': {
            'candidate_id': candidate['candidate_id'],
            'probe': {'status': result['status'], 'result': public_probe(result.get('probe_result'), args['mask_key'])},
            'train': candidate['train'],
        },
    })
    return 'probe %s for %s' % (result['status'], candidate['candidate_id'][:12])


COMMANDS = {'collect': collect, 'plan': plan, 'outcome': outcome}


def main(argv):
    parser = argparse.ArgumentParser(description='file work of one lease between the trainer and the probe')
    parser.add_argument('command', choices=sorted(COMMANDS))
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv[1:])
    try:
        print(COMMANDS[args.command](load_config(args.config)))
    except StepError as exc:
        sys.stderr.write('%s: %s\n' % (args.command, exc))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
