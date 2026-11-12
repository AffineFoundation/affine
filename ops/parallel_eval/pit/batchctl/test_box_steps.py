"""The file work between the trainer and the probe on a box (``box/steps.py``)
and the names of probed tasks (``masking.py``).

The tests build what the trainer of the live lane leaves on a box: its log,
its descriptor, its report, the weights and the fp32 master in the store.
The plan step runs the real plan builder of the live lane.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from . import layout as L
from . import masking as M
from .box import job as J
from .box import steps as S

STEPS = Path(__file__).parent / 'box' / 'steps.py'
PLAN_BUILDER = Path(__file__).parents[2] / 'live' / 'pe_probe_plan.py'
EXPECTED = '12' * 32
BASE_CHECKPOINT = '34' * 32
BASE_BF16 = '56' * 32
ENTRY = 'ab' * 32
NEW_MASTER = 'cd' * 32
OTHER = '99' * 32
INDICES = [70123, 815, 4242, 9]
SEED = 'plan-seed-of-the-round'
ENV_ID = 'affine_gatepool'
KEY = M.mask_key(SEED, 'run0001', 'e7-0123456789ab')


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Trained:
    """A box after the train command: every file that the trainer of the live
    lane writes, as dicts that a test changes before ``write``."""

    def __init__(self, root):
        self.root = root
        self.pst, self.fp32, self.ckpt, self.work = root / 'pst', root / 'fp32-master', root / 'ckpt', root / 'lease'
        self.out, self.model = self.pst / 'out', self.pst / 'out' / 'model'
        self.store = self.fp32 / ENTRY
        self.weights = {
            'model-00001-of-00002.safetensors': b'one' * 5000, 'model-00002-of-00002.safetensors': b'two' * 4000,
            'config.json': b'{"hidden": 4096}',
        }
        self.descriptor = {'files': {name: sha(data) for name, data in self.weights.items()}}
        self.checkpoint = self.descriptor['id'] = sha(json.dumps(self.descriptor['files'], sort_keys=True).encode())
        self.shards = {
            'fp32-master-00001-of-00002.safetensors': b'm1' * 9000, 'fp32-master-00002-of-00002.safetensors': b'm2' * 100,
        }
        self.meta = {
            'version': 'v833f-fp32-master-v1', 'bf16_sha256': ENTRY, 'master_digest': NEW_MASTER,
            'input_bf16_sha256': BASE_BF16, 'input_master_digest': EXPECTED, 'source': 'disk', 'optimizer_step': 6,
            'files': {name: {'sha256': sha(data), 'bytes': len(data)} for name, data in self.shards.items()},
        }
        block = {
            'version': 'v833f-fp32-master-v1', 'source': 'disk', 'input_bf16_sha256': BASE_BF16,
            'input_master_digest': EXPECTED, 'input_store_entry': BASE_BF16, 'output_bf16_sha256': ENTRY,
            'output_master_digest': NEW_MASTER, 'output_store_entry': ENTRY, 'store': str(self.fp32),
            'fp32_master_source': 'disk', 'fp32_master_hash_in': EXPECTED, 'fp32_master_hash_out': NEW_MASTER,
        }
        self.report = {
            'status': 'updated', 'optimizer_step': 6, 'input_checkpoint': BASE_CHECKPOINT, 'trained_rollouts': 64,
            'trained_tokens': 123456, 'preparation_seconds': 1.5, 'optimizer_seconds': 100.25,
            'preclip_gradient_norm': 0.75, 'objective_loss': 1.25,
            'microbatch': {'token_budget': 32768, 'steps': 6, 'peak_gib': [27.1, 26.9], 'profile': 'rtx5090-u6'},
            'optimizer': {'sgd': {
                'fp32_master': block, 'fp32_master_source': 'disk', 'fp32_master_hash_in': EXPECTED,
                'fp32_master_hash_out': NEW_MASTER, 'sgd_lr': 1e-6,
            }},
            'rollout_text': 'a rollout of a miner ' * 50,
        }
        self.events = [
            {'event': 'root-admission-checked', 'seconds': 0.1, 'engine_fp32': True, 'host': str(root / 'host-v834f')},
            {'event': 'zero3-policy', 'seconds': 0.2, 'per_gpu_mem_gib': 31.5, 'zero3': True,
             'host_version': 'host-v834f', 'token_budget_cap': 32768, 'backend': 'nccl',
             'profile': {'name': 'rtx5090-u6', 'token_budget': 32768, 'layer_reduce': 'bfloat16'}},
            {'event': 'zero3-installed', 'seconds': 9.0, 'dp_world': 8, 'dp_token_budget': 32768},
            {'event': 'native-grpo-complete', 'seconds': 500.0, 'status': 'updated', 'optimizer_step': 6,
             'trained_rollouts': 64, 'loss': 1.25},
            {'event': 'zero3-microbatch', 'seconds': 500.1, 'token_budget': 32768, 'steps': 6,
             'peak_gib': [27.1, 26.9]},
            {'event': 'staged', 'seconds': 520.0, 'output_checkpoint': self.checkpoint},
            {'event': 'done', 'seconds': 600.0, 'checkpoint': self.checkpoint, 'torch_max_alloc_gib': 27.5,
             'host': 'host-v834f', 'fp32': {
                 'source': 'disk', 'input_master_digest': EXPECTED, 'output_master_digest': NEW_MASTER,
                 'input_bf16': BASE_BF16, 'output_bf16': ENTRY}},
        ]
        self.rc = 0
        self.log_text = 'helper-preload rank 3\nTraceback is not here\n'
        self.config = {
            'schema': S.STEPS_SCHEMA, 'expected_master': EXPECTED, 'base_checkpoint': BASE_CHECKPOINT,
            'trainer_host': 'host-v834f', 'sharded': True, 'zero3_profile': 'rtx5090-u6',
            'train_log': str(self.pst / 'train.log'), 'out': str(self.out), 'fp32': str(self.fp32),
            'ckpt': str(self.ckpt), 'state': str(self.work / 'state.json'),
            'candidate': str(self.work / 'candidate.json'),
            'probe_args': str(self.work / 'private' / 'probe-args.json'), 'plan_builder': str(PLAN_BUILDER),
            'plan': str(self.work / 'private' / 'eval-plan.json'),
            'probe_result': str(self.work / 'probe' / 'RESULT.json'), 'outcome': str(self.work / 'outcome.json'),
        }
        self.probe_args = {
            'schema': S.PROBE_ARGS_SCHEMA, 'indices': list(INDICES), 'plan_seed': SEED, 'mask_key': KEY,
            'env_id': ENV_ID, 'k': 8, 'max_tokens': 65536, 'max_num_seqs': 112,
            'harness': {'harness': {'name': 'gate'}, 'mimo_answer_projection': 'boxed', 'sampling_contract': {'t': 1}},
        }
        for path in (self.model, self.store, self.work / 'private', self.work / 'probe'):
            path.mkdir(parents=True)

    def event(self, name):
        return next(event for event in self.events if event['event'] == name)

    def drop(self, name):
        self.events = [event for event in self.events if event['event'] != name]

    @property
    def block(self):
        return self.report['optimizer']['sgd']['fp32_master']

    def write(self):
        for name, data in self.weights.items():
            (self.model / name).write_bytes(data)
        for name, data in self.shards.items():
            (self.store / name).write_bytes(data)
        if self.meta is not None:
            (self.store / 'meta.json').write_text(json.dumps(self.meta, sort_keys=True, indent=1))
        (self.out / 'descriptor.json').write_text(json.dumps(self.descriptor))
        (self.out / 'report.private.json').write_text(json.dumps(self.report))
        lines = [self.log_text.rstrip('\n')] + [json.dumps(event) for event in self.events]
        if self.rc is not None:
            lines.append('rc=%d' % self.rc)
        (self.pst / 'train.log').write_text('\n'.join(lines) + '\n')
        (self.work / 'steps.json').write_text(json.dumps(self.config))
        (self.work / 'private' / 'probe-args.json').write_text(json.dumps(self.probe_args))
        return self

    def cfg(self):
        return S.load_config(str(self.work / 'steps.json'))

    def state(self):
        return json.loads((self.work / 'state.json').read_text())

    def candidate(self):
        return json.loads((self.work / 'candidate.json').read_text())

    def probed(self, **changes):
        """What the probe tool leaves in its job directory."""
        cells = {str(index): {'n': 8, 'c': 3 + place, 'w': 4 - place, 'u': 1, 'b': place, 'x': 0}
                 for place, index in enumerate(INDICES)}
        doc = {
            'ok': True, 'status': 'ok', 'candidate_id': self.checkpoint, 'at': 1.5, 'run_name': 'pe-abc-def-rtx5090-tp2',
            'probe_result': {
                'seed': SEED, 'run_name': 'pe-abc-def-rtx5090-tp2', 'checkpoint_id': self.checkpoint,
                'samples': 32, 'planned_samples': 32, 'partial': False, 'concurrency': 448, 'wall_s': 5000.5,
                'requests_per_replica': [9, 8, 8, 7],
                'envs': {ENV_ID: {'per_task': cells, 'samples': 32, 'indices': list(INDICES), 'mean': 0.5}},
                'serve_layout': {
                    'layout': {'name': 'rtx5090-tp2', 'gpus_per_replica': 2, 'max_num_seqs': 112},
                    'runtime': {'venv': '/opt/venvs/vllm-0.33.0', 'vllm_version': '0.33.0'},
                    'replicas': [{'replica': 0, 'gpus': [0, 1], 'pid': 4321}], 'serve_s': 300.5,
                },
            },
        }
        doc.update(changes)
        (self.work / 'probe' / 'RESULT.json').write_text(json.dumps(doc))
        return doc

    def outcome(self):
        return json.loads((self.work / 'outcome.json').read_text())


@pytest.fixture
def box(tmp_path, monkeypatch):
    monkeypatch.setenv(S.EXPECTED_MASTER_ENV, EXPECTED)
    return Trained(tmp_path)


# ----------------------------------------------------------------------------
# One format on both sides
# ----------------------------------------------------------------------------

def test_the_steps_the_job_and_the_host_share_their_formats():
    assert (S.BOX_STATE_SCHEMA, S.BOX_OUTCOME_SCHEMA, S.OUTCOME_PROBED) == (
        J.BOX_STATE_SCHEMA, J.BOX_OUTCOME_SCHEMA, J.OUTCOME_PROBED)
    assert S.EXPECTED_MASTER_ENV == J.ENV_EXPECTED_MASTER
    assert (S.ROLE_BF16, S.ROLE_FP32, S.ROLE_AUX) == (L.ROLE_BF16_WEIGHTS, L.ROLE_FP32_MASTER, L.ROLE_AUX)
    assert (S.HEX64.pattern, S.SAFE_NAME.pattern) == (L.HEX64.pattern, L.SAFE_NAME.pattern)
    assert S.MASK_HEX == M.MASK_HEX
    for env_id, index in ((ENV_ID, 7), (ENV_ID, '7'), ('other', 70123), ('', 0)):
        assert S.mask(KEY, env_id, index) == M.mask(KEY, env_id, index)
        assert len(S.mask(KEY, env_id, index)) == M.MASK_HEX
    assert S.mask(KEY, ENV_ID, 7) == S.mask(KEY, ENV_ID, '7') != S.mask(KEY, ENV_ID, 8)


# ----------------------------------------------------------------------------
# collect
# ----------------------------------------------------------------------------

def test_collect_names_every_file_of_the_new_state_for_the_job(box):
    box.write()
    said = S.collect(box.cfg())
    assert said == 'collected %s: 3 weight files, 3 fp32 master files' % box.checkpoint[:12]
    state = box.state()
    assert state['schema'] == J.BOX_STATE_SCHEMA
    assert state['trainer'] == {'input_master_digest': EXPECTED, 'output_master_digest': NEW_MASTER}
    assert state['meta'] == {'fp32_master_digest': NEW_MASTER, 'checkpoint_id': box.checkpoint, 'fp32_entry': ENTRY}
    files = {item['path']: item for item in state['files']}
    assert sorted(files) == sorted(
        ['bf16/' + name for name in box.weights] + ['fp32/' + name for name in box.shards]
        + ['fp32/meta.json', 'aux/descriptor.json'])
    for name, data in box.weights.items():
        assert files['bf16/' + name] == {
            'path': 'bf16/' + name, 'role': 'bf16_weights', 'source': str(box.model / name), 'sha256': sha(data)}
    for name, data in box.shards.items():
        assert files['fp32/' + name] == {
            'path': 'fp32/' + name, 'role': 'fp32_master', 'source': str(box.store / name), 'sha256': sha(data)}
    # The next box gets meta.json and the descriptor byte for byte.
    assert files['fp32/meta.json']['sha256'] == sha((box.store / 'meta.json').read_bytes())
    assert files['aux/descriptor.json'] == {
        'path': 'aux/descriptor.json', 'role': 'aux', 'source': str(box.out / 'descriptor.json'),
        'sha256': sha((box.out / 'descriptor.json').read_bytes())}
    # The job takes this state file as it is.
    meta, listed = J.load_box_state(str(box.work / 'state.json'), {'expected_fp32_master_digest': EXPECTED})
    assert meta == state['meta'] and len(listed) == 7


def test_collect_puts_the_weights_where_the_probe_server_reads_them(box):
    box.write()
    S.collect(box.cfg())
    served = box.ckpt / box.checkpoint
    assert sorted(path.name for path in served.iterdir()) == sorted(list(box.weights) + ['.verified'])
    for name in box.weights:
        assert os.path.samefile(served / name, box.model / name)
    assert (box.ckpt / (box.checkpoint + '.descriptor.json')).read_bytes() == (box.out / 'descriptor.json').read_bytes()
    # A second run after a restart of the step puts the same files there again.
    S.collect(box.cfg())
    assert sorted(path.name for path in served.iterdir()) == sorted(list(box.weights) + ['.verified'])


def test_collect_keeps_a_summary_of_the_training_without_paths_or_text(box):
    box.write()
    S.collect(box.cfg())
    doc = box.candidate()
    assert doc['schema'] == S.CANDIDATE_SCHEMA and doc['candidate_id'] == box.checkpoint
    train = doc['train']
    assert (train['checkpoint'], train['input_checkpoint'], train['host']) == (
        box.checkpoint, BASE_CHECKPOINT, 'host-v834f')
    assert train['fp32'] == {
        'source': 'disk', 'input_master_digest': EXPECTED, 'output_master_digest': NEW_MASTER, 'entry': ENTRY}
    assert train['events']['zero3-policy']['profile'] == {
        'name': 'rtx5090-u6', 'token_budget': 32768, 'layer_reduce': 'bfloat16'}
    assert train['events']['zero3-microbatch']['peak_gib'] == [27.1, 26.9]
    assert train['events']['done']['torch_max_alloc_gib'] == 27.5
    assert train['report'] == {key: box.report[key] for key in S.REPORT_KEYS}
    text = json.dumps(doc)
    assert str(box.root) not in text and 'a rollout of a miner' not in text and 'root-admission-checked' not in text


def test_what_is_kept_of_the_training_is_small_and_has_numbers_and_short_words_only(box):
    assert S.brief({'a': 1, 'b': 1.5, 'c': True, 'd': None, 'e': 'word', 'f': [1, 'x', {'g': 2}]}) == {
        'a': 1, 'b': 1.5, 'c': True, 'd': None, 'e': 'word', 'f': [1, 'x', {'g': 2}]}
    assert S.brief({'path': '/workspace/x', 'long': 'x' * 201, 'short': 'x' * 200, 'nan': float('nan'),
                    'inf': float('-inf'), 'a/b': 1, 'k' * 81: 1, 7: 1, 'set': (1, 2)}) == {
        'path': None, 'long': None, 'short': 'x' * 200, 'nan': None, 'inf': None, 'set': None}
    deep = {'a': {'a': {'a': {'a': {'a': {'a': 1}}}}}}
    assert S.brief(deep) == {'a': {'a': {'a': {'a': {'a': None}}}}}
    assert S.fits({'a': 'x' * 1000}) == {'a': 'x' * 1000} and S.fits({'a': ['x' * 100] * 3000}) is None
    # A report or an event that is too large is left out as a whole. The facts that the job checks stay.
    box.report['microbatch'] = {'steps': list(range(100000))}
    box.report['objective_loss'] = float('nan')
    box.event('zero3-microbatch')['note'] = 'see /workspace/pst/train.log'
    box.write()
    S.collect(box.cfg())
    train = box.candidate()['train']
    assert train['report'] is None and train['fp32']['output_master_digest'] == NEW_MASTER
    assert train['events']['zero3-microbatch']['note'] is None
    box.report['microbatch'] = {'steps': 6}
    box.write()
    S.collect(box.cfg())
    assert box.candidate()['train']['report']['objective_loss'] is None


def test_a_descriptor_may_name_its_files_with_their_size(box):
    box.descriptor['files'] = {name: {'sha256': sha(data), 'bytes': len(data)} for name, data in box.weights.items()}
    box.write()
    S.collect(box.cfg())
    assert {item['sha256'] for item in box.state()['files'] if item['role'] == 'bf16_weights'} == {
        sha(data) for data in box.weights.values()}


def test_a_trainer_with_one_set_of_settings_needs_no_name_for_it(box):
    box.config['zero3_profile'] = None
    del box.event('zero3-policy')['profile']
    box.write()
    S.collect(box.cfg())
    assert box.state()['meta']['checkpoint_id'] == box.checkpoint


def test_a_box_that_trains_on_one_gpu_reports_no_sharded_update(box):
    box.config.update(sharded=False, zero3_profile=None, trainer_host='host-v834e')
    box.drop('zero3-policy')
    box.event('done')['host'] = 'host-v834e'
    box.write()
    S.collect(box.cfg())
    assert box.candidate()['train']['host'] == 'host-v834e'


def other_master(target):
    target['input_master_digest'] = OTHER


REFUSALS = {
    'the trainer ended with an error': (lambda box: setattr(box, 'rc', 1), 'training did not finish'),
    'the trainer did not end': (lambda box: setattr(box, 'rc', None), 'training did not finish'),
    'no done event': (lambda box: box.drop('done'), 'training did not finish'),
    'no admission event': (lambda box: box.drop('root-admission-checked'), 'ROOT admission check'),
    'another trainer host': (lambda box: box.event('done').update(host='host-v834e'), 'trainer host was host-v834e'),
    'update not sharded': (lambda box: box.event('zero3-policy').update(zero3=False), 'the update ran on one GPU'),
    'no policy event': (lambda box: box.drop('zero3-policy'), 'the update ran on one GPU'),
    'other settings': (lambda box: box.event('zero3-policy')['profile'].update(name='reference-u2'),
                       'ran the settings reference-u2'),
    'settings without a name': (lambda box: box.event('zero3-policy').pop('profile'), 'ran the settings None'),
    'two checkpoints': (lambda box: box.event('done').update(checkpoint=OTHER), 'name two checkpoints'),
    'another input checkpoint': (lambda box: box.report.update(input_checkpoint=OTHER),
                                 'did not start from the checkpoint of the base state'),
    'no input checkpoint': (lambda box: box.report.pop('input_checkpoint'),
                            'did not start from the checkpoint of the base state'),
    'override used': (lambda box: box.report.update(used_fp32_bootstrap_override=True), 'used a bootstrap override'),
    'master from a bootstrap': (lambda box: box.block.update(source='bootstrap-upcast-root-override'),
                                'came from bootstrap-upcast-root-override'),
    'done says bootstrap': (lambda box: box.event('done')['fp32'].update(source='bootstrap-upcast'),
                            'came from bootstrap-upcast'),
    'optimizer says bootstrap': (
        lambda box: box.report['optimizer']['sgd'].update(fp32_master_source='bootstrap-upcast'),
        'came from bootstrap-upcast'),
    'block says bootstrap twice': (lambda box: box.block.update(fp32_master_source='bootstrap-upcast'),
                                   'came from bootstrap-upcast'),
    'report says bootstrap': (lambda box: box.report.update(fp32_master={'source': 'bootstrap-upcast'}),
                              'came from bootstrap-upcast'),
    'report says bootstrap in fp32': (lambda box: box.report.update(fp32={'source': 'bootstrap-upcast'}),
                                      'came from bootstrap-upcast'),
    'no source': (lambda box: box.block.pop('source'), 'came from None'),
    'done without a source': (lambda box: box.event('done')['fp32'].pop('source'), 'came from None'),
    'report read another master': (lambda box: other_master(box.block), 'did not start from the fp32 master'),
    'done read another master': (lambda box: other_master(box.event('done')['fp32']),
                                 'did not start from the fp32 master'),
    'two new masters': (lambda box: box.event('done')['fp32'].update(output_master_digest=OTHER),
                        'names no fp32 master digest for the new state'),
    'no new master': (lambda box: box.block.pop('output_master_digest'),
                      'names no fp32 master digest for the new state'),
    'new master is no digest': (lambda box: (box.block.update(output_master_digest='x'),
                                             box.event('done')['fp32'].update(output_master_digest='x')),
                                'names no fp32 master digest for the new state'),
    'another store entry': (lambda box: box.block.update(output_store_entry=OTHER), 'names no store entry'),
    'done names another entry': (lambda box: box.event('done')['fp32'].update(output_bf16=OTHER),
                                 'names no store entry'),
    'entry is no digest': (lambda box: (box.block.update(output_bf16_sha256='../x', output_store_entry='../x'),
                                        box.event('done')['fp32'].update(output_bf16='../x')),
                           'names no store entry'),
    'master not finished': (lambda box: box.block.update(finish_error='RuntimeError()'),
                            'did not finish its fp32 master'),
    'no master block': (lambda box: box.report['optimizer']['sgd'].pop('fp32_master'), 'no fp32 master block'),
    'no optimizer': (lambda box: box.report.pop('optimizer'), 'no fp32 master block'),
    'done without fp32': (lambda box: box.event('done').pop('fp32'), 'done event has no fp32 block'),
    'master not saved': (lambda box: setattr(box, 'meta', None), 'not in the store'),
    'meta names another master': (lambda box: box.meta.update(master_digest=OTHER), 'names another master digest'),
    'meta names another entry': (lambda box: box.meta.update(bf16_sha256=OTHER), 'names another store entry'),
    'meta without files': (lambda box: box.meta.update(files={}), 'names no files'),
    'meta with a path': (lambda box: box.meta['files'].update({'../x': {'sha256': OTHER, 'bytes': 1}}),
                         'file entry that is not valid'),
    'meta without a hash': (lambda box: next(iter(box.meta['files'].values())).pop('sha256'),
                            'file entry that is not valid'),
    'meta that names itself': (lambda box: box.meta['files'].update({'meta.json': {'sha256': OTHER, 'bytes': 1}}),
                               'file entry that is not valid'),
    'shard of another size': (lambda box: next(iter(box.meta['files'].values())).update(bytes=7),
                              'is missing or has another size'),
    'shard that is not there': (lambda box: box.meta['files'].update(
        {'fp32-master-00003-of-00002.safetensors': {'sha256': OTHER, 'bytes': 1}}), 'is missing or has another size'),
    'weight file changed': (lambda box: box.weights.update({'config.json': b'{"hidden": 1}'}),
                            'not the files of the descriptor: config.json'),
    'weight file missing': (lambda box: box.descriptor['files'].update({'tokenizer.json': OTHER}),
                            'weight file tokenizer.json is missing'),
    'descriptor with a path': (lambda box: box.descriptor['files'].update({'sub/x.bin': OTHER}),
                               'file entry that is not valid'),
    'descriptor without a hash': (lambda box: box.descriptor['files'].update({'config.json': 'x'}),
                                  'file entry that is not valid'),
    'descriptor without an id': (lambda box: box.descriptor.pop('id'), 'names no checkpoint id'),
    'descriptor without files': (lambda box: box.descriptor.update(files={}), 'names no files'),
    'no model files': (lambda box: box.weights.update(
        {'weights.safetensors': box.weights.pop('model-00001-of-00002.safetensors')}), 'wrote no model files'),
}


@pytest.mark.parametrize('name', sorted(REFUSALS))
def test_collect_refuses(box, name):
    change, why = REFUSALS[name]
    change(box)
    box.write()
    with pytest.raises(S.StepError, match=why):
        S.collect(box.cfg())
    # Nothing was named for the upload and nothing was put before the probe server.
    assert not (box.work / 'state.json').exists() and not (box.work / 'candidate.json').exists()
    assert not box.ckpt.exists()


@pytest.mark.parametrize('value', [OTHER, None])
def test_collect_runs_only_under_the_lease_of_its_config(box, monkeypatch, value):
    box.write()
    if value is None:
        monkeypatch.delenv(S.EXPECTED_MASTER_ENV)
    else:
        monkeypatch.setenv(S.EXPECTED_MASTER_ENV, value)
    with pytest.raises(S.StepError, match='does not run under the lease of its config'):
        S.collect(box.cfg())
    assert not (box.work / 'state.json').exists()


def test_only_the_last_exit_line_and_the_last_event_of_a_name_count(box):
    box.log_text = 'rc=1\n' + json.dumps({'event': 'done', 'checkpoint': OTHER}) + '\nnot json {\n{"no": "event"}\n[1]'
    box.write()
    S.collect(box.cfg())
    assert box.state()['meta']['checkpoint_id'] == box.checkpoint


@pytest.mark.parametrize('change', [
    lambda cfg: cfg.update(schema='other'),
    lambda cfg: cfg.update(expected_master='x'),
    lambda cfg: cfg.pop('base_checkpoint'),
    lambda cfg: cfg.update(trainer_host='../host'),
    lambda cfg: cfg.pop('sharded'),
    lambda cfg: cfg.update(sharded='yes'),
    lambda cfg: cfg.update(zero3_profile='a b'),
    lambda cfg: cfg.update(out='relative/out'),
    lambda cfg: cfg.update(state='/tmp/../etc/state.json'),
    lambda cfg: cfg.pop('probe_result'),
])
def test_a_config_that_breaks_a_rule_is_refused(box, change):
    change(box.config)
    box.write()
    with pytest.raises(S.StepError):
        box.cfg()


def run_step(box, command, env=None):
    return subprocess.run(
        [sys.executable, str(STEPS), command, '--config', str(box.work / 'steps.json')],
        capture_output=True, text=True, timeout=60, env=dict(os.environ, **(env or {})),
    )


def test_a_step_is_a_command_that_ends_with_zero_or_one(box):
    box.write()
    done = run_step(box, 'collect')
    assert done.returncode == 0 and done.stdout.startswith('collected '), done.stderr
    box.block['source'] = 'bootstrap-upcast-root-override'
    box.write()
    failed = run_step(box, 'collect')
    assert failed.returncode == 1 and 'collect: the fp32 master of the trainer came from bootstrap' in failed.stderr
    (box.work / 'steps.json').write_text('{}')
    assert run_step(box, 'collect').returncode == 1
    assert subprocess.run([sys.executable, str(STEPS), 'other', '--config', 'x'], capture_output=True).returncode == 2


def test_the_end_of_the_train_log_is_in_the_output_of_a_collect_that_found_no_training(box):
    # ``job.py`` looks in the output of the failed step for the words that put the fault on the batch.
    box.rc, box.log_text = 3, 'x' * 10000 + '\nAssertionError: dispatch refused: bad rollout'
    box.drop('done')
    box.write()
    failed = run_step(box, 'collect')
    assert failed.returncode == 1
    assert 'dispatch refused: bad rollout' in failed.stderr and 'exit code 3, done event missing' in failed.stderr
    assert len(failed.stderr) < 6000


# ----------------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------------

def live_builder():
    spec = importlib.util.spec_from_file_location('live_plan_builder', PLAN_BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_probe_plan


def test_the_plan_is_the_plan_of_the_live_builder(box):
    # No value here is a default of the builder.
    box.probe_args.update(k=4, max_tokens=1234, env_id='other_env', max_num_seqs=64)
    box.write()
    S.collect(box.cfg())
    assert S.plan(box.cfg()) == 'plan for %s: 4 tasks' % box.checkpoint[:12]
    path = box.work / 'private' / 'eval-plan.json'
    built = json.loads(path.read_text())
    assert built == live_builder()(
        box.checkpoint, INDICES, SEED, k=4, max_tokens=1234, env_id='other_env', max_num_seqs=64,
        harness=box.probe_args['harness'])
    assert built['envs'] == [{'env_id': 'other_env', 'indices': INDICES, 'k': 4, 'max_tokens': 1234}]
    assert (built['checkpoint_id'], built['seed'], built['concurrency_per_replica']) == (box.checkpoint, SEED, 64)
    assert built['manifest_projection'] == {'mimo_answer_projection': 'boxed', 'sampling_contract': {'t': 1}}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize('change', [
    lambda doc: doc.update(schema='other'),
    lambda doc: doc.update(candidate_id='x'),
    lambda doc: doc.pop('train'),
    lambda doc: doc.update(train=[1]),
])
def test_a_candidate_file_that_collect_did_not_write_is_refused(box, change):
    box.write()
    S.collect(box.cfg())
    doc = box.candidate()
    change(doc)
    (box.work / 'candidate.json').write_text(json.dumps(doc))
    box.probed()
    for step in (S.plan, S.outcome):
        with pytest.raises(S.StepError, match='the candidate file is not valid'):
            step(box.cfg())


@pytest.mark.parametrize('change', [
    lambda args: args.update(schema='other'),
    lambda args: args.update(indices=[]),
    lambda args: args.pop('indices'),
    lambda args: args.update(plan_seed=''),
    lambda args: args.update(plan_seed=7),
    lambda args: args.update(env_id=''),
    lambda args: args.update(k=0),
    lambda args: args.update(max_tokens=True),
    lambda args: args.pop('max_num_seqs'),
    lambda args: args.update(mask_key='short'),
    lambda args: args.pop('mask_key'),
    lambda args: args.update(harness=[1]),
])
def test_probe_arguments_that_break_a_rule_are_refused(box, change):
    box.write()
    S.collect(box.cfg())
    change(box.probe_args)
    box.write()
    with pytest.raises(S.StepError, match='probe'):
        S.plan(box.cfg())
    assert not (box.work / 'private' / 'eval-plan.json').exists()
    box.probed()
    with pytest.raises(S.StepError, match='probe'):
        S.outcome(box.cfg())
    assert not (box.work / 'outcome.json').exists()


def test_no_plan_without_a_collected_checkpoint_or_a_builder(box, tmp_path):
    box.write()
    with pytest.raises(S.StepError, match='cannot read the candidate file'):
        S.plan(box.cfg())
    S.collect(box.cfg())
    broken = tmp_path / 'builder.py'
    for text in ('raise RuntimeError("x")', 'def other(): pass', 'import nothing_like_this'):
        broken.write_text(text)
        box.config['plan_builder'] = str(broken)
        box.write()
        with pytest.raises(S.StepError, match='cannot load the plan builder'):
            S.plan(box.cfg())
    broken.write_text('def build_probe_plan(cid, *a, **k):\n    return {"checkpoint_id": "x"}\n')
    with pytest.raises(S.StepError, match='the plan is for another checkpoint'):
        S.plan(box.cfg())


# ----------------------------------------------------------------------------
# outcome
# ----------------------------------------------------------------------------

def collected(box):
    box.write()
    S.collect(box.cfg())
    return box


def test_the_outcome_names_no_task_no_seed_and_no_run(box):
    collected(box)
    probed = box.probed()
    assert S.outcome(box.cfg()) == 'probe ok for %s' % box.checkpoint[:12]
    raw = (box.work / 'outcome.json').read_text()
    doc = json.loads(raw)
    assert (doc['schema'], doc['outcome']) == (J.BOX_OUTCOME_SCHEMA, 'probed')
    payload = doc['payload']
    assert payload['candidate_id'] == box.checkpoint and payload['train'] == box.candidate()['train']
    assert payload['probe']['status'] == 'ok'
    result = payload['probe']['result']
    cells = result['envs'][ENV_ID]['per_task']
    assert sorted(cells) == sorted(M.mask(KEY, ENV_ID, index) for index in INDICES)
    for index in INDICES:
        assert cells[M.mask(KEY, ENV_ID, index)] == probed['probe_result']['envs'][ENV_ID]['per_task'][str(index)]
    # The numbers of the probe stay. The seed, the run name, the task ids and every path do not.
    assert {key: result[key] for key in ('samples', 'planned_samples', 'partial', 'concurrency', 'wall_s')} == {
        'samples': 32, 'planned_samples': 32, 'partial': False, 'concurrency': 448, 'wall_s': 5000.5}
    assert result['requests_per_replica'] == [9, 8, 8, 7]
    assert result['envs'][ENV_ID]['samples'] == 32 and result['envs'][ENV_ID]['mean'] == 0.5
    assert set(result['envs'][ENV_ID]) == {'per_task', 'samples', 'mean'}
    assert result['serve_layout']['layout'] == {'name': 'rtx5090-tp2', 'gpus_per_replica': 2, 'max_num_seqs': 112}
    assert result['serve_layout']['runtime'] == {'venv': None, 'vllm_version': '0.33.0'}
    for private in [SEED, KEY, 'pe-abc-def', '/opt/venvs'] + ['"%d"' % index for index in INDICES] + [
            '%d' % index for index in INDICES if index > 100]:
        assert private not in raw, private
    assert 'seed' not in result and 'run_name' not in result and 'checkpoint_id' not in result


def test_the_judge_gets_the_task_ids_back_with_the_key_of_the_batch(box):
    collected(box)
    probed = box.probed()
    S.outcome(box.cfg())
    public = box.outcome()['payload']['probe']['result']
    again = M.unmask(public, KEY, {ENV_ID: INDICES})
    assert again['envs'][ENV_ID]['per_task'] == probed['probe_result']['envs'][ENV_ID]['per_task']
    assert again['samples'] == 32 and again['envs'][ENV_ID]['mean'] == 0.5
    assert public['envs'][ENV_ID]['per_task'] != again['envs'][ENV_ID]['per_task']
    # A round with more tasks than the box probed keeps the cells that are there.
    assert M.unmask(public, KEY, {ENV_ID: INDICES + [5]})['envs'][ENV_ID]['per_task'] == again['envs'][ENV_ID][
        'per_task']
    with pytest.raises(M.MaskError, match='no task of the round'):
        M.unmask(public, KEY, {ENV_ID: INDICES[1:]})
    with pytest.raises(M.MaskError, match='no task of the round'):
        M.unmask(public, M.mask_key(SEED, 'run0001', 'e8-other'), {ENV_ID: INDICES})
    with pytest.raises(M.MaskError, match='not an environment of the round'):
        M.unmask(public, KEY, {'other_env': INDICES})
    for broken in ({}, {'envs': {}}, {'envs': {ENV_ID: {}}}, {'envs': {ENV_ID: {'per_task': [1]}}}, None, []):
        with pytest.raises(M.MaskError):
            M.unmask(broken, KEY, {ENV_ID: INDICES})


def test_the_key_of_a_batch_is_made_from_private_text_and_differs_for_every_batch():
    assert M.mask_key(SEED, 'run0001', 'e7-0123456789ab') == KEY and len(KEY) == 64 >= S.MIN_MASK_KEY
    others = {
        M.mask_key(SEED + 'x', 'run0001', 'e7-0123456789ab'), M.mask_key(SEED, 'run0002', 'e7-0123456789ab'),
        M.mask_key(SEED, 'run0001', 'e8-0123456789ab'), M.mask_key(SEED, 'run0001e7', '-0123456789ab'),
    }
    assert len(others) == 4 and KEY not in others
    for private in ('', None, 7):
        with pytest.raises(M.MaskError):
            M.mask_key(private, 'run0001', 'e7')


def test_a_probe_that_was_closed_early_is_an_outcome_and_says_so(box):
    collected(box)
    box.probed(status='finalized', ok=False)
    S.outcome(box.cfg())
    assert box.outcome()['payload']['probe']['status'] == 'finalized'


def test_numbers_that_are_no_numbers_and_text_do_not_leave_the_box(box):
    collected(box)
    doc = box.probed()
    result = doc['probe_result']
    result.update(note='free text', nan=float('nan'), inf=float('inf'), huge=10 ** 400, nothing=None, flag=True)
    result['envs'][ENV_ID].update(label='x', worst=float('-inf'))
    first = result['envs'][ENV_ID]['per_task'][str(INDICES[0])]
    first.update(text='rollout text', rate=float('nan'), nested={'a': 1}, samples=[1, 2], t=1.5)
    result['requests_per_replica'] = [3, 'x', None, -1, True]
    (box.work / 'probe' / 'RESULT.json').write_text(json.dumps(doc))
    S.outcome(box.cfg())
    public = box.outcome()['payload']['probe']['result']
    assert 'note' not in public and 'nan' not in public and 'inf' not in public and 'nothing' not in public
    assert public['huge'] == 10 ** 400 and public['flag'] is True
    assert set(public['envs'][ENV_ID]) == {'per_task', 'samples', 'mean'}
    assert public['envs'][ENV_ID]['per_task'][M.mask(KEY, ENV_ID, INDICES[0])] == {
        'n': 8, 'c': 3, 'w': 4, 'u': 1, 'b': 0, 'x': 0, 't': 1.5}
    assert public['requests_per_replica'] == [3, 0, 0, 0, 0]


@pytest.mark.parametrize('change, why', [
    (dict(status='failed', ok=False, error='ServeError: no layout of the plan serves'), 'no layout of the plan serves'),
    (dict(status='running'), 'the probe did not finish'),
    (dict(candidate_id=OTHER), 'the probe ran on another checkpoint'),
    (dict(probe_result=None), 'has no environments'),
    (dict(probe_result={'envs': {}}), 'has no environments'),
    (dict(probe_result={'envs': {ENV_ID: {'per_task': {}}}}), 'has no counters of tasks'),
    (dict(probe_result={'envs': {ENV_ID: {'samples': 1}}}), 'has no counters of tasks'),
    (dict(probe_result={'envs': {ENV_ID: {'per_task': {'7': 3}}}}), 'has a task without counters'),
])
def test_a_probe_without_a_usable_result_is_no_outcome(box, change, why):
    collected(box)
    box.probed(**change)
    with pytest.raises(S.StepError, match=why):
        S.outcome(box.cfg())
    assert not (box.work / 'outcome.json').exists()


def test_no_outcome_without_the_result_of_the_probe(box):
    collected(box)
    with pytest.raises(S.StepError, match='cannot read the result of the probe'):
        S.outcome(box.cfg())
    (box.work / 'probe' / 'RESULT.json').write_text('[]')
    with pytest.raises(S.StepError, match='is not valid'):
        S.outcome(box.cfg())


# ----------------------------------------------------------------------------
# HEAD blocks
# ----------------------------------------------------------------------------

HEAD_SEEDS = {0: 'seed-of-block-0', 2: 'seed-of-block-2'}


def with_head(box):
    """A lease that runs the HEAD blocks 0 and 2 after its own probe."""
    actor, aux = box.pst / 'in' / 'actor' / BASE_CHECKPOINT, box.work / 'base'
    actor.mkdir(parents=True)
    aux.mkdir()
    (actor / 'model-00001-of-00001.safetensors').write_bytes(b'head' * 100)
    (actor / 'config.json').write_bytes(b'{}')
    (aux / 'descriptor.json').write_text(json.dumps({'id': BASE_CHECKPOINT, 'files': {}}))
    box.config['head'] = {
        'weights': str(actor), 'descriptor': str(aux / 'descriptor.json'),
        'blocks': {str(n): {'plan': str(box.work / 'private' / ('head-plan-%d.json' % n)),
                            'result': str(box.work / ('head-%d' % n) / 'RESULT.json')} for n in HEAD_SEEDS},
    }
    box.probe_args['head'] = [{'block': n, 'plan_seed': seed, 'k': 4} for n, seed in HEAD_SEEDS.items()]
    return collected(box)


def head_probed(box, number, **changes):
    doc = box.probed()
    doc.update(candidate_id=BASE_CHECKPOINT)
    doc.update(changes)
    folder = box.work / ('head-%d' % number)
    folder.mkdir(exist_ok=True)
    (folder / 'RESULT.json').write_text(json.dumps(doc))
    return doc


def test_headplan_writes_the_plan_of_each_block_for_the_base_checkpoint(box):
    with_head(box)
    assert S.headplan(box.cfg()) == 'plans for 2 HEAD blocks of %s' % BASE_CHECKPOINT[:12]
    build = live_builder()
    for number, seed in HEAD_SEEDS.items():
        plan = json.loads((box.work / 'private' / ('head-plan-%d.json' % number)).read_text())
        assert plan == build(BASE_CHECKPOINT, INDICES, seed, k=4, max_tokens=65536, env_id=ENV_ID, max_num_seqs=112,
                             harness=box.probe_args['harness'])
        assert plan['checkpoint_id'] == BASE_CHECKPOINT and plan['envs'][0]['k'] == 4 and plan['seed'] == seed
    served = box.ckpt / BASE_CHECKPOINT
    assert sorted(path.name for path in served.iterdir()) == [
        '.verified', 'config.json', 'model-00001-of-00001.safetensors']
    assert json.loads((box.ckpt / (BASE_CHECKPOINT + '.descriptor.json')).read_text())['id'] == BASE_CHECKPOINT


def test_headplan_needs_blocks_and_the_weights_of_the_base(box):
    collected(box)
    with pytest.raises(S.StepError, match='runs no HEAD block'):
        S.headplan(box.cfg())
    with_head_box = Trained(box.root / 'second')
    with_head(with_head_box)
    (with_head_box.work / 'base' / 'descriptor.json').unlink()
    with pytest.raises(S.StepError, match='weights of the base are not on the box'):
        S.headplan(with_head_box.cfg())


def test_the_outcome_holds_each_head_block_under_masked_names(box):
    with_head(box)
    S.plan(box.cfg())
    S.headplan(box.cfg())
    probed = box.probed()
    head_probed(box, 0)
    assert S.outcome(box.cfg()) == 'probe ok for %s, HEAD blocks: 0 ok, 2 missing' % box.checkpoint[:12]
    payload = box.outcome()['payload']
    head = payload['head']
    assert head['checkpoint_id'] == BASE_CHECKPOINT and [entry['block'] for entry in head['blocks']] == [0, 2]
    first, second = head['blocks']
    assert (first['k'], first['max_tokens'], first['status']) == (4, 65536, 'ok')
    again = M.unmask(first['result'], KEY, {ENV_ID: INDICES})
    assert again['envs'][ENV_ID]['per_task'] == probed['probe_result']['envs'][ENV_ID]['per_task']
    assert second == {'block': 2, 'k': 4, 'max_tokens': 65536, 'status': 'missing'}
    raw = (box.work / 'outcome.json').read_text()
    for private in list(HEAD_SEEDS.values()) + [SEED, KEY]:
        assert private not in raw


@pytest.mark.parametrize('change, status', [
    (dict(status='failed', ok=False, error='ServeError: no layout of the plan serves'), 'failed'),
    (dict(candidate_id=OTHER), 'unusable'),
    (dict(probe_result={'envs': {}}), 'unusable'),
])
def test_a_head_block_that_did_not_finish_never_stops_the_outcome_of_the_batch(box, change, status):
    with_head(box)
    box.probed()
    head_probed(box, 0, **change)
    (box.work / 'head-2').mkdir()
    (box.work / 'head-2' / 'RESULT.json').write_text('{')
    S.outcome(box.cfg())
    payload = box.outcome()['payload']
    assert payload['probe']['status'] == 'ok'
    first, second = payload['head']['blocks']
    assert first['status'] == status and 'result' not in first
    assert second['status'] == 'unusable' and 'result' not in second


def test_the_serve_record_gets_the_sampling_of_the_plan_when_the_bundle_has_no_manifest(box):
    collected(box)
    S.plan(box.cfg())
    box.probed()
    S.outcome(box.cfg())
    harness = box.probe_args['harness']
    want = {'harness': harness['harness'], 'sampling_contract': harness['sampling_contract']}
    assert box.outcome()['payload']['probe']['result']['serve_layout']['sampling'] == want
    # A serve record that names its sampling keeps it, whole.
    doc = box.probed()
    doc['probe_result']['serve_layout']['sampling'] = {'harness': 'of/the/bundle', 'sampling_contract': {'n': 1}}
    (box.work / 'probe' / 'RESULT.json').write_text(json.dumps(doc))
    S.outcome(box.cfg())
    assert box.outcome()['payload']['probe']['result']['serve_layout']['sampling'] == {
        'harness': 'of/the/bundle', 'sampling_contract': {'n': 1}}
    # No plan and no manifest: the record says nothing, and the judge gives no verdict.
    box.probed()
    (box.work / 'private' / 'eval-plan.json').unlink()
    S.outcome(box.cfg())
    assert box.outcome()['payload']['probe']['result']['serve_layout']['sampling'] is None


@pytest.mark.parametrize('change', [
    lambda box: box.config['head'].pop('weights'),
    lambda box: box.config['head'].update(descriptor='relative/descriptor.json'),
    lambda box: box.config['head'].update(blocks={'x': {'plan': '/a/p', 'result': '/a/r'}}),
    lambda box: box.config['head'].update(blocks={'1': {'plan': '/a/p'}}),
    lambda box: box.config.update(head=[]),
])
def test_a_head_section_that_breaks_a_rule_is_refused(box, change):
    with_head(box)
    change(box)
    box.write()
    with pytest.raises(S.StepError, match='HEAD block'):
        box.cfg()


@pytest.mark.parametrize('head', [
    'x', [{'block': -1, 'plan_seed': 's', 'k': 4}], [{'block': 0, 'plan_seed': '', 'k': 4}],
    [{'block': 0, 'plan_seed': 's', 'k': 0}], [{'block': True, 'plan_seed': 's', 'k': 4}], [3],
])
def test_head_blocks_in_the_probe_arguments_that_break_a_rule_are_refused(box, head):
    collected(box)
    box.probe_args['head'] = head
    box.write()
    with pytest.raises(S.StepError, match='HEAD block'):
        S.load_probe_args(box.cfg())
