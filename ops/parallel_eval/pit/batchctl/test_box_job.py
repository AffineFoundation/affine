"""The job that runs on a rented box (``box/job.py``).

The job is one file that imports the standard library only, so it has its
own copy of the formats. The first tests compare that copy with the
controller's. The others run the job against a store behind signed URLs:
functions of the job in this process, and the whole job as its own process,
with small scripts in the place of the trainer and the probe.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from . import files as F
from . import layout as L
from .box import job as J
from .test_fakes import CONTENT_TYPE, ObjectServer, model_ids, put_model
from ...durable import canonical

BOX = Path(__file__).parent / 'box'
JOB = BOX / 'job.py'
RUN = 'run0001'
BATCH = 'e7-0123456789ab'
FENCE = 3
CHUNK = 100_000
CANDIDATE = 'd' * 64
NEW_MASTER = 'e' * 64
NEW_ENTRY = 'f' * 64

# The trainer of these tests: it reads the weights and the fp32 master of the
# base from the places of the job, writes a new file for each, and writes the
# state file that names them.
TRAIN = r'''
import hashlib, json, os, sys
cfg = json.load(open(sys.argv[1]))
os.makedirs(cfg['out'], exist_ok=True)
expected = os.environ['AFFINE_FP32_EXPECTED_MASTER_HASH']
files = []
for group, role in (('bf16', 'bf16_weights'), ('fp32', 'fp32_master')):
    for name in sorted(os.listdir(cfg[group])):
        data = open(os.path.join(cfg[group], name), 'rb').read() + b'+trained'
        path = os.path.join(cfg['out'], group + '-' + name)
        open(path, 'wb').write(data)
        files.append({'path': group + '/' + name, 'role': role, 'source': path,
                      'sha256': hashlib.sha256(data).hexdigest()})
desc = os.path.join(cfg['out'], 'descriptor.json')
open(desc, 'w').write(json.dumps({'id': cfg['candidate']}))
files.append({'path': 'aux/descriptor.json', 'role': 'aux', 'source': desc})
state = {'schema': 'affine-batchctl-box-state-v1',
         'trainer': {'input_master_digest': cfg.get('input_master', expected),
                     'output_master_digest': cfg['output_master']},
         'meta': {'fp32_master_digest': cfg['output_master'], 'checkpoint_id': cfg['candidate'],
                  'fp32_entry': cfg['entry']},
         'files': files}
state.update(cfg.get('state_changes', {}))
json.dump(state, open(cfg['state'], 'w'))
json.dump({'expected': expected, 'override': sorted(k for k in os.environ if 'BOOTSTRAP' in k),
           'extra': os.environ.get('EXTRA')}, open(cfg['seen'], 'w'))
'''


@pytest.fixture
def server():
    made = ObjectServer()
    yield made
    made.close()


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(J, 'RETRY_PAUSE_S', 0.01)


def gone(pid):
    """No process with this id runs. One that ended and waits for its parent counts as gone."""
    try:
        with open('/proc/%d/stat' % int(pid)) as handle:
            return handle.read().rpartition(')')[2].split()[0] == 'Z'
    except OSError:
        return True


class Lease:
    """One lease on one box: a base state in the store, the directories of
    the box under ``tmp_path``, and the job document."""

    def __init__(self, server, tmp_path, *, seed='base'):
        self.server, self.store = server, server.store
        self.layout = L.Layout('shadow/batchctl/%s/' % RUN)
        self.manifest, self.files = put_model(self.store, self.layout, seed, chunk=CHUNK)
        self.checkpoint, self.entry, self.master = model_ids(self.files)
        self.base = self.manifest.state_id
        self.tmp = tmp_path
        self.work = tmp_path / 'lease'
        self.work.mkdir()
        self.box = tmp_path / 'box'
        self.dirs = {name: self.box / name for name in ('actor', 'fp32', 'aux', 'out', 'served')}
        self.train = tmp_path / 'train.py'
        self.train.write_text(TRAIN)
        self.seen = tmp_path / 'seen.json'
        self.train_cfg = {
            'bf16': str(self.dirs['actor']), 'fp32': str(self.dirs['fp32']), 'out': str(self.dirs['out']),
            'candidate': CANDIDATE, 'output_master': NEW_MASTER, 'entry': NEW_ENTRY,
            'state': str(self.work / 'state.json'), 'seen': str(self.seen),
        }
        self.meminfo = tmp_path / 'meminfo'
        self.meminfo.write_text('MemTotal:       528000000 kB\nMemFree: 1 kB\n')
        self.outcome = self.work / 'outcome.json'

    def staging(self, name):
        return self.layout.staging_prefix(BATCH, FENCE) + name

    def slots(self, count=None):
        names = F.slots_for(self.manifest, CHUNK) if count is None else [F.slot_name(n) for n in range(count)]
        return {name: self.server.put_url(self.staging('obj/' + name)) for name in list(names) + [F.INDEX_NAME]}

    def outcome_script(self, doc):
        return "cat > %s <<'EOF'\n%s\nEOF\n" % (self.outcome, json.dumps(doc))

    def steps(self):
        probed = {'schema': J.BOX_OUTCOME_SCHEMA, 'outcome': 'probed', 'payload': {'samples': 8, 'score': 0.5}}
        return [
            {'name': 'train', 'script': '%s %s %s' % (sys.executable, self.train, self.work / 'train.json'),
             'timeout_s': 60, 'state': str(self.work / 'state.json')},
            {'name': 'probe', 'script': self.outcome_script(probed), 'timeout_s': 60},
        ]

    def job(self, **changes):
        doc = {
            'schema': J.JOB_SCHEMA, 'run_id': RUN, 'batch_id': BATCH, 'fence': FENCE, 'attempt': 1, 'base_seq': 0,
            'base_state': self.base, 'expected_fp32_master_digest': self.master,
            'heartbeat_every_s': 0.05, 'job_ttl_s': 120, 'content_type': CONTENT_TYPE, 'chunk_bytes': 1 << 20,
            'http': {'tries': 4, 'pause_s': 0.01, 'timeout_s': 20},
            'base': self.base_urls(self.manifest),
            'put': {
                'objects': self.slots(),
                'heartbeat': self.server.put_url(self.staging('HEARTBEAT.json')),
                'result': self.server.put_url(self.staging('result.json')),
                'complete': self.server.put_url(self.staging('COMPLETE.json')),
            },
            'env': {'EXTRA': 'from-job'},
            'require': {
                'gpus': 2, 'gpu_name': 'RTX5090', 'gpu_query': ['printf', 'NVIDIA GeForce RTX 5090\\nRTX 5090\\n'],
                'ram_gb': 300, 'meminfo': str(self.meminfo), 'disk_free_gb': 0.001,
            },
            'binding': {'checkpoint_id': self.checkpoint, 'epoch': 7},
            'place': [
                {'prefix': 'bf16/', 'dir': str(self.dirs['actor']), 'links': [str(self.dirs['served'])]},
                {'prefix': 'fp32/', 'dir': str(self.dirs['fp32'])},
                {'prefix': 'aux/', 'dir': str(self.dirs['aux'])},
            ],
            'expect': [
                {'file': str(self.dirs['fp32'] / 'meta.json'), 'key': ['master_digest'], 'equals': self.master},
                {'file': str(self.dirs['fp32'] / 'meta.json'), 'key': ['bf16_sha256'], 'equals': self.entry},
            ],
            'steps': self.steps(), 'outcome': str(self.outcome),
            'wipe': [str(self.box), str(self.work / 'private')], 'workers': 4,
        }
        doc.update(changes)
        return doc

    def base_urls(self, manifest):
        return {
            'manifest': self.server.get_url(self.layout.state_manifest(manifest.state_id)),
            'objects': {ref.name: self.server.get_url(self.layout.state_object(manifest.state_id, ref.name))
                        for ref in manifest.objects},
        }

    def write(self, job, train_cfg=None):
        (self.work / 'train.json').write_text(json.dumps(dict(self.train_cfg, **(train_cfg or {}))))
        (self.work / 'private').mkdir(exist_ok=True)
        (self.work / 'private' / 'probe-args.json').write_text('{"indices": [1, 2, 3]}')
        path = self.work / 'job.json'
        path.write_text(json.dumps(job))
        return path

    def run(self, job=None, *, train_cfg=None, env=None, timeout=120):
        path = self.write(job if job is not None else self.job(), train_cfg)
        return subprocess.run(
            [sys.executable, str(JOB), str(path)], capture_output=True, text=True, timeout=timeout,
            env=dict(os.environ, **(env or {})),
        )

    def staged(self, name):
        if self.store.head(self.staging(name)) is None:
            return None
        return self.store.get(self.staging(name), max_bytes=1 << 30).data

    def puts(self):
        prefix = self.layout.staging_prefix(BATCH, FENCE)
        return [key[len(prefix):] for _method, key, status in self.server.seen('PUT') if status == 200]

    def package(self):
        return L.parse_complete(self.staged('COMPLETE.json'), batch_id=BATCH, fence=FENCE, base_state=self.base)

    def index(self):
        raw = self.store.get(self.layout.state_object(self.base, F.INDEX_NAME), max_bytes=F.MAX_INDEX_BYTES).data
        return F.parse_index(raw, {ref.name: ref for ref in self.manifest.objects})


# ----------------------------------------------------------------------------
# One format on both sides
# ----------------------------------------------------------------------------

def test_the_box_and_the_controller_share_every_format_name_and_limit():
    assert J.STATE_SCHEMA == L.SCHEMA_STATE
    assert J.COMPLETE_SCHEMA == L.SCHEMA_COMPLETE
    assert J.HEARTBEAT_SCHEMA == L.SCHEMA_HEARTBEAT
    assert J.FILES_SCHEMA == F.FILES_SCHEMA
    assert J.ROLES == L.ROLES and J.REQUIRED_ROLES == L.REQUIRED_ROLES
    assert J.META_FP32_MASTER_DIGEST == L.META_FP32_MASTER_DIGEST
    assert (J.INDEX_NAME, J.SLOT_NAME) == (F.INDEX_NAME, F.SLOT_NAME)
    assert (J.MAX_OBJECTS, J.MAX_RESULT_BYTES, J.MAX_COMPLETE_BYTES) == (
        L.MAX_OBJECTS, L.MAX_RESULT_BYTES, L.MAX_COMPLETE_BYTES)
    assert J.MAX_INDEX_BYTES == F.MAX_INDEX_BYTES
    assert J.MAX_MANIFEST_BYTES <= L.MAX_COMPLETE_BYTES
    assert (J.HEX64.pattern, J.SAFE_NAME.pattern, J.META_KEY.pattern, J.PATH.pattern) == (
        L.HEX64.pattern, L.SAFE_NAME.pattern, L.META_KEY.pattern, F.PATH.pattern)
    for value in ({'b': [1, 2, {'a': None}], 'a': 'x'}, {'z': 'é', 'n': 1.5, 't': True}, []):
        assert J.canonical(value) == canonical(value)
    for path in ('bf16/a.safetensors', 'meta.json', 'a/../b', 'a/b/c', '', '/x', 'bf16/x y', 'BF16/x', 'bf16/x\n',
                 'bf16/a..b', 'bf16/a.b', None, 7):
        assert J.check_path(path) is F.is_path(path)
    # The number of the box starts above every number of the host, and both fit a heartbeat.
    body = L.heartbeat_body(fence=FENCE, seq=J.HEARTBEAT_BASE + 1, stage='x')
    assert L.parse_heartbeat(body, fence=FENCE) == J.HEARTBEAT_BASE + 1


def test_the_box_code_imports_the_standard_library_only():
    stdlib = set(sys.stdlib_module_names)
    found = sorted(BOX.glob('*.py'))
    assert JOB in found
    for path in found:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, '%s has a relative import' % path.name
                names = [node.module or '']
            else:
                continue
            for name in names:
                assert name.split('.')[0] in stdlib, '%s imports %s' % (path.name, name)
        text = path.read_text().lower()
        # Nothing on a box holds a credential or talks to the chain, the website, Discord or the provider.
        for word in ('bittensor', 'substrate', 'extrinsic', 'set_weights', 'discord', 'webhook', 'wallet',
                     'boto', 'access_key', 'secret', 'lium', 'api_key'):
            assert word not in text, '%s mentions %s' % (path.name, word)


def test_what_the_box_uploads_is_what_the_controller_parses(server, tmp_path):
    lease = Lease(server, tmp_path)
    done = lease.run()
    assert done.returncode == J.EXIT_DONE, done.stderr
    raw = lease.staged('COMPLETE.json')
    package = lease.package()
    result = lease.staged('result.json')
    assert (package.result_sha, package.result_size) == (L.sha256_hex(result), len(result))
    manifest = package.manifest
    assert manifest is not None and manifest.fp32_master_digest == NEW_MASTER
    assert (manifest.parent, manifest.batch_id) == (lease.base, BATCH)
    assert dict(manifest.meta) == {
        'fp32_master_digest': NEW_MASTER, 'checkpoint_id': CANDIDATE, 'fp32_entry': NEW_ENTRY,
    }
    objects = {ref.name: ref for ref in manifest.objects}
    for ref in manifest.objects:
        data = lease.staged('obj/' + ref.name)
        assert (len(data), L.sha256_hex(data)) == (ref.size, ref.sha256), ref.name
    index = F.parse_index(lease.staged('obj/' + F.INDEX_NAME), objects)
    assert (index.checkpoint_id, index.fp32_entry, index.fp32_master_digest) == (CANDIDATE, NEW_ENTRY, NEW_MASTER)
    # Every file of the new state is the file that the training step wrote.
    want = {path: data + b'+trained' for path, data in lease.files.items() if not path.startswith('aux/')}
    want['aux/descriptor.json'] = json.dumps({'id': CANDIDATE}).encode()
    assert sorted(entry.path for entry in index.files) == sorted(want)
    for entry in index.files:
        data = b''.join(lease.staged('obj/' + name) for name in entry.parts)
        assert data == want[entry.path] and entry.sha256 == L.sha256_hex(data)
    # The index bytes are the bytes that the host side writes for the same files.
    assert lease.staged('obj/' + F.INDEX_NAME) == F.index_body(
        index.files, checkpoint_id=CANDIDATE, fp32_entry=NEW_ENTRY, fp32_master_digest=NEW_MASTER)
    # The manifest in COMPLETE.json is the canonical manifest: its id is the hash of its body.
    assert json.loads(raw)['state'] == manifest.doc()

    doc = json.loads(result)
    assert doc['schema'] == J.RESULT_SCHEMA and doc['outcome'] == 'probed' and doc['reason'] == ''
    assert (doc['run_id'], doc['batch_id'], doc['fence'], doc['base_state']) == (RUN, BATCH, FENCE, lease.base)
    assert doc['candidate'] == {
        'state_id': manifest.state_id, 'checkpoint_id': CANDIDATE, 'fp32_master_digest': NEW_MASTER,
    }
    assert doc['base'] == {'checkpoint_id': lease.checkpoint, 'fp32_master_digest': lease.master}
    assert doc['binding'] == {'checkpoint_id': lease.checkpoint, 'epoch': 7}
    assert doc['payload'] == {'samples': 8, 'score': 0.5}
    assert doc['box']['gpus'] == ['NVIDIA GeForce RTX 5090', 'RTX 5090'] and doc['box']['ram_gb'] > 500
    assert doc['box']['base_bytes'] == sum(len(data) for data in lease.files.values())
    assert {'checks', 'base', 'fetch', 'step:train', 'step:probe', 'upload', 'total'} <= set(doc['seconds'])


def test_a_large_file_is_uploaded_in_parts(server, tmp_path):
    lease = Lease(server, tmp_path)
    big = tmp_path / 'big.bin'
    data = hashlib.sha256(b'big').digest() * 80_000          # 2.56 MB: three parts of 1 MiB
    big.write_bytes(data)
    small = tmp_path / 'small.bin'
    small.write_bytes(b'master')
    empty = tmp_path / 'empty.bin'
    empty.write_bytes(b'')
    files = [
        {'path': 'bf16/big', 'role': 'bf16_weights', 'source': str(big), 'sha256': L.sha256_hex(data)},
        {'path': 'fp32/small', 'role': 'fp32_master', 'source': str(small)},
        {'path': 'aux/empty', 'role': 'aux', 'source': str(empty)},
    ]
    job = lease.job()
    job['put']['objects'] = lease.slots(count=8)
    done = lease.run(job, train_cfg={'state_changes': {'files': files}})
    assert done.returncode == J.EXIT_DONE, done.stderr
    manifest = lease.package().manifest
    objects = {ref.name: ref for ref in manifest.objects}
    index = F.parse_index(lease.staged('obj/' + F.INDEX_NAME), objects)
    entry = index.file('bf16/big')
    assert [objects[name].size for name in entry.parts] == [1 << 20, 1 << 20, len(data) - (2 << 20)]
    assert b''.join(lease.staged('obj/' + name) for name in entry.parts) == data
    assert index.file('aux/empty').parts == () and index.file('aux/empty').size == 0
    assert len(manifest.objects) == 5


# ----------------------------------------------------------------------------
# The whole job
# ----------------------------------------------------------------------------

def test_the_job_commits_last_wipes_the_box_and_starts_once(server, tmp_path):
    lease = Lease(server, tmp_path)
    beats = []
    server.before_put = lambda key, data: beats.append(json.loads(data)) if key.endswith('HEARTBEAT.json') else None
    done = lease.run()
    assert done.returncode == J.EXIT_DONE, done.stderr
    puts = [name for name in lease.puts() if name != 'HEARTBEAT.json']
    # Everything of the state, then the result, then COMPLETE.json as the last object.
    assert puts[-2:] == ['result.json', 'COMPLETE.json']
    assert all(name.startswith('obj/') for name in puts[:-2]) and puts[-3] == 'obj/' + F.INDEX_NAME
    # The heartbeat number starts above every number of the host and only grows.
    numbers = [beat['seq'] for beat in beats]
    assert numbers[0] == J.HEARTBEAT_BASE + 1 and numbers == sorted(set(numbers)) and len(numbers) >= 3
    assert all(beat['fence'] == FENCE and beat['schema'] == L.SCHEMA_HEARTBEAT for beat in beats)
    assert beats[-1]['stage'] == 'done'
    assert L.parse_heartbeat(lease.staged('HEARTBEAT.json'), fence=FENCE) == numbers[-1]
    # The weights, the inputs and the signed URLs are gone. The claim and the logs stay.
    assert not lease.box.exists() and not (lease.work / 'private').exists()
    assert not (lease.work / 'job.json').exists()
    assert (lease.work / 'claimed').is_dir() and (lease.work / 'logs' / 'train.log').exists()
    # The steps got the digest of the lease, and no bootstrap override.
    assert json.loads(lease.seen.read_text()) == {'expected': lease.master, 'override': [], 'extra': 'from-job'}

    # A second start of the same lease does nothing.
    before = dict(lease.store.objs)
    count = len(server.log)
    again = lease.run()
    assert again.returncode == J.EXIT_STARTED and 'already started' in again.stderr
    assert lease.store.objs == before and len(server.log) == count


def test_the_bootstrap_override_never_reaches_a_step(server, tmp_path):
    lease = Lease(server, tmp_path)
    done = lease.run(env={'AFFINE_FP32_BOOTSTRAP_OVERRIDE': 'x', 'MY_BOOTSTRAP_OVERRIDE': 'y',
                          'AFFINE_FP32_EXPECTED_MASTER_HASH': '0' * 64})
    assert done.returncode == J.EXIT_DONE, done.stderr
    seen = json.loads(lease.seen.read_text())
    assert seen['override'] == [] and seen['expected'] == lease.master


@pytest.mark.parametrize('change', [
    lambda job: job['env'].update(AFFINE_FP32_BOOTSTRAP_OVERRIDE='1'),
    lambda job: job['env'].update(affine_fp32_bootstrap_override='1'),
    lambda job: job['env'].update(AFFINE_FP32_EXPECTED_MASTER_HASH='0' * 64),
    lambda job: job['steps'][0].update(env={'AFFINE_FP32_BOOTSTRAP_OVERRIDE': '1'}),
    lambda job: job['steps'][0].update(env={'AFFINE_FP32_EXPECTED_MASTER_HASH': '0' * 64}),
    lambda job: job['steps'][0].update(script='AFFINE_FP32_BOOTSTRAP_OVERRIDE=1 true'),
    lambda job: job['steps'][0].update(script='export AFFINE_FP32_BOOTSTRAP_OVERRIDE = 1; true'),
    lambda job: job.update(expected_fp32_master_digest=''),
    lambda job: job.pop('expected_fp32_master_digest'),
    lambda job: job.update(base_state='x'),
    lambda job: job.update(schema='affine-batchctl-job-v0'),
    lambda job: job.update(fence=0),
    lambda job: job['put'].pop('complete'),
    lambda job: job['put']['objects'].pop(F.INDEX_NAME),
    lambda job: job['put']['objects'].update({'../x': 'http://x'}),
    lambda job: job.update(place=[]),
    lambda job: job['place'][0].update(dir='relative'),
    lambda job: job['steps'].append(dict(job['steps'][0])),
    lambda job: job['steps'][1].update(state='/tmp/second-state.json'),
    lambda job: job['steps'][0].update(timeout_s=0),
    lambda job: job['steps'][0].update(batch_caused=['(']),
    lambda job: job.update(wipe=['/']),
    lambda job: job.update(wipe=['/root']),
    lambda job: job.update(wipe=['/workspace']),
    lambda job: job.update(wipe=['relative/path']),
    lambda job: job.update(wipe=['/tmp/../etc']),
    lambda job: job.update(http={'tries': 0}),
    lambda job: job.update(http={'other': 1}),
    lambda job: job.update(binding={'epoch': 7}),
])
def test_a_job_that_breaks_a_rule_does_not_start(server, tmp_path, change):
    lease = Lease(server, tmp_path)
    job = lease.job()
    change(job)
    done = lease.run(job)
    assert done.returncode == J.EXIT_BAD_JOB, done.stderr
    # Nothing was read, nothing was written, and the lease is not taken.
    assert server.log == [] and not (lease.work / 'claimed').exists()


def test_a_script_may_remove_the_override_but_never_set_it(server, tmp_path):
    # The train command of the live lane starts with ``env -u AFFINE_FP32_BOOTSTRAP_OVERRIDE NAME=value``.
    lease = Lease(server, tmp_path)
    job = lease.job()
    job['steps'][0]['script'] = 'env -u AFFINE_FP32_BOOTSTRAP_OVERRIDE AFFINE_FP32_EXPECTED_MASTER_HASH=x true'
    J.check_job(job)
    # A value that a step takes from a file cannot bring the override in either.
    source = tmp_path / 'value.json'
    source.write_text(json.dumps({'v': 'AFFINE_FP32_BOOTSTRAP_OVERRIDE=1'}))
    step = {'name': 'x', 'script': 'env @@VALUE@@ true', 'timeout_s': 1,
            'subst': {'@@VALUE@@': {'file': str(source), 'key': ['v'], 'match': '.*'}}}
    with pytest.raises(J.JobError, match='the bootstrap override is never set'):
        J.Runner(job, str(lease.work)).script(step)


def test_the_job_never_wipes_the_directory_that_holds_its_claim(server, tmp_path):
    lease = Lease(server, tmp_path)
    for wiped in (str(lease.work), str(lease.work.parent)):
        done = lease.run(lease.job(wipe=[wiped]))
        assert done.returncode == J.EXIT_BAD_JOB and 'must stay' in done.stderr
    assert server.log == []


def no_decision(lease, done, code=J.EXIT_FAILED):
    """The job ended without a decision: no COMPLETE.json, and the box is clean."""
    assert done.returncode == code, done.stderr
    assert lease.staged('COMPLETE.json') is None and lease.staged('result.json') is None
    assert not lease.box.exists() and not (lease.work / 'job.json').exists()
    assert not (lease.work / 'private').exists()
    return done.stderr


def test_a_step_that_fails_is_no_decision(server, tmp_path):
    lease = Lease(server, tmp_path)
    job = lease.job()
    job['steps'][1]['script'] = 'echo the probe broke; exit 7'
    assert 'step probe ended with code 7' in no_decision(lease, lease.run(job))
    assert json.loads(lease.staged('HEARTBEAT.json'))['stage'] == 'failed'


def test_a_step_that_says_the_batch_is_at_fault_is_a_decision_without_a_state(server, tmp_path):
    lease = Lease(server, tmp_path)
    job = lease.job()
    job['steps'][0].update(script='echo "dispatch refused: bad rollout"; exit 3', batch_caused=[r'dispatch refused'])
    done = lease.run(job)
    assert done.returncode == J.EXIT_DONE, done.stderr
    assert lease.package().manifest is None
    doc = json.loads(lease.staged('result.json'))
    assert (doc['outcome'], doc['reason'], doc['candidate']) == ('not_trained', 'batch_rejected:train', None)
    assert doc['base']['checkpoint_id'] == lease.checkpoint
    assert not any(name.startswith('obj/') for name in lease.puts())


def test_the_same_failure_without_the_words_of_the_rule_is_a_failure_of_the_box(server, tmp_path):
    lease = Lease(server, tmp_path)
    job = lease.job()
    job['steps'][0].update(script='echo "CUDA error"; exit 3', batch_caused=[r'dispatch refused'])
    assert 'step train ended with code 3' in no_decision(lease, lease.run(job))


def test_the_last_step_can_say_that_the_batch_was_not_trained(server, tmp_path):
    lease = Lease(server, tmp_path)
    job = lease.job()
    job['steps'][1]['script'] = lease.outcome_script(
        {'schema': J.BOX_OUTCOME_SCHEMA, 'outcome': 'not_trained', 'reason': 'no_rollouts'})
    done = lease.run(job)
    assert done.returncode == J.EXIT_DONE, done.stderr
    assert lease.package().manifest is None
    doc = json.loads(lease.staged('result.json'))
    assert (doc['outcome'], doc['reason'], doc['candidate'], doc['payload']) == ('not_trained', 'no_rollouts', None, {})


@pytest.mark.parametrize('outcome', [
    {'schema': 'other', 'outcome': 'probed'},
    {'schema': J.BOX_OUTCOME_SCHEMA, 'outcome': 'green'},
    {'schema': J.BOX_OUTCOME_SCHEMA, 'outcome': 'probed', 'payload': [1]},
    {'schema': J.BOX_OUTCOME_SCHEMA, 'outcome': 'probed', 'reason': 7},
    [],
    None,
])
def test_an_outcome_file_that_is_not_valid_is_no_decision(server, tmp_path, outcome):
    lease = Lease(server, tmp_path)
    job = lease.job()
    job['steps'][1]['script'] = 'true' if outcome is None else lease.outcome_script(outcome)
    assert 'the outcome file of the last step is not valid' in no_decision(lease, lease.run(job))


def test_a_job_without_a_step_that_writes_a_state_is_no_decision(server, tmp_path):
    lease = Lease(server, tmp_path)
    job = lease.job()
    job['steps'][0].pop('state')
    assert 'no step wrote a state' in no_decision(lease, lease.run(job))


def test_a_batch_that_is_bound_to_another_checkpoint_is_refused_before_any_weight_is_read(server, tmp_path):
    lease = Lease(server, tmp_path)
    done = lease.run(lease.job(binding={'checkpoint_id': '9' * 64}))
    assert done.returncode == J.EXIT_DONE, done.stderr
    doc = json.loads(lease.staged('result.json'))
    assert (doc['outcome'], doc['reason'], doc['candidate']) == ('not_trained', J.REASON_OTHER_CHECKPOINT, None)
    assert doc['base']['checkpoint_id'] == lease.checkpoint and doc['binding'] == {'checkpoint_id': '9' * 64}
    assert lease.package().manifest is None
    # Only the manifest and the index were read. No step ran.
    read = sorted(key.rsplit('/', 1)[-1] for _method, key, _status in server.seen('GET'))
    assert read == [F.INDEX_NAME, 'manifest.json']
    assert not lease.seen.exists() and not (lease.work / 'logs').exists()


def test_a_step_can_use_a_value_that_an_earlier_step_wrote(server, tmp_path):
    lease = Lease(server, tmp_path)
    job = lease.job()
    told = tmp_path / 'told.txt'
    token = {'file': str(lease.work / 'state.json'), 'key': ['meta', 'checkpoint_id'], 'match': '[0-9a-f]{64}'}
    job['steps'][1].update(script='echo @@CANDIDATE@@ > %s\n%s' % (told, job['steps'][1]['script']),
                           subst={'@@CANDIDATE@@': token})
    done = lease.run(job)
    assert done.returncode == J.EXIT_DONE, done.stderr
    assert told.read_text().strip() == CANDIDATE
    # A value of another form than the step allows never reaches a shell.
    (lease.work / 'state.json').write_text(json.dumps({'meta': {'checkpoint_id': 'x; touch /tmp/owned'}}))
    with pytest.raises(J.JobError, match='has no usable meta.checkpoint_id'):
        J.Runner(job, str(lease.work)).script(job['steps'][1])
    (lease.work / 'state.json').unlink()
    with pytest.raises(J.JobError, match='cannot read state.json'):
        J.Runner(job, str(lease.work)).script(job['steps'][1])


def test_a_step_that_runs_too_long_is_ended_with_everything_it_started(server, tmp_path):
    lease = Lease(server, tmp_path)
    job = lease.job()
    one, two = tmp_path / 'one.pid', tmp_path / 'two.pid'
    job['steps'][0].update(
        script='sleep 300 & echo $! > %s\n(sleep 300; true) & echo $! > %s\nwait' % (one, two), timeout_s=0.5)
    started = time.time()
    assert 'step train ran longer than' in no_decision(lease, lease.run(job))
    assert time.time() - started < 20
    assert gone(one.read_text()) and gone(two.read_text())


def test_what_a_step_leaves_running_ends_with_the_step(server, tmp_path):
    lease = Lease(server, tmp_path)
    job = lease.job()
    left = tmp_path / 'left.pid'
    job['steps'][0]['script'] += '\nnohup sleep 300 > /dev/null 2>&1 &\necho $! > %s\n' % left
    done = lease.run(job)
    assert done.returncode == J.EXIT_DONE, done.stderr
    assert gone(left.read_text())


def test_the_job_ends_itself_after_its_lifetime(server, tmp_path):
    lease = Lease(server, tmp_path)
    job = lease.job(job_ttl_s=0.6)
    pid_file = tmp_path / 'step.pid'
    job['steps'][0]['script'] = 'echo $$ > %s; sleep 300' % pid_file
    assert 'longer than its lifetime' in no_decision(lease, lease.run(job), J.EXIT_DEADLINE)
    assert not pid_file.exists() or gone(pid_file.read_text())


def test_a_stop_signal_ends_the_step_and_wipes_the_box(server, tmp_path):
    lease = Lease(server, tmp_path)
    job = lease.job()
    pid_file = tmp_path / 'step.pid'
    job['steps'][0]['script'] = 'echo $$ > %s; sleep 300' % pid_file
    path = lease.write(job)
    proc = subprocess.Popen([sys.executable, str(JOB), str(path)], stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 30
    while not (pid_file.exists() and pid_file.read_text().strip()) and time.time() < deadline:
        time.sleep(0.05)
    proc.send_signal(signal.SIGTERM)
    _out, err = proc.communicate(timeout=60)
    assert proc.returncode == J.EXIT_FAILED and 'told to stop' in err
    assert not lease.box.exists() and not (lease.work / 'job.json').exists()
    assert not (lease.work / 'private').exists()
    assert lease.staged('COMPLETE.json') is None
    assert gone(pid_file.read_text())


@pytest.mark.parametrize('train_cfg,why', [
    ({'input_master': '0' * 64}, 'did not start from the fp32 master of the base'),
    ({'state_changes': {'trainer': {'input_master_digest': None, 'output_master_digest': NEW_MASTER}}},
     'did not start from the fp32 master of the base'),
    ({'state_changes': {'trainer': {'output_master_digest': NEW_MASTER}}},
     'did not start from the fp32 master of the base'),
    ({'state_changes': {'meta': {'fp32_master_digest': '1' * 64}}}, 'does not name the fp32 master'),
    ({'state_changes': {'meta': {'checkpoint_id': CANDIDATE}}}, 'does not name the fp32 master'),
    ({'state_changes': {'meta': {'fp32_master_digest': NEW_MASTER, 'checkpoint_id': 7}}}, 'not valid'),
    ({'state_changes': {'meta': {'fp32_master_digest': NEW_MASTER, 'fp32_entry': ''}}}, 'not valid'),
    ({'state_changes': {'meta': {'fp32_master_digest': NEW_MASTER, 'Bad-Key': 'x'}}}, 'not valid'),
    ({'state_changes': {'schema': 'other'}}, 'not valid'),
    ({'state_changes': {'files': []}}, 'not valid'),
    ({'state_changes': {'files': [{'path': 'bf16/w', 'role': 'bf16_weights', 'source': '/nowhere/w'}]}},
     'not valid'),
    ({'state_changes': {'trainer': 'x'}}, 'not valid'),
])
def test_a_state_that_the_trainer_did_not_make_from_the_base_master_is_not_uploaded(
        server, tmp_path, train_cfg, why):
    lease = Lease(server, tmp_path)
    assert why in no_decision(lease, lease.run(train_cfg=train_cfg))
    assert not any(name.startswith('obj/') for name in lease.puts())


def test_a_state_needs_weights_and_an_fp32_master(server, tmp_path):
    lease = Lease(server, tmp_path)
    source = tmp_path / 'only.bin'
    source.write_bytes(b'weights')
    files = [{'path': 'bf16/w', 'role': 'bf16_weights', 'source': str(source)}]
    assert 'has no fp32_master file' in no_decision(lease, lease.run(train_cfg={'state_changes': {'files': files}}))


def test_a_file_that_changed_after_its_step_named_it_is_not_committed(server, tmp_path):
    lease = Lease(server, tmp_path)
    job = lease.job()
    # One weight file of the new state changes after the trainer measured it, as on a bad disk.
    victim = lease.dirs['out'] / 'bf16-model-00001-of-00003.safetensors'
    job['steps'][0]['script'] += '\nprintf x >> %s\n' % victim
    assert 'is not the file that its step named' in no_decision(lease, lease.run(job))


def test_a_state_that_needs_more_parts_than_the_lease_has_slots_is_not_committed(server, tmp_path):
    lease = Lease(server, tmp_path)
    job = lease.job()
    job['put']['objects'] = lease.slots(count=3)
    assert 'more parts than the lease has slots' in no_decision(lease, lease.run(job))


@pytest.mark.parametrize('require,why', [
    ({'gpus': 8}, 'the box has 2 GPUs, the job needs 8'),
    ({'gpu_name': 'B300'}, 'another GPU class'),
    ({'ram_gb': 600}, 'GB RAM'),
    ({'disk_free_gb': 10 ** 9}, 'free disk'),
    ({'gpu_query': ['false']}, 'cannot list the GPUs'),
    ({'meminfo': '/nonexistent/meminfo'}, 'cannot read the host RAM'),
])
def test_a_box_that_is_too_small_does_no_work(server, tmp_path, require, why):
    lease = Lease(server, tmp_path)
    job = lease.job()
    job['require'].update(require)
    assert why in no_decision(lease, lease.run(job))
    assert server.seen('GET') == []


def test_an_upload_that_the_store_refuses_once_is_sent_again(server, tmp_path):
    lease = Lease(server, tmp_path)
    server.faults[('PUT', lease.staging('obj/p-000002'))] = ['500', '500']
    server.faults[('PUT', lease.staging('result.json'))] = ['500']
    server.faults[('PUT', lease.staging('COMPLETE.json'))] = ['500']
    # A heartbeat that cannot be written does not stop the work.
    server.faults[('PUT', lease.staging('HEARTBEAT.json'))] = ['403', '500', '403']
    done = lease.run()
    assert done.returncode == J.EXIT_DONE, done.stderr
    manifest = lease.package().manifest
    for ref in manifest.objects:
        assert L.sha256_hex(lease.staged('obj/' + ref.name)) == ref.sha256
    assert 'heartbeat not written' in done.stderr


def test_without_the_result_there_is_no_complete(server, tmp_path):
    lease = Lease(server, tmp_path)
    server.faults[('PUT', lease.staging('result.json'))] = ['403']
    assert 'result: HTTP 403' in no_decision(lease, lease.run())


# ----------------------------------------------------------------------------
# The base state
# ----------------------------------------------------------------------------

def in_process(lease, **changes):
    job = lease.job(**changes)
    job['_path'] = str(lease.work / 'job.json')
    return J.check_job(job)


def fetch(job):
    objects, files, info = J.open_base(job)
    return J.fetch_files(job, objects, files, 4), info


def placed(lease):
    out = {}
    for group, name in (('bf16', 'actor'), ('fp32', 'fp32'), ('aux', 'aux')):
        for path in lease.dirs[name].iterdir():
            out['%s/%s' % (group, path.name)] = path.read_bytes()
    return out


def test_the_base_state_is_downloaded_to_its_places_and_checked(server, tmp_path):
    lease = Lease(server, tmp_path)
    size, info = fetch(in_process(lease))
    assert placed(lease) == lease.files and size == sum(len(data) for data in lease.files.values())
    assert info == {'checkpoint_id': lease.checkpoint, 'fp32_entry': lease.entry, 'fp32_master_digest': lease.master}
    # The weights are also where the probe server reads them: the same file, not a second copy.
    for path in lease.dirs['actor'].iterdir():
        assert (lease.dirs['served'] / path.name).samefile(path)
    J.check_expect(in_process(lease))


@pytest.mark.parametrize('fault', ['500', 'cut', 'corrupt'])
def test_a_download_that_breaks_goes_on_and_ends_with_the_right_bytes(server, tmp_path, fault):
    lease = Lease(server, tmp_path)
    part = lease.index().file('bf16/model-00002-of-00003.safetensors').parts[1]
    key = lease.layout.state_object(lease.base, part)
    server.faults[('GET', key)] = [fault, fault]
    server.faults[('GET', lease.layout.state_manifest(lease.base))] = ['500']
    fetch(in_process(lease))
    assert placed(lease) == lease.files
    assert len(server.seen('GET', key)) == 3


def test_a_broken_download_resumes_at_the_missing_byte(server, tmp_path):
    lease = Lease(server, tmp_path)
    part = lease.index().file('bf16/model-00001-of-00003.safetensors').parts[0]
    key = lease.layout.state_object(lease.base, part)
    data = lease.store.get(key, max_bytes=1 << 30).data
    out_path = tmp_path / 'one.bin'
    for faults, statuses in ((['cut'], [-1, 206]), (['cut', 'norange'], [-1, 200])):
        del server.log[:]
        server.faults[('GET', key)] = list(faults)
        with open(out_path, 'wb') as out:
            out.write(b'before')
            sha = J.http_get_into(
                server.get_url(key), out, len(data), L.sha256_hex(data), hashlib.sha256(b'before'), 'part')
        assert out_path.read_bytes() == b'before' + data
        assert sha.hexdigest() == L.sha256_hex(b'before' + data)
        assert [status for _method, _key, status in server.seen('GET', key)] == statuses


def test_a_small_document_that_ends_early_is_read_again(server, tmp_path):
    lease = Lease(server, tmp_path)
    key = lease.layout.state_manifest(lease.base)
    server.faults[('GET', key)] = ['cut']
    assert J.http_get_bytes(server.get_url(key), 1 << 20, 'manifest') == lease.manifest.body()
    with pytest.raises(J.JobError, match='larger than'):
        J.http_get_bytes(server.get_url(key), 10, 'manifest')


@pytest.mark.parametrize('what,why', [
    ('part', 'object p-[0-9]+: ValueError'),
    ('index', 'the file index is not the object of the manifest'),
    ('manifest', 'the manifest is not the manifest of the base state'),
])
def test_bytes_that_are_not_the_bytes_of_the_base_state_are_refused(server, tmp_path, what, why):
    lease = Lease(server, tmp_path)
    part = lease.index().file('bf16/config.json').parts[0]
    key = {
        'part': lease.layout.state_object(lease.base, part),
        'index': lease.layout.state_object(lease.base, F.INDEX_NAME),
        'manifest': lease.layout.state_manifest(lease.base),
    }[what]
    data = lease.store.get(key, max_bytes=1 << 30).data
    if what == 'part':
        other = data[:-1] + bytes([data[-1] ^ 1])
    elif what == 'index':
        # Another index that is valid in itself and has the same size.
        doc = json.loads(data)
        other = canonical(dict(doc, checkpoint_id=('0' if doc['checkpoint_id'][0] != '0' else '1') + doc[
            'checkpoint_id'][1:]))
        assert len(other) == len(data)
    else:
        # Another manifest that is valid in itself.
        other = canonical(dict(json.loads(data), batch_id='other'))
    lease.store.put(key, other)
    with pytest.raises(J.JobError, match=why):
        fetch(in_process(lease))
    if what == 'part':
        # The file of that part never gets its name.
        assert not (lease.dirs['actor'] / 'config.json').exists()
        assert not (lease.dirs['served'] / 'config.json').exists()
    else:
        # No weight was asked for.
        assert not lease.box.exists()
        assert not any('/p-' in key for _method, key, _status in server.seen('GET'))


def test_a_url_that_the_store_refuses_is_not_tried_again(server, tmp_path):
    lease = Lease(server, tmp_path)
    key = lease.layout.state_manifest(lease.base)
    server.faults[('GET', key)] = ['403']
    with pytest.raises(J.JobError, match='HTTP 403'):
        fetch(in_process(lease))
    assert len(server.seen('GET', key)) == 1


def index_doc(lease):
    raw = lease.store.get(lease.layout.state_object(lease.base, F.INDEX_NAME), max_bytes=F.MAX_INDEX_BYTES).data
    return json.loads(raw)


def restate(lease, index, meta=None):
    """The state of ``lease`` again, with another index or other meta. The
    manifest of the new state is valid: its id names these bytes."""
    index = canonical(index)
    parts = [ref for ref in lease.manifest.objects if ref.name != F.INDEX_NAME]
    manifest = F.manifest_of(
        parent=None, batch_id='genesis', parts=parts, index=index, meta=meta or dict(lease.manifest.meta))
    for ref in parts:
        data = lease.store.get(lease.layout.state_object(lease.base, ref.name), max_bytes=1 << 30).data
        lease.store.put(lease.layout.state_object(manifest.state_id, ref.name), data)
    lease.store.put(lease.layout.state_object(manifest.state_id, F.INDEX_NAME), index)
    lease.store.put(lease.layout.state_manifest(manifest.state_id), manifest.body())
    return manifest


@pytest.mark.parametrize('where,why', [
    ('lease', 'the fp32 master digest of the base is not the digest that the lease expects'),
    ('manifest', 'the fp32 master digest of the base is not the digest that the lease expects'),
    ('index', 'the file index names another fp32 master digest'),
])
def test_another_fp32_master_digest_than_the_lease_expects_stops_the_job_before_any_weight(
        server, tmp_path, where, why):
    lease = Lease(server, tmp_path)
    if where == 'lease':
        job = lease.job(expected_fp32_master_digest='0' * 64)
    else:
        # A state whose manifest and whose index name two digests. The lease expects the other one.
        if where == 'manifest':
            manifest = restate(lease, index_doc(lease), dict(lease.manifest.meta, fp32_master_digest='0' * 64))
        else:
            manifest = restate(lease, dict(index_doc(lease), fp32_master_digest='0' * 64))
        job = lease.job(base_state=manifest.state_id, base=lease.base_urls(manifest))
    assert why in no_decision(lease, lease.run(job))
    assert not any('/p-' in key for _method, key, _status in server.seen('GET'))
    assert not lease.seen.exists()


BIG = 'bf16/model-00002-of-00003.safetensors'


def entry_of(doc, path):
    return next(item for item in doc['files'] if item['path'] == path)


@pytest.mark.parametrize('change,why', [
    # The parts of one file in another order: every part is right, the file is not.
    (lambda doc: entry_of(doc, BIG)['parts'].reverse(), 'is not the file of the index'),
    (lambda doc: entry_of(doc, BIG).update(sha256='0' * 64), 'is not the file of the index'),
    (lambda doc: entry_of(doc, BIG).update(role='aux'), 'the file index of the base state is not valid'),
    (lambda doc: entry_of(doc, BIG)['parts'].pop(), 'the file index of the base state is not valid'),
    (lambda doc: entry_of(doc, BIG).update(size=entry_of(doc, BIG)['size'] + 1), 'index of the base state is not valid'),
    (lambda doc: entry_of(doc, 'bf16/config.json').update(parts=entry_of(doc, 'aux/descriptor.json')['parts']),
     'the file index of the base state is not valid'),
    (lambda doc: entry_of(doc, BIG).update(parts=entry_of(doc, BIG)['parts'][:3] + [F.INDEX_NAME]),
     'the file index of the base state is not valid'),
    (lambda doc: entry_of(doc, BIG).update(path='../../etc/passwd'), 'the file index of the base state is not valid'),
    (lambda doc: entry_of(doc, BIG).update(path='/etc/passwd'), 'the file index of the base state is not valid'),
    (lambda doc: doc['files'].append(dict(entry_of(doc, 'bf16/config.json'), parts=[], size=0)),
     'the file index of the base state is not valid'),
    (lambda doc: entry_of(doc, BIG).update(path='other/model.safetensors'), 'the job names no place'),
    # Two files with the same parts.
    (lambda doc: doc['files'].append(dict(entry_of(doc, BIG), path='bf16/second')),
     'the file index of the base state is not valid'),
    # An object of the manifest that no file uses.
    (lambda doc: doc['files'].remove(entry_of(doc, 'fp32/meta.json')), 'the file index of the base state is not valid'),
    (lambda doc: doc.update(schema='affine-batchctl-files-v0'), 'the file index of the base state is not valid'),
])
def test_an_index_that_does_not_fit_its_parts_is_refused(server, tmp_path, change, why):
    lease = Lease(server, tmp_path)
    doc = index_doc(lease)
    change(doc)
    manifest = restate(lease, doc)
    job = in_process(lease, base_state=manifest.state_id, base=lease.base_urls(manifest))
    with pytest.raises(J.JobError, match=why):
        fetch(job)
    assert not (lease.dirs['actor'] / BIG.split('/')[1]).exists()
    assert not Path('/etc/passwd.part').exists()


def test_the_index_is_no_part_of_a_file_on_the_box_either():
    sha = 'a' * 64
    objects = {'p-000000': ('bf16_weights', sha, 1), F.INDEX_NAME: ('aux', sha, 5)}
    files = [{'path': 'bf16/w', 'role': 'bf16_weights', 'size': 1, 'sha256': sha, 'parts': ['p-000000']}]
    index = {'schema': J.FILES_SCHEMA, 'files': files, 'checkpoint_id': None, 'fp32_entry': None,
             'fp32_master_digest': sha}
    assert J.parse_index(canonical(index), objects)[0] == files
    files.append({'path': 'aux/self', 'role': 'aux', 'size': 5, 'sha256': sha, 'parts': [F.INDEX_NAME]})
    with pytest.raises(J.JobError, match='the file index of the base state is not valid'):
        J.parse_index(canonical(index), objects)


@pytest.mark.parametrize('change', [
    lambda doc: doc['objects'][0].update(name='../x'),
    lambda doc: doc['objects'][0].update(role='weights'),
    lambda doc: doc['objects'][0].update(sha256='x'),
    lambda doc: doc['objects'][0].update(size=0),
    lambda doc: doc['objects'][0].update(size='7'),
    lambda doc: doc['objects'].append(dict(doc['objects'][0])),
    lambda doc: doc.update(objects=[]),
    lambda doc: doc.update(schema='affine-batchctl-state-v0'),
    lambda doc: doc.update(meta=[]),
    lambda doc: doc.pop('meta'),
])
def test_a_manifest_that_breaks_a_rule_is_refused(change):
    doc = {'schema': J.STATE_SCHEMA, 'parent': None, 'batch_id': 'genesis', 'meta': {},
           'objects': [{'name': 'p-000000', 'role': 'bf16_weights', 'sha256': 'a' * 64, 'size': 1}]}
    J.parse_manifest(canonical(doc), L.sha256_hex(canonical(doc)))
    change(doc)
    with pytest.raises(J.JobError, match='the manifest of the base state is not valid'):
        J.parse_manifest(canonical(doc), L.sha256_hex(canonical(doc)))


def test_the_master_on_the_disk_must_be_the_master_of_the_lease(server, tmp_path):
    lease = Lease(server, tmp_path)
    job = in_process(lease)
    fetch(job)
    J.check_expect(job)
    meta = lease.dirs['fp32'] / 'meta.json'
    doc = json.loads(meta.read_text())
    meta.write_text(json.dumps(dict(doc, master_digest='0' * 64)))
    with pytest.raises(J.JobError, match='master_digest is not what the lease expects'):
        J.check_expect(job)
    meta.unlink()
    with pytest.raises(J.JobError, match='cannot read'):
        J.check_expect(job)


def test_a_master_on_the_disk_that_is_not_the_one_of_the_lease_stops_the_job_before_training(server, tmp_path):
    lease = Lease(server, tmp_path)
    job = lease.job()
    job['expect'][1]['equals'] = '0' * 64
    assert 'bf16_sha256 is not what the lease expects' in no_decision(lease, lease.run(job))
    assert not lease.seen.exists()


def test_a_url_reads_or_writes_one_key_only(server, tmp_path):
    lease = Lease(server, tmp_path)
    head = server.put_url(lease.layout.head)
    other = server.put_url(lease.staging('COMPLETE.json')).replace('COMPLETE.json', 'result.json')
    as_get = server.put_url(lease.staging('COMPLETE.json'))
    with pytest.raises(J.JobError, match='HTTP 403'):
        J.http_put_bytes(other, b'{}', CONTENT_TYPE, 'other key')
    with pytest.raises(J.JobError, match='HTTP 403'):
        J.http_get_bytes(as_get, 100, 'read with a write URL')
    with pytest.raises(J.JobError, match='HTTP 403'):
        J.http_put_bytes(head, b'{}', 'text/plain', 'wrong content type')
    assert lease.store.head(lease.layout.head) is None and lease.staged('result.json') is None
