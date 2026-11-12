"""Tests of the genesis seed against moto (a fake S3 in this process).

A fake live HEAD is put into a bucket: the weights under ``<ckpt_prefix>/<H>/model/`` and the fp32
master under ``fp32-master/<H>/``. Every sha256 in the two documents is the real hash of the test
bytes, unless a test says that a claim is wrong.

"The controller accepts the state" is tested with the real ``effects._verify_state``, on the guarded
store of the run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from ...durable import canonical
from ..shadow import ShadowWriteError
from . import effects as E
from . import files as F
from . import genesis as GEN
from . import guard as G
from .genesis import DEFAULT_CKPT_PREFIX, GenesisError, live_head, plan, read_live_head, seed
from .interfaces import StoreError
from .layout import GENESIS_BATCH, META_FP32_MASTER_DIGEST, Layout, StateManifest, load_json, parse_manifest, sha256_hex
from .r2 import R2Store

BUCKET = 'batchctl-test'
OTHER = 'batchctl-test-other'
RUN = 'gen00001'
PREFIX = G.shadow_prefix(RUN)
# Every other call of the client counts as a write.
READS = frozenset({'head_object', 'get_object', 'list_objects_v2', 'head_bucket'})


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


H = sha(b'checkpoint')
ENTRY = sha(b'bf16 entry')
DIGEST = sha(b'fp32 master')
WEIGHTS = {
    'config.json': b'{"layers":2}',
    'model-00001-of-00002.safetensors': b'bf16 weights, part one',
    'model-00002-of-00002.safetensors': b'bf16 weights, part two!',
}
MASTER = {
    'meta.json': b'{"master":"fp32"}',
    'shard-00000.bin': b'fp32 shard zero',
    'shard-00001.bin': b'fp32 shard one, longer',
}
LAST_WEIGHT = 'model-00002-of-00002.safetensors'


def weight_key(name: str) -> str:
    return '%s/%s/model/%s' % (DEFAULT_CKPT_PREFIX, H, name)


def master_key(name: str) -> str:
    return 'fp32-master/%s/%s' % (H, name)


def documents() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The descriptor and the master index of the fake live HEAD. The descriptor has both entry forms."""
    files: Dict[str, Any] = {name: sha(data) for name, data in WEIGHTS.items()}
    files['config.json'] = {'sha256': files['config.json'], 'bytes': len(WEIGHTS['config.json'])}
    index = {
        'bf16_sha256': ENTRY, 'master_digest': DIGEST,
        'files': {
            name: {'sha256': sha(data), 'bytes': len(data), 'r2_key': master_key(name)}
            for name, data in MASTER.items()
        },
        'meta_json': {'bf16_sha256': ENTRY, 'master_digest': DIGEST},
    }
    return {'id': H, 'files': files}, index


def put_live(raw: Any, bucket: str = BUCKET) -> None:
    for name, data in WEIGHTS.items():
        raw.put_object(Bucket=bucket, Key=weight_key(name), Body=data)
    for name, data in MASTER.items():
        raw.put_object(Bucket=bucket, Key=master_key(name), Body=data)


def listing(raw: Any, bucket: str = BUCKET, prefix: str = '') -> Dict[str, Tuple[int, str]]:
    return {info.key: (info.size, info.etag) for info in R2Store(raw, bucket).list(prefix)}


def stored_manifest(store: R2Store, state_id: str) -> StateManifest:
    body = store.get(Layout(PREFIX).state_manifest(state_id), max_bytes=1 << 20).data
    assert sha256_hex(body) == state_id
    return parse_manifest(load_json(body), batch_id=GENESIS_BATCH, parent=None)


def verified(store: R2Store, state_id: str) -> Tuple[str, Any]:
    """What the controller says about the state. Only the store and the layout are needed for that."""
    fx = E.Effects.__new__(E.Effects)
    fx.store, fx.layout = G.GuardedStore(store, RUN), Layout(PREFIX)
    return fx._verify_state(state_id, batch_id=GENESIS_BATCH, parent=None, evidence=(), full=True)


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def job(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    return {'payload': {'manifest': {'payload': {'checkpoint': checkpoint}}}}


def tree(root: Path) -> Dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob('*')) if path.is_file()}


class Spy:
    """Forwards to a client. Records each call. A hook runs before a call and may raise."""

    def __init__(self, client: Any):
        self._client = client
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.before: Dict[str, Callable[[Dict[str, Any]], None]] = {}

    def __getattr__(self, name: str) -> Any:
        inner = getattr(self._client, name)

        def call(**kwargs: Any) -> Any:
            self.calls.append((name, kwargs))
            hook = self.before.get(name)
            if hook is not None:
                hook(kwargs)
            return inner(**kwargs)

        return call

    def writes(self) -> List[Tuple[str, Dict[str, Any]]]:
        return [(name, kwargs) for name, kwargs in self.calls if name not in READS]


@pytest.fixture
def raw():
    with mock_aws():
        client = boto3.client(
            's3', region_name='us-east-1', aws_access_key_id='test', aws_secret_access_key='test',
        )
        client.create_bucket(Bucket=BUCKET)
        client.create_bucket(Bucket=OTHER)
        yield client


@pytest.fixture
def spy(raw):
    return Spy(raw)


@pytest.fixture
def store(spy):
    return R2Store(spy, BUCKET)


# ----------------------------------------------------------------------------
# The seed
# ----------------------------------------------------------------------------

def test_the_seed_makes_a_first_state_that_the_controller_accepts(raw, store):
    put_live(raw)
    state_id = seed(store, live_head(*documents()), prefix=PREFIX)
    lay = Layout(PREFIX)
    manifest = stored_manifest(store, state_id)
    assert dict(manifest.meta) == {META_FP32_MASTER_DIGEST: DIGEST, F.META_CHECKPOINT: H, F.META_FP32_ENTRY: ENTRY}

    # R2 holds the objects of the manifest and no other object, each with the size and the sha256 of the manifest.
    objects = {ref.name: ref for ref in manifest.objects}
    assert set(listing(raw, prefix=lay.state_prefix(state_id) + 'obj/')) == {
        lay.state_object(state_id, name) for name in objects
    }
    for name, ref in objects.items():
        digest = store.digest(lay.state_object(state_id, name))
        assert (digest.size, digest.sha256) == (ref.size, ref.sha256)
    assert verified(store, state_id)[0] == E.OK

    index = F.parse_index(store.get(lay.state_object(state_id, F.INDEX_NAME), max_bytes=1 << 20).data, objects)
    assert (index.checkpoint_id, index.fp32_entry, index.fp32_master_digest) == (H, ENTRY, DIGEST)
    assert [entry.path for entry in index.group(F.GROUP_BF16)] == ['bf16/' + name for name in sorted(WEIGHTS)]
    assert [entry.path for entry in index.group(F.GROUP_FP32)] == ['fp32/' + name for name in sorted(MASTER)]
    assert [entry.path for entry in index.group(F.GROUP_AUX)] == [F.DESCRIPTOR_PATH]
    # One part for each file, numbered in the sorted order of the paths, with the role of its group.
    assert [entry.parts for entry in index.files] == [(F.slot_name(number),) for number in range(len(index.files))]
    assert all(entry.role == F.GROUP_ROLE[entry.path.split('/')[0]] for entry in index.files)

    wanted = {'bf16/' + name: data for name, data in WEIGHTS.items()}
    wanted.update({'fp32/' + name: data for name, data in MASTER.items()})
    wanted[F.DESCRIPTOR_PATH] = canonical({'id': H, 'files': {name: sha(data) for name, data in WEIGHTS.items()}})
    for entry in index.files:
        data = store.get(lay.state_object(state_id, entry.parts[0]), max_bytes=1 << 20).data
        assert data == wanted[entry.path] and (entry.size, entry.sha256) == (len(data), sha(data))


def test_a_second_seed_returns_the_same_id_and_writes_nothing(raw, store, spy):
    put_live(raw)
    head = live_head(*documents())
    first = seed(store, head, prefix=PREFIX)
    before = listing(raw)
    del spy.calls[:]
    assert seed(store, head, prefix=PREFIX) == first
    assert spy.writes() == [] and listing(raw) == before


def test_a_seed_that_broke_is_finished_by_the_next_seed(raw, store, spy):
    put_live(raw)
    head = live_head(*documents())

    def third_copy_fails(kwargs: Dict[str, Any]) -> None:
        if len([name for name, _ in spy.calls if name == 'copy_object']) == 3:
            raise ClientError(
                {'Error': {'Code': 'InternalError', 'Message': 'm'}, 'ResponseMetadata': {'HTTPStatusCode': 500}},
                'CopyObject',
            )

    spy.before['copy_object'] = third_copy_fails
    with pytest.raises(StoreError):
        seed(store, head, prefix=PREFIX)
    # The manifest is written last: a state that is not complete has none.
    assert len(listing(raw, prefix=PREFIX)) == 2
    assert not any(key.endswith('manifest.json') for key in listing(raw, prefix=PREFIX))

    del spy.before['copy_object'], spy.calls[:]
    lines: List[str] = []
    state_id = seed(store, head, prefix=PREFIX, log=lines.append)
    copies = [name for name, _ in spy.calls if name == 'copy_object']
    assert len(copies) == len(WEIGHTS) + len(MASTER) - 2
    assert len([line for line in lines if line.startswith('kept ')]) == 2
    assert verified(store, state_id)[0] == E.OK


@pytest.mark.parametrize('source', [None, OTHER])
def test_the_live_side_is_only_read_and_every_write_is_under_the_prefix(raw, store, spy, source):
    live_bucket = source or BUCKET
    put_live(raw, live_bucket)
    raw.put_object(Bucket=BUCKET, Key='live/HEAD.json', Body=b'live')
    before = {bucket: listing(raw, bucket) for bucket in (BUCKET, OTHER)}

    state_id = seed(store, live_head(*documents()), prefix=PREFIX, src_bucket=source)

    after = {bucket: listing(raw, bucket) for bucket in (BUCKET, OTHER)}
    # No key outside the prefix of the run was written, changed or deleted. The etags of the live keys are the same.
    assert {key: value for key, value in after[BUCKET].items() if not key.startswith(PREFIX)} == before[BUCKET]
    assert after[OTHER] == before[OTHER]
    writes = spy.writes()
    assert writes and all(kwargs['Bucket'] == BUCKET and kwargs['Key'].startswith(PREFIX) for _name, kwargs in writes)
    # A live key is used for ``head`` and as the source of a server-side copy, and for nothing else.
    assert {name for name, kwargs in spy.calls if not kwargs.get('Key', '').startswith(PREFIX)} == {'head_object'}
    sources = [kwargs['CopySource'] for name, kwargs in spy.calls if name == 'copy_object']
    assert {item['Bucket'] for item in sources} == {live_bucket}
    assert {item['Key'] for item in sources} == {key for key in before[live_bucket] if key != 'live/HEAD.json'}
    assert verified(store, state_id)[0] == E.OK


def no_weight_object(raw, descriptor, index):
    raw.delete_object(Bucket=BUCKET, Key=weight_key('config.json'))


def no_master_object(raw, descriptor, index):
    raw.delete_object(Bucket=BUCKET, Key=master_key('shard-00001.bin'))


def no_meta_file(raw, descriptor, index):
    del index['files']['meta.json']


def only_the_meta_file(raw, descriptor, index):
    index['files'] = {'meta.json': index['files']['meta.json']}


def another_size(raw, descriptor, index):
    index['files']['shard-00000.bin']['bytes'] += 1


def bad_checkpoint_id(raw, descriptor, index):
    descriptor['id'] = H.upper()


def bad_weight_sha256(raw, descriptor, index):
    descriptor['files'][LAST_WEIGHT] = 'abc'


def bad_master_digest(raw, descriptor, index):
    index['master_digest'] = DIGEST[:-1]


def bad_file_name(raw, descriptor, index):
    descriptor['files']['../' + LAST_WEIGHT] = descriptor['files'].pop(LAST_WEIGHT)


def meta_names_another_master(raw, descriptor, index):
    index['meta_json']['master_digest'] = sha(b'another master')


def live_key_under_the_prefix(raw, descriptor, index):
    index['files']['shard-00000.bin']['r2_key'] = PREFIX + 'states/shard-00000.bin'


def empty_weight_object(raw, descriptor, index):
    raw.put_object(Bucket=BUCKET, Key=weight_key(LAST_WEIGHT), Body=b'')
    descriptor['files'][LAST_WEIGHT] = sha(b'')


def empty_master_object(raw, descriptor, index):
    raw.put_object(Bucket=BUCKET, Key=master_key('shard-00000.bin'), Body=b'')
    index['files']['shard-00000.bin'].update(sha256=sha(b''), bytes=0)


DAMAGE = [
    no_weight_object, no_master_object, no_meta_file, only_the_meta_file, another_size, bad_checkpoint_id,
    bad_weight_sha256, bad_master_digest, bad_file_name, meta_names_another_master, live_key_under_the_prefix,
    empty_weight_object, empty_master_object,
]


@pytest.mark.parametrize('damage', DAMAGE, ids=lambda damage: damage.__name__)
def test_a_live_head_that_is_not_valid_is_refused_and_nothing_is_written(raw, store, spy, damage):
    put_live(raw)
    descriptor, index = documents()
    damage(raw, descriptor, index)
    with pytest.raises(GenesisError):
        seed(store, live_head(descriptor, index), prefix=PREFIX)
    assert spy.writes() == [] and listing(raw, prefix=PREFIX) == {}


@pytest.mark.parametrize('prefix', [
    '', 'live/', 'pit-shadow/other/gen00001/', 'pit-shadow/batchctl//', PREFIX.rstrip('/'), '/' + PREFIX,
    DEFAULT_CKPT_PREFIX + '/',
])
def test_a_prefix_that_is_not_a_shadow_prefix_is_refused_before_any_write(raw, store, spy, prefix):
    put_live(raw)
    before = listing(raw)
    with pytest.raises(ShadowWriteError):
        seed(store, live_head(*documents()), prefix=prefix)
    assert spy.writes() == [] and listing(raw) == before


# ----------------------------------------------------------------------------
# The plan
# ----------------------------------------------------------------------------

def test_plan_is_pure_and_deterministic():
    sizes = {weight_key(name): len(data) for name, data in WEIGHTS.items()}
    sizes.update({master_key(name): len(data) for name, data in MASTER.items()})
    head = live_head(*documents())
    first, second = plan(head, sizes), plan(live_head(*documents()), dict(sizes))
    assert first == second and first.manifest.state_id == second.manifest.state_id
    assert (first.manifest.batch_id, first.manifest.parent) == (GENESIS_BATCH, None)
    assert {(src, size) for src, _name, size in first.copies} == set(sizes.items())
    # ``aux/descriptor.json`` is the first path, so its part is the first slot.
    assert [name for name, _data in first.local] == [F.slot_name(0), F.INDEX_NAME]
    assert first.local[1][1] == first.index
    F.parse_index(first.index, {ref.name: ref for ref in first.manifest.objects})

    bigger = dict(sizes)
    bigger[weight_key('config.json')] += 1
    assert plan(head, bigger).manifest.state_id != first.manifest.state_id
    with pytest.raises(GenesisError):
        plan(head, {key: size for key, size in sizes.items() if key != weight_key('config.json')})


def test_a_wrong_claimed_sha256_gives_a_state_that_the_controller_refuses(raw, store):
    put_live(raw)
    descriptor, index = documents()
    descriptor['files'][LAST_WEIGHT] = sha(b'not the bytes of this file')
    state_id = seed(store, live_head(descriptor, index), prefix=PREFIX)
    lay = Layout(PREFIX)
    manifest = stored_manifest(store, state_id)
    wrong = [
        ref.name for ref in manifest.objects
        if store.digest(lay.state_object(state_id, ref.name)).sha256 != ref.sha256
    ]
    files = F.parse_index(
        store.get(lay.state_object(state_id, F.INDEX_NAME), max_bytes=1 << 20).data,
        {ref.name: ref for ref in manifest.objects},
    )
    assert tuple(wrong) == files.file('bf16/' + LAST_WEIGHT).parts
    assert verified(store, state_id) == (E.INVALID, 'object_mismatch:%s' % wrong[0])


# ----------------------------------------------------------------------------
# The files of the host
# ----------------------------------------------------------------------------

def test_read_live_head_takes_the_descriptor_of_current_and_writes_nothing(tmp_path):
    root = tmp_path / 'own'
    descriptor, index = documents()
    write_json(root / 'life' / 'CURRENT.json', {'epoch_index': 12, 'checkpoint': descriptor})
    write_json(root / 'life' / 'epochs' / '12' / 'TRAINING-JOB.json', job(dict(descriptor, files={'x': sha(b'x')})))
    write_json(root / 'fp32' / (H + '.json'), index)
    before = tree(root)
    assert read_live_head(root / 'life', root / 'fp32') == live_head(descriptor, index)
    assert tree(root) == before


def test_read_live_head_falls_back_to_the_newest_epoch_that_names_the_checkpoint(tmp_path):
    life, masters = tmp_path / 'own' / 'life', tmp_path / 'own' / 'fp32'
    descriptor, index = documents()
    other = {'id': sha(b'another checkpoint'), 'files': {'other.safetensors': sha(b'other')}}
    write_json(life / 'CURRENT.json', {'epoch_index': 11, 'checkpoint': {'id': H}})
    # 10 is above 9 as a number and below it as text.
    write_json(life / 'epochs' / '9' / 'TRAINING-JOB.json', job(dict(descriptor, files={'old': sha(b'old')})))
    write_json(life / 'epochs' / '10' / 'TRAINING-JOB.json', job(descriptor))
    write_json(life / 'epochs' / '11' / 'TRAINING-JOB.json', job(other))
    write_json(life / 'epochs' / 'notes' / 'TRAINING-JOB.json', job(dict(descriptor, files={'no': sha(b'no')})))
    (life / 'epochs' / '12').mkdir()
    write_json(masters / (H + '.json'), index)
    write_json(masters / (other['id'] + '.json'), index)
    assert read_live_head(life, masters) == live_head(descriptor, index)
    assert read_live_head(life, masters, checkpoint=other['id']) == live_head(other, index)


def test_read_live_head_refuses_what_it_cannot_find(tmp_path):
    life, masters = tmp_path / 'own' / 'life', tmp_path / 'own' / 'fp32'
    descriptor, index = documents()
    write_json(life / 'CURRENT.json', {'epoch_index': 12, 'checkpoint': descriptor})
    with pytest.raises(GenesisError, match='master index'):
        read_live_head(life, masters)
    write_json(masters / (H + '.json'), index)
    assert read_live_head(life, masters).checkpoint_id == H
    with pytest.raises(GenesisError, match='no descriptor'):
        read_live_head(life, masters, checkpoint=sha(b'a checkpoint that no epoch names'))
    # A value that is not an id is never used as a file name.
    with pytest.raises(GenesisError, match='no checkpoint id'):
        read_live_head(life, masters, checkpoint='../fp32/' + H)
    (life / 'CURRENT.json').unlink()
    with pytest.raises(GenesisError, match='no checkpoint id'):
        read_live_head(life, masters)


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def test_the_command_line_seeds_the_state_and_prints_one_json_line(raw, tmp_path, monkeypatch, capsys):
    put_live(raw)
    live = listing(raw)
    root = tmp_path / 'own'
    descriptor, index = documents()
    write_json(root / 'life' / 'CURRENT.json', {'epoch_index': 12, 'checkpoint': descriptor})
    write_json(root / 'fp32' / (H + '.json'), index)
    argv = [
        '--run-id', RUN, '--bucket-config', str(root / 'bucket.json'), '--life-root', str(root / 'life'),
        '--master-index-dir', str(root / 'fp32'),
    ]

    def run(args: List[str]) -> Tuple[int, List[Any]]:
        code = GEN.main(args)
        return code, [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    # No bucket config: an error line, and no client is made.
    code, out = run(argv)
    assert code == 1 and len(out) == 1 and set(out[0]) == {'error'}

    monkeypatch.setattr(GEN, 'client_from_bucket_config', lambda path: (raw, BUCKET))
    code, out = run(argv)
    assert code == 0 and len(out) == 1
    state_id = out[0]['genesis_state']
    stored = listing(raw, prefix=Layout(PREFIX).state_prefix(state_id) + 'obj/')
    assert out[0] == {
        'genesis_state': state_id, 'checkpoint': H, 'fp32_master_digest': DIGEST,
        'objects': len(WEIGHTS) + len(MASTER) + 2, 'bytes': sum(size for size, _etag in stored.values()),
    }
    assert len(stored) == out[0]['objects'] and verified(R2Store(raw, BUCKET), state_id)[0] == E.OK

    for bad in (argv + ['--checkpoint', sha(b'unknown')], ['--run-id', 'NOT-A-RUN'] + argv[2:]):
        code, out = run(bad)
        assert code == 1 and len(out) == 1 and set(out[0]) == {'error'}
    after = listing(raw)
    assert all(after[key] == live[key] for key in live)
    assert all(key.startswith(PREFIX) for key in set(after) - set(live))
