"""The file index of a model state: one strict format, checked against the
manifest that holds it."""
from __future__ import annotations

import json

import pytest

from . import files as F
from .layout import (
    MAX_OBJECTS, ROLE_AUX, ROLE_BF16_WEIGHTS, ROLE_FP32_MASTER, Layout, ObjectRef, parse_manifest, sha256_hex,
)
from .test_fakes import LockedStore, model_files, model_ids, put_state

LAYOUT = Layout('pit-shadow/batchctl/run0001/')
MASTER = 'c' * 64


def small_state(**more):
    files = model_files('files')
    checkpoint, entry, master = model_ids(files)
    store = LockedStore()
    manifest = put_state(
        store, LAYOUT, files, fp32_master_digest=master, checkpoint_id=checkpoint, fp32_entry=entry,
        chunk=100_000, **more,
    )
    return store, manifest, files


def index_of(store, manifest):
    return store.get(LAYOUT.state_object(manifest.state_id, F.INDEX_NAME), max_bytes=F.MAX_INDEX_BYTES).data


def objects_of(manifest):
    return {ref.name: ref for ref in manifest.objects}


def test_a_state_is_parts_plus_one_index_and_the_id_fixes_every_byte():
    store, manifest, files = small_state()
    raw = index_of(store, manifest)
    index = F.parse_index(raw, objects_of(manifest))
    checkpoint, entry, master = model_ids(files)
    assert (index.checkpoint_id, index.fp32_entry, index.fp32_master_digest) == (checkpoint, entry, master)
    assert manifest.fp32_master_digest == master
    assert [entry.path for entry in index.files] == sorted(files)
    for entry in index.files:
        parts = [store.get(LAYOUT.state_object(manifest.state_id, name), max_bytes=1 << 30).data for name in entry.parts]
        assert b''.join(parts) == files[entry.path]
        assert (entry.size, entry.sha256) == (len(files[entry.path]), sha256_hex(files[entry.path]))
        assert entry.role == F.GROUP_ROLE[entry.path.split('/')[0]]
        assert all(objects_of(manifest)[name].role == entry.role for name in entry.parts)
    # A large file has more than one part, and every object but the index is a part of one file.
    assert len(index.file('bf16/model-00001-of-00003.safetensors').parts) == 4
    used = [name for entry in index.files for name in entry.parts]
    assert sorted(used + [F.INDEX_NAME]) == sorted(objects_of(manifest))
    # The manifest is a valid manifest of the controller, and it names the index by its sha256.
    parsed = parse_manifest(json.loads(manifest.body()), batch_id='genesis', parent=None)
    assert parsed.state_id == manifest.state_id
    assert objects_of(manifest)[F.INDEX_NAME] == ObjectRef(F.INDEX_NAME, ROLE_AUX, sha256_hex(raw), len(raw))
    assert [e.path for e in index.group(F.GROUP_FP32)] == [p for p in sorted(files) if p.startswith('fp32/')]


def test_the_index_bytes_do_not_depend_on_the_order_of_the_files():
    one = F.FileEntry('bf16/a', ROLE_BF16_WEIGHTS, 1, 'a' * 64, ('p-000000',))
    two = F.FileEntry('fp32/b', ROLE_FP32_MASTER, 2, 'b' * 64, ('p-000001',))
    make = lambda files: F.index_body(files, checkpoint_id='c', fp32_entry=None, fp32_master_digest=MASTER)
    assert make([one, two]) == make([two, one])
    assert json.loads(make([one, two]))['fp32_entry'] is None


def test_a_file_of_size_zero_has_no_part():
    store = LockedStore()
    files = {'bf16/w': b'weights', 'fp32/m': b'master', 'aux/empty': b''}
    manifest = put_state(store, LAYOUT, files, fp32_master_digest=MASTER)
    index = F.parse_index(index_of(store, manifest), objects_of(manifest))
    assert index.file('aux/empty') == F.FileEntry('aux/empty', ROLE_AUX, 0, sha256_hex(b''), ())
    assert index.checkpoint_id is None and index.fp32_entry is None


def changed(raw, change):
    doc = json.loads(raw)
    change(doc)
    return json.dumps(doc).encode()


def first(doc):
    return doc['files'][0]


def empty_file(**changes):
    return dict({'path': 'aux/empty', 'role': ROLE_AUX, 'size': 0, 'sha256': sha256_hex(b''), 'parts': []}, **changes)


@pytest.mark.parametrize('change', [
    lambda d: d.update(schema='affine-batchctl-files-v0'),
    lambda d: d.update(extra=1),
    lambda d: d.pop('fp32_entry'),
    lambda d: d.update(fp32_master_digest='c' * 63),
    lambda d: d.update(fp32_master_digest=None),
    lambda d: d.update(checkpoint_id=7),
    lambda d: d.update(checkpoint_id=''),
    lambda d: d.update(fp32_entry='x' * 257),
    lambda d: d.update(files=[]),
    lambda d: d.update(files={}),
    lambda d: d['files'].append(dict(first(d))),
    lambda d: first(d).update(path='../x'),
    lambda d: first(d).update(path='a/b/c'),
    lambda d: first(d).update(path='/abs'),
    lambda d: first(d).update(path='bf16/' + 'x' * 201),
    lambda d: first(d).update(role='weights'),
    lambda d: first(d).update(size=-1),
    lambda d: first(d).update(size=True),
    lambda d: first(d).update(size=first(d)['size'] + 1),
    lambda d: first(d).update(sha256='A' * 64),
    lambda d: first(d).update(parts='p-000000'),
    lambda d: first(d).update(parts=[7]),
    lambda d: first(d).update(parts=first(d)['parts'] + ['p-999999']),
    lambda d: first(d).update(parts=first(d)['parts'] + [F.INDEX_NAME]),
    lambda d: first(d).update(parts=first(d)['parts'] + d['files'][1]['parts']),
    lambda d: first(d).update(more=1),
    lambda d: first(d).pop('sha256'),
    # Two files with the same parts.
    lambda d: d['files'].append(dict(first(d), path='aux/second')),
    # An object of the manifest that no file uses.
    lambda d: d['files'].pop(),
    lambda d: d['files'][-1].update(parts=[], size=0),
    # A file without a part still keeps every rule of a file.
    lambda d: d['files'].extend([empty_file(), empty_file()]),
    lambda d: d['files'].append(empty_file(role='weights')),
    lambda d: d['files'].append(empty_file(size=False)),
    lambda d: d['files'].append(empty_file(parts={})),
    lambda d: d['files'].append(empty_file(parts=None)),
    lambda d: d['files'].append(empty_file(path='aux/a..b')),
    # The part of a weight file is used for a file of another role.
    lambda d: d['files'][-1].update(role=ROLE_AUX),
])
def test_an_index_that_breaks_a_rule_is_refused(change):
    store, manifest, _files = small_state()
    raw = index_of(store, manifest)
    F.parse_index(raw, objects_of(manifest))
    with pytest.raises(F.IndexInvalid):
        F.parse_index(changed(raw, change), objects_of(manifest))


def test_a_file_without_a_part_is_valid_next_to_the_other_files():
    store, manifest, _files = small_state()
    raw = changed(index_of(store, manifest), lambda d: d['files'].append(empty_file()))
    assert F.parse_index(raw, objects_of(manifest)).file('aux/empty').parts == ()


def test_the_index_is_no_part_of_a_file():
    store, manifest, _files = small_state()
    raw = index_of(store, manifest)
    itself = {'path': 'aux/self', 'role': ROLE_AUX, 'size': len(raw), 'sha256': sha256_hex(raw),
              'parts': [F.INDEX_NAME]}
    with pytest.raises(F.IndexInvalid, match='file_part'):
        F.parse_index(changed(raw, lambda d: d['files'].append(itself)), objects_of(manifest))


@pytest.mark.parametrize('raw', [
    b'', b'[]', b'null', b'{', b'{"schema": 1}', b'[' * 100000,
    F.index_body([], checkpoint_id=None, fp32_entry=None, fp32_master_digest=MASTER),
])
def test_bytes_that_are_no_index_are_refused(raw):
    with pytest.raises(F.IndexInvalid):
        F.parse_index(raw, {})


def test_an_index_above_the_size_limit_is_refused_before_it_is_parsed():
    with pytest.raises(F.IndexInvalid, match='index_too_large'):
        F.parse_index(b' ' * (F.MAX_INDEX_BYTES + 1), {})


def test_a_manifest_needs_the_fp32_master_digest_and_fits_the_object_limit():
    part = ObjectRef('p-000000', ROLE_BF16_WEIGHTS, 'a' * 64, 1)
    with pytest.raises(F.IndexInvalid, match='missing_fp32_master_digest'):
        F.manifest_of(parent=None, batch_id='genesis', parts=[part], index=b'{}', meta={})
    many = [ObjectRef(F.slot_name(n), ROLE_BF16_WEIGHTS, 'a' * 64, 1) for n in range(MAX_OBJECTS)]
    with pytest.raises(F.IndexInvalid, match='too_many_objects'):
        F.manifest_of(parent=None, batch_id='genesis', parts=many, index=b'{}', meta={'fp32_master_digest': MASTER})


def test_upload_slots_cover_a_state_of_the_size_of_the_base():
    _store, manifest, files = small_state()
    chunk = 100_000
    slots = F.slots_for(manifest, chunk)
    # The same files, cut again, never need more parts than there are slots.
    need = sum(-(-len(data) // chunk) for data in files.values())
    assert len(slots) >= need + 16 and len(set(slots)) == len(slots)
    assert slots[0] == 'p-000000' and F.INDEX_NAME not in slots
    assert all(F.is_path(name) for name in slots)
    # Never more than a manifest can hold next to the index.
    assert len(F.slots_for(manifest, 1)) == MAX_OBJECTS - 1
    with pytest.raises(ValueError):
        F.slots_for(manifest, 0)


@pytest.mark.parametrize('path,ok', [
    ('bf16/model.safetensors', True), ('meta.json', True), ('fp32/a_b-c.d', True),
    ('', False), ('bf16/', False), ('/x', False), ('a/../b', False), ('a/b/c', False), ('BF16/x', False),
    ('bf16/.hidden', False), ('bf16/x\n', False), ('bf16/x y', False), (None, False), (7, False),
    ('bf16/a..b', False), ('bf16/a.b', True),
])
def test_file_paths(path, ok):
    assert F.is_path(path) is ok
