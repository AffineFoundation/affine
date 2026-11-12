"""R2 layout and strict parsers: every malformed document fails closed."""
from __future__ import annotations

import copy
import json

import pytest

from ops.parallel_eval.durable import canonical
from ops.parallel_eval.pit.batchctl import layout as L

H1, H2, H3, H4 = ('%064x' % n for n in (1, 2, 3, 4))


def manifest_doc(parent=H4, batch_id='b1'):
    return {
        'schema': L.SCHEMA_STATE,
        'parent': parent,
        'batch_id': batch_id,
        'objects': [
            {'name': 'weights.bf16.safetensors', 'role': L.ROLE_BF16_WEIGHTS, 'sha256': H2, 'size': 20},
            {'name': 'master.fp32.safetensors', 'role': L.ROLE_FP32_MASTER, 'sha256': H1, 'size': 10},
        ],
        'meta': {L.META_FP32_MASTER_DIGEST: H3, 'step': 7},
    }


def complete_raw(state=True, **over):
    doc = {
        'schema': L.SCHEMA_COMPLETE,
        'batch_id': 'b1',
        'fence': 5,
        'base_state': H4,
        'result': {'sha256': H1, 'size': 12},
        'state': manifest_doc() if state else None,
    }
    doc.update(over)
    return canonical(doc)


def test_manifest_parses_and_sorts_objects():
    m = L.parse_manifest(manifest_doc(), batch_id='b1', parent=H4)
    assert [o.name for o in m.objects] == ['master.fp32.safetensors', 'weights.bf16.safetensors']
    assert m.fp32_master_digest == H3
    assert L.is_hex64(m.state_id)
    assert json.loads(m.body())['objects'][0]['role'] == L.ROLE_FP32_MASTER


def test_state_id_does_not_depend_on_object_order():
    doc = manifest_doc()
    swapped = copy.deepcopy(doc)
    swapped['objects'].reverse()
    a = L.parse_manifest(doc, batch_id='b1', parent=H4)
    b = L.parse_manifest(swapped, batch_id='b1', parent=H4)
    assert a.state_id == b.state_id


def test_state_id_changes_with_any_field():
    base = L.parse_manifest(manifest_doc(), batch_id='b1', parent=H4).state_id
    other_parent = L.parse_manifest(manifest_doc(parent=H3), batch_id='b1', parent=H3).state_id
    other_batch = L.parse_manifest(manifest_doc(batch_id='b2'), batch_id='b2', parent=H4).state_id
    doc = manifest_doc()
    doc['objects'][0]['sha256'] = H3
    other_bytes = L.parse_manifest(doc, batch_id='b1', parent=H4).state_id
    assert len({base, other_parent, other_batch, other_bytes}) == 4


def _bad_manifests():
    def drop_role(doc, role):
        doc['objects'] = [o for o in doc['objects'] if o['role'] != role]

    cases = {}
    d = manifest_doc(); drop_role(d, L.ROLE_FP32_MASTER); cases['missing_role_fp32_master'] = d
    d = manifest_doc(); drop_role(d, L.ROLE_BF16_WEIGHTS); cases['missing_role_bf16_weights'] = d
    d = manifest_doc(); del d['meta'][L.META_FP32_MASTER_DIGEST]; cases['missing_fp32_master_digest'] = d
    d = manifest_doc(); d['meta'][L.META_FP32_MASTER_DIGEST] = 'ABC'; cases['missing_fp32_master_digest.2'] = d
    d = manifest_doc(); d['objects'][0]['name'] = '../HEAD.json'; cases['object_name'] = d
    d = manifest_doc(); d['objects'][0]['name'] = 'a/b'; cases['object_name.2'] = d
    d = manifest_doc(); d['objects'][1]['name'] = d['objects'][0]['name']; cases['object_name_duplicate'] = d
    d = manifest_doc(); d['objects'][0]['sha256'] = 'AB' * 32; cases['object_sha256'] = d
    d = manifest_doc(); d['objects'][0]['size'] = 0; cases['object_size'] = d
    d = manifest_doc(); d['objects'][0]['size'] = True; cases['object_size.2'] = d
    d = manifest_doc(); d['objects'][0]['role'] = 'other'; cases['object_role'] = d
    d = manifest_doc(); d['objects'][0]['extra'] = 1; cases['object_keys'] = d
    d = manifest_doc(); d['objects'] = []; cases['manifest_objects'] = d
    d = manifest_doc(); d['schema'] = 'x'; cases['manifest_schema'] = d
    d = manifest_doc(); d['extra'] = 1; cases['manifest_keys'] = d
    d = manifest_doc(); d['meta']['Bad-Key'] = 1; cases['meta_key'] = d
    d = manifest_doc(); d['meta']['k'] = [1]; cases['meta_value'] = d
    d = manifest_doc(batch_id='other'); cases['manifest_batch_mismatch'] = d
    d = manifest_doc(parent=H1); cases['manifest_parent_mismatch'] = d
    return sorted(cases.items())


@pytest.mark.parametrize('reason,doc', _bad_manifests())
def test_bad_manifest_fails_closed(reason, doc):
    with pytest.raises(L.PackageInvalid) as err:
        L.parse_manifest(doc, batch_id='b1', parent=H4)
    assert str(err.value) == reason.split('.')[0]


def test_complete_parses_green_and_red():
    green = L.parse_complete(complete_raw(), batch_id='b1', fence=5, base_state=H4)
    assert green.manifest is not None and green.result_sha == H1 and green.result_size == 12
    red = L.parse_complete(complete_raw(state=False), batch_id='b1', fence=5, base_state=H4)
    assert red.manifest is None
    assert green.complete_sha == L.sha256_hex(complete_raw())


@pytest.mark.parametrize('kwargs,reason', [
    (dict(batch_id='b2', fence=5, base_state=H4), 'complete_batch_mismatch'),
    (dict(batch_id='b1', fence=6, base_state=H4), 'complete_fence_mismatch'),
    (dict(batch_id='b1', fence=5, base_state=H3), 'complete_base_mismatch'),
])
def test_complete_is_bound_to_the_lease(kwargs, reason):
    with pytest.raises(L.PackageInvalid) as err:
        L.parse_complete(complete_raw(), **kwargs)
    assert str(err.value) == reason


@pytest.mark.parametrize('raw,reason', [
    (b'not json', 'complete_not_json'),
    (b'[1]', 'complete_not_object'),
    (b'\xff\xfe', 'complete_not_json'),
    (complete_raw(extra=1), 'complete_keys'),
    (complete_raw(schema='x'), 'complete_schema'),
    (complete_raw(fence='5'), 'complete_fence_mismatch'),
    (complete_raw(fence=True), 'complete_fence_mismatch'),
    (complete_raw(result={'sha256': H1}), 'result_keys'),
    (complete_raw(result={'sha256': 'zz', 'size': 3}), 'result_sha256'),
    (complete_raw(result={'sha256': H1, 'size': 0}), 'result_size'),
    (b' ' * (L.MAX_COMPLETE_BYTES + 1), 'complete_too_large'),
])
def test_bad_complete_fails_closed(raw, reason):
    with pytest.raises(L.PackageInvalid) as err:
        L.parse_complete(raw, batch_id='b1', fence=5, base_state=H4)
    assert str(err.value) == reason


def test_complete_doc_roundtrip():
    manifest = L.parse_manifest(manifest_doc(), batch_id='b1', parent=H4)
    doc = L.complete_doc(batch_id='b1', fence=5, base_state=H4, result_sha=H1, result_size=12, manifest=manifest)
    pkg = L.parse_complete(canonical(doc), batch_id='b1', fence=5, base_state=H4)
    assert pkg.manifest.state_id == manifest.state_id


def test_link_and_head_are_pure_functions_of_their_fields():
    fields = dict(seq=3, state_id=H1, parent=H2, batch_id='b1', result_sha=H3, fence=9)
    assert L.link_id(**fields) == L.link_id(**fields) == L.sha256_hex(L.link_body(**fields))
    assert L.link_id(**dict(fields, fence=10)) != L.link_id(**fields)
    body = L.head_body(seq=3, link=H1, state_id=H2)
    assert body == L.head_body(seq=3, link=H1, state_id=H2)
    assert L.parse_head(body) == L.HeadPointer(3, H1, H2)


@pytest.mark.parametrize('raw', [
    b'{}', b'[]', b'{"schema":"x","seq":1,"link":"a","state":"b"}',
    canonical({'schema': L.SCHEMA_HEAD, 'seq': -1, 'link': H1, 'state': H2}),
    canonical({'schema': L.SCHEMA_HEAD, 'seq': True, 'link': H1, 'state': H2}),
    canonical({'schema': L.SCHEMA_HEAD, 'seq': 1, 'link': 'x', 'state': H2}),
    canonical({'schema': L.SCHEMA_HEAD, 'seq': 1, 'link': H1, 'state': H2, 'more': 1}),
])
def test_bad_head_pointer_is_rejected(raw):
    with pytest.raises(ValueError):
        L.parse_head(raw)


def test_layout_keys():
    lay = L.Layout('pit-shadow/batchctl/run1/')
    assert lay.head == 'pit-shadow/batchctl/run1/HEAD.json'
    assert lay.link(H1) == 'pit-shadow/batchctl/run1/links/%s.json' % H1
    assert lay.state_manifest(H1) == 'pit-shadow/batchctl/run1/states/%s/manifest.json' % H1
    assert lay.state_object(H1, 'w.bin') == 'pit-shadow/batchctl/run1/states/%s/obj/w.bin' % H1
    assert lay.result(H2) == 'pit-shadow/batchctl/run1/results/%s.json' % H2
    assert lay.outcome('b1') == 'pit-shadow/batchctl/run1/outcomes/b1.json'
    assert lay.staging_prefix('b1', 7) == 'pit-shadow/batchctl/run1/staging/b1/000000000007/'
    assert lay.staging_complete('b1', 7).endswith('/000000000007/COMPLETE.json')
    assert lay.parse_staging_key(lay.staging_object('b1', 7, 'w.bin')) == ('b1', 7)
    assert lay.parse_staging_key(lay.head) is None
    assert lay.parse_staging_key(lay.staging_root + 'b1/notanumber/x') is None
    assert lay.parse_staging_key(lay.staging_root + 'b1') is None


@pytest.mark.parametrize('fence', ['\u00b2', '\u0663', '1' * 19, '9' * 5000, '-1', '1.0', ' 1', ''])
def test_a_staging_key_from_outside_never_makes_the_parser_fail(fence):
    # ``str.isdigit`` is true for digits that ``int`` refuses, and ``int``
    # refuses a very long number. Neither may reach ``int``.
    lay = L.Layout('pit-shadow/batchctl/run1/')
    assert lay.parse_staging_key('%sb1/%s/x' % (lay.staging_root, fence)) is None


DEEP = b'[' * 200000 + b']' * 200000


def test_json_that_is_nested_too_deep_is_a_plain_invalid_document():
    # The parser of the standard library raises RecursionError here. A box
    # must not be able to raise an error that the controller does not expect.
    with pytest.raises(RecursionError):
        json.loads(DEEP)
    with pytest.raises(ValueError):
        L.load_json(DEEP)
    with pytest.raises(L.PackageInvalid) as err:
        L.parse_complete(DEEP, batch_id='b1', fence=5, base_state=H4)
    assert str(err.value) == 'complete_not_json'
    with pytest.raises(ValueError):
        L.parse_head(DEEP[:L.MAX_HEAD_BYTES])
    with pytest.raises(ValueError):
        L.parse_heartbeat(DEEP, fence=7)
    nested = b'{"a":' * 100000 + b'1' + b'}' * 100000
    with pytest.raises(ValueError):
        L.load_json(nested)


@pytest.mark.parametrize('prefix', ['', '/abs/', 'no-slash', 'a//b/'])
def test_layout_rejects_bad_prefix(prefix):
    with pytest.raises(ValueError):
        L.Layout(prefix)


@pytest.mark.parametrize('value,ok', [
    ('b1', True), ('epoch-12.batch_3', True), ('genesis', False), ('', False),
    ('a/b', False), ('..', False), ('a..b', False), ('-x', False), (5, False), ('x' * 129, False),
])
def test_safe_batch_id(value, ok):
    assert L.is_safe_batch_id(value) is ok


def test_a_trailing_newline_never_passes_a_name_check():
    assert not L.is_safe_batch_id('b1\n')
    assert not L.is_hex64(H1 + '\n')
    assert not L.is_hex64('AB' * 32)
    doc = manifest_doc()
    doc['objects'][0]['name'] += '\n'
    with pytest.raises(L.PackageInvalid):
        L.parse_manifest(doc, batch_id='b1', parent=H4)
    doc = manifest_doc()
    doc['meta']['step\n'] = 1
    with pytest.raises(L.PackageInvalid):
        L.parse_manifest(doc, batch_id='b1', parent=H4)
    doc = manifest_doc()
    doc['objects'][0]['sha256'] += '\n'
    with pytest.raises(L.PackageInvalid):
        L.parse_manifest(doc, batch_id='b1', parent=H4)


def test_heartbeat_roundtrip_and_fence_binding():
    lay = L.Layout('pit-shadow/batchctl/run1/')
    assert lay.staging_heartbeat('b1', 7) == lay.staging_prefix('b1', 7) + 'HEARTBEAT.json'
    assert lay.parse_staging_key(lay.staging_heartbeat('b1', 7)) == ('b1', 7)
    raw = L.heartbeat_body(fence=7, seq=3, stage='train')
    assert L.parse_heartbeat(raw, fence=7) == 3
    assert len(raw) < L.MAX_HEARTBEAT_BYTES
    # The heartbeat of another fence renews nothing.
    with pytest.raises(ValueError):
        L.parse_heartbeat(raw, fence=8)


@pytest.mark.parametrize('raw', [
    b'', b'not json', b'[]', b'7', b'{}',
    canonical({'schema': 'other', 'fence': 7, 'seq': 1}),
    canonical({'schema': L.SCHEMA_HEARTBEAT, 'fence': 7}),
    canonical({'schema': L.SCHEMA_HEARTBEAT, 'fence': 7, 'seq': -1}),
    canonical({'schema': L.SCHEMA_HEARTBEAT, 'fence': 7, 'seq': '1'}),
    canonical({'schema': L.SCHEMA_HEARTBEAT, 'fence': 7, 'seq': True}),
    canonical({'schema': L.SCHEMA_HEARTBEAT, 'fence': '7', 'seq': 1}),
])
def test_bad_heartbeat_is_rejected(raw):
    with pytest.raises(ValueError):
        L.parse_heartbeat(raw, fence=7)


def test_run_binding_names_the_journal():
    lay = L.Layout('pit-shadow/batchctl/run1/')
    assert lay.run == 'pit-shadow/batchctl/run1/RUN.json'
    one = L.run_body(run_id='run1', journal_id=H1)
    assert one == L.run_body(run_id='run1', journal_id=H1)
    assert one != L.run_body(run_id='run1', journal_id=H2)
    assert json.loads(one) == {'schema': L.SCHEMA_RUN, 'run_id': 'run1', 'journal': H1}
