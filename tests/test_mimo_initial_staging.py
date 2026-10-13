import copy
import hashlib
from types import SimpleNamespace

import pytest

from ops.mimo_stage_initial_checkpoint import stage
from subnet.mimo_release import MODEL_ID, MODEL_REVISION
from subnet.mimo_grpo.data import digest
from test_mimo_grpo_publication import S3


def fixture(tmp_path):
    files = {'config.json': b'{}', 'model.safetensors': b'tiny-real-bytes-fixture'}
    for name, raw in files.items(): (tmp_path/name).write_bytes(raw)
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
    return dict(model=MODEL_ID, revision=MODEL_REVISION, directory=str(tmp_path),
        bytes=sum(len(x) for x in files.values()), descriptor=dict(id=digest(hashes), files=hashes))


def test_initial_upload_retry_is_immutable_and_not_a_training_receipt(tmp_path):
    fetched = fixture(tmp_path); client = S3(); bucket = SimpleNamespace(client=client, name='fixture')
    options = dict(namespace='private/initial-test', expected_checkpoint=fetched['descriptor']['id'])
    first = stage(bucket, fetched, **options); uploads = list(client.uploads)
    assert stage(bucket, fetched, **options) == first
    assert client.uploads == uploads
    assert first['qualified'] is False and first['training_update'] is False
    assert first['discovery_changed'] is False
    assert not any('CURRENT' in key or 'optimizer' in key for key in client.objects)
    key = first['objects']['model.safetensors']['key']; original = client.objects[key]
    client.objects[key] = b'X' * len(original)
    with pytest.raises(ValueError, match='GET hash differs'): stage(bucket, fetched, **options)
    assert client.objects[key] == b'X' * len(original)


def test_wrong_revision_inventory_and_public_destination_rejected(tmp_path):
    fetched = fixture(tmp_path); client = S3(); bucket = SimpleNamespace(client=client, name='fixture')
    options = dict(namespace='private/initial-test', expected_checkpoint=fetched['descriptor']['id'])
    bad = copy.deepcopy(fetched); bad['revision'] = '0'*40
    with pytest.raises(ValueError, match='revision'): stage(bucket, bad, **options)
    with pytest.raises(ValueError, match='private namespace'):
        stage(bucket, fetched, **dict(options, namespace='public/current'))
    (tmp_path/'optimizer.pt').write_bytes(b'do not upload')
    with pytest.raises(ValueError, match='inventory'): stage(bucket, fetched, **options)
    assert not client.objects
