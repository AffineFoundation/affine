"""Tests of the R2 adapter.

Three kinds:

* against moto (a fake S3 in this process): the store contract of
  ``storecheck``, multipart upload and multipart copy;
* against scripted answers: the mapping of every answer of R2 to the error
  rule of the interfaces, and the resumed read of ``digest``;
* against real R2, only when ``BATCHCTL_R2_CONFIG`` names a bucket document.
  They write under a sandbox prefix ``pit-shadow/batchctl/<run_id>/`` and nowhere
  else. moto cannot prove what real R2 does with a condition, so these runs
  are the proof that counts.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import boto3
import pytest
import urllib3
from botocore.exceptions import (
    ClientError, ConnectionClosedError, EndpointConnectionError, ReadTimeoutError,
)
from moto import mock_aws

from ..shadow import ShadowWriteError
from . import guard as G
from . import r2 as R
from . import storecheck
from .interfaces import Digest, NotFound, PreconditionFailed, StoreError, TooLarge
from .r2 import MIB, R2Config, R2Presigner, R2Store, ReadOnlyStore

BUCKET = 'batchctl-test'
OTHER = 'batchctl-test-other'
RUN = 'r2test01'
SMALL = R2Config(
    put_part_bytes=5 * MIB, put_multipart_above=5 * MIB,
    copy_part_bytes=5 * MIB, copy_multipart_above=5 * MIB, workers=4,
)
DEFINITE = (NotFound, PreconditionFailed, TooLarge)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def same_bytes(digest: Digest, size: int, sha256: str) -> bool:
    return (digest.size, digest.sha256) == (size, sha256)


def pattern(size: int, seed: bytes = b'r2') -> bytes:
    block = hashlib.sha256(seed).digest()
    return (block * (size // len(block) + 1))[:size]


class Spy:
    """Forwards to a client. Records each call. A hook runs before a call
    and may raise. ``put_object`` runs under a lock: moto does not promise
    that a conditional put is atomic, and real R2 does."""

    def __init__(self, client: Any):
        self._client = client
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.before: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        self._lock = threading.Lock()
        self._serial = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        inner = getattr(self._client, name)

        def call(**kwargs: Any) -> Any:
            with self._lock:
                self.calls.append((name, kwargs))
            hook = self.before.get(name)
            if hook is not None:
                hook(kwargs)
            if name == 'put_object':
                with self._serial:
                    return inner(**kwargs)
            return inner(**kwargs)

        return call

    def named(self, name: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [kwargs for called, kwargs in self.calls if called == name]


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
    return R2Store(spy, BUCKET, SMALL)


def uploads(raw: Any, bucket: str = BUCKET) -> List[Any]:
    return raw.list_multipart_uploads(Bucket=bucket).get('Uploads') or []


def client_error(code: str, status: int, operation: str = 'Op') -> ClientError:
    return ClientError(
        {'Error': {'Code': code, 'Message': 'm'}, 'ResponseMetadata': {'HTTPStatusCode': status}},
        operation,
    )


def no_answer() -> List[BaseException]:
    return [
        EndpointConnectionError(endpoint_url='https://r2.invalid'),
        ReadTimeoutError(endpoint_url='https://r2.invalid'),
        ConnectionClosedError(endpoint_url='https://r2.invalid'),
        ConnectionResetError('reset'),
        TimeoutError('timed out'),
        urllib3.exceptions.ProtocolError('broken'),
    ]


# ----------------------------------------------------------------------------
# The store contract
# ----------------------------------------------------------------------------

def test_the_adapter_keeps_the_store_contract(store):
    done = storecheck.check_store(store, 'contract/', big_bytes=12 * MIB + 123)
    assert 'big_object' in done and 'one_winner' in done and 'copy_by_etag' in done
    assert store.list('contract/') == []


def test_the_adapter_keeps_the_store_contract_behind_the_shadow_guard(store, raw):
    guarded = G.GuardedStore(store, RUN)
    prefix = G.shadow_prefix(RUN) + 'contract/'
    raw.put_object(Bucket=BUCKET, Key='live/HEAD.json', Body=b'live')
    storecheck.check_store(guarded, prefix, many=30, big_bytes=6 * MIB)
    # Nothing outside the prefix of the run was written or removed.
    assert [item['Key'] for item in raw.list_objects_v2(Bucket=BUCKET)['Contents']] == ['live/HEAD.json']
    with pytest.raises(ShadowWriteError):
        guarded.get('live/HEAD.json', max_bytes=10)
    with pytest.raises(ShadowWriteError):
        guarded.put('live/HEAD.json', b'x')


# ----------------------------------------------------------------------------
# Multipart upload
# ----------------------------------------------------------------------------

def test_a_large_put_is_a_multipart_upload_with_equal_parts(store, spy, raw):
    data = pattern(12 * MIB + 77)
    etag = store.put('big', data)
    sizes = sorted(len(kwargs['Body']) for kwargs in spy.named('upload_part'))
    assert sizes == [2 * MIB + 77, 5 * MIB, 5 * MIB]
    assert not spy.named('put_object')
    assert etag.strip('"').endswith('-3')
    assert store.digest('big') == Digest(len(data), sha(data), etag)
    assert uploads(raw) == []


def test_a_put_at_the_limit_is_one_request(store, spy):
    store.put('edge', pattern(5 * MIB))
    assert len(spy.named('put_object')) == 1 and not spy.named('create_multipart_upload')


@pytest.mark.parametrize('condition', ['create', 'swap'])
def test_a_conditional_put_is_always_one_request(store, spy, condition):
    data = pattern(6 * MIB)
    if condition == 'create':
        store.put('doc', data, if_none_match=True)
    else:
        first = store.put('doc', b'old', if_none_match=True)
        store.put('doc', data, if_match=first)
    assert not spy.named('create_multipart_upload')
    assert store.digest('doc').sha256 == sha(data)


def test_a_part_that_fails_is_sent_again(store, spy):
    failed = []

    def hook(kwargs):
        if kwargs['PartNumber'] == 2 and len(failed) < 2:
            failed.append(1)
            raise EndpointConnectionError(endpoint_url='https://r2.invalid')

    spy.before['upload_part'] = hook
    data = pattern(11 * MIB)
    store.put('big', data)
    assert len(failed) == 2
    assert store.digest('big').sha256 == sha(data)


def test_a_multipart_upload_that_cannot_finish_is_aborted_and_writes_nothing(store, spy, raw):
    def hook(kwargs):
        if kwargs['PartNumber'] == 2:
            raise ReadTimeoutError(endpoint_url='https://r2.invalid')

    spy.before['upload_part'] = hook
    with pytest.raises(StoreError) as caught:
        store.put('big', pattern(16 * MIB))
    assert not isinstance(caught.value, DEFINITE)
    assert store.head('big') is None
    assert uploads(raw) == []
    assert len([k for k in spy.named('upload_part') if k['PartNumber'] == 2]) == SMALL.attempts


def test_a_failed_multipart_upload_keeps_the_old_object(store, spy):
    store.put('big', b'old')
    spy.before['complete_multipart_upload'] = lambda kwargs: (_ for _ in ()).throw(
        ConnectionClosedError(endpoint_url='https://r2.invalid'))
    with pytest.raises(StoreError):
        store.put('big', pattern(6 * MIB))
    assert store.get('big', max_bytes=10).data == b'old'


def test_put_file_uploads_a_large_file_in_parts(store, spy, tmp_path):
    data = pattern(11 * MIB + 5)
    path = tmp_path / 'shard.safetensors'
    path.write_bytes(data)
    etag = store.put_file('shard', path)
    assert len(spy.named('upload_part')) == 3
    assert store.digest('shard') == Digest(len(data), sha(data), etag)


def test_put_file_uploads_a_small_file_in_one_request(store, spy, tmp_path):
    path = tmp_path / 'small.json'
    path.write_bytes(b'{}')
    store.put_file('small', path)
    assert len(spy.named('put_object')) == 1
    assert store.get('small', max_bytes=2).data == b'{}'


def test_put_file_refuses_a_file_that_shrinks_during_the_upload(store, spy, raw, tmp_path):
    path = tmp_path / 'shard'
    path.write_bytes(pattern(11 * MIB))

    def hook(kwargs):
        path.write_bytes(b'short')

    spy.before['create_multipart_upload'] = hook
    with pytest.raises(StoreError):
        store.put_file('shard', path)
    assert store.head('shard') is None and uploads(raw) == []


# ----------------------------------------------------------------------------
# Copy
# ----------------------------------------------------------------------------

def test_a_small_copy_is_one_request_tied_to_the_etag(store, spy):
    etag = store.put('src', b'bytes')
    store.copy('src', 'dst', src_etag=etag)
    [call] = spy.named('copy_object')
    assert call['CopySourceIfMatch'] == etag
    assert call['CopySource'] == {'Bucket': BUCKET, 'Key': 'src'}
    assert store.get('dst', max_bytes=5).data == b'bytes'


def test_a_large_copy_is_a_multipart_copy_of_byte_ranges(store, spy, raw):
    data = pattern(12 * MIB + 9)
    etag = store.put('src', data)
    store.copy('src', 'dst', src_etag=etag)
    parts = sorted(spy.named('upload_part_copy'), key=lambda kwargs: kwargs['PartNumber'])
    assert [kwargs['CopySourceRange'] for kwargs in parts] == [
        'bytes=0-%d' % (5 * MIB - 1),
        'bytes=%d-%d' % (5 * MIB, 10 * MIB - 1),
        'bytes=%d-%d' % (10 * MIB, 12 * MIB + 8),
    ]
    assert all(kwargs['CopySourceIfMatch'] == etag for kwargs in parts)
    assert not spy.named('copy_object')
    assert same_bytes(store.digest('dst'), len(data), sha(data))
    assert uploads(raw) == []


def test_a_source_that_changes_during_a_large_copy_is_refused(store, spy, raw):
    etag = store.put('src', pattern(12 * MIB))
    changed = []

    def hook(kwargs):
        if kwargs['PartNumber'] == 2 and not changed:
            changed.append(1)
            raw.put_object(Bucket=BUCKET, Key='src', Body=pattern(12 * MIB, b'other'))

    spy.before['upload_part_copy'] = hook
    with pytest.raises(PreconditionFailed):
        store.copy('src', 'dst', src_etag=etag)
    # No object made of two versions, and no upload left behind.
    assert store.head('dst') is None
    assert uploads(raw) == []


def test_a_large_copy_whose_part_is_refused_writes_nothing(store, spy, raw):
    etag = store.put('src', pattern(12 * MIB))
    spy.before['upload_part_copy'] = lambda kwargs: (_ for _ in ()).throw(
        client_error('PreconditionFailed', 412, 'UploadPartCopy'))
    with pytest.raises(PreconditionFailed):
        store.copy('src', 'dst', src_etag=etag)
    assert store.head('dst') is None and uploads(raw) == []
    # A refusal is definite: it is not tried again.
    assert len(spy.named('upload_part_copy')) <= SMALL.workers


def test_a_large_copy_that_breaks_is_an_unknown_outcome_and_leaves_no_upload(store, spy, raw):
    etag = store.put('src', pattern(12 * MIB))
    spy.before['upload_part_copy'] = lambda kwargs: (_ for _ in ()).throw(
        client_error('InternalError', 500, 'UploadPartCopy'))
    with pytest.raises(StoreError) as caught:
        store.copy('src', 'dst', src_etag=etag)
    assert not isinstance(caught.value, DEFINITE)
    assert store.head('dst') is None and uploads(raw) == []


@pytest.mark.parametrize('size', [9, 11 * MIB])
def test_copy_reads_the_source_from_another_bucket(spy, raw, size):
    data = pattern(size)
    source = R2Store(spy, OTHER, SMALL)
    etag = source.put('live/shard', data)
    target = R2Store(spy, BUCKET, SMALL)
    target.copy('live/shard', 'shadow/shard', src_etag=etag, src_bucket=OTHER)
    assert same_bytes(target.digest('shadow/shard'), size, sha(data))
    # The source bucket got no new key and no upload.
    assert [info.key for info in source.list('')] == ['live/shard']
    assert uploads(raw, OTHER) == []
    with pytest.raises(PreconditionFailed):
        target.copy('live/shard', 'shadow/other', src_etag='"stale"', src_bucket=OTHER)
    with pytest.raises(NotFound):
        target.copy('live/none', 'shadow/other', src_etag=etag, src_bucket=OTHER)


# ----------------------------------------------------------------------------
# Every answer of R2, and no answer at all
# ----------------------------------------------------------------------------

OPS: Dict[str, Tuple[str, Callable[[R2Store], Any]]] = {
    'head': ('head_object', lambda s: s.head('k')),
    'get': ('get_object', lambda s: s.get('k', max_bytes=10)),
    'digest': ('get_object', lambda s: s.digest('k')),
    'list': ('list_objects_v2', lambda s: s.list('k')),
    'put': ('put_object', lambda s: s.put('k', b'x')),
    'create': ('put_object', lambda s: s.put('k', b'x', if_none_match=True)),
    'swap': ('put_object', lambda s: s.put('k', b'x', if_match='"e"')),
    'copy-look': ('head_object', lambda s: s.copy('k', 'd', src_etag='"e"')),
    'delete': ('delete_object', lambda s: s.delete('k')),
    'bucket': ('head_bucket', lambda s: s.check_bucket()),
}


class Raising:
    """A client where one call raises, and every other call is a bug."""

    def __init__(self, name: str, error: BaseException):
        self._name = name
        self._error = error

    def __getattr__(self, name: str) -> Any:
        if name != self._name:
            raise AssertionError('unexpected call %s' % name)

        def call(**kwargs: Any) -> Any:
            raise self._error

        return call


@pytest.mark.parametrize('op', sorted(OPS))
@pytest.mark.parametrize('error', no_answer(), ids=lambda error: type(error).__name__)
def test_a_request_without_an_answer_is_never_a_definite_error(op, error):
    name, call = OPS[op]
    with pytest.raises(StoreError) as caught:
        call(R2Store(Raising(name, error), BUCKET))
    assert not isinstance(caught.value, DEFINITE)


ANSWERS = [
    # op, code, status, what the adapter must do
    ('head', '404', 404, None),
    ('head', 'NotFound', 404, None),
    ('head', '403', 403, StoreError),
    ('head', '500', 500, StoreError),
    ('get', 'NoSuchKey', 404, NotFound),
    ('get', 'NoSuchBucket', 404, StoreError),
    ('get', 'AccessDenied', 403, StoreError),
    ('get', 'SlowDown', 503, StoreError),
    ('digest', 'NoSuchKey', 404, NotFound),
    ('digest', 'InternalError', 500, StoreError),
    ('list', 'NoSuchBucket', 404, StoreError),
    ('put', 'PreconditionFailed', 412, StoreError),
    ('put', 'InternalError', 500, StoreError),
    ('create', 'PreconditionFailed', 412, PreconditionFailed),
    ('create', 'ConditionalRequestConflict', 409, StoreError),
    ('create', 'NoSuchBucket', 404, StoreError),
    ('create', 'InternalError', 500, StoreError),
    ('swap', 'PreconditionFailed', 412, PreconditionFailed),
    ('swap', 'NoSuchKey', 404, PreconditionFailed),
    ('swap', 'NoSuchBucket', 404, StoreError),
    ('swap', 'ConditionalRequestConflict', 409, StoreError),
    ('swap', 'ServiceUnavailable', 503, StoreError),
    ('copy-look', '403', 403, StoreError),
    ('delete', 'NoSuchKey', 404, None),
    ('delete', 'AccessDenied', 403, StoreError),
    ('bucket', '404', 404, StoreError),
    ('bucket', '403', 403, StoreError),
]


@pytest.mark.parametrize('op,code,status,want', ANSWERS)
def test_each_answer_of_r2_maps_to_the_error_rule(op, code, status, want):
    name, call = OPS[op]
    target = R2Store(Raising(name, client_error(code, status)), BUCKET)
    if want is None:
        assert call(target) is None
        return
    with pytest.raises(want) as caught:
        call(target)
    if want is StoreError:
        assert not isinstance(caught.value, DEFINITE)


def test_a_copy_answer_maps_to_the_error_rule(store, spy):
    etag = store.put('src', b'x')
    for code, status, want in (
        ('PreconditionFailed', 412, PreconditionFailed), ('NoSuchKey', 404, NotFound),
        ('NoSuchBucket', 404, StoreError), ('InternalError', 500, StoreError),
    ):
        spy.before['copy_object'] = lambda kwargs, code=code, status=status: (_ for _ in ()).throw(
            client_error(code, status, 'CopyObject'))
        with pytest.raises(want) as caught:
            store.copy('src', 'dst', src_etag=etag)
        assert type(caught.value) is want
    spy.before['copy_object'] = lambda kwargs: (_ for _ in ()).throw(ConnectionResetError())
    with pytest.raises(StoreError) as caught:
        store.copy('src', 'dst', src_etag=etag)
    assert type(caught.value) is StoreError


def test_an_error_message_holds_no_secret_and_no_url():
    error = EndpointConnectionError(endpoint_url='https://account-id.r2.invalid')
    with pytest.raises(StoreError) as caught:
        R2Store(Raising('put_object', error), BUCKET).put('k', b'x')
    assert 'account-id' not in str(caught.value)


# ----------------------------------------------------------------------------
# Reads
# ----------------------------------------------------------------------------

class Body:
    """A response body. It gives ``data`` and then ends, or raises ``error``."""

    def __init__(self, data: bytes, error: Optional[BaseException] = None):
        self._data = data
        self._error = error
        self.closed = False
        self.reads = 0

    def read(self, amount: Optional[int] = None) -> bytes:
        self.reads += 1
        if not self._data:
            if self._error is not None:
                raise self._error
            return b''
        cut = len(self._data) if amount is None else amount
        chunk, self._data = self._data[:cut], self._data[cut:]
        return chunk

    def close(self) -> None:
        self.closed = True


class Scripted:
    """A client that gives one prepared answer for each ``get_object``."""

    def __init__(self, answers: List[Any]):
        self.answers = list(answers)
        self.calls: List[Dict[str, Any]] = []

    def get_object(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def whole(data: bytes, size: int, body: Body, etag: str = '"v1"') -> Dict[str, Any]:
    return {'ETag': etag, 'ContentLength': size, 'Body': body, 'ResponseMetadata': {'HTTPStatusCode': 200}}


def rest(total: int, offset: int, body: Body, status: int = 206, etag: str = '"v1"') -> Dict[str, Any]:
    return {
        'ETag': etag, 'ContentLength': total - offset, 'Body': body,
        'ContentRange': 'bytes %d-%d/%d' % (offset, total - 1, total),
        'ResponseMetadata': {'HTTPStatusCode': status},
    }


def test_get_does_not_read_an_object_above_the_limit():
    body = Body(b'x' * 100)
    client = Scripted([whole(b'', 100, body)])
    with pytest.raises(TooLarge):
        R2Store(client, BUCKET).get('k', max_bytes=99)
    assert body.reads == 0 and body.closed


@pytest.mark.parametrize('error', [None, ConnectionResetError('reset'), urllib3.exceptions.ProtocolError('x')])
def test_get_of_a_body_that_ends_early_is_an_unknown_outcome(error):
    body = Body(b'only-half', error)
    client = Scripted([whole(b'', 100, body)])
    with pytest.raises(StoreError) as caught:
        R2Store(client, BUCKET).get('k', max_bytes=100)
    assert not isinstance(caught.value, DEFINITE) and body.closed


@pytest.mark.parametrize('error', [None, ConnectionResetError('reset'), ReadTimeoutError(endpoint_url='u')])
def test_digest_goes_on_after_a_broken_connection(error):
    data = pattern(1000)
    first, second, third = Body(data[:300], error), Body(data[300:640], error), Body(data[640:])
    client = Scripted([whole(data, 1000, first), rest(1000, 300, second), rest(1000, 640, third)])
    target = R2Store(client, BUCKET, R2Config(read_chunk_bytes=64))
    assert target.digest('k') == Digest(1000, sha(data), '"v1"')
    assert [call.get('Range') for call in client.calls] == [None, 'bytes=300-', 'bytes=640-']
    # Every further read is tied to the version of the first read.
    assert [call.get('IfMatch') for call in client.calls] == [None, '"v1"', '"v1"']
    assert first.closed and second.closed and third.closed


@pytest.mark.parametrize('answer', [client_error('PreconditionFailed', 412), client_error('NoSuchKey', 404)])
def test_digest_never_hashes_a_mix_of_two_versions(answer):
    data = pattern(1000)
    client = Scripted([whole(data, 1000, Body(data[:300])), answer])
    with pytest.raises(StoreError) as caught:
        R2Store(client, BUCKET).digest('k')
    assert not isinstance(caught.value, DEFINITE)


def test_digest_refuses_a_store_that_ignores_the_byte_range():
    data = pattern(1000)
    again = Body(data)
    client = Scripted([whole(data, 1000, Body(data[:300])), rest(1000, 0, again, status=200)])
    with pytest.raises(StoreError):
        R2Store(client, BUCKET).digest('k')
    assert again.closed and again.reads == 0


def test_digest_refuses_an_answer_for_another_range():
    data = pattern(1000)
    client = Scripted([whole(data, 1000, Body(data[:300])), rest(1000, 200, Body(data[200:]))])
    with pytest.raises(StoreError):
        R2Store(client, BUCKET).digest('k')


def test_digest_refuses_more_bytes_than_the_object_has():
    data = pattern(1000)
    client = Scripted([whole(data, 900, Body(data))])
    with pytest.raises(StoreError):
        R2Store(client, BUCKET).digest('k')


def test_digest_gives_up_after_the_configured_number_of_tries():
    data = pattern(1000)
    answers = [whole(data, 1000, Body(data[:100]))] + [
        rest(1000, 100 * n, Body(data[100 * n:100 * n + 100])) for n in range(1, 9)
    ]
    client = Scripted(answers)
    with pytest.raises(StoreError):
        R2Store(client, BUCKET, R2Config(attempts=3)).digest('k')
    assert len(client.calls) == 3


def test_digest_of_a_real_object_is_the_hash_of_its_bytes(store):
    data = pattern(3 * MIB + 1)
    etag = store.put('k', data)
    assert store.digest('k') == Digest(len(data), sha(data), etag)
    etag = store.put('empty', b'')
    assert store.digest('empty') == Digest(0, sha(b''), etag)


class Pages:
    def __init__(self, pages: List[Dict[str, Any]]):
        self.pages = list(pages)
        self.tokens: List[Optional[str]] = []

    def list_objects_v2(self, **kwargs: Any) -> Dict[str, Any]:
        self.tokens.append(kwargs.get('ContinuationToken'))
        return self.pages.pop(0)


def test_list_follows_every_page():
    item = lambda key: {'Key': key, 'Size': 1, 'ETag': '"e"'}  # noqa: E731
    client = Pages([
        {'Contents': [item('a'), item('b')], 'IsTruncated': True, 'NextContinuationToken': 't1'},
        {'IsTruncated': True, 'NextContinuationToken': 't2'},
        {'Contents': [item('c')], 'IsTruncated': False},
    ])
    assert [info.key for info in R2Store(client, BUCKET).list('')] == ['a', 'b', 'c']
    assert client.tokens == [None, 't1', 't2']


def test_list_refuses_a_cut_page_without_a_token():
    client = Pages([{'Contents': [], 'IsTruncated': True}])
    with pytest.raises(StoreError):
        R2Store(client, BUCKET).list('')


def test_the_read_only_store_has_no_call_that_writes(store):
    store.put('live/doc', b'data')
    reader = ReadOnlyStore(store)
    assert reader.get('live/doc', max_bytes=4).data == b'data'
    assert reader.head('live/doc').size == 4
    assert reader.digest('live/doc').sha256 == sha(b'data')
    public = {name for name in dir(reader) if not name.startswith('_')}
    assert public == {'bucket', 'digest', 'get', 'head'}


# ----------------------------------------------------------------------------
# Settings and the client
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('change', [
    {'put_part_bytes': 5 * MIB - 1}, {'copy_part_bytes': 4096 * MIB + 1},
    {'put_multipart_above': 0}, {'copy_multipart_above': 4096 * MIB + 1},
    {'workers': 0}, {'attempts': 0}, {'read_chunk_bytes': 0},
])
def test_settings_outside_the_limits_of_r2_are_refused(change):
    with pytest.raises(ValueError):
        R2Config(**change)


def test_parts_are_equal_and_never_more_than_ten_thousand():
    assert R._plan(12 * MIB, 5 * MIB) == (5 * MIB, 3)
    assert R._plan(10 * MIB, 5 * MIB) == (5 * MIB, 2)
    assert R._plan(1, 5 * MIB) == (5 * MIB, 1)
    part, count = R._plan(200 * 1024 * MIB, 5 * MIB)
    assert count <= R.MAX_PARTS and part % MIB == 0 and part * count >= 200 * 1024 * MIB
    with pytest.raises(ValueError):
        R._plan(R.MAX_PARTS * R.SINGLE_MAX_BYTES + 1, 5 * MIB)


def test_the_client_never_repeats_a_request_by_itself():
    listener = socket.socket()
    listener.bind(('localhost', 0))
    listener.listen(16)
    listener.settimeout(0.2)
    seen: List[int] = []
    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            try:
                connection, _address = listener.accept()
            except OSError:
                continue
            seen.append(1)
            connection.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        client = R.make_client(
            endpoint='http://localhost:%d' % listener.getsockname()[1],
            access_key_id='test', secret_access_key='test',
            connect_timeout_s=2, read_timeout_s=2,
        )
        with pytest.raises(StoreError) as caught:
            R2Store(client, BUCKET).put('k', b'x', if_none_match=True)
        assert not isinstance(caught.value, DEFINITE)
        time.sleep(0.3)
    finally:
        stop.set()
        thread.join()
        listener.close()
    assert len(seen) == 1
    assert client.meta.config.retries['total_max_attempts'] == 1


def test_the_bucket_document_is_read_in_each_shape(tmp_path):
    secret = tmp_path / 'r2.env'
    secret.write_text('# comment\nR2_ACCESS_KEY_ID = "id-1"\nR2_SECRET_ACCESS_KEY=\'key-1\'\nOTHER=x\n')
    inner = {'bucket': 'name', 'endpoint': 'https://r2.invalid', 'credentials_file': str(secret)}
    want = {
        'bucket': 'name', 'endpoint': 'https://r2.invalid',
        'access_key_id': 'id-1', 'secret_access_key': 'key-1',
    }
    for number, shape in enumerate((inner, {'bucket': inner}, {'payload': {'bucket': inner}})):
        path = tmp_path / ('doc%d.json' % number)
        path.write_text(json.dumps(shape))
        assert R.read_bucket_config(path) == want
        client, bucket = R.client_from_bucket_config(path)
        assert bucket == 'name' and client.meta.endpoint_url == 'https://r2.invalid'


# ----------------------------------------------------------------------------
# Presigned URLs
# ----------------------------------------------------------------------------

def test_a_presigned_url_names_one_key_and_holds_no_secret():
    client = R.make_client(
        endpoint='https://account.r2.invalid', access_key_id='AKIDEXAMPLE', secret_access_key='very-secret-key',
    )
    signer = R2Presigner(client, BUCKET)
    get_url = signer.get_url('shadow/a/b.bin', 600)
    put_url = signer.put_url('shadow/a/c.bin', 900)
    assert get_url.startswith('https://account.r2.invalid/%s/shadow/a/b.bin?' % BUCKET)
    assert put_url.startswith('https://account.r2.invalid/%s/shadow/a/c.bin?' % BUCKET)
    for url, seconds in ((get_url, 600), (put_url, 900)):
        assert 'X-Amz-Signature=' in url and 'X-Amz-Expires=%d' % seconds in url
        assert 'very-secret-key' not in url
        assert 'x-amz-checksum' not in url.lower() and 'x-amz-sdk-checksum' not in url.lower()
    assert 'content-type' in put_url.lower()


@pytest.mark.parametrize('seconds', [0, -5, 0.5, 7 * 24 * 3600 + 1])
def test_a_presigned_url_has_a_bounded_life(seconds):
    client = R.make_client(endpoint='https://account.r2.invalid', access_key_id='a', secret_access_key='b')
    with pytest.raises(ValueError):
        R2Presigner(client, BUCKET).get_url('k', seconds)


# ----------------------------------------------------------------------------
# Real R2, in a sandbox prefix. Skipped unless the operator names a bucket.
# ----------------------------------------------------------------------------

REAL_CONFIG = os.environ.get('BATCHCTL_R2_CONFIG')


class Sandbox(NamedTuple):
    inner: R2Store
    guarded: G.GuardedStore
    signer: R2Presigner
    root: str

    def key(self, name: str) -> str:
        return G.check_key(self.root, self.root + 'contract/' + name)
real = pytest.mark.skipif(not REAL_CONFIG, reason='real R2; set BATCHCTL_R2_CONFIG to a bucket document')


@pytest.fixture
def sandbox():
    """A fresh run id, and a store that can reach the prefix of that run only."""
    client, bucket = R.client_from_bucket_config(Path(REAL_CONFIG))
    run_id = os.environ.get('BATCHCTL_R2_SANDBOX_RUN') or 'it' + uuid.uuid4().hex[:10]
    part = int(os.environ.get('BATCHCTL_R2_PART_MIB', '16')) * MIB
    inner = R2Store(client, bucket, R2Config(
        put_part_bytes=part, put_multipart_above=part, copy_part_bytes=part, copy_multipart_above=part,
    ))
    inner.check_bucket()
    box = Sandbox(inner, G.GuardedStore(inner, run_id), R2Presigner(client, bucket), G.shadow_prefix(run_id))
    try:
        yield box
    finally:
        for info in list(box.guarded.list(box.root)):
            box.guarded.delete(info.key)


@real
def test_real_r2_keeps_the_store_contract(sandbox):
    big = 2 * sandbox.inner.cfg.put_part_bytes + 12345
    done = storecheck.check_store(sandbox.guarded, sandbox.root + 'contract/', big_bytes=big)
    print('real R2 store contract:', ', '.join(done))


@real
def test_real_r2_takes_a_shard_of_the_configured_size(sandbox, tmp_path):
    """Multipart upload, read-back and multipart copy of one large file.
    ``BATCHCTL_R2_SHARD_MIB=4608`` makes it a shard above 4 GiB."""
    size = int(os.environ.get('BATCHCTL_R2_SHARD_MIB', '48')) * MIB + 4097
    path = tmp_path / 'shard.bin'
    digest = hashlib.sha256()
    block = pattern(MIB, b'shard')
    with path.open('wb') as handle:
        left = size
        while left:
            chunk = block[:min(left, len(block))]
            handle.write(chunk)
            digest.update(chunk)
            left -= len(chunk)
    src, dst, never = sandbox.key('shard/src'), sandbox.key('shard/dst'), sandbox.key('shard/never')
    started = time.time()
    etag = sandbox.inner.put_file(src, path)
    assert sandbox.guarded.digest(src) == Digest(size, digest.hexdigest(), etag)
    sandbox.guarded.copy(src, dst, src_etag=etag)
    assert same_bytes(sandbox.guarded.digest(dst), size, digest.hexdigest())
    with pytest.raises(PreconditionFailed):
        sandbox.guarded.copy(src, never, src_etag='"stale"')
    assert sandbox.guarded.head(never) is None
    print('real R2 shard: %d bytes up, twice down, one multipart copy, %.0f s' % (size, time.time() - started))


@real
def test_real_r2_takes_a_presigned_upload_and_download(sandbox):
    key, other = sandbox.key('presigned/object'), sandbox.key('presigned/other')
    signer = sandbox.signer
    data = pattern(3 * MIB)
    request = urllib.request.Request(
        signer.put_url(key, 300), data=data, method='PUT', headers={'Content-Type': R.CONTENT_TYPE},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        assert response.status == 200
    assert same_bytes(sandbox.guarded.digest(key), len(data), sha(data))
    with urllib.request.urlopen(signer.get_url(key, 300), timeout=120) as response:
        assert response.read() == data
    # A URL is for one key and one method.
    moved = signer.put_url(key, 300).replace('presigned/object', 'presigned/other')
    for url in (moved, signer.get_url(key, 300)):
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(urllib.request.Request(
                url, data=b'x', method='PUT', headers={'Content-Type': R.CONTENT_TYPE}), timeout=60)
        assert caught.value.code == 403
    assert sandbox.guarded.head(other) is None
    assert same_bytes(sandbox.guarded.digest(key), len(data), sha(data))
    # A URL ends.
    short = signer.get_url(key, 1)
    time.sleep(3)
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(short, timeout=60)
    assert caught.value.code in (400, 403)
