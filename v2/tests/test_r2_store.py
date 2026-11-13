"""R2 store against an S3-compatible fake (moto, served over real HTTP on localhost)."""

from __future__ import annotations

import hashlib
import urllib.request
import uuid
from unittest.mock import Mock

import boto3
import pytest
from botocore.exceptions import ClientError, ReadTimeoutError
from moto.server import ThreadedMotoServer

from sn120_v2.errors import ConfigError, ContractError, InfraError
from sn120_v2.r2 import MAX_PRESIGN_S, MIB, R2Store
from sn120_v2.seams import ObjectStore, Presigner

SECRET = "fake-secret-value-never-shown"
BIG = bytes(range(256)) * (12 * MIB // 256) + b"tail"  # three 5 MiB parts
TIMEOUT = ReadTimeoutError(endpoint_url="http://r2")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def endpoint():
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def env(endpoint):  # the environment for one fresh bucket
    env = {"SN120_R2_ENDPOINT": endpoint, "SN120_R2_BUCKET": f"b-{uuid.uuid4().hex}",
           "SN120_R2_ACCESS_KEY_ID": "fake-id", "SN120_R2_SECRET_ACCESS_KEY": SECRET}
    raw = boto3.client("s3", endpoint_url=endpoint, region_name="auto", aws_access_key_id="fake-id", aws_secret_access_key=SECRET)
    raw.create_bucket(Bucket=env["SN120_R2_BUCKET"], CreateBucketConfiguration={"LocationConstraint": "auto"})
    return env


@pytest.fixture
def store(env):
    return R2Store.from_env(env, part_size=5 * MIB, attempts=1)


def raw_put(store, key, data, **extra):
    """Write behind the store's back, as a box with an upload link could."""
    store._s3.put_object(Bucket=store._bucket, Key=store._k(key), Body=data, **extra)


def test_round_trip_and_missing_keys(store, tmp_path):
    assert isinstance(store, ObjectStore) and isinstance(store, Presigner)
    store.put("private/a.json", b"one")
    store.put("private/a.json", b"two!")  # overwrite
    assert store.get("private/a.json") == b"two!"
    assert store.exists("private/a.json") and store.size("private/a.json") == 4
    store.copy("private/a.json", "private/a.json")  # no-op
    store.copy("private/a.json", "public/a.json")
    assert store.sha256("public/a.json") == sha(b"two!")
    store.delete("private/a.json")
    store.delete("private/a.json")  # a missing key is not an error
    assert not store.exists("private/a.json")
    for call in (store.get, store.size, store.sha256, lambda k: store.copy(k, "x"), lambda k: store.get_file(k, tmp_path / "f")):
        with pytest.raises(KeyError):
            call("private/a.json")
    assert not (tmp_path / "f").exists() and not store.exists("x")
    store.put("private/empty", b"")
    store.get_file("private/empty", tmp_path / "empty")
    assert (tmp_path / "empty").read_bytes() == b"" and store.sha256("private/empty") == sha(b"")
    for bad in ("", "/abs"):
        with pytest.raises(ContractError):
            store.put(bad, b"")


def test_sha256_is_the_hash_of_the_stored_bytes_not_a_claim(store):
    """v1 lesson: verify sha256 before promote. A hash that the uploader claims must not count."""
    raw_put(store, "pending/x.bin", b"real bytes", Metadata={"sha256": sha(b"claimed")})
    assert store.sha256("pending/x.bin") == sha(b"real bytes")
    raw_put(store, "pending/x.bin", b"REAL BYTES")  # same size, new content
    assert store.sha256("pending/x.bin") == sha(b"REAL BYTES")


def test_large_file_upload_download_and_multipart_copy(store, tmp_path):
    """v1 lesson: one CopyObject on a ~4 GB shard timed out, so a large copy must use UploadPartCopy."""
    (tmp_path / "shard").write_bytes(BIG)
    store.put_file("pending/folds/v1/base/shard", tmp_path / "shard")
    assert store._head("pending/folds/v1/base/shard")["ETag"].strip('"').endswith("-3")  # a multipart upload
    ranges, real = [], store._s3.upload_part_copy
    store._s3.upload_part_copy = lambda **kw: ranges.append(kw["CopySourceRange"]) or real(**kw)
    store._s3.copy_object = Mock(side_effect=AssertionError("a large copy must not use one CopyObject call"))
    store.copy("pending/folds/v1/base/shard", "public/bases/h/shard")
    assert sorted(ranges) == [f"bytes=0-{5 * MIB - 1}", f"bytes={10 * MIB}-{len(BIG) - 1}", f"bytes={5 * MIB}-{10 * MIB - 1}"]
    assert store.size("public/bases/h/shard") == len(BIG) and store.sha256("public/bases/h/shard") == sha(BIG)
    target = tmp_path / "deep" / "dir" / "out.bin"
    store.get_file("public/bases/h/shard", target)
    assert target.read_bytes() == BIG and [p.name for p in target.parent.iterdir()] == ["out.bin"]


@pytest.mark.parametrize("error", [ClientError({"Error": {"Code": "InternalError"}}, "UploadPartCopy"), TIMEOUT])
def test_failed_copy_promotes_nothing(store, error):
    """v1 lesson: never promote on a timeout. A copy that fails part-way leaves no object at the target."""
    store.put("pending/big", BIG)
    real = store._s3.upload_part_copy
    store._s3.upload_part_copy = lambda **kw: real(**kw) if kw["PartNumber"] == 1 else Mock(side_effect=error)()
    with pytest.raises(InfraError):
        store.copy("pending/big", "public/big")
    assert not store.exists("public/big")
    assert not store._s3.list_multipart_uploads(Bucket=store._bucket).get("Uploads")  # the upload was aborted


def test_partial_or_wrong_write_is_never_a_success(store, tmp_path):
    """A call that the store accepts, but that leaves the wrong bytes, must raise and leave nothing behind."""
    store.put("pending/src", b"0123456789")
    (tmp_path / "f").write_bytes(b"0123456789")
    store._s3.copy_object = store._s3.upload_file = lambda *args, **kw: raw_put(store, "public/dst", b"0123")
    for write in (lambda: store.copy("pending/src", "public/dst"), lambda: store.put_file("public/dst", tmp_path / "f")):
        with pytest.raises(InfraError, match="expected 10"):
            write()
        assert not store.exists("public/dst")
    real_put = store._s3.put_object
    store._s3.put_object = lambda **kw: real_put(**kw) | {"ETag": '"0"'}
    with pytest.raises(InfraError, match="MD5"):
        store.put("public/dst", b"payload")
    assert not store.exists("public/dst")


def test_copy_fails_when_the_source_changes_under_it(store):
    store.put("pending/src", b"version one")
    real_copy = store._s3.copy_object
    store._s3.copy_object = lambda **kw: (real_copy(**kw), raw_put(store, "pending/src", b"version two"))
    with pytest.raises(InfraError, match="source changed"):
        store.copy("pending/src", "public/dst")
    assert not store.exists("public/dst")


def test_failed_download_leaves_no_partial_file(store, tmp_path):
    store.put("public/a", b"abc")
    store._s3.download_file = Mock(side_effect=TIMEOUT)
    with pytest.raises(InfraError):
        store.get_file("public/a", tmp_path / "a")
    assert not list(tmp_path.iterdir())


def test_list_matches_whole_path_segments(store):
    """Lease l1 must not list lease l10: a pending prefix is deleted when its round closes."""
    for key in ("pending/rounds/r1/l10/result.json", "pending/rounds/r1/l1/result.json", "pending/rounds/r1/l1/loras/000/w.bin"):
        store.put(key, b"x")
    assert store.list("pending/rounds/r1/l1") == ["pending/rounds/r1/l1/loras/000/w.bin", "pending/rounds/r1/l1/result.json"]
    assert store.list("pending/rounds/r1/l1/result.json") == ["pending/rounds/r1/l1/result.json"]
    assert len(store.list("pending/rounds/r1/")) == len(store.list("")) == 3
    assert store.list("pending/rounds/r2") == []


def test_prefix_keeps_a_run_inside_its_own_namespace(env):
    """A smoke run under a prefix cannot read, list or delete a key outside that prefix."""
    root = R2Store.from_env(env, attempts=1)
    smoke = R2Store.from_env(env | {"SN120_R2_PREFIX": "/smoke/run1/"}, attempts=1)
    root.put("public/current.json", b"live")
    smoke.put("public/current.json", b"smoke")
    assert smoke.list("") == ["public/current.json"] and smoke.get("public/current.json") == b"smoke"
    assert root.list("") == ["public/current.json", "smoke/run1/public/current.json"]
    smoke.delete("public/current.json")
    assert root.get("public/current.json") == b"live"


def test_presigned_links_work_without_credentials(store):
    """v1 lesson: boxes and miners use presigned links; store credentials never leave the controller."""
    key = "private/ff/v1/miners/hk1/batch.json"
    url, headers = store.presign_put(key, 86400)
    assert headers == {"Content-Type": "application/octet-stream"}
    assert "X-Amz-Expires=86400" in url and "content-type" in url and SECRET not in url
    with urllib.request.urlopen(urllib.request.Request(url, data=b'{"v": 1}', headers=headers, method="PUT")) as reply:
        assert reply.status == 200
    assert store.get(key) == b'{"v": 1}'
    link = store.presign_get(key, 1800)
    assert "X-Amz-Signature=" in link and "X-Amz-Expires=1800" in link and SECRET not in link
    with urllib.request.urlopen(link) as reply:
        assert reply.read() == b'{"v": 1}'
    for ttl in (0, -5, MAX_PRESIGN_S + 1):
        with pytest.raises(ContractError):
            store.presign_get(key, ttl)


def test_bad_configuration_fails_closed_and_hides_the_secret(env):
    """v1 lessons: a missing env var fails closed; an outage is InfraError, never "the file is missing"."""
    for name in env:
        with pytest.raises(ConfigError, match=name):
            R2Store.from_env({k: v for k, v in env.items() if k != name})
    with pytest.raises(InfraError) as missing_bucket:
        R2Store.from_env(env | {"SN120_R2_BUCKET": "no-such-bucket"}, attempts=1)
    with pytest.raises(InfraError) as unreachable:
        R2Store.from_env(env | {"SN120_R2_ENDPOINT": "http://127.0.0.1:1"}, attempts=1)
    store = R2Store.from_env(env, attempts=1)
    store._s3.head_object = store._s3.get_object = Mock(side_effect=TIMEOUT)
    for call in (store.exists, store.size, store.sha256, store.get):
        with pytest.raises(InfraError) as outage:
            call("private/ff/v1/miners/hk1/batch.json")
    for caught in (missing_bucket, unreachable, outage):
        assert SECRET not in str(caught.value) and SECRET not in repr(caught.value.__cause__)
    with pytest.raises(ContractError):
        R2Store.from_env(env, part_size=MIB)
