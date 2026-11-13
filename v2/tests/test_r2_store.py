"""R2 store against an S3-compatible fake (moto, served over real HTTP on localhost)."""

from __future__ import annotations

import hashlib
import re
import urllib.request
import uuid
from unittest.mock import Mock

import boto3
import pytest
from botocore.exceptions import ClientError, ReadTimeoutError
from moto.server import ThreadedMotoServer

from sn120_v2 import layout, links
from sn120_v2.errors import ConfigError, ContractError, InfraError
from sn120_v2.r2 import MAX_PRESIGN_S, MIB, R2Store, main, preflight, self_test, written_roots
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


def record_calls(store):
    """Every S3 call that the store sends from now on, as (operation, key)."""
    calls = []
    store._s3.meta.events.register("before-parameter-build.s3", lambda params, model, **kw: calls.append((model.name, params.get("Key"))))
    return calls


# What v1 has in the bucket that production shares, by its source: none of it is a key that v2 writes.
V1_KEYS = ("public/current.json", "public/checkpoints/c1/model.safetensors", "public/streams/live/current.json", "public/head.json.bak",
           "private/ff/v1/turns/e1/batch.json", "private/ff/v1/miners/5Hotkey/batch.json", "mimo/run/checkpoints/c1/fp32-master/w.bin",
           "pendingx/y", "selftest/keep")


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
    root.put(layout.CURRENT, b"live")
    smoke.put(layout.CURRENT, b"smoke")
    assert smoke.list("") == [layout.CURRENT] and smoke.get(layout.CURRENT) == b"smoke"
    assert root.list("") == [layout.CURRENT, f"smoke/run1/{layout.CURRENT}"]
    smoke.delete(layout.CURRENT)
    assert root.get(layout.CURRENT) == b"live"


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


def test_self_test_checks_every_rule_and_leaves_nothing(env, store, capsys):
    """The first smoke-test step: one line per check, no link or secret printed, nothing left in the bucket.

    With no key prefix (production) it writes and deletes only under ``selftest/<random id>/`` at the bucket root.
    """
    for key in V1_KEYS:
        store.put(key, b"v1")
    assert main(["--self-test"], env) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 10 and all(line.startswith("ok  ") for line in lines[:9]) and lines[9].startswith("PASS")
    assert store.list("") == sorted(V1_KEYS) and SECRET not in "".join(lines) and "X-Amz" not in "".join(lines)
    calls = record_calls(store)
    store._s3.upload_part_copy = Mock(side_effect=ClientError({"Error": {"Code": "SignatureDoesNotMatch"}}, "UploadPartCopy"))
    assert self_test(store) is False
    sent = [(name, key) for name, key in calls if key]  # every call of the self-test that names a key
    failed = [line for line in capsys.readouterr().out.splitlines() if line.startswith("FAIL")]
    assert len(failed) == 3 and "multipart copy" in failed[0] and "SignatureDoesNotMatch" in failed[0]  # copy, download, summary
    assert store.list("") == sorted(V1_KEYS) and all(store.get(key) == b"v1" for key in V1_KEYS)
    own = {key.split("/")[1] for _, key in sent}
    assert len(own) == 1 and re.fullmatch("[0-9a-f]{32}", own.pop()) and {"PutObject", "DeleteObject"} <= {name for name, _ in sent}
    assert all(key.startswith("selftest/") for _, key in sent)
    assert main(["--self-test"], {k: v for k, v in env.items() if k != "SN120_R2_BUCKET"}) == 1
    assert "SN120_R2_BUCKET" in capsys.readouterr().out


def test_preflight_only_lists_and_passes_only_when_no_v2_key_exists(env, store, capsys):
    """The cutover gate for a bucket that v1 also used: read-only, and exit 0 only when no key that v2 writes holds an object."""
    for key in V1_KEYS:
        store.put(key, b"v1")
    calls = record_calls(store)
    assert preflight(store) is True and {name for name, _ in calls} == {"ListObjectsV2"}
    out = capsys.readouterr().out
    assert "private/ff/v1/miners  1 objects: the miner files" in out and out.splitlines()[-1].startswith("PASS") and "EXISTS" not in out
    assert main(["--preflight"], env) == 0 and store.list("") == sorted(V1_KEYS)
    store.put(layout.public_base_file("h", "model.bin"), b"genesis")
    assert main(["--preflight"], env) == 1
    assert main(["--preflight", "--allow", layout.public_base_prefix("h") + "/"], env) == 0
    assert "1 objects, all under --allow" in capsys.readouterr().out
    theirs = [layout.CURRENT, "pending/left/by/anyone", layout.round_file("r000001"), layout.journal_record(0), f"{layout.LINK_FILES}/5Hotkey.json"]
    theirs += [layout.irt_snapshot(f"s{n}") for n in range(12)]
    for key in theirs:
        store.put(key, b"x")
    assert main(["--preflight", "--allow", "public/bases/h"], env) == 1
    lines = capsys.readouterr().out.splitlines()
    assert all(f"          {key}" in lines for key in theirs[:5]) and "          ... and 2 more" in lines and lines[-1].startswith("FAIL")
    assert [line.split()[1] for line in lines if line.startswith("EXISTS")] == ["pending", "private/journal", "public/ff/links", "public/head.json", "public/irt", "public/rounds"]
    for key in theirs + [layout.public_base_file("h", "model.bin")]:
        store.delete(key)
    store._s3.get_paginator = Mock(side_effect=TIMEOUT)  # a prefix that cannot be listed is not known to be empty
    assert preflight(store) is False and "FAIL    pending  r2 list" in capsys.readouterr().out


def test_preflight_asks_for_each_key_and_prefix_by_name_and_reads_every_page(store, capsys):
    """A listing of the bucket's top levels stops at 1000 names. The preflight asks for each root by name and follows the pages."""
    sent = []
    store._s3.meta.events.register("before-parameter-build.s3", lambda params, model, **kw: sent.append((model.name, dict(params))))
    v1 = [f"public/checkpoints/{n:06d}/model.safetensors" for n in range(1001)]  # more of v1 than one page, before every public key of v2
    for key in v1:
        raw_put(store, key, b"")
    del sent[:]
    assert preflight(store) is True
    roots, _ = written_roots()
    assert all(name == "ListObjectsV2" for name, _ in sent) and [params["Prefix"] for _, params in sent] == roots
    assert not any({"Delimiter", "MaxKeys", "StartAfter"} & set(params) for _, params in sent), "no top-level listing and no cut"
    last = layout.frozen_batch("r000001", "zzz")  # the last of 1001 names under one prefix of v2: on the second page
    for key in [layout.frozen_batch("r000001", f"hk{n:04d}") for n in range(1000)] + [last, layout.round_file("r000001")]:
        raw_put(store, key, b"")
    del sent[:]
    capsys.readouterr()
    assert preflight(store, show=2000) is False
    out = capsys.readouterr().out
    assert "EXISTS  private/rounds  1001 objects" in out and f"          {last}" in out and "EXISTS  public/rounds  1 objects" in out
    paged = [params for _, params in sent if params["Prefix"] == "private/rounds"]
    assert len(paged) == 2 and "ContinuationToken" not in paged[0] and paged[1]["ContinuationToken"], "the second page was asked for"
    assert len(sent) == len(roots) + 1 and len(store.list("")) == 2003


def test_preflight_takes_every_key_from_the_layout_module(monkeypatch):
    """A key that is added to ``layout.py`` is checked with no change to the preflight."""
    roots, miners = written_roots()
    assert miners == "private/ff/v1/miners" == layout.miner_object("x").rsplit("/", 2)[0] and miners in roots
    assert {layout.CURRENT, layout.LLMS_TXT, layout.MINING_JSON, layout.BENCHMARKS, layout.JOURNAL, layout.PENDING, "private/rounds", "public/rounds"} <= set(roots)
    assert all(root.split("/")[0] in ("public", "private", "pending") for root in roots)
    for name, value in vars(layout).items():  # every key that a layout function makes is under one listed root
        if callable(value) and getattr(value, "__module__", "") == layout.__name__ and not name.startswith("_"):
            key = value(*["a"] * (value.__code__.co_argcount - (name in ("pending_lora_prefix", "journal_record"))), *[7] * (name in ("pending_lora_prefix", "journal_record")))
            assert any(key == root or key.startswith(root + "/") for root in roots), name
    assert (f"{layout.LINK_FILES}/", layout.LINK_INDEX) == (links.PUBLIC_PREFIX + links.LINKS_DIR, links.PUBLIC_PREFIX + links.INDEX)

    def new_zone(name: str, seq: int) -> str:
        return f"public/new-zone/{name}/{seq:03d}.json"

    def anywhere(name: str) -> str:
        return f"{name}/file.json"

    new_zone.__module__ = anywhere.__module__ = layout.__name__
    monkeypatch.setattr(layout, "new_zone", new_zone, raising=False)
    monkeypatch.setattr(layout, "NEW_POINTER", "public/new-pointer.json", raising=False)
    assert {"public/new-zone", "public/new-pointer.json"} <= set(written_roots()[0])
    monkeypatch.setattr(layout, "anywhere", anywhere, raising=False)
    with pytest.raises(ContractError, match="anywhere"):
        written_roots()
    monkeypatch.delattr(layout, "anywhere")
    monkeypatch.setattr(layout, "WHOLE_ZONE", "public/", raising=False)  # v1 has objects in public/ and private/
    with pytest.raises(ContractError, match="whole zone"):
        written_roots()
