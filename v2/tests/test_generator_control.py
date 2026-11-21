"""Fake-store/chain tests for Generator's deadline snapshot and round robin."""
from __future__ import annotations

import datetime as dt
import hashlib

import pytest
from nacl.signing import SigningKey

from sn120_v2.canonical import sha256_hex
from sn120_v2.generator.epoch import epoch_deadline, epoch_start, submission_object
from sn120_v2.generator.protocol import sign_submission
from sn120_v2.generator.snapshot import build_snapshot, round_robin, seed_for
from sn120_v2.links import public_key
from sn120_v2.store import LocalStore

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
EPOCH = "20261120T0000Z"


def address(key):
    raw = bytes([42]) + bytes(key.verify_key)
    number = int.from_bytes(raw + hashlib.blake2b(b"SS58PRE" + raw, digest_size=64).digest()[:2], "big")
    text = ""
    while number:
        number, digit = divmod(number, 58)
        text = B58[digit] + text
    return text


def hotkey(label):
    key = SigningKey(hashlib.sha256(label.encode()).digest())
    return address(key), key


def tid(number):
    return f"{number:020x}"


def upload(store, epoch, hotkey, key, ids, when, body=None):
    path = submission_object(epoch, hotkey)
    raw = body if body is not None else sign_submission(
        epoch, hotkey, ids, lambda message: key.sign(message).signature, require_activation=False
    ).canonical_json()
    store.put(path, raw)
    store._path(path).touch()
    import os
    os.utime(store._path(path), (when, when))
    return path


def test_snapshot_verifies_every_file_and_earliest_upload_wins(tmp_path):
    store = LocalStore(tmp_path)
    a, ka = hotkey("control-a")
    b, kb = hotkey("control-b")
    owner, ko = hotkey("owner")
    start = epoch_start(EPOCH).timestamp()
    upload(store, EPOCH, a, ka, [tid(1), tid(2)], start + 200)
    upload(store, EPOCH, b, kb, [tid(1), tid(3)], start + 100)
    upload(store, EPOCH, owner, ko, [tid(4)], start + 100)  # uid zero: ineligible
    late = sign_submission(epoch=EPOCH, hotkey=a, task_ids=[tid(5)], sign=lambda m: ka.sign(m).signature, require_activation=False)
    doc = late.to_dict() | {"epoch": "20261120T0400Z"}
    import json
    upload(store, EPOCH, a, ka, [tid(5)], start + 300, json.dumps(doc).encode())
    # The second object of one hotkey is not an allowed layout, so use a bad signature on a separate hotkey.
    c, kc = hotkey("control-c")
    bad = sign_submission(epoch=EPOCH, hotkey=c, task_ids=[tid(6)], sign=lambda m: kc.sign(m).signature, require_activation=False)
    upload(store, EPOCH, c, kc, [tid(6)], start + 100, bad.canonical_json()[:-4] + b"0000")
    manifest, job = build_snapshot(
        store, epoch=EPOCH, registered={0: owner, 1: a, 2: b, 3: c}, activations={a, b, c},
        known_task_ids={tid(n) for n in range(1, 7)}, previously_accepted={tid(2)}, king="head-1", seed_commitment="ab" * 32,
    )
    assert [row["task_id"] for row in job["tasks"]] == [tid(1), tid(3)]
    assert job["tasks"][0]["hotkey"] == b and job["tasks"][0]["uid"] == 2
    assert all("task_ids" not in str(row) for row in manifest["files"])
    assert all("id" not in row for row in manifest["files"])
    assert next(row for row in manifest["files"] if row["hotkey"] == a)["sha256"] == store.sha256(submission_object(EPOCH, a))
    assert not any(row["valid"] for row in manifest["files"] if row["hotkey"] in (owner, c))
    assert job["king"] == "head-1" and job["tasks"][0]["head_version_at_deadline"] == "head-1"


def test_timestamp_tie_breaks_on_file_sha_and_pool_membership(tmp_path):
    store = LocalStore(tmp_path)
    a, ka = hotkey("tie-a")
    b, kb = hotkey("tie-b")
    when = epoch_start(EPOCH).timestamp() + 1
    first = upload(store, EPOCH, a, ka, [tid(1)], when)
    second = upload(store, EPOCH, b, kb, [tid(1), tid(99)], when)
    hashes = {a: store.sha256(first), b: store.sha256(second)}
    winner = min(hashes, key=hashes.get)
    manifest, job = build_snapshot(
        store, epoch=EPOCH, registered={1: a, 2: b}, activations={a, b}, known_task_ids={tid(1)},
        previously_accepted=set(), king="king", seed_commitment="00" * 32,
    )
    assert len(job["tasks"]) == 1 and job["tasks"][0]["hotkey"] == winner
    assert job["tasks"][0]["file_sha256"] == hashes[winner]


def test_round_robin_is_fair_and_stable():
    owners = {
        tid(i): {"hotkey": "a" if i < 4 else "b", "uid": 1, "uploaded_at": float(i), "file_sha256": f"{i:064x}"}
        for i in range(1, 5)
    }
    rows = round_robin(owners)
    assert [row["hotkey"] for row in rows] == ["a", "b", "a", "a"]
    assert [row["task_id"] for row in rows] == [tid(1), tid(4), tid(2), tid(3)]
    assert all(row["hotkey"] == ("a" if int(row["task_id"]) < 4 else "b") for row in rows)
    assert round_robin(owners) == rows


def test_sampling_seed_matches_published_definition():
    secret = b"seed secret"
    expected = int(hashlib.sha256(b"seed secret:" + tid(1).encode() + b":0").hexdigest()[:12], 16)
    assert seed_for(secret, tid(1), 0) == expected


def test_snapshot_rejects_post_deadline_upload(tmp_path):
    store = LocalStore(tmp_path)
    a, key = hotkey("late")
    upload(store, EPOCH, a, key, [tid(1)], epoch_deadline(EPOCH).timestamp())
    manifest, job = build_snapshot(
        store, epoch=EPOCH, registered={1: a}, activations={a}, known_task_ids={tid(1)},
        previously_accepted=set(), king="king", seed_commitment="00" * 32,
    )
    assert not job["tasks"] and manifest["files"][0]["reason"] == "outside_submission_window"
