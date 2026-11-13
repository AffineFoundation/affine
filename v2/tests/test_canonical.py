from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from sn120_v2.canonical import Record, canonical_json, digest, file_sha256, is_sha256, manifest_digest, sha256_hex
from sn120_v2.errors import ContractError

HASH_A, HASH_B = "a" * 64, "b" * 64


def test_canonical_json_has_one_byte_form():
    assert canonical_json({"b": [1, 2], "a": {"d": None, "c": True}}) == b'{"a":{"c":true,"d":null},"b":[1,2]}'
    assert canonical_json({"x": 1, "y": 2}) == canonical_json({"y": 2, "x": 1})
    assert canonical_json({"k": "é"}) == '{"k":"é"}'.encode("utf-8")
    assert digest({"a": 1}) == hashlib.sha256(b'{"a":1}').hexdigest()
    for bad in ({"x": [float("nan")]}, {"x": float("inf")}, {1: "a"}, {"x": {1, 2}}):
        with pytest.raises(ContractError):
            canonical_json(bad)


def test_manifest_digest_is_sha256_of_the_name_to_hash_object(tmp_path):
    expected = hashlib.sha256(f'{{"config.json":"{HASH_B}","model.safetensors":"{HASH_A}"}}'.encode()).hexdigest()
    assert manifest_digest({"model.safetensors": HASH_A, "config.json": HASH_B}) == expected
    assert manifest_digest({"a.bin": HASH_A}) != manifest_digest({"a.bin": HASH_B})
    for bad in ({}, {"a.bin": "not-a-hash"}, {"a.bin": HASH_A.upper()}, {"": HASH_A}, {"modèle.bin": HASH_A}):
        with pytest.raises(ContractError):
            manifest_digest(bad)
    assert is_sha256(HASH_A) and not is_sha256(HASH_A[:-1] + "g") and not is_sha256(None)
    path = tmp_path / "blob.bin"
    path.write_bytes(b"affine" * 1000)
    assert file_sha256(path) == sha256_hex(b"affine" * 1000)


@dataclass(frozen=True)
class Inner(Record):
    name: str
    score: float


@dataclass(frozen=True)
class Outer(Record):
    count: int
    flag: bool
    items: tuple[Inner, ...]
    pair: tuple[str, int]
    table: dict[str, int]
    note: str | None = None


def sample() -> Outer:
    return Outer(count=2, flag=True, items=(Inner("a", 1.5), Inner("b", 2)), pair=("x", 1), table={"k": 3})


def test_record_round_trips_and_its_digest_survives_the_round_trip():
    record = sample()
    again = Outer.from_json(record.to_json())
    assert again == record and Outer.from_dict(record.to_dict()) == record
    assert type(record.to_dict()["items"][1]["score"]) is float, "a whole number in a float field is written as a float"
    assert again.digest() == record.digest() != Outer(3, True, (), ("x", 1), {}).digest()
    assert Inner.from_dict({"name": "a", "score": 2}) == Inner("a", 2.0)


@pytest.mark.parametrize(
    "patch",
    [
        {"count": True}, {"count": 1.0}, {"flag": 1}, {"note": 5}, {"items": [{"name": "a", "score": "high"}]},
        {"items": [{"name": "a", "score": 1.0, "extra": 1}]}, {"pair": ["x", 1, 2]}, {"table": {"k": "v"}}, {"surprise": 1},
    ],
)
def test_record_rejects_a_wrong_or_unknown_field(patch):
    with pytest.raises(ContractError):
        Outer.from_dict(sample().to_dict() | patch)


def test_record_rejects_a_missing_field_and_text_that_is_not_json():
    data = sample().to_dict()
    del data["count"]
    with pytest.raises(ContractError):
        Outer.from_dict(data)
    with pytest.raises(ContractError):
        Outer.from_json(b"{not json")
