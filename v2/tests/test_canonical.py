from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from sn120_v2.canonical import (
    Record,
    canonical_json,
    digest,
    file_sha256,
    is_sha256,
    manifest_digest,
    sha256_hex,
)
from sn120_v2.errors import ContractError

HASH_A = "a" * 64
HASH_B = "b" * 64


def test_canonical_json_sorts_keys_and_uses_no_spaces():
    assert canonical_json({"b": [1, 2], "a": {"d": None, "c": True}}) == b'{"a":{"c":true,"d":null},"b":[1,2]}'


def test_canonical_json_does_not_depend_on_insertion_order():
    assert canonical_json({"x": 1, "y": 2}) == canonical_json({"y": 2, "x": 1})


def test_canonical_json_keeps_non_ascii_as_utf8():
    assert canonical_json({"k": "é"}) == '{"k":"é"}'.encode("utf-8")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_canonical_json_rejects_values_with_no_json_form(bad):
    with pytest.raises(ContractError):
        canonical_json({"x": [bad]})


def test_canonical_json_rejects_a_key_that_is_not_a_string():
    with pytest.raises(ContractError):
        canonical_json({1: "a"})


def test_canonical_json_rejects_an_unknown_type():
    with pytest.raises(ContractError):
        canonical_json({"x": {1, 2}})


def test_digest_is_sha256_of_the_canonical_bytes():
    assert digest({"a": 1}) == hashlib.sha256(b'{"a":1}').hexdigest()


def test_is_sha256_wants_64_lowercase_hex_digits():
    assert is_sha256(HASH_A)
    assert not is_sha256(HASH_A.upper())
    assert not is_sha256(HASH_A[:-1])
    assert not is_sha256(HASH_A[:-1] + "g")
    assert not is_sha256(None)


def test_manifest_digest_is_sha256_of_the_name_to_hash_object():
    expected = hashlib.sha256(f'{{"config.json":"{HASH_B}","model.safetensors":"{HASH_A}"}}'.encode()).hexdigest()
    assert manifest_digest({"model.safetensors": HASH_A, "config.json": HASH_B}) == expected


def test_manifest_digest_ignores_the_order_of_the_files():
    one = manifest_digest({"a.bin": HASH_A, "b.bin": HASH_B})
    two = manifest_digest({"b.bin": HASH_B, "a.bin": HASH_A})
    assert one == two


def test_manifest_digest_changes_when_a_file_changes():
    assert manifest_digest({"a.bin": HASH_A}) != manifest_digest({"a.bin": HASH_B})


@pytest.mark.parametrize(
    "files",
    [{}, {"a.bin": "not-a-hash"}, {"a.bin": HASH_A.upper()}, {"": HASH_A}, {"modèle.bin": HASH_A}],
)
def test_manifest_digest_fails_closed_on_a_bad_manifest(files):
    with pytest.raises(ContractError):
        manifest_digest(files)


def test_file_sha256_matches_the_hash_of_the_bytes(tmp_path):
    payload = b"affine" * 1000
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert file_sha256(path) == sha256_hex(payload)


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
    return Outer(count=2, flag=True, items=(Inner("a", 1.5), Inner("b", 2.0)), pair=("x", 1), table={"k": 3})


def test_record_round_trips_through_json():
    record = sample()
    assert Outer.from_json(record.to_json()) == record
    assert Outer.from_dict(record.to_dict()) == record


def test_record_digest_is_stable_and_sensitive():
    assert sample().digest() == sample().digest()
    changed = Outer(count=3, flag=True, items=(), pair=("x", 1), table={})
    assert changed.digest() != sample().digest()


def test_record_accepts_an_integer_where_a_float_is_expected():
    assert Inner.from_dict({"name": "a", "score": 2}) == Inner("a", 2.0)


@pytest.mark.parametrize(
    "patch",
    [
        {"count": True},
        {"count": 1.0},
        {"flag": 1},
        {"note": 5},
        {"items": [{"name": "a", "score": "high"}]},
        {"items": [{"name": "a", "score": 1.0, "extra": 1}]},
        {"pair": ["x", 1, 2]},
        {"table": {"k": "v"}},
        {"surprise": 1},
    ],
)
def test_record_rejects_a_wrong_or_unknown_field(patch):
    data = sample().to_dict() | patch
    with pytest.raises(ContractError):
        Outer.from_dict(data)


def test_record_rejects_a_missing_required_field():
    data = sample().to_dict()
    del data["count"]
    with pytest.raises(ContractError):
        Outer.from_dict(data)


def test_record_rejects_text_that_is_not_json():
    with pytest.raises(ContractError):
        Outer.from_json(b"{not json")
