"""Generator submission bytes, strict schema and signature/activation checks."""

from __future__ import annotations

import base64
import copy
import hashlib
import json

import pytest
from nacl.signing import SigningKey

from sn120_v2 import links
from sn120_v2.errors import ContractError
from sn120_v2.generator import constants, protocol

ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
EPOCH = "20261120T0000Z"
TASKS = [f"{number:020x}" for number in range(1, 33)]


def ss58(public: bytes, prefix: int = 42) -> str:
    body = bytes([prefix]) + public
    number = int.from_bytes(body + hashlib.blake2b(b"SS58PRE" + body, digest_size=64).digest()[:2], "big")
    text = ""
    while number:
        number, digit = divmod(number, 58)
        text = ALPHABET[digit] + text
    return text


class Miner:
    def __init__(self, label: str = "Generator test miner") -> None:
        self.key = SigningKey(hashlib.sha256(label.encode()).digest())
        self.hotkey = ss58(bytes(self.key.verify_key))

    def sign(self, message: bytes) -> bytes:
        return self.key.sign(message).signature

    def activation(self) -> str:
        return links.activation_text(self.hotkey, self.sign(links.activation_message(self.hotkey, 120)))


A, B = Miner(), Miner("another Generator miner")


def make(**changes):
    submission = protocol.sign_submission(EPOCH, A.hotkey, TASKS[:3], A.sign)
    return submission.to_dict() | changes


def test_fixed_signed_message_and_canonical_hash():
    expected_ids = hashlib.sha256("\n".join(TASKS[:3]).encode()).hexdigest()
    message = f"affine-generator-v1|120|1|{EPOCH}|{A.hotkey}|{constants.POOL_ID}|{expected_ids}".encode()
    submission = protocol.sign_submission(EPOCH, A.hotkey, TASKS[:3], A.sign)
    assert submission.signing_message() == message
    assert A.key.verify_key.verify(message, bytes.fromhex(submission.signature)) == message
    # The order of IDs matters and no trailing newline enters their digest.
    assert protocol.task_ids_digest(TASKS[:3]) != protocol.task_ids_digest(TASKS[:3][::-1])
    assert protocol.task_ids_digest(TASKS[:3]) != hashlib.sha256(("\n".join(TASKS[:3]) + "\n").encode()).hexdigest()
    expected_hash = hashlib.sha256(json.dumps(submission.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert submission.digest() == expected_hash
    assert protocol.canonical_submission_hash(submission) == expected_hash
    assert json.loads(submission.canonical_json()) == submission.to_dict()
    assert len(submission.canonical_json()) < constants.MAX_SUBMISSION_BYTES


def test_signing_is_deterministic_and_round_trips_from_wire_bytes():
    one = protocol.sign_submission(EPOCH, A.hotkey, TASKS[:2], A.sign)
    two = protocol.sign_submission(EPOCH, A.hotkey, TASKS[:2], A.sign)
    assert one == two
    assert protocol.Submission.from_json(one.canonical_json()) == one
    assert one.verify_signature()
    assert protocol.verify_submission(one, registered=True, activation=A.activation(), known_task_ids=lambda _: True)


def test_pool_pin_is_full_release_name_and_raw_manifest_sha():
    assert constants.POOL_ID == f"sn120-math-prior-v1:{constants.POOL_MANIFEST_SHA256}"
    assert constants.POOL_ID in protocol.signing_message(EPOCH, A.hotkey, TASKS[:1]).decode()


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "affine-generator-v2"),
        ("version", True),
        ("netuid", True),
        ("netuid", 120.0),
        ("netuid", "120"),
        ("netuid", 121),
        ("mech_id", True),
        ("mech_id", 1.0),
        ("mech_id", "1"),
        ("mech_id", 0),
        ("epoch", 123),
        ("hotkey", None),
        ("pool", constants.POOL_MANIFEST_SHA256),
        ("pool", "sn120-math-prior-v2:" + constants.POOL_MANIFEST_SHA256),
        ("task_ids", (TASKS[0],)),
        ("task_ids", TASKS[0]),
        ("task_ids", []),
        ("task_ids", TASKS + ["f" * 20]),
        ("task_ids", [TASKS[0], TASKS[0]]),
        ("task_ids", [1]),
        ("task_ids", [True]),
        ("task_ids", ["A" + TASKS[1][1:]]),
        ("task_ids", ["mp:123"]),
        ("task_ids", ["f" * 19]),
        ("task_ids", ["f" * 21]),
        ("task_ids", ["g" * 20]),
        ("signature", "00" * 63),
        ("signature", "AA" * 64),
        ("signature", 7),
    ],
)
def test_invalid_schema_values_are_rejected(field, value):
    with pytest.raises(ContractError):
        protocol.Submission.from_dict(make(**{field: value}))


def test_missing_and_unknown_fields_are_rejected():
    for field in make():
        row = make()
        del row[field]
        with pytest.raises(ContractError, match="required fields"):
            protocol.Submission.from_dict(row)
    with pytest.raises(ContractError, match="required fields"):
        protocol.Submission.from_dict(make(extra="no"))


@pytest.mark.parametrize("raw", [None, [], b"", b"{", b"\xff", b"null", b"[]", b"true", b'"json"', b"x" * (16 * 1024 + 1)])
def test_invalid_json_shape_encoding_and_size_are_rejected(raw):
    with pytest.raises(ContractError):
        protocol.Submission.from_json(raw)


def test_size_limit_is_on_raw_upload_bytes_not_parsed_length():
    raw = make()
    encoded = json.dumps(raw).encode()
    assert len(encoded) > len(json.dumps(raw, separators=(",", ":")).encode())
    parsed = protocol.Submission.from_json(encoded)
    padding = b" " * (16 * 1024 - len(encoded))
    assert len(encoded + padding) == 16 * 1024
    assert protocol.Submission.from_json(encoded + padding) == parsed
    with pytest.raises(ContractError, match="bytes"):
        protocol.Submission.from_json(encoded + padding + b" ")


@pytest.mark.parametrize("epoch", ["2026-11-20T00:00:00Z", "20261120T0100Z", "20261301T0000Z", "20261120T0001Z", "20261120T0000z"])
def test_noncanonical_epoch_is_rejected(epoch):
    with pytest.raises(ContractError):
        protocol.Submission.from_json(json.dumps(make(epoch=epoch)).encode())


def test_bad_hotkey_is_rejected_and_verification_fails_closed():
    with pytest.raises(ContractError):
        protocol.Submission.from_json(json.dumps(make(hotkey="not-an-ss58-address")).encode())
    assert not protocol.verify_submission(make(hotkey="bad"))
    assert not protocol.verify_submission(make(epoch="not-an-epoch"))


@pytest.mark.parametrize("field", ["version", "netuid", "mech_id", "epoch", "hotkey", "pool", "task_ids", "signature"])
def test_every_tampered_signed_field_fails_signature_verification(field):
    row = make()
    values = {
        "version": "affine-generator-v1 ",
        "netuid": 121,
        "mech_id": 2,
        "epoch": "20261120T0400Z",
        "hotkey": B.hotkey,
        "pool": constants.POOL_ID + " ",
        "task_ids": TASKS[1:4],
        "signature": ("0" if row["signature"][0] != "0" else "1") + row["signature"][1:],
    }
    # Preserve the schema while altering domain-separated content where possible.
    row[field] = values[field]
    assert not protocol.verify_submission(row)


def test_reordered_tasks_and_cross_hotkey_replay_do_not_verify():
    original = protocol.sign_submission(EPOCH, A.hotkey, TASKS[:3], A.sign)
    reordered = original.to_dict() | {"task_ids": TASKS[:3][::-1]}
    assert not protocol.verify_submission(reordered)
    forged = original.to_dict() | {"hotkey": B.hotkey}
    assert not protocol.verify_submission(forged)


def test_activation_is_not_silently_assumed_and_wrong_ones_fail():
    row = make()
    assert protocol.verify_submission(row, registered=True)
    assert not protocol.verify_submission(row, registered=False)
    assert not protocol.verify_submission(row, activation=B.activation())
    assert not protocol.verify_submission(row, activation=A.activation().replace("affine2", "affine3"))
    other_subnet = links.activation_text(A.hotkey, A.sign(links.activation_message(A.hotkey, 7)))
    assert not protocol.verify_submission(row, activation=other_subnet)
    with pytest.raises(ContractError, match="activation"):
        protocol.sign_submission(EPOCH, B.hotkey, TASKS[:1], A.sign)


def test_optional_pool_membership_is_checked_without_changing_ids():
    assert protocol.verify_submission(make(), known_task_ids=lambda task_id: task_id in TASKS)
    assert not protocol.verify_submission(make(), known_task_ids=lambda _: False)


def test_signer_contract_and_optional_activation_precheck_for_test_keys():
    with pytest.raises(ContractError, match="64-byte"):
        protocol.sign_submission(EPOCH, A.hotkey, TASKS[:1], lambda _: b"short", require_activation=False)
    submission = protocol.sign_submission(EPOCH, A.hotkey, TASKS[:1], A.sign, require_activation=False)
    assert submission.verify_signature()
    with pytest.raises(ContractError):
        protocol.sign_submission(EPOCH, A.hotkey, [True], A.sign)
    with pytest.raises(ContractError):
        protocol.sign_submission(EPOCH, A.hotkey, TASKS + ["f" * 20], A.sign)
