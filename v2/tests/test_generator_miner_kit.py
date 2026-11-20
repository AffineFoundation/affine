"""Generator miner kit: real signing, strict offline checks and secret-safe links."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from nacl.public import SealedBox
from nacl.signing import SigningKey

from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.errors import ConfigError, ContractError
from sn120_v2.generator import epoch, pool, protocol

_SPEC = importlib.util.spec_from_file_location("generator_miner_tool", Path(__file__).resolve().parents[1] / "miner" / "generator_miner.py")
kit = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(kit)

EPOCH = "20261120T0000Z"
START = int(epoch.epoch_start(EPOCH).timestamp())
DEADLINE = int(epoch.epoch_deadline(EPOCH).timestamp())
TASKS = [f"{n:020x}" for n in range(1, 33)]
ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def ss58(public: bytes, prefix: int = 42) -> str:
    body = bytes([prefix]) + public
    number = int.from_bytes(body + hashlib.blake2b(b"SS58PRE" + body, digest_size=64).digest()[:2], "big")
    text = ""
    while number:
        number, digit = divmod(number, 58)
        text = ALPHABET[digit] + text
    return text


class Miner:
    def __init__(self, label: str = "miner-kit-test") -> None:
        self.seed = hashlib.sha256(label.encode()).digest()
        self.key = SigningKey(self.seed)
        self.hotkey = ss58(bytes(self.key.verify_key))

    def sign(self, message: bytes) -> bytes:
        return self.key.sign(message).signature


A = Miner()


def key_args(**changes) -> argparse.Namespace:
    values = dict(hotkey=A.hotkey, key_file=None, seed=A.seed.hex(), wallet=None, hotkey_name=None, wallet_path=Path("."), password=None)
    return argparse.Namespace(**(values | changes))


def sealed_link(*, epoch_name: str = EPOCH, hotkey: str = A.hotkey, expiry: int | None = None, url: str = "https://upload.test/private/generator/sub/20261120T0000Z/object.json?X-Amz-Signature=TOPSECRET"):
    expiry = int(epoch.epoch_deadline(epoch_name).timestamp()) if expiry is None else expiry
    link = {
        "version": kit.LINK_VERSION,
        "hotkey": hotkey,
        "epoch": epoch_name,
        "object_key": f"private/generator/sub/{epoch_name}/{hotkey}.json",
        "put_url": url,
        "put_headers": {"Content-Type": "application/octet-stream", "x-amz-meta-test": "yes"},
        "expires_at": expiry,
        "issued_at": START,
        "not_before": int(epoch.epoch_start(epoch_name).timestamp()),
        "deadline": int(epoch.epoch_deadline(epoch_name).timestamp()),
        "upload_mode": "overwrite_until_deadline",
        "first_upload_final": False,
        "overwrite_until_deadline": True,
    }
    box = SealedBox(A.key.verify_key.to_curve25519_public_key())
    return {
        "version": kit.LINK_VERSION,
        "hotkey": hotkey,
        "epoch": epoch_name,
        "deadline": int(epoch.epoch_deadline(epoch_name).timestamp()),
        "upload_mode": link["upload_mode"],
        "sealed_to": bytes(A.key.verify_key).hex(),
        "current": base64.b64encode(box.encrypt(canonical_json(link))).decode(),
        "expires_at": expiry,
    }, link


def test_kit_and_validator_agree_on_message_signature_and_canonical_batch():
    ids = TASKS[:4]
    document = kit.batch_document(EPOCH, A.hotkey, ids, A.key)
    validator = protocol.sign_submission(EPOCH, A.hotkey, ids, A.sign, require_activation=False)
    assert document == validator.to_dict()
    assert canonical_json(document) == validator.canonical_json()
    assert document["signature"] == A.key.sign(protocol.signing_message(EPOCH, A.hotkey, ids)).signature.hex()
    assert protocol.verify_submission(document, known_task_ids=lambda task_id: task_id in TASKS)
    assert kit.check_document(canonical_json(document)).digest() == validator.digest()
    assert kit.protocol is protocol


def test_key_file_matches_wallet_and_bad_keys_fail(tmp_path):
    path = tmp_path / "hotkey.json"
    path.write_text(json.dumps({"secretSeed": "0x" + A.seed.hex(), "ss58Address": A.hotkey}))
    address, key = kit.signing_key(key_args(seed=None, key_file=path))
    assert address == A.hotkey and bytes(key.verify_key) == bytes(A.key.verify_key)
    with pytest.raises(ConfigError, match="does not match"):
        kit.signing_key(key_args(hotkey=Miner("other").hotkey))
    path.write_text(json.dumps({"secretSeed": "00" * 31}))
    with pytest.raises(ConfigError, match="32-byte"):
        kit.read_seed_file(path)
    with pytest.raises(ConfigError, match="32 bytes"):
        kit.signing_key(key_args(seed="abcd"))


def test_encrypted_wallet_is_unlocked_through_sdk_not_silent_seed(monkeypatch, tmp_path):
    class FakeKey:
        ss58_address = A.hotkey
        private_key = "0x" + A.seed.hex()
        is_encrypted = True
        decrypted = False

        def decrypt(self, password):
            assert password == "test-password"
            self.decrypted = True

    fake = FakeKey()
    sdk = SimpleNamespace(Wallet=lambda **kwargs: SimpleNamespace(hotkey=fake))
    monkeypatch.setitem(__import__("sys").modules, "bittensor", sdk)
    address, key = kit.load_wallet_key(tmp_path, "w", "h", "test-password")
    assert fake.decrypted and address == A.hotkey and bytes(key.verify_key) == bytes(A.key.verify_key)


def test_open_link_returns_secret_but_prints_only_public_timing(capsys):
    document, link = sealed_link()
    opened = kit.open_sealed_link(document, A.hotkey, A.key, START + 10)
    assert opened["put_url"] == link["put_url"]
    kit.print_link(opened)
    printed = capsys.readouterr().out
    assert EPOCH in printed and "2026-11-19T20:00:00Z" in printed and "2026-11-20T00:00:00Z" in printed
    assert "TOPSECRET" not in printed and "https://upload.test" not in printed


def test_show_url_is_the_only_explicit_url_output(capsys):
    _, link = sealed_link()
    kit.print_link(link, show_url=True)
    assert "TOPSECRET" in capsys.readouterr().out


def test_link_epoch_must_match_expiry_and_current_window():
    document, _ = sealed_link(epoch_name=EPOCH)
    with pytest.raises(ConfigError, match="closed or not open"):
        kit.open_sealed_link(document, A.hotkey, A.key, DEADLINE)
    with pytest.raises(ConfigError, match="less than 60"):
        kit.open_sealed_link(document, A.hotkey, A.key, DEADLINE - 59)
    document, _ = sealed_link(expiry=DEADLINE + 1)
    with pytest.raises(ConfigError, match="expiry exceeds deadline"):
        kit.open_sealed_link(document, A.hotkey, A.key, START + 10)


def test_next_epoch_box_is_selected_and_validated():
    document, link = sealed_link()
    following = epoch.epoch_id(DEADLINE)
    next_doc, next_link = sealed_link(epoch_name=following)
    document["next"] = next_doc["current"]
    document["next_epoch"] = following
    opened = kit.open_sealed_link(document, A.hotkey, A.key, DEADLINE + 10, field="next")
    assert opened["epoch"] == following
    assert kit._linked_epoch(opened) == following
    with pytest.raises(ConfigError, match="closed or not open"):
        kit.open_sealed_link(document, A.hotkey, A.key, DEADLINE + 10, field="current")


def test_tampered_link_or_wrong_recipient_fails_closed():
    document, _ = sealed_link()
    other = Miner("other")
    with pytest.raises(ConfigError, match="not sealed"):
        kit.open_sealed_link(document, other.hotkey, other.key, START + 10)
    document["current"] = document["current"][:-4] + "AAAA"
    with pytest.raises(ContractError, match="cannot open"):
        kit.open_sealed_link(document, A.hotkey, A.key, START + 10)
    document, _ = sealed_link(url="http://upload.test/private/generator/sub/20261120T0000Z/a.json")
    with pytest.raises(ConfigError, match="HTTPS"):
        kit.open_sealed_link(document, A.hotkey, A.key, START + 10)


def test_upload_uses_exact_signed_headers_and_never_logs_url(monkeypatch, capsys):
    _, link = sealed_link()
    seen = []

    class Reply:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            return b""

    def call(request, timeout):
        seen.append(request)
        return Reply()

    monkeypatch.setattr(kit.urllib.request, "urlopen", call)
    kit.upload(link, b"signed-batch", replace=True)
    request = seen[0]
    assert request.full_url == link["put_url"]
    assert request.data == b"signed-batch"
    assert request.get_method() == "PUT"
    assert request.get_header("Content-type") == "application/octet-stream"
    assert request.get_header("X-amz-meta-test") == "yes"
    assert "TOPSECRET" not in capsys.readouterr().out


def test_upload_http_error_does_not_expose_presigned_url(monkeypatch, capsys):
    _, link = sealed_link()

    class Error(kit.urllib.error.HTTPError):
        def __str__(self):
            return link["put_url"]

    def fail(*args, **kwargs):
        raise Error(link["put_url"], 403, "forbidden", {}, io.BytesIO(b"private"))

    monkeypatch.setattr(kit.urllib.request, "urlopen", fail)
    with pytest.raises(Exception, match="HTTP 403") as caught:
        kit.upload(link, b"x", replace=True)
    assert "TOPSECRET" not in str(caught.value)


def test_task_ids_file_and_jsonl_accept_original_ids_only(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("# selected\n" + "\n".join(TASKS[:2]) + "\n")
    assert kit.read_task_ids(path) == TASKS[:2]
    path.write_text("\n".join(canonical_json({"id": task_id, "local_pass_rate": 0.5}).decode() for task_id in TASKS[:2]))
    assert kit.read_task_ids(path) == TASKS[:2]
    for ids in ([], TASKS + ["f" * 20], [TASKS[0], TASKS[0]]):
        path.write_text("\n".join(ids))
        with pytest.raises(ConfigError):
            kit.read_task_ids(path)


def test_pool_command_downloads_hashes_and_builds_validator_index(tmp_path, monkeypatch, capsys):
    manifest = pool.PoolManifest(pool.POOL_ID, pool.POOL_MANIFEST_URL, pool.POOL_MANIFEST_SHA256, tuple(pool.PoolShard(f"prior-{n:02d}-of-08.parquet", 1, "0" * 64) for n in range(8)), rows=3)
    calls = []

    class Registry:
        _locations = {task_id: object() for task_id in TASKS}

        def __init__(self, root, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(kit.pool, "load_manifest", lambda location: manifest)
    monkeypatch.setattr(kit.pool, "prepare_shards", lambda root, document, fetch: tuple(tmp_path / f"shard-{n}" for n in range(8)) if fetch else None)
    monkeypatch.setattr(kit.pool, "PriorPool", Registry)
    args = argparse.Namespace(cache=tmp_path, manifest_url=pool.POOL_MANIFEST_URL, index=True)
    assert kit.cmd_pool(args) == 0
    assert calls == [{"manifest": manifest, "build_index": True}]
    output = capsys.readouterr().out
    assert pool.POOL_MANIFEST_SHA256 in output
    assert "8 shards" in output and "3 rows" in output


def test_offline_check_rejects_tampering_and_unknown_pool_id():
    document = kit.batch_document(EPOCH, A.hotkey, TASKS[:2], A.key)
    known = {task_id: object() for task_id in TASKS[:2]}
    registry = SimpleNamespace(__contains__=known.__contains__)
    assert kit.check_document(canonical_json(document), registry).task_ids == tuple(TASKS[:2])
    tampered = dict(document)
    tampered["task_ids"] = list(reversed(tampered["task_ids"]))
    with pytest.raises(ConfigError, match="invalid"):
        kit.check_document(canonical_json(tampered), registry)
    unknown = kit.batch_document(EPOCH, A.hotkey, ["f" * 20], A.key)
    with pytest.raises(ConfigError, match="invalid"):
        kit.check_document(canonical_json(unknown), registry)


def test_status_filters_public_results_for_hotkey(tmp_path, capsys):
    state = tmp_path / "state.json"
    result = tmp_path / "results.json"
    state.write_text(json.dumps({"epoch": EPOCH, "seed_commitment": "abc", "private_url": "https://secret.example/?X-Amz-Signature=NO"}))
    result.write_text(json.dumps({"epoch": EPOCH, "hotkeys": {A.hotkey: {"submitted": 2, "accepted": 1}}}))
    args = argparse.Namespace(
        hotkey=A.hotkey,
        epochs=[EPOCH],
        accepted=False,
        state_url=str(state),
        results_url=str(tmp_path / "{epoch}.json"),
        accepted_url="unused",
    )
    # Use the actual epoch filename in the test's local URL template.
    result.rename(tmp_path / f"{EPOCH}.json")
    assert kit.cmd_status(args) == 0
    printed = capsys.readouterr().out
    assert '"accepted": 1' in printed and "private_url" not in printed
    assert kit.status_for_hotkey({"results": [{"hotkey": A.hotkey, "accepted": 1}]}, A.hotkey)["accepted"] == 1


def test_sealed_and_public_cap_refuse_over_cap_before_upload(tmp_path):
    _, link = sealed_link()
    link["max_tasks"] = 1
    raw = canonical_json(kit.batch_document(EPOCH, A.hotkey, TASKS[:2], A.key))
    with pytest.raises(ConfigError, match=f"submission has 2 task ids; the per-hotkey cap for {EPOCH} is 1"):
        kit.upload(link, raw, replace=True, cap=link["max_tasks"], epoch_name=EPOCH)
    del link["max_tasks"]
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"current_epoch": EPOCH, "submission_cap": {"epoch": EPOCH, "cap": 1}}))
    args = SimpleNamespace(state_url=str(state))
    assert kit.public_cap(args, EPOCH) == 1
    with pytest.raises(ConfigError, match=f"submission has 2 task ids; the per-hotkey cap for {EPOCH} is 1"):
        kit.upload(link, raw, replace=True, cap=kit.public_cap(args, EPOCH), epoch_name=EPOCH)


def test_guide_and_llms_section_name_protocol_and_files():
    guide = (Path(__file__).parents[1] / "site/static/generator-guide.md").read_text()
    section = (Path(__file__).parents[1] / "sn120_v2/generator/LLMS_SECTION.md").read_text()
    for document in (guide, section):
        assert "mechanism 1" in document and "10%" in document and "811,609" in document
        assert "one batch per hotkey per epoch" in document
        assert "first upload wins" in document.lower() and "once ever" in document.lower()
        assert "Beta(1,1)" in document and "q = P(0.25" in document
        assert "00, 04, 08, 12, 16, 20" in document
        assert "python -m sn120_v2.links activate" in document
        assert "accepted-all.jsonl" in document
        assert "America/" not in document and not re.search(r"UTC[+-][0-9]", document)
