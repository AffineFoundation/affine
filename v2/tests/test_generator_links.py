"""Sealed, activation-gated and epoch-bound Generator upload links."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from nacl.public import SealedBox
from nacl.signing import SigningKey

from sn120_v2 import links as core_links
from sn120_v2.errors import ConfigError, ContractError, InfraError
from sn120_v2.generator import epoch
from sn120_v2.generator import links
from sn120_v2.types import ChainSnapshot, Registration

UTC = dt.timezone.utc
EPOCH = "20261120T0000Z"
DEADLINE = epoch.epoch_deadline(EPOCH)
NOW = int(DEADLINE.timestamp()) - 3600
NEXT = epoch.epoch_id(DEADLINE)
URL_SECRET = "presigned-url-must-not-escape"


def ss58(public: bytes) -> str:
    body = b"\x2a" + public
    number = int.from_bytes(body + hashlib.blake2b(b"SS58PRE" + body, digest_size=64).digest()[:2], "big")
    address = ""
    while number:
        number, digit = divmod(number, 58)
        address = core_links._B58[digit] + address
    return address


class Miner:
    def __init__(self, label: str) -> None:
        self.key = SigningKey(hashlib.sha256(label.encode()).digest())
        self.hotkey = ss58(bytes(self.key.verify_key))

    def activation(self) -> str:
        return core_links.activation_text(
            self.hotkey, self.key.sign(core_links.activation_message(self.hotkey, 120)).signature
        )


class FakePresigner:
    def __init__(self, *, conditional: bool = False, binding: bool = True) -> None:
        self.conditional = conditional
        self.binding = binding
        self.calls: list[tuple[str, str, int]] = []
        self.conditional_calls: list[tuple[str, int]] = []
        self.checks = 0

    def first_upload_supported(self) -> bool:
        self.checks += 1
        return self.conditional

    def presign_first_upload_put(self, key: str, ttl_s: int) -> tuple[str, dict[str, str]]:
        self.conditional_calls.append((key, ttl_s))
        signed = "host;content-type"
        if self.binding:
            signed += ";if-none-match"
        return f"https://invalid.test/{key}?{URL_SECRET}&X-Amz-SignedHeaders={signed}", dict(links.FIRST_UPLOAD_HEADERS)

    def presign_put(self, key: str, ttl_s: int) -> tuple[str, dict[str, str]]:
        self.calls.append(("put", key, ttl_s))
        return f"https://invalid.test/{key}?{URL_SECRET}", {"Content-Type": "application/octet-stream"}

    def presign_get(self, key: str, ttl_s: int) -> str:
        self.calls.append(("get", key, ttl_s))
        return f"https://invalid.test/{key}?{URL_SECRET}"


A, B = Miner("Generator link A"), Miner("Generator link B")


def open_sealed(miner: Miner, document: dict, field: str = "current") -> dict:
    box = SealedBox(miner.key.to_curve25519_private_key())
    return json.loads(box.decrypt(base64.b64decode(document[field])))


def one_file(miner: Miner = A, now=NOW, **kwargs) -> tuple[dict, dict, dict, dict]:
    files = links.link_files([miner.hotkey], FakePresigner(), now, **kwargs)
    document = json.loads(files[f"generator/links/{miner.hotkey}.json"])
    return files, document, open_sealed(miner, document), json.loads(files["generator/links.json"])


def test_current_link_is_sealed_to_the_exact_epoch_object_and_deadline_cap():
    _, document, link, _ = one_file(ttl_s=7 * 86400)
    assert set(document) == {
        "version", "hotkey", "issued_at", "epoch", "deadline", "expires_at", "sealed_to", "upload_mode", "current"
    }
    assert document["sealed_to"] == bytes(A.key.verify_key).hex()
    assert link["object_key"] == epoch.submission_object(EPOCH, A.hotkey)
    assert link["object_key"] == f"private/generator/sub/{EPOCH}/{A.hotkey}.json"
    assert link["epoch"] == document["epoch"] == EPOCH
    assert link["deadline"] == document["deadline"] == int(DEADLINE.timestamp())
    assert link["expires_at"] == document["expires_at"] == link["deadline"]
    assert link["not_before"] == int(epoch.epoch_start(EPOCH).timestamp())
    assert link["upload_mode"] == links.OVERWRITE_UNTIL_DEADLINE
    assert link["overwrite_until_deadline"] is True and link["first_upload_final"] is False
    assert link["put_headers"] == {"Content-Type": "application/octet-stream"}
    # Independent verification is possible, but no presigned material is public.
    assert open_sealed(A, document) == link


@pytest.mark.parametrize("remaining,next_present", [(601, False), (600, True), (599, True), (300, True)])
def test_prefetched_next_link_is_separately_sealed_only_inside_ten_minutes(remaining, next_present):
    files = links.link_files([A.hotkey], FakePresigner(), int(DEADLINE.timestamp()) - remaining, ttl_s=7200)
    document = json.loads(files[f"generator/links/{A.hotkey}.json"])
    assert ("next" in document) is next_present
    assert ("next_epoch" in document) is next_present
    if not next_present:
        return
    current, following = open_sealed(A, document), open_sealed(A, document, "next")
    assert following["kind"] == "next" and following["epoch"] == NEXT
    assert following["object_key"] == f"private/generator/sub/{NEXT}/{A.hotkey}.json"
    assert following["object_key"] != current["object_key"]
    assert following["not_before"] == int(DEADLINE.timestamp())
    assert following["deadline"] == int(DEADLINE.timestamp()) + 4 * 3600
    assert following["expires_at"] == min(following["deadline"], document["issued_at"] + 7200)


def test_requested_ttl_is_capped_at_each_deadline():
    fake = FakePresigner()
    files = links.link_files([A.hotkey], fake, int(DEADLINE.timestamp()) - 120, ttl_s=3600)
    document = json.loads(files[f"generator/links/{A.hotkey}.json"])
    current, following = open_sealed(A, document), open_sealed(A, document, "next")
    assert current["expires_at"] - current["issued_at"] == 120
    assert following["expires_at"] - following["issued_at"] == 3600
    assert ("put", current["object_key"], 120) in fake.calls
    assert ("put", following["object_key"], 3600) in fake.calls


def test_utc_helpers_normalize_timezone_and_validate_time_and_ttl():
    local = DEADLINE.astimezone(dt.timezone(dt.timedelta(hours=-6))) - dt.timedelta(hours=1)
    files, document, _, _ = one_file(now=local, ttl_s=7200)
    assert document["epoch"] == EPOCH and document["issued_at"] == NOW
    assert list(files) == sorted(files, key=lambda name: name == "generator/links.json")
    with pytest.raises(ValueError, match="timezone"):
        links.link_files([], FakePresigner(), dt.datetime(2026, 11, 19, 19))
    for bad_ttl in (0, -1, True, 1.5, 7 * 86400 + 1):
        with pytest.raises(ConfigError):
            links.link_files([], FakePresigner(), NOW, ttl_s=bad_ttl)


def test_index_and_outer_files_contain_no_presigned_url_or_headers():
    files, document, _, index = one_file()
    public_bytes = b"".join(files.values())
    assert URL_SECRET.encode() not in public_bytes
    assert "http" not in public_bytes.decode()
    for forbidden in ("put_url", "put_headers", "get_url", "X-Amz", "If-None-Match"):
        assert forbidden not in public_bytes.decode()
    assert index["hotkeys"] == [A.hotkey]
    assert index["link_file"] == "generator/links/<hotkey>.json"
    assert index["upload_mode"] == links.OVERWRITE_UNTIL_DEADLINE
    assert document["current"] != open_sealed(A, document)["put_url"]


def test_invalid_empty_duplicate_and_non_curve_addresses_are_safe():
    invalid_curve = ss58(b"\x02" + bytes(31))
    fake = FakePresigner()
    files = links.link_files(
        ["", "not-an-address", "../escape", invalid_curve, B.hotkey, A.hotkey, A.hotkey], fake, NOW
    )
    assert set(files) == {
        "generator/links.json", f"generator/links/{A.hotkey}.json", f"generator/links/{B.hotkey}.json"
    }
    assert all("../" not in call[1] and "not-an-address" not in call[1] for call in fake.calls)
    assert json.loads(files["generator/links.json"])["hotkeys"] == sorted([A.hotkey, B.hotkey])
    assert set(links.link_files([], FakePresigner(), NOW)) == {"generator/links.json"}


def registrations(*rows):
    return [Registration(uid, hotkey, 1) for uid, hotkey in rows]


def test_eligibility_requires_registration_nonzero_uid_and_activation():
    registered = registrations((0, A.hotkey), (1, B.hotkey), (2, "not-activated"))
    assert links.eligible(registered, {A.hotkey, B.hotkey, "unregistered"}) == [B.hotkey]
    assert links.eligible(registered, set()) == []


def test_mandatory_activation_cli_and_read_only_chain_cleanup(tmp_path, monkeypatch, capsys):
    assert "--no-activation-check" not in Path(links.__file__).read_text()
    with pytest.raises(SystemExit):
        links.main(["issue", "--site-dir", str(tmp_path), "--no-activation-check"])
    with pytest.raises(SystemExit):
        links.main(["issue", "--network", "finney"])  # --site-dir is required
    capsys.readouterr()

    site = tmp_path / "site"
    site.mkdir()
    registered = ChainSnapshot(120, 5, NOW, tuple(registrations((0, A.hotkey), (1, B.hotkey))))
    client = SimpleNamespace(closed=False)
    client.close = lambda: setattr(client, "closed", True)
    calls = []
    monkeypatch.setattr(links, "R2Store", SimpleNamespace(from_env=lambda environ: "store"))
    monkeypatch.setattr(links.SubtensorClient, "connect", classmethod(lambda cls, netuid, network: calls.append(("connect", netuid, network)) or client))
    monkeypatch.setattr(links.Reader, "snapshot", lambda self: calls.append(("snapshot", self.netuid)) or registered)
    monkeypatch.setattr(links, "read_activated", lambda passed, hotkeys, netuid: calls.append(("activated", hotkeys, netuid)) or {B.hotkey})
    seen = []
    monkeypatch.setattr(links, "link_files", lambda hotkeys, presigner, now, **kwargs: seen.append(hotkeys) or {"generator/links.json": b'{"epoch":"20261120T0000Z","upload_mode":"overwrite_until_deadline"}'})
    assert links.main(["issue", "--site-dir", str(site), "--network", "finney"]) == 0
    assert calls == [
        ("connect", 120, "finney"), ("snapshot", 120), ("activated", [B.hotkey], 120)
    ]
    assert seen == [[B.hotkey]] and client.closed
    printed = capsys.readouterr()
    assert "http" not in printed.out + printed.err and URL_SECRET not in printed.out + printed.err


def test_cli_credentials_and_explicit_failures_do_not_leak_urls(tmp_path, monkeypatch, capsys):
    site = tmp_path / "site"
    site.mkdir()
    credentials = tmp_path / "r2.env"
    credentials.write_text("R2_ACCESS_KEY_ID=fake-id\nR2_SECRET_ACCESS_KEY=fake-secret\n")
    environ_seen = []
    monkeypatch.setattr(links, "R2Store", SimpleNamespace(from_env=lambda environ: environ_seen.append(environ) or object()))
    monkeypatch.setattr(links, "_registered_and_activated", lambda network, netuid: ([], []))
    assert links.main(["issue", "--site-dir", str(site), "--credentials-file", str(credentials)]) == 0
    assert environ_seen[0]["SN120_R2_ACCESS_KEY_ID"] == "fake-id"
    assert environ_seen[0]["SN120_R2_SECRET_ACCESS_KEY"] == "fake-secret"
    capsys.readouterr()

    monkeypatch.setattr(links, "_registered_and_activated", lambda network, netuid: (_ for _ in ()).throw(RuntimeError(URL_SECRET)))
    # main does not turn arbitrary programming errors into successful output; the
    # chain helper itself wraps failures without including their text.
    with pytest.raises(RuntimeError):
        links.main(["issue", "--site-dir", str(site)])

    class BrokenPresigner:
        def presign_put(self, key, ttl):
            raise RuntimeError(URL_SECRET)

    with pytest.raises(InfraError) as failure:
        links.link_files([A.hotkey], BrokenPresigner(), NOW)
    assert URL_SECRET not in str(failure.value)


def test_first_upload_final_uses_conditional_signature_when_probe_succeeds():
    fake = FakePresigner(conditional=True)
    files = links.link_files([A.hotkey], fake, NOW, first_upload_final=True)
    document = json.loads(files[f"generator/links/{A.hotkey}.json"])
    current = open_sealed(A, document)
    assert current["upload_mode"] == links.FIRST_UPLOAD_FINAL
    assert current["first_upload_final"] is True and current["overwrite_until_deadline"] is False
    assert current["put_headers"] == {"Content-Type": "application/octet-stream", "If-None-Match": "*"}
    assert fake.conditional_calls == [(current["object_key"], 3600)]
    assert fake.calls == [("get", current["object_key"], 3600)]
    assert json.loads(files["generator/links.json"])["upload_mode"] == links.FIRST_UPLOAD_FINAL


@pytest.mark.parametrize("binding", [True, False])
def test_unverified_or_invalid_conditional_support_fails_to_explicit_overwrite(binding):
    for first in (False, True):
        for conditional in (False, True):
            fake = FakePresigner(conditional=conditional, binding=binding)
            files = links.link_files([A.hotkey], fake, NOW, first_upload_final=first)
            current = open_sealed(A, json.loads(files[f"generator/links/{A.hotkey}.json"]))
            expected = links.FIRST_UPLOAD_FINAL if first and conditional and binding else links.OVERWRITE_UNTIL_DEADLINE
            assert current["upload_mode"] == expected
            if expected == links.OVERWRITE_UNTIL_DEADLINE:
                assert current["first_upload_final"] is False and current["overwrite_until_deadline"] is True
                assert current["put_headers"] == {"Content-Type": "application/octet-stream"}


def test_explicit_probe_is_injectable_and_probe_errors_use_overwrite_mode():
    fake = FakePresigner()
    files = links.link_files([A.hotkey], fake, NOW, first_upload_final=True, conditional_support=lambda store: False)
    current = open_sealed(A, json.loads(files[f"generator/links/{A.hotkey}.json"]))
    assert current["upload_mode"] == links.OVERWRITE_UNTIL_DEADLINE
    files = links.link_files(
        [A.hotkey], fake, NOW, first_upload_final=True,
        conditional_support=lambda store: (_ for _ in ()).throw(RuntimeError(URL_SECRET)),
    )
    current = open_sealed(A, json.loads(files[f"generator/links/{A.hotkey}.json"]))
    assert current["upload_mode"] == links.OVERWRITE_UNTIL_DEADLINE


def test_conditional_generator_adapter_uses_raw_r2_seam_and_needs_explicit_probe():
    raw = []
    store = SimpleNamespace(
        _presign=lambda operation, key, ttl, extra: raw.append((operation, key, ttl, extra))
        or "https://invalid.test/object?X-Amz-SignedHeaders=host%3Bcontent-type%3Bif-none-match",
        presign_get=lambda key, ttl: "https://invalid.test/object",
        presign_put=lambda key, ttl: ("https://invalid.test/object", {"Content-Type": "application/octet-stream"}),
    )
    adapter = links.GeneratorPresigner(store, conditional_support=lambda passed: passed is store)
    assert adapter.first_upload_supported()
    url, headers = adapter.presign_first_upload_put("key", 60)
    assert raw == [("put_object", "key", 60, {"ContentType": "application/octet-stream", "IfNoneMatch": "*"})]
    assert links._binds_if_none_match(url) and headers == links.FIRST_UPLOAD_HEADERS
    assert not links.GeneratorPresigner(store).first_upload_supported()


def test_publication_is_atomic_idempotent_index_last_and_removes_stale_files(tmp_path, monkeypatch):
    root = tmp_path / "site"
    root.mkdir()
    path = root / "generator" / "links" / f"{A.hotkey}.json"
    first = {f"generator/links/{A.hotkey}.json": b'{"one":1}\n', "generator/links.json": b'{"hotkeys":["A"]}\n'}
    assert links.write_files(first, root) == 0
    before = path.stat().st_mtime_ns
    replaced = []
    original = links.os.replace
    monkeypatch.setattr(links.os, "replace", lambda source, target: replaced.append((source, target)) or original(source, target))
    assert links.write_files(first, root) == 0
    assert path.stat().st_mtime_ns == before and replaced == []
    path.write_bytes(b"old")
    stale = root / "generator" / "links" / "stale.json"
    stale.write_text("{}")
    keep = root / "generator" / "links" / "README.txt"
    keep.write_text("not a link")
    assert links.write_files(first, root) == 1
    assert path.read_bytes() == first[f"generator/links/{A.hotkey}.json"]
    assert not stale.exists() and keep.exists()
    second = first | {"generator/links.json": b'{"hotkeys":["A"],"changed":true}\n'}
    assert links.write_files(second, root) == 0
    assert Path(replaced[-1][1]) == root / "generator/links.json"
    assert not list(root.rglob(".*.tmp"))
    with pytest.raises(ConfigError):
        links.write_files(first, tmp_path / "missing")


def test_optional_r2_mirror_uses_only_public_generator_keys():
    class Mirror:
        def __init__(self):
            self.data = {links.PUBLIC_LINKS_DIR + "stale.json": b"old"}

        def put(self, key, data):
            self.data[key] = data

        def list(self, prefix):
            return sorted(key for key in self.data if key.startswith(prefix))

        def delete(self, key):
            del self.data[key]

    mirror = Mirror()
    files = {"generator/links.json": b"{}", f"generator/links/{A.hotkey}.json": b"sealed"}
    assert links.mirror_files(files, mirror) == 1
    assert mirror.data == {
        links.PUBLIC_INDEX: b"{}", links.PUBLIC_LINKS_DIR + f"{A.hotkey}.json": b"sealed"
    }
