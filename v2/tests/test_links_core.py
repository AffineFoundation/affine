"""Sealed links and activation: identity, the activation check, the link format, who gets a link."""

from __future__ import annotations

import base64
import hashlib
import json
from types import SimpleNamespace

import pytest
from nacl.public import SealedBox
from nacl.signing import SigningKey

from sn120_v2 import layout, links
from sn120_v2.errors import ConfigError, ContractError, InfraError
from sn120_v2.publisher import refuse_private
from sn120_v2.state import State
from sn120_v2.store import LocalStore

NOW = 1_800_000_000
URL = "https://site.test/ff/links"  # the public address of the link files
# Public chain data, read from Finney: one real v1 activation on subnet 120.
REAL_HOTKEY = "5C5BmN1XCueA5BC1dxEYztCRWhHcYXy8BYbZ6PP9xTnu5VHE"
REAL_ACTIVATION = f"affine2|activate|{REAL_HOTKEY}|96dT4mRhb_vF7kyFzvfhDGz08ox7iblxLMDUmrxWVVGrYT8oP5YU65wwjZ85efYT1Unp2rlwFgSQDjIE3sHLCg"
REAL_ON_CHAIN = "a\x02" + REAL_ACTIVATION  # as the SDK returns it: a 2-byte length prefix, then the text
OWNER = "5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8"


def ss58(public: bytes, prefix: int = 42) -> str:
    body = bytes([prefix]) + public
    number = int.from_bytes(body + hashlib.blake2b(b"SS58PRE" + body, digest_size=64).digest()[:2], "big")
    text = ""
    while number:
        number, digit = divmod(number, 58)
        text = links._B58[digit] + text
    return text


class Miner:
    """A hotkey with an Ed25519 key, as a miner holds it."""

    def __init__(self, label: str) -> None:
        self.seed = hashlib.sha256(label.encode()).digest()
        self.key = SigningKey(self.seed)
        self.hotkey = ss58(bytes(self.key.verify_key))

    def activation(self, netuid: int = 120) -> str:
        return links.activation_text(self.hotkey, self.key.sign(links.activation_message(self.hotkey, netuid)).signature)

    def wallet(self) -> SimpleNamespace:
        sign = lambda message: self.key.sign(message).signature
        return SimpleNamespace(hotkey=SimpleNamespace(ss58_address=self.hotkey, sign=sign))


class FakePresigner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def presign_get(self, key: str, ttl_s: int) -> str:
        self.calls.append(("get", key, ttl_s))
        return f"https://r2.test/{key}?op=get&sig=SECRET"

    def presign_put(self, key: str, ttl_s: int) -> tuple[str, dict[str, str]]:
        self.calls.append(("put", key, ttl_s))
        return f"https://r2.test/{key}?op=put&sig=SECRET", {"Content-Type": "application/octet-stream"}


A, B, C = Miner("a"), Miner("b"), Miner("c")


# ---- identity ----


def test_public_key_of_an_address():
    assert links.public_key(A.hotkey) == bytes(A.key.verify_key)
    assert len(links.public_key(OWNER)) == 32 and ss58(links.public_key(OWNER)) == OWNER


@pytest.mark.parametrize("bad", ["", "hk1", A.hotkey[:-1] + ("a" if A.hotkey[-1] != "a" else "b"), A.hotkey + "1111", "0OIl" * 12, "../" + A.hotkey])
def test_public_key_refuses_a_bad_address(bad):
    with pytest.raises(ContractError):
        links.public_key(bad)


# ---- activation ----


def test_activation_message_is_the_v1_message():
    expected = f"affine-activate|v1|120|{OWNER}|feda8015ef5480f767c30b93be94421d6613f09dc652b3c5ce377c5b835fe2f1"
    assert links.activation_message(OWNER, 120) == expected.encode()


def test_a_real_v1_activation_from_the_chain_stays_valid():
    assert links.valid_activation(REAL_ACTIVATION, REAL_HOTKEY, 120)
    assert not links.valid_activation(REAL_ACTIVATION, REAL_HOTKEY, 121)
    assert links.activated([(REAL_HOTKEY, [(REAL_ON_CHAIN, 9238640)])], [REAL_HOTKEY], 120) == {REAL_HOTKEY}


def test_activation_needs_the_signature_of_this_hotkey_for_this_subnet():
    good = A.activation()
    assert links.valid_activation(good, A.hotkey, 120)
    assert not links.valid_activation(good, A.hotkey, 7)  # signed for another subnet
    assert not links.valid_activation(good, B.hotkey, 120)  # another hotkey
    stolen = links.activation_text(A.hotkey, B.key.sign(links.activation_message(A.hotkey, 120)).signature)
    assert not links.valid_activation(stolen, A.hotkey, 120)  # signed by another key
    for broken in ("", "affine2|activate|" + A.hotkey, good + "|x", good[:-4] + "!!!!", good[:-10], good.replace("affine2", "affine3")):
        assert not links.valid_activation(broken, A.hotkey, 120)


def test_a_key_that_is_not_ed25519_cannot_activate():
    other = ss58(hashlib.sha256(b"sr25519 public key").digest())  # 32 bytes that A's secret key does not match
    forged = links.activation_text(other, A.key.sign(links.activation_message(other, 120)).signature)
    assert not links.valid_activation(forged, other, 120)


def test_activated_reads_each_form_of_the_chain_value():
    text = A.activation().encode()
    prefixed = ((len(text) << 2) | 1).to_bytes(2, "little") + text
    rows = [
        (A.hotkey, [('{"model": "x"}', 5), (prefixed.decode("latin-1"), 9)]),  # an old other commitment, then the activation
        (B.hotkey, [("0x" + (b"\x61\x02" + B.activation().encode()).hex(), 9)]),  # hex
        (C.hotkey, [(b"\x61\x02" + C.activation().encode(), 9)]),  # bytes
    ]
    assert links.activated(rows, [A.hotkey, B.hotkey, C.hotkey], 120) == {A.hotkey, B.hotkey, C.hotkey}
    assert links.activated(rows, [B.hotkey], 120) == {B.hotkey}  # only the hotkeys that were asked for
    assert links.activated(rows, [A.hotkey, B.hotkey, C.hotkey], 121) == set()


def test_activated_ignores_bad_rows():
    rows = [
        (A.hotkey, [("a\x02" + B.activation(), 1)]),  # another hotkey's activation under this key
        (B.hotkey, [("0xzz", 1), ("", 2), (b"\xff\xfe", 3), ("x" * 9 + B.activation(), 4), None]),  # text too far from the start
        (C.hotkey, None),
    ]
    assert links.activated(rows, [A.hotkey, B.hotkey, C.hotkey], 120) == set()


def fake_client(rows=None, error: Exception | None = None):
    seen = []

    def query_map(item, params):
        seen.append((item, params))
        if error:
            raise error
        return iter(rows)

    sdk = SimpleNamespace(storage=SimpleNamespace(Commitments=SimpleNamespace(RevealedCommitments="REVEALED")))
    return SimpleNamespace(chain=SimpleNamespace(query_map=query_map), sdk=sdk, seen=seen)


def test_read_activated_reads_the_revealed_commitments_of_the_subnet():
    client = fake_client([(A.hotkey, [("a\x02" + A.activation(), 1)]), (B.hotkey, [("other", 1)])])
    assert links.read_activated(client, [A.hotkey, B.hotkey], 120) == {A.hotkey}
    assert client.seen == [("REVEALED", [120])]


@pytest.mark.parametrize("client", [fake_client(error=TimeoutError("node is down")), fake_client(rows=[])])
def test_a_failed_or_empty_chain_read_is_an_error_not_an_empty_answer(client):
    with pytest.raises(InfraError):
        links.read_activated(client, [A.hotkey], 120)


class FakeSdk:
    """The parts of the bittensor SDK that ``activate`` uses."""

    def __init__(self, uid) -> None:
        self.uid, self.submitted, self.locked = uid, [], []
        self.storage = SimpleNamespace(SubtensorModule=SimpleNamespace(Uids="UIDS"))
        self.timelock = SimpleNamespace(encrypt=self._encrypt)
        self.calls = SimpleNamespace(Commitments=SimpleNamespace(set_commitment=lambda netuid, info: ("set_commitment", netuid, info)))

    def _encrypt(self, data, reveal_in):
        self.locked.append((data, reveal_in))
        return SimpleNamespace(ciphertext=b"locked:" + data, reveal_round=77)

    def Subtensor(self, network):
        def submit_call(call, wallet, signer):
            self.submitted.append((call, wallet, signer))
            return SimpleNamespace(raise_for_failure=lambda: None, block_hash="0xabc")

        return SimpleNamespace(query=lambda item, params: self.uid, submit_call=submit_call)


def test_activate_dry_run_prints_a_valid_activation_and_writes_nothing():
    sdk = FakeSdk(uid=5)
    out = links.activate(A.wallet(), sdk, 120, "finney", execute=False)
    assert out["status"] == "dry_run" and links.valid_activation(out["payload"], A.hotkey, 120)
    assert sdk.submitted == [] and sdk.locked == []


def test_activate_execute_writes_one_timelocked_commitment_signed_by_the_hotkey():
    sdk, wallet = FakeSdk(uid=5), A.wallet()
    out = links.activate(wallet, sdk, 120, "finney", execute=True)
    assert out == {"status": "submitted", "netuid": 120, "uid": 5, "block_hash": "0xabc"}
    (data, reveal_in), = sdk.locked
    assert reveal_in == "180s" and int.from_bytes(data[:2], "little") >> 2 == len(data) - 2 and data[0] & 3 == 1
    assert links.activated([(A.hotkey, [(data, 1)])], [A.hotkey], 120) == {A.hotkey}  # the reader accepts what the writer makes
    (call, used_wallet, signer), = sdk.submitted
    assert used_wallet is wallet and signer == "hotkey"
    assert call == ("set_commitment", 120, {"fields": [{"TimelockEncrypted": {"encrypted": b"locked:" + data, "reveal_round": 77}}]})


def test_activate_refuses_an_unregistered_hotkey_and_a_key_that_is_not_ed25519():
    sdk = FakeSdk(uid=None)
    with pytest.raises(ConfigError, match="not registered"):
        links.activate(A.wallet(), sdk, 120, "finney", execute=True)
    wrong = SimpleNamespace(hotkey=SimpleNamespace(ss58_address=A.hotkey, sign=lambda message: b"\x01" * 64))
    with pytest.raises(ConfigError, match="Ed25519"):
        links.activate(wrong, FakeSdk(uid=5), 120, "finney", execute=True)
    assert sdk.submitted == []


# ---- the link format ----


def test_link_files_have_the_v1_shape_and_only_the_owner_can_open_them():
    presigner = FakePresigner()
    files = links.link_files([B.hotkey, A.hotkey], presigner, NOW, URL)
    assert sorted(files) == sorted([f"ff/links/{A.hotkey}.json", f"ff/links/{B.hotkey}.json", "ff/links.json"])
    raw = files[f"ff/links/{A.hotkey}.json"]
    outer = json.loads(raw)
    assert raw == (json.dumps(outer, indent=1, sort_keys=True) + "\n").encode()
    assert set(outer) == {"version", "hotkey", "issued_at", "expires_at", "sealed_to", "sealed"}
    assert (outer["version"], outer["hotkey"], outer["issued_at"], outer["expires_at"]) == ("affine-ff-link-v1", A.hotkey, NOW, NOW + 86400)
    assert outer["sealed_to"] == bytes(A.key.verify_key).hex()
    # the way a v1 miner opens it: the Ed25519 seed, converted to a Curve25519 key, opens a libsodium sealed box
    inner = json.loads(SealedBox(A.key.to_curve25519_private_key()).decrypt(base64.b64decode(outer["sealed"])))
    key = layout.miner_object(A.hotkey)
    assert set(inner) == {"version", "hotkey", "object_key", "issued_at", "expires_at", "put_url", "put_headers", "get_url", "format", "spec"}
    assert inner["object_key"] == key == f"private/ff/v1/miners/{A.hotkey}/batch.json"
    assert inner["put_url"] == f"https://r2.test/{key}?op=put&sig=SECRET" and inner["get_url"].endswith("op=get&sig=SECRET")
    assert inner["put_headers"] == {"Content-Type": "application/octet-stream"}
    assert (inner["hotkey"], inner["issued_at"], inner["expires_at"]) == (A.hotkey, NOW, NOW + 86400)
    assert links.open_link(outer, A.hotkey, A.seed, NOW) == inner
    assert ("put", key, 86400) in presigner.calls and ("get", key, 86400) in presigner.calls


def test_no_public_file_shows_an_address_in_the_clear():
    files = links.link_files([A.hotkey, B.hotkey], FakePresigner(), NOW, URL)
    for content in files.values():
        assert b"r2.test" not in content and b"SECRET" not in content and b"private/" not in content


def test_the_index_lists_the_hotkeys_that_have_a_link():
    index = json.loads(links.link_files([B.hotkey, A.hotkey, A.hotkey], FakePresigner(), NOW, URL, ttl_s=7200, reissue_s=600)["ff/links.json"])
    assert index == {
        "version": "affine-ff-link-v1",
        "issued_at": NOW,
        "issued_at_cr": "2027-01-15 02:00:00 CR (UTC-6)",  # 08:00:00 UTC
        "expires_at": NOW + 7200,
        "hotkeys": sorted([A.hotkey, B.hotkey]),
        "reissue_every_seconds": 600,
        "link_file": "https://site.test/ff/links/<hotkey>.json",
    }
    assert json.loads(links.link_files([], FakePresigner(), NOW, URL)["ff/links.json"])["hotkeys"] == []


@pytest.mark.parametrize("ttl_s, reissue_s", [(7 * 86400 + 1, 1800), (0, 0), (1800, 1800), (600, 1800), (86400, 0)])
def test_link_times_have_limits(ttl_s, reissue_s):
    with pytest.raises(ConfigError):
        links.link_files([A.hotkey], FakePresigner(), NOW, URL, ttl_s=ttl_s, reissue_s=reissue_s)
    assert links.link_files([A.hotkey], FakePresigner(), NOW, URL, ttl_s=7 * 86400, reissue_s=1800)


def test_a_hotkey_that_cannot_be_sealed_to_gets_no_link_and_stops_no_other():
    presigner = FakePresigner()
    not_a_curve_point = ss58(b"\x02" + bytes(31))  # a valid address whose key is not an Ed25519 public key
    files = links.link_files([A.hotkey, "not-an-address", "../../etc", not_a_curve_point], presigner, NOW, URL)
    assert set(files) == {f"ff/links/{A.hotkey}.json", "ff/links.json"}
    assert json.loads(files["ff/links.json"])["hotkeys"] == [A.hotkey]
    assert all("etc" not in key and "not-an-address" not in key for _, key, _ in presigner.calls)  # a bad name signs nothing


def test_open_link_refuses_the_wrong_key_another_hotkey_an_old_link_and_a_changed_file():
    files = links.link_files([A.hotkey, B.hotkey], FakePresigner(), NOW, URL)
    mine, theirs = (json.loads(files[f"ff/links/{m.hotkey}.json"]) for m in (A, B))
    with pytest.raises(ConfigError, match="not the Ed25519 key"):
        links.open_link(mine, A.hotkey, B.seed, NOW)
    with pytest.raises(ContractError, match="cannot open"):
        links.open_link(theirs, A.hotkey, A.seed, NOW)  # sealed to B: A's key does not open it
    with pytest.raises(ContractError, match="expired"):
        links.open_link(mine, A.hotkey, A.seed, NOW + 86400 - 59)
    assert links.open_link(mine, A.hotkey, A.seed, NOW + 86400 - 60)
    changed = mine | {"sealed": base64.b64encode(base64.b64decode(mine["sealed"])[:-1] + b"\x00").decode()}
    for broken in (changed, mine | {"sealed": "not base64!"}, {"hotkey": A.hotkey}):
        with pytest.raises(ContractError):
            links.open_link(broken, A.hotkey, A.seed, NOW)
    relabeled = A.hotkey  # a link whose inside names another hotkey, sealed to A
    forged = {"sealed": links.seal({"hotkey": B.hotkey, "expires_at": NOW + 999}, relabeled)}
    with pytest.raises(ContractError, match="another hotkey"):
        links.open_link(forged, A.hotkey, A.seed, NOW)


def test_read_seed_reads_a_hex_file_and_a_bittensor_hotkey_file(tmp_path):
    plain, wallet, bad = tmp_path / "seed.hex", tmp_path / "hotkey", tmp_path / "bad"
    plain.write_text(A.seed.hex() + "\n")
    wallet.write_text(json.dumps({"ss58Address": A.hotkey, "secretSeed": "0x" + A.seed.hex()}))
    assert links.read_seed(plain) == links.read_seed(wallet) == A.seed
    for content in ("abcd", "zz" * 32, "{not json", json.dumps({"ss58Address": A.hotkey})):
        bad.write_text(content)
        with pytest.raises(ConfigError) as caught:
            links.read_seed(bad)
        assert content not in str(caught.value)  # an error never repeats what the key file holds
    with pytest.raises(ConfigError):
        links.read_seed(tmp_path / "missing")


# ---- who gets a link, and where the files go ----


def test_every_seat_that_is_not_evaluated_yet_may_upload(build):
    state = State(
        queue=build.queue(
            build.seat(0, hotkey="burn"),
            build.seat(3, hotkey="waiting"),
            build.seat(4, hotkey="in-a-round", leased=True),
            build.seat(5, hotkey="held", held=True),
            build.seat(6, hotkey="demoted", demoted=True),
            build.seat(7, hotkey="front", front=True),
            build.seat(8, hotkey="had-its-turn"),
        ),
        served_hotkeys={"had-its-turn", "gone"},
    )
    assert links.eligible(state, burn_uid=0) == sorted(["waiting", "in-a-round", "held", "demoted", "front"])
    assert "burn" in links.eligible(state, burn_uid=99)
    assert links.eligible(State(), burn_uid=0) == []


@pytest.mark.parametrize("prefix", ["", "public/"])
def test_write_files_replaces_links_and_removes_those_of_hotkeys_that_left(tmp_path, prefix):
    sink = LocalStore(tmp_path)
    sink.put(prefix + "ff/links/gone.json", b"old")
    sink.put(prefix + f"ff/links/{A.hotkey}.json", b"old")
    sink.put(prefix + "ff/status.json", b"not mine")
    sink.put(prefix + "llms.txt", b"not mine")
    files = links.link_files([A.hotkey, B.hotkey], FakePresigner(), NOW, URL)
    assert links.write_files(files, sink, prefix) == 1
    assert sink.list(prefix + "ff/") == sorted(prefix + name for name in [*files, "ff/status.json"])
    assert sink.get(prefix + f"ff/links/{A.hotkey}.json") == files[f"ff/links/{A.hotkey}.json"]
    assert sink.get(prefix + "llms.txt") == b"not mine"
    assert links.write_files(links.link_files([], FakePresigner(), NOW, URL), sink, prefix) == 2  # nobody waits: no link file stays


# ---- the public snapshot ----


def test_upload_entry_is_public_and_names_the_link_file():
    entry = links.upload_entry("https://site.test/ff/links/")
    assert entry["sealed_links"] == "https://site.test/ff/links/<hotkey>.json" and entry["index"] == "https://site.test/ff/links.json"
    assert (entry["method"], entry["content_type"]) == ("PUT", "application/octet-stream")
    assert (entry["link_ttl_seconds"], entry["reissue_every_seconds"]) == (86400, 1800)
    assert entry["open"].endswith("--links-url https://site.test/ff/links")
    refuse_private({"upload": entry})  # the publisher's own guard: no upload address, no private object key
    assert json.loads(json.dumps(entry)) == entry
    other = links.upload_entry("https://example.test/public/ff/links", ttl_s=3600, reissue_s=600)
    assert other["sealed_links"] == "https://example.test/public/ff/links/<hotkey>.json" and other["link_ttl_seconds"] == 3600


@pytest.mark.parametrize("bad", ["", "site.test/ff/links", "ff/links", "https://site.test/ff/links.json", "private/ff/links"])
def test_the_public_address_of_the_link_files_is_given_and_checked(bad):
    with pytest.raises(ConfigError):
        links.upload_entry(bad)
    with pytest.raises(ConfigError):
        links.link_files([A.hotkey], FakePresigner(), NOW, bad)
