"""The link commands end to end: a real presigned PUT against an S3 fake (moto, over HTTP on localhost)."""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import threading
import urllib.error
import urllib.request
import uuid

import boto3
import pytest
from moto.server import ThreadedMotoServer
from nacl.signing import SigningKey

from sn120_v2 import links
from sn120_v2.journal import JOURNAL_FILE, Journal, Mirror
from sn120_v2.r2 import R2Store
from sn120_v2.sim import SimWorld, sim_params
from sn120_v2.state import State

SECRET = "fake-secret-value-never-shown"
PREFIX = "test-run/"
URL = "https://site.test/ff/links"  # the public address of the link files, as the operator gives it


class Miner:
    """A hotkey with an Ed25519 key, as a miner holds it."""

    def __init__(self, label: str) -> None:
        self.seed = hashlib.sha256(label.encode()).digest()
        self.key = SigningKey(self.seed)
        body = b"\x2a" + bytes(self.key.verify_key)  # SS58: prefix 42, the public key, 2 checksum bytes, in base58
        number = int.from_bytes(body + hashlib.blake2b(b"SS58PRE" + body, digest_size=64).digest()[:2], "big")
        self.hotkey = ""
        while number:
            number, digit = divmod(number, 58)
            self.hotkey = links._B58[digit] + self.hotkey

    def activation(self) -> str:
        return links.activation_text(self.hotkey, self.key.sign(links.activation_message(self.hotkey, 120)).signature)


A, B, C = Miner("a"), Miner("b"), Miner("c")


@pytest.fixture(scope="module")
def endpoint():
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def bucket(endpoint, monkeypatch):
    """A fresh bucket, and the store settings in the environment as the operator has them."""
    name = f"b-{uuid.uuid4().hex}"
    raw = boto3.client("s3", endpoint_url=endpoint, region_name="auto", aws_access_key_id="fake-id", aws_secret_access_key=SECRET)
    raw.create_bucket(Bucket=name, CreateBucketConfiguration={"LocationConstraint": "auto"})
    settings = {"ENDPOINT": endpoint, "BUCKET": name, "ACCESS_KEY_ID": "fake-id", "SECRET_ACCESS_KEY": SECRET, "PREFIX": PREFIX}
    for key, value in settings.items():
        monkeypatch.setenv(f"SN120_R2_{key}", value)
    read = lambda key: raw.get_object(Bucket=name, Key=key)["Body"].read()
    keys = lambda: sorted(item["Key"] for item in raw.list_objects_v2(Bucket=name).get("Contents", []))
    return type("Bucket", (), {"read": staticmethod(read), "keys": staticmethod(keys)})


@pytest.fixture
def queue(build, monkeypatch):
    """The controller's state: A and B wait, C had its turn, UID 0 is the burn UID."""
    seats = build.queue(build.seat(0, hotkey=C.hotkey), build.seat(3, hotkey=A.hotkey), build.seat(4, hotkey=B.hotkey, leased=True))
    state = State(genesis=build.genesis_spec(), queue=seats)
    monkeypatch.setattr(links, "load", lambda path: state)
    return state


@pytest.fixture
def site(tmp_path):
    """The website root, served over HTTP as the public site serves it."""
    root = tmp_path / "site"
    root.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *args: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{server.server_address[1]}/ff/links"
    server.shutdown()


def put(url: str, data: bytes, headers: dict[str, str]) -> int:
    request = urllib.request.Request(url, data=data, headers=headers, method="PUT")
    try:
        with urllib.request.urlopen(request, timeout=30) as reply:
            return reply.status
    except urllib.error.HTTPError as exc:
        return exc.code


def seed_file(tmp_path, miner):
    path = tmp_path / f"{miner.hotkey}.seed"
    path.write_text(miner.seed.hex())
    return str(path)


def chain_with(monkeypatch, rows=None, error=None):
    def query_map(item, params):
        if error:
            raise error
        return iter(rows)

    sdk = type("Sdk", (), {"storage": type("S", (), {"Commitments": type("C", (), {"RevealedCommitments": "REVEALED"})})})
    client = type("Client", (), {"chain": type("Chain", (), {"query_map": staticmethod(query_map)}), "sdk": sdk})
    monkeypatch.setattr(links.SubtensorClient, "connect", classmethod(lambda cls, netuid, network: client))


def test_a_miner_gets_its_link_from_the_site_and_uploads_its_batch(bucket, queue, site, tmp_path, monkeypatch, capsys):
    root, address = site
    chain_with(monkeypatch, rows=[(A.hotkey, [("a\x02" + A.activation(), 1)]), (B.hotkey, [('{"model": "x"}', 1)])])
    assert links.main(["issue", "--links-url", URL, "--state-dir", str(tmp_path), "--site-dir", str(root), "--network", "finney"]) == 0
    printed = capsys.readouterr()
    assert "issued 1 of 2 waiting hotkeys (not activated 1, not Ed25519 0), removed 0" in printed.out
    assert "X-Amz" not in printed.out + printed.err and "http" not in printed.out  # the job never prints an address
    assert sorted(path.name for path in (root / "ff" / "links").iterdir()) == [f"{A.hotkey}.json"]  # B is not activated
    assert json.loads((root / "ff" / "links.json").read_text())["hotkeys"] == [A.hotkey]

    assert links.main(["open", "--hotkey", A.hotkey, "--key", seed_file(tmp_path, A), "--links-url", address, "--json"]) == 0
    link = json.loads(capsys.readouterr().out)
    assert link["put_headers"] == {"Content-Type": "application/octet-stream"}
    key = PREFIX + f"private/ff/v1/miners/{A.hotkey}/batch.json"  # where the controller reads the batch
    assert put(link["put_url"], b'{"version": 1}', link["put_headers"]) == 200 and bucket.read(key) == b'{"version": 1}'
    assert put(link["put_url"], b'{"version": 2}', link["put_headers"]) == 200 and bucket.read(key) == b'{"version": 2}'  # replace
    with urllib.request.urlopen(link["get_url"], timeout=30) as reply:
        assert reply.read() == b'{"version": 2}'
    # The fake does not check signatures, so a refused upload cannot be shown here. What can be shown: the
    # Content-Type header is one of the signed headers, and the address names this hotkey's object only.
    assert "X-Amz-SignedHeaders=content-type%3Bhost" in link["put_url"] and f"/{key}?" in link["put_url"]
    assert bucket.keys() == [key]

    assert links.main(["open", "--hotkey", A.hotkey, "--key", seed_file(tmp_path, A), "--links-url", address]) == 0
    assert capsys.readouterr().out.strip() == link["put_url"]  # by default: only the upload address
    assert links.main(["open", "--hotkey", B.hotkey, "--key", seed_file(tmp_path, B), "--links-url", address]) == 1
    assert "no link for this hotkey" in capsys.readouterr().err


def test_a_failed_chain_read_writes_nothing_and_keeps_the_old_links(bucket, queue, site, tmp_path, monkeypatch, capsys):
    root, _ = site
    args = ["issue", "--links-url", URL, "--state-dir", str(tmp_path), "--site-dir", str(root)]
    assert links.main(args + ["--no-activation-check"]) == 0  # the operator's switch: every waiting hotkey gets a link
    before = {path.name: path.read_bytes() for path in (root / "ff" / "links").iterdir()}
    assert sorted(before) == sorted(f"{m.hotkey}.json" for m in (A, B))  # C holds the burn UID: no link
    capsys.readouterr()

    chain_with(monkeypatch, error=ConnectionError("node is down"))
    assert links.main(args + ["--network", "finney"]) == 1
    assert "cannot read the activations" in capsys.readouterr().err
    chain_with(monkeypatch, rows=[])
    assert links.main(args + ["--network", "finney"]) == 1
    monkeypatch.setattr(links.SubtensorClient, "connect", classmethod(lambda cls, netuid, network: 1 / 0))
    assert links.main(args + ["--network", "finney"]) == 1
    assert "cannot connect to the chain" in capsys.readouterr().err
    assert links.main(args) == 1  # neither a chain nor the switch
    assert {path.name: path.read_bytes() for path in (root / "ff" / "links").iterdir()} == before


def test_a_hotkey_that_had_its_turn_loses_its_link_file(bucket, queue, site, tmp_path):
    root, _ = site
    args = ["issue", "--links-url", URL, "--state-dir", str(tmp_path), "--site-dir", str(root), "--no-activation-check"]
    assert links.main(args) == 0
    seat = next(key for key, seat in queue.queue.items() if seat.reg.hotkey == A.hotkey)
    del queue.queue[seat]
    queue.served_hotkeys.add(A.hotkey)
    assert links.main(args) == 0
    assert sorted(path.name for path in (root / "ff" / "links").iterdir()) == [f"{B.hotkey}.json"]
    assert json.loads((root / "ff" / "links.json").read_text())["hotkeys"] == [B.hotkey]


def test_the_files_can_go_to_the_public_part_of_the_store(bucket, queue, tmp_path, capsys):
    assert links.main(["issue", "--links-url", URL, "--state-dir", str(tmp_path), "--to-store", "--no-activation-check", "--ttl-s", "3600", "--reissue-s", "600"]) == 0
    names = [PREFIX + "public/ff/links.json", *(PREFIX + f"public/ff/links/{m.hotkey}.json" for m in sorted((A, B), key=lambda m: m.hotkey))]
    assert bucket.keys() == sorted(names)
    doc = json.loads(bucket.read(PREFIX + f"public/ff/links/{B.hotkey}.json"))
    link = links.open_link(doc, B.hotkey, B.seed, doc["issued_at"])
    assert link["expires_at"] - link["issued_at"] == 3600 and put(link["put_url"], b"batch", link["put_headers"]) == 200
    assert links.main(["issue", "--links-url", URL, "--state-dir", str(tmp_path), "--to-store", "--no-activation-check", "--ttl-s", str(8 * 86400)]) == 1
    assert "ttl" in capsys.readouterr().err


def test_open_reads_a_local_file_and_refuses_the_wrong_key(bucket, queue, site, tmp_path, capsys):
    root, _ = site
    assert links.main(["issue", "--links-url", URL, "--state-dir", str(tmp_path), "--site-dir", str(root), "--no-activation-check"]) == 0
    path = str(root / "ff" / "links" / f"{A.hotkey}.json")
    capsys.readouterr()
    assert links.main(["open", "--hotkey", A.hotkey, "--key", seed_file(tmp_path, A), "--file", path]) == 0
    assert capsys.readouterr().out.startswith("http")
    assert links.main(["open", "--hotkey", A.hotkey, "--key", seed_file(tmp_path, B), "--file", path]) == 1
    printed = capsys.readouterr()
    assert printed.out == "" and "not the Ed25519 key" in printed.err
    assert links.main(["open", "--hotkey", A.hotkey, "--key", seed_file(tmp_path, A)]) == 1  # neither an address nor a file
    assert "--links-url" in capsys.readouterr().err
    (tmp_path / "torn.json").write_text("{")
    assert links.main(["open", "--hotkey", A.hotkey, "--key", seed_file(tmp_path, A), "--file", str(tmp_path / "torn.json")]) == 1
    assert "cannot read the link file" in capsys.readouterr().err


def test_issue_needs_the_journal(bucket, site, tmp_path, capsys):
    (site[0] / "ff" / "links").mkdir(parents=True)
    (site[0] / "ff" / "links" / "old.json").write_text("{}")
    assert links.main(["issue", "--links-url", URL, "--state-dir", str(tmp_path / "wrong"), "--site-dir", str(site[0]), "--no-activation-check"]) == 1
    assert "cannot read the journal" in capsys.readouterr().err
    assert (site[0] / "ff" / "links" / "old.json").exists()  # a wrong state directory removes no link


def test_issue_reads_the_queue_from_a_real_journal(bucket, site, tmp_path, capsys):
    world = SimWorld(tmp_path / "state", sim_params())
    for uid in (1, 2, 3):
        world.register(uid, 0.1)
    args = ["issue", "--links-url", URL, "--state-dir", str(tmp_path / "state"), "--site-dir", str(site[0]), "--no-activation-check"]
    assert links.main(args) == 0  # the simulation's hotkeys are not SS58 addresses: they wait, but cannot get a link
    assert "issued 0 of 3 waiting hotkeys (not activated 0, not Ed25519 3)" in capsys.readouterr().out
    world.round()  # every entry is evaluated
    assert links.main(args) == 0
    assert "issued 0 of 0 waiting hotkeys" in capsys.readouterr().out


def test_issue_reads_the_journal_copy_in_the_store_on_a_host_without_the_controller(bucket, site, tmp_path, capsys):
    cache = tmp_path / "cache"
    args = ["issue", "--links-url", URL, "--state-dir", str(cache), "--journal-from-store", "--site-dir", str(site[0]), "--no-activation-check"]
    (site[0] / "ff" / "links").mkdir(parents=True)
    (site[0] / "ff" / "links" / "old.json").write_text("{}")
    assert links.main(args) == 1  # the store holds no journal yet: that is not "nobody waits"
    assert "no genesis" in capsys.readouterr().err and (site[0] / "ff" / "links" / "old.json").exists()

    world = SimWorld(tmp_path / "controller", sim_params())  # the controller's host: its journal, and a copy in the store
    for uid in (1, 2, 3):
        world.register(uid, 0.1)
    Journal(tmp_path / "controller" / JOURNAL_FILE, "test", world.now, Mirror(R2Store.from_env()))
    assert links.main(args) == 0
    assert "issued 0 of 3 waiting hotkeys (not activated 0, not Ed25519 3), removed 1" in capsys.readouterr().out
    assert (cache / JOURNAL_FILE).exists() and not (site[0] / "ff" / "links" / "old.json").exists()


def test_issue_needs_the_public_address_of_the_link_files(bucket, queue, site, tmp_path, capsys):
    args = ["issue", "--state-dir", str(tmp_path), "--site-dir", str(site[0]), "--no-activation-check"]
    with pytest.raises(SystemExit):
        links.main(args)  # no default: the address is never guessed
    assert links.main(args + ["--links-url", "ff/links"]) == 1
    assert "public address of the link files" in capsys.readouterr().err and not (site[0] / "ff").exists()


def test_issue_needs_the_store_settings(queue, site, tmp_path, monkeypatch, capsys):
    for name in ("ENDPOINT", "BUCKET", "ACCESS_KEY_ID", "SECRET_ACCESS_KEY"):
        monkeypatch.delenv(f"SN120_R2_{name}", raising=False)
    assert links.main(["issue", "--links-url", URL, "--state-dir", str(tmp_path), "--site-dir", str(site[0]), "--no-activation-check"]) == 1
    assert "SN120_R2_ENDPOINT" in capsys.readouterr().err and not (site[0] / "ff").exists()
