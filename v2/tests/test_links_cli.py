"""The link commands end to end: a real presigned PUT against an S3 fake (moto, over HTTP on localhost)."""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

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
    """The state in the journal copy that the store holds: A and B wait, C had its turn, UID 0 is the burn UID."""
    seats = build.queue(build.seat(0, hotkey=C.hotkey), build.seat(3, hotkey=A.hotkey), build.seat(4, hotkey=B.hotkey, leased=True))
    state = State(genesis=build.genesis_spec(), queue=seats)
    monkeypatch.setattr(links, "load_from_store", lambda store, cache_path: state)
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


def controller(root, miners: int, rounds: int):
    """A controller's host: its own journal on disk, and the copy of it in the store."""
    world = SimWorld(root, sim_params())
    for uid in range(1, miners + 1):
        world.register(uid, 0.1)
    for _ in range(rounds):
        world.round()  # every entry is evaluated
    Journal(root / JOURNAL_FILE, "test", world.now, Mirror(R2Store.from_env()))


def old_link(site):
    (site[0] / "ff" / "links").mkdir(parents=True)
    (site[0] / "ff" / "links" / "old.json").write_text("{}")
    return site[0] / "ff" / "links" / "old.json"


def test_issue_needs_a_journal_in_the_store(bucket, site, tmp_path, capsys):
    old = old_link(site)
    assert links.main(["issue", "--links-url", URL, "--state-dir", str(tmp_path / "copy"), "--site-dir", str(site[0]), "--no-activation-check"]) == 1
    assert "no journal with a genesis" in capsys.readouterr().err
    assert old.exists()  # a store without a journal is not "nobody waits": no link is removed


@pytest.mark.parametrize("rounds, waiting", [(0, 3), (1, 0)])
def test_issue_reads_the_queue_from_the_journal_copy_in_the_store(bucket, site, tmp_path, capsys, rounds, waiting):
    old, copy = old_link(site), tmp_path / "job" / "copy"
    controller(tmp_path / "controller", 3, rounds)
    keys = bucket.keys()
    # a host without the controller; --journal-from-store is accepted and changes nothing
    args = ["issue", "--links-url", URL, "--state-dir", str(copy), "--site-dir", str(site[0]), "--no-activation-check", "--journal-from-store"]
    for _ in range(2):  # the second run finds its local copy up to date
        assert links.main(args) == 0  # the simulation's hotkeys are not SS58 addresses: they wait, but cannot get a link
        assert f"issued 0 of {waiting} waiting hotkeys (not activated 0, not Ed25519 {waiting})" in capsys.readouterr().out
    assert (copy / JOURNAL_FILE).read_bytes() == (tmp_path / "controller" / JOURNAL_FILE).read_bytes()
    assert bucket.keys() == keys and not old.exists()  # the read wrote nothing to the store


def test_a_local_copy_of_another_journal_stops_the_run(bucket, site, tmp_path, monkeypatch, capsys):
    args = ["issue", "--links-url", URL, "--state-dir", str(tmp_path / "copy"), "--site-dir", str(site[0]), "--no-activation-check"]
    controller(tmp_path / "smoke", 3, 0)
    assert links.main(args) == 0
    before = (site[0] / "ff" / "links.json").read_bytes()
    monkeypatch.setenv("SN120_R2_PREFIX", "production/")  # the same job directory, now with the settings of another store
    controller(tmp_path / "production", 2, 0)
    assert links.main(args) == 1
    assert "differ" in capsys.readouterr().err and (site[0] / "ff" / "links.json").read_bytes() == before


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


# ------------------------------------------------------------ the store token from a file, and the unit files

V1_FILE = f"""# the v1 link job's credentials file: KEY=VALUE lines, read in place
R2_ACCOUNT_ID=not-used
R2_ACCESS_KEY_ID=fake-id
R2_SECRET_ACCESS_KEY="{SECRET}"
"""
SYSTEMD = Path(links.__file__).resolve().parents[1] / "systemd"


def test_the_store_token_comes_from_a_v1_credentials_file_in_place(bucket, queue, site, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SN120_R2_ACCESS_KEY_ID")
    monkeypatch.delenv("SN120_R2_SECRET_ACCESS_KEY")
    file = tmp_path / "r2.env"
    file.write_text(V1_FILE)
    args = ["issue", "--links-url", URL, "--state-dir", str(tmp_path / "copy"), "--site-dir", str(site[0]), "--no-activation-check"]
    assert links.main(args) == 1 and "SN120_R2_ACCESS_KEY_ID" in capsys.readouterr().err  # without the file there is no token
    assert links.main(args + ["--credentials-file", str(file)]) == 0
    printed = capsys.readouterr()
    assert SECRET not in printed.out + printed.err and "fake-id" not in printed.out + printed.err
    assert "SN120_R2_SECRET_ACCESS_KEY" not in os.environ and file.read_text() == V1_FILE  # nothing is exported, nothing is copied
    doc = json.loads((site[0] / "ff" / "links" / f"{A.hotkey}.json").read_text())
    link = links.open_link(doc, A.hotkey, A.seed, int(time.time()))
    assert "X-Amz-Credential=fake-id" in link["put_url"] and put(link["put_url"], b"{}", link["put_headers"]) == 200

    file.write_text(f"export SN120_R2_ACCESS_KEY_ID = 'fake-id'\nSN120_R2_SECRET_ACCESS_KEY={SECRET}\n")  # the v2 names work too
    assert links.store_token(file) == {"SN120_R2_ACCESS_KEY_ID": "fake-id", "SN120_R2_SECRET_ACCESS_KEY": SECRET}


@pytest.mark.parametrize("text, missing", [("R2_ACCESS_KEY_ID=fake-id\n", "R2_SECRET_ACCESS_KEY"), (f"# R2_ACCESS_KEY_ID=x\nR2_SECRET_ACCESS_KEY={SECRET}\n", "R2_ACCESS_KEY_ID")])
def test_a_credentials_file_without_a_token_part_stops_the_run_and_shows_no_value(bucket, queue, site, tmp_path, capsys, text, missing):
    file = tmp_path / "r2.env"
    file.write_text(text)
    args = ["issue", "--links-url", URL, "--state-dir", str(tmp_path / "copy"), "--site-dir", str(site[0]), "--no-activation-check"]
    assert links.main(args + ["--credentials-file", str(file)]) == 1  # the token in the environment does not fill the gap
    error = capsys.readouterr().err
    assert missing in error and SECRET not in error and "fake-id" not in error and not (site[0] / "ff").exists()
    assert links.main(args + ["--credentials-file", str(tmp_path / "missing.env")]) == 1
    assert "cannot read the credentials file" in capsys.readouterr().err


def unit_lines(name: str) -> list[str]:
    return [line for line in (SYSTEMD / name).read_text().splitlines() if line and not line.startswith("#")]


def test_the_unit_runs_the_job_as_the_operator_installs_it(bucket, queue, site, tmp_path, monkeypatch, capsys):
    file = tmp_path / "v1" / "r2-direct.env"
    file.parent.mkdir()
    file.write_text(V1_FILE)
    blanks = {
        "LINKS_R2_ENDPOINT": os.environ["SN120_R2_ENDPOINT"],
        "LINKS_R2_BUCKET": os.environ["SN120_R2_BUCKET"],
        "LINKS_STATE_DIR": str(tmp_path / "links-job"),
        "LINKS_CREDENTIALS_FILE": str(file),
        "LINKS_SITE_DIR": str(site[0]),
        "LINKS_PUBLIC_URL": URL,
        "LINKS_NETWORK": "finney",
        "LINKS_PYTHON": "python",
        "LINKS_USER": "operator",
    }
    text = (SYSTEMD / "sn120-v2-links.service").read_text()
    lines = unit_lines("sn120-v2-links.service")
    assert set(re.findall(r"LINKS_[A-Z0-9_]+", "\n".join(lines))) == set(blanks) == set(re.findall(r"^#   (LINKS_[A-Z0-9_]+)", text, re.M))
    for blank in sorted(blanks, key=len, reverse=True):
        lines = [line.replace(blank, blanks[blank]) for line in lines]
    for name in [name for name in os.environ if name.startswith("SN120_R2_")]:
        monkeypatch.delenv(name)  # systemd starts the job with the unit's variables only
    for line in lines:
        if line.startswith("Environment="):
            monkeypatch.setenv(*line.removeprefix("Environment=").split("=", 1))
    assert sorted(name for name in os.environ if name.startswith("SN120_R2_")) == ["SN120_R2_BUCKET", "SN120_R2_ENDPOINT"]
    assert not any(line.startswith("EnvironmentFile=") for line in lines) and "UMask=0022" in lines and "Type=oneshot" in lines
    (command,) = [line.removeprefix("ExecStart=").split() for line in lines if line.startswith("ExecStart=")]
    assert command[:3] == ["python", "-m", "sn120_v2.links"] and "--loop" not in command  # one run; the timer makes the next

    chain_with(monkeypatch, rows=[(A.hotkey, [(A.activation(), 7)])])
    before = os.umask(0o022)  # the unit's UMask
    try:
        assert links.main(command[3:]) == 0
    finally:
        os.umask(before)
    assert "issued 1 of 2 waiting hotkeys (not activated 1" in capsys.readouterr().out
    written = site[0] / "ff" / "links" / f"{A.hotkey}.json"
    assert written.stat().st_mode & 0o777 == 0o644 and (site[0] / "ff" / "links.json").stat().st_mode & 0o777 == 0o644  # the web server can read them
    link = links.open_link(json.loads(written.read_text()), A.hotkey, A.seed, int(time.time()))
    assert f"/{os.environ['SN120_R2_BUCKET']}/private/ff/v1/miners/{A.hotkey}/batch.json?" in link["put_url"]  # no prefix: the key that v1 used

    timer = unit_lines("sn120-v2-links.timer")
    assert "Unit=sn120-v2-links.service" in timer and "OnCalendar=*-*-* *:10,40:00" in timer  # two runs an hour
    assert 3600 // 2 == links.REISSUE_S  # the index publishes this number, so the timer must keep it
