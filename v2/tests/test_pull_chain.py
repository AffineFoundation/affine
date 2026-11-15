"""The public pull (``v2/site/pull.py``) on a store whose box keeps the folded weight files (``SN120_HEAD_PUBLISH=chain``).

The pull runs as a command against an S3 fake (moto, over HTTP on localhost).
The head is real and tiny (``sim_head.py``), so a reader's rebuild through
the links of the pull is the real arithmetic. No test counts on a fixed head
or on a fixed count of past folds.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path

import boto3
import pytest
from moto.server import ThreadedMotoServer

torch = pytest.importorskip("torch")

from sn120_v2.sim_head import ChainWorld, is_weight  # noqa: E402 - after the skip guard

from sn120_v2 import headfold, layout, replicate  # noqa: E402
from sn120_v2.gpu.resident import hash_tree  # noqa: E402
from sn120_v2.headchain import CHAIN, FIXED_ORDER, FULL, MATMUL, heads_of  # noqa: E402
from sn120_v2.publisher import render  # noqa: E402
from sn120_v2.r2 import R2Store  # noqa: E402
from sn120_v2.sim import sim_params  # noqa: E402

PULL = Path(__file__).parents[1] / "site" / "pull.py"
URL, LINKS = "https://site.example", "public/read_links.json"
SECRET = "fake-secret-value-never-shown"
PAST = (0, 2)  # the count of kings that were folded before the switch to chain mode


@pytest.fixture(scope="module")
def endpoint():
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def env(endpoint):
    """The store settings of the pull, for a new empty bucket."""
    env = {"SN120_R2_ENDPOINT": endpoint, "SN120_R2_BUCKET": f"b-{uuid.uuid4().hex}", "SN120_R2_ACCESS_KEY_ID": "fake-id", "SN120_R2_SECRET_ACCESS_KEY": SECRET}
    raw = boto3.client("s3", endpoint_url=endpoint, region_name="auto", aws_access_key_id="fake-id", aws_secret_access_key=SECRET)
    raw.create_bucket(Bucket=env["SN120_R2_BUCKET"], CreateBucketConfiguration={"LocationConstraint": "auto"})
    return env


def pull(env: dict, site: Path) -> subprocess.CompletedProcess:
    """One run of the pull as a process. Its store settings are ``env`` alone."""
    site.mkdir(exist_ok=True)
    ours = {name: value for name, value in os.environ.items() if not name.startswith(("SN120_", "R2_"))}
    done = subprocess.run([sys.executable, str(PULL), "--site-dir", str(site), "--public-url", URL], capture_output=True, text=True, env={**ours, **env})
    assert "X-Amz" not in done.stdout + done.stderr and SECRET not in done.stdout + done.stderr  # no signed link in a log
    return done


@pytest.fixture
def world(tmp_path) -> ChainWorld:
    return ChainWorld(tmp_path / "world", sim_params())


def sync(world: ChainWorld, env: dict) -> R2Store:
    """Make the bucket equal to the store of the world, with the two public files of its last snapshot."""
    world.wait()
    store = R2Store.from_env(env, attempts=1)
    keys = set(world.store.list(""))
    for key in keys:
        store.put(key, world.store.get(key))
    for key in set(store.list("")) - keys:
        store.delete(key)
    mining, llms = render(json.loads(json.dumps(world.publisher.snapshots[-1])), URL)
    store.put(layout.MINING_JSON, mining)
    store.put(layout.LLMS_TXT, llms)
    return store


def read(site: Path) -> tuple[dict, dict]:
    return json.loads((site / LINKS).read_bytes()), json.loads((site / layout.MINING_JSON).read_bytes())


def linked(group: dict) -> dict[str, str]:
    """``{file name: sha256}`` of one file set of the links file. Each link is read, and it gives the bytes that the row names."""
    for row in group["files"]:
        with urllib.request.urlopen(row["url"]) as reply:
            data = reply.read()
        assert hashlib.sha256(data).hexdigest() == row["sha256"] and len(data) == row["size"]
    assert group["bytes"] == sum(row["size"] for row in group["files"])
    return {row["name"]: row["sha256"] for row in group["files"]}


@pytest.mark.parametrize("past", PAST)
def test_a_head_that_is_not_a_checkpoint_gets_the_links_of_the_newest_checkpoint_and_of_the_loras_after_it(world, env, tmp_path, past, capsys):
    for _ in range(past):
        world.king()
    start = world.head  # the last head whose full files are in the store
    world.start(CHAIN)
    heads = [world.king(), world.king()]
    store, site = sync(world, env), tmp_path / "site"
    assert not [key for key in store.list(layout.public_base_prefix(heads[-1].head.base_hash)) if is_weight(key)], "the store has no weight file of the head"
    done = pull(env, site)
    assert done.returncode == 0, done.stderr
    links, mining = read(site)
    chain = mining["head_chain"]
    assert links["head_version_id"] == heads[-1].head.version_id == chain["head_version_id"]
    assert links["base"] is None and links["master"] is None and links["loras"] == []
    assert [point["version_id"] for point in links["checkpoints"]] == [start.head.version_id] == [chain["start"]["version_id"]]
    assert linked(links["checkpoints"][0]["base"]) == start.base_files and linked(links["checkpoints"][0]["master"]) == start.master_files
    steps = chain["steps"]
    assert [step["kernel"] for step in steps] == [FIXED_ORDER] * 2 and [step["version_id"] for step in steps] == [head.head.version_id for head in heads]
    assert [group["id"] for group in links["chain_loras"]] == [lora["lora_hash"] for step in steps for lora in step["loras"]]
    assert [linked(group) for group in links["chain_loras"]] == [lora["files"] for step in steps for lora in step["loras"]]
    assert "python -m sn120_v2.headfold rebuild" in links["rebuild"]
    whole = json.loads((site / layout.HEAD_CHAIN).read_bytes())
    assert (site / layout.HEAD_CHAIN).read_bytes() == store.get(layout.HEAD_CHAIN) and chain["file"] == layout.HEAD_CHAIN
    assert [step["kernel"] for step in whole["steps"]] == [MATMUL] * past + [FIXED_ORDER] * 2, "the whole chain, from the launch model"

    # A reader with the two files of the site and no other access makes the base of the head, by the rebuild command and by ``fetch``.
    files = ["--chain", str(site / layout.MINING_JSON), "--links", str(site / LINKS)]
    assert headfold.main(["rebuild", *files, "--work", str(tmp_path / "reader")]) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["version_id"] == heads[-1].head.version_id and built["steps"] == 2
    assert hash_tree(Path(built["base_dir"])) == heads[-1].base_files and hash_tree(Path(built["master_dir"])) == heads[-1].master_files
    contract = replicate.Contract.load(str(site / layout.MINING_JSON))
    fetched = replicate.fetch(contract, tmp_path / "miner", links_source=str(site / LINKS))
    assert fetched["replayed_folds"] == 2 and hash_tree(Path(fetched["dir"])) == heads[-1].base_files
    assert replicate.fetch(contract, tmp_path / "miner", links_source=str(site / LINKS))["fetched"] == 0, "a second fetch finds the base"


def test_from_the_marker_of_the_head_the_rebuild_command_works_with_the_mining_json_of_before_the_marker(world, env, tmp_path, capsys):
    world.start(CHAIN)
    head = world.king()
    site = tmp_path / "site"
    world.controller.head.checkpoint(world.state)  # the files of the head, then its marker; the box has not published again
    sync(world, env)
    done = pull(env, site)
    assert done.returncode == 0, done.stderr
    links, mining = read(site)
    chain = mining["head_chain"]
    assert chain["start"]["version_id"] != head.head.version_id and chain["steps"][-1]["checkpoint"] is False, "mining.json is the one of before the marker"
    assert [point["version_id"] for point in links["checkpoints"]] == [head.head.version_id] and links["chain_loras"] == []
    capsys.readouterr()
    assert headfold.main(["rebuild", "--chain", str(site / layout.MINING_JSON), "--links", str(site / LINKS), "--work", str(tmp_path / "miner")]) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["version_id"] == built["started_from"] == head.head.version_id and built["steps"] == 0
    assert hash_tree(Path(built["base_dir"])) == head.base_files and hash_tree(Path(built["master_dir"])) == head.master_files


def test_after_the_marker_of_the_head_the_links_have_the_form_of_a_store_with_each_head_in_full(world, env, tmp_path):
    world.start(CHAIN)
    head = world.king()
    site = tmp_path / "site"
    sync(world, env)
    assert pull(env, site).returncode == 0 and read(site)[0]["base"] is None
    world.controller.head.checkpoint(world.state)  # the ``checkpoint`` command: the files of the head, then its marker
    sync(world, env)
    done = pull(env, site)
    assert done.returncode == 0, done.stderr
    links, mining = read(site)
    assert links["base"]["id"] == head.head.base_hash and links["master"]["id"] == head.head.base_fp32_master_digest
    assert linked(links["base"]) == head.base_files and linked(links["master"]) == head.master_files and "note" in links["master"]
    assert [point["version_id"] for point in links["checkpoints"]] == [head.head.version_id] and links["chain_loras"] == []
    assert heads_of(json.loads((site / layout.HEAD_CHAIN).read_bytes()))[-1]["checkpoint"] is True, "the chain file of the site shows the marker"
    contract = replicate.Contract.load(str(site / layout.MINING_JSON))
    fetched = replicate.fetch(contract, tmp_path / "miner", links_source=str(site / LINKS))
    assert "replayed_folds" not in fetched and fetched["fetched"] == len(head.base_files), "the base is a download: no fold on the reader's machine"
    assert hash_tree(Path(fetched["dir"])) == head.base_files


def test_a_store_with_each_head_in_full_has_the_links_of_before_and_a_chain_file_of_another_head_is_not_used(world, env, tmp_path):
    first = world.king()
    site = tmp_path / "site"
    sync(world, env)
    assert pull(env, site).returncode == 0
    links, mining = read(site)
    assert "head_chain" not in mining and not {"checkpoints", "chain_loras", "rebuild"} & set(links) and not (site / layout.HEAD_CHAIN).exists()
    assert links["base"]["id"] == first.head.base_hash and links["master"]["id"] == first.head.base_fp32_master_digest

    world.start(CHAIN)
    world.king()
    sync(world, env)
    assert pull(env, site).returncode == 0 and (site / layout.HEAD_CHAIN).is_file() and read(site)[0]["base"] is None
    world.start(FULL)  # back to full mode: the chain file of the store stays at the head of the last fold in chain mode
    last = world.king()
    store = sync(world, env)
    assert json.loads(store.get(layout.HEAD_CHAIN))["head_version_id"] != last.head.version_id
    done = pull(env, site)
    assert done.returncode == 0, done.stderr
    links, mining = read(site)
    assert "head_chain" not in mining and not {"checkpoints", "chain_loras", "rebuild"} & set(links)
    assert linked(links["base"]) == last.base_files and linked(links["master"]) == last.master_files
    assert not (site / layout.HEAD_CHAIN).exists(), "the copy of the chain of an older head is removed"


def test_a_head_with_no_weight_file_and_no_chain_of_its_own_fails_the_step_and_keeps_the_last_good_links(world, env, tmp_path):
    world.start(CHAIN)
    world.king()
    site = tmp_path / "site"
    store = sync(world, env)
    assert pull(env, site).returncode == 0
    good, chain = (site / LINKS).read_bytes(), store.get(layout.HEAD_CHAIN)
    world.king()
    store = sync(world, env)
    store.put(layout.HEAD_CHAIN, chain)  # as between two writes of the controller: the pointer is new, the chain file is not
    done = pull(env, site)
    assert done.returncode == 1 and "FAILED head and read links" in done.stderr
    assert (site / LINKS).read_bytes() == good and (site / layout.HEAD_CHAIN).read_bytes() == chain
    store.put(layout.HEAD_CHAIN, b'{"schema": "other"}')  # a chain file that this code does not read: the same refusal, no guess
    done = pull(env, site)
    assert done.returncode == 1 and "VerifyError" in done.stderr and (site / LINKS).read_bytes() == good
