"""The public pull (``v2/site/pull.py``), run as a command against an S3 fake (moto, over HTTP on localhost)."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import subprocess
import sys
import time
import importlib.util
import urllib.request
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import boto3
import pytest
from moto.server import ThreadedMotoServer

from sn120_v2 import layout, links, sim
from sn120_v2.errors import ConfigError, InfraError
from sn120_v2.publisher import StorePublisher, render
from sn120_v2.public import winner_stats_from_snapshot
from sn120_v2.r2 import R2Store
from sn120_v2.r2_names import PLAIN_NAMES, store_env
from sn120_v2.types import HeadManifest

PULL = Path(__file__).parents[1] / "site" / "pull.py"
URL = "https://site.example"
SECRET = "fake-secret-value-never-shown"
LINKS, INDEX = "public/read_links.json", "public/pull_index.json"
BUILT = {"index.html", "v1.html", "pool.html", "pool/index.html", "pool.js", "site.css", "site.js", "site-data.json", "favicon.svg", "network-haffer.ttf", "network-mono.ttf",
         "generator.html", "generator/index.html", "generator.js", "generator.css", "generator-guide.md", "generator/generator-guide.md", "generator-guide.html", "generator-guide/index.html"}
FOREIGN = {  # what other software keeps under public/ of the same bucket, and keys that only look like v2's
    "public/41/current.json": b'{"epoch": 41}',
    "public/streams/main/current.json": b'{"stream": "main"}',
    "public/heads/not-a-head.json": b'{"epoch": 41}',
    "public/heads/deep/x.json": b"{}",
    "public/rounds/r000001/extra.bin": b"x",
    "public/rounds/r000001/deep/round.json": b"{}",
    "public/irt/unnamed.json": b"{}",
    "public/other.json": b"{}",
}


@pytest.fixture(scope="module")
def endpoint():
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def env(endpoint):
    env = {"SN120_R2_ENDPOINT": endpoint, "SN120_R2_BUCKET": f"b-{uuid.uuid4().hex}",
           "SN120_R2_ACCESS_KEY_ID": "fake-id", "SN120_R2_SECRET_ACCESS_KEY": SECRET}
    raw = boto3.client("s3", endpoint_url=endpoint, region_name="auto", aws_access_key_id="fake-id", aws_secret_access_key=SECRET)
    raw.create_bucket(Bucket=env["SN120_R2_BUCKET"], CreateBucketConfiguration={"LocationConstraint": "auto"})
    return env


@pytest.fixture
def store(env, tmp_path):
    """The bucket after one applied round and with a second round open: every object of the simulation, and the two public files."""
    world = sim.SimWorld(tmp_path / "world", sim.sim_params())
    for uid, quality in ((3, 0.0), (4, 0.3), (6, None), (7, -0.2)):
        world.register(uid, quality)
    assert world.round().outcome == "applied"
    world.register(8, 0.0)
    world.controller.open_round(world.clock(), "entropy")
    store = R2Store.from_env(env, attempts=1)
    for key in world.store.list(""):
        store.put(key, world.store.get(key))
    snapshot = json.loads(json.dumps(world.publisher.snapshots[-1]))
    StorePublisher(store, URL + "/").publish(snapshot)  # rich standalone object, transport-free mining.json
    for key, data in FOREIGN.items():
        store.put(key, data)
    return store


def command(env: dict, script: str, *args: str) -> subprocess.CompletedProcess:
    """One command of ``v2/site`` as a process. Its store settings are ``env`` alone: no ``SN120_*`` or ``R2_*`` name of the caller's shell reaches it."""
    ours = {name: value for name, value in os.environ.items() if not name.startswith(("SN120_", "R2_"))}
    done = subprocess.run([sys.executable, str(PULL.with_name(script)), *args], capture_output=True, text=True, env={**ours, **env})
    assert "X-Amz" not in done.stdout + done.stderr and SECRET not in done.stdout + done.stderr  # no signed link in a log
    return done


def pull(env: dict, site: Path, *more: str) -> subprocess.CompletedProcess:
    if site.parent.is_dir():
        site.mkdir(exist_ok=True)
    return command(env, "pull.py", "--site-dir", str(site), "--public-url", URL, *more)


def tree(site: Path) -> dict[str, bytes]:
    return {str(path.relative_to(site)): path.read_bytes() for path in sorted(site.rglob("*")) if path.is_file()}


def times(site: Path) -> dict[str, int]:
    return {str(path.relative_to(site)): path.stat().st_mtime_ns for path in site.rglob("*") if path.is_file()}


def test_one_run_copies_the_keys_of_the_layout_writes_the_links_and_builds_the_site(env, store, tmp_path):
    site = tmp_path / "site"
    done = pull(env, site)
    assert done.returncode == 0, done.stderr
    assert "site built" in done.stdout and done.stderr.count("\n") == 1  # one note: the file under heads/ that is not a head
    files = tree(site)
    mining = json.loads(files[layout.MINING_JSON])
    small = [key for key in store.list("public/") if key.startswith(("public/heads/", "public/rounds/")) and key not in FOREIGN]
    tables = {layout.irt_snapshot(mining["irt_snapshot_id"])} | {layout.irt_snapshot(row["irt_snapshot_id"]) for row in mining["history"]}
    copied = {layout.MINING_JSON, layout.LLMS_TXT, layout.WINNER_STATS, layout.CURRENT, *small, *tables, layout.ROLLOUT_INDEX, "public/rollouts/SCHEMA.md"}
    assert len(small) == 4 and set(files) == copied | {LINKS, INDEX, "llms.txt", "mining.json", layout.WINNER_STATS_ALIAS} | BUILT
    for key in copied - {layout.ROLLOUT_INDEX, "public/rollouts/SCHEMA.md"}:  # mirrored objects are byte-identical
        assert files[key] == store.get(key)
    assert json.loads(files[layout.ROLLOUT_INDEX]) == {"schema": "sn120-v2-rollouts-index-2", "backfill": None, "bench": [], "gate": [], "schema_doc": "SCHEMA.md"}
    assert (site / "public/rollouts/SCHEMA.md").read_bytes() == (PULL.parent / "static/rollouts/SCHEMA.md").read_bytes()
    assert files["llms.txt"] == files[layout.LLMS_TXT] and files["mining.json"] == files[layout.MINING_JSON]
    assert files[layout.WINNER_STATS_ALIAS] == files[layout.WINNER_STATS] == store.get(layout.WINNER_STATS_ALIAS)
    assert json.loads(files["site-data.json"])["winner_stats_url"] == f"{URL}/{layout.WINNER_STATS}"
    assert [row["uid"] for row in json.loads(files[layout.WINNER_STATS])["winners"]] == [4]
    assert set(json.loads(files[INDEX])["files"]) == copied - {"public/rollouts/SCHEMA.md"} | {LINKS}
    assert not list(site.rglob(".tmp-*")) and "site-data.json" in files["index.html"].decode()

    manifest = HeadManifest.from_json(files[layout.CURRENT])
    links = json.loads(files[LINKS])
    assert links["schema"] == "sn120-v2-read-links-1" and links["head_version_id"] == manifest.head.version_id
    assert links["ttl_s"] == 86400 and links["expires_at"] == links["generated_at"] + 86400
    assert abs(links["generated_at"] - time.time()) < 120
    sets = [(links["base"], manifest.base_files), (links["master"], manifest.master_files)]
    sets += [(row, manifest.lora_files[row["id"]]) for row in links["loras"]]
    assert [row["id"] for row in links["loras"]] == [ref.lora_hash for ref in manifest.head.loras]
    assert links["base"]["id"] == manifest.head.base_hash and links["master"]["id"] == manifest.head.base_fp32_master_digest
    for row, expected in sets:  # a link gives the bytes that the head manifest names
        assert {item["name"]: item["sha256"] for item in row["files"]} == expected and expected
        for item in row["files"]:
            with urllib.request.urlopen(item["url"]) as response:
                data = response.read()
            assert hashlib.sha256(data).hexdigest() == item["sha256"] and len(data) == item["size"]
            assert "X-Amz-Expires=86400" in item["url"] and item["key"].startswith(("public/bases/", "public/masters/", "public/loras/"))
        assert row["bytes"] == sum(item["size"] for item in row["files"])

    before = times(site)
    time.sleep(1.1)
    again = pull(env, site, "--ttl-s", "600")
    assert again.returncode == 0 and "wrote 1," in again.stdout  # only the links file is new
    after = times(site)
    assert {name for name in after if after[name] != before[name]} == {LINKS}
    new = json.loads((site / LINKS).read_text())
    assert new["generated_at"] > links["generated_at"] and new["expires_at"] == new["generated_at"] + 600


def test_the_run_invokes_winner_stats_immediately_after_the_contract(env, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(PULL.parent))
    spec = importlib.util.spec_from_file_location("sn120_site_pull_test", PULL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    called = []
    monkeypatch.setattr(module.Pull, "step", lambda self, name, work: called.append((name, work)))
    monkeypatch.setattr(module.build, "write", lambda *args: None)
    assert module.run(store=None, site=tmp_path, public_url=URL, ttl_s=1, keep_rounds=2, max_bytes=1,
                      v1_history=None, now=1) == 0
    assert called[1][0] == "llms.txt and mining.json" and called[2][0] == "winner stats"
    assert called[2][1].__func__ is module.Pull.winner_stats
    assert layout.WINNER_STATS in module.FIXED


def test_enriched_stats_verify_and_forgery_is_refused(env, store, tmp_path):
    from sn120_v2.canonical import sha256_hex
    from sn120_v2.envs.math_pool import load_pool
    from sn120_v2.envs.reserve import Catalog, empty_manifest

    mining = json.loads(store.get(layout.MINING_JSON))
    mining["winner_stats"] = json.loads(store.get(layout.WINNER_STATS))
    winner = mining["winner_stats"]["winners"][0]
    task = load_pool().task(load_pool().task_ids[0])
    raw = json.dumps({"version": "affine-batch-v2", "hotkey": winner["hotkey"], "groups": [
        {"prompt": task.batch_prompt(), "rollouts": [{"output": "work \\boxed{1}", "reward": 1.0}]},
        {"prompt": "custom public prompt", "rollouts": [{"output": "no box", "reward": 0.0}]},
    ]}).encode()
    winner["batch_sha256"] = sha256_hex(raw)
    crowned = next(entry for entry in mining["history"][0]["entries"] if entry["status"] == "winner")
    crowned["batch_sha256"] = winner["batch_sha256"]
    store.put(layout.winner_batch(winner["round_id"]), raw)
    store.put(layout.RESERVE_MANIFEST, empty_manifest())
    store.put(layout.RESERVE_WITHDRAWN, b"")
    mining.pop("snapshot_digest", None)
    mining.pop("contract", None)
    pool = Catalog(load_pool(), load_pool().__class__((), "no-reserve-fixture"))
    StorePublisher(store, URL + "/", prompt_pool=pool).publish(mining)
    stats = json.loads(store.get(layout.WINNER_STATS))
    assert stats["winners"][0]["group_count"] == 2 and stats["winners"][0]["boxed_frac"] == 0.5
    site = tmp_path / "fresh"
    done = pull(env, site)
    assert done.returncode == 0, done.stderr
    assert "winner enrichment check deferred" in done.stderr
    assert (site / layout.WINNER_STATS).read_bytes() == store.get(layout.WINNER_STATS_ALIAS) == store.get(layout.WINNER_STATS)
    assert (site / layout.RESERVE_MANIFEST).read_bytes() == empty_manifest()

    good_stats = (site / layout.WINNER_STATS).read_bytes()
    stats["winners"] = [dict(stats["winners"][0], group_count=999)]
    store.put(layout.WINNER_STATS, json.dumps(stats).encode())
    store.put(layout.WINNER_STATS_ALIAS, store.get(layout.WINNER_STATS))
    done = pull(env, site)
    assert done.returncode == 1 and "FAILED winner stats:" in done.stderr and "group_count" in done.stderr
    assert (site / layout.WINNER_STATS).read_bytes() == good_stats
    assert (site / layout.WINNER_STATS_ALIAS).read_bytes() == good_stats


def test_enriched_stats_fail_after_the_mirror_if_the_catalog_is_unavailable(env, store, tmp_path):
    stats = json.loads(store.get(layout.WINNER_STATS))
    store.put(layout.WINNER_STATS, json.dumps(stats | {"winners": [dict(stats["winners"][0], group_count=1)]}).encode())
    done = pull(env, tmp_path / "fresh")
    assert done.returncode == 1 and "FAILED winner stats enriched fields" in done.stderr


def test_only_keys_that_the_layout_names_are_read(env, store, tmp_path):
    site = tmp_path / "site"
    done = pull(env, site)
    assert done.returncode == 0 and "public/heads/not-a-head.json is not a v2 head manifest" in done.stderr
    files = tree(site)
    assert not set(files) & set(FOREIGN) and not (site / "public" / "41").exists() and not (site / "public" / "streams").exists()
    for zone in ("bases", "masters", "loras"):  # no weight file is copied
        assert not (site / "public" / zone).exists()
    source = PULL.read_text()
    assert re.findall(r"\.list\(([^)]*)\)", source) == ["HEADS", "ROUNDS", "layout.ROLLOUT_BENCH_PREFIX", "layout.ROLLOUT_GATE_PREFIX"]  # the whole prefix is never listed

    store.put(layout.CURRENT, b'{"epoch": 41, "model": "v1"}')  # a pointer of other software at the key of the head pointer
    good = (site / layout.CURRENT).read_bytes()
    links = (site / LINKS).read_bytes()
    done = pull(env, site)
    assert done.returncode == 1 and "FAILED head and read links" in done.stderr
    assert (site / layout.CURRENT).read_bytes() == good and (site / LINKS).read_bytes() == links


def test_a_file_that_the_pull_did_not_write_is_never_replaced_or_removed(env, store, tmp_path):
    site = tmp_path / "site"
    theirs = {
        "ff/links.json": b'{"links": 1}', "ff/links/hk3.json": b'{"url": "x"}', "ff/status.json": b"v1 status",
        "k1.json": b"v1 k1", "scores.json": b"v1 scores", "public/operator-note.txt": b"keep",
        "public/rounds/r000001/round.json": b"not from the pull", "public/rounds/r000001/note.txt": b"keep",
    }
    replaced = {"index.html": b"v1 page", "mining.json": b"v1 mining", "llms.txt": b"v1 llms"}
    for name, data in {**theirs, **replaced}.items():
        (site / name).parent.mkdir(parents=True, exist_ok=True)
        (site / name).write_bytes(data)
    done = pull(env, site, "--keep-rounds", "2")
    assert done.returncode == 1 and "FAILED rounds: Refused: public/rounds/r000001/round.json" in done.stderr
    files = tree(site)
    assert {name: files[name] for name in theirs} == theirs
    assert files["mining.json"] == store.get(layout.MINING_JSON) and files["llms.txt"] == store.get(layout.LLMS_TXT)
    assert b"site-data.json" in files["index.html"] and layout.CURRENT in files and LINKS in files  # the other steps ran

    (site / "public/rounds/r000001/round.json").unlink()
    for number in range(2, 7):  # more rounds than --keep-rounds: the old ones go, and only the files of the pull
        store.put(layout.round_file(f"r{number:06d}"), b"{}")
        store.put(layout.winner_batch(f"r{number:06d}"), b"[]")
    assert pull(env, site, "--keep-rounds", "6").returncode == 0
    for name in ("public/rounds/r000003/note.txt", "public/rounds/r000004/round.json.bak"):
        (site / name).write_bytes(b"keep")
    done = pull(env, site, "--keep-rounds", "2")
    assert done.returncode == 0 and "removed 4," in done.stdout
    kept = sorted(path.name for path in (site / "public/rounds").iterdir())
    assert kept == ["r000001", "r000002", "r000003", "r000004", "r000005", "r000006"]  # mining.json names the first two
    assert sorted(tree(site / "public/rounds/r000002")) == ["batch.json", "round.json"]
    assert tree(site / "public/rounds/r000003") == {"note.txt": b"keep"} and tree(site / "public/rounds/r000004") == {"round.json.bak": b"keep"}
    assert {name: data for name, data in tree(site).items() if name in theirs and "round.json" not in name} == {
        name: data for name, data in theirs.items() if "round.json" not in name}

    (site / INDEX).write_text('{"schema": "other"}')
    before = tree(site)
    done = pull(env, site)
    assert done.returncode == 1 and "is not the index of this command" in done.stderr and tree(site) == before


def test_task_tables_and_large_files_keep_the_disk_bounded(env, store, tmp_path):
    site = tmp_path / "site"
    mining = json.loads(store.get(layout.MINING_JSON))
    named = {mining["irt_snapshot_id"]} | {row["irt_snapshot_id"] for row in mining["history"]}
    for table in ("t1", "t2", "t3"):  # each round names a new task table; the digest stays, so llms.txt still matches
        store.put(layout.irt_snapshot(table), table.encode())
        store.put(layout.MINING_JSON, json.dumps({**mining, "irt_snapshot_id": table}).encode())
        assert pull(env, site, "--keep-rounds", "3").returncode == 0
        time.sleep(0.02)
    names = {path.stem for path in (site / "public/irt").iterdir()}
    assert names == named | {"t2", "t3"} and len(names) == 3  # the tables that the rounds name, the current one, one older

    store.put(layout.winner_batch("r000002"), b"x" * (2 << 20))
    (site / "public/.tmp-killed").write_bytes(b"part of a file")
    os.utime(site / "public/.tmp-killed", (time.time() - 7200,) * 2)
    done = pull(env, site, "--max-file-mb", "1")
    assert done.returncode == 0 and "above --max-file-mb: not copied" in done.stderr
    assert not (site / layout.winner_batch("r000002")).exists() and not (site / "public/.tmp-killed").exists()


def test_a_store_that_does_not_answer_keeps_the_last_good_files(env, store, tmp_path):
    site = tmp_path / "site"
    assert pull(env, site).returncode == 0
    good = tree(site)

    for bad in (["--ttl-s", "0"], ["--ttl-s", "700000"], ["--public-url", "site.example"], ["--keep-rounds", "1"]):
        assert pull(env, site, *bad).returncode == 2 and tree(site) == good
    (tmp_path / "v1.json").write_text('{"schema": "other"}')  # a bad v1 file fails its own step; the v2 site is still built
    done = pull(env, site, "--v1-history", str(tmp_path / "v1.json"))
    assert done.returncode == 1 and "FAILED v1 history" in done.stderr and "failed steps 1;" in done.stdout
    assert {name: data for name, data in tree(site).items() if name != LINKS} == {name: data for name, data in good.items() if name != LINKS}
    good = tree(site)
    done = pull({**env, "SN120_R2_SECRET_ACCESS_KEY": ""}, site)
    assert done.returncode == 1 and "nothing written" in done.stderr and tree(site) == good
    done = pull({**env, "SN120_R2_BUCKET": "no-such-bucket"}, site)
    assert done.returncode == 1 and "nothing written" in done.stderr and tree(site) == good
    done = pull(env, tmp_path / "site" / "missing" / "deeper")
    assert done.returncode == 1 and not (site / "missing").exists()  # a site directory is never made

    manifest = HeadManifest.from_json(good[layout.CURRENT])
    name = sorted(manifest.base_files)[0]
    weight = store.get(layout.public_base_file(manifest.head.base_hash, name))
    store.delete(layout.public_base_file(manifest.head.base_hash, name))  # a link to a missing file is never published
    store.put(layout.BENCHMARKS, b"not json")
    done = pull(env, site)
    assert done.returncode == 1 and "FAILED head and read links: missing in the store" in done.stderr
    assert (site / LINKS).read_bytes() == good[LINKS] and (site / layout.BENCHMARKS).read_bytes() == b"not json"
    store.put(layout.public_base_file(manifest.head.base_hash, name), weight)

    store.put(layout.LLMS_TXT, b"# llms.txt of another snapshot")  # half of a publish: the pair is refused
    done = pull(env, site)
    assert done.returncode == 1 and "not of the same snapshot" in done.stderr
    for path in ("llms.txt", "mining.json", layout.LLMS_TXT, layout.MINING_JSON):
        assert (site / path).read_bytes() == good[path]
    store.put(layout.LLMS_TXT, good["llms.txt"])
    store.delete(layout.BENCHMARKS)
    assert pull(env, site).returncode == 0  # the benchmark file that was copied stays; the site is built again


def test_the_reference_models_of_the_v1_history_are_in_the_table_from_the_first_run_and_stay_under_the_stores_table(env, store, tmp_path):
    def cell(score: float) -> dict:
        return {"score": score, "se": 0.01, "partial": False, "samples": 100, "expected": 100, "cap_rate": 0.0}

    site, history = tmp_path / "site", tmp_path / "v1-history.json"
    history.write_text(json.dumps({"schema": "sn120-v1-archive-1", "archived_at": 1.0, "kings": [], "turns": [], "bench": {
        "updated_at": 5, "columns": [{"id": "b1", "label": "B1", "in_avg": True}],
        "rows": [{"label": "Ref-8B", "kind": "reference", "status": "scored", "cells": {"b1": cell(0.75)}},
                 {"label": "E12 abcd", "kind": "king", "status": "scored", "cells": {"b1": cell(0.9)}}]}}))

    def rows() -> list:
        done = pull(env, site, "--v1-history", str(history))
        assert done.returncode == 0, done.stderr
        bench = json.loads((site / "site-data.json").read_text())["bench"]
        return [(row["label"].split()[0], row["kind"], row["status"], row["avg"]) for row in bench["rows"]]

    waiting = [("r000001", "king", "none", None), ("Ref-8B", "reference", "scored", 0.75), ("Genesis", "genesis", "none", None)]
    assert rows() == waiting  # the first run: the history is installed before the data is built
    assert not (site / layout.BENCHMARKS).exists() and layout.BENCHMARKS not in json.loads((site / INDEX).read_text())["files"]
    folded = next(line for line in json.loads(store.get(layout.MINING_JSON))["heads"] if line["folded"] and line["round_id"] == "r000001")
    store.put(layout.BENCHMARKS, json.dumps({  # the first table of the benchmark service: its own rows, no reference model
        "updated_at": 9, "benchmarks": [dict(id="b1", label="B1", suite="math", order=1, enabled=True)],
        "models": [dict(model_id="king-1", label="r000001 base", kind="king", status="evaluating", round_id="r000001",
                        head_version_id=folded["version_id"], crowned_at=folded["applied_at"]),
                   dict(model_id="genesis-1", label="genesis Launch", kind="genesis", status="queued", round_id="genesis")],
        "results": []}).encode())
    assert rows() == [("r000001", "king", "running", None), waiting[1], ("Genesis", "genesis", "queued", None)]
    assert layout.BENCHMARKS in json.loads((site / INDEX).read_text())["files"]


def test_bad_winner_stats_keep_the_last_good_mirror_and_site_data(env, store, tmp_path):
    site = tmp_path / "site"
    assert pull(env, site).returncode == 0
    good = (site / layout.WINNER_STATS).read_bytes()
    good_data = (site / "site-data.json").read_bytes()
    stats = json.loads(good)
    for bad in (stats | {"schema": "other"},
                stats | {"private": "pending/batch"},
                stats | {"winners": [dict(stats["winners"][0], uid=999)]},
                stats | {"winners": [dict(stats["winners"][0], z=999)]},
                stats | {"winners": [dict(stats["winners"][0], batch_url="https://other.example/public/rounds/r000001/batch.json")]},
                stats | {"winners": [dict(stats["winners"][0], train={"groups": ["private batch text"]})]}):
        store.put(layout.WINNER_STATS, json.dumps(bad).encode())
        done = pull(env, site)
        assert done.returncode == 1 and "FAILED winner stats" in done.stderr
        assert (site / layout.WINNER_STATS).read_bytes() == good
        assert (site / "site-data.json").read_bytes() == good_data


def test_a_legacy_store_without_winner_stats_gets_the_safe_public_projection(env, store, tmp_path):
    """The old mining.json has no winner train/counts. Never invent them when introducing the new mirror."""
    store.delete(layout.WINNER_STATS)
    # A legacy publisher's mining.json carries no winner-only transport, unlike today's in-memory fixture.
    mining = json.loads(store.get(layout.MINING_JSON))
    store.put(layout.MINING_JSON, json.dumps({key: value for key, value in mining.items() if key != "winner_stats"}).encode())
    site = tmp_path / "site"
    done = pull(env, site)
    assert done.returncode == 0, done.stderr
    stats = json.loads((site / layout.WINNER_STATS).read_bytes())
    assert stats["schema"] == "sn120-v2-winner-stats-1" and [row["uid"] for row in stats["winners"]] == [4]
    assert stats["winners"][0]["batch_url"] == f"{URL}/{layout.winner_batch(stats['winners'][0]['round_id'])}"
    assert all(stats["winners"][0][key] is None for key in ("batch_prompts", "batch_rollouts", "batch_format", "train"))
    assert json.loads((site / "site-data.json").read_text())["winner_stats_url"] == f"{URL}/{layout.WINNER_STATS}"


def test_a_round_with_no_result_is_on_the_site_after_one_run(env, tmp_path):
    """No training finished: the round has no result, and mining.json gives its entry no ``status``."""
    world = sim.SimWorld(tmp_path / "world", sim.sim_params())
    world.register(3, 0.0)
    world.register(4, 0.3)
    assert world.round().outcome == "applied"
    world.register(8, 0.0)
    world.trainer.fail_hotkeys = {"hk8"}
    assert world.round().outcome == "no_decision"
    store = R2Store.from_env(env, attempts=1)
    for key in world.store.list(""):
        store.put(key, world.store.get(key))
    snapshot = json.loads(json.dumps(world.publisher.snapshots[-1]))
    mining, llms = render(snapshot, URL)
    store.put(layout.MINING_JSON, mining)
    store.put(layout.LLMS_TXT, llms)
    store.put(layout.WINNER_STATS, json.dumps(winner_stats_from_snapshot(snapshot, URL)).encode())
    assert "status" not in json.loads(mining)["history"][-1]["entries"][0]
    site = tmp_path / "site"
    done = pull(env, site)
    assert done.returncode == 0 and "failed steps 0; site built" in done.stdout, done.stderr
    data = json.loads((site / "site-data.json").read_text())
    assert [(row["uid"], row["status"]) for row in data["history"] if row["round_id"] == "r000002"] == [(8, None)]
    assert [(row["round_id"], row["outcome"]) for row in data["rounds"]] == [("r000002", "no_decision"), ("r000001", "applied")]


def test_a_store_with_the_pointer_only_at_its_old_key_fails_one_step_until_the_controller_writes_the_new_key(env, store, tmp_path):
    """The head pointer moved from ``public/current.json``. The pull never reads the old key: at an empty prefix it is an object of v1."""
    site, old, pointer = tmp_path / "site", "public/current.json", store.get(layout.CURRENT)
    store.put(old, pointer)
    store.delete(layout.CURRENT)
    done = pull(env, site)
    assert done.returncode == 1 and f"pull: FAILED head and read links: missing in the store: '{layout.CURRENT}'" in done.stderr
    assert "failed steps 1;" in done.stdout
    files = tree(site)
    assert not {old, layout.CURRENT, LINKS} & set(files) and {layout.MINING_JSON, "llms.txt", "index.html"} <= set(files)
    store.put(layout.CURRENT, pointer)  # what the first pass of ``run`` does
    done = pull(env, site)
    assert done.returncode == 0 and tree(site)[layout.CURRENT] == pointer and old not in tree(site)

    units = PULL.parents[1] / "systemd"
    unit = (units / "sn120-v2-site-pull.service").read_text()
    start = next(line for line in unit.splitlines() if line.startswith("ExecStart="))
    flags = re.findall(r"--[\w-]+", start)
    assert "SN120_V2_DIR/site/pull.py" in start and flags and all(f'"{flag}"' in PULL.read_text() for flag in flags)
    assert set(re.findall(r"SN120_[A-Z0-9_]+", unit.split("[Unit]")[1])) == set(re.findall(r"^#   (SN120_[A-Z0-9_]+)", unit, re.M))
    assert "Unit=sn120-v2-site-pull.service" in (units / "sn120-v2-site-pull.timer").read_text()


class ReadOnlyToken(BaseHTTPRequestHandler):
    """The store as a token with object read only sees it: GET and HEAD go to the fake, every other call is denied."""

    def forward(self):
        upstream = http.client.HTTPConnection(*self.server.target)
        upstream.request(self.command, self.path, headers=dict(self.headers))
        answer = upstream.getresponse()
        body = answer.read()
        self.send_response(answer.status)
        for name, value in answer.getheaders():
            if name.lower() not in ("transfer-encoding", "connection", "content-length", "server", "date"):
                self.send_header(name, value)
        self.send_header("Content-Length", answer.getheader("Content-Length") or str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        upstream.close()

    def deny(self):
        self.server.denied.append(f"{self.command} {self.path}")
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = b'<?xml version="1.0" encoding="UTF-8"?><Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>'
        self.send_response(403)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_HEAD = forward
    do_PUT = do_POST = do_DELETE = do_PATCH = deny  # PUT: object, part, copy. POST: multipart, delete of many. DELETE.

    def log_message(self, *args):
        pass


@pytest.fixture
def read_only(endpoint):
    server = ThreadingHTTPServer(("127.0.0.1", 0), ReadOnlyToken)
    host, _, port = endpoint.removeprefix("http://").partition(":")
    server.target, server.denied = (host, int(port)), []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def test_the_whole_run_passes_with_a_token_that_cannot_write_and_a_file_with_the_plain_names(env, store, read_only, tmp_path):
    site, proxy = tmp_path / "site", f"http://127.0.0.1:{read_only.server_address[1]}"
    for number in range(2, 6):  # more rounds than --keep-rounds, so that the run also trims
        store.put(layout.round_file(f"r{number:06d}"), b"{}")
        store.put(layout.winner_batch(f"r{number:06d}"), b"[]")
    store.put(layout.BENCHMARKS, b"{}")
    token = {"R2_ACCOUNT_ID": "0123abcd", "R2_ACCESS_KEY_ID": "fake-id", "R2_SECRET_ACCESS_KEY": SECRET,
             "R2_BUCKET": env["SN120_R2_BUCKET"], "R2_ENDPOINT": proxy}  # the names of the operator's file; no SN120_R2_* name
    first = pull(token, site, "--keep-rounds", "4")
    second = pull(token, site, "--keep-rounds", "2")
    assert first.returncode == 0 and second.returncode == 0, first.stderr + second.stderr
    assert "removed 2," in second.stdout and read_only.denied == []  # the run made no write call, and no call was refused
    item = json.loads((site / LINKS).read_text())["base"]["files"][0]
    with urllib.request.urlopen(item["url"]) as response:  # a link signed with that token reads the file
        assert hashlib.sha256(response.read()).hexdigest() == item["sha256"] and item["url"].startswith(proxy)
    (site / "llms.txt").unlink()  # the reset tool's copy of llms.txt takes the same names and makes no write call either
    reset = command(token, "reset.py", "llms", "--site-dir", str(site))
    assert reset.returncode == 0 and (site / "llms.txt").read_bytes() == store.get(layout.LLMS_TXT) and read_only.denied == [], reset.stderr

    writer = R2Store.from_env({**env, "SN120_R2_ENDPOINT": proxy}, attempts=1)  # the same store does refuse a write
    for call in (lambda: writer.put("public/x", b"x"), lambda: writer.delete(layout.MINING_JSON), lambda: writer.copy(layout.MINING_JSON, "public/y")):
        with pytest.raises(InfraError):
            call()
    assert [line.split()[0] for line in read_only.denied] == ["PUT", "DELETE", "PUT"] and store.exists(layout.MINING_JSON)

    done = pull({name: value for name, value in token.items() if name not in ("R2_ENDPOINT", "R2_ACCOUNT_ID")}, site)
    assert done.returncode == 1 and "nothing written" in done.stderr  # a missing name is named, and no value is shown
    assert all(name in done.stderr for name in ("SN120_R2_ENDPOINT", "R2_ENDPOINT", "R2_ACCOUNT_ID")) and env["SN120_R2_BUCKET"] not in done.stderr


def test_reset_llms_reads_the_store_with_only_the_plain_names(env, store, tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "llms.txt").write_text("the old contract")
    token = {PLAIN_NAMES[name]: value for name, value in env.items()}  # the operator's file: no SN120_R2_* name
    done = command(token, "reset.py", "llms", "--site-dir", str(site))
    assert done.returncode == 0 and (site / "llms.txt").read_bytes() == store.get(layout.LLMS_TXT), done.stderr
    assert sorted(path.name for path in site.iterdir()) == ["llms.txt"]

    (site / "llms.txt").write_text("the old contract")
    done = command({**token, "SN120_R2_BUCKET": "not-the-bucket"}, "reset.py", "llms", "--site-dir", str(site))  # a set v2 name wins
    assert done.returncode == 1 and (site / "llms.txt").read_text() == "the old contract"
    done = command({name: value for name, value in token.items() if name != "R2_BUCKET"}, "reset.py", "llms", "--site-dir", str(site))
    assert done.returncode == 1 and (site / "llms.txt").read_text() == "the old contract"  # a missing name is named, and no value is shown
    assert "SN120_R2_BUCKET" in done.stderr and "R2_BUCKET is not set" in done.stderr and "fake-id" not in done.stderr


def test_a_store_setting_comes_from_its_plain_name_only_when_the_v2_name_is_not_set():
    host = "0123abcd.r2.cloudflarestorage.com"
    plain = {"R2_ENDPOINT": host, "R2_BUCKET": "bucket-a", "R2_ACCESS_KEY_ID": "plain-id", "R2_SECRET_ACCESS_KEY": SECRET, "PATH": "/bin"}
    want = {"SN120_R2_ENDPOINT": f"https://{host}", "SN120_R2_BUCKET": "bucket-a", "SN120_R2_ACCESS_KEY_ID": "plain-id", "SN120_R2_SECRET_ACCESS_KEY": SECRET}
    for form in (host, f" {host}/ ", f"https://{host}", f"https://{host}/", f"{host}/bucket-a", f"https://{host}/bucket-a/"):
        assert store_env({**plain, "R2_ENDPOINT": form}) == want  # a host name or a URL, with or without the bucket
    assert store_env({**plain, "R2_ENDPOINT": "", "R2_ACCOUNT_ID": "0123abcd"}) == want  # no endpoint: built from the account id
    assert store_env({**plain, "R2_ACCOUNT_ID": "ffff"}) == want  # the endpoint wins over the account id
    assert store_env({**plain, "R2_ENDPOINT": "http://127.0.0.1:5000"})["SN120_R2_ENDPOINT"] == "http://127.0.0.1:5000"
    v2 = {"SN120_R2_ENDPOINT": "http://127.0.0.1:1", "SN120_R2_BUCKET": "bucket-b", "SN120_R2_ACCESS_KEY_ID": "v2-id",
          "SN120_R2_SECRET_ACCESS_KEY": "v2-secret", "SN120_R2_PREFIX": "smoke"}
    assert store_env({**plain, **v2}) == v2  # a set v2 name wins, one name at a time
    assert store_env({**plain, "SN120_R2_BUCKET": "bucket-b"}) == {**want, "SN120_R2_BUCKET": "bucket-b"}
    for name, plain_name in PLAIN_NAMES.items():
        with pytest.raises(ConfigError) as caught:
            store_env({key: value for key, value in plain.items() if key != plain_name})
        text = str(caught.value)
        assert name in text and plain_name in text and SECRET not in text and "plain-id" not in text and host not in text
    for bad in (f"https://{host}/other/path", f"ftp://{host}", f"https://user:{SECRET}@{host}", f"{host}?x=1", "https://", f"{host}:port"):
        with pytest.raises(ConfigError) as caught:
            store_env({**plain, "R2_ENDPOINT": bad})
        assert "R2_ENDPOINT" in str(caught.value) and bad not in str(caught.value) and SECRET not in str(caught.value)
    with pytest.raises(ConfigError, match="R2_ACCOUNT_ID"):
        store_env({**plain, "R2_ENDPOINT": "", "R2_ACCOUNT_ID": "evil.example/x"})
    assert links.TOKEN_NAMES == {name: PLAIN_NAMES[name] for name in ("SN120_R2_ACCESS_KEY_ID", "SN120_R2_SECRET_ACCESS_KEY")}
    assert "store_env" not in (PULL.parents[1] / "sn120_v2" / "r2.py").read_text()  # R2Store.from_env reads only SN120_R2_*
