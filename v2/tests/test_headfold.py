"""The fold, the checkpoint and the rebuild of a head whose weight files stay on the box (``headfold.py``).

The head is real and tiny (``sim_head.py``), so each fold here is the real
arithmetic. No test counts on a fixed head or on a fixed count of past folds.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil
import threading
import time
from collections import namedtuple
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sn120_v2.sim_head import ChainWorld, is_weight, tiny_base, tiny_lora  # noqa: E402 - after the skip guard

from sn120_v2 import layout  # noqa: E402
from sn120_v2.canonical import file_sha256, manifest_digest  # noqa: E402
from sn120_v2.errors import ConfigError, ContractError, InfraError, VerifyError  # noqa: E402
from sn120_v2.fold_kernel import upcast_master  # noqa: E402
from sn120_v2.gpu.resident import hash_tree  # noqa: E402
from sn120_v2.headchain import CHAIN, FIXED_ORDER, FOLD_SCHEMA, FULL, MATMUL, chain_links, checkpoints, heads_of, small_files, tail  # noqa: E402
from sn120_v2.headfold import (  # noqa: E402
    BASE,
    BLAS,
    DEVICES_ENV,
    IMPLEMENTATION_ENV,
    MASTER,
    REFERENCE,
    THREADS_ENV,
    TWICE_ENV,
    FoldJob,
    FoldSettings,
    LinkSource,
    StoreSource,
    fold_checked,
    in_children,
    inline,
    main,
    rebuild,
    run_fold,
)
from sn120_v2.sim import sim_params  # noqa: E402
from sn120_v2.types import HeadManifest  # noqa: E402

Usage = namedtuple("Usage", "total used free")


@pytest.fixture
def world(tmp_path) -> ChainWorld:
    return ChainWorld(tmp_path / "world", sim_params())


@pytest.fixture
def job(tmp_path) -> FoldJob:
    """One fold of a tiny head with two LoRAs of different rank. The outputs do not exist yet."""
    base = tiny_base(tmp_path / "base")
    upcast_master(base, tmp_path / "master")
    loras = []
    for name, rank, alpha in (("first", 4, 37.7), ("second", 2, 4.0)):
        (tmp_path / name).mkdir()
        for file, data in tiny_lora(name, rank, alpha).items():
            (tmp_path / name / file).write_bytes(data)
        loras.append((tmp_path / name, rank))
    return FoldJob(tmp_path / "master", base, tuple(loras), tmp_path / "out" / "master", tmp_path / "out" / "base")


def is_head(world: ChainWorld, manifest: HeadManifest, root: Path | None = None) -> bool:
    """True when ``<root>/<id>/`` (default: the cache of the box) holds exactly the base and the master of ``manifest``."""
    root = root or world.box.work_dir / "cache"
    return hash_tree(root / manifest.head.base_hash) == manifest.base_files and hash_tree(root / manifest.head.base_fp32_master_digest) == manifest.master_files


def weights_of(manifest: HeadManifest) -> set[str]:
    """The keys of the folded weight files of one head."""
    keys = {layout.public_base_file(manifest.head.base_hash, name) for name in manifest.base_files}
    return {key for key in keys | {layout.public_master_file(manifest.head.base_fp32_master_digest, name) for name in manifest.master_files} if is_weight(key)}


def links_of(world: ChainWorld, doc: dict | None = None) -> dict:
    """A links file as the site's pull job makes it: the newest checkpoint and the LoRAs after it."""
    doc = doc or world.chain_file()
    return {"expires_at": int(time.time()) + 600} | chain_links(world.store, doc, 600, target=heads_of(doc)[-1]["version_id"])


# ---- the settings ----


def test_the_fold_settings_come_from_the_environment_and_a_wrong_value_is_an_error():
    assert FoldSettings.from_env({}) == FoldSettings(REFERENCE, True, ("cpu",), None) == FoldSettings()
    got = FoldSettings.from_env({IMPLEMENTATION_ENV: "blas", DEVICES_ENV: " cuda:0, cuda:1 ", THREADS_ENV: "8", TWICE_ENV: "1"})
    assert got == FoldSettings(BLAS, True, ("cuda:0", "cuda:1"), 8)
    assert FoldSettings.from_env({TWICE_ENV: "0"}).twice is False
    for wrong, name in (({IMPLEMENTATION_ENV: "fast"}, IMPLEMENTATION_ENV), ({DEVICES_ENV: "gpu0"}, DEVICES_ENV), ({DEVICES_ENV: "cuda"}, DEVICES_ENV), ({THREADS_ENV: "0"}, THREADS_ENV)):
        with pytest.raises(ConfigError, match=name):
            FoldSettings.from_env(wrong)
    # On the box only the second fold proves the bytes of ``blas``. A reader has the record for that, and makes no second fold.
    with pytest.raises(ConfigError, match="needs the second fold"):
        FoldSettings.from_env({IMPLEMENTATION_ENV: "blas", TWICE_ENV: "0"})
    assert FoldSettings.from_env({IMPLEMENTATION_ENV: "blas", TWICE_ENV: "0"}, box=False) == FoldSettings(BLAS, False)
    assert FoldSettings.from_env({}, box=False).twice is False


# ---- one fold ----


def test_one_fold_gives_the_same_hashes_written_or_only_hashed_and_with_any_count_of_threads(job):
    written = run_fold(job)
    assert written.base_files == hash_tree(job.out_base) and written.master_files == hash_tree(job.out_master) and written.device == "cpu" and not written.note
    assert set(written.base_files) == set(hash_tree(job.base_dir)) and written.base_files != hash_tree(job.base_dir)
    assert written.ids == (manifest_digest(written.base_files), manifest_digest(written.master_files))
    out = job.out_base.parent
    shutil.rmtree(out)
    for threads in (1, 3):
        hashed = run_fold(dataclasses.replace(job, out_master=None, out_base=None, threads=threads))
        assert hashed.ids == written.ids and not out.exists()


def test_a_device_that_is_not_there_gives_the_fold_to_the_cpu_with_a_note(job):
    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    plain = run_fold(dataclasses.replace(job, out_master=None, out_base=None))
    moved = run_fold(dataclasses.replace(job, device="cuda:0"))
    assert moved.device == "cpu" and "cuda:0 did not pass its check" in moved.note and moved.ids == plain.ids
    assert hash_tree(job.out_master) == moved.master_files


def test_the_reference_code_gives_the_bytes_of_the_old_fold_for_the_old_kernel(job, tmp_path):
    old = run_fold(dataclasses.replace(job, kernel=MATMUL, implementation=BLAS))
    again = run_fold(dataclasses.replace(job, kernel=MATMUL, out_master=None, out_base=None, device="cuda:0"))  # the old kernel is a CPU product
    assert again.ids == old.ids and again.device == "cpu" and not again.note


def test_a_fold_that_cannot_be_done_is_refused_with_its_reason(job):
    with pytest.raises(VerifyError, match="does not know the kernel 'fixed-order-v2'.*newer version"):
        run_fold(dataclasses.replace(job, kernel="fixed-order-v2"))
    with pytest.raises(ContractError, match="cannot hash only"):
        run_fold(dataclasses.replace(job, implementation=BLAS, out_base=None))
    with pytest.raises(VerifyError, match="rank"):
        run_fold(dataclasses.replace(job, loras=((job.loras[0][0], 8),)))
    job.out_master.mkdir(parents=True)
    with pytest.raises(InfraError, match="FileExistsError"):
        run_fold(job)


def test_the_second_fold_only_hashes_takes_the_second_device_and_must_agree(job):
    seen = []

    def run(jobs):
        seen.append(list(jobs))
        return inline([dataclasses.replace(one, device="cpu") for one in jobs])

    results = fold_checked(job, FoldSettings(REFERENCE, True, ("cuda:0", "cuda:1"), 2), run, "test")
    first, second = seen[-1]
    assert (first.implementation, first.device, first.out_master, first.threads) == (REFERENCE, "cuda:0", job.out_master, 2)
    assert (second.implementation, second.device, second.out_master, second.out_base, second.threads) == (REFERENCE, "cuda:1", None, None, 2)
    assert len(results) == 2 and results[0].ids == results[1].ids
    shutil.rmtree(job.out_base.parent)
    fold_checked(job, FoldSettings(BLAS, True, ("cuda:3",)), lambda jobs: seen.append(list(jobs)) or [results[0]] * len(jobs), "test")
    assert [(one.implementation, one.device) for one in seen[-1]] == [(BLAS, "cuda:3"), (REFERENCE, "cuda:3")]
    assert len(fold_checked(job, FoldSettings(twice=False), lambda jobs: seen.append(list(jobs)) or [results[0]] * len(jobs), "test")) == 1 and len(seen[-1]) == 1

    def other(jobs):
        return [results[0], dataclasses.replace(results[0], master_files=results[0].master_files | {"added.safetensors": "0" * 64})]

    with pytest.raises(VerifyError, match="the two folds gave different files"):
        fold_checked(job, FoldSettings(), other, "test")


def test_the_folds_of_the_box_run_in_child_processes_and_an_error_of_a_child_comes_back(job):
    plain = run_fold(dataclasses.replace(job, out_master=None, out_base=None))
    results = in_children([job, dataclasses.replace(job, out_master=None, out_base=None)])
    assert [result.ids for result in results] == [plain.ids] * 2 and hash_tree(job.out_master) == plain.master_files
    with pytest.raises(VerifyError, match="does not know the kernel"):
        in_children([dataclasses.replace(job, out_master=None, out_base=None), dataclasses.replace(job, kernel="other", out_master=None, out_base=None)])


# ---- the fold of the box ----


def test_a_fold_in_chain_mode_keeps_the_weight_files_on_the_box_and_publishes_the_small_files_and_the_record(world):
    world.start(CHAIN)
    before, old = set(world.store.weights()), world.head
    manifest = world.king()
    head = manifest.head
    assert set(world.store.weights()) == before and not weights_of(manifest) & before  # no weight file of the new head went to the store
    assert manifest.base_files != old.base_files and set(manifest.base_files) == set(old.base_files) and len(manifest.master_files) == len(old.master_files)
    small = small_files(manifest.base_files)
    assert world.store.list(layout.public_base_prefix(head.base_hash)) == sorted(layout.public_base_file(head.base_hash, name) for name in small) and small
    for name, sha in small.items():
        assert world.store.sha256(layout.public_base_file(head.base_hash, name)) == sha
    assert not world.store.list(layout.public_master_prefix(head.base_fp32_master_digest)) and not world.store.list("pending/")
    assert HeadManifest.from_json(world.store.get(layout.CURRENT)) == manifest == HeadManifest.from_json(world.store.get(layout.head_manifest(head.version_id)))
    assert is_head(world, manifest)  # the next round starts from these files; it downloads nothing
    record = json.loads(world.store.get(layout.head_fold(head.parent_version_id)))
    assert record == {
        "schema": FOLD_SCHEMA, "stacked_version_id": head.parent_version_id, "base_hash": head.base_hash, "base_fp32_master_digest": head.base_fp32_master_digest,
        "kernel": FIXED_ORDER, "torch": torch.__version__, "image_digest": world.box.image_digest,
        "folds": [{"implementation": REFERENCE, "device": "cpu"}] * 2,
    }  # no time and no size: the record is the same after a second fold of the same head
    phases = [json.loads(line) for line in (world.box.work_dir / "logs" / "phases.jsonl").read_text().splitlines()]
    assert phases[-1]["phase"] == "fold" and phases[-1]["mode"] == "chain" and phases[-1]["version_id"] == head.parent_version_id
    assert not list((world.box.work_dir / "fold").iterdir())
    later = world.king()  # a fold from local files alone: the store has no master of the head before it
    assert later.head.parent_version_id != head.version_id and world.state.manifests[later.head.parent_version_id].head.parent_version_id == head.version_id
    assert is_head(world, later) and set(world.store.weights()) == before


def test_two_folds_that_differ_keep_nothing_and_the_head_does_not_move(world):
    world.start(CHAIN)
    start = world.king()

    def faulty(jobs):
        results = inline(jobs)
        return [results[0], dataclasses.replace(results[1], master_files={name: "0" * 64 for name in results[1].master_files})]

    world.folder.run = faulty
    world.register(90, 0.3)
    world.round()
    stacked = world.state.head
    assert not stacked.folded and "the two folds gave different files; nothing was kept" in world.controller.last_error
    assert sorted(path.name for path in (world.box.work_dir / "cache").iterdir()) == sorted({stacked.base_hash, stacked.base_fp32_master_digest, *(ref.lora_hash for ref in stacked.loras)})
    assert not list((world.box.work_dir / "fold").iterdir()) and not world.store.exists(layout.head_fold(stacked.version_id))
    assert world.head == start and world.writer.doc(world.state)["stacked"]["version_id"] == stacked.version_id
    world.folder.run = inline
    world.now += world.params.apply_retry_s
    world.round()
    assert world.state.head.folded and world.state.head.parent_version_id == stacked.version_id and is_head(world, world.head)


def test_a_fold_needs_the_free_disk_of_one_head_and_a_checkpoint_the_disk_of_two(world, monkeypatch, capsys):
    world.start(CHAIN, every=1)
    size = sum(file.stat().st_size for identity in (world.head.head.base_hash, world.head.head.base_fp32_master_digest) for file in world.local(identity).iterdir())
    free = [size - 1]
    # The free disk of the work directory. The cache directory below it is asked for each download: that answer stays large.
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(1 << 40, 0, free[0] if Path(path) == world.box.work_dir else 1 << 39))
    world.register(90, 0.3)
    world.round()
    assert not world.state.head.folded and "not enough free disk for the fold" in world.controller.last_error
    free[0] = 2 * size - 1  # room for the fold, not for the two heads that a waiting checkpoint needs
    world.now += world.params.apply_retry_s
    capsys.readouterr()
    world.round()
    world.wait()
    assert world.state.head.folded and "checkpoint: not started: InfraError" in capsys.readouterr().err
    assert not world.keeper.links.exists() and world.head.head.version_id not in checkpoints(world.store)
    world.round()  # a pass of ``run`` with no round: the same problem is not printed again
    assert "checkpoint" not in capsys.readouterr().err
    free[0] = 2 * size
    world.round()
    world.wait()
    assert world.head.head.version_id in checkpoints(world.store)


def test_a_pass_of_run_with_no_new_head_does_not_read_the_store_for_the_checkpoint_rule(world):
    world.start(CHAIN, every=3)
    asked, real = [], world.keeper.due
    world.keeper.due = lambda state: asked.append(state.head.version_id) or real(state)
    first = world.king()
    for _ in range(3):  # ``run`` calls ``after_round`` after each pass of its loop, each 30 s when no file waits
        assert world.round() is None
    second = world.king()
    assert asked == [first.head.version_id, second.head.version_id], "one read of the markers for each new head"


# ---- the lost local head, and the rebuild ----


@pytest.mark.parametrize("past", [0, 2])
def test_a_box_that_lost_its_local_head_rebuilds_it_at_the_start_of_run(world, past, capsys):
    for _ in range(past):
        world.king()
    world.start(CHAIN)
    for _ in range(2):
        world.king()
    manifest = world.head
    shutil.rmtree(world.box.work_dir / "cache")
    world.register(90, 0.3)
    world.round()
    assert not world.state.head.folded and "Stop run and start it again" in world.controller.last_error
    capsys.readouterr()
    world.start()
    assert f"the local files of head {manifest.head.version_id[:12]} are not there" in capsys.readouterr().err
    assert is_head(world, manifest) and world.head == manifest
    world.now += world.params.apply_retry_s
    world.round()
    assert world.state.head.folded and is_head(world, world.head)


@pytest.mark.parametrize("past", [0, 3])
def test_a_reader_rebuilds_the_head_from_genesis_through_every_past_and_new_step(world, tmp_path, past):
    for _ in range(past):
        world.king()
    world.start(CHAIN)
    for _ in range(2):
        world.king()
    world.wait()
    doc = world.chain_file()
    for step in doc["steps"]:  # a reader that trusts only genesis: the old kernel of each past step is replayed too
        step["checkpoint"] = False
    assert [step["kernel"] for step in doc["steps"]] == [MATMUL] * past + [FIXED_ORDER] * 2
    source = StoreSource(world.store)
    result = rebuild(doc, tmp_path / "reader", source)
    manifest = world.head
    assert result["version_id"] == manifest.head.version_id and result["started_from"] == doc["genesis"]["version_id"] and result["steps"] == past + 2
    assert hash_tree(Path(result["base_dir"])) == manifest.base_files and hash_tree(Path(result["master_dir"])) == manifest.master_files
    assert result["fetched_bytes"] == source.fetched > 0
    # Only the head stays: each older head and each LoRA is deleted after its step.
    assert sorted(path.name for path in (tmp_path / "reader" / "cache").iterdir()) == sorted([manifest.head.base_hash, manifest.head.base_fp32_master_digest])
    assert not list((tmp_path / "reader" / "fold").iterdir())


def test_a_reader_with_the_links_file_starts_at_the_newest_checkpoint_and_later_at_the_head_that_it_holds(world, tmp_path):
    world.king()
    world.start(CHAIN)
    world.wait()
    for _ in range(2):
        world.king()
    doc, work = world.chain_file(), tmp_path / "reader"
    assert doc["start"]["steps_to_head"] == 2 and doc["start"]["version_id"] != doc["genesis"]["version_id"]
    source = LinkSource(links_of(world))
    first = rebuild(tail(doc), work, source)  # the part of the chain that mining.json shows is enough
    assert first["started_from"] == doc["start"]["version_id"] and first["steps"] == 2 and is_head(world, world.head, work / "cache")
    again = LinkSource(links_of(world))
    assert rebuild(tail(doc), work, again) == first | {"started_from": first["version_id"], "steps": 0, "fetched_bytes": 0}

    world.king()
    doc, source = world.chain_file(), LinkSource(links_of(world))
    third = rebuild(tail(doc), work, source)
    lora = sum(row["size"] for group in links_of(world)["chain_loras"][-1:] for row in group["files"])
    assert third["started_from"] == first["version_id"] and third["steps"] == 1 and third["fetched_bytes"] == lora  # one LoRA, no weight file
    assert is_head(world, world.head, work / "cache") and len(list((work / "cache").iterdir())) == 2

    older = rebuild(doc, tmp_path / "older", LinkSource({"expires_at": int(time.time()) + 600} | chain_links(world.store, doc, 600)), first["version_id"])
    assert older["version_id"] == first["version_id"] and older["steps"] == 2  # a head before the last one, with the links of every checkpoint
    assert hash_tree(Path(older["master_dir"])) == world.state.manifests[first["version_id"]].master_files

    with pytest.raises(ConfigError, match="the links have expired"):
        LinkSource(links_of(world) | {"expires_at": int(time.time()) - 1})
    with pytest.raises(ConfigError, match="is not complete"):
        LinkSource(links_of(world) | {"chain_loras": [{"id": "x"}]})
    with pytest.raises(ConfigError, match="has no links for the file set"):
        rebuild(tail(doc), tmp_path / "bare", LinkSource(links_of(world) | {"checkpoints": []}))
    stale = links_of(world)
    stale["chain_loras"][0]["files"][0]["sha256"] = "0" * 64
    with pytest.raises(VerifyError, match="does not list adapter_config.json of the lora"):
        rebuild(tail(doc), tmp_path / "stale", LinkSource(stale))


def test_a_step_with_other_hashes_than_the_record_is_deleted_and_the_text_names_the_way_out(world, tmp_path):
    world.king()
    world.start(CHAIN)
    world.king()
    good = world.chain_file()
    for step in good["steps"]:
        step["checkpoint"] = False
    old, new = good["steps"][0], good["steps"][-1]
    assert (old["kernel"], new["kernel"]) == (MATMUL, FIXED_ORDER)

    def wrong(step_at: int) -> dict:
        doc = copy.deepcopy(good)
        doc["steps"][step_at]["base_fp32_master_digest"] = "0" * 64
        return doc

    cases = (
        (0, FoldSettings(), "old", f"kernel {MATMUL}, implementation {REFERENCE}.*only a machine like the box repeats it. Start from a checkpoint after this step"),
        (-1, FoldSettings(), "new", f"kernel {FIXED_ORDER}, implementation {REFERENCE}.*this machine or a file on it has a fault. Run: python -m sn120_v2.fold_kernel marks"),
        (-1, FoldSettings(BLAS, False), "blas", f"kernel {FIXED_ORDER}, implementation {BLAS}.*Set {IMPLEMENTATION_ENV}={REFERENCE} \\(or remove the variable\\) and run again"),
    )
    for step_at, settings, name, text in cases:
        work, doc = tmp_path / name, wrong(step_at)
        with pytest.raises(VerifyError, match=text) as caught:
            rebuild(doc, work, StoreSource(world.store), settings=settings)
        step = doc["steps"][step_at]
        assert f"the step to head {step['version_id'][:12]}" in str(caught.value) and "they were deleted" in str(caught.value)
        before = heads_of(doc)[len(doc["steps"]) + step_at if step_at < 0 else step_at]
        kept = {path.name for path in (work / "cache").iterdir()}
        assert {before["base_hash"], before["base_fp32_master_digest"]} <= kept  # the head before the step stays: the next try starts there
        assert not {"0" * 64, good["steps"][step_at]["base_hash"], good["steps"][step_at]["base_fp32_master_digest"]} & kept
        assert not list((work / "fold").iterdir())
    result = rebuild(good, tmp_path / "new", StoreSource(world.store))  # the same work directory, with the right record
    assert result["steps"] == 1 and result["started_from"] == old["version_id"] and is_head(world, world.head, tmp_path / "new" / "cache")


class Dirs:
    """A source that gives the files of two directories, and that lists what was asked of it."""

    def __init__(self, base: Path, master: Path) -> None:
        self.dirs, self.fetched, self.asked = {BASE: base, MASTER: master}, 0, []

    def checkpoint(self, head):
        return hash_tree(self.dirs[BASE]), hash_tree(self.dirs[MASTER])

    def get(self, kind, set_id, name, sha256, path):
        self.asked.append((kind, name))
        shutil.copyfile(self.dirs[kind] / name, path)
        assert file_sha256(path) == sha256
        self.fetched += path.stat().st_size


@pytest.mark.parametrize("plain", [False, True])
def test_the_base_of_a_checkpoint_is_the_cast_of_its_master_and_is_fetched_only_when_the_cast_differs(tmp_path, plain, capsys):
    base = tiny_base(tmp_path / "base", plain=plain)
    master_files = upcast_master(base, tmp_path / "master")
    base_files = hash_tree(base)
    head = {"version_id": "1" * 64, "base_hash": manifest_digest(base_files), "base_fp32_master_digest": manifest_digest(master_files), "checkpoint": True}
    doc = {"schema": "sn120-head-chain-v1", "head_version_id": head["version_id"], "genesis": head, "steps": [], "stacked": None}
    source, work = Dirs(base, tmp_path / "master"), tmp_path / "work"
    result = rebuild(doc, work, source)
    assert result["steps"] == 0 and hash_tree(Path(result["base_dir"])) == base_files and hash_tree(Path(result["master_dir"])) == master_files
    weights = {(BASE, name) for name in base_files if name not in small_files(base_files)}
    assert {(MASTER, name) for name in master_files} | {(BASE, name) for name in small_files(base_files)} <= set(source.asked) and len(weights) == 2
    assert (weights <= set(source.asked)) is plain and (weights & set(source.asked) == set()) is not plain
    assert ("the base weight files are fetched" in capsys.readouterr().err) is plain
    assert not (work / "fetch").exists() or not list((work / "fetch").iterdir())
    count = len(source.asked)
    assert rebuild(doc, work, source)["steps"] == 0 and len(source.asked) == count  # a head that is held is not fetched again


def test_the_rebuild_command_reads_mining_json_or_the_chain_file_and_says_refused_with_the_reason(world, tmp_path, capsys):
    world.king()
    world.start(CHAIN)
    world.wait()
    world.king()
    manifest = world.head
    chain, links, mining = tmp_path / "head_chain.json", tmp_path / "read_links.json", tmp_path / "mining.json"
    chain.write_bytes(world.store.get(layout.HEAD_CHAIN))
    links.write_text(json.dumps(links_of(world)))
    mining.write_text(json.dumps({"head_chain": world.publisher.snapshots[-1]["head_chain"], "contract": {"public_url": tmp_path.as_uri(), "read_links": "read_links.json"}}))
    capsys.readouterr()
    assert main(["rebuild", "--chain", str(chain), "--links", str(links), "--work", str(tmp_path / "one")]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["version_id"] == manifest.head.version_id and first["steps"] == 1 and is_head(world, manifest, tmp_path / "one" / "cache")
    assert main(["rebuild", "--chain", mining.as_uri(), "--work", str(tmp_path / "two"), "--device", "cpu"]) == 0  # the links file that mining.json names
    assert json.loads(capsys.readouterr().out) | {"base_dir": "", "master_dir": ""} == first | {"base_dir": "", "master_dir": ""}
    start = world.chain_file()["start"]["version_id"]
    assert main(["rebuild", "--chain", str(chain), "--links", str(links), "--work", str(tmp_path / "three"), "--head", start]) == 0
    assert json.loads(capsys.readouterr().out)["steps"] == 0

    refused = (
        (["--chain", str(chain), "--work", str(tmp_path / "x")], "refused: ConfigError: the chain file names no links file; pass --links or --store"),
        (["--chain", str(tmp_path / "none.json"), "--work", str(tmp_path / "x")], "refused: ConfigError: cannot read"),
        (["--chain", str(links), "--links", str(links), "--work", str(tmp_path / "x")], "refused: VerifyError: the head chain is not complete"),
        (["--chain", str(chain), "--links", str(links), "--work", str(tmp_path / "x"), "--device", "gpu"], f"refused: ConfigError: {DEVICES_ENV} must be"),
        (["--chain", str(chain), "--links", str(links), "--work", str(tmp_path / "x"), "--head", "0" * 64], "refused: VerifyError: head 000000000000 is not a folded head"),
    )
    for argv, text in refused:
        assert main(["rebuild", *argv]) == 2
        out, err = capsys.readouterr()
        assert text in err and not out


# ---- the checkpoints ----


def test_a_checkpoint_is_due_after_the_set_count_of_folds_and_never_by_itself_with_zero(world):
    world.start(CHAIN, every=0)
    before = set(world.store.weights())
    for _ in range(3):
        world.king()
        assert not world.keeper.due(world.state)
    world.wait()
    assert checkpoints(world.store) == set() and set(world.store.weights()) == before and not world.keeper.links.exists()

    world.start(every=2)  # two folds have no checkpoint, so one is due at this start
    world.wait()
    marked = [world.head.head.version_id]
    assert checkpoints(world.store) == set(marked)
    for count in range(1, 5):
        head = world.king().head
        world.wait()
        marked += [head.version_id] if count % 2 == 0 else []
        assert checkpoints(world.store) == set(marked)
    doc = world.chain_file()
    assert [head["version_id"] for head in heads_of(doc) if head["checkpoint"]] == [doc["genesis"]["version_id"], *marked] and doc["start"]["steps_to_head"] == 0


@pytest.mark.parametrize("past", [1, 3])
def test_the_last_head_of_full_mode_is_marked_at_the_switch_and_no_file_is_sent_again(world, past):
    for _ in range(past):
        world.king()
    manifest, at = world.head, len(world.store.wrote)
    world.start(CHAIN, every=0)  # also with "never by itself": this head costs no upload
    world.wait()
    assert world.store.wrote[at:] == [layout.head_checkpoint(manifest.head.version_id)]
    marker = json.loads(world.store.get(layout.head_checkpoint(manifest.head.version_id)))
    assert marker["version_id"] == manifest.head.version_id and marker["base_hash"] == manifest.head.base_hash
    assert set(marker["base_sizes"]) == set(manifest.base_files) and set(marker["master_sizes"]) == set(manifest.master_files)
    assert all(size == world.store.size(layout.public_master_file(manifest.head.base_fp32_master_digest, name)) for name, size in marker["master_sizes"].items())
    assert set(marker) == {"schema", "version_id", "base_hash", "base_fp32_master_digest", "base_sizes", "master_sizes"}  # no time: a second write is the same bytes
    world.king()
    assert world.chain_file()["start"] == {"version_id": manifest.head.version_id, "base_hash": manifest.head.base_hash, "checkpoint": True, "steps_to_head": 1} | {
        "base_fp32_master_digest": manifest.head.base_fp32_master_digest
    }


def test_with_a_checkpoint_for_every_king_the_round_goes_on_and_the_marker_of_each_head_is_written_last(world):
    world.start(CHAIN, every=1)
    world.store.gate = threading.Event()  # the upload of a weight file waits: a slow store
    before, at = set(world.store.weights()), len(world.store.wrote)
    first = world.king()
    assert set(world.store.weights()) == before and checkpoints(world.store) == set()  # the head moved, and the store has no weight file of it
    assert world.chain_file()["start"]["steps_to_head"] == 1
    second = world.king()  # a whole round and its fold, while the upload of the first head waits
    assert set(world.store.weights()) == before and checkpoints(world.store) == set() and world.chain_file()["start"]["steps_to_head"] == 2
    assert sorted(path.name for path in world.keeper.links.iterdir()) == sorted([first.head.version_id, second.head.version_id])
    world.store.gate.set()
    world.wait()
    wrote = world.store.wrote[at:]
    markers = [wrote.index(layout.head_checkpoint(manifest.head.version_id)) for manifest in (first, second)]
    for manifest, low, high in ((first, 0, markers[0]), (second, markers[0], markers[1])):
        sent = {key for key in wrote if key in weights_of(manifest)}
        assert sent == weights_of(manifest) and len(sent) == len(manifest.master_files) + len(manifest.base_files) - len(small_files(manifest.base_files))
        assert all(low <= wrote.index(key) < high for key in sent)  # each file before the marker of its head, and the first king before the second
        for key in sent:
            assert world.store.sha256(key) in {**manifest.base_files, **manifest.master_files}.values()
    doc = world.chain_file()
    assert [step["checkpoint"] for step in doc["steps"][-2:]] == [True, True]
    assert doc["start"]["version_id"] == second.head.version_id and doc["start"]["steps_to_head"] == 0
    assert not list(world.keeper.links.iterdir())  # the links go only after the marker
    assert is_head(world, world.king())  # the cache gave up the two old heads, and the next fold still has its own


@pytest.mark.parametrize("stop", ["first file", "last file", "marker"])
def test_an_upload_that_stops_leaves_no_marker_and_the_next_start_sends_only_what_is_missing(world, stop, capsys):
    world.start(CHAIN, every=1)
    refused = []

    def refuse(key: str) -> bool:
        target = sorted(weights_of(world.head))[0 if stop == "first file" else -1] if stop != "marker" else layout.head_checkpoint(world.head.head.version_id)
        if key == target:
            refused.append(key)
        return key == target

    world.store.refuse = refuse
    capsys.readouterr()
    manifest = world.king()
    world.wait()
    assert refused and f"checkpoint of {manifest.head.version_id[:12]}: stopped by InfraError" in capsys.readouterr().err
    assert checkpoints(world.store) == set() and world.chain_file()["start"]["steps_to_head"] == 1
    assert (world.keeper.links / manifest.head.version_id).is_dir()  # the files of the head stay on the box
    missing = weights_of(manifest) - set(world.store.weights())
    assert not missing if stop == "marker" else set(refused) <= missing  # a file that had not started is not sent either
    world.store.refuse, at = None, len(world.store.wrote)
    world.start()  # as after a stop of ``run``: the links are found, and the upload goes on
    world.wait()
    wrote = world.store.wrote[at:]
    assert {key for key in wrote if is_weight(key)} == missing and wrote.index(layout.head_checkpoint(manifest.head.version_id)) >= len(missing)
    assert checkpoints(world.store) == {manifest.head.version_id} and not list(world.keeper.links.iterdir())
    world.king()
    assert world.chain_file()["start"]["version_id"] == manifest.head.version_id


def test_the_checkpoint_command_uploads_the_head_now_and_does_nothing_for_a_head_that_is_one(world):
    world.start(CHAIN, every=0)
    manifest, at = world.king(), len(world.store.wrote)
    command = world.controller.head.checkpoint
    assert command(world.state) == {"version_id": manifest.head.version_id, "checkpoint": True, "uploaded": True}
    wrote = world.store.wrote[at:]
    assert {key for key in wrote if is_weight(key)} == weights_of(manifest) and wrote[-2:] == [layout.head_checkpoint(manifest.head.version_id), layout.HEAD_CHAIN]
    assert world.chain_file()["start"]["steps_to_head"] == 0 and not list(world.keeper.links.iterdir())
    at = len(world.store.wrote)
    assert command(world.state) == {"version_id": manifest.head.version_id, "checkpoint": True, "uploaded": False} and world.store.wrote[at:] == []
    later = world.king()
    shutil.rmtree(world.box.work_dir / "cache")
    with pytest.raises(InfraError, match=f"the local files of head {later.head.version_id[:12]} cannot be linked"):
        command(world.state)
    assert checkpoints(world.store) == {manifest.head.version_id}


def test_full_mode_after_chain_mode_needs_the_local_head_or_a_checkpoint(world):
    world.start(CHAIN, every=0)
    world.king()
    world.start(FULL)
    manifest = world.king()  # the local files are intact: the fold of full mode reads them and uploads the new head
    assert weights_of(manifest) <= set(world.store.weights())
    world.start(CHAIN, every=0)
    world.wait()
    later = world.king()
    assert world.chain_file()["start"]["version_id"] == manifest.head.version_id and [step["kernel"] for step in world.chain_file()["steps"]] == [FIXED_ORDER, MATMUL, FIXED_ORDER]
    world.start(FULL)
    for path in (world.box.work_dir / "cache").iterdir():
        shutil.rmtree(path)
    world.register(90, 0.3)
    world.round()
    assert not world.state.head.folded and world.head == later and world.controller.last_error.startswith("fold failed")
    world.start(CHAIN, every=0)  # the store has no weight file of this head: one start in chain mode rebuilds it on the box
    assert is_head(world, later)
    world.start(FULL)
    world.now += world.params.apply_retry_s
    world.round()
    assert world.state.head.folded and weights_of(world.head) <= set(world.store.weights())
