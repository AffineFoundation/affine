"""The head chain (``headchain.py``): what the record says of a journal, and what a reader may trust in it.

No test here counts on a fixed head or on a fixed count of past folds: each one
reads the heads of the journal, as the code does.
"""

from __future__ import annotations

import copy
import json

import pytest

torch = pytest.importorskip("torch")

from sn120_v2.sim_head import ChainWorld, is_weight  # noqa: E402 - after the skip guard

from sn120_v2 import headfold, layout, publisher  # noqa: E402
from sn120_v2.errors import ConfigError, VerifyError  # noqa: E402
from sn120_v2.headchain import (  # noqa: E402
    CHAIN,
    DEFAULT_EVERY,
    EVERY_ENV,
    FIXED_ORDER,
    FOLD_RECORD,
    FULL,
    MATMUL,
    MODE_ENV,
    SCHEMA,
    ChainWriter,
    chain_links,
    check_chain,
    checkpoint_every,
    checkpoints,
    heads_of,
    mode,
    route,
    small_files,
    tail,
    with_start,
)
from sn120_v2.sim import sim_params  # noqa: E402
from sn120_v2.types import HeadManifest  # noqa: E402


@pytest.fixture
def world(tmp_path) -> ChainWorld:
    return ChainWorld(tmp_path / "world", sim_params())


def folded(world: ChainWorld) -> list[HeadManifest]:
    """The manifest of each folded head of the journal, oldest first. The first one is genesis."""
    return [world.state.manifests[head.version_id] for head in world.state.heads if head.folded]


def test_the_mode_and_the_checkpoint_count_come_from_the_environment_and_a_wrong_value_is_an_error():
    assert mode({}) == FULL and mode({MODE_ENV: " "}) == FULL and mode({MODE_ENV: "chain"}) == CHAIN and mode({MODE_ENV: "full"}) == FULL
    with pytest.raises(ConfigError, match=MODE_ENV):
        mode({MODE_ENV: "Chain"})
    assert checkpoint_every({}) == DEFAULT_EVERY and checkpoint_every({EVERY_ENV: "0"}) == 0 and checkpoint_every({EVERY_ENV: " 3 "}) == 3
    for wrong in ("-1", "1.5", "often"):
        with pytest.raises(ConfigError, match=EVERY_ENV):
            checkpoint_every({EVERY_ENV: wrong})


def test_full_mode_writes_no_part_of_the_chain_and_uploads_each_head(world):
    for _ in range(2):
        world.king()
    assert not [key for key in world.store.list("public/") if "head_chain" in key or "head_checkpoints" in key or "head_folds" in key]
    assert all("head_chain" not in snapshot for snapshot in world.publisher.snapshots)
    for manifest in folded(world):
        for name in manifest.base_files:
            assert world.store.exists(layout.public_base_file(manifest.head.base_hash, name))
        for name in manifest.master_files:
            assert world.store.exists(layout.public_master_file(manifest.head.base_fp32_master_digest, name))


@pytest.mark.parametrize("past", [0, 1, 4])
def test_the_chain_has_one_step_for_each_fold_whatever_the_count_of_past_folds(world, past):
    for _ in range(past):
        world.king()
    world.start(CHAIN)
    new = [world.king().head.version_id for _ in range(2)]
    world.wait()
    doc, heads = world.chain_file(), folded(world)
    check_chain(doc)
    assert doc["schema"] == SCHEMA and doc["head_version_id"] == world.state.head.version_id and doc["stacked"] is None
    assert [head["version_id"] for head in heads_of(doc)] == [manifest.head.version_id for manifest in heads]
    assert len(doc["steps"]) == past + 2 and [step["version_id"] for step in doc["steps"][past:]] == new
    # A fold of full mode left no fold record, so it has the kernel of gpu/folder.py. Each new fold names its own.
    assert [step["kernel"] for step in doc["steps"]] == [MATMUL] * past + [FIXED_ORDER] * 2
    for before, step, manifest in zip(heads, doc["steps"], heads[1:]):
        stacked = world.state.manifests[step["stacked_version_id"]]
        assert step["from_version_id"] == before.head.version_id and stacked.head.parent_version_id == before.head.version_id
        assert (step["base_hash"], step["base_fp32_master_digest"]) == (manifest.head.base_hash, manifest.head.base_fp32_master_digest)
        assert [lora["lora_hash"] for lora in step["loras"]] == [ref.lora_hash for ref in stacked.head.loras] and step["loras"]
        for lora in step["loras"]:
            assert lora["files"] == stacked.lora_files[lora["lora_hash"]] and lora["rank"] == world.params.recipe.rank
    # The last head of full mode has its files in the store, so the start of ``run`` marked it. No later head is in the store.
    marked = [head["version_id"] for head in heads_of(doc) if head["checkpoint"]]
    assert marked == ([heads[0].head.version_id] if not past else [heads[0].head.version_id, heads[past].head.version_id])
    assert doc["start"] == {"version_id": marked[-1], "base_hash": heads[past].head.base_hash, "checkpoint": True, "steps_to_head": 2} | {
        "base_fp32_master_digest": heads[past].head.base_fp32_master_digest
    }
    shown = world.publisher.snapshots[-1]["head_chain"]
    assert shown == tail(doc) and shown["file"] == layout.HEAD_CHAIN and "genesis" not in shown and "recipe" not in shown
    assert [step["version_id"] for step in shown["steps"]] == new
    check_chain(shown)


def test_a_head_with_a_lora_that_is_not_folded_is_in_the_chain_as_stacked(world):
    world.start(CHAIN)
    world.king()

    class NoFold:
        def fold(self, head, params, out_prefix):
            raise VerifyError("simulated fold failure")

    world.controller.folder = NoFold()
    world.register(90, 0.3)
    world.round()
    head = world.state.head
    assert not head.folded and "simulated fold failure" in world.controller.last_error
    doc = world.writer.doc(world.state)
    check_chain(doc)
    assert doc["head_version_id"] == head.version_id and doc["stacked"]["version_id"] == head.version_id
    assert [lora["lora_hash"] for lora in doc["stacked"]["loras"]] == [ref.lora_hash for ref in head.loras]
    assert heads_of(doc)[-1]["version_id"] == head.parent_version_id and doc["start"]["steps_to_head"] == 1


def test_a_chain_that_does_not_link_up_or_that_names_other_files_is_refused(world):
    world.start(CHAIN)
    for _ in range(3):
        world.king()
    good = world.chain_file()
    check_chain(good)

    def changed(edit) -> dict:
        doc = copy.deepcopy(good)
        edit(doc)
        return doc

    cases = {
        "the schema": lambda doc: doc.update(schema="sn120-head-chain-v2"),
        "does not start at a checkpoint": lambda doc: doc["genesis"].update(checkpoint=False),
        "does not start at the head before it": lambda doc: doc["steps"].pop(1),
        "do not hash to its id": lambda doc: doc["steps"][-1]["loras"][0]["files"].update({"adapter_config.json": "0" * 64}),
        "does not end at the head that it names": lambda doc: doc.update(head_version_id=doc["steps"][0]["version_id"]),
        "is not complete": lambda doc: doc["steps"][0].pop("loras"),
    }
    for text, edit in cases.items():
        with pytest.raises(VerifyError, match=text):
            check_chain(changed(edit))
    with pytest.raises(VerifyError, match="is not complete"):
        check_chain({"schema": SCHEMA})
    with pytest.raises(VerifyError, match="does not start at a checkpoint"):  # a tail whose start is only said to be one
        check_chain(tail(good) | {"start": good["start"] | {"checkpoint": "yes"}})


def test_the_route_starts_at_the_nearest_head_that_the_machine_holds_else_at_the_nearest_checkpoint(world):
    world.start(CHAIN)
    for _ in range(4):
        world.king()
    doc = world.chain_file()
    ids = [head["version_id"] for head in heads_of(doc)]
    nothing = lambda head: False  # noqa: E731
    start, steps = route(doc, ids[-1], nothing)
    assert start["version_id"] == ids[0] and [step["version_id"] for step in steps] == ids[1:]
    doc["steps"][1]["checkpoint"] = True  # the head ids[2]
    assert with_start(doc)["start"]["version_id"] == ids[2] and doc["start"]["steps_to_head"] == 2
    start, steps = route(doc, ids[-1], nothing)
    assert start["version_id"] == ids[2] and [step["version_id"] for step in steps] == ids[3:]
    start, steps = route(doc, ids[1], nothing)  # a head before the newest checkpoint: from the checkpoint before it
    assert start["version_id"] == ids[0] and [step["version_id"] for step in steps] == ids[1:2]
    start, steps = route(doc, ids[-1], lambda head: head["version_id"] in (ids[1], ids[3]))  # a held head is nearer than the checkpoint
    assert start["version_id"] == ids[3] and [step["version_id"] for step in steps] == ids[4:]
    start, steps = route(doc, ids[-1], lambda head: True)
    assert start["version_id"] == ids[-1] and steps == []
    with pytest.raises(VerifyError, match="not a folded head of this chain"):
        route(doc, world.state.manifests[doc["steps"][0]["stacked_version_id"]].head.version_id, nothing)


def test_a_fold_without_its_own_fold_record_does_not_move_the_head(world):
    world.start(CHAIN)
    real = world.controller.folder

    class Edited:
        """The real fold, then a change of its record."""

        def __init__(self, edit) -> None:
            self.edit = edit

        def fold(self, head, params, out_prefix):
            fold = real.fold(head, params, out_prefix)
            self.edit(f"{out_prefix}/{FOLD_RECORD}")
            return fold

    def other(key: str) -> None:
        world.store.put(key, json.dumps(json.loads(world.store.get(key)) | {"stacked_version_id": "0" * 64}).encode())

    def no_kernel(key: str) -> None:
        world.store.put(key, json.dumps(json.loads(world.store.get(key)) | {"kernel": ""}).encode())

    world.register(90, 0.3)
    for edit, text in ((world.store.delete, "no complete fold record"), (other, "of another fold"), (no_kernel, "of another fold")):
        world.controller.folder = Edited(edit)
        world.now += world.params.apply_retry_s
        world.round()
        assert not world.state.head.folded and text in world.controller.last_error
        assert not world.store.list("public/head_folds")
    world.controller.folder = real
    world.now += world.params.apply_retry_s
    world.round()
    assert world.state.head.folded and world.chain_file()["steps"][-1]["kernel"] == FIXED_ORDER
    assert world.store.list("public/head_folds") == [layout.head_fold(world.state.head.parent_version_id)]


def test_only_a_marker_key_makes_a_head_a_checkpoint(world):
    world.start(CHAIN)
    head = world.king().head
    assert checkpoints(world.store) == set()
    for key in ("public/head_checkpoints/deep/x.json", f"public/head_checkpoints/{head.version_id}.txt", f"public/head_checkpoints/{head.version_id}.json.bak"):
        world.store.put(key, b"{}")
    assert checkpoints(world.store) == set() and world.writer.doc(world.state)["start"]["steps_to_head"] == 1
    world.store.put(layout.head_checkpoint(head.version_id), b"{}")  # as another process writes one
    assert checkpoints(world.store) == {head.version_id} and world.writer.doc(world.state)["start"]["steps_to_head"] == 0


def test_the_chain_file_is_written_only_when_it_changed_and_a_marker_shows_at_once(world):
    world.start(CHAIN)
    head = world.king().head
    writer: ChainWriter = world.writer
    count = world.store.wrote.count(layout.HEAD_CHAIN)
    assert writer(world.state) == tail(world.chain_file()) and world.store.wrote.count(layout.HEAD_CHAIN) == count
    writer.mark("0" * 64)  # not a head of the chain: no write
    assert world.store.wrote.count(layout.HEAD_CHAIN) == count
    writer.mark(head.version_id)
    doc = world.chain_file()
    assert world.store.wrote.count(layout.HEAD_CHAIN) == count + 1 and doc["steps"][-1]["checkpoint"] is True
    assert doc["start"]["version_id"] == head.version_id and doc["start"]["steps_to_head"] == 0
    writer.mark(head.version_id)
    assert world.store.wrote.count(layout.HEAD_CHAIN) == count + 1


def test_the_links_of_a_chain_are_the_checkpoints_and_the_loras_of_its_steps(world):
    world.king()
    world.start(CHAIN)
    for _ in range(2):
        world.king()
    world.wait()
    doc, heads = world.chain_file(), folded(world)
    links = chain_links(world.store, doc, 600)
    marked = [head for head in heads_of(doc) if head["checkpoint"]]
    assert [point["version_id"] for point in links["checkpoints"]] == [head["version_id"] for head in marked] and len(marked) == 2
    for point, head in zip(links["checkpoints"], marked):
        manifest = world.state.manifests[head["version_id"]]
        assert point["base"]["id"] == head["base_hash"] and point["master"]["id"] == head["base_fp32_master_digest"]
        assert {row["name"]: row["sha256"] for row in point["base"]["files"]} == manifest.base_files
        assert {row["name"]: row["sha256"] for row in point["master"]["files"]} == manifest.master_files
        assert point["master"]["bytes"] == sum(row["size"] for row in point["master"]["files"]) > 0
    assert [row["id"] for row in links["chain_loras"]] == [lora["lora_hash"] for step in doc["steps"] for lora in step["loras"]]

    newest = chain_links(world.store, doc, 600, target=heads[-1].head.version_id)  # what a reader of the head needs, and no more
    assert [point["version_id"] for point in newest["checkpoints"]] == [doc["start"]["version_id"]]
    assert [row["id"] for row in newest["chain_loras"]] == [lora["lora_hash"] for step in doc["steps"][-2:] for lora in step["loras"]]

    assert not any(is_weight(key) for key in world.store.list(layout.public_base_prefix(heads[-1].head.base_hash)))
    unmarked = copy.deepcopy(doc)
    unmarked["steps"][-1]["checkpoint"] = True  # the chain says so, the store has no weight file of that head
    with pytest.raises(KeyError):
        chain_links(world.store, unmarked, 600)
    assert small_files(heads[-1].base_files) and all(not is_weight(layout.public_base_file("x", name)) for name in small_files(heads[-1].base_files))
    swapped = copy.deepcopy(doc)
    swapped["genesis"]["base_hash"] = heads[-1].head.base_hash
    with pytest.raises(VerifyError, match="is of another head"):
        chain_links(world.store, swapped, 600)


def test_llms_txt_names_the_chain_the_rebuild_command_and_the_implementation_setting_only_when_the_box_keeps_the_weight_files(world):
    world.king()

    def text() -> tuple[str, dict]:
        mining, llms = publisher.render(json.loads(json.dumps(world.publisher.snapshots[-1])), "https://site.example")
        return " ".join(llms.decode().split()), json.loads(mining)

    words = ("head_chain", "checkpoint", publisher.REBUILD_MODULE, publisher.FOLD_IMPLEMENTATION_ENV, FIXED_ORDER, MATMUL)
    flat, mining = text()
    assert "head_chain" not in mining and not [word for word in words if word in flat], "full mode: the text of before"
    world.start(CHAIN)
    head = world.king()
    flat, mining = text()
    chain = mining["head_chain"]
    assert chain["head_version_id"] == head.head.version_id and not [word for word in words if word not in flat]
    assert f"At this snapshot the newest checkpoint is head `{chain['start']['version_id']}`. The base of head M is {len(chain['steps'])} fold(s) after it (kernel `{FIXED_ORDER}`)." in flat
    assert f"`python -m {publisher.REBUILD_MODULE} rebuild --chain https://site.example/mining.json --work <directory>`" in flat
    assert f"`{publisher.FOLD_IMPLEMENTATION_ENV}=blas`" in flat and "`refused: `" in flat and f"`{layout.HEAD_CHAIN}`" in flat
    assert publisher.REBUILD_COST in flat and "21 minutes on 4 CPU cores" in publisher.REBUILD_COST
    assert publisher.REBUILD_MODULE == headfold.__name__ and publisher.FOLD_IMPLEMENTATION_ENV == headfold.IMPLEMENTATION_ENV
    assert headfold.REFERENCE == "reference" and headfold.BLAS == "blas", "the two values that the text names"
    world.controller.head.checkpoint(world.state)
    world.wait()
    world.register(world.uid, 0.0)  # a publish after the marker
    world.round()
    flat, mining = text()
    assert mining["head_chain"]["start"]["version_id"] == head.head.version_id and mining["head_chain"]["steps"] == []
    assert f"At this snapshot the newest checkpoint is head `{head.head.version_id}`. The base of head M is that checkpoint." in flat
