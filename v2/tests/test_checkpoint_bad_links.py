"""A links directory of a checkpoint that holds a bad file (``headfold.HeadKeeper``).

The links directory ``<work>/checkpoint/<version_id>/`` holds hard links of the files of one head, for the upload
in the background. A file in it can be bad: ``overlap`` makes the links before the second sha256 of the head
ended, and a disk can change a file after its check in each order. Two rules are proven here.

* No file goes to a public key before its sha256 is the one that the chain says for that key: in each order, in
  the thread of ``run``, with the command ``checkpoint``, and after each restart.
* A links directory with a bad file does not stop the checkpoints of later heads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sn120_v2.sim_head import ChainWorld  # noqa: E402 - after the skip guard

from sn120_v2 import layout  # noqa: E402
from sn120_v2.afterking import OVERLAP, SERIAL  # noqa: E402
from sn120_v2.canonical import file_sha256  # noqa: E402
from sn120_v2.errors import HeadLost, VerifyError  # noqa: E402
from sn120_v2.gpu.resident import hash_tree  # noqa: E402
from sn120_v2.headchain import CHAIN, check_chain, checkpoints  # noqa: E402
from sn120_v2.headfold import MASTER, inline  # noqa: E402
from sn120_v2.sim import sim_params  # noqa: E402
from sn120_v2.types import HeadManifest  # noqa: E402

ORDERS = [SERIAL, OVERLAP]
BAD_LINKS = "do not hash to the record of the chain"


def a_world(path: Path, order: str, every: int = 1) -> ChainWorld:
    world = ChainWorld(path, sim_params(), order=order, every=every)
    world.start(CHAIN)
    return world


def links(world: ChainWorld) -> list[str]:
    root = world.box.work_dir / "checkpoint"
    return sorted(path.name for path in root.iterdir()) if root.is_dir() else []


def master_keys(manifest: HeadManifest) -> dict[str, str]:
    """{public key: sha256 by the chain} of each master file of one head."""
    return {layout.public_master_file(manifest.head.base_fp32_master_digest, name): sha for name, sha in manifest.master_files.items()}


def base_keys(manifest: HeadManifest) -> dict[str, str]:
    return {layout.public_base_file(manifest.head.base_hash, name): sha for name, sha in manifest.base_files.items()}


def public_is_true(world: ChainWorld) -> None:
    """Each weight file under a public key of a head of the journal has the sha256 that the journal says for that key."""
    wanted: dict[str, str] = {}
    for manifest in world.state.manifests.values():
        wanted |= master_keys(manifest) | base_keys(manifest)
    for key in world.store.weights():
        assert world.store.sha256(key) == wanted[key], f"{key} holds other bytes than the chain says"


def never_sent(world: ChainWorld, key: str) -> bool:
    return key not in world.store.wrote and not world.store.exists(key)


def a_bad_file_in_the_links(world: ChainWorld, manifest: HeadManifest) -> str:
    """One master file of the links directory of ``manifest`` gets other bytes, as a file that a disk changed. Returns its public key.

    The link is replaced by a new file, so the file of the cache stays as it is.
    """
    name = sorted(manifest.master_files)[0]
    path = world.box.work_dir / "checkpoint" / manifest.head.version_id / MASTER / name
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.unlink()
    path.write_bytes(bytes(raw))
    return layout.public_master_file(manifest.head.base_fp32_master_digest, name)


def changed_on_disk(world: ChainWorld) -> None:
    """Each fold from now writes a master file whose bytes on the disk are not the bytes that it hashed: a fault of the disk."""

    def faulty(jobs):
        results = inline(jobs)
        target = sorted(file for file in jobs[0].out_master.iterdir() if file.is_file())[0]
        raw = bytearray(target.read_bytes())
        raw[-1] ^= 1
        target.write_bytes(bytes(raw))
        return results

    world.folder.run = faulty


def new_run(world: ChainWorld) -> None:
    """The process of ``run`` died and the loop started a new one: nothing is verified in its memory, and its start proves the local head."""
    world.wait()
    if getattr(world.folder, "_read_back", None) is not None:
        world.folder._read_back.wait()  # noqa: SLF001 - a dead process leaves no thread; the test waits for it instead
    world.folder.resident._verified.clear()  # noqa: SLF001
    world.folder.run = inline
    world.start()


def stopped_upload(world: ChainWorld) -> ChainWorld:
    """A world with one checkpoint, and then a king whose upload did not start: its links directory is on the disk."""
    world.king()
    world.wait()
    world.store.refuse = lambda key: key.startswith("public/masters/")  # in the mode chain only a checkpoint writes there
    return world


@pytest.mark.parametrize("order", ORDERS)
def test_a_bad_file_of_a_links_directory_reaches_no_public_key_and_the_directory_is_removed(tmp_path, order, capsys):
    world = stopped_upload(a_world(tmp_path / "world", order))
    manifest = world.king()
    world.wait()
    assert links(world) == [manifest.head.version_id] and manifest.head.version_id not in checkpoints(world.store)
    key = a_bad_file_in_the_links(world, manifest)
    world.store.refuse = None
    world.store.wrote.clear()
    capsys.readouterr()
    world.controller.head.after_round(world.state)  # the next pass of ``run`` starts the upload again
    world.wait()
    assert never_sent(world, key), "the bad file was sent to its public key"
    public_is_true(world)
    said = capsys.readouterr().err
    assert BAD_LINKS in said and manifest.head.version_id[:12] in said
    assert manifest.head.version_id not in checkpoints(world.store), "no marker for a head whose files were not all sent"
    # The cache holds the true files of this head, so the next pass links them again and the checkpoint is made.
    world.controller.head.after_round(world.state)
    world.wait()
    assert manifest.head.version_id in checkpoints(world.store) and links(world) == []
    assert world.store.sha256(key) == master_keys(manifest)[key]
    public_is_true(world)


@pytest.mark.parametrize("order", ORDERS)
def test_a_links_directory_with_a_bad_file_does_not_stop_the_checkpoints_of_later_heads(tmp_path, order):
    world = stopped_upload(a_world(tmp_path / "world", order))
    old = world.king()
    world.wait()
    key = a_bad_file_in_the_links(world, old)
    world.store.refuse = None
    later = [world.king() for _ in range(3)]
    world.wait()
    world.controller.head.after_round(world.state)
    world.wait()
    marked = checkpoints(world.store)
    assert [manifest.head.version_id in marked for manifest in later] == [True, True, True], "each later head has its marker"
    assert old.head.version_id not in marked and never_sent(world, key), "the old head is not the head of the cache any more: it gets no checkpoint, and its bad file went nowhere"
    assert links(world) == [], "no links directory stays on the disk"
    public_is_true(world)
    check_chain(world.chain_file())


def test_a_head_whose_second_sha256_failed_after_its_links_were_made_is_not_sent_by_the_next_run_and_the_kings_after_it_get_their_checkpoints(tmp_path, capsys):
    """The fault of the review of PR 225, with ``overlap`` and a checkpoint for each king."""
    world = a_world(tmp_path / "world", OVERLAP)
    world.king()
    world.wait()
    changed_on_disk(world)
    bad = world.king()  # ``after_round`` made the links of this head; its second sha256 fails
    world.wait()
    with pytest.raises(HeadLost):
        world.folder.proven(True)
    assert links(world) == [bad.head.version_id], "the links of the head were made before its second sha256 ended"
    bad_key = next(key for key, sha in master_keys(bad).items() if file_sha256(world.box.work_dir / "checkpoint" / bad.head.version_id / MASTER / Path(key).name) != sha)
    world.store.wrote.clear()
    capsys.readouterr()
    new_run(world)  # ``run`` stopped with 75; this start rebuilds the head from the chain
    world.wait()
    said = capsys.readouterr().err
    assert f"the local files of head {bad.head.version_id[:12]} are not there" in said and BAD_LINKS in said
    assert never_sent(world, bad_key), "the new run sent the bad file of the old links to its public key"
    public_is_true(world)
    world.controller.head.after_round(world.state)  # the next pass: the links are made again, from the rebuilt head
    world.wait()
    assert bad.head.version_id in checkpoints(world.store) and world.store.sha256(bad_key) == master_keys(bad)[bad_key]
    later = [world.king() for _ in range(3)]
    world.wait()
    marked = checkpoints(world.store)
    assert [manifest.head.version_id in marked for manifest in later] == [True, True, True] and links(world) == []
    public_is_true(world)
    check_chain(world.chain_file())
    assert hash_tree(world.local(world.head.head.base_fp32_master_digest)) == world.head.master_files


@pytest.mark.parametrize("order", ORDERS)
def test_a_file_that_is_bad_in_the_cache_too_is_linked_one_more_time_and_then_no_more_in_this_run(tmp_path, order, capsys):
    """The links are made again from the cache one time. When they are bad again, the head gets no checkpoint in this run, and no pass sends a byte of it."""
    world = stopped_upload(a_world(tmp_path / "world", order))
    manifest = world.king()
    world.wait()
    if world.controller.head.proven is not None:
        world.controller.head.proven(True)  # the second sha256 of the head passed: the file changes after it
    name = sorted(manifest.master_files)[0]
    key = layout.public_master_file(manifest.head.base_fp32_master_digest, name)
    path = world.local(manifest.head.base_fp32_master_digest) / name
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(bytes(raw))  # the same file as the link: the disk changed it after its check
    world.store.refuse = None
    world.store.wrote.clear()
    capsys.readouterr()
    for _ in range(4):
        world.controller.head.after_round(world.state)
        world.wait()
    said = capsys.readouterr().err
    assert said.count(BAD_LINKS) == 2 and "no checkpoint of this head is made in this run" in said
    assert never_sent(world, key) and links(world) == [] and manifest.head.version_id not in checkpoints(world.store)
    public_is_true(world)


@pytest.mark.parametrize("order", ORDERS)
def test_the_checkpoint_command_refuses_a_bad_file_and_sends_nothing_of_it(tmp_path, order):
    world = a_world(tmp_path / "world", order, every=0)
    manifest = world.king()
    world.keeper.link(manifest)
    key = a_bad_file_in_the_links(world, manifest)
    world.store.wrote.clear()
    with pytest.raises(VerifyError, match=BAD_LINKS):
        world.controller.head.checkpoint(world.state)
    assert never_sent(world, key) and links(world) == [] and manifest.head.version_id not in checkpoints(world.store)
    public_is_true(world)
    assert world.controller.head.checkpoint(world.state) == {"version_id": manifest.head.version_id, "checkpoint": True, "uploaded": True}, "the next call links the true files of the cache"
    public_is_true(world)


def test_a_job_for_links_that_are_gone_is_dropped_and_stops_no_later_checkpoint(tmp_path, capsys):
    """A pass of ``run`` can list a links directory in the instant before the upload thread removes it. The job that it then adds has no files."""
    world = stopped_upload(a_world(tmp_path / "world", OVERLAP))
    manifest = world.king()
    world.wait()
    a_bad_file_in_the_links(world, manifest)
    world.store.refuse = None
    world.controller.head.after_round(world.state)
    world.wait()
    assert links(world) == []
    capsys.readouterr()
    world.keeper._jobs[manifest.head.version_id] = manifest  # noqa: SLF001 - the job of that pass
    world.keeper._work()  # noqa: SLF001
    assert not world.keeper._jobs and "stopped by" not in capsys.readouterr().err  # noqa: SLF001
    later = world.king()
    world.wait()
    assert later.head.version_id in checkpoints(world.store)
    public_is_true(world)
