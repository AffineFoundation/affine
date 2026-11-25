from __future__ import annotations

import pytest

from sn120_v2.errors import ContractError, FencedError, JournalCorrupt, VerifyError
from sn120_v2.journal import Journal, read_journal
from sn120_v2.seams import ObjectStore
from sn120_v2.store import LocalStore, delete_pending, promote_files, verify_files


def test_journal_keeps_records_in_order_with_a_hash_chain(tmp_path):
    journal = Journal(tmp_path / "j.jsonl", "a", at=1)
    journal.append("x", {"n": 1}, at=2)
    journal.append("y", {"n": 2}, at=3)
    records = read_journal(tmp_path / "j.jsonl")
    assert [(r.seq, r.kind) for r in records] == [(0, "leader"), (1, "x"), (2, "y")]
    assert records[0].prev == "" and all(r.prev for r in records[1:])
    assert records == journal.records


def test_a_second_writer_fences_the_first(tmp_path):
    first = Journal(tmp_path / "j.jsonl", "first", at=1)
    second = Journal(tmp_path / "j.jsonl", "second", at=2)
    assert second.fence == first.fence + 1
    with pytest.raises(FencedError):
        first.append("x", {}, at=3)
    second.append("y", {}, at=4)
    assert [r.kind for r in read_journal(tmp_path / "j.jsonl")] == ["leader", "leader", "y"]


def test_a_torn_last_line_is_dropped_and_any_other_damage_fails_closed(tmp_path):
    path = tmp_path / "j.jsonl"
    journal = Journal(path, "a", at=1)
    journal.append("x", {"n": 1}, at=2)
    good = path.read_bytes()
    path.write_bytes(good + b'{"seq": 2, "kind": "half')
    assert [r.kind for r in read_journal(path)] == ["leader", "x"]
    assert Journal(path, "b", at=3).records[-1].kind == "leader" and b"half" not in path.read_bytes()
    lines = good.split(b"\n")
    for damaged in (lines[1] + b"\n", good.replace(b'"n":1', b'"n":9'), lines[0] + b"\nnot json\n"):
        path.write_bytes(damaged)
        with pytest.raises(JournalCorrupt):
            read_journal(path)


def test_local_store_follows_the_object_store_rules(tmp_path):
    store = LocalStore(tmp_path / "store")
    assert isinstance(store, ObjectStore)
    for key in ("pending/rounds/r1/r1-l1/a", "pending/rounds/r1/r1-l10/b", "pending/rounds/r1/r1-l1/sub/c", "public/x"):
        store.put(key, key.encode())
    assert store.list("pending/rounds/r1/r1-l1") == ["pending/rounds/r1/r1-l1/a", "pending/rounds/r1/r1-l1/sub/c"]
    assert store.list("pending/rounds/r1/r1-l1/a") == ["pending/rounds/r1/r1-l1/a"]
    assert store.list("pending/rounds/r1/r1-l") == [], "a prefix matches whole path segments only"
    assert store.list("public/") == ["public/x"] and len(store.list("")) == 4
    for missing in (store.get, store.size, store.sha256, lambda k: store.copy(k, "public/y"), lambda k: store.get_file(k, tmp_path / "out")):
        with pytest.raises(KeyError):
            missing("public/absent")
    assert not store.exists("public/absent")
    store.delete("public/absent")
    store.put("public/x", b"new")
    store.copy("public/x", "public/x")
    store.copy("public/x", "pending/rounds/r1/r1-l1/a")
    assert store.get("pending/rounds/r1/r1-l1/a") == b"new" and store.size("public/x") == 3
    store.get_file("public/x", tmp_path / "deep" / "dir" / "x.bin")
    assert (tmp_path / "deep" / "dir" / "x.bin").read_bytes() == b"new"
    for unsafe in ("", "/abs", "a/../b", "a//b", "a/./b"):
        with pytest.raises(ContractError):
            store.put(unsafe, b"")


def test_promotion_verifies_before_and_after_the_copy_and_fails_closed(tmp_path, build):
    store = LocalStore(tmp_path / "store")
    store.put("pending/x/w.bin", b"weights")
    files, sizes = {"w.bin": build.h("unused")}, {"w.bin": 7}
    with pytest.raises(VerifyError, match="sha256"):
        promote_files(store, "pending/x", "public/loras/abc", files, sizes)
    assert store.list("public/") == [], "nothing is promoted on a wrong hash"
    files = {"w.bin": store.sha256("pending/x/w.bin")}
    with pytest.raises(VerifyError, match="size"):
        verify_files(store, "pending/x", files, {"w.bin": 8})
    with pytest.raises(VerifyError, match="missing"):
        verify_files(store, "pending/absent", files)
    promote_files(store, "pending/x", "public/loras/abc", files, sizes)
    assert store.get("public/loras/abc/w.bin") == b"weights"
    store.delete("pending/x/w.bin")
    promote_files(store, "pending/x", "public/loras/abc", files, sizes)  # already promoted: nothing to do


def test_cleanup_can_only_delete_staged_objects(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.put("pending/rounds/r1/l/a", b"1")
    store.put("public/bases/head/model.bin", b"2")
    assert delete_pending(store, "pending/rounds/r1") == 1
    for protected in ("public/bases/head", "private/rounds/r1", "", "pending", "pending/", "pending/rounds", "pending/rounds/", "pending//r1", "pendingx/rounds/r1"):
        with pytest.raises(ContractError):
            delete_pending(store, protected)
    assert store.exists("public/bases/head/model.bin")
    store.put("pending/rounds/r10/l/a", b"3")
    assert delete_pending(store, "pending/rounds/r1") == 0 and store.exists("pending/rounds/r10/l/a")
