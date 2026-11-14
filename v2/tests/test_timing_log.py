"""The timing lines of the promotion, the fold, the close and the open: in the log of ``run`` only."""

from __future__ import annotations

import re

import pytest

from sn120_v2.errors import VerifyError
from sn120_v2.journal import JOURNAL_FILE
from sn120_v2.sim import SimWorld, sim_params
from sn120_v2.store import LocalStore, promote_files, timed

LINE = re.compile(r"timing \d\d:\d\d:\d\d UTC: (.+): (ended by (\w+) after )?\d+\.\d s")


def timing(capsys) -> list[str]:
    """The step of each timing line that was logged since the last call. A failed step ends with the name of its error."""
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("timing ")]
    matches = [LINE.fullmatch(line) for line in lines]
    assert all(matches), lines
    return [match[1] + (f" [{match[3]}]" if match[3] else "") for match in matches]


def test_a_promotion_logs_each_of_its_steps_and_a_second_one_logs_that_nothing_was_copied(tmp_path, capsys):
    store = LocalStore(tmp_path / "store")
    store.put("pending/x/a.bin", b"weights")
    store.put("pending/x/b.bin", b"more")
    files = {name: store.sha256(f"pending/x/{name}") for name in ("a.bin", "b.bin")}
    promote_files(store, "pending/x", "public/bases/abc", files)
    assert timing(capsys) == [
        "public/bases/abc: no verified copy is there yet",
        "public/bases/abc: the check of 2 files in pending/",
        "public/bases/abc: the copy of 2 files",
        "public/bases/abc: the check of the 2 copied files",
    ]
    promote_files(store, "pending/x", "public/bases/abc", files)
    assert timing(capsys) == ["public/bases/abc: a verified copy is there already, so nothing is copied"]
    store.put("pending/y/a.bin", b"other bytes")
    with pytest.raises(VerifyError):
        promote_files(store, "pending/y", "public/bases/def", {"a.bin": files["a.bin"]})
    assert timing(capsys) == ["public/bases/def: no verified copy is there yet", "public/bases/def: the check of 1 files in pending/ [VerifyError]"]
    assert store.list("public/bases/def") == [], "a step that fails is logged, and nothing is copied"


def test_a_step_that_an_error_ends_is_logged_with_the_error_and_the_error_goes_on(capsys):
    with pytest.raises(KeyError):
        with timed("a step"):
            raise KeyError("x")
    assert timing(capsys) == ["a step [KeyError]"]


def test_a_round_with_a_king_logs_the_time_of_each_step_and_no_record_or_public_file_holds_one(tmp_path, capsys):
    world = SimWorld(tmp_path, sim_params())
    for uid, quality in ((1, 0.3), (2, 0.0)):
        world.register(uid, quality)
    capsys.readouterr()
    record = world.round()
    assert record.outcome == "applied" and world.controller.state.head.folded
    steps = timing(capsys)
    head = world.controller.state.heads[-2].version_id[:12]  # the head with the king's LoRA, before the fold
    wanted = [
        "open: the check of the waiting files",
        f"open: round {record.round_id} is open, 2 seats. From the file check to here",
        f"open: the public files of round {record.round_id}",
        f"king of {record.round_id}: the winner's LoRA to public/",
        f"king of {record.round_id}: the winner's batch file to public/",
        f"fold of {head}: the box (the sum, the upload to pending/ and its check)",
        f"fold of {head}: the promotion of the base and the master, all steps",
        f"fold of {head}: the delete of its files in pending/",
        f"close of {record.round_id} (applied): the task table, the round file and the record",
        f"close of {record.round_id}: the head pointer",
        f"close of {record.round_id}: the delete of its files in pending/",
        f"close of {record.round_id}: the public files",
    ]
    assert [step for step in steps if not step.startswith("public/")] == wanted
    promoted = [step for step in steps if step.startswith("public/")]
    assert [step.split("/")[1] for step in promoted if step.endswith("copied files")] == ["loras", "bases", "masters"]
    assert steps.index(wanted[6]) - steps.index(wanted[5]) == 9, "the base and the master: four lines each, inside the fold's promotion"
    stored = [world.store.get(key) for key in world.store.list("public/")] + [(world.root / JOURNAL_FILE).read_bytes()]
    assert stored and not any(b"timing" in data for data in stored)
