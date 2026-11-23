"""The operator's way out when ``run`` refuses a chain sync: ``force-sync``.

The state is the one that commit 515d7e9f wrote (``tests/data/state_515d7e9f/idle``):
57 waiting seats, hk4 to hk60. The chain here is a list of registrations that the test changes.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pytest

from sn120_v2 import cli, sim
from sn120_v2.cli import main
from sn120_v2.journal import JOURNAL_FILE, read_journal
from sn120_v2.sim import STORE_DIR, SimAdmission
from sn120_v2.state import load, replay
from sn120_v2.store import LocalStore
from sn120_v2.types import ChainSnapshot, Registration

STATE = Path(__file__).parent / "data" / "state_515d7e9f" / "idle"
RUN = ["--rounds", "1", "--max-idle", "1", "--idle-sleep", "0"]


class Chain:
    """A chain that the test edits. Each read is at a later block, unless ``stuck``."""

    def __init__(self) -> None:
        self.uids = set(range(1, 61))
        self.block, self.stuck, self.reads = 500, False, 0
        self.leave_at_read: dict[int, int] = {}  # read number -> a UID that leaves the chain right before that read

    def __call__(self) -> ChainSnapshot:
        self.reads += 1
        self.block += 0 if self.stuck else 1
        self.uids.discard(self.leave_at_read.get(self.reads, 0))
        regs = (Registration(0, "owner", 1), *(Registration(uid, f"hk{uid}", 100 + uid, 1) for uid in sorted(self.uids)))
        return ChainSnapshot(120, self.block, 60 + self.reads, regs)  # the clock time of a read differs from read to read


@pytest.fixture
def box(tmp_path, monkeypatch):
    directory = tmp_path / "state"
    shutil.copytree(STATE, directory)
    chain = Chain()
    cpu_stack = sim.stack
    monkeypatch.setattr(sim, "stack", lambda context: dataclasses.replace(cpu_stack(context), chain=chain))
    monkeypatch.setattr(cli, "CHAIN_READ_WAIT_S", 0.0)
    chain.directory = directory
    chain.common = ["--state-dir", str(directory), "--stack", "sn120_v2.sim:stack"]
    return chain


def records(box) -> list:
    return [record.kind for record in read_journal(box.directory / JOURNAL_FILE)]


def test_run_refuses_a_sync_that_removes_many_seats_and_the_operator_applies_it_once(box, capsys):
    assert len(load(box.directory / JOURNAL_FILE).queue) == 57
    box.uids -= set(range(41, 61))  # 20 of the 57 seats left the chain while run was down: more than a quarter
    SimAdmission(LocalStore(box.directory / STORE_DIR)).upload("hk4", 0.3)
    assert main(["run", *box.common, *RUN]) == 0
    err = capsys.readouterr().err
    assert "waiting: sync would remove 20 of 57 seats; refusing without force" in err and "force-sync" in err, "run never forces, and names the command"
    assert len(load(box.directory / JOURNAL_FILE).queue) == 57 and len(load(box.directory / JOURNAL_FILE).rounds) == 1, "no round opens"
    before = records(box)

    assert main(["force-sync", *box.common]) == 0, "without the count: the list only"
    out = capsys.readouterr().out
    assert "(41 registrations) give the same changes to the queue of 57 seats" in out and "remove 20 seats, add 0 seats, change the hotkey of 0 seats" in out
    assert out.count("\n  remove uid") == 20 and "remove uid   41 registration block       141 hk41 (normal)" in out
    assert out.rstrip().endswith("give the same command with --confirm-remove 20")
    assert records(box) == before, "the list writes nothing and does not take over the journal"

    assert main(["force-sync", *box.common, "--confirm-remove", "19"]) == 2, "another count than the list shows"
    assert "the chain now removes 20 seats, not the confirmed 19; nothing was changed" in capsys.readouterr().err
    assert len(load(box.directory / JOURNAL_FILE).queue) == 57

    assert main(["force-sync", *box.common, "--confirm-remove", "20"]) == 0
    assert "applied at chain block" in capsys.readouterr().out
    state = load(box.directory / JOURNAL_FILE)
    assert sorted(seat.reg.uid for seat in state.queue.values()) == list(range(4, 41)) and records(box)[-1] == "registrations_synced"
    assert replay(read_journal(box.directory / JOURNAL_FILE)).queue == state.queue

    assert main(["run", *box.common, *RUN]) == 0, "run goes on by itself"
    out = capsys.readouterr()
    assert "waiting:" not in out.err and "SN120 v2 round r000002: APPLIED" in out.out
    assert main(["force-sync", *box.common]) == 0 and "remove 0 seats" in capsys.readouterr().out


def test_force_sync_changes_nothing_when_two_chain_reads_differ_or_the_chain_gives_no_later_block(box, capsys):
    box.uids -= set(range(41, 61))
    before = load(box.directory / JOURNAL_FILE).queue
    box.leave_at_read = {box.reads + 2: 40}  # one more seat is gone in the second read
    assert main(["force-sync", *box.common, "--confirm-remove", "20"]) == 2
    err = capsys.readouterr().err
    assert "two chain reads give different changes of the queue (remove 20, add 0 at block 501 and remove 21, add 0 at block 502)" in err
    assert load(box.directory / JOURNAL_FILE).queue == before
    box.stuck = True
    assert main(["force-sync", *box.common, "--confirm-remove", "21"]) == 2
    assert "the chain stayed at block 502 for 0 s" in capsys.readouterr().err and load(box.directory / JOURNAL_FILE).queue == before
    box.stuck, box.uids = False, set()
    assert main(["force-sync", *box.common]) == 0, "a chain with none of the seats, as from another network: the list says so"
    assert "WARNING: not one of the 57 seats is on this chain" in capsys.readouterr().out and load(box.directory / JOURNAL_FILE).queue == before


def test_force_sync_needs_a_stack_that_reads_the_chain(tmp_path, capsys):
    shutil.copytree(STATE, tmp_path / "state")
    assert main(["force-sync", "--state-dir", str(tmp_path / "state"), "--stack", "sn120_v2.sim:stack"]) == 2
    assert "this stack reads no chain" in capsys.readouterr().err
