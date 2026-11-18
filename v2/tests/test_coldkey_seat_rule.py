"""One seat for each coldkey in a round. The coldkey comes from the chain read of the registrations.

A queue that an older commit saved has seats with the coldkey "". The first sync fills them: no seat moves,
``served_hotkeys`` stays, and an open round stays as it is.
"""

from __future__ import annotations

import dataclasses
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from sn120_v2 import publisher, sim
from sn120_v2.chain import FakeChain, Reader, SubtensorClient
from sn120_v2.cli import main
from sn120_v2.errors import InfraError
from sn120_v2.journal import JOURNAL_FILE, read_journal
from sn120_v2.queue import plan_sync, waiting
from sn120_v2.sim import STORE_DIR, SimAdmission, SimWorld, sim_params
from sn120_v2.state import REGISTRATIONS_SYNCED, load, replay
from sn120_v2.store import LocalStore
from sn120_v2.types import ChainSnapshot

NOW = 1_000_000
STATES = Path(__file__).parent / "data" / "state_515d7e9f"
RUN = ["--rounds", "1", "--max-idle", "1", "--idle-sleep", "0"]
ITEMS = ("SubnetworkN", "Keys", "BlockAtRegistration", "Owner")
SDK = SimpleNamespace(storage=SimpleNamespace(SubtensorModule=SimpleNamespace(**{name: name for name in ITEMS})))


class Subtensor:
    """The SDK calls of one registration read, with the names and shapes read from bittensor 11.0.2.

    ``owners`` is hotkey -> coldkey, in UID order. Storage ``Owner`` is the chain's map from a hotkey to its coldkey.
    """

    block = 900

    def __init__(self, owners: dict[str, object]) -> None:
        self.owners, self.batches = dict(owners), []

    def block_info(self, block):
        return SimpleNamespace(number=block, hash=f"0xhash{block}", timestamp=datetime.fromtimestamp(NOW, timezone.utc))

    def query(self, item, params, *, block=None):
        return {"SubnetworkN": len(self.owners)}[item]

    def query_map(self, item, params, *, block=None):
        return [(uid, hotkey if item == "Keys" else 800 + uid) for uid, hotkey in enumerate(self.owners)]

    def query_batch(self, item, param_sets, *, block=None):
        self.batches.append((item, param_sets, block))
        return [self.owners[hotkey] for (hotkey,) in param_sets]


def sdk_reader(chain: Subtensor) -> Reader:
    return Reader(SubtensorClient(120, chain, sdk=SDK), 120, clock=lambda: NOW)


def seats(record) -> list[int]:
    return [entry.reg.uid for lease in record.leases for entry in lease.entries]


def sync_records(journal: Path) -> list:
    return [record for record in read_journal(journal) if record.kind == REGISTRATIONS_SYNCED]


def test_two_seats_of_one_coldkey_do_not_sit_in_one_round_and_the_second_sits_in_the_next(tmp_path):
    world = SimWorld(tmp_path, sim_params())
    chain = Subtensor({"owner": "ck-owner", "hk1": "ck-a", "hk2": "ck-a", "hk3": "ck-b"})  # UID 0 to 3
    world.controller.sync_registrations(sdk_reader(chain).snapshot(), world.clock())
    for hotkey in ("hk1", "hk2", "hk3"):
        world.admission.upload(hotkey)  # each of the three seats has a valid file
    first = world.round()
    assert seats(first) == [1, 3], "hk1 and hk2 have one coldkey: only the first of them sits"

    # the second seat waits in its place: it is not a skip, it has no flag, and nothing is charged or counted for it
    state = world.controller.state
    (waits,) = state.queue.values()
    assert first.skipped == () and waits.reg.uid == 2 and waits.reg.coldkey == "ck-a"
    assert not (waits.front or waits.demoted or waits.held or waits.leased) and waits.infra_attempts == 0
    assert state.served_hotkeys == {"hk1", "hk3"} and waits.reg.key not in state.served_regs
    shown = world.publisher.snapshots[-1]["queue"]["next"]  # what mining.json and the website show for it
    assert shown == [{"position": 0, "uid": 2, "hotkey": "hk2", "registration_block": 802, "file_valid": True, "front": False, "demoted": False, "held": False}]
    mining, text = (raw.decode() for raw in publisher.render(world.publisher.snapshots[-1]))
    assert "A second seat of the same hotkey or coldkey waits for a later round." in text and "| 0 | 2 | hk2 | 802 | yes |" in text
    assert "ck-a" not in mining + text, "the coldkey itself is in no public file"

    # the next round has no seat of its coldkey, so it sits there, before a seat that registered later
    chain.owners |= {"hk4": "ck-c"}
    world.admission.upload("hk4")
    world.controller.sync_registrations(sdk_reader(chain).snapshot(), world.clock())
    second = world.round()
    assert seats(second) == [2, 4] and world.controller.state.served_hotkeys == {"hk1", "hk2", "hk3", "hk4"}
    assert replay(read_journal(world.root / JOURNAL_FILE)).queue == world.controller.state.queue == {}
    public_objects = world.store.list("public/")
    assert any("/rounds/" in key for key in public_objects) and not any(b"ck-" in world.store.get(key) for key in public_objects), "no public object has a coldkey"


def test_the_registration_read_gets_every_coldkey_in_one_batched_read():
    chain = Subtensor({"owner": "ck-owner", "hk1": "ck-a", "hk2": "ck-a", "hk3": "ck-b"})
    snap = sdk_reader(chain).snapshot()
    assert [(reg.uid, reg.hotkey, reg.coldkey) for reg in snap.registrations] == [(0, "owner", "ck-owner"), (1, "hk1", "ck-a"), (2, "hk2", "ck-a"), (3, "hk3", "ck-b")]
    assert chain.batches == [("Owner", [["owner"], ["hk1"], ["hk2"], ["hk3"]], 900)], "one request for all hotkeys, at the block of the snapshot"
    fake = FakeChain({uid: f"hk{uid}" for uid in range(3)}, "hk0")
    fake.owners = {"hk1": "ck-a", "hk2": "ck-a"}
    assert [reg.coldkey for reg in Reader(fake, 120, clock=lambda: fake.now).snapshot().registrations] == ["coldkey-of-hk0", "ck-a", "ck-a"]


def no_answer(*args, **kwargs):
    raise TimeoutError("no answer")


@pytest.mark.parametrize(
    "damage",
    [
        lambda chain: setattr(chain, "coldkeys", no_answer),  # the read fails
        lambda chain: setattr(chain, "coldkeys", lambda block, hotkeys: ["ck"] * (len(hotkeys) - 1)),  # one coldkey is missing
        lambda chain: chain.owners.update({"hk2": ""}),  # a coldkey is empty
        lambda chain: chain.owners.update({"hk2": 7}),  # a coldkey is not a text
    ],
)
def test_a_coldkey_that_cannot_be_read_fails_the_registration_read(damage):
    chain = FakeChain({uid: f"hk{uid}" for uid in range(5)}, "hk0")
    reader = Reader(chain, 120, clock=lambda: chain.now)
    assert len(reader.snapshot().registrations) == 5
    damage(chain)
    with pytest.raises(InfraError):
        reader.snapshot()


def test_the_sdk_client_never_makes_a_coldkey_from_a_missing_owner():
    chain = Subtensor({"owner": "ck-owner", "hk1": None})  # the chain has no owner for hk1
    with pytest.raises(InfraError, match="bad registration for UID 1"):
        sdk_reader(chain).snapshot()


def chain_sync(build, queue, regs, **kw):
    return plan_sync(queue, kw.pop("served_regs", ()), kw.pop("served_hotkeys", ()), ChainSnapshot(120, 5000, 9000, tuple(regs)), netuid=120, burn_uid=0, **kw)


def test_sync_sets_the_coldkey_of_a_waiting_seat_and_changes_nothing_else_of_it(build):
    queue = build.queue(
        build.seat(1, registered_at=50), build.seat(2, coldkey="ck-old", front=True), build.seat(3, coldkey="ck-c", demoted=True),
        build.seat(4, leased=True), build.seat(5, coldkey="ck-e"), build.seat(6, held=True),
    )
    chain = [build.reg(uid, coldkey=f"ck-{name}", registered_at=99) for uid, name in ((1, "a"), (2, "b"), (3, "c"), (4, "d"))]
    plan = chain_sync(build, queue, [*chain, build.reg(5), build.reg(6, hotkey="swapped", coldkey="ck-f")])
    assert plan.coldkeyed == (build.reg(1, coldkey="ck-a", registered_at=50), build.reg(2, coldkey="ck-b")), "an empty coldkey is filled, a changed one follows the chain"
    assert not plan.added and not plan.removed and not plan.empty, "a leased seat is not touched, and a snapshot with no coldkey erases none"
    assert plan.rehotkeyed == (build.reg(6, hotkey="swapped", coldkey="ck-f"),), "a hotkey swap brings its coldkey with it"
    assert chain_sync(build, build.queue(build.seat(1, coldkey="ck-a")), [build.reg(1, coldkey="ck-a")]).empty


def old_state(tmp_path, monkeypatch, name: str, now: int, owners: dict[str, str]):
    """A state directory that commit 515d7e9f wrote, and a chain with the same 60 registrations and with coldkeys."""
    directory = tmp_path / name
    shutil.copytree(STATES / name, directory)
    chain = FakeChain({0: "owner", **{uid: f"hk{uid}" for uid in range(1, 61)}}, "owner", now=now)
    chain.born, chain.owners = {0: 1, **{uid: 100 + uid for uid in range(1, 61)}}, dict(owners)
    reader, cpu_stack = Reader(chain, 120, clock=lambda: chain.now), sim.stack
    monkeypatch.setattr(sim, "stack", lambda context: dataclasses.replace(cpu_stack(context), chain=reader.snapshot))
    return directory, ["--state-dir", str(directory), "--stack", "sn120_v2.sim:stack"], SimAdmission(LocalStore(directory / STORE_DIR))


def without_coldkeys(queue):
    return [dataclasses.replace(seat, reg=dataclasses.replace(seat.reg, coldkey="")) for seat in waiting(queue)]


def test_the_first_sync_on_a_state_of_an_older_commit_fills_the_coldkeys_and_moves_no_seat(tmp_path, monkeypatch):
    directory, stack, admission = old_state(tmp_path, monkeypatch, "idle", int(time.time()), {"hk4": "ck-a", "hk5": "ck-a", "hk7": "ck-a"})
    before = load(directory / JOURNAL_FILE)
    assert len(before.queue) == 57 and {seat.reg.coldkey for seat in before.queue.values()} == {""}, "the old commit saved no coldkey"
    for hotkey in ("hk4", "hk5", "hk6", "hk7"):
        admission.upload(hotkey)
    assert main(["run", *stack, *RUN]) == 0

    (synced,) = sync_records(directory / JOURNAL_FILE)[1:]  # the first such record is the one of the old commit
    data = synced.data
    assert len(data["coldkeyed"]) == 57 and data["added"] == data["removed"] == data["rehotkeyed"] == []
    after = replay(record for record in read_journal(directory / JOURNAL_FILE) if record.seq <= synced.seq)  # the state right after the first sync
    assert without_coldkeys(after.queue) == waiting(before.queue), "the same seats in the same order with the same flags and times"
    assert {seat.reg.hotkey: seat.reg.coldkey for seat in waiting(after.queue)[:4]} == {"hk4": "ck-a", "hk5": "ck-a", "hk6": "coldkey-of-hk6", "hk7": "ck-a"}
    assert (after.served_hotkeys, after.served_regs, after.kings, after.rounds, after.heads) == (before.served_hotkeys, before.served_regs, before.kings, before.rounds, before.heads)

    state = load(directory / JOURNAL_FILE)
    assert seats(state.rounds[-1]) == [4, 6], "the round that follows already applies the rule"
    assert [seat.reg.uid for seat in waiting(state.queue)[:2]] == [5, 7] and state.served_hotkeys == before.served_hotkeys | {"hk4", "hk6"}
    assert main(["run", *stack, *RUN]) == 0
    assert len(sync_records(directory / JOURNAL_FILE)) == 2, "a sync with nothing to change writes no record"
    assert seats(load(directory / JOURNAL_FILE).rounds[-1]) == [5], "the second seat of the coldkey sits in the next round, the third waits again"


def test_the_first_sync_leaves_an_open_round_and_its_seats_as_they_are(tmp_path, monkeypatch):
    opened_at = load(STATES / "open_round" / JOURNAL_FILE).round.opened_at
    monkeypatch.setattr(time, "time", lambda: opened_at + 5.0)  # the lease of the stored ticket still runs
    directory, stack, _ = old_state(tmp_path, monkeypatch, "open_round", opened_at + 5, {"hk4": "ck-a", "hk5": "ck-a", "hk6": "ck-a"})
    before = load(directory / JOURNAL_FILE)
    assert seats(before.round) == [4, 5], "the old commit put two seats of what is now one coldkey into the round"
    assert main(["run", *stack, *RUN]) == 0

    synced = sync_records(directory / JOURNAL_FILE)[-1]
    after = replay(record for record in read_journal(directory / JOURNAL_FILE) if record.seq <= synced.seq)
    assert after.round == before.round, "the open round is the same record"
    in_round = {key: seat for key, seat in after.queue.items() if seat.leased}
    assert in_round == {key: seat for key, seat in before.queue.items() if seat.leased} and set(in_round) == {(4, 104), (5, 105)}
    assert len(synced.data["coldkeyed"]) == len(before.queue) - 2 and without_coldkeys(after.queue) == waiting(before.queue)
    assert after.served_hotkeys == before.served_hotkeys

    state = load(directory / JOURNAL_FILE)
    assert state.round is None and (state.rounds[-1].round_id, state.rounds[-1].outcome) == ("r000002", "applied")
    assert {entry.uid: entry.status for entry in state.rounds[-1].decision.entries} == {4: "winner", 5: "lost"}, "the open round ran with both of its seats"


def test_a_failed_coldkey_read_costs_the_pass_and_changes_nothing(tmp_path, monkeypatch, capsys):
    directory, stack, admission = old_state(tmp_path, monkeypatch, "idle", int(time.time()), {})
    admission.upload("hk4")
    monkeypatch.setattr(FakeChain, "coldkeys", no_answer)
    before = (directory / JOURNAL_FILE).read_bytes()
    assert main(["run", *stack, *RUN]) == 0
    assert "waiting: chain read of the coldkeys failed: no answer" in capsys.readouterr().err
    state = load(directory / JOURNAL_FILE)
    assert len(sync_records(directory / JOURNAL_FILE)) == 1 and len(state.rounds) == 1 and state.round is None, "no sync and no round in a pass with no coldkeys"
    assert {seat.reg.coldkey for seat in state.queue.values()} == {""} and (directory / JOURNAL_FILE).read_bytes().startswith(before)
