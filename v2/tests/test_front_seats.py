"""``front-seats``: the operator sends waiting seats to the front, for turns outside v2 that were never judged."""

from __future__ import annotations

import json

import pytest

from sn120_v2.cli import main
from sn120_v2.errors import JournalCorrupt, V2Error
from sn120_v2.journal import read_journal
from sn120_v2.queue import FRONT, squat_trim, waiting, zone
from sn120_v2.sim import JOURNAL_FILE, SimWorld, sim_params
from sn120_v2.state import SEATS_FRONTED, load, replay
from sn120_v2.types import King, V1Import

YEAR = 31536000  # squat_min_wait_s of the mainnet parameters
REASON = "v1 turns e1321, e1324, e1325 were never judged"


@pytest.fixture
def world(tmp_path) -> SimWorld:
    """The shape of the mainnet queue: two front seats of the import, three normal seats, three seats that v1 had moved back."""
    v1 = V1Import(
        "k1.json", 9500000, kings=(King(50, "hk50", "v1-e1", "", 5),), served_regs=(), served_hotkeys=("hk60",),
        demoted=((7, 1002), (8, 1006), (9, 1009)), front=((1, 1001), (2, 1010)),
    )
    made = SimWorld(tmp_path, sim_params(squat_min_wait_s=YEAR), v1=v1)
    for uid, block in ((1, 1001), (2, 1010), (3, 1003), (4, 1004), (5, 1005), (7, 1002), (8, 1006), (9, 1009)):
        made.register(uid, None, block=block)
    return made


def order(state) -> list[int]:
    return [seat.reg.uid for seat in waiting(state.queue)]


def files(world: SimWorld) -> dict[str, bytes]:
    return {str(path.relative_to(world.root)): path.read_bytes() for path in sorted(world.root.rglob("*")) if path.is_file()}


def command(world: SimWorld, *hotkeys: str, flag: str | None = None, reason: str = REASON) -> list[str]:
    names = [part for hotkey in hotkeys for part in ("--hotkey", hotkey)]
    return ["front-seats", "--state-dir", str(world.root), "--stack", "sn120_v2.sim:stack", *names, "--reason", reason, *([flag] if flag else [])]


def test_the_dry_run_shows_the_change_and_the_real_run_writes_exactly_it(world, capsys):
    before = world.controller.state
    assert order(before) == [1, 2, 3, 4, 5, 7, 8, 9], "front, normal, back; in each zone by registration block"
    untouched = files(world)

    for flag in ("--dryrun", "--dry-run"):
        assert main(command(world, "hk8", "hk7", "hk3", flag=flag)) == 0
        out, err = capsys.readouterr()
        shown = json.loads(out)
        assert (shown["dry_run"], shown["written"], shown["seats_to_change"], shown["charged"]) == (True, False, 3, 0)
        assert [(s["uid"], s["position"], s["front"], s["demoted"]) for s in shown["seats"]] == [
            (8, {"before": 6, "after": 3}, {"before": False, "after": True}, {"before": True, "after": False}),
            (7, {"before": 5, "after": 1}, {"before": False, "after": True}, {"before": True, "after": False}),
            (3, {"before": 2, "after": 2}, {"before": False, "after": True}, {"before": False, "after": False}),
        ]
        assert shown["front_seats_after"] == [1, 7, 3, 8, 2], "by registration block, not behind the front seats of the import"
        assert shown["other_seats_whose_position_shifts"] == 3, "UIDs 2, 4 and 5 move back; UID 1 and UID 9 keep their place"
        assert "nothing was written" in err and "AffineValidator" in err and SEATS_FRONTED in err and "older commit" in err
        assert files(world) == untouched, "a dry run writes nothing: not the journal, not the store"

    assert main(command(world, "hk8", "hk7", "hk3")) == 0
    out, err = capsys.readouterr()
    done = json.loads(out)
    assert (done["dry_run"], done["written"], done["replay_matches"]) == (False, True, True)
    assert {key: done[key] for key in shown if key not in ("dry_run", "written")} == {key: shown[key] for key in shown if key not in ("dry_run", "written")}
    records = read_journal(world.root / JOURNAL_FILE)
    assert [record.kind for record in records].count(SEATS_FRONTED) == 1 and records[-1].seq == done["journal_record"]
    assert records[-1].data == {"keys": [[8, 1006], [7, 1002], [3, 1003]], "reason": REASON}
    assert f"journal record {records[-1].seq}" in err

    after = world.reopen().state
    assert after.queue == load(world.root / JOURNAL_FILE).queue == replay(records).queue, "a replay of the journal gives the same state"
    assert order(after) == [1, 7, 3, 8, 2, 4, 5, 9]
    named = {(7, 1002), (8, 1006), (3, 1003)}
    assert all(after.queue[key].front and not after.queue[key].demoted for key in named)
    assert {key: seat for key, seat in after.queue.items() if key not in named} == {key: seat for key, seat in before.queue.items() if key not in named}
    assert all(seat.infra_attempts == 0 and not seat.held and not seat.leased for seat in after.queue.values()), "no failure is counted"
    assert (after.served_regs, after.served_hotkeys, after.turn_times, after.kings) == (before.served_regs, before.served_hotkeys, before.turn_times, before.kings)

    written = files(world)
    assert main(command(world, "hk8", "hk7", "hk3")) == 0, "a second run changes nothing and says so"
    out, err = capsys.readouterr()
    assert (json.loads(out)["written"], json.loads(out)["seats_to_change"]) == (False, 0) and "nothing to change" in err
    assert files(world) == written, "not even a takeover of the journal"


def test_one_wrong_hotkey_or_an_open_round_refuses_the_whole_command(world, capsys):
    untouched = files(world)
    assert main(command(world, "hk7", "hk99", "hk60")) == 2
    err = capsys.readouterr().err
    assert "refused: V2Error: nothing was changed" in err and "hk99: no seat in the queue has this hotkey" in err
    assert "hk60: it had its turn and was charged, so it has no seat" in err
    assert main(command(world, "hk7", reason="  ")) == 2 and "--reason is empty" in capsys.readouterr().err
    assert files(world) == untouched and not world.reopen().state.queue[(7, 1002)].front

    world.register(20, 0.0, block=1020)
    assert world.controller.open_round(world.clock(), "entropy"), "a round is open"
    with pytest.raises(V2Error, match="only between rounds"):
        world.controller.front_seats([(7, 1002)], REASON, world.clock())
    records = len(read_journal(world.root / JOURNAL_FILE))
    for hotkeys in (("hk7",), ("hk20",)):
        assert main(command(world, *hotkeys)) == 2
        assert "nothing was changed: round r000001 is open (phase RUNNING)" in capsys.readouterr().err
    assert len(read_journal(world.root / JOURNAL_FILE)) == records, "a refused command does not take over the journal"


def test_a_seat_at_the_front_stays_there_and_a_checkout_without_the_record_kind_refuses_the_journal(world, capsys, monkeypatch):
    assert main(command(world, "hk7", "hk3")) == 0
    capsys.readouterr()
    world.reopen()
    del world.chain[7]
    world.sync()
    assert (7, 1002) not in world.controller.state.queue
    world.register(7, None, block=1002)  # the same registration comes back, as after a chain read that missed it
    seat = world.controller.state.queue[(7, 1002)]
    assert seat.front and not seat.demoted, "as a front seat of the import: the flag belongs to the registration"

    records = read_journal(world.root / JOURNAL_FILE)
    monkeypatch.delitem(__import__("sn120_v2.state", fromlist=["HANDLERS"]).HANDLERS, SEATS_FRONTED)
    with pytest.raises(JournalCorrupt, match="unknown journal record kind 'seats_fronted'"):
        replay(records)


def test_squat_trim_cannot_touch_a_seat_at_the_front_at_the_mainnet_wait(world, capsys):
    """The trim looks only at seats of the normal zone. A front seat is not in it, however old it is and with no file."""
    params = world.controller.params
    assert (params.squat_min_wait_s, params.squat_age_factor) == (YEAR, 2.0)
    times = [seat.reg.registered_at for seat in world.controller.state.queue.values()]
    late = max(times) + 3 * YEAR  # older than 2 x 365 days: every normal seat with no file is due

    def trimmed(state) -> list[int]:
        return sorted(uid for uid, _ in squat_trim(state.queue, {}, late, turns_last_day=5, min_wait_s=params.squat_min_wait_s, age_factor=params.squat_age_factor))

    assert trimmed(world.controller.state) == [3, 4, 5], "before the command UID 3 is a normal seat, and the trim reaches it"
    assert squat_trim(world.controller.state.queue, {}, min(times) + 2 * YEAR, 5, params.squat_min_wait_s, params.squat_age_factor) == [], "not before 730 days"
    assert squat_trim(world.controller.state.queue, {}, late, 0, params.squat_min_wait_s, params.squat_age_factor) == [], "and never with no charged entry in the last day"

    assert main(command(world, "hk8", "hk7", "hk3")) == 0
    capsys.readouterr()
    state = world.reopen().state
    assert all(zone(state.queue[key]) == FRONT for key in ((7, 1002), (8, 1006), (3, 1003)))
    assert trimmed(state) == [4, 5], "no seat of the command, at any age"
