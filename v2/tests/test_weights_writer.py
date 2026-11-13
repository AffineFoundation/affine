"""The hourly weights writer, end to end on the in-memory chain."""

from __future__ import annotations

import json

import pytest

from sn120_v2.chain import FakeChain, Reader, Setter
from sn120_v2.errors import ConfigError, V2Error
from sn120_v2.params import Params
from sn120_v2.types import King
from sn120_v2.weights_writer import STATE_FILE, load_kings_file, load_state, main, run_once, single_writer

VALIDATOR = "hk0"
HOUR_BLOCKS = 300  # 300 blocks of 12 s


def king(uid: int, order: int) -> King:
    return King(uid, f"hk{uid}", f"r{order:06d}", f"head{order}", 1000 + order)


def fail() -> list[King]:
    raise OSError("journal not readable")


class Rig:
    """One writer on a fake chain of 16 UIDs. The chain's clock is the writer's clock."""

    def __init__(self, state_dir) -> None:
        self.chain = FakeChain({uid: f"hk{uid}" for uid in range(16)}, VALIDATOR)
        self.kings = [king(1, 0), king(2, 1)]
        self.load_kings = lambda: self.kings
        self.state_dir = state_dir

    def run(self, **options):
        return run_once(
            load_kings=self.load_kings,
            reader=Reader(self.chain, 120, clock=lambda: self.chain.now),
            setter=Setter(self.chain, VALIDATOR),
            params=Params(),
            state_dir=self.state_dir,
            now=self.chain.now,
            **options,
        )

    def last_valid(self) -> dict[str, float]:
        return load_state(self.state_dir)["last_valid"]["weights"]


@pytest.fixture
def rig(tmp_path) -> Rig:
    return Rig(tmp_path)


def test_the_last_ten_kings_get_an_equal_share(rig):
    rig.kings = [king(uid, order) for order, uid in enumerate(range(1, 12))]
    assert rig.run().status == "set"
    assert rig.chain.weights == {uid: pytest.approx(0.1) for uid in range(2, 12)}  # the oldest king, UID 1, is out
    assert sum(rig.chain.weights.values()) == pytest.approx(1.0) and all(rig.chain.weights.values())


def test_a_king_whose_hotkey_left_its_uid_drops_and_the_rest_split(rig):
    rig.kings.append(king(3, 2))
    rig.chain.keys[2] = "a-new-miner"
    assert rig.run().status == "set" and rig.chain.weights == {1: 0.5, 3: 0.5}


def test_no_paid_payee_burns_when_the_snapshot_is_complete_and_the_state_loaded(rig):
    rig.kings = []
    assert rig.run().status == "set" and rig.chain.weights == {0: 1.0}


@pytest.mark.parametrize(
    "damage",
    [
        lambda rig: setattr(rig.chain, "down", "connection refused"),  # chain outage
        lambda rig: rig.chain.born.pop(3),  # partial chain read
        lambda rig: setattr(rig, "load_kings", fail),  # the state fails to load
        lambda rig: rig.kings.clear(),  # the state loads empty: the kings vanished
        lambda rig: rig.kings.pop(),  # the state loads an older kings list
    ],
)
def test_an_outage_keeps_the_last_valid_weights_and_defers(rig, damage):
    assert rig.run().status == "set"
    rig.chain.advance(HOUR_BLOCKS)
    damage(rig)
    receipt = rig.run()
    assert receipt.status == "deferred" and "the last valid weights stay" in receipt.reason
    assert rig.chain.submissions == 1 and rig.chain.weights == {1: 0.5, 2: 0.5}  # nothing new went out: no burn
    assert rig.last_valid() == {"1": 0.5, "2": 0.5}


@pytest.mark.parametrize("damage", [lambda rig: setattr(rig.chain, "down", "x"), lambda rig: setattr(rig, "load_kings", fail)])
def test_a_first_run_in_an_outage_submits_nothing(rig, damage):
    damage(rig)
    assert rig.run().status == "deferred"
    assert rig.chain.submissions == 0 and "last_valid" not in load_state(rig.state_dir)


def test_one_attempt_per_hour_and_the_next_hour_tries_again(rig):
    rig.chain.refusals = ["stake too low"]
    assert rig.run().status == "deferred"
    rig.chain.refusals = []
    again = rig.run()  # the same hour: no second attempt, and no chain call at all
    assert again.status == "deferred" and "already had its attempt" in again.reason and rig.chain.submissions == 0
    assert rig.run(force=True).status == "set"  # an operator can force one
    rig.chain.advance(HOUR_BLOCKS)
    rig.kings.append(king(3, 2))
    assert rig.run().status == "set" and rig.chain.submissions == 2 and len(rig.chain.weights) == 3


def test_a_failed_submission_is_one_deferred_attempt(rig):
    rig.chain.submit = lambda table: fail()
    assert rig.run().status == "deferred" and load_state(rig.state_dir)["last_receipt"]["status"] == "deferred"


def test_a_second_writer_defers_while_the_lock_is_held(rig):
    with single_writer(rig.state_dir) as alone:
        receipt = rig.run()
    assert alone and receipt.status == "deferred" and "lock" in receipt.reason and rig.chain.submissions == 0


def test_an_unreadable_state_file_defers(rig):
    (rig.state_dir / STATE_FILE).write_text("{not json")
    assert rig.run().status == "deferred" and rig.chain.submissions == 0


def test_kings_file_round_trip_and_a_malformed_file_raises(tmp_path):
    path = tmp_path / "kings.json"
    path.write_text(json.dumps([king(1, 0).to_dict(), king(2, 1).to_dict()]))
    assert load_kings_file(path) == (king(1, 0), king(2, 1))
    path.write_text(json.dumps({"kings": []}))
    with pytest.raises(V2Error):
        load_kings_file(path)


def test_the_command_fails_closed_without_its_settings(monkeypatch):
    for name in ("PARAMS_FILE", "KINGS_FILE", "CHAIN_NETWORK", "WEIGHTS_STATE_DIR", "VALIDATOR_HOTKEY", "WALLET"):
        monkeypatch.delenv(f"SN120_V2_{name}", raising=False)
    with pytest.raises(ConfigError):
        main([])
