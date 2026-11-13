"""The hourly weights writer, end to end on the in-memory chain and on a journal of the CPU simulation."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import pytest

from sn120_v2 import chain as chain_module
from sn120_v2.chain import FakeChain, Reader, Setter, SubtensorClient
from sn120_v2.params import Params
from sn120_v2.sim import SimWorld, sim_params
from sn120_v2.types import King
from sn120_v2.weights_writer import JOURNAL_FILE, STATE_FILE, load_kings, load_state, main, run_once, single_writer

VALIDATOR = "hk0"
SIM_VALIDATOR = "5SimValidator"  # the validator hotkey in the genesis of the simulation
HOUR_BLOCKS = 300  # 300 blocks of 12 s


def king(uid: int, order: int) -> King:
    return King(uid, f"hk{uid}", f"r{order:06d}", f"head{order}", 1000 + order)


def fail(*args):
    raise OSError("journal not readable")


class Rig:
    """One writer on a fake chain of 16 UIDs. The chain's clock is the writer's clock."""

    def __init__(self, state_dir) -> None:
        self.chain = FakeChain({uid: f"hk{uid}" for uid in range(16)}, VALIDATOR)
        self.kings = [king(1, 0), king(2, 1)]
        self.pinned = None  # the parameters pinned in the journal
        self.load_kings = lambda: (self.kings, self.chain.validator_hotkey, self.pinned)
        self.state_dir = state_dir

    def run(self, **options):
        validator = self.chain.validator_hotkey
        return run_once(
            load_kings=self.load_kings,
            reader=Reader(self.chain, 120, clock=lambda: self.chain.now),
            setter=Setter(self.chain, validator),
            params=Params(),
            state_dir=self.state_dir,
            now=self.chain.now,
            **{"wallet_hotkey": validator} | options,
        )

    def last_valid(self) -> dict[str, float]:
        return load_state(self.state_dir)["last_valid"]["weights"]


@pytest.fixture
def rig(tmp_path) -> Rig:
    return Rig(tmp_path)


@pytest.fixture
def journal_rig(tmp_path) -> Rig:
    """A writer that reads the journal of a simulated validator: UID 1 won one round, so it is the only king."""
    world = SimWorld(tmp_path, sim_params())
    world.register(1, 0.3)
    world.register(2, 0.0)
    assert world.round().outcome == "applied"
    rig = Rig(tmp_path)
    rig.chain = FakeChain({0: SIM_VALIDATOR, 1: "hk1", 2: "hk2"}, SIM_VALIDATOR, now=int(time.time()))
    rig.load_kings = partial(load_kings, tmp_path)
    return rig


def test_the_last_ten_kings_get_an_equal_share(rig):
    rig.kings = [king(uid, order) for order, uid in enumerate(range(1, 12))]
    assert rig.run().status == "set"
    assert rig.chain.weights == {uid: pytest.approx(0.1) for uid in range(2, 12)}  # the oldest king, UID 1, is out
    assert sum(rig.chain.weights.values()) == pytest.approx(1.0) and all(rig.chain.weights.values())


def test_a_king_whose_hotkey_left_its_uid_drops_and_the_rest_split(rig):
    rig.kings.append(king(3, 2))
    rig.chain.keys[2] = "a-new-miner"
    assert rig.run().status == "set" and rig.chain.weights == {1: 0.5, 3: 0.5}


def test_fewer_kings_than_the_window_split_all_the_weight_and_uid_0_gets_none(rig):
    rig.kings.append(king(0, 2))  # even a king on UID 0 is never paid
    assert rig.run().status == "set" and rig.chain.weights == {1: 0.5, 2: 0.5}
    rig.chain.advance(HOUR_BLOCKS)
    rig.chain.keys[1] = "a-new-miner"  # one of the two kings left: the other gets all of it, not UID 0
    assert rig.run().status == "set" and rig.chain.weights == {2: 1.0}


@pytest.mark.parametrize(
    "damage",
    [
        lambda rig: setattr(rig.chain, "down", "connection refused"),  # chain outage
        lambda rig: rig.chain.born.pop(3),  # partial chain read
        lambda rig: setattr(rig, "load_kings", fail),  # the state fails to load
        lambda rig: rig.kings.clear(),  # the state loads empty: the kings vanished
        lambda rig: rig.kings.pop(),  # the state loads a shorter kings list
        lambda rig: rig.kings.__setitem__(1, king(2, 0)),  # the state loads an older kings list
        lambda rig: rig.chain.keys.update({1: "new-miner-a", 2: "new-miner-b"}),  # no king is registered: no burn
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


@pytest.mark.parametrize(
    "damage",
    [
        lambda rig: setattr(rig.chain, "down", "x"),
        lambda rig: setattr(rig, "load_kings", fail),
        lambda rig: rig.kings.clear(),  # no king yet: nothing to pay, and no burn
        lambda rig: rig.chain.keys.update({1: "new-miner-a", 2: "new-miner-b"}),  # no king is registered: no burn
    ],
)
def test_a_first_run_with_an_outage_or_with_no_registered_king_submits_nothing(rig, damage):
    damage(rig)
    assert rig.run().status == "deferred"
    assert rig.chain.submissions == 0 and "last_valid" not in load_state(rig.state_dir)


def test_the_writer_reads_the_kings_and_the_validator_hotkey_from_the_journal(journal_rig):
    kings, validator_hotkey, pinned = load_kings(journal_rig.state_dir)
    assert [(king.uid, king.hotkey) for king in kings] == [(1, "hk1")] and validator_hotkey == SIM_VALIDATOR
    assert pinned == sim_params()
    assert journal_rig.run().status == "set" and journal_rig.chain.weights == {1: 1.0}


def test_the_payout_uses_the_parameters_pinned_in_the_journal_not_the_ones_of_the_command(rig):
    rig.pinned = replace(Params(), kings_window=1)
    assert rig.run().status == "set" and rig.chain.weights == {2: 1.0}  # a window of one: only the newest king
    rig.chain.advance(HOUR_BLOCKS)
    rig.pinned = replace(Params(), netuid=7)  # a journal of another subnet than the connection
    receipt = rig.run()
    assert receipt.status == "deferred" and "pins netuid 7" in receipt.reason and rig.chain.submissions == 1


@pytest.mark.parametrize(
    "damage",
    [
        lambda path: path.write_bytes(path.read_bytes() + b"{damaged\n"),  # JournalCorrupt
        lambda path: path.write_bytes(path.read_bytes().replace(b"hk1", b"hk7", 1)),  # JournalCorrupt: a changed line
        lambda path: path.write_bytes(b""),  # an empty journal has no genesis; it must not read as "no kings"
        lambda path: path.unlink(),
    ],
)
def test_a_journal_that_does_not_load_keeps_the_last_valid_weights(journal_rig, damage):
    assert journal_rig.run().status == "set"
    journal_rig.chain.advance(HOUR_BLOCKS)
    damage(journal_rig.state_dir / JOURNAL_FILE)
    receipt = journal_rig.run()
    assert receipt.status == "deferred" and "the state failed to load" in receipt.reason
    assert journal_rig.chain.submissions == 1 and journal_rig.chain.weights == {1: 1.0}


def test_a_wallet_that_is_not_the_validator_hotkey_of_the_genesis_writes_nothing(journal_rig):
    receipt = journal_rig.run(wallet_hotkey="hk2")
    assert receipt.status == "deferred" and "the genesis names 5SimValidator" in receipt.reason
    assert journal_rig.chain.submissions == 0 and "last_valid" not in load_state(journal_rig.state_dir)


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
    rig.chain.submit = fail
    assert rig.run().status == "deferred" and load_state(rig.state_dir)["last_receipt"]["status"] == "deferred"


def test_a_second_writer_defers_while_the_lock_is_held(rig):
    with single_writer(rig.state_dir) as alone:
        receipt = rig.run()
    assert alone and receipt.status == "deferred" and "lock" in receipt.reason and rig.chain.submissions == 0


def test_an_unreadable_state_file_defers(rig):
    (rig.state_dir / STATE_FILE).write_text("{not json")
    assert rig.run().status == "deferred" and rig.chain.submissions == 0


def test_the_dry_run_prints_the_table_and_writes_nothing(journal_rig, monkeypatch, capsys):
    chain, calls, state_dir = journal_rig.chain, [], journal_rig.state_dir
    monkeypatch.setattr(SubtensorClient, "connect", lambda *settings: calls.append(settings) or chain)
    files = sorted(path.name for path in state_dir.iterdir())
    argv = ["--state-dir", str(state_dir), "--network", "a-node", "--dry-run"]
    assert main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "dry_run": True, "status": "would set", "block": chain.block, "kings": 1, "weights": {"1": 1.0},
        "validator_hotkey": SIM_VALIDATOR, "validator_registered": True,
        "last_write_block": 0, "blocks_since_last_write": chain.block,
    }  # fmt: skip
    chain.down = "connection refused"
    assert main(argv) == 0 and json.loads(capsys.readouterr().out)["status"] == "would defer"
    assert calls == [(120, "a-node", "", "")] * 2  # no wallet goes to the connection
    assert chain.submissions == 0 and sorted(path.name for path in state_dir.iterdir()) == files


def test_the_dry_run_with_a_wallet_runs_the_checks_before_a_submission_and_writes_nothing(journal_rig, monkeypatch, capsys):
    chain, state_dir = journal_rig.chain, journal_rig.state_dir
    monkeypatch.setattr(SubtensorClient, "connect", lambda *settings: chain)
    files = sorted(path.name for path in state_dir.iterdir())
    argv = ["--state-dir", str(state_dir), "--network", "a-node", "--dry-run", "--wallet", "cold/hot"]

    def report() -> dict:
        assert main(argv) == 0
        return json.loads(capsys.readouterr().out)

    passed = report()
    assert passed["status"] == "would set" and passed["preflight"] == "passed" and passed["wallet_hotkey"] == SIM_VALIDATOR
    chain.updated[0] = chain.block - 10  # another writer set weights 10 blocks ago
    recent = report()
    assert recent["status"] == "would defer" and "recent write" in recent["reason"] and recent["blocks_since_last_write"] == 10
    chain.signer = lambda: "hk2"  # the hotkey file is not the validator hotkey of the genesis
    assert report()["status"] == "would defer"
    assert chain.submissions == 0 and sorted(path.name for path in state_dir.iterdir()) == files


def test_the_command_sets_the_weights_with_the_wallet(journal_rig, monkeypatch, capsys):
    chain, calls, state_dir = journal_rig.chain, [], journal_rig.state_dir
    monkeypatch.setattr(SubtensorClient, "connect", lambda *settings: calls.append(settings) or chain)
    argv = ["--state-dir", str(state_dir), "--network", "a-node", "--wallet", "cold/hot"]
    assert main(argv) == 0 and json.loads(capsys.readouterr().out)["status"] == "set"
    assert calls == [(120, "a-node", "cold/hot", "")] and chain.weights == {1: 1.0}
    assert main(argv) == 0 and "already had its attempt" in json.loads(capsys.readouterr().out)["reason"]
    chain.validator_hotkey = "hk2"  # the wallet now signs as another hotkey than the genesis names
    assert main(argv + ["--force"]) == 0 and "the genesis names" in json.loads(capsys.readouterr().out)["reason"]
    assert chain.submissions == 1


@pytest.mark.parametrize("argv", [[], ["--network", "finney", "--dry-run"], ["--state-dir", "x", "--network", "finney"]])
def test_the_command_refuses_a_missing_setting_before_any_connection(argv, monkeypatch):
    monkeypatch.setattr(SubtensorClient, "connect", fail)
    with pytest.raises(SystemExit):
        main(argv)


def test_a_missing_hotkey_file_is_one_clear_line_and_no_connection(tmp_path, monkeypatch, capsys):
    class NoHotkeyWallet:
        def __init__(self, name, path):
            self.hotkey_file = SimpleNamespace(path=f"{path}/{name.replace('/', '/hotkeys/')}")

        @property
        def hotkey(self):
            raise FileNotFoundError("keyfile does not exist")

    monkeypatch.setattr(chain_module, "bittensor", SimpleNamespace(Wallet=NoHotkeyWallet, Subtensor=fail))
    argv = ["--state-dir", str(tmp_path), "--network", "a-node", "--wallet", "cold/hot", "--wallet-path", "/wallets"]
    assert main(argv) == 2 and main(argv + ["--dry-run"]) == 2  # ``fail`` would raise if a connection opened
    error = capsys.readouterr().err
    assert "nothing submitted: the hotkey file /wallets/cold/hotkeys/hot does not exist" in error
    assert list(tmp_path.iterdir()) == []
