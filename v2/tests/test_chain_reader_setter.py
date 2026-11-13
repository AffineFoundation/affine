"""The chain reader and the weight setter, on the in-memory chain and on a stub of the SDK."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from sn120_v2.chain import BLOCK_SECONDS, FakeChain, Reader, Setter, SubtensorClient, main
from sn120_v2.errors import ConfigError, ContractError, InfraError
from sn120_v2.seams import ChainReader, WeightSetter
from sn120_v2.types import ChainSnapshot

VALIDATOR = "hk0"
TABLE = {1: 0.5, 4: 0.5}


def fake(**settings) -> FakeChain:
    return FakeChain({uid: f"hk{uid}" for uid in range(5)}, VALIDATOR, **settings)


def setter(chain: FakeChain, **settings) -> Setter:
    return Setter(chain, VALIDATOR, **settings)


def fail(*args, **kwargs):
    raise TimeoutError("no answer")


def test_snapshot_holds_every_registration_at_one_block():
    chain = fake()
    reader = Reader(chain, 120, clock=lambda: chain.now)
    snap = reader.snapshot()
    assert isinstance(reader, ChainReader)
    assert (snap.netuid, snap.block, snap.taken_at) == (120, 5000, chain.now)
    assert snap.hotkeys() == {uid: f"hk{uid}" for uid in range(5)}  # UID 0 is in the snapshot
    seat = snap.registrations[3]
    assert (seat.uid, seat.registration_block) == (3, 4003)
    assert seat.registered_at == chain.now - BLOCK_SECONDS * (5000 - 4003)
    assert reader.block_hash(7) == chain.block_hash(7)


@pytest.mark.parametrize(
    "damage",
    [
        lambda chain: chain.keys.pop(2),  # a hotkey is missing
        lambda chain: chain.born.pop(2),  # a registration block is missing
        lambda chain: setattr(chain, "uid_count", lambda block: 9),  # the UID count disagrees
        lambda chain: chain.keys.update({2: ""}),  # a hotkey is empty
        lambda chain: chain.born.update({2: chain.block + 1}),  # a registration after the head
        lambda chain: setattr(chain, "down", "connection refused"),  # the node is down
        lambda chain: setattr(chain, "registration_blocks", fail),  # one read times out
        lambda chain: setattr(chain, "now", chain.now - 3600),  # the node is an hour behind
    ],
)
def test_a_partial_failed_or_stale_read_raises_infra_error(damage):
    chain = fake()
    reader = Reader(chain, 120, clock=lambda now=chain.now: now)
    damage(chain)
    with pytest.raises(InfraError):
        reader.snapshot()


def test_setter_plans_then_submits_once():
    chain = fake()
    receipt = setter(chain).set_weights(TABLE)
    assert isinstance(setter(chain), WeightSetter)
    assert (receipt.status, receipt.block) == ("set", 5000)
    assert chain.weights == TABLE and chain.submissions == 1


def test_a_recent_write_is_deferred_without_a_submission():
    chain = fake()
    setter(chain).set_weights(TABLE)
    chain.advance(99)
    receipt = setter(chain).set_weights({2: 1.0})
    assert receipt.status == "deferred" and "rate limit" in receipt.reason
    assert chain.submissions == 1 and chain.weights == TABLE
    chain.advance(1)
    assert setter(chain, min_gap_blocks=500).set_weights({2: 1.0}).status == "deferred"  # our own floor also holds
    assert setter(chain).set_weights({2: 1.0}).status == "set" and chain.submissions == 2


@pytest.mark.parametrize(
    "damage, reason",
    [
        (lambda chain: chain.refusals.append("stake too low"), "plan refused: stake too low"),
        (lambda chain: chain.keys.update({0: "another-key"}), "validator hotkey is not registered"),
        (lambda chain: (chain.keys.pop(4), chain.updated.pop(4)), "UIDs [4] are not on chain"),
        (lambda chain: setattr(chain, "plan", lambda table: ("another-key", [])), "does not sign as the validator"),
        (lambda chain: setattr(chain, "rate_limit", fail), "plan failed"),
        (lambda chain: setattr(chain, "down", "connection refused"), "plan failed"),
    ],
)
def test_a_refused_plan_is_deferred_without_a_submission(damage, reason):
    chain = fake()
    damage(chain)
    receipt = setter(chain).set_weights(TABLE)
    assert receipt.status == "deferred" and reason in receipt.reason
    assert chain.submissions == 0 and chain.weights == {}


def test_a_refusal_or_a_failure_at_submission_is_deferred_after_one_try():
    chain = fake(limit=1000)
    setter(chain).set_weights(TABLE)
    chain.rate_limit = lambda block: 0  # the plan sees no limit, so the chain refuses at submission: a race
    receipt = setter(chain).set_weights({2: 1.0})
    assert receipt.status == "deferred" and "rate limit" in receipt.reason
    assert chain.submissions == 2 and chain.weights == TABLE  # one try, and no second one
    chain.submit = fail
    receipt = setter(chain).set_weights({2: 1.0})
    assert receipt.status == "deferred" and "submission failed" in receipt.reason


@pytest.mark.parametrize("table", [{}, {1: 0.0, 2: 0.0}, {1: 0.4}, {1: float("nan")}])
def test_an_empty_or_all_zero_table_is_never_submitted(table):
    chain = fake()
    with pytest.raises(ContractError):
        setter(chain).set_weights(table)
    assert chain.submissions == 0


class StubSubtensor:
    """The few SDK calls the client makes, with the names and shapes read from bittensor 11.0.2."""

    block = 900

    def __init__(self) -> None:
        self.executed: list[tuple] = []

    def block_info(self, block):
        return SimpleNamespace(number=block, hash=f"0xhash{block}", timestamp=datetime.fromtimestamp(1_800_000_000, timezone.utc))

    def query(self, item, params, *, block=None):
        return {"SubnetworkN": 2, "LastUpdate": [10, 20], "WeightsSetRateLimit": 100, "WeightsVersionKey": 7}[item]

    def query_map(self, item, params, *, block=None):
        return {"Keys": [(0, "hk0"), (1, "hk1")], "BlockAtRegistration": [(0, 5), (1, 800)]}[item]

    def plan(self, intent, wallet):
        return SimpleNamespace(signer_address="hk0", violations=["too soon"])

    def execute(self, intent, wallet, **options):
        self.executed.append((intent, wallet, options))
        return SimpleNamespace(success=True, message="", block_hash="0xabc")


ITEMS = ("SubnetworkN", "LastUpdate", "WeightsSetRateLimit", "WeightsVersionKey", "Keys", "BlockAtRegistration")
STUB_SDK = SimpleNamespace(
    storage=SimpleNamespace(SubtensorModule=SimpleNamespace(**{name: name for name in ITEMS})),
    SetWeights=lambda **fields: fields,
)


def test_sdk_client_reads_at_one_block_and_submits_exactly_once():
    chain = StubSubtensor()
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hk0"))
    client = SubtensorClient(120, chain, wallet=wallet, sdk=STUB_SDK)
    assert client.signer() == "hk0"
    snap = Reader(client, 120, clock=lambda: 1_800_000_000).snapshot()
    assert snap.block == 900 and snap.hotkeys() == {0: "hk0", 1: "hk1"}
    assert snap.registrations[1].registration_block == 800 and client.block_hash(3) == "0xhash3"
    assert client.last_update(900) == [10, 20] and client.rate_limit(900) == 100
    assert client.plan({1: 1.0}) == ("hk0", ["too soon"])
    assert client.submit({1: 0.25, 0: 0.75}) == (True, "0xabc")
    [(intent, used, options)] = chain.executed
    assert intent == dict(netuid=120, uids=[0, 1], weights=[0.75, 0.25], version_key=7) and used is wallet
    assert options["retries"] == 0
    for call in (lambda reader: reader.submit({1: 1.0}), lambda reader: reader.signer()):
        with pytest.raises(ConfigError):
            call(SubtensorClient(120, chain, sdk=STUB_SDK))  # no wallet: this client can only read


def test_the_snapshot_command_needs_no_wallet_and_writes_only_after_a_complete_read(tmp_path, monkeypatch):
    chain, out, calls = fake(now=int(time.time())), tmp_path / "chain_snapshot.json", []
    monkeypatch.setattr(SubtensorClient, "connect", lambda *settings: calls.append(settings) or chain)
    argv = ["--netuid", "120", "--network", "a-node", "--out", str(out)]
    chain.born.pop(2)  # a partial read
    assert main(argv) == 1 and list(tmp_path.iterdir()) == []
    chain.born[2] = 4002
    assert main(argv) == 0 and calls == [(120, "a-node")] * 2  # no wallet goes to the connection
    written = out.read_bytes()
    assert ChainSnapshot.from_json(written) == Reader(chain, 120).snapshot()
    chain.down = "connection refused"
    assert main(argv) == 1 and out.read_bytes() == written and list(tmp_path.iterdir()) == [out]
