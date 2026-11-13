"""Chain access: the reader of registrations and the one setter of weights.

``Reader`` and ``Setter`` implement the ``ChainReader`` and ``WeightSetter``
seams on top of a narrow ``ChainClient``. ``FakeChain`` is the in-memory
client for tests and the CPU simulation. ``SubtensorClient`` is the client
over the bittensor SDK: it is the only code here that needs the SDK or a
wallet, and the wallet's private key never leaves the SDK.

``python -m sn120_v2.chain --netuid 120 --network finney --out FILE`` writes
one snapshot to a file. It only reads the chain and it uses no wallet.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .errors import ConfigError, InfraError, V2Error
from .payout import check_weights
from .types import ChainSnapshot, Registration, WeightReceipt

try:
    import bittensor
except ImportError:  # the SDK is the optional "chain" extra: the control plane and the tests run without it
    bittensor = None

BLOCK_SECONDS = 12  # Bittensor block time, used to estimate the time of an old block


class ChainClient(Protocol):
    """The few chain calls v2 needs. Every read names its block, so one snapshot is one block."""

    def head(self) -> tuple[int, str, int]: ...  # the newest block: number, hash, Unix time
    def block_hash(self, block: int) -> str: ...
    def uid_count(self, block: int) -> int: ...
    def hotkeys(self, block: int) -> Mapping[int, str]: ...  # UID -> hotkey
    def registration_blocks(self, block: int) -> Mapping[int, int]: ...  # UID -> block of its registration
    def last_update(self, block: int) -> Sequence[int]: ...  # by UID: block of its last weight write
    def rate_limit(self, block: int) -> int: ...  # blocks a validator must wait between two weight writes
    def plan(self, weights: Mapping[int, float]) -> tuple[str, list[str]]: ...  # dry run: signer address, refusals
    def submit(self, weights: Mapping[int, float]) -> tuple[bool, str]: ...  # one signed submission: accepted, message


def _read(what: str, call: Callable[[], Any]) -> Any:
    """Run one chain read. Any failure is ours, so it becomes ``InfraError``."""
    try:
        return call()
    except Exception as exc:
        raise InfraError(f"chain read of {what} failed: {exc}") from exc


class Reader:
    """``ChainReader``: every registration of the subnet, read at one block."""

    def __init__(self, client: ChainClient, netuid: int, *, max_lag_s: int = 600, clock: Callable[[], float] = time.time) -> None:
        self.client, self.netuid, self.max_lag_s, self.clock = client, netuid, max_lag_s, clock

    def snapshot(self) -> ChainSnapshot:
        block, _, taken_at = _read("the head", self.client.head)
        lag = int(self.clock()) - taken_at
        if lag > self.max_lag_s:
            raise InfraError(f"the chain head is {lag} s old: the node is behind")
        count = _read("the UID count", lambda: self.client.uid_count(block))
        hotkeys = dict(_read("the hotkeys", lambda: self.client.hotkeys(block)))
        born = dict(_read("the registration blocks", lambda: self.client.registration_blocks(block)))
        uids = list(range(count))
        if count < 1 or sorted(hotkeys) != uids or sorted(born) != uids:
            raise InfraError(
                f"partial chain read at block {block}: {len(hotkeys)} hotkeys and {len(born)} registration blocks for {count} UIDs"
            )
        registrations = []
        for uid in uids:
            if not (isinstance(hotkeys[uid], str) and hotkeys[uid]) or not 0 <= born[uid] <= block:
                raise InfraError(f"partial chain read at block {block}: bad registration for UID {uid}")
            registered_at = max(1, taken_at - BLOCK_SECONDS * (block - born[uid]))
            registrations.append(Registration(uid, hotkeys[uid], born[uid], registered_at))
        return ChainSnapshot(self.netuid, block, taken_at, tuple(registrations))

    def block_hash(self, block: int) -> str:
        return _read(f"the hash of block {block}", lambda: self.client.block_hash(block))


class Setter:
    """``WeightSetter``: plan first, then at most one submission. Every refusal is a ``deferred`` receipt."""

    def __init__(self, client: ChainClient, validator_hotkey: str, min_gap_blocks: int = 0) -> None:
        self.client, self.validator_hotkey, self.min_gap_blocks = client, validator_hotkey, min_gap_blocks

    def set_weights(self, weights: Mapping[int, float]) -> WeightReceipt:
        table = check_weights(weights)  # an empty or all-zero table raises here and is never submitted
        block = 0
        try:
            block = self.client.head()[0]
            refusal = self._plan(table, block)
        except Exception as exc:
            refusal = f"plan failed: {exc}"
        if refusal:
            return WeightReceipt("deferred", block, refusal)
        try:
            accepted, message = self.client.submit(table)
        except Exception as exc:
            return WeightReceipt("deferred", block, f"submission failed: {exc}")
        return WeightReceipt("set" if accepted else "deferred", block, message)

    def _plan(self, table: Mapping[int, float], block: int) -> str:
        """Read-only checks. Return the reason to defer, or "" when a submission may go out."""
        hotkeys = self.client.hotkeys(block)
        mine = [uid for uid, hotkey in hotkeys.items() if hotkey == self.validator_hotkey]
        if not mine:
            return "the validator hotkey is not registered on the subnet"
        missing = sorted(set(table) - set(hotkeys))
        if missing:
            return f"UIDs {missing} are not on chain"
        gap = max(self.client.rate_limit(block), self.min_gap_blocks)
        wait = self.client.last_update(block)[mine[0]] + gap - block
        if wait > 0:
            return f"recent write: the rate limit allows the next one in {wait} blocks"
        signer, refusals = self.client.plan(table)
        if signer != self.validator_hotkey:
            return "the wallet does not sign as the validator hotkey"
        return "plan refused: " + "; ".join(refusals) if refusals else ""


class FakeChain:
    """An in-memory ``ChainClient`` for tests and the CPU simulation. It keeps the rate limit as the real chain does."""

    def __init__(
        self, hotkeys: Mapping[int, str], validator_hotkey: str, *, block: int = 5000, limit: int = 100, now: int = 1_800_000_000
    ) -> None:
        self.keys = dict(hotkeys)
        self.born = {uid: block - 1000 + uid for uid in self.keys}
        self.updated = {uid: 0 for uid in self.keys}
        self.validator_hotkey, self.block, self.limit, self.now = validator_hotkey, block, limit, now
        self.weights: dict[int, float] = {}  # the table on chain; empty until the first accepted write
        self.submissions = 0
        self.refusals: list[str] = []  # what ``plan`` reports
        self.down = ""  # when set, every call fails with this message

    def _up(self) -> None:
        if self.down:
            raise ConnectionError(self.down)

    def advance(self, blocks: int) -> None:
        self.block += blocks
        self.now += blocks * BLOCK_SECONDS

    def head(self) -> tuple[int, str, int]:
        self._up()
        return self.block, self.block_hash(self.block), self.now

    def block_hash(self, block: int) -> str:
        self._up()
        return f"0x{block:064x}"

    def uid_count(self, block: int) -> int:
        self._up()
        return len(self.keys)

    def hotkeys(self, block: int) -> dict[int, str]:
        self._up()
        return dict(self.keys)

    def registration_blocks(self, block: int) -> dict[int, int]:
        self._up()
        return dict(self.born)

    def last_update(self, block: int) -> list[int]:
        self._up()
        return [self.updated[uid] for uid in sorted(self.updated)]

    def rate_limit(self, block: int) -> int:
        self._up()
        return self.limit

    def plan(self, weights: Mapping[int, float]) -> tuple[str, list[str]]:
        self._up()
        return self.validator_hotkey, list(self.refusals)

    def submit(self, weights: Mapping[int, float]) -> tuple[bool, str]:
        self._up()
        self.submissions += 1
        uid = next(uid for uid, hotkey in self.keys.items() if hotkey == self.validator_hotkey)
        if self.block - self.updated[uid] < self.limit:
            return False, "the chain refused: rate limit"
        self.weights, self.updated[uid] = dict(weights), self.block
        return True, f"accepted at block {self.block}"


class SubtensorClient:
    """``ChainClient`` over the bittensor SDK. Without a wallet it can only read."""

    def __init__(self, netuid: int, subtensor: Any, wallet: Any = None, sdk: Any = None) -> None:
        self.netuid, self.chain, self.wallet, self.sdk = netuid, subtensor, wallet, sdk or bittensor
        if self.sdk is None:
            raise ConfigError("the bittensor SDK is not installed: install the 'chain' extra")
        self.items = self.sdk.storage.SubtensorModule

    @classmethod
    def connect(cls, netuid: int, network: str, wallet: str = "") -> SubtensorClient:
        """Open a connection. ``wallet`` is a ``name/hotkey`` wallet on this machine, or "" to read only."""
        if bittensor is None:
            raise ConfigError("the bittensor SDK is not installed: install the 'chain' extra")
        return cls(netuid, bittensor.Subtensor(network), bittensor.Wallet(wallet) if wallet else None)

    def _query(self, item: str, block: int | None = None) -> Any:
        return self.chain.query(getattr(self.items, item), [self.netuid], block=block)

    def _map(self, item: str, block: int) -> dict[int, Any]:
        return {int(uid): value for uid, value in self.chain.query_map(getattr(self.items, item), [self.netuid], block=block)}

    def head(self) -> tuple[int, str, int]:
        info = self.chain.block_info(self.chain.block)
        return int(info.number), str(info.hash), int(info.timestamp.timestamp())

    def block_hash(self, block: int) -> str:
        return str(self.chain.block_info(block).hash)

    def uid_count(self, block: int) -> int:
        return int(self._query("SubnetworkN", block))

    def hotkeys(self, block: int) -> dict[int, str]:
        return {uid: str(hotkey) for uid, hotkey in self._map("Keys", block).items()}

    def registration_blocks(self, block: int) -> dict[int, int]:
        return {uid: int(born) for uid, born in self._map("BlockAtRegistration", block).items()}

    def last_update(self, block: int) -> list[int]:
        return [int(value) for value in self._query("LastUpdate", block)]

    def rate_limit(self, block: int) -> int:
        return int(self._query("WeightsSetRateLimit", block))

    def _intent(self, weights: Mapping[int, float]) -> Any:
        if self.wallet is None:
            raise ConfigError("this chain client has no wallet: it can only read")
        uids = sorted(weights)
        version_key = int(self._query("WeightsVersionKey") or 0)
        return self.sdk.SetWeights(netuid=self.netuid, uids=uids, weights=[weights[uid] for uid in uids], version_key=version_key)

    def plan(self, weights: Mapping[int, float]) -> tuple[str, list[str]]:
        plan = self.chain.plan(self._intent(weights), self.wallet)
        return str(plan.signer_address), [str(violation) for violation in plan.violations]

    def submit(self, weights: Mapping[int, float]) -> tuple[bool, str]:
        # retries=0: one submission. The SDK sends a timelocked commit when the subnet has commit-reveal on;
        # the chain reveals it later, so an accepted commit is not yet a change of the weights.
        result = self.chain.execute(self._intent(weights), self.wallet, wait_for_inclusion=True, wait_for_finalization=True, retries=0)
        return bool(result.success), str(result.message or result.block_hash or "")


def main(argv: Sequence[str] | None = None) -> int:
    """Read one snapshot, then write its record as JSON. A partial or failed read writes nothing."""
    parser = argparse.ArgumentParser(prog="python -m sn120_v2.chain", description="Write one chain snapshot to a file.")
    parser.add_argument("--netuid", type=int, required=True, help="the subnet to read: 120 for SN120")
    parser.add_argument("--network", required=True, help="the chain to read: finney, or the address of a node")
    parser.add_argument("--out", type=Path, required=True, help="the file for the ChainSnapshot record, as JSON")
    args = parser.parse_args(argv)
    try:
        client = _read("the connection", lambda: SubtensorClient.connect(args.netuid, args.network))  # no wallet
        snapshot = Reader(client, args.netuid).snapshot()
    except V2Error as exc:
        print(f"no snapshot written: {exc}", file=sys.stderr)
        return 1
    scratch = args.out.with_name(args.out.name + ".tmp")
    scratch.write_bytes(snapshot.to_json())
    os.replace(scratch, args.out)
    print(f"wrote {args.out}: netuid {snapshot.netuid}, block {snapshot.block}, {len(snapshot.registrations)} registrations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
