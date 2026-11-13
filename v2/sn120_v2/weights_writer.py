"""The single hourly writer of chain weights.

One run is one attempt. It ends in ``set`` or ``deferred`` and it never loops:
the next attempt is the next hourly run.

The table is the payout formula over the kings and one complete chain
snapshot: the registered kings of the window split all the weight. There is
no burn, in any case: no table with weight on ``burn_uid`` (UID 0, the owner)
is ever submitted. When no king of the window is registered, when the chain
read fails or is partial, or when the state does not load, the run submits
nothing, so the last valid weights stay on chain.

Run it from one hourly timer, on the machine that holds the validator hotkey
(the unit files are in ``v2/systemd``):

    python -m sn120_v2.weights_writer --state-dir DIR --network finney --wallet NAME/HOTKEY

The write is signed by the hotkey. The run reads the hotkey file and no other
key file: the coldkey is never needed on this machine.

With ``--dry-run`` it computes and prints the table. Then it needs no wallet,
and it writes nothing to the chain or to a file. With ``--dry-run --wallet``
it also checks the hotkey file and runs every check that comes before a
submission, still with no chain write.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from . import payout
from .chain import Reader, Setter, SubtensorClient
from .errors import ConfigError, ContractError, InfraError, V2Error
from .params import Params, load_params
from .seams import ChainReader, WeightSetter
from .state import load as replay_journal
from .types import ChainSnapshot, King, WeightReceipt

JOURNAL_FILE = "journal.jsonl"
LOCK_FILE = "weights.lock"
STATE_FILE = "weights_state.json"

KingsSource = Callable[[], tuple[Sequence[King], str, Params | None]]


def load_kings(state_dir: Path) -> tuple[list[King], str, Params | None]:
    """The kings, oldest first, the one hotkey that may set weights, and the parameters pinned in the journal.

    This is the only place where the writer reads the controller's state:
    the journal in ``state_dir``. A journal that cannot be read raises.
    """
    view = replay_journal(Path(state_dir) / JOURNAL_FILE)
    if view.genesis is None:
        raise ContractError("the journal holds no genesis: the state is not bootstrapped")
    return list(view.kings), view.genesis.validator_hotkey, view.params


@contextmanager
def single_writer(state_dir: Path) -> Iterator[bool]:
    """Hold the writer lock for one run. Yields False when another writer holds it."""
    state_dir.mkdir(parents=True, exist_ok=True)
    with open(state_dir / LOCK_FILE, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


def load_state(state_dir: Path) -> dict[str, Any]:
    """What the writer remembers: its last run and the last valid table. No file means a first run."""
    path = state_dir / STATE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_state(state_dir: Path, state: dict[str, Any]) -> None:
    scratch = state_dir / (STATE_FILE + ".tmp")
    scratch.write_text(json.dumps(state, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(scratch, state_dir / STATE_FILE)


class Keep(Exception):
    """A reason to submit nothing in this run, so that the last valid weights stay."""

    def __init__(self, block: int, reason: str) -> None:
        super().__init__(reason)
        self.block, self.reason = block, reason


def build_table(
    kings_source: KingsSource, reader: ChainReader, params: Params, last_valid: dict[str, Any] | None = None
) -> tuple[dict[int, float], ChainSnapshot, int, int, str]:
    """The payout table from one complete chain snapshot and a state that loaded. Anything less raises ``Keep``.

    So does a table with no registered king: it would put weight on ``burn_uid``, and the writer never burns.

    The payout uses the parameters pinned in the journal when it has them, so that
    the writer and the controller cannot disagree. ``params`` is the fallback.

    Returns the table, the snapshot, the number of kings, the time of the newest king, and the validator hotkey.
    """
    try:
        snapshot = reader.snapshot()
    except InfraError as exc:
        raise Keep(0, f"no complete chain snapshot: {exc}") from exc
    if snapshot.netuid != params.netuid:
        raise Keep(snapshot.block, f"the chain snapshot is for netuid {snapshot.netuid}, not {params.netuid}")
    try:
        kings, validator_hotkey, pinned = kings_source()
        kings, params = tuple(kings), pinned or params
        newest = max((king.applied_at for king in kings), default=0)
    except Exception as exc:  # a corrupt journal, a missing file, a state with no genesis
        raise Keep(snapshot.block, f"the state failed to load: {exc}") from exc
    if snapshot.netuid != params.netuid:
        raise Keep(snapshot.block, f"the journal pins netuid {params.netuid}, but the chain snapshot is for netuid {snapshot.netuid}")
    # The kings list only grows. A shorter or older list means a bad state read.
    last = last_valid or {}
    if len(kings) < last.get("kings", 0) or newest < last.get("newest_king_at", 0):
        raise Keep(snapshot.block, "the kings list went backwards, so the state did not load correctly")
    try:
        table = payout.weights(kings, snapshot.hotkeys(), params)
    except V2Error as exc:
        raise Keep(snapshot.block, f"the payout refused the table: {exc}") from exc
    if not table or params.burn_uid in table:
        raise Keep(snapshot.block, "no king of the window is registered on chain, and the writer never burns")
    return table, snapshot, len(kings), newest, validator_hotkey


def run_once(
    *,
    load_kings: KingsSource,
    reader: ChainReader,
    setter: WeightSetter,
    wallet_hotkey: str,
    params: Params,
    state_dir: Path,
    now: int,
    force: bool = False,
) -> WeightReceipt:
    """One attempt to set the weights.

    ``load_kings`` returns the applied kings, oldest first, and the validator
    hotkey of the genesis. ``wallet_hotkey`` is the address the wallet signs
    as: when the two differ, nothing is submitted.

    Each clock hour (``params.weights_cadence_s``) gets one attempt, whatever
    its result. ``force`` lets an operator run a second one.
    """
    state_dir = Path(state_dir)
    with single_writer(state_dir) as alone:
        if not alone:
            return WeightReceipt("deferred", 0, "another weights writer holds the lock")
        try:
            state = load_state(state_dir)
        except (OSError, ValueError) as exc:
            return WeightReceipt("deferred", 0, f"the writer state file is unreadable: {exc}")
        if not force and state.get("last_run_at", -1) // params.weights_cadence_s == now // params.weights_cadence_s:
            return WeightReceipt("deferred", 0, "this hour already had its attempt")
        state["last_run_at"] = now
        save_state(state_dir, state)  # saved before any chain call: a crash cannot become a retry loop
        receipt = _attempt(load_kings, reader, setter, wallet_hotkey, params, state, now)
        state["last_receipt"] = receipt.to_dict()
        save_state(state_dir, state)
        return receipt


def _attempt(
    kings_source: KingsSource,
    reader: ChainReader,
    setter: WeightSetter,
    wallet_hotkey: str,
    params: Params,
    state: dict[str, Any],
    now: int,
) -> WeightReceipt:
    try:
        table, snapshot, count, newest, validator_hotkey = build_table(kings_source, reader, params, state.get("last_valid"))
    except Keep as keep:
        return WeightReceipt("deferred", keep.block, f"{keep.reason}; nothing submitted, the last valid weights stay")
    if wallet_hotkey != validator_hotkey:
        reason = f"the wallet signs as {wallet_hotkey}, but the genesis names {validator_hotkey} as the validator hotkey"
        return WeightReceipt("deferred", snapshot.block, f"{reason}; nothing submitted")
    state["last_valid"] = {
        "weights": {str(uid): weight for uid, weight in table.items()},
        "block": snapshot.block,
        "computed_at": now,
        "kings": count,
        "newest_king_at": newest,
    }
    try:
        return setter.set_weights(table)
    except Exception as exc:
        return WeightReceipt("deferred", snapshot.block, f"the weight setter failed: {exc}")


def dry_run(kings_source: KingsSource, client: Any, reader: ChainReader, params: Params, wallet_hotkey: str = "") -> dict[str, Any]:
    """What a run would do now. It writes nothing: not to the chain, and not to a file.

    With ``wallet_hotkey`` (the address read from the hotkey file) it also runs the setter's checks before a submission.
    """
    try:
        table, snapshot, count, _, validator_hotkey = build_table(kings_source, reader, params)
    except Keep as keep:
        return {"status": "would defer", "block": keep.block, "reason": keep.reason}
    report: dict[str, Any] = {"status": "would set", "block": snapshot.block, "kings": count, "weights": table, "validator_hotkey": validator_hotkey}
    uids = [uid for uid, hotkey in snapshot.hotkeys().items() if hotkey == validator_hotkey]
    report["validator_registered"] = bool(uids)
    if uids:
        try:
            report["last_write_block"] = int(client.last_update(snapshot.block)[uids[0]])
            report["blocks_since_last_write"] = snapshot.block - report["last_write_block"]
        except Exception:  # an extra for the operator: a failed read leaves it out
            pass
    if wallet_hotkey:
        report["wallet_hotkey"] = wallet_hotkey
        if wallet_hotkey != validator_hotkey:
            report |= {"status": "would defer", "reason": "the wallet does not sign as the validator hotkey of the genesis"}
        else:
            refusal = Setter(client, wallet_hotkey, params.weights_rate_limit_blocks).preflight(table)[1]
            report |= {"status": "would defer", "reason": refusal} if refusal else {"preflight": "passed"}
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sn120_v2.weights_writer", description="Set the SN120 chain weights once: the last 10 kings, 1/N each."
    )
    # No setting has a default path, network or wallet. A missing one is an error.
    parser.add_argument("--state-dir", type=Path, required=True, help="directory of the controller's journal; the writer's lock and memory go here too")
    parser.add_argument("--network", required=True, help="the chain: finney, or the address of a node")
    parser.add_argument("--wallet", help="NAME/HOTKEY of the validator wallet on this machine: only its hotkey file is read; required unless --dry-run")
    parser.add_argument("--wallet-path", default="", help="directory that holds the wallets; default: the SDK's ~/.bittensor/wallets")
    parser.add_argument("--params", help="JSON file of parameter overrides. The payout uses the parameters pinned in the journal when it has them")
    parser.add_argument("--dry-run", action="store_true", help="compute and print the table: no chain write, no file written; --wallet adds the checks before a submission")
    parser.add_argument("--force", action="store_true", help="run even if this hour already had its attempt")
    args = parser.parse_args(argv)
    if not args.dry_run and not args.wallet:
        parser.error("--wallet is required unless --dry-run")
    kings_source = partial(load_kings, args.state_dir)
    try:
        params = load_params(args.params) if args.params else Params()
        client = SubtensorClient.connect(params.netuid, args.network, args.wallet or "", args.wallet_path)
        wallet_hotkey = client.signer() if args.wallet else ""
    except ConfigError as exc:  # for example a missing hotkey file: one clear line, and no chain call
        print(f"nothing submitted: {exc}", file=sys.stderr)
        return 2
    reader = Reader(client, params.netuid)
    if args.dry_run:
        print(json.dumps({"dry_run": True} | dry_run(kings_source, client, reader, params, wallet_hotkey), sort_keys=True))
        return 0
    receipt = run_once(
        load_kings=kings_source,
        reader=reader,
        setter=Setter(client, wallet_hotkey, params.weights_rate_limit_blocks),
        wallet_hotkey=wallet_hotkey,
        params=params,
        state_dir=args.state_dir,
        now=int(time.time()),
        force=args.force,
    )
    print(receipt.to_json().decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
