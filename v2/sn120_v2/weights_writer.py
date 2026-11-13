"""The single hourly writer of chain weights.

One run is one attempt. It ends in ``set`` or ``deferred`` and it never loops:
the next attempt is the next hourly run.

The table is the payout formula over the kings and one complete chain
snapshot. The burn table (all weight to ``burn_uid`` when no payee is paid) is
a result of that formula. It is never a fallback: when the chain read fails
or is partial, or the state does not load, the run submits nothing, so the
last valid weights stay on chain.

Run it from one hourly timer: ``python -m sn120_v2.weights_writer``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from . import payout
from .chain import Reader, Setter, SubtensorClient
from .errors import ContractError, InfraError, V2Error
from .params import Params, load_params, require_env
from .seams import ChainReader, WeightSetter
from .types import King, WeightReceipt

LOCK_FILE = "weights.lock"
STATE_FILE = "weights_state.json"


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


def load_kings_file(path: Path) -> tuple[King, ...]:
    """Kings, oldest first, from a JSON list of ``King`` records. A missing or malformed file raises."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ContractError("the kings file must hold a JSON list of kings")
    return tuple(King.from_dict(row) for row in rows)


def run_once(
    *,
    load_kings: Callable[[], Sequence[King]],
    reader: ChainReader,
    setter: WeightSetter,
    params: Params,
    state_dir: Path,
    now: int,
    force: bool = False,
) -> WeightReceipt:
    """One attempt to set the weights. ``load_kings`` returns the applied kings, oldest first.

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
        receipt = _attempt(load_kings, reader, setter, params, state, now)
        state["last_receipt"] = receipt.to_dict()
        save_state(state_dir, state)
        return receipt


def _attempt(
    load_kings: Callable[[], Sequence[King]],
    reader: ChainReader,
    setter: WeightSetter,
    params: Params,
    state: dict[str, Any],
    now: int,
) -> WeightReceipt:
    def kept(block: int, reason: str) -> WeightReceipt:
        return WeightReceipt("deferred", block, f"{reason}; nothing submitted, the last valid weights stay")

    try:
        snapshot = reader.snapshot()
    except InfraError as exc:
        return kept(0, f"no complete chain snapshot: {exc}")
    if snapshot.netuid != params.netuid:
        return kept(snapshot.block, f"the chain snapshot is for netuid {snapshot.netuid}, not {params.netuid}")
    try:
        kings = tuple(load_kings())
        newest = max((king.applied_at for king in kings), default=0)
    except Exception as exc:
        return kept(snapshot.block, f"the state failed to load: {exc}")
    # The kings list only grows. An older or empty list means a bad state read, which must not turn into a burn.
    if newest < state.get("last_valid", {}).get("newest_king_at", 0):
        return kept(snapshot.block, "the kings list went backwards, so the state did not load correctly")
    try:
        table = payout.weights(kings, snapshot.hotkeys(), params)
    except V2Error as exc:
        return kept(snapshot.block, f"the payout refused the table: {exc}")
    state["last_valid"] = {
        "weights": {str(uid): weight for uid, weight in table.items()},
        "block": snapshot.block,
        "computed_at": now,
        "newest_king_at": newest,
    }
    try:
        return setter.set_weights(table)
    except Exception as exc:
        return WeightReceipt("deferred", snapshot.block, f"the weight setter failed: {exc}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set the SN120 chain weights once: last-10-kings, 1/N each.")
    parser.add_argument("--dry-run", action="store_true", help="read the chain and print the table; submit nothing")
    parser.add_argument("--force", action="store_true", help="run even if this hour already had its attempt")
    args = parser.parse_args(argv)
    # Every setting is required. A missing one is an error, never a default path, network or wallet.
    params = load_params(require_env("SN120_V2_PARAMS_FILE"))
    kings_file = Path(require_env("SN120_V2_KINGS_FILE"))
    network = require_env("SN120_V2_CHAIN_NETWORK")
    if args.dry_run:
        registered = Reader(SubtensorClient.connect(params.netuid, network), params.netuid).snapshot().hotkeys()
        table = payout.weights(load_kings_file(kings_file), registered, params)
        print(json.dumps({"dry_run": True, "weights": table}, sort_keys=True))
        return 0
    state_dir = Path(require_env("SN120_V2_WEIGHTS_STATE_DIR"))
    validator_hotkey = require_env("SN120_V2_VALIDATOR_HOTKEY")
    client = SubtensorClient.connect(params.netuid, network, require_env("SN120_V2_WALLET"))
    receipt = run_once(
        load_kings=lambda: load_kings_file(kings_file),
        reader=Reader(client, params.netuid),
        setter=Setter(client, validator_hotkey, params.weights_rate_limit_blocks),
        params=params,
        state_dir=state_dir,
        now=int(time.time()),
        force=args.force,
    )
    print(receipt.to_json().decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
