"""Bring the journal on the weights host up to date, from the journal copy in the object store.

    python -m sn120_v2.kings --state-dir <dir>
    python -m sn120_v2.weights_writer --state-dir <dir> ...

For the host that holds the wallet. It does not run the controller, so it
has no journal of its own. This command pulls the records it does not have
yet into ``<dir>/journal.jsonl``, checks the whole chain of checksums, and
prints the kings. The weights writer then reads the kings from that file.
Run it before each run of the writer.

It only reads the store (``SN120_R2_*``; a read-only token is enough). When
the store does not answer, the command fails and the local file stays as it
is: the writer then uses the last valid kings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

from .errors import V2Error
from .journal import JOURNAL_FILE
from .r2 import R2Store
from .seams import ObjectStore
from .state import State, load_from_store


def sync(store: ObjectStore, state_dir: Path) -> State:
    state = load_from_store(store, Path(state_dir) / JOURNAL_FILE)
    if state.genesis is None:
        raise V2Error("the stored journal has no genesis yet")
    return state


def main(argv: Sequence[str] | None = None, open_store: Callable[[], ObjectStore] = R2Store.from_env) -> int:
    parser = argparse.ArgumentParser(prog="python -m sn120_v2.kings", description="Pull the journal copy from the store and print the kings.")
    parser.add_argument("--state-dir", type=Path, required=True, help="the --state-dir of the weights writer on this host")
    args = parser.parse_args(argv)
    try:
        state = sync(open_store(), args.state_dir)
    except V2Error as exc:
        print(f"journal not changed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    kings = [{"uid": king.uid, "hotkey": king.hotkey, "round_id": king.round_id} for king in state.kings]
    print(json.dumps({"kings": kings, "head_version_id": state.head.version_id, "validator_hotkey": state.genesis.validator_hotkey}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
