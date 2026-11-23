"""Freeze the public history of SN120 v1 into one file, once, at cutover.

    python v2/site/v1_archive.py --from <v1 site: URL or directory> --out <file>

It reads four public files of the v1 website and nothing else: ``k1.json``
(the kings and the last turns), ``k1/turns/index.json`` (the older turns),
``scores.json`` (the weight shares) and ``benchmarks.json``. It writes one
JSON file. ``build.py --v1-history <file>`` copies that file into the v2 site,
and the page ``v1.html`` shows it. After this command, the v1 page needs no v1
service and no v1 file.

All four files must be readable, or nothing is written: an archive is taken
once, so it must be whole. Every output field is named here. Run it after the
v1 services stop and before the v1 website is replaced. Keep the file: it
cannot be made again when the v1 website is gone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from build import V1_SCHEMA, bench_view, read, write

FILES = {"k1": "k1.json", "index": "k1/turns/index.json", "scores": "scores.json", "bench": "benchmarks.json"}
TURN_FIELDS = ("uid", "hotkey", "verdict", "z", "c_pass", "h_pass", "tasks", "rollouts", "candidate", "head")


def archive(files: dict[str, dict[str, Any]], source: str, now: float) -> dict[str, Any]:
    """The v1 files, cut down to what ``v1.html`` shows: the kings, every decided or open turn, and the benchmarks."""
    k1 = files["k1"]
    share = {(miner["uid"], miner["hotkey"]): miner.get("share_now") for miner in files["scores"]["miners"]}
    kings = [{
        "rank": rank, "uid": king["uid"], "hotkey": king["hotkey"], "epoch": king["green_epoch"], "z": king.get("z"),
        "share": share.get((king["uid"], king["hotkey"])),
    } for rank, king in enumerate(k1["payees"], 1)]

    turns: dict[int, dict[str, Any]] = {}
    for old in files["index"]["turns"]:  # the older turns: the result, the UID, the task count and the candidate only
        turns[old["epoch"]] = {
            "epoch": old["epoch"], "uid": old.get("uid"), "verdict": old.get("decision"), "tasks": old.get("tasks"),
            "candidate": old.get("candidate"),
        }
    for turn in k1["turns"]:  # the last turns, with their scores; a "skip" epoch trained nobody and is not a turn
        if turn.get("status") == "turn":
            row = {name: turn.get(name) for name in TURN_FIELDS}
            turns[turn["epoch_index"]] = row | {"epoch": turn["epoch_index"], "cap_rate": turn.get("cap"), "at": turn.get("turn_at")}
    return {
        "schema": V1_SCHEMA, "archived_at": now, "source": source, "v1_generated_at": k1["generated_at"],
        "kings": kings, "turns": [turns[epoch] for epoch in sorted(turns, reverse=True)], "bench": bench_view(files["bench"]),
    }


def main(argv: list[str] | None = None, clock: Callable[[], float] = time.time) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--from", dest="source", required=True, help="the v1 website: an http(s) URL or its directory")
    parser.add_argument("--out", required=True, type=Path, help="the file to write, for build.py --v1-history")
    args = parser.parse_args(argv)
    try:
        files = {name: json.loads(read(args.source, key)) for name, key in FILES.items()}
        data = json.dumps(archive(files, args.source, clock()), separators=(",", ":"), allow_nan=False).encode()
    except Exception as error:  # noqa: BLE001 - a part that cannot be read means no archive at all
        print(f"v1 archive: nothing written: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(args.out, data)
    made = json.loads(data)
    print(f"v1 archive: wrote {args.out} ({len(data)} bytes): {len(made['kings'])} kings, {len(made['turns'])} turns, "
          f"{len(made['bench']['rows'])} benchmark rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
