"""One minute systemd entry point. A nonblocking lock prevents overlapping ticks."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

from ..errors import ConfigError, V2Error
from .common import open_store, utc_text
from .config import Config, add_config
from .control import tick as run_tick
from .remote import ScorerRemote


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sn120_v2.generator.tick")
    add_config(parser)
    parser.add_argument("--now", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        config = Config.load(args.config)
        config.work_dir.mkdir(parents=True, exist_ok=True)
        with (config.work_dir / "tick.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("tick: another run holds the lock")
                return 0
            from ..chain import SubtensorClient
            from .pool import PriorPool
            client = SubtensorClient.connect(120, config.network)
            try:
                events = run_tick(
                    config, now=time.time() if args.now is None else args.now,
                    client=client, store=open_store(config.credentials_file), pool=PriorPool(config.pool_dir),
                    remote=ScorerRemote.from_config(config.ssh, config.work_dir) if config.ssh else None,
                )
            finally:
                client.chain.close()
        print(json.dumps({"tick": utc_text(time.time()), "events": events}, sort_keys=True))
        return 0
    except V2Error as exc:
        print(f"tick failed; completed files are preserved and the next run resumes: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
