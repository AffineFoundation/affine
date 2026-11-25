"""CLI for resumable collection of scorer output into the private work directory."""
from __future__ import annotations

import argparse
import sys

from ..errors import V2Error
from .config import Config, add_config
from .epoch import epoch_id
from .remote import ScorerRemote


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sn120_v2.generator.collect")
    add_config(parser)
    parser.add_argument("--epoch", required=True)
    args = parser.parse_args(argv)
    try:
        config = Config.load(args.config)
        remote = ScorerRemote.from_config(config.ssh, config.work_dir)
        result = remote.collect(epoch_id(args.epoch))
        print(f"collect {result.epoch}: {result.state}")
        return 0
    except V2Error as exc:
        print(f"collect failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
