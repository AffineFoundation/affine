#!/usr/bin/env python
"""Emission kill-switch for SN120: pin all weights on the burn UID, forever.

While this loop runs, every weight-set from the validator hotkey routes 100%
of the weight vector to [subnet].burn_uid from affine/affine.toml — miners
receive no incentive. This is the same call the production validator makes
when its king chain is empty (affine.chain.set_rolling_weights fallback).

HOW TO TURN ON / OFF — use the wrapper, not this file directly:

    ~/subnet120/ops/burnctl.sh on        # start burning (background loop)
    ~/subnet120/ops/burnctl.sh off       # stop burning
    ~/subnet120/ops/burnctl.sh status    # running? + last log lines

Log: ops/burn.log   Pidfile: ops/burn.pid

WARNING: do not run this at the same time as the production validator —
both set weights with the same hotkey and the last writer wins each
interval. Turn this OFF before starting the validator (and vice versa).
"""

import logging
import signal
import time
import tomllib
from pathlib import Path

import bittensor as bt

ROOT = Path(__file__).resolve().parent.parent
CONTRACT_TOML = ROOT / "affine" / "affine.toml"
# Matches [validator].weight_interval_s; chain-side rate limiting may reject
# some attempts, which is harmless — the loop just tries again next tick.
INTERVAL_S = 300

log = logging.getLogger("burn")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = tomllib.loads(CONTRACT_TOML.read_text())
    sub = cfg["subnet"]
    netuid = int(sub["netuid"])
    burn_uid = int(sub["burn_uid"])
    version_key = int(sub["weight_version_key"])
    wallet = bt.Wallet(name=cfg["wallet"]["name"], hotkey=cfg["wallet"]["hotkey"])
    subtensor = bt.subtensor(network=sub["network"])

    stopping = False

    def _stop(signum, _frame):
        nonlocal stopping
        stopping = True
        log.info("signal %d received; stopping after current tick", signum)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    log.info(
        "burn loop up: netuid=%d burn_uid=%d version_key=%d wallet=%s/%s interval=%ds",
        netuid, burn_uid, version_key, wallet.name, wallet.hotkey_str, INTERVAL_S,
    )
    while not stopping:
        try:
            res = subtensor.execute(
                bt.SetWeights(netuid=netuid, uids=[burn_uid], weights=[1.0],
                              version_key=version_key),
                wallet,
            )
            res.raise_for_failure()
            log.info("weights pinned to burn uid %d (block=%s)", burn_uid, res.block_hash)
        except Exception:
            log.exception("set_weights failed (rate limit or chain error); retrying next tick")
        # Sleep in 1s slices so on/off toggling is responsive.
        for _ in range(INTERVAL_S):
            if stopping:
                break
            time.sleep(1)
    log.info("burn loop stopped")


if __name__ == "__main__":
    main()
