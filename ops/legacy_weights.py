#!/usr/bin/env python
"""Pin SN120 weights on the previous winners of the old Affine mechanism.

Snapshot taken at archive block 8763869 (2026-08-03, before the burn loop
flushed incentive to the burn UID): five hotkeys each held incentive 0.20.
Every tick this loop resolves those hotkeys against the live metagraph and
splits the weight vector equally across the ones still registered
(deregistered hotkeys are skipped; if none remain, falls back to the burn
UID from affine/affine.toml).

HOW TO TURN ON / OFF — use the wrapper, not this file directly:

    ~/subnet120/ops/legacyctl.sh on        # start pinning (background loop)
    ~/subnet120/ops/legacyctl.sh off       # stop pinning
    ~/subnet120/ops/legacyctl.sh status    # running? + last log lines

Log: ops/legacy.log   Pidfile: ops/legacy.pid

WARNING: do not run this at the same time as the burn loop or the
production validator — all three set weights with the same hotkey and the
last writer wins each interval.
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

# Incentive holders on netuid 120 at archive block 8763869 (0.20 each),
# i.e. the winners crowned by the old Affine mechanism before the burn.
LEGACY_WINNERS = [
    "5EpvnXGu8jUAVc67oPGgJ3brR4JZqjBUSaTKhZuBoNAAzSJF",  # uid 58 at snapshot
    "5ECeJJpEMjW4pxM9eGyJ5ua3Sebfyr8kcVwLAdaiJLUC8pkW",  # uid 107
    "5CQLBK7Mmw1vsk7eQcBok9Qn44JNU5YVrfNmZpJHPxLV271B",  # uid 191
    "5EkhZHopy9CAoUhKmVTDsyGQi7Voo9gURYPnNDiMZX1pQZxp",  # uid 195 (dereg'd 2026-08-04)
    "5DSW4cTwQt2U8rck6mFN1nNqoj37j1waqwszQDuz2zh9zC7z",  # uid 201
]

log = logging.getLogger("legacy")


def resolve_uids(subtensor, netuid: int) -> list[int]:
    meta = subtensor.subnets.metagraph(netuid)
    uid_of = {hk: uid for uid, hk in enumerate(meta.hotkeys)}
    uids = []
    for hk in LEGACY_WINNERS:
        uid = uid_of.get(hk)
        if uid is None:
            log.warning("legacy winner %s… no longer registered; skipping", hk[:16])
        else:
            uids.append(uid)
    return uids


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
        "legacy-winner loop up: netuid=%d version_key=%d wallet=%s/%s interval=%ds",
        netuid, version_key, wallet.name, wallet.hotkey_str, INTERVAL_S,
    )
    while not stopping:
        try:
            uids = resolve_uids(subtensor, netuid)
            if not uids:
                log.error("no legacy winners registered; falling back to burn uid %d", burn_uid)
                uids = [burn_uid]
            w = 1.0 / len(uids)
            res = subtensor.execute(
                bt.SetWeights(netuid=netuid, uids=uids, weights=[w] * len(uids),
                              version_key=version_key),
                wallet,
            )
            res.raise_for_failure()
            log.info("weights pinned to legacy winners uids=%s weight=%.3f each (block=%s)",
                     uids, w, res.block_hash)
        except Exception:
            log.exception("set_weights failed (rate limit or chain error); retrying next tick")
        # Sleep in 1s slices so on/off toggling is responsive.
        for _ in range(INTERVAL_S):
            if stopping:
                break
            time.sleep(1)
    log.info("legacy-winner loop stopped")


if __name__ == "__main__":
    main()
