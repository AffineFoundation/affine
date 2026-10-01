#!/usr/bin/env python3
"""SN120 owner-only weight heartbeat. Default is read-only; --execute submits."""
import argparse
import fcntl
import json
import logging
import os
import signal
from pathlib import Path

import bittensor as bt

NETUID = 120
EXPECTED_OWNER = "5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8"
STATE = Path.home() / ".local/state/affine-burn"
LOG = logging.getLogger("affine-burn")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (STATE / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOG.info("another invocation is active; skipped")
            return
        chain = bt.subtensor(network="finney")
        block = chain.block
        storage = bt.storage.SubtensorModule

        def query(name, params):
            return chain.query(getattr(storage, name), params, block=block)

        owner = query("SubnetOwnerHotkey", [NETUID])
        if owner != EXPECTED_OWNER:
            raise RuntimeError("subnet owner hotkey changed; operator review required")
        owner_uid = query("Uids", [NETUID, owner])
        if owner_uid is None:
            raise RuntimeError("owner hotkey is not registered; refusing to choose another UID")
        owner_uid = int(owner_uid)
        wallet = bt.Wallet(name="default", hotkey="default", path=str(Path.home() / ".bittensor/wallets"))
        if wallet.hotkey.ss58_address != EXPECTED_OWNER:
            raise RuntimeError("wallet hotkey does not match expected owner")
        version = int(query("WeightsVersionKey", [NETUID]) or 0)
        rate = int(query("WeightsSetRateLimit", [NETUID]) or 0)
        last_updates = query("LastUpdate", [NETUID])
        last = int(last_updates[owner_uid]) if owner_uid < len(last_updates) else 0
        remaining = max(0, rate - (block - last)) if last else 0
        # The SDK and chain exempt a single self-weight from minimum count/max clip.
        intent = bt.SetWeights(netuid=NETUID, uids=[owner_uid], weights=[1.0], version_key=version)
        status = dict(netuid=NETUID, network="finney", block=block, owner_hotkey=owner,
                      owner_uid=owner_uid, weight=1.0, version_key=version,
                      commit_reveal=bool(query("CommitRevealWeightsEnabled", [NETUID])),
                      rate_limit_blocks=rate, remaining_blocks=remaining,
                      status="validated")
        LOG.info("owner destination validated: %s", json.dumps(status, sort_keys=True))
        if remaining:
            status["status"] = "deferred_rate_limit"
            LOG.info("last weight update at block %d; deferred %d blocks; next hourly tick will retry", last, remaining)
        else:
            # Read-only plan validates construction, policy and fees before any submission.
            plan = chain.plan(intent, wallet)
            if not plan.ok:
                raise RuntimeError("SDK plan denied owner weight submission")
            status["status"] = "planned"
            LOG.info("SDK read-only plan allowed owner-only submission")
            if args.execute:
                result = chain.execute(intent, wallet, wait_for_inclusion=True,
                                       wait_for_finalization=True, retries=0)
                result.raise_for_failure()
                status.update(status="submitted", block_hash=str(result.block_hash))
                LOG.info("owner weights accepted: block_hash=%s", result.block_hash)
        temporary = STATE / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(STATE / "status.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    def timed_out(_signum, _frame):
        raise TimeoutError("chain invocation exceeded 11 minutes")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(660)
    try:
        main()
    except bt.ChainError as error:
        # The live legacy validator may win the race after our preflight.
        if error.code == bt.ErrorCode.RATE_LIMITED:
            LOG.info("chain rate limit raced our preflight; deferred to next hourly tick")
        else:
            LOG.error("chain submission failed (%s); next hourly tick will retry", error.code)
            raise SystemExit(1)
    except Exception as error:
        # Do not print wallet objects or private key material, including exception payloads.
        LOG.error("burn invocation failed: %s; inspect configuration/chain before retrying", type(error).__name__)
        raise SystemExit(1)
