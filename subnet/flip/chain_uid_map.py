#!/usr/bin/env python3
"""Read-only SN120 hotkey->uid map from chain (SubtensorModule.Keys[netuid]). No signing, no writes on chain.
Prints one JSON object: {netuid, block, taken_at, n, hotkey_to_uid}. Used by flip_scorer.chain_hotkey_uid_map."""
import json, sys, time
import bittensor as bt
from bittensor._generated.storage import SubtensorModule as S

NETUID = int(sys.argv[1]) if len(sys.argv) > 1 else 120
with bt.Subtensor("finney") as c:
    blk = c.block() if callable(c.block) else c.block
    keys = dict(c.query_map(S.Keys, [NETUID], block=blk))
    n = int(c.query(S.SubnetworkN, [NETUID], block=blk))
m = {str(hk): int(uid) for uid, hk in keys.items()}
if not m or len(keys) != n:
    raise SystemExit("incomplete metagraph keys: %d of %d" % (len(keys), n))
print(json.dumps(dict(netuid=NETUID, block=int(blk), taken_at=time.time(), n=n, hotkey_to_uid=m), sort_keys=True))
