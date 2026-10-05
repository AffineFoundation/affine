"""Software bootstrap for explicitly reviewed, quiescent historical caches.

ROOT signs the exact ownership catalogue and durable R2 inventory, then runs
this only after stopping all original readers of those roots. No discovery,
recursive delete, model hashing, credential access, or source changes occur.
The same lifecycle code used after every job retires the adopted old inputs.
"""
import argparse
import json
from pathlib import Path
import time
from subnet.cache_lifecycle import CacheLifecycle
from subnet.distributed_roles import authenticate


def apply(envelope,authority,*,now=None):
    payload=authenticate(envelope,authority)
    now=time.time() if now is None else now
    if payload.get('revision')!='owned-verifier-cache-catalog-v1' or not payload.get('quiescent_readers_confirmed'):
        raise ValueError('reviewed quiescent cache catalogue required')
    if not payload['created_at']<=now<payload['expires_at']:raise ValueError('ownership catalogue lifetime')
    results=[]
    for entry in payload['roots']:
        root=Path(entry['root'])
        if not root.is_absolute() or root!=root.resolve():raise ValueError('exact owned root')
        cache=CacheLifecycle(root)
        # Caller ROOT attests authenticated durable R2 inventory already exists;
        # input eviction does not require another full checkpoint download.
        for cp in entry['checkpoints']:
            with cache.lease_checkpoint(cp['id'],blocking=False):
                cache.adopt_checkpoint(cp['id'],root/'checkpoints'/cp['id'],cp['files'],cp['durability_ack'])
        removed=cache.evict_checkpoints(exclude=entry.get('keep',[]),keep=0)
        results.append(dict(root=str(root),removed=removed,available_bytes=__import__('shutil').disk_usage(root).free))
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--catalog',required=True);parser.add_argument('--authority',required=True)
    args=parser.parse_args();print(json.dumps(apply(json.loads(Path(args.catalog).read_text()),args.authority),sort_keys=True))

if __name__=='__main__':main()
