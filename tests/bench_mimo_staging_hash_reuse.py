"""Bounded local write/hash microbenchmark; no model, GPU, bucket or live state."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time
import os

from subnet.mimo_grpo import transaction as t
from subnet.mimo_grpo.optimizer import _hash_file as legacy_hash
from test_mimo_transaction_hash_memo import fixture


def run(root,size,shared):
    tx,callbacks,_=fixture(root);counts={'bytes_hashed':0,'full_hashes':0}
    original=hashlib.sha256
    class Digest:
        def __init__(self):self.h=original();counts['full_hashes']+=1
        def update(self,data):counts['bytes_hashed']+=len(data);self.h.update(data)
        def hexdigest(self):return self.h.hexdigest()
        def digest(self):return self.h.digest()
    def counted(*args,**kwargs):return original(*args,**kwargs)if args or kwargs else Digest()
    def write(root):
        names=['model/config.json','model/model.safetensors','optimizer/state.bin']
        for name in names:
            p=root/name;p.parent.mkdir(exist_ok=True)
            with p.open('xb')as f:
                if name.startswith('optimizer/'):
                    block=b'o'*(1024**2)
                    for _ in range(size//len(block)):f.write(block)
                else:f.write(b'{}'if name.endswith('.json')else b'model')
                f.flush();os.fsync(f.fileno())
        return names
    def stage_legacy(report,root):
        names=write(root)
        for name in names:legacy_hash(root/name) # save_local + model inventory
        for name in names:
            if name.startswith('optimizer/'):legacy_hash(root/name) # state inventory
        return names
    def stage_shared(report,root,hasher):
        names=write(root)
        for name in names:hasher(root/name)
        for name in names:
            if name.startswith('optimizer/'):hasher(root/name)
        return names
    callbacks.pop('stage')
    callbacks['stage_with_hasher'if shared else'stage']=stage_shared if shared else stage_legacy
    hashlib.sha256=counted;start=time.perf_counter()
    try:result=tx.run(**callbacks)
    finally:elapsed=time.perf_counter()-start;hashlib.sha256=original
    return dict(seconds=elapsed,**counts),result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mib',type=int,default=64);parser.add_argument('--rounds',type=int,default=3);args=parser.parse_args()
    if not 1<=args.mib<=256 or not 1<=args.rounds<=5:raise ValueError('bounded local benchmark only')
    rows=[]
    with tempfile.TemporaryDirectory(prefix='mimo-hash-bench-')as tmp:
        root=Path(tmp)
        for i in range(args.rounds):
            baseline,a=run(root/f'old-{i}',args.mib*1024**2,False)
            optimized,b=run(root/f'new-{i}',args.mib*1024**2,True)
            assert t.canonical(a)==t.canonical(b)
            rows.append(dict(baseline=baseline,optimized=optimized,journal_exact=True))
    print(json.dumps(dict(optimizer_mib=args.mib,rounds=rows,
        median_baseline_seconds=statistics.median(r['baseline']['seconds']for r in rows),
        median_optimized_seconds=statistics.median(r['optimized']['seconds']for r in rows),
        production_mutations=0,claim='local file I/O microbenchmark, not production epoch speed'),indent=2))

if __name__=='__main__':main()
