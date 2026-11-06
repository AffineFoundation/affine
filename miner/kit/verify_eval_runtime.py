#!/usr/bin/env python3
"""Verify published live-probe bytes and eligible-task indexing; no network/GPU.
Does not predict the private draw or certify a GPU replay. --check-packages also
checks installed package versions (optional; no packages are installed here).
"""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path


def verify(root, check_packages=False, check_worker_packages=False):
    root=Path(root).resolve()
    pins=json.loads((root/'validator/probe/runtime/PINS.json').read_text())
    for rel,expected in pins['files'].items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root): raise ValueError('pin escapes miner root: '+rel)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected: raise ValueError('sha256 mismatch: '+rel)
    snap=pins['snapshot']; packed=(root/snap['path']).resolve()
    if not packed.is_relative_to(root): raise ValueError('snapshot escapes miner root')
    with gzip.open(packed,'rb') as f:
        raw=f.read(64*1024*1024+1)
    if len(raw)>64*1024*1024: raise ValueError('snapshot exceeds bound')
    if hashlib.sha256(raw).hexdigest()!=snap['uncompressed_sha256']: raise ValueError('snapshot sha256 mismatch')
    rows=json.loads(raw)
    if len(rows)!=snap['indexed_rows']: raise ValueError('snapshot row count mismatch')
    eligible=[json.loads(l) for l in (root/'data/gate-pool-v2.jsonl').read_text().splitlines() if l.strip()]
    if len(eligible)!=snap['eligible_tasks']: raise ValueError('eligible pool count mismatch')
    indices=set()
    for row in eligible:
        env,index=row['id'].split(':'); index=int(index)+(snap['dapo_offset'] if env=='dapo' else 0)
        actual=rows[index]['data']
        if (actual['problem'],actual['answer'])!=(row['problem'],row['answer']): raise ValueError('probe index mismatch: '+row['id'])
        if index in indices: raise ValueError('duplicate eligible probe index')
        indices.add(index)
    if set(range(len(rows)))-indices!=set(snap['excluded_math_indices']): raise ValueError('excluded index set mismatch')
    serve=(root/'validator/probe/serve.sh').read_text()
    if '--dtype bfloat16' not in serve or '--max-model-len 69632' not in serve or '${MAX_SEQS:-512}' not in serve:
        raise ValueError('serve dtype/context/concurrency mismatch')
    if '--quantization' in serve or '--kv-cache-dtype' in serve: raise ValueError('unapproved quantization flag')
    harness=(root/'validator/probe/runtime/complete_group_harness.py').read_text()
    if "1 <= value['max_output_tokens'] <= 65536" not in harness or "value['max_output_tokens'] < value['max_context_tokens'] <= 69632" not in harness:
        raise ValueError('harness cap mismatch')
    if check_packages and check_worker_packages:
        raise ValueError('serving and worker are separate environments; select one')
    if check_packages or check_worker_packages:
        versions=pins['worker_runtime'] if check_worker_packages else pins['runtime']
        for package,expected in versions.items():
            if importlib.metadata.version(package)!=expected: raise ValueError('package version mismatch: '+package)
    return dict(verified=True,files=len(pins['files']),indexed_rows=len(rows),eligible_tasks=len(eligible),
                eligible_index_mapping='all matched',packages_checked=check_packages,worker_packages_checked=check_worker_packages,
                parallel_eval_live=False,gpu_replay_verified=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--miner-root',type=Path,default=Path(__file__).resolve().parent.parent)
    env=ap.add_mutually_exclusive_group()
    env.add_argument('--check-packages',action='store_true',help='serving environment package versions')
    env.add_argument('--check-worker-packages',action='store_true',help='separate probe-worker environment versions')
    a=ap.parse_args()
    print(json.dumps(verify(a.miner_root,a.check_packages,a.check_worker_packages),sort_keys=True))


if __name__=='__main__': main()
