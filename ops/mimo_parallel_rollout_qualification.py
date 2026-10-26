#!/usr/bin/env python3
"""Historical full-length B1 replay qualification, never a new submission.

Every original nonce is regenerated under the original signed context. The
expired opening is evidence being replayed, not extended mining permission.
Independent processes own models/caches; one shared gate bounds proof prefill.
"""
import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

_ACTOR=None
_INDEX=None
_GATE=None


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def initialize(config,gate):
    global _ACTOR,_INDEX,_GATE
    helper=Path(config['helper'])
    if hashlib.sha256(helper.read_bytes()).hexdigest()!=config['helper_sha256']:
        raise ValueError('pinned qualification initializer changed')
    spec=importlib.util.spec_from_file_location('original_prefix_qualification',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.initialize(config)
    _ACTOR,_INDEX,_GATE=module._ACTOR,config['index'],gate


def generate(nonce):
    import torch
    from subnet.complete_group_mining import _rollout_from_runtime
    actor=_ACTOR.for_environment(_ACTOR.env_config,_ACTOR.harness)
    original_compute=actor.compute
    def compute(*args,**kwargs):
        with _GATE:return original_compute(*args,**kwargs)
    actor.compute=compute
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    row,arrays=actor.rollout(_INDEX,nonce)
    row=_rollout_from_runtime(actor,row,nonce)
    torch.cuda.synchronize()
    return dict(nonce=nonce,row=row,arrays=[a.tolist()for a in arrays],
        pid=os.getpid(),elapsed_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated())


def run(args):
    source=Path(args.source_root).resolve();sys.path.insert(0,str(source))
    from subnet.backend_jobs import signed
    from subnet.complete_group_codec import decode,limits
    opening=json.loads(Path(args.opening).read_bytes());m=signed(opening,args.authority)
    original_raw=Path(args.original).read_bytes()
    if hashlib.sha256(original_raw).hexdigest()!=args.original_sha256:
        raise ValueError('preserved original accepted archive digest')
    [(batch,arrays)]=decode(original_raw,limits(m))
    if batch['miner']!=args.miner or batch['epoch']!=m['epoch'] or len(batch['rollouts'])!=8:
        raise ValueError('exact original private N8 batch required')
    output=Path(args.output).absolute();output.mkdir(mode=0o700,parents=True,exist_ok=False)
    config=dict(source=str(source),release=args.release,opening=args.opening,
        authority=args.authority,miner=args.miner,checkpoint=args.checkpoint,
        environment=batch['env_id'],index=batch['index'],prefix_tokens=1,
        helper=args.helper,helper_sha256=hashlib.sha256(Path(args.helper).read_bytes()).hexdigest())
    mp=multiprocessing.get_context('spawn');gate=mp.Semaphore(1)
    began=time.monotonic();rows={}
    with ProcessPoolExecutor(max_workers=4,mp_context=mp,initializer=initialize,initargs=(config,gate))as pool:
        jobs={pool.submit(generate,n):n for n in range(8)}
        for future in as_completed(jobs):
            n=jobs[future];result=future.result()
            expected=batch['rollouts'][n]
            checks=dict(tokens=result['row']['turns'][0]['output']==expected['turns'][0]['output'],
                original_row=result['row']==expected,
                selected_probabilities=result['arrays']==[a.tolist()for a in arrays[n]])
            path=output/(str(n)+'.json');path.write_bytes(canonical(result));path.chmod(0o600)
            summary=dict(nonce=n,checks=checks,tokens=len(expected['turns'][0]['output']),
                elapsed_seconds=result['elapsed_seconds'],pid=result['pid'],
                peak_allocated_bytes=result['peak_allocated_bytes'])
            rows[n]=summary
            (output/'PROGRESS.json').write_bytes(canonical(dict(completed=list(rows.values()),
                elapsed_seconds=time.monotonic()-began)))
            print(json.dumps(summary),flush=True)
    result=dict(version='mimo-private-full-original-replica-qualification-v1',
        batch_dimension=1,replicas=4,historical_replay=True,new_submission=False,
        extended_epoch_permission=False,public_activation=False,
        original_archive_sha256=args.original_sha256,
        original_opening_sha256=hashlib.sha256(canonical(opening)).hexdigest(),
        exact_original_parity=all(all(r['checks'].values())for r in rows.values()),
        rows=[rows[n]for n in range(8)],elapsed_seconds=time.monotonic()-began)
    (output/'RESULT.json').write_bytes(canonical(result))
    if not result['exact_original_parity']:raise ValueError('full replica replay differs from original execution')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','release','opening','authority','miner','checkpoint','original',
                 'original-sha256','output','helper'):
        parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
