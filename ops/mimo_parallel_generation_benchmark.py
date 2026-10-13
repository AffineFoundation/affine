#!/usr/bin/env python3
"""Private B1 replica prefix benchmark; never creates admissible mining batches.

Runs the original signed sampler/context without changing its output horizon,
then deliberately stops the diagnostic prefix early. Each process owns its HF
wrapper, model, recurrent/KV caches and proof computation. This is prefix parity
and throughput evidence, not full-rollout or public release qualification.
"""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

_ACTOR = None
_AUTHORITY = None
_STEPS = None


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def initialize(config):
    global _ACTOR,_AUTHORITY,_STEPS
    source=Path(config['source']).resolve();sys.path.insert(0,str(source));os.chdir(source)
    release=json.loads(Path(config['release']).read_bytes())
    if hashlib.sha256(canonical(release['source'])).hexdigest()!=release['source_sha256']:
        raise ValueError('original source inventory digest')
    for name,row in release['source'].items():
        path=source/name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('original pinned source changed: '+name)
    from subnet.backend_jobs import signed
    manifest=signed(json.loads(Path(config['opening']).read_bytes()),config['authority'])
    if manifest.get('payable') is not False or not manifest['epoch'].startswith(('private-','test-','nonpayable-')):
        raise ValueError('isolated private original opening required')
    if manifest['source_bundle']['sha256']!=release['source_sha256']:
        raise ValueError('original sampler/source context differs')
    from subnet.protocol import entry,harness_for
    from subnet.mimo_runtime import MimoRuntime
    from subnet.complete_group_runtime import MimoGroupRuntime
    from subnet.complete_group_native import resolve_task
    definition=entry(manifest,config['environment']);h=harness_for(definition,config['index'])
    backend=MimoRuntime.load(config['checkpoint'],{k:manifest['checkpoint'][k]for k in ('id','files')},
                             manifest['mimo_runtime'],device='cuda',trainable=False)
    _ACTOR=MimoGroupRuntime(config['checkpoint'],manifest,definition['spec'],h,
                           miner=config['miner'],backend=backend)
    _AUTHORITY=resolve_task(manifest,config['environment'],config['index'],backend.tokenizer)
    _STEPS=config['prefix_tokens']
    if not 1<=_STEPS<=h['max_output_tokens']:
        raise ValueError('bounded diagnostic prefix within unchanged original horizon')


def ready(delay=.05):
    time.sleep(delay)
    return dict(pid=os.getpid(),checkpoint=_ACTOR.backend.checkpoint_id)


def generate(nonce):
    import torch
    from subnet.complete_group_sampling import prepare_draws,supported_logits
    from subnet.complete_group_distribution import distribution,pick,eos_ids
    actor=_ACTOR;backend=actor.backend;prompt=list(_AUTHORITY.prompt)
    draw=prepare_draws(actor.sampling_context,actor.spec.id,_AUTHORITY.task_hash,
                      _AUTHORITY.index,nonce,0,prompt)
    inputs=prompt;cache=None;output=[];probabilities=[]
    enqueue_seconds=0.;sampling_seconds=0.;torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize();start=time.monotonic()
    with torch.inference_mode():
        for position in range(_STEPS):
            t=time.monotonic()
            logits,cache=backend.next_token_logits(inputs,past_key_values=cache,use_cache=True)
            enqueue_seconds+=time.monotonic()-t;t=time.monotonic()
            p=distribution(supported_logits(actor,logits),actor.harness['temperature'],
                           actor.harness['top_p'],actor.harness['top_k'])
            token=pick(p,draw(position));output.append(token);probabilities.append(float(p[token].log()))
            sampling_seconds+=time.monotonic()-t;inputs=[token]
            if token in eos_ids(actor.sampling_context['token_geometry']):break
    torch.cuda.synchronize();sampling_elapsed=time.monotonic()-start
    # Same B1 prefill and TOPLOC builder as the original actor; this proves only
    # the diagnostic prefix, never pretends an early stop is an eligible output.
    del cache
    proof_started=time.monotonic();acts,full=actor.compute(prompt,output)
    from toploc import build_proofs_base64
    proofs=build_proofs_base64(acts,decode_batching_size=16,topk=128)
    selected=[float(full[i,t])for i,t in enumerate(output)]
    torch.cuda.synchronize()
    return dict(nonce=nonce,pid=os.getpid(),prompt_sha256=hashlib.sha256(canonical(prompt)).hexdigest(),
        task_hash=_AUTHORITY.task_hash,output=output,behavior_logprobs=probabilities,
        raw_t1_logprobs=selected,proofs=proofs,diagnostic_prefix_only=True,
        sampling_seconds=sampling_elapsed,forward_enqueue_host_seconds=enqueue_seconds,
        distribution_and_wait_seconds=sampling_seconds,proof_prefill_seconds=time.monotonic()-proof_started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start)


def run(args):
    output=Path(args.output).absolute();output.mkdir(parents=True,exist_ok=False,mode=0o700)
    parallel=Path(args.parallel_module).resolve()
    spec=importlib.util.spec_from_file_location('isolated_parallel_collector',parallel)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from nacl.signing import VerifyKey
    opening=json.loads(Path(args.opening).read_bytes())
    VerifyKey(bytes.fromhex(args.authority)).verify(canonical(opening['payload']),base64.b64decode(opening['signature']))
    if opening['signer']!=args.authority:raise ValueError('original opening authority')
    m=opening['payload'];n=m['group_contract']['group_size']
    if n!=8:raise ValueError('this study compares complete N8')
    config=dict(source=str(Path(args.source_root).resolve()),release=str(Path(args.release).resolve()),
        opening=str(Path(args.opening).resolve()),authority=args.authority,miner=args.miner,
        checkpoint=str(Path(args.checkpoint).resolve()),environment=args.environment,index=args.index,
        prefix_tokens=args.prefix_tokens)
    baseline=None;runs=[]
    for workers in args.replicas:
        if workers not in (1,2,4):raise ValueError('explicit bounded B1 replica count')
        spool=module.NonceSpool(output/f'replicas-{workers}',
            dict(opening_sha256=hashlib.sha256(canonical(opening)).hexdigest(),miner=args.miner,
                 environment=args.environment,index=args.index,prefix_tokens=args.prefix_tokens,workers=workers),n)
        started=time.monotonic()
        try:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                     initializer=initialize,initargs=(config,))as pool:
                pids=set()
                while len(pids)<workers:
                    pids.update(x.result()['pid']for x in [pool.submit(ready,.25)for _ in range(workers*2)])
                load_seconds=time.monotonic()-started;started=time.monotonic()
                def encode(nonce,result):
                    if result['nonce']!=nonce or result['diagnostic_prefix_only']is not True:
                        raise ValueError('original prescribed diagnostic attempt')
                    return canonical(result)
                raw=module.collect(spool,lambda nonce:pool.submit(generate,nonce),encode,workers=workers)
                elapsed=time.monotonic()-started
            rows=[json.loads(v)for v in raw]
        finally:spool.close()
        relevant=lambda row:{k:row[k]for k in ('nonce','prompt_sha256','task_hash','output','behavior_logprobs','raw_t1_logprobs','proofs')}
        reference=[relevant(row)for row in rows]
        if baseline is None:baseline=reference
        equal=reference==baseline
        summary=dict(replicas=workers,processes=len({r['pid']for r in rows}),group_size=n,
            prefix_tokens=args.prefix_tokens,total_tokens=sum(len(r['output'])for r in rows),
            elapsed_seconds=elapsed,load_seconds=load_seconds,exact_serial_prefix_parity=equal,
            per_rollout_seconds=[r['elapsed_seconds']for r in rows],
            max_process_peak_allocated_bytes=max(r['peak_allocated_bytes']for r in rows))
        runs.append(summary);(output/'PROGRESS.json').write_bytes(canonical(dict(runs=runs)))
        print(json.dumps(summary),flush=True)
        if not equal:raise ValueError('parallel prefix changed original B1 tokens, probabilities, or proofs')
    result=dict(version='mimo-private-b1-replica-prefix-benchmark-v1',runs=runs,
        speedup=runs[0]['elapsed_seconds']/runs[-1]['elapsed_seconds'],
        original_opening_sha256=hashlib.sha256(canonical(opening)).hexdigest(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        collector_sha256=hashlib.sha256(parallel.read_bytes()).hexdigest(),
        full_rollout_qualified=False,public_activation=False,batch_dimension=1)
    (output/'RESULT.json').write_bytes(canonical(result));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','release','opening','authority','miner','checkpoint','environment','output','parallel-module'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--prefix-tokens',type=int,default=64)
    parser.add_argument('--replicas',type=int,nargs='+',default=[1,2])
    run(parser.parse_args())
