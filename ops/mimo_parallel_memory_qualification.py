#!/usr/bin/env python3
"""Private memory/parity control; synthetic long tokens are never rollouts.

Load the exact selected runtime source with five resident models. Recompute one
original completed rollout in each independent B1 replica; then exercise the
8192-token proof tensor geometry. No sampling, submission, reward or training.
"""
from pathlib import Path
import argparse, gc, hashlib, importlib.util, json, multiprocessing, os, sys, time
from concurrent.futures import ProcessPoolExecutor

_CONTEXT=None
_PEER_CACHE=None


def load(config):
    import torch
    from subnet.mimo_runtime import MimoRuntime,validate_runtime_spec
    from subnet.complete_group_codec import decode,limits
    manifest=json.loads(Path(config['job']).read_bytes())['payload']['manifest']['payload']
    raw=Path(config['original_nonce']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=config['original_nonce_sha256']:raise ValueError('original nonce digest')
    row,arrays=decode(raw,limits(manifest))[0]
    rollout=row['rollout'];turn=rollout['turns'][0]
    model=MimoRuntime.load(config['checkpoint'],{k:manifest['checkpoint'][k]for k in ('id','files')},
        validate_runtime_spec(manifest['mimo_runtime']),device='cuda:0',trainable=False)
    spec=importlib.util.spec_from_file_location('subnet.parallel_memory_candidate',config['adapter'])
    candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
    return model,turn,arrays[0][0],candidate._release_cuda_temporaries


def initialize(config,gate,barrier):
    global _CONTEXT,_PEER_CACHE
    _CONTEXT=(*load(config),gate)
    count=config.get('max_peer_cache_tokens',0)
    if type(count)is not int or count not in (0,8192):raise ValueError('explicit peer cache control')
    if count:
        model,turn,_,release,_=_CONTEXT
        original=turn['output'];tokens=(original*((count+len(original)-1)//len(original)))[:count]
        # Conservatively keep all four maximum-length caches while one proof
        # runs. Production has at most three other decoding peers. These are
        # synthetic geometry inputs, never claimed to be sampled rollouts.
        with gate:
            logits,_PEER_CACHE=model.next_token_logits(list(turn['prompt'])+tokens,use_cache=True)
            del logits
            release()
    barrier.wait(timeout=480)


def run(mode):
    import numpy as np
    import torch
    from toploc import build_proofs_base64
    model,turn,arrays,release,gate=_CONTEXT
    prompt=list(turn['prompt']);original=list(turn['output'])
    output=original if mode=='original' else (original*((8192+len(original)-1)//len(original)))[:8192]
    began=time.monotonic()
    with gate:
        release();torch.cuda.reset_peak_memory_stats()
        acts,full=model.compute(prompt,output)
        selected=full[np.arange(len(output)),output].astype(np.float32).reshape(-1,1)
        if mode=='original':
            proofs=build_proofs_base64(acts,decode_batching_size=16,topk=128)
            if not np.array_equal(selected,arrays):raise ValueError('original selected logprobs changed')
            if proofs!=turn['proofs']:raise ValueError('original TOPLOC proof changed')
        peak=torch.cuda.max_memory_allocated();del acts,full,selected
        release()
        return dict(mode=mode,pid=os.getpid(),retained_peer_cache=_PEER_CACHE is not None,
            original_tokens=len(original),tokens=len(output),
            exact_selected_logprobs_and_toploc=mode=='original',synthetic_geometry_only=mode!='original',
            allocated_after=torch.cuda.memory_allocated(),reserved_after=torch.cuda.memory_reserved(),
            peak_allocated=peak,seconds=time.monotonic()-began)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=json.loads(Path(args.config).read_bytes())
    sys.path.insert(0,config['source']);out=Path(config['output']);out.mkdir(mode=0o700,exist_ok=True)
    if (out/'RESULT.json').exists():raise ValueError('original qualification result already exists')
    began=time.monotonic();parent=load(config);mp=multiprocessing.get_context('spawn')
    gate=mp.Semaphore(1);barrier=mp.Barrier(4);results=[]
    pool=ProcessPoolExecutor(4,mp_context=mp,initializer=initialize,initargs=(config,gate,barrier))
    try:
        for mode in ('original','synthetic8192'):
            futures=[pool.submit(run,mode)for _ in range(4)]
            values=[f.result(timeout=600)for f in futures];results.extend(values)
            (out/(mode+'.json')).write_text(json.dumps(values,sort_keys=True,indent=2))
    finally:
        from subnet.mimo_parallel_mining import stop_process_pool
        stop_process_pool(pool)
        parent[0].close();del parent;gc.collect()
    result=dict(version='private-five-resident-models-proof-memory-v1',elapsed_seconds=time.monotonic()-began,
        actual_model=True,source=config['source'],adapter_sha256=hashlib.sha256(Path(config['adapter']).read_bytes()).hexdigest(),
        original_nonce_sha256=config['original_nonce_sha256'],results=results,
        generated_sampling_rollouts=False,submitted_batches=False,training_inputs_created=False,
        long_rollout_sampling_qualified=False)
    (out/'RESULT.json').write_text(json.dumps(result,sort_keys=True,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
