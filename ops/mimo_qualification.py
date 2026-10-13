#!/usr/bin/env python3
"""Prepare and execute isolated MiMo runtime diagnostics, without signing them.

No downloader, fleet access, production state or authority credentials are used.
The resulting report is evidence for qualification, never an approval itself.
Run from a staged source directory with a local immutable HF checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subnet.mimo_runtime import MimoRuntime, canonical, digest, file_hash, make_runtime_spec


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value))


def prepare(args):
    from transformers import AutoTokenizer
    root = Path(args.checkpoint).resolve()
    files = {p.name:file_hash(p) for p in root.iterdir() if p.is_file()}
    if len(files) != len(list(root.iterdir())):
        raise ValueError("checkpoint must contain only ordinary flat model assets")
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    spec = make_runtime_spec(chat_template=tokenizer.chat_template,
        enable_thinking=not args.disable_thinking,max_context_tokens=args.max_context_tokens,
        attention=args.attention)
    write_new(args.output,{"checkpoint":{"id":digest(files),"files":files},
        "mimo_runtime":spec.to_dict(),"upstream_revision":args.revision,
        "status":"unqualified-runtime-candidate"})


def run(args):
    import torch
    pin=json.loads(Path(args.pin).read_text())
    timings={}
    began=time.monotonic()
    runtime=MimoRuntime.load(Path(args.checkpoint).resolve(),pin["checkpoint"],pin["mimo_runtime"],
        device="cuda",trainable=args.backward)
    timings["load_seconds"]=time.monotonic()-began
    try:
        prompt=runtime.prepare_prompt([{"role":"user","content":args.prompt}])
        generated=[];sample_lp=[];cache=None;cached_intervals=[]
        started=time.monotonic()
        # Diagnostic draws have their own domain and do not mint protocol batches.
        from subnet.complete_group_distribution import distribution, pick
        for position in range(args.tokens):
            logits,cache=runtime.next_token_logits(prompt if position == 0 else [generated[-1]],cache)
            probs=distribution(logits,temperature=1.0,top_p=.95,top_k=20)
            raw=hashlib.sha256(canonical(["mimo-runtime-diagnostic-v1",args.seed,position])).digest()
            draw=(int.from_bytes(raw[:8],"big") >> 11)/2**53
            token=pick(probs,draw)
            if args.cdf:
                high=float(probs.cumsum(-1)[token]);mass=float(probs[token])
                cached_intervals.append(dict(position=position,token=token,draw=draw,
                    low=high-mass,high=high,mass=mass))
            generated.append(token)
            sample_lp.append(float(torch.log_softmax(logits.float(),-1)[token]))
            if token in runtime.eos_token_ids:break
        torch.cuda.synchronize()
        timings["generation_seconds"]=time.monotonic()-started
        del cache
        started=time.monotonic()
        selected=runtime.score(prompt,generated,gradients=args.backward)
        torch.cuda.synchronize()
        timings["prefill_seconds"]=time.monotonic()-started
        differences=(selected.detach().cpu()-torch.tensor(sample_lp)).abs()
        cdf_evidence=None
        if args.cdf:
            started=time.monotonic()
            _,raw_log_probs=runtime.compute(prompt,generated)
            prefills=distribution(torch.from_numpy(raw_log_probs),1.,.95,20)
            intervals=[]
            for row,probs in zip(cached_intervals,prefills):
                high=float(probs.cumsum(-1)[row['token']]);mass=float(probs[row['token']]);low=high-mass
                intervals.append(dict(**row,prefill_low=low,prefill_high=high,prefill_mass=mass,
                    prefill_accepts=mass>0 and low<=row['draw']<high,
                    low_difference=abs(low-row['low']),high_difference=abs(high-row['high'])))
            cdf_evidence=dict(scope='observed-generated-token-intervals-not-global-error-bound',
                intervals=intervals,max_endpoint_difference=max(max(r['low_difference'],r['high_difference'])for r in intervals),
                exact_prefill_reject_count=sum(not r['prefill_accepts']for r in intervals),
                support_mismatch_count=sum(r['prefill_mass']==0 for r in intervals))
            timings['cdf_diagnostic_seconds']=time.monotonic()-started
        backward=None
        if args.backward:
            started=time.monotonic()
            (-selected.mean()).backward()
            torch.cuda.synchronize()
            timings["backward_seconds"]=time.monotonic()-started
            missing=[name for name,p in runtime.model.named_parameters() if p.requires_grad and p.grad is None]
            finite=all(bool(torch.isfinite(p.grad).all()) for p in runtime.model.parameters() if p.grad is not None)
            backward={"finite_gradients":finite,"missing_trainable_gradients":missing,
                "trainable_parameters":sum(p.numel() for p in runtime.model.parameters() if p.requires_grad),
                "frozen_parameters":sum(p.numel() for p in runtime.model.parameters() if not p.requires_grad)}
            if missing or not finite:raise ValueError("invalid actual MiMo backward gradients")
        report={"version":"mimo-runtime-diagnostic-v1","qualification":False,
            "checkpoint":pin["checkpoint"]["id"],"runtime_profile":runtime.spec.to_dict(),
            "prompt":prompt,"output":generated,"output_text":runtime.tokenizer.decode(generated),
            "stop":"eos" if generated[-1] in runtime.eos_token_ids else "budget_cap",
            "cached_raw_log_probs":sample_lp,"prefill_raw_log_probs":selected.detach().cpu().tolist(),
            "max_logprob_difference":float(differences.max()),"mean_logprob_difference":float(differences.mean()),
            "timings":timings,"backward":backward,"cdf_evidence":cdf_evidence,
            "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "gpu":torch.cuda.get_device_name(),"compute_capability":list(torch.cuda.get_device_capability())}
        write_new(args.output,report)
        print(json.dumps({"report":args.output,"generated_tokens":len(generated),"timings":timings,
            "max_logprob_difference":report["max_logprob_difference"],"qualification":False}))
    finally:runtime.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare");prep.add_argument("--checkpoint",required=True)
    prep.add_argument("--revision",required=True);prep.add_argument("--output",required=True)
    prep.add_argument("--disable-thinking",action="store_true")
    prep.add_argument("--max-context-tokens",type=int,default=12288)
    prep.add_argument("--attention",choices=["eager","sdpa"],default="eager")
    run_parser=sub.add_parser("run");run_parser.add_argument("--checkpoint",required=True)
    run_parser.add_argument("--pin",required=True);run_parser.add_argument("--output",required=True)
    run_parser.add_argument("--prompt",default="Solve 17 * 23. Explain briefly and put the final answer in \\boxed{}.")
    run_parser.add_argument("--tokens",type=int,default=64);run_parser.add_argument("--seed",type=int,default=0)
    run_parser.add_argument("--backward",action="store_true")
    run_parser.add_argument("--cdf",action="store_true",help="Record cached/prefill sampled-token CDF intervals")
    args=parser.parse_args()
    if args.command=="run" and not 1<=args.tokens<=8192:parser.error("--tokens must be between 1 and 8192")
    (prepare if args.command=="prepare" else run)(args)


if __name__=="__main__":main()
