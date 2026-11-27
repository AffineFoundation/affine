#!/usr/bin/env python3
"""Speed bench (test pod only, synthetic data). Reuses qual_step model/loss; times phases per step.
Variants: v834c | v835 | v835gd (prototype: GPU-direct async reduce in post-accumulate hook, no CPU staging)
Gather modes (v835*): gpu (as shipped: full FP32 dict on rank0 GPU) | cpu (stream each tensor to rank0 pinned CPU)
"""
import argparse, json, os, sys, time
os.environ.setdefault("TRITON_CACHE_DIR", "/root/.triton/cache-r" + os.environ.get("LOCAL_RANK", "0"))
import torch, torch.distributed as dist
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import qual_step as Q

def sync():
    torch.cuda.synchronize()

class GD:
    """prototype: owner FP32 shard on GPU; non-owned grads cast to fp32 on GPU and reduced to owner from the hook."""
    def __init__(self, params, rank, world, device, max_inflight=4):
        from v835_sharded_accum import ownership_map
        self.parameters = tuple((n, p) for n, p in params if p.requires_grad)
        self.owners = ownership_map(self.parameters, world); self.rank = rank; self.device = device
        self.buffers = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in self.parameters if self.owners[n] == rank}
        self.pending = []; self.max_inflight = max_inflight; self.hooks = []; self.stream = torch.cuda.Stream()
    def install_hooks(self):
        for n, p in self.parameters:
            def mk(n):
                def hook(param):
                    g = param.grad
                    owner = self.owners[n]
                    cur = torch.cuda.current_stream()
                    self.stream.wait_stream(cur)
                    with torch.cuda.stream(self.stream):
                        t = g.float()
                        if owner == self.rank:
                            w = dist.reduce(t, dst=owner, op=dist.ReduceOp.SUM, async_op=True)
                            self.pending.append((w, t, n, g))
                        else:
                            w = dist.reduce(t, dst=owner, op=dist.ReduceOp.SUM, async_op=True)
                            self.pending.append((w, t, None, g))
                    param.grad = None
                    while len(self.pending) > self.max_inflight:
                        self._drain_one()
                return hook
            self.hooks.append(p.register_post_accumulate_grad_hook(mk(n)))
    def _drain_one(self):
        w, t, n, g = self.pending.pop(0); w.wait()
        if n is not None:
            with torch.cuda.stream(self.stream):
                self.buffers[n].add_(t)
    def finish(self):
        while self.pending: self._drain_one()
        torch.cuda.current_stream().wait_stream(self.stream)
    def remove(self):
        for h in self.hooks: h.remove()

def gather(acc_buffers, owners, params, rank, device, mode):
    full = {}
    for n, p in params:
        o = owners[n]
        t = acc_buffers[n] if rank == o else torch.empty(p.shape, dtype=torch.float32, device=device)
        dist.broadcast(t, src=o) if mode == "bcast" else None
        if mode == "gpu":
            dist.broadcast(t, src=o)
            if rank == 0: full[n] = t
        elif mode == "cpu":
            # point-to-point to rank0 only, then D2H into pinned host (host optimizer lives on CPU)
            if o != 0:
                if rank == o: dist.send(t, dst=0)
                elif rank == 0: dist.recv(t, src=o)
            if rank == 0:
                h = torch.empty(p.shape, dtype=torch.float32, pin_memory=True); h.copy_(t); full[n] = h
        if rank != 0 and rank != o: del t
    return full if rank == 0 else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True); ap.add_argument("--output-len", type=int, required=True)
    ap.add_argument("--prompt-len", type=int, default=4096); ap.add_argument("--gather", default="gpu")
    ap.add_argument("--offload", action="store_true", help="shortcut for --activation-offload on")
    ap.add_argument("--activation-offload", type=str, default=None,
                    help="0/1/auto (default env ACTIVATION_OFFLOAD or auto)")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--logit-chunk", type=int, default=4096); ap.add_argument("--json-out", required=True)
    ap.add_argument("--parse-only", action="store_true", help="Parse args and exit (no GPU)")
    a = ap.parse_args()
    if a.parse_only:
        print(json.dumps(dict(parse_only=True, args=vars(a)), sort_keys=True)); return
    dist.init_process_group("nccl"); local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local)
    dev = torch.device("cuda", local); rank = dist.get_rank(); world = dist.get_world_size()
    model = Q.build_model("random://qwen3_5_9b", dev); Q.install_checkpoint(model)
    for p in model.parameters(): dist.broadcast(p.data, src=0)
    FK = Q._load("v83_fast_kernels", "v83_fast_kernels.py"); FK.install(model)
    dp = Q._load("v835_dp", "v835_dp.py")
    offload_val = "on" if a.offload else a.activation_offload
    dp.configure_activation_offload(offload_val)
    if dp.ACTIVATION_OFFLOAD_MODE != "off":
        class R: pass
        rt = R(); rt.model = model; rt.trainable = True; dp._install_activation_offload(rt)
    prompt, output = Q.synthetic_batch(a.prompt_len, a.output_len, model.config.vocab_size)
    support = dict(temperature=1.0, top_k=20, top_p=0.95)
    named = list(model.named_parameters())
    steps = []
    for s in range(a.steps):
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache(); dist.barrier(); sync()
        T = {}; t0 = time.monotonic(); err = None
        try:
            for p in model.parameters(): p.grad = None
            if a.variant == "v834c":
                from subnet.mimo_grpo.optimizer import FP32Accumulator
                acc = FP32Accumulator(model.named_parameters())
                loss = Q.packed_selected_loss(model, FK, [(prompt, output)], a.logit_chunk, support); sync(); T["fwd"] = time.monotonic() - t0
                t = time.monotonic(); loss.backward(); acc.capture(); sync(); T["bwd"] = time.monotonic() - t
                t = time.monotonic()
                for n in sorted(acc.buffers): dist.reduce(acc.buffers[n], dst=0, op=dist.ReduceOp.SUM)
                sync(); T["reduce"] = time.monotonic() - t; full = acc.buffers if rank == 0 else None; T["gather"] = 0.0
                params = acc.parameters
            else:
                if a.variant == "v835":
                    from v835_sharded_accum import ShardedFP32Accumulator
                    acc = ShardedFP32Accumulator(named, rank=rank, world=world, device=dev); acc.install_hooks()
                else:
                    acc = GD(named, rank, world, dev); acc.install_hooks()
                loss = Q.packed_selected_loss(model, FK, [(prompt, output)], a.logit_chunk, support); sync(); T["fwd"] = time.monotonic() - t0
                t = time.monotonic(); loss.backward()
                if a.variant == "v835": acc.capture()
                sync(); T["bwd"] = time.monotonic() - t
                t = time.monotonic()
                if a.variant == "v835": acc.reduce_shards()
                else: acc.finish(); acc.remove()
                sync(); T["reduce"] = time.monotonic() - t
                t = time.monotonic(); full = gather(acc.buffers, acc.owners, acc.parameters, rank, dev, a.gather); sync(); T["gather"] = time.monotonic() - t
                params = acc.parameters
            t = time.monotonic()
            if rank == 0:
                norms = [torch.linalg.vector_norm(g, dtype=torch.float32) for g in full.values()]
                tot = torch.linalg.vector_norm(torch.stack([x.to(norms[0].device) for x in norms])); sc = torch.clamp(1.0 / (tot + 1e-6), max=1.)
                for g in full.values(): g.mul_(sc.to(g.device))
            sync(); T["clip"] = time.monotonic() - t
            t = time.monotonic()
            if rank == 0:  # host optimizer boundary: grads to CPU (as optimizer.step does), not AdamW math
                for n, p in params:
                    _ = full[n] if full[n].device.type == "cpu" else full[n].to("cpu")
            sync(); T["to_host"] = time.monotonic() - t
            if a.variant == "v835": acc.clear()
            del full
            T["norm"] = float(tot) if rank == 0 else None
        except torch.cuda.OutOfMemoryError as e:
            err = "OOM: " + str(e).split("\n")[0][:200]
        except Exception as e:
            import traceback; err = f"{type(e).__name__}: {e}"[:400]; traceback.print_exc()
        sync(); T["total"] = time.monotonic() - t0
        T["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30; T["peak_res_gib"] = torch.cuda.max_memory_reserved() / 2**30
        T["err"] = err
        g = [None] * world; dist.all_gather_object(g, T); steps.append(g)
        if rank == 0: print(json.dumps(dict(step=s, r0=g[0], max_total=max(x["total"] for x in g), max_peak=max(x["peak_alloc_gib"] for x in g))), flush=True)
        if any(x["err"] for x in g): break
    if rank == 0:
        ctx = a.prompt_len + a.output_len
        json.dump(dict(vars(a), ctx=ctx, world=world, tokens_per_step=ctx * world, steps=steps), open(a.json_out, "w"), indent=1)
    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__": main()
