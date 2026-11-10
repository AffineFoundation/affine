#!/usr/bin/env python3
"""Numerics of the sharded actor (stage/v835_zero3.Actor) on CPU over gloo, against the v8.3.4e definition.

W processes each load the same tiny production-class model in the requested precision, shard it with torch FSDP
exactly as the trainer does, receive rank 0's weights, and run their own micro-batches in lockstep (ranks with
fewer micro-batches run zero-weight steps). Rank 0 gathers the fp32 gradient. Three checks:

  weights    every rank's all-gathered weights are byte-identical to rank 0's unsharded weights.
  reduction  v8.3.4e adds each micro-batch's autograd gradient (parameter dtype) into an fp32 accumulator on its
             rank and SUM-reduces the accumulators. The same autograd gradients are captured here inside each
             rank (leaf hooks on the unsharded parameters), summed that way, and compared with the gathered one.
             This is exact arithmetic on the same numbers, so it does not depend on the kernels.
  unsharded  the gathered gradient against one process running every micro-batch on the unsharded model.

bfloat16 is the production precision (the fixture profile is fp32, so the loaded model is cast). The CPU bf16
kernels of these torch builds are not reproducible between processes that share cores: several processes
computing the same UNSHARDED bf16 gradient at the same time disagree at the 1e-3..1e-2 level, with fp32 they
agree bit for bit. `unsharded` is therefore required only for float32 and reported for bfloat16. Every process
uses one thread.

Prints one JSON line. Exit code 0 only when the required checks hold.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fixture  # noqa: E402  (sibling module; this file runs as a script)

LOGIT_CHUNK = 2          # several lm_head calls per micro-batch, a different number on each rank
REL_TOL = 1e-6           # fp32 summation order only; a bf16 rounding of the reduced gradient would be near 4e-3


def batches(world, seed):
    """rank -> list of micro-batches, each a list of (prompt, output) token tuples. Rank r gets 1 + r % 3 of them."""
    g = torch.Generator().manual_seed(seed)
    tokens = lambda n: tuple(int(x) for x in torch.randint(4, 60, (n,), generator=g))
    out = {}
    for rank in range(world):
        out[rank] = [[(tokens(int(torch.randint(2, 6, (1,), generator=g))), tokens(int(torch.randint(2, 12, (1,), generator=g))))
                      for _ in range(1 + (rank + step) % 3)] for step in range(1 + rank % 3)]
    return out


def runtime_of(head, descriptor, spec, dtype):
    from subnet.mimo_runtime import MimoRuntime
    runtime = MimoRuntime.load(head, descriptor, spec, device='cpu', trainable=True, allow_fixture=True)
    if dtype != 'float32':
        runtime.model.to(getattr(torch, dtype))
        runtime._expected_dtype = getattr(torch, dtype)
    return runtime


def enter(src, hostdir):
    os.chdir(src)
    sys.path[:0] = [str(src), str(hostdir)]
    torch.set_num_threads(1)
    # as the trainer and its helper ranks do on the box: the declared training kernels read as the pinned None
    fixture.load_module('v83_kernels_policy', Path(hostdir) / 'v83_kernels_policy.py').allow()


def worker(rank, world, port, work, dtype, seed, offload, queue):
    try:
        src, hostdir = Path(work) / 'src', Path(work) / 'host-v834f'
        enter(src, hostdir)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group('gloo', rank=rank, world_size=world)
        ctrl = dist.new_group(backend='gloo')
        zero3 = fixture.load_module('v835_zero3', hostdir / 'v835_zero3.py')
        B = fixture.load_module('v824_batched', hostdir / 'v8211_batched.py')
        zero3.LOGIT_CHUNK = LOGIT_CHUNK
        dp = type('DP', (), dict(STATE=dict(world=world, ctrl=ctrl)))
        zero3.make_train_update(dp, B, None)                 # installs the sharded forward on v824_batched
        meta = json.loads((Path(work) / 'head.json').read_text())
        head = str(Path(work) / 'head')
        runtime = runtime_of(head, meta['descriptor'], meta['spec'], dtype)
        actor = zero3.Actor(runtime, world, rank)
        dist.barrier()
        zero3.WORLD['actor'] = actor
        # count what the host-memory hooks receive (the hooks themselves are the staged ones)
        packed, pack = [], zero3._SavedInputsOnHost._pack
        zero3._SavedInputsOnHost._pack = staticmethod(lambda tensor: (packed.append(tuple(tensor.shape)), pack(tensor))[1])
        actor.offload_inputs = offload
        canonical = dict(runtime_of(head, meta['descriptor'], meta['spec'], dtype).model.named_parameters()) if rank == 0 else None
        actor.sync(ctrl, canonical)
        # unshard one unit at a time: hash the gathered weights and hook the unsharded leaves (this rank's autograd gradients)
        local, digest, hooked = {}, hashlib.sha256(), []

        def capture(name):
            def hook(grad):
                grad = grad.detach().to(torch.float32)
                local[name] = local[name] + grad if name in local else grad.clone()
            return hook
        for unit in actor.layers + [actor.embed, actor.head, actor.root]:
            unit.unshard()
            for name, param in actor.model.named_parameters():
                if param.requires_grad and type(param).__name__ != 'DTensor' and name not in hooked:
                    hooked.append(name)
                    # hash a copy: exporting the FSDP buffer itself to numpy would pin its storage against resharding
                    digest.update(name.encode() + param.detach().clone().view(torch.uint8).numpy().tobytes())
                    param.register_hook(capture(name))
            unit.reshard()
        want = hashlib.sha256()
        if rank == 0:
            for name in hooked:
                want.update(name.encode() + canonical[name].detach().clone().view(torch.uint8).numpy().tobytes())
        versions = B._versions(actor.model)
        plan = batches(world, seed)
        mine, steps, losses = plan[rank], max(len(v) for v in plan.values()), []
        for step in range(steps):
            if B._versions(actor.model) != versions:
                raise AssertionError('sharded parameters changed identity or version between steps')
            if step < len(mine):
                with torch.enable_grad():
                    outs = B.batched_log_probs(runtime, mine[step], gradients=True, engine='packed', logit_chunk=LOGIT_CHUNK, support=None)
                loss = -torch.stack([o.sum() for o in outs]).sum()
                losses.append(float(loss.detach()))
                loss.backward()
                actor.end_step()
            else:
                actor.null_step(B, [plan[0][0][0]])
        params = dict(actor.model.named_parameters())
        torch.save(local, str(Path(work) / ('local-gradient-%d.pt' % rank)))
        info = dict(rank=rank, ok=True, losses=losses, null_steps=steps - len(mine), steps=steps,
                    layers=len(actor.layers), packed_inputs=sum(1 for shape in packed if len(shape) == 3),
                    hooked=len(hooked), weights=digest.hexdigest(), canonical_weights=want.hexdigest() if rank == 0 else None,
                    names=[n for n, _ in actor.names],
                    sharded=all(type(params[n]).__name__ == 'DTensor' for n, _ in actor.names),
                    shard_bytes=sum(actor.shard_bytes().values()),
                    full_bytes=sum(torch.Size(shape).numel() * (4 if n in actor.native else actor.embed.weight.element_size())
                                   for n, shape in actor.names),
                    native_dtypes=sorted({str(params[n].dtype) for n in actor.native}),
                    explicit_dtypes=sorted({str(params[n].dtype) for rows in actor.explicit.values() for n, _ in rows}))
        buffers = {n: torch.zeros(shape, dtype=torch.float32) for n, shape in actor.names} if rank == 0 else None
        actor.gather(ctrl, buffers)
        if rank == 0:
            torch.save(buffers, str(Path(work) / 'sharded-gradient.pt'))
        queue.put(info)
        dist.barrier()
        dist.destroy_process_group()
    except Exception:
        queue.put(dict(rank=rank, ok=False, error=traceback.format_exc()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', type=int, default=2)
    ap.add_argument('--dtype', default='bfloat16', choices=('bfloat16', 'float32'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--port', type=int, default=29591)
    ap.add_argument('--work', default=None)
    ap.add_argument('--offload-inputs', action='store_true',
                    help='every rank keeps its checkpointed layer inputs behind the host-memory hooks')
    a = ap.parse_args(argv)
    work = Path(a.work or tempfile.mkdtemp(prefix='zero3-numerics-')).resolve()
    src = fixture.source_root(work / 'src')
    hostdir = fixture.host_dir(work / 'host-v834f', fixture.STAGE)
    enter(src, hostdir)
    if not (work / 'head.json').exists():
        descriptor, spec = fixture.tiny_checkpoint(work / 'head')
        (work / 'head.json').write_text(json.dumps(dict(descriptor=descriptor, spec=spec)))
    meta = json.loads((work / 'head.json').read_text())
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, a.world, a.port, str(work), a.dtype, a.seed, a.offload_inputs, queue))
             for r in range(a.world)]
    for p in procs:
        p.start()
    outs = sorted((queue.get(timeout=600) for _ in procs), key=lambda o: o['rank'])
    for p in procs:
        p.join(60)
    failed = [o for o in outs if not o['ok']]
    if failed:
        print(json.dumps(dict(ok=False, world=a.world, dtype=a.dtype, error=failed[0]['error'][-1500:])))
        return 2
    # the v8.3.4e definition on the unsharded model: fp32 accumulator += each micro-batch's gradient
    B = fixture.load_module('v824_batched_reference', hostdir / 'v8211_batched.py')
    runtime = runtime_of(str(work / 'head'), meta['descriptor'], meta['spec'], a.dtype)
    reference = {n: torch.zeros(p.shape, dtype=torch.float32) for n, p in runtime.model.named_parameters() if p.requires_grad}
    losses = {}
    for rank, plan in batches(a.world, a.seed).items():
        for items in plan:
            for p in runtime.model.parameters():
                p.grad = None
            with torch.enable_grad():
                o = B.batched_log_probs(runtime, items, gradients=True, engine='padded', logit_chunk=LOGIT_CHUNK, support=None)
            loss = -torch.stack([x.sum() for x in o]).sum()
            losses.setdefault(rank, []).append(float(loss.detach()))
            loss.backward()
            for n, p in runtime.model.named_parameters():
                if p.requires_grad:
                    reference[n] += p.grad
    sharded = torch.load(str(work / 'sharded-gradient.pt'))
    summed = {}                 # the v8.3.4e accumulator: SUM over ranks of each rank's fp32 sum of its own gradients
    for rank in range(a.world):
        for n, g in torch.load(str(work / ('local-gradient-%d.pt' % rank))).items():
            summed[n] = summed[n] + g if n in summed else g.clone()

    def compare(target):
        num = torch.sqrt(sum((sharded[n] - target[n]).double().pow(2).sum() for n in target))
        den = torch.sqrt(sum(target[n].double().pow(2).sum() for n in target))
        return dict(rel_diff=float(num / den), max_abs_diff=max(float((sharded[n] - target[n]).abs().max()) for n in target),
                    exact_tensors=sum(bool(torch.equal(sharded[n], target[n])) for n in target), gradient_norm=float(den))
    ref_max = max(float(reference[n].abs().max()) for n in reference)
    reduction, unsharded = compare(summed), compare(reference)
    result = dict(
        world=a.world, dtype=a.dtype, seed=a.seed, tensors=len(reference),
        names_equal=list(sharded) == list(reference) and all(o['names'] == list(reference) for o in outs)
        and set(summed) == set(reference) and all(o['hooked'] == len(reference) for o in outs),
        all_sharded=all(o['sharded'] for o in outs),
        weights_exact=all(o['weights'] == outs[0]['canonical_weights'] for o in outs),
        losses_equal=all(o['losses'] == losses[o['rank']] for o in outs),
        null_steps=[o['null_steps'] for o in outs], shard_bytes=[o['shard_bytes'] for o in outs],
        offload_inputs=a.offload_inputs, steps=outs[0]['steps'], layers=outs[0]['layers'],
        packed_inputs=[o['packed_inputs'] for o in outs], gradient_file=str(work / 'sharded-gradient.pt'),
        full_bytes=outs[0]['full_bytes'],
        native_dtypes=outs[0]['native_dtypes'], explicit_dtypes=outs[0]['explicit_dtypes'],
        reduction=reduction, unsharded=unsharded, reference_max_abs=ref_max,
        # the two fields ops.parallel_eval.pit.train_shard reports: the required comparison of this precision
        max_abs_diff=(unsharded if a.dtype == 'float32' else reduction)['max_abs_diff'],
        rel_tol=REL_TOL, tol=1e-5 * max(1.0, ref_max))
    required = [reduction] + ([unsharded] if a.dtype == 'float32' else [])
    result['ok'] = bool(result['names_equal'] and result['all_sharded'] and result['weights_exact']
                        and (result['losses_equal'] or a.dtype != 'float32')
                        and all(c['rel_diff'] <= REL_TOL and c['max_abs_diff'] <= result['tol'] for c in required))
    print(json.dumps(result))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
