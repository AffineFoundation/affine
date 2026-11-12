#!/usr/bin/env python3
"""Numerics of the sharded actor (stage/v835_zero3.Actor) on CPU over gloo, against the v8.3.4e definition.

W processes each load the same tiny production-class model in the requested precision, shard it with torch FSDP
exactly as the trainer does, receive rank 0's weights, and run their own micro-batches in lockstep (ranks with
fewer micro-batches run zero-weight steps). Rank 0 gathers the fp32 gradient. Three checks:

  weights    every rank's all-gathered weights are byte-identical to rank 0's unsharded weights.
  reduction  The autograd gradients of every step are captured inside each rank (leaf hooks on the unsharded
             parameters) and summed here as the settings of the run define the sum. The result is compared with
             the gathered gradient. This is exact arithmetic on the same numbers, so it does not depend on the
             kernels. Two definitions:
               fp32    v8.3.4e: each micro-batch's gradient (parameter dtype) goes into an fp32 accumulator on
                       its rank and the accumulators are SUM-reduced. lm_head and the embedding always; the
                       decoder layers when the settings reduce them in float32.
               native  the gradients of the ranks of one step are added in the parameter dtype (bf16 in
                       production), and the steps are added in fp32. The decoder layers of the default settings.
             `fp32_reduction` is the distance of the gathered gradient from the fp32 definition for every tensor.
             `explicit_step_reduction` compares lm_head and the embedding with the fp32 sum taken in the order of
             the code (the ranks of one step, then the steps): with two ranks that is the same bits.
  unsharded  the gathered gradient against one process running every micro-batch on the unsharded model.

--profile names the settings (v835_zero3.PROFILES); without it the run has the default ones. The result also says
what the run did with them: the dtypes of the reduce-scatter inputs, the group that carried the gradient shards to
rank 0, how many all-gathers were issued before each decoder layer ran, and the host blocks of the layer inputs.
With two ranks the bf16 sum of a layer gradient has one value, so the comparison with the `native` definition is
exact. With more ranks the backend adds in its own order: then lm_head and the embedding keep the exact check and
the layers are only required to stay within LAYER_BF16_TOL of the fp32 definition (`layers_near_fp32`).

bfloat16 is the production precision (the fixture profile is fp32, so the loaded model is cast). The CPU bf16
kernels of these torch builds are not reproducible between processes that share cores: several processes
computing the same UNSHARDED bf16 gradient at the same time disagree at the 1e-3..1e-2 level, with fp32 they
agree bit for bit. `unsharded` is therefore required only for float32 and reported for bfloat16. Every process
uses one thread.

Prints one JSON line. Exit code 0 only when the required checks hold.
"""
import argparse
import collections
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
# Layer gradients added over the ranks in bf16, against the same sum in fp32. Each addition rounds its result to
# 8 bits (at most 2**-9 of it), so the distance grows with the number of ranks. This is a limit for this fixture
# and not a proof: the runs here gave 1.1e-3 with 2 ranks and 3.1e-3 with 8.
LAYER_BF16_TOL = 2e-2


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


def count_collectives(ctrl):
    """Pass-through counters on the collectives that FSDP and the staged code issue in this process."""
    seen = dict(all_gather=0, reduce=collections.Counter(), gather=collections.Counter())
    all_gather, reduce_scatter, gather = dist.all_gather_single, dist.reduce_scatter_single, dist.gather

    def all_gather_single(*args, **kwargs):
        seen['all_gather'] += 1
        return all_gather(*args, **kwargs)

    def reduce_scatter_single(*args, **kwargs):
        seen['reduce'][str((kwargs['input'] if 'input' in kwargs else args[1]).dtype)] += 1
        return reduce_scatter(*args, **kwargs)

    def gather_to_rank_0(tensor, *args, **kwargs):
        seen['gather']['control' if kwargs.get('group') is ctrl else 'data'] += 1
        return gather(tensor, *args, **kwargs)
    dist.all_gather_single, dist.reduce_scatter_single, dist.gather = all_gather_single, reduce_scatter_single, gather_to_rank_0
    return seen


def worker(rank, world, port, work, dtype, seed, offload, profile, block_bytes, queue):
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
        if profile:
            os.environ[zero3.PROFILE_ENV] = profile
        if block_bytes:      # a layer input of the tiny model is far below one block of the real size
            zero3.HOST_BLOCK_BYTES = block_bytes
        seen = count_collectives(ctrl)
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
        zero3._SavedInputsOnHost._pack = lambda hooks, tensor: (packed.append(tuple(tensor.shape)), pack(hooks, tensor))[1]
        actor.offload_inputs = offload
        # all-gathers issued since the step began, at the moment each decoder layer runs its forward
        mark, before_layer = dict(all_gather=0), []

        def watched(inner):
            def forward(*args, **kwargs):
                before_layer[-1].append(seen['all_gather'] - mark['all_gather'])
                return inner(*args, **kwargs)
            return forward
        for layer in actor.layers:
            layer.forward = watched(layer.forward)
        canonical = dict(runtime_of(head, meta['descriptor'], meta['spec'], dtype).model.named_parameters()) if rank == 0 else None
        actor.sync(ctrl, canonical)
        # unshard one unit at a time: hash the gathered weights and hook the unsharded leaves (this rank's autograd gradients)
        local, by_step, digest, hooked = {}, [], hashlib.sha256(), []

        def capture(name):
            def hook(grad):
                grad = grad.detach()
                by_step[-1][name] = by_step[-1][name] + grad if name in by_step[-1] else grad.clone()
                grad = grad.to(torch.float32)
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
        mine, steps, losses, blocks_used = plan[rank], max(len(v) for v in plan.values()), [], []
        for step in range(steps):
            if B._versions(actor.model) != versions:
                raise AssertionError('sharded parameters changed identity or version between steps')
            by_step.append({})
            before_layer.append([])
            mark['all_gather'] = seen['all_gather']
            if step < len(mine):
                with torch.enable_grad():
                    outs = B.batched_log_probs(runtime, mine[step], gradients=True, engine='packed', logit_chunk=LOGIT_CHUNK, support=None)
                loss = -torch.stack([o.sum() for o in outs]).sum()
                losses.append(float(loss.detach()))
                loss.backward()
                actor.end_step()
            else:
                actor.null_step(B, [plan[0][0][0]])
            if actor.host_blocks is not None:       # the blocks this step filled; the next step starts at block 0 again
                blocks_used.append(actor.host_blocks.used)
        params = dict(actor.model.named_parameters())
        torch.save(dict(total=local, steps=by_step), str(Path(work) / ('local-gradient-%d.pt' % rank)))
        info = dict(rank=rank, ok=True, losses=losses, null_steps=steps - len(mine), steps=steps,
                    layers=len(actor.layers), packed_inputs=sum(1 for shape in packed if len(shape) == 3),
                    profile=actor.settings._asdict(), reduce_inputs=dict(seen['reduce']),
                    gathers_before_layer=before_layer[0], native=list(actor.native),
                    host_blocks=None if actor.host_blocks is None else len(actor.host_blocks.blocks),
                    host_blocks_used=None if actor.host_blocks is None else blocks_used,
                    hooked=len(hooked), weights=digest.hexdigest(), canonical_weights=want.hexdigest() if rank == 0 else None,
                    names=[n for n, _ in actor.names],
                    sharded=all(type(params[n]).__name__ == 'DTensor' for n, _ in actor.names),
                    shard_bytes=sum(actor.shard_bytes().values()),
                    full_bytes=sum(torch.Size(shape).numel() * (4 if n in actor.native else actor.embed.weight.element_size())
                                   for n, shape in actor.names),
                    native_dtypes=sorted({str(params[n].dtype) for n in actor.native}),
                    explicit_dtypes=sorted({str(params[n].dtype) for rows in actor.explicit.values() for n, _ in rows}))
        buffers = {n: torch.zeros(shape, dtype=torch.float32) for n, shape in actor.names} if rank == 0 else None
        seen['gather'].clear()
        actor.gather(ctrl, buffers)
        if rank == 0:
            torch.save(buffers, str(Path(work) / 'sharded-gradient.pt'))
        info['gather_groups'] = dict(seen['gather'])
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
    ap.add_argument('--profile', default=None, help='settings of the sharded update (v835_zero3.PROFILES); default: its default')
    ap.add_argument('--host-block-bytes', type=int, default=None,
                    help='size of one host block of the layer inputs (v835_zero3.HOST_BLOCK_BYTES)')
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
    procs = [ctx.Process(target=worker, args=(r, a.world, a.port, str(work), a.dtype, a.seed, a.offload_inputs,
                                              a.profile, a.host_block_bytes, queue))
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
    local = [torch.load(str(work / ('local-gradient-%d.pt' % rank))) for rank in range(a.world)]
    summed = {}                 # the v8.3.4e accumulator: SUM over ranks of each rank's fp32 sum of its own gradients
    for rank in range(a.world):
        for n, g in local[rank]['total'].items():
            summed[n] = summed[n] + g if n in summed else g.clone()
    profile, native = outs[0]['profile'], set(outs[0]['native'])

    def step_sum(n, widen):
        """The ranks of one step added in rank order, then the steps added in fp32. `widen`: the ranks are added
        in fp32; else in the dtype autograd gave the gradient."""
        total = None
        for step in range(outs[0]['steps']):
            parts = [local[rank]['steps'][step][n] for rank in range(a.world) if n in local[rank]['steps'][step]]
            if parts:
                across = parts[0].to(torch.float32) if widen else parts[0].clone()
                for part in parts[1:]:
                    across = across + (part.to(torch.float32) if widen else part)
                total = across.to(torch.float32) if total is None else total + across.to(torch.float32)
        return total
    # the sum the settings of this run define: `summed` for a tensor reduced in fp32; for a tensor reduced as
    # autograd gave it, the ranks of one step added in that dtype and the steps added in fp32
    defined = dict(summed)
    if profile['layer_reduce'] == 'native':
        defined.update({n: step_sum(n, widen=False) for n in native})
    # lm_head and the embedding in the order of the explicit collective: one fp32 SUM over the ranks in every step
    explicit_steps = {n: step_sum(n, widen=True) for n in summed if n not in native}

    def compare(target, names=None):
        names = [n for n in target if names is None or n in names]
        num = torch.sqrt(sum((sharded[n] - target[n]).double().pow(2).sum() for n in names))
        den = torch.sqrt(sum(target[n].double().pow(2).sum() for n in names))
        got = torch.sqrt(sum(sharded[n].double().pow(2).sum() for n in names))
        dot = sum((sharded[n].double() * target[n].double()).sum() for n in names)
        return dict(rel_diff=float(num / den), max_abs_diff=max(float((sharded[n] - target[n]).abs().max()) for n in names),
                    exact_tensors=sum(bool(torch.equal(sharded[n], target[n])) for n in names), tensors=len(names),
                    gradient_norm=float(den), cosine=float(dot / (den * got)))
    ref_max = max(float(reference[n].abs().max()) for n in reference)
    reduction, unsharded = compare(defined), compare(reference)
    # Layers added over the ranks in bf16: two ranks give one correctly rounded sum, so the gathered gradient must
    # be the defined sum bit for bit. More ranks are added in an order that belongs to the backend. Then lm_head
    # and the embedding keep the exact check, and the layers must stay near the fp32 definition.
    in_bf16 = profile['layer_reduce'] == 'native' and a.dtype != 'float32'
    one_order = not in_bf16 or a.world <= 2
    result = dict(
        world=a.world, dtype=a.dtype, seed=a.seed, tensors=len(reference), profile=profile,
        profiles_equal=all(o['profile'] == profile for o in outs),
        reduce_inputs=outs[0]['reduce_inputs'], reduce_inputs_equal=all(o['reduce_inputs'] == outs[0]['reduce_inputs'] for o in outs),
        gather_groups=[o['gather_groups'] for o in outs], gathers_before_layer=outs[0]['gathers_before_layer'],
        host_blocks=[o['host_blocks'] for o in outs], host_blocks_used=[o['host_blocks_used'] for o in outs],
        host_block_bytes=a.host_block_bytes, layers_in_bf16=in_bf16, one_order=one_order,
        layer_reduction=compare(defined, native), explicit_reduction=compare(summed, set(summed) - native),
        explicit_step_reduction=compare(explicit_steps),
        fp32_reduction=compare(summed), layer_fp32_reduction=compare(summed, native),
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
        rel_tol=REL_TOL, layer_bf16_tol=LAYER_BF16_TOL, tol=1e-5 * max(1.0, ref_max))
    required = [reduction if one_order else result['explicit_reduction']] + ([unsharded] if a.dtype == 'float32' else [])
    result['layers_near_fp32'] = not in_bf16 or result['layer_fp32_reduction']['rel_diff'] <= LAYER_BF16_TOL
    result['ok'] = bool(result['names_equal'] and result['all_sharded'] and result['weights_exact']
                        and result['profiles_equal'] and result['reduce_inputs_equal'] and result['layers_near_fp32']
                        and (result['losses_equal'] or a.dtype != 'float32')
                        and all(c['rel_diff'] <= REL_TOL and c['max_abs_diff'] <= result['tol'] for c in required))
    print(json.dumps(result))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
