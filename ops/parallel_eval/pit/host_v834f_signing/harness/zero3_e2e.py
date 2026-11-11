#!/usr/bin/env python3
"""Rank 0 of a CPU run of the host launcher, in the order train_pit.py uses on the box:

    host.checked(ROOT-ADMITTED) -> zero3.activate(host) -> host.install(cfg, start_dp=True) -> actor ->
    CPUAdamW -> v824_batched.train_update_batched -> stage_for_publication_model_only -> stop

modes
  prepare    build the stand-in closure, the tiny head checkpoint and its runtime profile (shared by the other modes)
  sharded    host-v834f (stage/) with PIT_ZERO3: the real install(start_dp=True) starts the helper ranks through the
             v835_zero3 entry over gloo and one sharded update runs
  reference  the admitted host-v834e files, one process, the original v8211_batched.train_update_batched
             (padded engine) with the host's optimizer step: the unsharded update on the same head and batch
  plain      host-v834f on a GPU that does not need sharding: activate() and install(start_dp=False) only, to
             show that the launcher then installs the v8.3.4e update (v83_dp) and leaves the token budget alone
  fp32-load  the admitted host-v834e files in a fresh process: build the optimizer on the candidate another run
             published (--source) and load the fp32 master that run saved with host-v834e's own _fp32_load
  bf16-master  production precision, which the launcher runs above cannot have (subnet.mimo_runtime loads fp32 on a
             CPU, so there the master equals the weights). One process, host-v834f launcher, no DP world: the
             tiny model is cast to bf16 and its head master gets a residual below half a bf16 step. The launcher's
             own optimizer step runs on CPU bf16 weights, as the sharded update runs it on rank 0, with the fp32
             gradient that zero3_numerics.py gathered from FSDP ranks in bf16 (<work>/sharded-gradient.pt). The
             saved candidate master is then loaded with the admitted host-v834e _fp32_load.

--observe (sharded mode, rank 0) puts pass-through instruments on the staged code and adds what they saw to the
result. Every instrumented function runs unchanged; see observe(). It answers three questions about memory that
only show on a real GPU as an out-of-memory error: does a full-size original Parameter outlive the fully_shard call
that replaced it, does backward keep the recomputed logits of the head chunks, and does the zero-weight probe step
run the backward of the real loss. --logit-chunk lowers rank 0's head chunk so that one micro-batch has several.

Every run has its own fp32 master store, <work>/<run>/fp32-master, so an entry there was written by that run.
The launcher splits a master into files of 4 GiB; the tiny master is 142 kB, so the harness lowers that one
constant (FP32_SHARD_BYTES) to make the same code write several files.

train_pit.py itself is not run here: it needs the box paths, a signed dispatch and CUDA. See fixture.py for what is
a stand-in. The result is written to <work>/<mode>/result.json.
"""
import argparse
import collections
import dataclasses
import gc
import inspect
import json
import os
import shutil
import sys
import time
import types
import warnings
import weakref
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fixture  # noqa: E402  (sibling module; this file runs as a script)

TEST_SEED = bytes(range(32))      # throwaway signing key of the fixture config; never a real authority seed
FIXTURE_SHARD_BYTES = 48 * 1024   # stands in for the launcher's 4 GiB FP32_SHARD_BYTES (see the module docstring)
ADMITTED_MODES = ('reference', 'fp32-load')


def plan_probe(dp):
    """The installed planner (the function the update calls) on four 14,000-token rollouts for two ranks.
    Two of them are 28,000 tokens: one bin under the signed 40,960 budget, two bins under the 24,576 cut."""
    rows = [dict(group=types.SimpleNamespace(prompt=range(2000)), rollout=types.SimpleNamespace(tokens=range(12000)))
            for _ in range(4)]
    assign, bins, _ = dp._plan(rows, list(range(4)), 2)
    return dict(bins=bins, largest_bin_tokens=max(14000 * len(batch) for batches in assign for batch in batches))


NODE = 'autograd::engine::evaluate_function: '         # the profiler's name for one autograd node that backward ran


def _sizes(items):
    return [[len(prompt), len(output)] for prompt, output in items]


def _recorded(seen, kind, call, items_of):
    """-> call, which also adds the autograd nodes that ran during it to seen['nodes']."""
    import torch      # after install(), as everywhere in this file

    def run(*args, **kwargs):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            out = call(*args, **kwargs)
        nodes = collections.Counter(e.name[len(NODE):] for e in profile.events() if e.name.startswith(NODE))
        seen['nodes'].append(dict(kind=seen['tag'] or kind, items=items_of(*args), nodes=dict(nodes)))
        return out
    return run


def observe(z3, B):
    """Wrap four staged functions of rank 0 so that each call is recorded and then runs unchanged. -> the record.

    shard   after every real fully_shard(unit): how many of the unit's original trainable Parameters are alive.
            FSDP has replaced each of them by this rank's shard, so a live one is a full-size tensor that some
            reference still pins.
    head    when backward reaches the decoder output (Actor._head_done; every head chunk is done by then): how
            many fp32 [tokens, vocab] tensors are alive. They are recomputed logits that the head checkpoint holds.
    nodes   the autograd nodes that the backward of each zero-weight probe step and of each real step ran.
    """
    # torch imports only now: the closure is on sys.path and install() has patched it (train_pit order)
    import torch
    from torch.distributed.tensor import DTensor
    seen = dict(shard=[], head=[], nodes=[], tag=None)
    real_shard = z3.fully_shard

    def fully_shard(module, **kwargs):
        originals = [weakref.ref(p) for p in module.parameters() if p.requires_grad and not isinstance(p, DTensor)]
        out = real_shard(module, **kwargs)
        gc.collect()
        seen['shard'].append(dict(unit=type(module).__name__, originals=len(originals),
                                  alive=sum(ref() is not None for ref in originals)))
        return out
    z3.fully_shard = fully_shard

    real_done = z3.Actor._head_done

    def _head_done(self, grad):
        gc.collect()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')        # the scan touches deprecated module attributes
            logits = sum(1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.dtype == torch.float32
                         and obj.dim() == 2 and obj.shape[1] == self.inv['vocab'])
        seen['head'].append(dict(phase=seen['tag'] or ('step' if self.accumulate else 'probe'), logits_alive=logits))
        return real_done(self, grad)
    z3.Actor._head_done = _head_done

    z3.Actor.null_step = _recorded(seen, 'probe', z3.Actor.null_step, lambda actor, module, items: _sizes(items))
    B._gradient_pass = _recorded(seen, 'step', B._gradient_pass, lambda runtime, rows, plan, *rest: _sizes(
        (rows[i]['group'].prompt, rows[i]['rollout'].tokens) for batch in plan for i in batch))
    return seen


def head_control(z3, B, stock, seen, items, support):
    """The same micro-batch and the same loss on log_q twice on the sharded actor: with the forward of the stock
    v8211_batched file (log_p keeps its graph) and with the staged sharded forward. Then the staged zero-weight
    probe step on the stock forward. -> what the runs kept alive in backward and which nodes they ran (recorded
    by observe() under control-stock, control-staged and control-probe), and whether the values and the gradients
    of the first two are the same bits."""
    import torch      # after install(), as everywhere in this file
    actor = z3.WORLD['actor']

    def run(tag, forward):
        def step():
            with torch.enable_grad():
                outs = forward()
            weights = [torch.linspace(0.5, 1.5, len(log_q)) for _, log_q, _ in outs]
            torch.stack([(w * log_q).sum() for w, (_, log_q, _) in zip(weights, outs)]).sum().backward()
            actor.end_step()
            return outs
        actor.clear()
        seen['tag'] = tag
        try:
            outs = _recorded(seen, tag, step, lambda: _sizes(items))()
        finally:
            seen['tag'] = None
        gradients = {name: actor.shard_grad(name).clone() for name, _ in actor.names}
        values = [tuple(t.detach().clone() for t in out) for out in outs]
        attached = [bool(out[0].requires_grad) for out in outs]
        actor.clear()
        return gradients, values, attached

    def stock_forward():        # what the sharded forward did before the head-memory fix: begin the step, stock outputs
        actor.begin_step()
        return stock(actor.runtime, items, gradients=True, engine='padded', logit_chunk=z3.LOGIT_CHUNK, support=support)
    want_g, want_v, want_attached = run('control-stock', stock_forward)
    got_g, got_v, got_attached = run('control-staged', lambda: B.batched_log_probs(
        actor.runtime, items, gradients=True, engine='packed', logit_chunk=z3.LOGIT_CHUNK, support=support))
    # The probe step must differentiate what the loss differentiates, whatever the forward hands back: here a
    # log_p that keeps its graph. Its state is the one _probe() sets.
    attached = types.SimpleNamespace(batched_log_probs=lambda *args, **kwargs: stock_forward(), SUPPORT_RENORM=support)
    seen['tag'], actor.accumulate = 'control-probe', False
    try:
        actor.null_step(attached, items)
    finally:
        seen['tag'], actor.accumulate = None, True
    actor.clear()
    return dict(
        items=_sizes(items), logit_chunk=z3.LOGIT_CHUNK,
        head_chunks=-(-sum(len(output) for _, output in items) // z3.LOGIT_CHUNK),
        stock_log_p_attached=want_attached, staged_log_p_attached=got_attached,
        values_equal=all(torch.equal(a, b) for want, got in zip(want_v, got_v) for a, b in zip(want, got)),
        gradient_tensors=len(want_g), gradient_nonzero=sum(bool(g.abs().sum() > 0) for g in want_g.values()),
        gradients_equal=list(want_g) == list(got_g) and all(torch.equal(want_g[n], got_g[n]) for n in want_g))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--mode', required=True, choices=('prepare', 'sharded', 'reference', 'plain', 'fp32-load', 'bf16-master'))
    ap.add_argument('--source', default='sharded', help='fp32-load: the run whose candidate and fp32 master are loaded')
    ap.add_argument('--world', type=int, default=2)
    ap.add_argument('--port', type=int, default=29583)
    ap.add_argument('--tag', default=None, help='result directory name (default: the mode)')
    ap.add_argument('--kill-helper', action='store_true', help='kill the helper rank before the update (watchdog check)')
    ap.add_argument('--guard-fail', type=int, default=0, help='make the N-th step-boundary guard call raise once, then retry')
    ap.add_argument('--fake-peak', default=None, choices=('first-over', 'always-over'),
                    help='there is no GPU here: stand in for the GPU memory rank 0 measures in the probe step. '
                         'first-over = over the limit until its layer inputs are on the host; always-over = always')
    ap.add_argument('--observe', action='store_true', help='sharded: record what rank 0 keeps alive (see observe())')
    ap.add_argument('--logit-chunk', type=int, default=None, help='sharded: head tokens per lm_head call on rank 0')
    a = ap.parse_args(argv)
    if a.observe and (a.mode != 'sharded' or a.world != 1):
        ap.error('--observe runs its control forward on rank 0 alone: it needs --mode sharded --world 1')
    work = Path(a.work).resolve()
    src = fixture.source_root(work / 'src')
    sharded = a.mode == 'sharded'
    admitted = a.mode in ADMITTED_MODES
    hostdir = fixture.host_dir(work / ('host-v834e' if admitted else 'host-v834f'),
                               fixture.ADMITTED if admitted else fixture.STAGE)
    os.chdir(src)                                           # the launcher requires cwd == the admitted closure root
    sys.path[:0] = [str(src), str(hostdir)]
    host = fixture.load_module('mimo_host_trainer_v83', hostdir / 'mimo_host_trainer_v83.py')
    host.TRITON_CACHE = str(work / 'triton-cache')
    host.FP32_SHARD_BYTES = FIXTURE_SHARD_BYTES
    head = work / 'head'
    if a.mode == 'prepare':
        # production profiles pin the declared training kernels to None; allow() makes the fingerprint do the same
        fixture.load_module('v83_kernels_policy', hostdir / 'v83_kernels_policy.py').allow()
        descriptor, spec = fixture.tiny_checkpoint(head)
        (work / 'head.json').write_text(json.dumps(dict(descriptor=descriptor, spec=spec)))
        print(json.dumps(dict(event='prepared', checkpoint=descriptor['id'])), flush=True)
        return
    out = work / (a.tag or (a.mode if a.mode != 'fp32-load' else 'fp32-load-' + a.source))
    out.mkdir(parents=True)
    store = out / 'fp32-master'
    host.FP32_STORE = str(store)
    meta = json.loads((work / 'head.json').read_text())
    descriptor, spec = meta['descriptor'], meta['spec']
    engine = dict(fixture.ENGINE_BLOCK, dp_world=a.world, dp_port=a.port)
    cfg = host.checked(fixture.signed_config(host, hostdir, src, engine, TEST_SEED))
    result = dict(mode=a.mode, host_files=sorted(host.HOST_FILES), launcher=str(Path(host.__file__).resolve()))
    if a.mode in ('fp32-load', 'bf16-master'):
        result.update(fp32_load(host, cfg, work, out, spec, a.source) if a.mode == 'fp32-load' else
                      bf16_master(host, cfg, work, descriptor, spec))
        (out / 'result.json').write_text(json.dumps(result, indent=1, default=str))
        print(json.dumps(dict(event='done', mode=a.mode)), flush=True)
        return
    if a.mode in ('sharded', 'plain'):
        z3 = host.zero3()
        if sharded:
            try:
                z3.activate(None)
                result['activate_none'] = 'accepted'
            except RuntimeError as x:
                result['activate_none'] = 'refused: %s' % x
        result['policy'] = z3.activate(host)
        if a.logit_chunk:
            z3.LOGIT_CHUNK = a.logit_chunk
    info = host.install(cfg, start_dp=sharded)
    seen = observe(z3, sys.modules['v824_batched']) if a.observe else None
    result['install'] = dict(dp_world=info['dp_world'], zero3=info.get('zero3'), dp_token_budget=info.get('dp_token_budget'))
    dp = sys.modules['v83_dp']
    helpers = dp.STATE.get('helpers') or []
    result['helper_argv'] = [list(map(str, h.args)) for h in helpers]
    result['forward'] = sys.modules['v824_batched'].batched_log_probs.__qualname__
    result['zero3_env'] = os.environ.get('PIT_ZERO3')
    result['plan_probe'] = plan_probe(dp)
    if a.mode == 'plain':
        (out / 'result.json').write_text(json.dumps(result, indent=1, default=str))
        print(json.dumps(dict(event='done', mode=a.mode)), flush=True)
        return

    # subnet.* and torch import only now: the closure is on sys.path and install() has patched it (train_pit order)
    import torch
    from subnet.mimo_grpo.config import TrainerConfig
    from subnet.mimo_grpo.data import InputBinding
    from subnet.mimo_grpo.optimizer import CPUAdamW, FP32Accumulator
    from subnet.mimo_runtime import MimoRuntime
    result['fp32_accumulator_signature'] = str(inspect.signature(FP32Accumulator.__init__))
    torch.set_num_threads(2)
    runtime = MimoRuntime.load(str(head), descriptor, spec, device='cpu', trainable=True, allow_fixture=True)
    config = TrainerConfig(**fixture.trainer_settings())
    step = 7
    opt = CPUAdamW(runtime.model.named_parameters(), config, input_checkpoint=descriptor['id'])
    opt.global_step = step
    # the head's fp32 master in the local store and its digest in the environment, as train_pit requires
    bf, _ = host._digest_map(list(opt.parameters))
    dig, per = host._digest_map([(n, opt.masters[n]) for n, _ in opt.parameters])
    events = []
    host._fp32_save(opt, bf, dict(version=host.FP32_VERSION, bf16_sha256=bf, master_digest=dig, tensor_sha256=per), (bf,), events.append)
    os.environ['AFFINE_FP32_EXPECTED_MASTER_HASH'] = dig
    ff = sys.modules['ff_trainer']
    groups = fixture.free_groups(ff)
    binding = InputBinding('fixture-run', 'fixture-epoch', descriptor['id'], '0' * 64, step)
    B = sys.modules['v824_batched']
    e = cfg['engine']
    records = []
    calls = dict(n=0, failed=False)
    stores_seen = set()

    def guard():
        calls['n'] += 1
        stores_seen.add(tuple(sorted(os.listdir(store))))      # what a poller of the store sees while gradients are computed
        if a.guard_fail and not calls['failed'] and calls['n'] == a.guard_fail:
            calls['failed'] = True
            raise RuntimeError('fixture guard: lease lost')

    if a.kill_helper:
        helpers[0].kill()
        helpers[0].wait()
    if a.fake_peak:
        measured = z3._measured_peak

        def fake_peak():
            on_host = z3.WORLD['actor'].offload_inputs
            high = a.fake_peak == 'always-over' or not on_host
            return dict(allocated_gib=29.0 if high else 12.0, reserved_gib=30.0 if high else 13.0)
        z3._measured_peak = fake_peak
    t0 = time.monotonic()
    if sharded:
        kwargs = dict(config=config, reference_runtime=None, engine=e['batched_engine'], token_budget=info['dp_token_budget'],
                      max_rows=e['max_rows'], guard=guard, record=records.append)
        try:
            report = B.train_update_batched(runtime, opt, groups, binding, **kwargs)
        except RuntimeError as x:
            if not (a.guard_fail or a.fake_peak == 'always-over'):
                raise
            result['first_error'] = '%s: %s' % (type(x).__name__, x)
            result['helper_alive_after_error'] = [h.poll() is None for h in helpers]
            if a.fake_peak:
                z3._measured_peak = measured          # the same world, now without the stand-in measurement
            report = B.train_update_batched(runtime, opt, groups, binding, **kwargs)
    else:   # the unsharded update: KL forced off as the host does, original engine, the host's CPUAdamW.step
        cfg0 = dataclasses.replace(config, kl_coefficient=0., kl_reference_checkpoint='').validate()
        opt.config = cfg0
        # install() put the CUDA-only packed forward on v824_batched; the padded forward is the same file's own function
        B.batched_log_probs = fixture.load_module('v8211_stock', hostdir / 'v8211_batched.py').batched_log_probs
        report = B.train_update_batched._original(runtime, opt, groups, binding, config=cfg0, reference_runtime=None,
                                                  engine='padded', token_budget=e['token_budget'], max_rows=e['max_rows'],
                                                  guard=guard, record=records.append)
    result['update_seconds'] = time.monotonic() - t0
    off = sys.modules['v8210_optimizer_offpath']
    staged = off.stage_for_publication_model_only(runtime, opt, report, str(out / 'staged'), workers=e['stage_workers'])
    staged = staged[0] if isinstance(staged, tuple) else staged
    (out / 'descriptor.json').write_text(json.dumps(dict(id=staged['output_checkpoint'], files=staged['checkpoint_files'])))
    host._PUBLISHED.set()
    with opt._v833f_lock:       # as train_pit: the background fp32 master save has finished when this returns
        pass
    fp = report['optimizer']['sgd']['fp32_master']
    result.update(
        status=report['status'], optimizer_step=report['optimizer_step'], objective_loss=report['objective_loss'],
        preclip_gradient_norm=report['preclip_gradient_norm'], trained_rollouts=report['trained_rollouts'],
        trained_tokens=report['trained_tokens'], microbatch=report['microbatch'], train_params=report.get('train_params'),
        losses={str(t['task_id'])[:8] + ':' + str(t['nonce']): t['loss'] for t in report['trajectories']},
        raw_update_norm=report['optimizer']['sgd']['raw_update_norm'],
        fp32=dict(source=fp['source'], input_master_digest=fp['input_master_digest'], expected=dig,
                  output_master_digest=fp.get('output_master_digest'), input_bf16_sha256=fp.get('input_bf16_sha256'),
                  output_bf16_sha256=fp.get('output_bf16_sha256'), output_store_entry=fp.get('output_store_entry'),
                  store=fp.get('store'), finish_error=fp.get('finish_error')),
        store_during_training=sorted(map(list, stores_seen)),
        store_after={d.name: sorted(f.name for f in d.iterdir()) for d in sorted(store.iterdir())},
        checkpoint=staged['output_checkpoint'], checkpoint_files=sorted(staged['checkpoint_files']),
        model_dir=str(out / 'staged' / 'model'), actor_device=str(runtime.device), guard_calls=calls['n'],
        record_stages=sorted({r.get('stage') for r in records}),
        trajectory_loss_records=sum(r.get('stage') == 'trajectory-loss' for r in records),
        masked_records=sum(r.get('stage') == 'masked-unresolved' for r in records))
    if a.observe:
        stock = fixture.load_module('v8211_stock', hostdir / 'v8211_batched.py').batched_log_probs
        items = [(g.prompt, t.tokens) for g in groups for t in g.trajectories][:3]
        control = head_control(z3, B, stock, seen, items, B.SUPPORT_RENORM)
        result['observed'] = dict(shard=seen['shard'], head=seen['head'], nodes=seen['nodes'], control=control,
                                  logit_chunk=z3.LOGIT_CHUNK, vocab=z3.WORLD['actor'].inv['vocab'])
    if sharded:
        dp._bcast(dict(op='stop'))
        result['helper_returncodes'] = [h.wait(timeout=60) for h in helpers]
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(dict(event='done', mode=a.mode, loss=result['objective_loss'], norm=result['preclip_gradient_norm'])), flush=True)


def fp32_load(host, cfg, work, out, spec, source):
    """The admitted host-v834e launcher loads, with its own _fp32_load, the fp32 master that run `source` saved,
    into a fresh optimizer built on the candidate checkpoint `source` published. Then two refusals on real files."""
    run = work / source
    host.FP32_STORE = str(run / 'fp32-master')
    host.install(cfg, start_dp=False)
    # subnet.* and torch import only now: the closure is on sys.path and install() has patched it (train_pit order)
    import torch
    from subnet.mimo_grpo.config import TrainerConfig
    from subnet.mimo_grpo.optimizer import CPUAdamW
    from subnet.mimo_runtime import MimoRuntime
    config = TrainerConfig(**fixture.trainer_settings())

    def optimizer(directory, descriptor):
        runtime = MimoRuntime.load(str(directory), descriptor, spec, device='cpu', trainable=True, allow_fixture=True)
        opt = CPUAdamW(runtime.model.named_parameters(), config, input_checkpoint=descriptor['id'])
        with torch.no_grad():
            for master in opt.masters.values():
                master.zero_()                      # whatever the masters hold after the load came from the files
        return opt

    def load(opt, entry):
        events = []
        try:
            meta = host._fp32_load(opt, list(opt.parameters), entry, events.append)
            return dict(accepted=True, meta=meta, events=events)
        except host.Fp32MasterError as x:
            return dict(accepted=False, error=str(x), events=events)

    candidate = json.loads((run / 'descriptor.json').read_text())
    opt = optimizer(run / 'staged' / 'model', candidate)
    entry, _ = host._digest_map(list(opt.parameters))
    loaded = load(opt, entry)
    digest, per = host._digest_map([(n, opt.masters[n]) for n, _ in opt.parameters])
    result = dict(
        source=source, store=host.FP32_STORE, candidate=candidate['id'], entry=entry, loaded=loaded,
        master_digest_after_load=digest, master_tensor_sha256=per,
        master_tensors=len(opt.masters), master_dtypes=sorted({str(m.dtype) for m in opt.masters.values()}),
        master_nonzero=all(bool(m.detach().abs().sum() > 0) for m in opt.masters.values()),
        master_projects_to_candidate=all(bool(torch.equal(opt.masters[n].detach().to(p.dtype), p.detach()))
                                         for n, p in opt.parameters))
    # refusal 1: the same entry with one byte of one weight file changed
    host.FP32_STORE = str(out / 'damaged-store')
    shutil.copytree(run / 'fp32-master' / entry, out / 'damaged-store' / entry)
    target = sorted((out / 'damaged-store' / entry).glob('*.safetensors'))[-1]
    raw = bytearray(target.read_bytes())
    raw[-1] ^= 0x01
    target.write_bytes(bytes(raw))
    result['damaged_file'] = load(optimizer(run / 'staged' / 'model', candidate), entry)
    # refusal 2: the intact candidate master offered for the HEAD weights (it does not project onto them)
    host.FP32_STORE = str(run / 'fp32-master')
    head = json.loads((work / 'head.json').read_text())['descriptor']
    result['wrong_weights'] = load(optimizer(work / 'head', head), entry)
    return result


def bf16_master(host, cfg, work, descriptor, spec):
    """See the module docstring (mode bf16-master)."""
    host.install(cfg, start_dp=False)
    # subnet.* and torch import only now: the closure is on sys.path and install() has patched it (train_pit order)
    import torch
    from subnet.mimo_grpo.config import TrainerConfig
    from subnet.mimo_grpo.optimizer import CPUAdamW
    from subnet.mimo_runtime import MimoRuntime
    runtime = MimoRuntime.load(str(work / 'head'), descriptor, spec, device='cpu', trainable=True, allow_fixture=True)
    runtime.model.to(torch.bfloat16)                 # the fixture profile is fp32; production weights are bf16
    runtime._expected_dtype = torch.bfloat16
    config = TrainerConfig(**fixture.trainer_settings())
    opt = CPUAdamW(runtime.model.named_parameters(), config, input_checkpoint=descriptor['id'])
    opt.global_step = 7
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for name, parameter in opt.parameters:       # |residual| <= |w| / 1024: below half a bf16 step of w
            master = opt.masters[name]
            master.add_(master.abs() * (torch.rand(master.shape, generator=generator) - 0.5) * 2.0 ** -9)
            if not torch.equal(master.to(torch.bfloat16), parameter):
                raise AssertionError('fixture head master must project onto the bf16 head: ' + name)
    head_bf16, _ = host._digest_map(list(opt.parameters))
    head_digest, per = host._digest_map([(n, opt.masters[n]) for n, _ in opt.parameters])
    head_residual = sum(int((opt.masters[n].detach() != p.detach().float()).sum()) for n, p in opt.parameters)
    events = []
    host._fp32_save(opt, head_bf16, dict(version=host.FP32_VERSION, bf16_sha256=head_bf16, master_digest=head_digest,
                                         tensor_sha256=per), (head_bf16,), events.append)
    os.environ['AFFINE_FP32_EXPECTED_MASTER_HASH'] = head_digest
    with torch.no_grad():                            # memory holds the plain upcast again: the step must use the store
        for name, parameter in opt.parameters:
            opt.masters[name].copy_(parameter.detach().float())
    gradients = torch.load(str(work / 'sharded-gradient.pt'))
    update = opt.step(gradients)                     # the launcher's step: verified master from disk, fp32 update, bf16 projection
    host._PUBLISHED.set()
    with opt._v833f_lock:
        pass
    fp = update['sgd']['fp32_master']
    # the admitted host-v834e loader, on a fresh optimizer over the candidate weights (bf16, CPU)
    admitted = fixture.load_module('admitted_host_v834e_launcher', fixture.ADMITTED / 'mimo_host_trainer_v83.py')
    admitted.FP32_STORE = host.FP32_STORE
    fresh = CPUAdamW(runtime.model.named_parameters(), config, input_checkpoint=descriptor['id'])
    with torch.no_grad():
        for master in fresh.masters.values():
            master.zero_()
    rejected = []
    entry, _ = admitted._digest_map(list(fresh.parameters))
    meta = admitted._fp32_load(fresh, list(fresh.parameters), entry, rejected.append)
    digest, _ = admitted._digest_map([(n, fresh.masters[n]) for n, _ in fresh.parameters])
    store = Path(host.FP32_STORE)
    return dict(
        weight_dtypes=sorted({str(p.dtype) for _, p in opt.parameters}), actor_device=str(runtime.device),
        gradient_tensors=len(gradients), head_entry=head_bf16, head_master_digest=head_digest,
        head_residual_elements=head_residual, source=fp['source'], input_master_digest=fp['input_master_digest'],
        output_master_digest=fp['output_master_digest'], output_store_entry=fp['output_store_entry'],
        finish_error=fp.get('finish_error'), projected_changed_elements=update['projected_changed_elements'],
        store_after={d.name: sorted(f.name for f in d.iterdir()) for d in sorted(store.iterdir())},
        admitted_launcher=str(Path(admitted.__file__).resolve()), entry=entry, meta_keys=sorted(meta),
        loaded_master_digest=digest, meta_master_digest=meta['master_digest'], rejected=rejected,
        master_dtypes=sorted({str(m.dtype) for m in fresh.masters.values()}),
        projects_to_candidate=all(bool(torch.equal(fresh.masters[n].detach().to(p.dtype), p.detach())) for n, p in fresh.parameters),
        equals_memory_master=all(bool(torch.equal(fresh.masters[n], opt.masters[n])) for n, _ in opt.parameters),
        # elements where the loaded master is not the plain fp32 value of the bf16 weight: the residual a bf16 file cannot hold
        residual_elements=sum(int((fresh.masters[n].detach() != p.detach().float()).sum()) for n, p in fresh.parameters),
        elements=sum(p.numel() for _, p in fresh.parameters))


if __name__ == '__main__':
    main()
    sys.stdout.flush()
    os._exit(0)      # as train_pit: do not wait on the launcher's non-daemon threads
