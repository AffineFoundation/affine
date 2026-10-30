"""v8.3 data-parallel packed engine for the v8.2.11 batched GRPO update (same loss code, same optimizer).

Semantics are those of v8211_batched.train_update_batched (same layout, loss aggregation, KL reference, derail mask,
rewards/advantages, cap policy, loss_scale, clip, ONE CPUAdamW step on the worker = rank 0). Only the execution differs:
  * rows are packed into varlen micro-batches (first-fit-decreasing, token budget) and spread over W GPUs
    (LPT by tokens); every rank runs the SAME v8211_batched._reference_pass/_gradient_pass on its own micro-batches
    with the 'packed' batched_log_probs replaced by fast_kernels.packed_log_probs (fla chunk_gated_delta_rule /
    causal_conv1d varlen, torch varlen flash attention; installed per layer instance, training forwards only);
  * the FP32 gradient accumulators are SUM-reduced to rank 0 (fp32), per-rollout details/losses gathered;
  * helpers (ranks 1..W-1) hold the frozen KL reference and a bit-exact copy of the actor (rank-0 broadcast each job).
Failure before the optimizer step on any rank -> every rank joins the ok-flag reduction and the update raises
(transaction fails exactly as a single-GPU failure would; the in-memory actor/optimizer are untouched).
"""
import math, os, pickle, sys, time, json
import torch, torch.distributed as dist
from datetime import timedelta

# v8.3.3g (RA/20261030-trainer-speed, Jacob pt841u/t847u): execution-only speedups, same update.
#  * install_fast_hash: subnet.mimo_runtime.file_hash = the ORIGINAL function run in a thread pool (siblings *.safetensors
#    prefetched in parallel) and memoized by (path, dev, ino, size, mtime_ns, ctime_ns); a changed file re-hashes.
#  * helper ranks preload the KL reference + an actor copy at process start from PRELOAD_HINT (written by rank 0 every job);
#    the actor parameters are still overwritten by the rank-0 broadcast at every job (bit-exact copy as before).
PRELOAD_HINT = '/workspace/affine-mimo-trainer-prod/v833g-helper-preload.json'


def install_fast_hash(workers=16):
    import stat as _st, threading
    from concurrent.futures import ThreadPoolExecutor
    import subnet.mimo_runtime as R
    if getattr(R.file_hash, '_v833g', False):
        return R.file_hash
    original = R.file_hash
    memo, lock = {}, threading.Lock()
    pool = ThreadPoolExecutor(workers, thread_name_prefix='v833g-hash')

    def key(path):
        st = os.lstat(path)
        if not _st.S_ISREG(st.st_mode):
            return None
        return (os.path.abspath(str(path)), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    def submit(path, k):
        with lock:
            f = memo.get(k)
            if f is None:
                f = memo[k] = pool.submit(original, path)
        return f

    def file_hash(path):
        try:
            k = key(path)
        except OSError:
            return original(path)
        if k is None:
            return original(path)
        f = submit(path, k)
        if str(path).endswith('.safetensors'):
            try:
                for sib in os.scandir(os.path.dirname(os.path.abspath(str(path)))):
                    if sib.name.endswith('.safetensors') and not sib.name.startswith('.'):
                        ks = key(sib.path)
                        if ks is not None:
                            submit(sib.path, ks)
            except OSError:
                pass
        try:
            value = f.result()
        except BaseException:
            with lock:
                memo.pop(k, None)
            raise
        if key(path) != k:            # changed after hashing: never serve a stale digest
            with lock:
                memo.pop(k, None)
            return original(path)
        return value
    file_hash._v833g = True
    file_hash.__doc__ = original.__doc__
    R.file_hash = file_hash
    return file_hash


def _write_hint(obj):
    try:
        tmp = PRELOAD_HINT + '.tmp-%d' % os.getpid()
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, sort_keys=True)
        os.replace(tmp, PRELOAD_HINT)
    except Exception:
        pass

STATE = dict(world=1, ref=None, actor=None, actor_ckpt=None, initialized=False)
BUDGET = 32768; MAX_ROWS = 256; LOGIT_CHUNK = 4096; NO_CKPT_LAYERS_RANK0 = 0; NO_CKPT_LAYERS_HELPER = 2


def init(world, port):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ['MASTER_PORT'] = str(port)
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', rank=0, world_size=world, device_id=torch.device('cuda', 0), timeout=timedelta(minutes=45))
    # v8.3.1: job hand-off on a CPU (gloo) group with a long timeout: helpers idle between epochs for > 45 min
    STATE['ctrl'] = dist.new_group(backend='gloo', timeout=timedelta(days=30))
    STATE.update(world=world, initialized=True)


def _plan(rows, active, world):
    lengths = {i: len(rows[i]['group'].prompt) + len(rows[i]['rollout'].tokens) for i in active}
    bins = []
    for i in sorted(active, key=lambda i: (-lengths[i], i)):
        for b in bins:
            if b[0] + lengths[i] <= BUDGET and len(b[1]) < MAX_ROWS:
                b[0] += lengths[i]; b[1].append(i); break
        else:
            bins.append([lengths[i], [i]])
    loads = [0] * world; assign = [[] for _ in range(world)]
    for cost, members in sorted(bins, key=lambda b: (-b[0], b[1])):
        r = min(range(world), key=lambda k: loads[k]); loads[r] += cost; assign[r].append(sorted(members))
    return assign, len(bins), loads


def _bcast(obj):
    box = [obj]; dist.broadcast_object_list(box, src=0, group=STATE['ctrl']); return box[0]


def _install_fast(B, FK, runtime, no_ckpt):
    if not getattr(runtime.model, '_v83_fast', False):
        FK.install(runtime.model)
        if runtime.trainable:
            for layer in list(runtime.model.model.language_model.layers)[:no_ckpt]:
                layer.__dict__.pop('forward', None)
        runtime.model._v83_fast = True


def _passes(B, FK, actor, ref, rows, plan, config, agg_config, loss_scale, active_rollouts, versions, guard, record):
    """One rank's share. Returns (accumulator|None, details, losses, err)."""
    from subnet.mimo_grpo.optimizer import FP32Accumulator
    accumulator, details, losses, err = None, {}, {}, None
    try:
        B._reference_pass(rows, plan, config, ref, engine='packed', logit_chunk=LOGIT_CHUNK, guard=guard)
        if B._versions(actor.model) != versions:
            raise ValueError('input actor changed during reference preparation')
        for p in actor.model.parameters(): p.grad = None
        accumulator = FP32Accumulator(actor.model.named_parameters())
        B._gradient_pass(actor, rows, plan, config, agg_config, loss_scale, active_rollouts, accumulator, versions,
                         details, losses, engine='packed', logit_chunk=LOGIT_CHUNK, guard=guard, record=record)
        torch.cuda.synchronize()
    except BaseException as e:   # every rank must reach the ok-flag reduction
        err = f'{type(e).__name__}: {e}'
    return accumulator, details, losses, err


def _reduce(B, rows, accumulator, details, losses, err, rank):
    world = STATE['world']
    ok = torch.tensor([0.0 if err else 1.0], device='cuda'); dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    errs = [None] * world; dist.all_gather_object(errs, err)
    if ok.item() < 1:
        if accumulator is not None: accumulator.buffers.clear()
        return None, [e for e in errs if e]
    for name in sorted(accumulator.buffers):
        dist.reduce(accumulator.buffers[name], dst=0, op=dist.ReduceOp.SUM)
    gathered = [None] * world
    dist.all_gather_object(gathered, (details, losses, {i: (rows[i]['keep'], rows[i]['derail_index']) for i in details}))
    if rank != 0:
        accumulator.buffers.clear(); return None, []
    all_d, all_l = {}, {}
    for d, l, keep in gathered:
        all_d.update(d); all_l.update(l)
        for i, (k, di) in keep.items(): rows[i]['keep'], rows[i]['derail_index'] = k, di
    return (all_d, all_l), []


def make_train_update(B, FK):
    """-> train_update_batched-compatible function (installed by the v8.3 host as v824_batched.train_update_batched)."""
    original = B.train_update_batched

    def fast_blp(runtime, items, *, gradients, engine, logit_chunk=2048):
        runtime.guard()
        if gradients and not runtime.trainable:
            raise ValueError('inference runtime cannot train')
        if engine != 'packed':
            raise ValueError('v8.3 DP engine is packed-only')
        return FK.packed_log_probs(runtime, items, gradients=gradients, logit_chunk=LOGIT_CHUNK)

    def train_update_dp(runtime, optimizer, groups, binding, *, config, guard=lambda: None, record=lambda row: None,
                        reference_runtime=None, engine='packed', token_budget=16384, max_rows=64, logit_chunk=2048,
                        timings=None):
        # Run the ORIGINAL preamble checks by calling the original with an empty-groups probe? No: replicate exactly by
        # delegating the no-update path and all validation to the original function's own code below.
        groups = tuple(groups)
        if not groups:
            return original(runtime, optimizer, groups, binding, config=config, guard=guard, record=record,
                            reference_runtime=reference_runtime, engine='padded', token_budget=token_budget,
                            max_rows=max_rows, logit_chunk=logit_chunk, timings=timings)
        return _dp_update(runtime, optimizer, groups, binding, config, guard, record, reference_runtime, timings)

    def _dp_update(runtime, optimizer, groups, binding, config, guard, record, reference_runtime, timings):
        PRE.validate(runtime, optimizer, groups, binding, config, reference_runtime)
        agg_config, loss_scale, aggregation = B.loss_aggregation(config)
        timings = {} if timings is None else timings
        started = time.monotonic(); guard()
        world = STATE['world']
        base_report = PRE.base_report(B, runtime, optimizer, groups, binding, config)
        runtime.model.eval(); versions = B._versions(runtime.model)
        if B.DROP_UNRESOLVED:
            rows, unresolved_filter = B.layout_drop_unresolved(groups, config)
        else:
            rows, unresolved_filter = B._layout(groups, config), dict(enabled=False)
        record(dict(stage='unresolved-filter', **unresolved_filter))
        active = [i for i, row in enumerate(rows) if row['active']]
        assign, nbins, loads = _plan(rows, active, world)
        _install_fast(B, FK, runtime, NO_CKPT_LAYERS_RANK0); _install_fast(B, FK, reference_runtime, 0)
        # helpers: job header, then the actor weights (bit-exact copy of the rank-0 actor)
        t = time.monotonic()
        _bcast(dict(op='job', groups=groups, config=config, drop=B.DROP_UNRESOLVED, mode=B.LOSS_AGG_MODE, scale=B.LOSS_SCALE,
                    actor_dir=str(runtime._checkpoint_guard.directory), actor_desc=runtime._checkpoint_guard.descriptor,
                    ref_dir=str(reference_runtime._checkpoint_guard.directory), ref_desc=reference_runtime._checkpoint_guard.descriptor,
                    spec=runtime.spec.to_dict(), assign=assign))
        for _, p in runtime.model.named_parameters():
            dist.broadcast(p.data, src=0)
        timings['dp_sync_seconds'] = time.monotonic() - t
        _write_hint(dict(spec=runtime.spec.to_dict(),
                         actor_dir=str(runtime._checkpoint_guard.directory), actor_desc=runtime._checkpoint_guard.descriptor,
                         ref_dir=str(reference_runtime._checkpoint_guard.directory), ref_desc=reference_runtime._checkpoint_guard.descriptor))
        prep = time.monotonic()
        acc, details, losses, err = _passes(B, FK, runtime, reference_runtime, rows, assign[0], config, agg_config, loss_scale,
                                             len(active), versions, guard, record)
        gathered, errs = _reduce(B, rows, acc, details, losses, err, 0)
        if gathered is None:
            raise RuntimeError('v8.3 DP update failed before the optimizer step: ' + '; '.join(errs))
        all_d, all_l = gathered
        for i in sorted(all_d):     # trajectory-loss events for helper-owned rollouts (rank-0 ones were recorded live)
            if i not in details: record(dict(stage='trajectory-loss', **all_d[i]))
        timings['gradient_seconds'] = time.monotonic() - prep
        norm = acc.clip(config.max_gradient_norm)
        guard()
        if B._versions(runtime.model) != versions:
            raise ValueError('actor changed before optimizer step')
        step_started = time.monotonic()
        update = optimizer.step(acc.buffers)
        if optimizer.global_step != binding.optimizer_step + 1:
            raise ValueError('exactly one optimizer step required')
        acc.buffers.clear()
        ordered = [all_d[i] for i in sorted(all_d)]
        return dict(base_report, status='updated', optimizer_step=optimizer.global_step,
            trained_rollouts=len(ordered), masked_rollouts=len(rows) - len(ordered),
            sampled_tokens=sum(len(r['rollout'].tokens) for r in rows),
            trained_tokens=sum(r['tokens'] for r in ordered),
            loss_tokens=sum(r['loss_tokens'] for r in ordered),
            derailed_rollouts=sum(r['derail_index'] is not None for r in ordered),
            unresolved_trained_rollouts=sum(r['outcome'] == 'unresolved' for r in ordered),
            outcome_counts={outcome: sum(r['rollout'].outcome == outcome for r in rows)
                            for outcome in ('correct', 'wrong', 'unresolved')},
            objective_loss=math.fsum(all_l[i] for i in sorted(all_l)), preclip_gradient_norm=norm,
            maximum_old_actor_logprob_drift=0.0, trajectories=ordered,
            optimizer=update, preparation_seconds=timings['gradient_seconds'],
            optimizer_seconds=time.monotonic() - step_started,
            update_seconds=time.monotonic() - started,
            microbatch=dict(engine='packed-dp-v83', token_budget=BUDGET, max_rows=MAX_ROWS, count=nbins, world=world,
                            rank_tokens=loads),
            unresolved_filter=unresolved_filter,
            data_sha256=digest([__import__('dataclasses').asdict(group) for group in groups]),
            loss_aggregation=aggregation,
            epoch_metrics=B.epoch_metrics(rows, all_d, config, aggregation, len(active), norm))

    from subnet.mimo_grpo.data import digest
    B.batched_log_probs = fast_blp
    train_update_dp._original = original
    return train_update_dp


class PRE:
    """The exact precondition block of v8211_batched.train_update_batched (lines up to base_report), factored out."""
    @staticmethod
    def validate(runtime, optimizer, groups, binding, config, reference_runtime):
        from subnet.mimo_grpo.config import SC_MIS, PROMPT_MEAN, TRAIN_UNRESOLVED
        from subnet.mimo_grpo.data import TaskGroup
        config.validate(); binding.validate()
        if config.ppo_epochs != 1:
            raise ValueError('actor-as-OLD requires exactly one PPO epoch / one optimizer step')
        if config.sampler_correction == SC_MIS:
            raise ValueError('sampler correction not supported by the batched engine')
        if config.loss_agg_mode == PROMPT_MEAN and config.derail_mask:
            raise ValueError('prompt-mean weights need derail keep before the gradient pass')
        if len(groups) > config.max_tasks or len({g.task_id for g in groups}) != len(groups):
            raise ValueError('distinct complete tasks within optimizer batch limit')
        if runtime.checkpoint_id != binding.checkpoint or optimizer.input_checkpoint != binding.checkpoint:
            raise ValueError('trusted runtime/optimizer must match input checkpoint')
        if optimizer.global_step != binding.optimizer_step or optimizer.config != config or optimizer.failed:
            raise ValueError('original healthy optimizer step and resolved config')
        if config.kl_coefficient > 0:
            if reference_runtime is None or reference_runtime.checkpoint_id != config.kl_reference_checkpoint:
                raise ValueError('KL anchoring requires the configured frozen reference runtime')
            if reference_runtime is runtime or getattr(reference_runtime, 'model', None) is runtime.model:
                raise ValueError('KL reference must be a separate frozen model')
        elif reference_runtime is not None:
            raise ValueError('reference runtime supplied without a KL objective')
        expected = tuple((name, id(p)) for name, p in runtime.model.named_parameters() if p.requires_grad)
        if expected != tuple((name, id(p)) for name, p in optimizer.parameters):
            raise ValueError('optimizer must own exact trainable actor parameters')
        for group in groups:
            if not isinstance(group, TaskGroup):
                raise ValueError('native trusted TaskGroup required')
            group.validate(config.group_size, vocab_size=runtime.vocab_size, eos_token_ids=set(runtime.eos_token_ids),
                           unresolved_failures=config.cap_policy == TRAIN_UNRESOLVED)
            if any(len(group.prompt) + len(r.tokens) > runtime.max_context_tokens for r in group.trajectories):
                raise ValueError('complete rollout exceeds qualified total context')

    @staticmethod
    def base_report(B, runtime, optimizer, groups, binding, config):
        return dict(version='affine-mimo-grpo-update-v1', config=config.document(), input=dict(
            run=binding.run, epoch=binding.epoch, checkpoint=binding.checkpoint,
            contract_sha256=binding.contract_sha256, optimizer_step=binding.optimizer_step),
            admitted_groups=[dict(task_id=g.task_id, admission_sha256=g.admission_sha256) for g in groups],
            group_count=len(groups), rollout_count=len(groups) * config.group_size,
            input_checkpoint=binding.checkpoint, optimizer_step_before=optimizer.global_step)


def helper_main(rank, world, port, host_dir, source_root):
    """Ranks 1..W-1: resident KL reference + actor copy; serve jobs until the group closes."""
    sys.path.insert(0, source_root); sys.path.insert(0, host_dir)
    import threading
    parent = os.getppid()
    def watchdog():   # the worker (rank 0) is our parent: never outlive it
        while True:
            time.sleep(5)
            if os.getppid() != parent: os._exit(3)
    threading.Thread(target=watchdog, daemon=True).start()
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ['MASTER_PORT'] = str(port)
    torch.cuda.set_device(0)   # CUDA_VISIBLE_DEVICES pins one GPU per helper
    dist.init_process_group('nccl', rank=rank, world_size=world, device_id=torch.device('cuda', 0), timeout=timedelta(minutes=45))
    STATE['ctrl'] = dist.new_group(backend='gloo', timeout=timedelta(days=30))
    import importlib.util
    def load(name, rel):
        spec = importlib.util.spec_from_file_location(name, os.path.join(host_dir, rel)); m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m; spec.loader.exec_module(m); return m
    B = load('v824_batched', 'v8211_batched.py'); FK = load('v83_fast_kernels', 'v83_fast_kernels.py')
    import v83_kernels_policy as KP
    KP.allow()
    from subnet.mimo_runtime import MimoRuntime
    STATE['world'] = world
    def fast_blp(runtime, items, *, gradients, engine, logit_chunk=2048):
        return FK.packed_log_probs(runtime, items, gradients=gradients, logit_chunk=LOGIT_CHUNK)
    B.batched_log_probs = fast_blp
    install_fast_hash()
    try:      # v8.3.3g preload (any failure -> lazy load at the first job, as before)
        hint = json.loads(open(PRELOAD_HINT).read())
        STATE['ref'] = MimoRuntime.load(hint['ref_dir'], hint['ref_desc'], hint['spec'], device='cuda', trainable=False)
        _install_fast(B, FK, STATE['ref'], 0)
        STATE['actor'] = MimoRuntime.load(hint['actor_dir'], hint['actor_desc'], hint['spec'], device='cuda', trainable=True)
        _install_fast(B, FK, STATE['actor'], NO_CKPT_LAYERS_HELPER)
        print(json.dumps(dict(component='trainer-host-v833g', event='helper-preloaded', rank=rank)), flush=True)
    except Exception as x:
        STATE['actor'] = None
        if STATE['ref'] is not None and not isinstance(x, FileNotFoundError):
            pass
        print(json.dumps(dict(component='trainer-host-v833g', event='helper-preload-skipped', rank=rank, error=repr(x)[:200])), flush=True)
        torch.cuda.empty_cache()
    while True:
        job = _bcast(None)
        if job is None or job.get('op') == 'stop':
            break
        B.DROP_UNRESOLVED, B.LOSS_AGG_MODE, B.LOSS_SCALE = job['drop'], job['mode'], job['scale']
        if STATE['ref'] is None or STATE['ref'].checkpoint_id != job['ref_desc']['id']:
            STATE['ref'] = MimoRuntime.load(job['ref_dir'], job['ref_desc'], job['spec'], device='cuda', trainable=False)
            _install_fast(B, FK, STATE['ref'], 0)
        if STATE['actor'] is None:
            STATE['actor'] = MimoRuntime.load(job['actor_dir'], job['actor_desc'], job['spec'], device='cuda', trainable=True)
            _install_fast(B, FK, STATE['actor'], NO_CKPT_LAYERS_HELPER)
        actor = STATE['actor']
        for _, p in actor.model.named_parameters():
            dist.broadcast(p.data, src=0)
        config = job['config']; agg_config, loss_scale, _ = B.loss_aggregation(config)
        groups = job['groups']
        rows = B.layout_drop_unresolved(groups, config)[0] if B.DROP_UNRESOLVED else B._layout(groups, config)
        active = [i for i, row in enumerate(rows) if row['active']]
        actor.model.eval(); versions = B._versions(actor.model)
        acc, details, losses, err = _passes(B, FK, actor, STATE['ref'], rows, job['assign'][rank], config, agg_config, loss_scale,
                                             len(active), versions, lambda: None, lambda row: None)
        _reduce(B, rows, acc, details, losses, err, rank)
        for p in actor.model.parameters(): p.grad = None
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == '__main__':
    r, w, port = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    helper_main(r, w, port, sys.argv[4], sys.argv[5])
