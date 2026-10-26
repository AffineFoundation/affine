"""ROOT-admitted v8.3 trainer host launcher over the unchanged worker closure.

v8.3 = v8.2.11 + a data-parallel packed execution of the SAME update (v83_dp.py), on W GPUs of one node:
the worker process is rank 0 (it owns the actor, the CPUAdamW optimizer, staging and publication exactly as in
v8.2.11); W-1 helper processes (one per extra GPU, started here, children of the worker) hold the frozen KL
reference and a per-job bit-exact copy of the actor and run their share of the packed micro-batches through the same
v8211_batched._reference_pass/_gradient_pass. FP32 gradients are SUM-reduced to rank 0, then ONE clip and ONE
CPUAdamW step (unchanged). Loss, normalization, rewards, cap policy, group budget, KL, lr: unchanged.
The fast training kernels (fla chunk_gated_delta_rule / causal_conv1d varlen, torch varlen flash attention) are
declared in engine.training_kernels and installed per layer instance for training forwards only (v83_kernels_policy).
Authorized by Jacob (owner) 2026-10-25 (pt619u; parity gate "at least as close to fp32 truth as prod" approved 15:13 CR).

v8.2.11 notes:

v8.2.11 = v8.2.10 + engine.loss_agg_mode ('config' | 'seq-mean-token-mean') + engine.loss_scale, authorized by
Jacob 2026-10-24 (t586u). v8211_batched.py replaces v825_batched.py (same update; aggregation selectable; adds
report.epoch_metrics). math_levels.json maps the math env task index to its MATH level for the cap-rate metric;
mimo_training_worker.prepare_signed_dispatch is wrapped only to record task_id -> (env_id, index) from the
authenticated receipts (return value unchanged). 'config' = byte-for-byte the v8.2.10 loss.

v8.2.10 = v8.2.5 + engine.optimizer_persistence ('none' | 'synchronous'), authorized by Jacob 2026-10-22
("saving the optimizer state should not be important can we remove this requirement"). 'none': the epoch
report never waits on writing the optimizer state (v8210_optimizer_offpath.py): the model is staged and
published exactly as before; the optimizer generation is a small local marker; the live optimizer stays in
process memory; cold recovery rebuilds fresh AdamW moments with the step kept when no state was persisted.
'synchronous' = exact v8.2.5 staging/recovery (pre-signed in-place rollback variant).

v8.2.5 = v8.2.4 + engine.drop_unresolved (bool): when true, unresolved rollouts are dropped from the
loss (see v825_batched.py). Rollback: drop_unresolved=false config or the v8.2.4 owned worker.

The frozen worker source closure (source_files / source_guard / registry) is not
modified on disk. This launcher verifies a ROOT admission over its own exact
configuration (host file digests + original-method source digests), installs the
v8.2.4 in-memory overrides, then runs `subnet.distributed_worker` unchanged:
  * padded micro-batched actor-as-OLD update (token_budget 16384, max_rows 64)
  * CPUAdamW.step -> pipelined (same torch AdamW, bitwise identical; pinned overlap)
  * fast staging (inline hashes seed the transaction memo) + model preupload overlap
  * FastR2ModelPublisher (v2 receipts: streamed sha256 + part Content-MD5 + HEAD)
  * v2-aware receipt checks (transaction._receipt, validate_publication, validate_result)
Host modules load under non-`subnet.` names so source_guard's loaded-module check holds.
"""
import sys
sys.dont_write_bytecode = True
import argparse, ast, hashlib, importlib.util, inspect, json, os, runpy, textwrap
from pathlib import Path

VERSION = 'affine-mimo-trainer-host-v832'
TRITON_CACHE = '/workspace/affine-mimo-trainer-prod/triton-cache-v831'
SCOPE = 'sn120-mimo-trainer-host-overrides-v8.3.2'
P = Path(__file__).resolve().parent
HOST_FILES = ('mimo_host_trainer_v83.py', 'v83_dp.py', 'v83_fast_kernels.py', 'v83_kernels_policy.py', 'math_levels.json', 'v8210_optimizer_offpath.py', 'v8211_batched.py', 'v824_optimizer_fast.py', 'v824_staging_fast.py',
              'v824_publication_fast.py', 'patched/mimo_grpo.transaction.py', 'patched/mimo_grpo.publication.py',
              'patched/mimo_grpo.service.py', 'patched/mimo_training_worker.py')
SITES = ('mimo_grpo.transaction._receipt', 'mimo_grpo.transaction.Transaction.run',
         'mimo_grpo.publication.validate_publication', 'mimo_grpo.service.execute_transaction',
         'mimo_training_worker.validate_result', 'mimo_grpo.optimizer.CPUAdamW.step',
         'mimo_grpo.publication.R2ModelPublisher.__init__', 'mimo_grpo.recovery.restore_transaction_state',
         'mimo_training_worker.prepare_signed_dispatch')
PERSISTENCE = ('none', 'synchronous')
ENGINE_KEYS = {'batched_engine', 'token_budget', 'max_rows', 'fast_stage', 'stage_workers',
               'part_concurrency', 'shard_concurrency', 'optimizer_step_threads', 'drop_unresolved',
               'optimizer_persistence', 'loss_agg_mode', 'loss_scale',
               'dp_world', 'dp_port', 'dp_token_budget', 'dp_max_rows', 'training_kernels'}
# v8.3.2 optimizer drift control (Jacob 2026-10-26 13:06 CR). OPTIONAL engine keys; absent / None / 0 / 1.0 = OFF (v8.3.1 behaviour).
DRIFT_KEYS = {'drift_cap_rel', 'restart_lr_warm_steps', 'restart_lr_scale'}
DRIFT_FP32_MARGIN = 1e-3
DRIFT_OFF = dict(drift_cap_rel=None, restart_lr_warm_steps=0, restart_lr_scale=1.0)


def drift_settings(engine):
    """-> (drift_cap_rel or None, restart_lr_warm_steps, restart_lr_scale), validated; defaults are OFF."""
    cap = engine.get('drift_cap_rel', DRIFT_OFF['drift_cap_rel'])
    warm = engine.get('restart_lr_warm_steps', DRIFT_OFF['restart_lr_warm_steps'])
    scale = engine.get('restart_lr_scale', DRIFT_OFF['restart_lr_scale'])
    if cap is not None and not (type(cap) is float and 0 < cap <= 1):
        raise ValueError('drift_cap_rel must be None or a float in (0, 1]')
    if not (type(warm) is int and 0 <= warm <= 100):
        raise ValueError('restart_lr_warm_steps must be an int in [0, 100]')
    if not (type(scale) is float and 0 < scale <= 1):
        raise ValueError('restart_lr_scale must be a float in (0, 1]')
    return cap, warm, scale


def _fresh_moments(opt):
    """True if every AdamW state is absent or has all-zero moments (new optimizer or fresh-moments cold recovery)."""
    for name, _ in opt.parameters:
        state = opt.optimizer.state.get(opt.masters[name])
        if state and (bool(state['exp_avg'].any()) or bool(state['exp_avg_sq'].any())):
            return False
    return True


def controlled_step(opt, gradients, inner, cap, warm, scale, emit=None):
    """v8.3.2: ONE AdamW step (inner = the unchanged v8.3.1 pipelined step) with
    (a) restart LR: lr x scale for the first `warm` steps after fresh optimizer moments (detected once per optimizer object);
    (b) drift cap: after the step, per tensor scale the fp32 master update so ||dW_t||/||W_t|| <= cap, then one global factor so
        ||dW||/||W|| <= cap over all tensors; the bf16 parameters are re-projected from the capped masters.
    Runs only on the worker (rank 0, which owns the single CPU AdamW); helper ranks receive these exact bf16 weights by
    dist.broadcast(src=0) at the next job, so all ranks are identical by construction. Adam moments are not modified."""
    import math, time, torch
    t0 = time.monotonic()
    if not getattr(opt, '_v832_checked', False):
        opt._v832_fresh_at_start = _fresh_moments(opt)
        opt._v832_since_fresh = 0 if opt._v832_fresh_at_start else None
        opt._v832_checked = True
    since = opt._v832_since_fresh
    lr_factor = scale if (since is not None and since < warm) else 1.0
    old = {name: opt.masters[name].detach().clone() for name, _ in opt.parameters} if cap else None
    groups = opt.optimizer.param_groups
    saved = [g['lr'] for g in groups]
    try:
        if lr_factor != 1.0:
            for g in groups:
                g['lr'] = g['lr'] * lr_factor
        result = inner(opt, gradients)
    finally:
        for g, lr in zip(groups, saved):
            g['lr'] = lr
    info = dict(version='v832-drift-control-v1', optimizer_step=opt.global_step, fresh_moments_at_start=opt._v832_fresh_at_start,
                steps_since_fresh_before=since, lr_factor=lr_factor, effective_lr=saved[0] * lr_factor if saved else None,
                restart_lr_warm_steps=warm, restart_lr_scale=scale, drift_cap_rel=cap)
    if cap:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(max(previous_threads, getattr(opt, 'step_threads', 1)))
        try:
            result, info = _drift_cap(opt, result, info, old, cap)
        finally:
            torch.set_num_threads(previous_threads)
    if since is not None:
        opt._v832_since_fresh = since + 1
    info['seconds'] = time.monotonic() - t0
    result = dict(result, drift_control=info)
    (emit or (lambda row: print(json.dumps(row, sort_keys=True), flush=True)))(dict(component='trainer-host-v832', event='drift-control', **info))
    return result


def _drift_cap(opt, result, info, old, cap):
    import math, torch
    if True:
        f64 = torch.float64
        rows = []
        for name, _ in opt.parameters:
            o = old[name]; m = opt.masters[name].detach()
            dn = float(torch.linalg.vector_norm(m - o, dtype=f64)); wn = float(torch.linalg.vector_norm(o, dtype=f64))
            rows.append((name, dn, wn))
        W = math.sqrt(math.fsum(wn * wn for _, _, wn in rows))
        pre = math.sqrt(math.fsum(dn * dn for _, dn, _ in rows)) / W
        target = cap * (1.0 - DRIFT_FP32_MARGIN)   # fp32 rounding of old + f*(new-old) must not push a tensor past `cap`
        per = [min(1.0, target * wn / dn) if dn > 0 and wn > 0 else 1.0 for _, dn, wn in rows]
        mid = math.sqrt(math.fsum((s * dn) ** 2 for s, (_, dn, _) in zip(per, rows))) / W
        gscale = min(1.0, target / mid) if mid > 0 else 1.0
        # (sum dn_t^2 s_t^2)/(sum W_t^2) <= cap^2 once every tensor is capped, so the global factor only binds through fp32 rounding;
        # it is still applied and re-measured so the logged global bound is a measured fact, not an assumption.
        params = dict(opt.parameters)
        def apply(name, f):
            o = old[name]; m = opt.masters[name]; parameter = params[name]
            m.copy_(o + (opt._v832_new[name] - o) * f)
            parameter.copy_(m.detach().to(device=parameter.device, dtype=parameter.dtype))
            return float(torch.linalg.vector_norm(m.detach() - o, dtype=f64))
        factors = {}; post = {}; capped = 0
        with torch.no_grad():
            opt._v832_new = {}
            for s, (name, dn, wn) in zip(per, rows):
                f = s * gscale; factors[name] = f; post[name] = dn
                capped += s < 1.0
                if f < 1.0:
                    opt._v832_new[name] = opt.masters[name].detach().clone()
                    post[name] = apply(name, f)
                    for _ in range(4):                     # exact per-tensor bound despite fp32 rounding
                        if wn == 0 or post[name] / wn <= cap:
                            break
                        factors[name] *= cap / (post[name] / wn) * (1.0 - DRIFT_FP32_MARGIN)
                        post[name] = apply(name, factors[name])
            post_rel = math.sqrt(math.fsum(v * v for v in post.values())) / W
            repass = 1.0
            if post_rel > cap:
                repass = cap / post_rel * (1.0 - DRIFT_FP32_MARGIN)
                for name, dn, wn in rows:
                    if name not in opt._v832_new:
                        opt._v832_new[name] = opt.masters[name].detach().clone()
                    factors[name] *= repass
                    post[name] = apply(name, factors[name])
                post_rel = math.sqrt(math.fsum(v * v for v in post.values())) / W
            del opt._v832_new
            changed = 0
            for name, parameter in opt.parameters:
                changed += int((opt.masters[name].detach().to(parameter.dtype) != old[name].to(parameter.dtype)).sum())
                old[name] = None
        max_post = max((post[n] / wn for n, _, wn in rows if wn > 0), default=0.0)
        max_pre = max((dn / wn for _, dn, wn in rows if wn > 0), default=0.0)
        if post_rel > cap or max_post > cap:
            opt.failed = True
            raise ValueError('v8.3.2 drift cap bound not met')
        if not all(torch.isfinite(opt.masters[n]).all() for n, _ in opt.parameters):
            opt.failed = True
            raise ValueError('v8.3.2 drift cap produced a non-finite master')
        info.update(pre_cap_rel=pre, post_cap_rel=post_rel, pre_cap_dW=pre * W, post_cap_dW=post_rel * W, weight_norm=W,
                    tensors=len(rows), tensors_capped=int(capped), tensor_cap_scale_min=min(per), global_scale=gscale * repass,
                    min_factor=min(factors.values()), cap_fired=bool(capped or gscale * repass < 1.0),
                    max_tensor_rel_pre=max_pre, max_tensor_rel_post=max_post,
                    zero_norm_tensors=sum(1 for _, _, wn in rows if wn == 0),
                    projected_changed_elements_precap=result.get('projected_changed_elements'))
        result = dict(result, projected_changed_elements=changed)
    return result, info
LOSS_AGG_MODES = ('config', 'seq-mean-token-mean')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def method_sha(obj):
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def private_json(path):
    path = Path(path)
    st = path.lstat()
    if not path.is_file() or path.is_symlink() or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise ValueError('private regular operator configuration required')
    return json.loads(path.read_bytes())


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, P / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def patched_def(rel, qualname):
    """Exact source of one def from the admitted patched copy."""
    tree = ast.parse((P / 'patched' / rel).read_text())
    parts = qualname.split('.')
    nodes = tree.body
    for i, part in enumerate(parts):
        hits = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part]
        if len(hits) != 1:
            raise ValueError('exact patched definition: ' + qualname)
        node = hits[0]; nodes = node.body
    lines = (P / 'patched' / rel).read_text().splitlines(keepends=True)
    return textwrap.dedent(''.join(lines[node.lineno - 1:node.end_lineno]))


def patched_assign(rel, names):
    text = (P / 'patched' / rel).read_text(); out = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], 'id', None) in names:
            out.append(ast.get_source_segment(text, node))
    if len(out) != len(names):
        raise ValueError('exact patched constants')
    return '\n'.join(out) + '\n'


def define(module, source, name, cls=None):
    """Compile in a copy of the module globals (methods) or the module itself (functions)."""
    if cls is None:
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
        return module.__dict__[name]
    # Methods compile against the live module globals (later global patches stay visible);
    # the temporary top-level name is removed again.
    if name in module.__dict__:
        raise ValueError('method name collides with a module global: ' + name)
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    function = module.__dict__.pop(name)
    original = getattr(cls, name)
    function.__qualname__, function.__module__ = original.__qualname__, original.__module__
    function.__defaults__, function.__kwdefaults__ = original.__defaults__, original.__kwdefaults__
    return function


def checked(config_path):
    config = private_json(config_path)
    need = lambda c, m: (_ for _ in ()).throw(ValueError(m)) if not c else None
    need(set(config) == {'version', 'scope', 'authority', 'source_root', 'host_files', 'original_methods',
                         'engine', 'admission'} and config['version'] == VERSION and config['scope'] == SCOPE,
         'exact v8.3 trainer host configuration')
    need(set(config['host_files']) == set(HOST_FILES) and set(config['original_methods']) == set(SITES)
         and ENGINE_KEYS <= set(config['engine']) <= ENGINE_KEYS | DRIFT_KEYS and type(config['engine']['drop_unresolved']) is bool
         and config['engine']['optimizer_persistence'] in PERSISTENCE and config['engine']['fast_stage'] is True
         and config['engine']['loss_agg_mode'] in LOSS_AGG_MODES and type(config['engine']['loss_scale']) is float
         and 0 < config['engine']['loss_scale'] <= 1024
         and (config['engine']['loss_agg_mode'] != 'config' or config['engine']['loss_scale'] == 1.0)
         and config['engine']['batched_engine'] == 'packed'
         and type(config['engine']['dp_world']) is int and 1 <= config['engine']['dp_world'] <= 8
         and type(config['engine']['dp_port']) is int and 1024 < config['engine']['dp_port'] < 65536
         and type(config['engine']['dp_token_budget']) is int and 8192 <= config['engine']['dp_token_budget'] <= 65536
         and type(config['engine']['dp_max_rows']) is int and 1 <= config['engine']['dp_max_rows'] <= 1024
         and type(config['engine']['training_kernels']) is dict,
         'exact v8.3 host inventory')
    drift_settings(config['engine'])
    for name, digest_ in config['host_files'].items():
        path = P / name
        need(path.is_file() and not path.is_symlink() and file_sha(path) == digest_, 'exact admitted host file ' + name)
    root = Path(config['source_root']).resolve()
    need(root.is_dir() and Path.cwd().resolve() == root, 'worker cwd is the admitted closure root')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from subnet.backend_jobs import signed
    from subnet.storage import canonical, sha
    proposal = {k: v for k, v in config.items() if k != 'admission'}
    expected = dict(version=VERSION, configuration_sha256=sha(canonical(proposal)), source_unchanged=True, scope=SCOPE)
    need(signed(config['admission'], config['authority']) == expected, 'exact ROOT trainer host admission')
    return config


def install(config, start_dp=False):
    import subnet.mimo_grpo.transaction as transaction
    import subnet.mimo_grpo.publication as publication
    import subnet.mimo_grpo.service as service
    import subnet.mimo_grpo.optimizer as optimizer
    import subnet.mimo_grpo.handover as handover
    import subnet.mimo_grpo.recovery as recovery
    import subnet.mimo_training_worker as worker
    import subnet.mimo_grpo as mimo_grpo
    import subnet.mimo_grpo.prepared as prepared
    pins = config['original_methods']
    objects = {
        'mimo_grpo.transaction._receipt': transaction._receipt,
        'mimo_grpo.transaction.Transaction.run': transaction.Transaction.run,
        'mimo_grpo.publication.validate_publication': publication.validate_publication,
        'mimo_grpo.service.execute_transaction': service.execute_transaction,
        'mimo_training_worker.validate_result': worker.validate_result,
        'mimo_grpo.optimizer.CPUAdamW.step': optimizer.CPUAdamW.step,
        'mimo_grpo.publication.R2ModelPublisher.__init__': publication.R2ModelPublisher.__init__,
        'mimo_grpo.recovery.restore_transaction_state': recovery.restore_transaction_state,
        'mimo_training_worker.prepare_signed_dispatch': worker.prepare_signed_dispatch,
    }
    for site, obj in objects.items():
        if method_sha(obj) != pins[site]:
            raise ValueError('exact original trainer method required: ' + site)
    if worker.R2ModelPublisher is not publication.R2ModelPublisher or worker.CPUAdamW is not optimizer.CPUAdamW \
            or worker.execute_transaction is not service.execute_transaction \
            or handover.validate_publication is not publication.validate_publication \
            or recovery._receipt is not transaction._receipt \
            or worker.restore_transaction_state is not recovery.restore_transaction_state \
            or worker.prepare_signed_dispatch is not prepared.prepare_signed_dispatch:
        raise ValueError('exact original trainer bindings required')
    engine = config['engine']
    # prepare (all compiled before any install)
    sources = dict(
        receipt=patched_def('mimo_grpo.transaction.py', '_receipt'),
        run=patched_def('mimo_grpo.transaction.py', 'Transaction.run'),
        receipt_consts=patched_assign('mimo_grpo.transaction.py', {'V1_RECEIPT', 'V2_RECEIPT'}),
        pub_consts=patched_assign('mimo_grpo.publication.py', {'VERSION_V2', 'UPLOAD_VERIFICATION_V2'}),
        verified=patched_def('mimo_grpo.publication.py', 'publication_verified'),
        validate_publication=patched_def('mimo_grpo.publication.py', 'validate_publication'),
        execute=patched_def('mimo_grpo.service.py', 'execute_transaction'),
        validate_result=patched_def('mimo_training_worker.py', 'validate_result'))
    for value in sources.values():
        compile(value, '<v824>', 'exec')
    # imported by the patched service as `v824_batched`; v8.2.5 module registered under that name
    batched = load('v824_batched', 'v8211_batched.py')
    batched.DROP_UNRESOLVED = engine['drop_unresolved']
    batched.LOSS_AGG_MODE = engine['loss_agg_mode']
    batched.LOSS_SCALE = engine['loss_scale']
    levels = json.loads((P / 'math_levels.json').read_bytes())
    if levels.get('version') != 'affine-math-task-levels-v1' or not all(
            type(v) is int and 1 <= v <= 5 for v in levels['levels'].values()):
        raise ValueError('exact math level table')
    batched.MATH_LEVELS = dict(levels['levels'])
    batched.loss_aggregation(__import__('subnet.mimo_grpo.config', fromlist=['TrainerConfig']).TrainerConfig(
        loss_agg_mode='seq-mean-token-sum-norm', loss_scale_factor=8192))  # fail fast on a bad mode/scale
    staging = load('v824_staging_fast', 'v824_staging_fast.py')
    fast_opt = load('v824_optimizer_fast', 'v824_optimizer_fast.py')
    fast_pub = load('v824_publication_fast', 'v824_publication_fast.py')
    offpath = load('v8210_optimizer_offpath', 'v8210_optimizer_offpath.py')
    # install
    exec(compile(sources['receipt_consts'], transaction.__file__, 'exec'), transaction.__dict__)
    recovery._receipt = define(transaction, sources['receipt'], '_receipt')   # recovery imported it by name
    transaction.Transaction.run = define(transaction, sources['run'], 'run', transaction.Transaction)
    exec(compile(sources['pub_consts'], publication.__file__, 'exec'), publication.__dict__)
    define(publication, sources['verified'], 'publication_verified')
    handover.validate_publication = define(publication, sources['validate_publication'], 'validate_publication')
    patched_execute = define(service, sources['execute'], 'execute_transaction')
    define(worker, sources['validate_result'], 'validate_result')

    def execute_transaction(*args, **kwargs):
        kwargs.update(batched_engine=engine['batched_engine'], token_budget=engine['token_budget'],
                      max_rows=engine['max_rows'], fast_stage=engine['fast_stage'],
                      stage_workers=engine['stage_workers'])
        return patched_execute(*args, **kwargs)
    worker.execute_transaction = execute_transaction

    original_prepare = worker.prepare_signed_dispatch

    def prepare_signed_dispatch(*args, **kwargs):
        prepared_ = original_prepare(*args, **kwargs)
        try:   # diagnostics only: never alters or blocks the authenticated preparation
            index = {g.task_id: (r['task']['env_id'], r['task']['index'])
                     for g, r in zip(prepared_['groups'], prepared_['receipts'], strict=True)}
        except Exception:
            index = {}
        batched.TASK_INDEX.clear()
        batched.TASK_INDEX.update(index)
        return prepared_
    prepare_signed_dispatch.__qualname__ = original_prepare.__qualname__
    worker.prepare_signed_dispatch = prepare_signed_dispatch

    class R2ModelPublisherV824(fast_pub.FastR2ModelPublisher):
        def __init__(self, bucket, signer, **kwargs):
            kwargs.setdefault('shard_concurrency', engine['shard_concurrency'])
            super().__init__(bucket, signer, part_concurrency=engine['part_concurrency'], **kwargs)
    worker.R2ModelPublisher = R2ModelPublisherV824

    pipelined = fast_opt.PipelinedCPUAdamW.step
    threads = engine['optimizer_step_threads']
    drift_cap, restart_warm, restart_scale = drift_settings(engine)

    def step(self, gradients):
        if not hasattr(self, '_pinned'):
            import torch
            self.step_threads = threads
            peak = max(p.numel() for _, p in self.parameters)
            # pinned host staging only when CUDA exists (production); plain host buffers otherwise
            self._pinned = [torch.empty(peak, dtype=torch.float32, pin_memory=torch.cuda.is_available()) for _ in range(2)]
            self._copy_stream = None
        if drift_cap is None and (restart_warm == 0 or restart_scale == 1.0):
            return pipelined(self, gradients)      # flags OFF: exactly the v8.3.1 step
        return controlled_step(self, gradients, pipelined, drift_cap, restart_warm, restart_scale)
    step.__qualname__ = optimizer.CPUAdamW.step.__qualname__
    optimizer.CPUAdamW.step = step
    if engine['optimizer_persistence'] == 'none':
        # The patched service imports `stage_for_publication_fast` from v824_staging_fast at call time.
        staging.stage_for_publication_fast = offpath.stage_for_publication_model_only
        restore = offpath.make_restore(recovery.restore_transaction_state)
        restore.__qualname__ = recovery.restore_transaction_state.__qualname__
        recovery.restore_transaction_state = restore
        worker.restore_transaction_state = restore     # imported by name from subnet.mimo_grpo
        mimo_grpo.restore_transaction_state = restore
    # ---- v8.3: declared training kernels + data-parallel packed execution of the same update
    # v8.3.1: private Triton kernel cache per rank on local disk (no cross-process compile race, no FUSE rename; /dev/shm is noexec)
    os.environ['TRITON_CACHE_DIR'] = TRITON_CACHE + '/rank0'
    kernels_policy = load('v83_kernels_policy', 'v83_kernels_policy.py')
    kernels = kernels_policy.allow(engine['training_kernels'])
    fast_kernels = load('v83_fast_kernels', 'v83_fast_kernels.py')
    dp = load('v83_dp', 'v83_dp.py')
    dp.BUDGET, dp.MAX_ROWS = engine['dp_token_budget'], engine['dp_max_rows']
    helpers = []
    if start_dp:
        import subprocess
        world, port = engine['dp_world'], engine['dp_port']
        for r in range(1, world):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(r), MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port),
                       TRITON_CACHE_DIR=TRITON_CACHE + '/rank%d' % r)
            helpers.append(subprocess.Popen([sys.executable, '-B', str(P / 'v83_dp.py'), str(r), str(world), str(port),
                                             str(P), str(Path(config['source_root']).resolve())],
                                            env=env, cwd=str(Path(config['source_root']).resolve())))
        dp.init(world, port)
        dp.STATE['helpers'] = helpers
    batched.train_update_batched = dp.make_train_update(batched, fast_kernels)
    return dict(component='trainer-host-v83', dp_world=engine['dp_world'] if start_dp else 0, training_kernels=kernels,
                engine=engine, drop_unresolved=batched.DROP_UNRESOLVED,
                loss_agg_mode=batched.LOSS_AGG_MODE, loss_scale=batched.LOSS_SCALE, math_levels=len(batched.MATH_LEVELS),
                drop_policy=batched.DROP_POLICY, optimizer_persistence=engine['optimizer_persistence'],
                drift_control=dict(drift_cap_rel=drift_cap, restart_lr_warm_steps=restart_warm, restart_lr_scale=restart_scale),
                sites=sorted(SITES))


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit('usage: mimo_host_trainer_v83.py --operator-config PATH [--check] -- <distributed_worker args>')
    split = argv.index('--')
    p = argparse.ArgumentParser(); p.add_argument('--operator-config', required=True); p.add_argument('--check', action='store_true')
    args = p.parse_args(argv[:split]); worker_args = argv[split + 1:]
    config = checked(args.operator_config)
    result = install(config, start_dp=not args.check)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.check:
        return 0
    sys.argv = [str(Path(config['source_root']) / 'subnet' / 'distributed_worker.py')] + worker_args
    runpy.run_module('subnet.distributed_worker', run_name='__main__', alter_sys=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
