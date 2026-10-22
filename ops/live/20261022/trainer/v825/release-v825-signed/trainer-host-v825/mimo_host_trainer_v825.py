"""ROOT-admitted v8.2.5 trainer host launcher over the unchanged worker closure.

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

VERSION = 'affine-mimo-trainer-host-v825'
SCOPE = 'sn120-mimo-trainer-host-overrides-v8.2.5'
P = Path(__file__).resolve().parent
HOST_FILES = ('mimo_host_trainer_v825.py', 'v825_batched.py', 'v824_optimizer_fast.py', 'v824_staging_fast.py',
              'v824_publication_fast.py', 'patched/mimo_grpo.transaction.py', 'patched/mimo_grpo.publication.py',
              'patched/mimo_grpo.service.py', 'patched/mimo_training_worker.py')
SITES = ('mimo_grpo.transaction._receipt', 'mimo_grpo.transaction.Transaction.run',
         'mimo_grpo.publication.validate_publication', 'mimo_grpo.service.execute_transaction',
         'mimo_training_worker.validate_result', 'mimo_grpo.optimizer.CPUAdamW.step',
         'mimo_grpo.publication.R2ModelPublisher.__init__')
ENGINE_KEYS = {'batched_engine', 'token_budget', 'max_rows', 'fast_stage', 'stage_workers',
               'part_concurrency', 'shard_concurrency', 'optimizer_step_threads', 'drop_unresolved'}


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
         'exact v8.2.5 trainer host configuration')
    need(set(config['host_files']) == set(HOST_FILES) and set(config['original_methods']) == set(SITES)
         and set(config['engine']) == ENGINE_KEYS and type(config['engine']['drop_unresolved']) is bool,
         'exact v8.2.5 host inventory')
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


def install(config):
    import subnet.mimo_grpo.transaction as transaction
    import subnet.mimo_grpo.publication as publication
    import subnet.mimo_grpo.service as service
    import subnet.mimo_grpo.optimizer as optimizer
    import subnet.mimo_grpo.handover as handover
    import subnet.mimo_grpo.recovery as recovery
    import subnet.mimo_training_worker as worker
    pins = config['original_methods']
    objects = {
        'mimo_grpo.transaction._receipt': transaction._receipt,
        'mimo_grpo.transaction.Transaction.run': transaction.Transaction.run,
        'mimo_grpo.publication.validate_publication': publication.validate_publication,
        'mimo_grpo.service.execute_transaction': service.execute_transaction,
        'mimo_training_worker.validate_result': worker.validate_result,
        'mimo_grpo.optimizer.CPUAdamW.step': optimizer.CPUAdamW.step,
        'mimo_grpo.publication.R2ModelPublisher.__init__': publication.R2ModelPublisher.__init__,
    }
    for site, obj in objects.items():
        if method_sha(obj) != pins[site]:
            raise ValueError('exact original trainer method required: ' + site)
    if worker.R2ModelPublisher is not publication.R2ModelPublisher or worker.CPUAdamW is not optimizer.CPUAdamW \
            or worker.execute_transaction is not service.execute_transaction \
            or handover.validate_publication is not publication.validate_publication \
            or recovery._receipt is not transaction._receipt:
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
    batched = load('v824_batched', 'v825_batched.py')
    batched.DROP_UNRESOLVED = engine['drop_unresolved']
    load('v824_staging_fast', 'v824_staging_fast.py')
    fast_opt = load('v824_optimizer_fast', 'v824_optimizer_fast.py')
    fast_pub = load('v824_publication_fast', 'v824_publication_fast.py')
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

    class R2ModelPublisherV824(fast_pub.FastR2ModelPublisher):
        def __init__(self, bucket, signer, **kwargs):
            kwargs.setdefault('shard_concurrency', engine['shard_concurrency'])
            super().__init__(bucket, signer, part_concurrency=engine['part_concurrency'], **kwargs)
    worker.R2ModelPublisher = R2ModelPublisherV824

    pipelined = fast_opt.PipelinedCPUAdamW.step
    threads = engine['optimizer_step_threads']

    def step(self, gradients):
        if not hasattr(self, '_pinned'):
            import torch
            self.step_threads = threads
            peak = max(p.numel() for _, p in self.parameters)
            # pinned host staging only when CUDA exists (production); plain host buffers otherwise
            self._pinned = [torch.empty(peak, dtype=torch.float32, pin_memory=torch.cuda.is_available()) for _ in range(2)]
            self._copy_stream = None
        return pipelined(self, gradients)
    step.__qualname__ = optimizer.CPUAdamW.step.__qualname__
    optimizer.CPUAdamW.step = step
    return dict(component='trainer-host-v825', engine=engine, drop_unresolved=batched.DROP_UNRESOLVED,
                drop_policy=batched.DROP_POLICY, sites=sorted(SITES))


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit('usage: mimo_host_trainer_v825.py --operator-config PATH [--check] -- <distributed_worker args>')
    split = argv.index('--')
    p = argparse.ArgumentParser(); p.add_argument('--operator-config', required=True); p.add_argument('--check', action='store_true')
    args = p.parse_args(argv[:split]); worker_args = argv[split + 1:]
    config = checked(args.operator_config)
    result = install(config)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.check:
        return 0
    sys.argv = [str(Path(config['source_root']) / 'subnet' / 'distributed_worker.py')] + worker_args
    runpy.run_module('subnet.distributed_worker', run_name='__main__', alter_sys=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
