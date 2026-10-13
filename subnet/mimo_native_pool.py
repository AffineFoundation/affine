"""Bounded, ordered CPU native grading in isolated spawn processes.

This changes scheduling only. Every frozen candidate receives the original
native grade; no sampling audit, GPU model, credential or signing key enters a
worker. Each seal owns its pool and each worker loads its authenticated tokenizer
once. Spawn is required because native math graders use process-local signals.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import multiprocessing
import os
from pathlib import Path

from .audit_policy import InvalidSample

_TOKENIZER = None
_MANIFEST = None
TOKENIZER_FILES = frozenset(('chat_template.jinja', 'config.json', 'generation_config.json',
                            'merges.txt', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json'))


def authenticated_tokenizer(directory, files, template_sha256):
    from .mimo_runtime import file_hash
    from transformers import AutoTokenizer
    root = Path(directory).absolute()
    if root.resolve() != root or not root.is_dir():
        raise ValueError('ordinary tokenizer directory required')
    if not TOKENIZER_FILES.issubset(files) or {path.name for path in root.iterdir()} != TOKENIZER_FILES:
        raise ValueError('complete explicit MiMo tokenizer file inventory required')
    for path in root.iterdir():
        if (not path.is_file() or path.is_symlink() or path.name not in files
                or path.name.endswith('.safetensors') or file_hash(path) != files[path.name]):
            raise ValueError('native worker tokenizer differs from qualified checkpoint')
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != template_sha256:
        raise ValueError('native worker tokenizer template differs')
    return tokenizer


def _initialize(directory, files, template_sha256, manifest, task_assets):
    global _TOKENIZER, _MANIFEST
    # Eight scalar/symbolic grading workers must not each create a full-host
    # BLAS/tokenizer thread pool. Set these only inside the spawned children,
    # before importing tokenizer/runtime dependencies; parent services unaffected.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from .mimo_runtime import file_hash
    root = Path(__file__).resolve().parents[1]
    for name, expected in task_assets.items():
        path = root / name
        if (path.resolve() != path or not path.is_relative_to(root) or not path.is_file()
                or path.stat().st_size != expected['bytes'] or file_hash(path) != expected['sha256']):
            raise ValueError('native worker task asset differs from qualified release')
    _TOKENIZER = authenticated_tokenizer(directory, files, template_sha256)
    _MANIFEST = copy.deepcopy(manifest)


def _grade(request):
    from .complete_group_native import grade_batch
    miner, _, batch = request
    try:
        return grade_batch(batch, _MANIFEST, _TOKENIZER, miner=miner)
    except InvalidSample as error:
        return InvalidSample(str(error))


def bounded_ordered_map(executor, function, requests, limit):
    """Submit at most limit full documents; yield in original population order.

    Exceptions propagate, and no additional input is consumed after one fails.
    Outstanding futures are cancelled on failure/early close; the owning pool
    waits for already running native graders before returning to retry.
    """
    if type(limit) is not int or not 1 <= limit <= 64:
        raise ValueError('bounded native grading in-flight limit')
    requests = iter(requests)
    pending = deque()
    try:
        for _ in range(limit):
            try:
                request = next(requests)
            except StopIteration:
                break
            pending.append((request, executor.submit(function, request)))
        while pending:
            request, future = pending.popleft()
            yield request, future.result()
            try:
                request = next(requests)
            except StopIteration:
                continue
            pending.append((request, executor.submit(function, request)))
    finally:
        for _, future in pending:
            future.cancel()
        if hasattr(requests, 'close'):
            requests.close()


class NativeGradingPool:
    def __init__(self, tokenizer_directory, checkpoint_files, template_sha256,
                 task_assets, *, workers=8):
        if type(workers) is not int or not 1 <= workers <= 32:
            raise ValueError('native grading workers must be between 1 and 32')
        self.workers = workers
        self.args = (str(tokenizer_directory), copy.deepcopy(checkpoint_files),
                     template_sha256, copy.deepcopy(task_assets))

    def __call__(self, requests, manifest):
        directory, files, template, assets = self.args
        # Workers are created only when a frozen population needs grading.
        # Recovered durable dispatches never create a pool or regrade inputs.
        with ProcessPoolExecutor(max_workers=self.workers,
                mp_context=multiprocessing.get_context('spawn'), initializer=_initialize,
                initargs=(directory, files, template, copy.deepcopy(manifest), assets)) as pool:
            yield from bounded_ordered_map(pool, _grade, requests, self.workers)
