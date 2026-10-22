"""v8.2.6 host capture scheduling; supersedes capture-latency-v1 CapturePass (same contract).

Only host scheduling changes. Original complete_group_capture.capture, Coordinator.collect,
Coordinator.seal, native grading, claim order and selection are untouched and still decide
everything. Only records returned by original Coordinator.collect in this process are reused.

v8.2.6 deltas over v1:
 1. Observation time per miner: each original collect/probe observes now=max(tick_now, wall
    clock), not the (possibly ~2 h old) tick start. A commitment uploaded during a long pass is
    no longer rejected as "after now" and re-collected. The signed [start, deadline) R2
    LastModified window checks are unchanged, so nothing that reached R2 at/after the deadline
    is ever admitted.
 2. resolve_task memo: TaskAuthority (frozen) for (env_id, index) under the exact scope
    (coordinator root + full manifest digest), filled only by the original resolver.
 3. Best-effort read prefetch: up to `workers` threads only read (GET commitment, GET documents,
    HEAD proofs; no native resolution, no writes) for miners that need a full original collect, a bounded window ahead of the
    serial loop. The original collect consumes each observation once; anything missing or
    failed is read by the original code itself. Writes (immutable PUTs), ledger, signing and
    capture publication stay original, serial and in sorted order.
 4. Probe transport errors fall back to a full original collect instead of failing the pass.
 5. ROOT CAUSE fix (pure-derivation memo): inside ONE original collect, every compact document
    and rollout re-derived the same pure functions of the immutable signed opening:
    forced_sampling.binding (~33 ms, ~14 calls/group, deep-copies the 12.5 MB opening's
    environment registry), protocol.entries (~35 ms, ~7 calls/group) and per native session
    environments._source_hash (~0.34 s: re-reads+hashes ~1,280 pinned task source files) and
    complete_group_admission.scope (~29 ms, ~5 calls/group; returns a digest string).
    That is ~1.6 s CPU per group, x1,756 groups per cold pass, x~3 under GIL contention with
    the in-process audit/evaluation threads = the observed 1.5-2 h. These four are memoized
    ONLY on the capture thread while the CapturePass runs an original collect (thread-local
    switch; audit/eval/other threads always call the originals). binding/entries are keyed by
    the identity of the exact manifest object of that one collect (held strongly, cleared
    after each collect) and return deep copies; _source_hash is keyed by the canonical spec
    and cleared every pass. Exceptions are never cached.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
import time

from subnet.backend_jobs import signed
from subnet.storage import canonical, sha
from subnet.complete_group_capture import _submission_get, SubmissionRejected
from subnet.complete_group_codec import limits
from subnet.complete_group_transport import paths
from subnet.mimo_release_policy import need

VERSION = 'affine-host-capture-cache-v826'
PURE = ('subnet.forced_sampling.binding', 'subnet.protocol.entries', 'subnet.environments._source_hash',
        'subnet.complete_group_admission.scope')
# modules that bound complete_group_admission.scope by name at import time
SCOPE_ALIASES = ('subnet.complete_group_capture', 'subnet.complete_group_transport')
_active = threading.local()


def _install_pure_memo():
    """Idempotent. Wrap the three pure derivations; the wrapper is a pass-through unless the
    calling thread is inside a CapturePass original collect."""
    import copy, importlib
    for dotted in PURE:
        module_name, name = dotted.rsplit('.', 1)
        module = importlib.import_module(module_name)
        original = getattr(module, name)
        if getattr(original, '_v826_original', None) is not None:
            continue
        if name == '_source_hash':
            def wrapper(spec, _o=original, _n=dotted):
                memo = getattr(_active, 'memo', None)
                if memo is None:
                    return _o(spec)
                key = (_n, json.dumps(spec.to_dict(), sort_keys=True, default=repr))
                if key not in memo['pass']:
                    memo['pass'][key] = _o(spec)
                return memo['pass'][key]
        else:
            def wrapper(manifest, *args, _o=original, _n=dotted, **kwargs):
                memo = getattr(_active, 'memo', None)
                if memo is None or type(manifest) is not dict:
                    return _o(manifest, *args, **kwargs)
                key = (_n, id(manifest), args, tuple(sorted(kwargs.items())))
                try:
                    hash(key)
                except TypeError:
                    return _o(manifest, *args, **kwargs)
                hit = memo['collect'].get(key)
                if hit is None or hit[0] is not manifest:
                    hit = (manifest, _o(manifest, *args, **kwargs))
                    memo['collect'][key] = hit
                return hit[1] if type(hit[1]) is str else copy.deepcopy(hit[1])
        wrapper.__name__, wrapper.__qualname__, wrapper.__module__ = original.__name__, original.__qualname__, original.__module__
        wrapper.__doc__ = original.__doc__
        wrapper._v826_original = original
        if name == 'scope':
            aliases = [importlib.import_module(a) for a in SCOPE_ALIASES]
            need(all(getattr(a, 'scope') is original for a in aliases), 'scope aliases are the original function')
            for a in aliases:
                setattr(a, 'scope', wrapper)
        setattr(module, name, wrapper)


class _Body:
    def __init__(self, data):
        self._data, self._pos = data, 0
    def read(self, n=-1):
        if n is None or n < 0:
            n = len(self._data) - self._pos
        out = self._data[self._pos:self._pos + n]; self._pos += len(out)
        return out
    def close(self):
        pass


class _Observations:
    """One-shot exact R2 observations (bytes + LastModified) made after the pass began."""
    def __init__(self):
        self._lock = threading.Lock(); self._items = {}
    def put(self, key, value):
        with self._lock:
            self._items[key] = value
    def take(self, key):
        with self._lock:
            return self._items.pop(key, None)
    def drop(self, prefix_keys):
        with self._lock:
            for key in prefix_keys:
                self._items.pop(key, None)


class _PrefetchClient:
    def __init__(self, real, observations, counters):
        self._real, self._obs, self._counters = real, observations, counters
    def get_object(self, **kw):
        if set(kw) == {'Bucket', 'Key'}:
            hit = self._obs.take(('get', kw['Bucket'], kw['Key']))
            if hit is not None:
                self._counters['prefetch_hits'] += 1
                length, modified, data = hit
                return dict(Body=_Body(data), ContentLength=length, LastModified=modified)
        return self._real.get_object(**kw)
    def head_object(self, **kw):
        if set(kw) == {'Bucket', 'Key'}:
            hit = self._obs.take(('head', kw['Bucket'], kw['Key']))
            if hit is not None:
                self._counters['prefetch_hits'] += 1
                length, modified = hit
                return dict(ContentLength=length, LastModified=modified)
        return self._real.head_object(**kw)
    def __getattr__(self, name):
        return getattr(self._real, name)


class _PrefetchBucket:
    def __init__(self, real, observations, counters):
        self._real = real; self.name = real.name
        self.client = _PrefetchClient(real.client, observations, counters)
    def __getattr__(self, name):
        return getattr(self._real, name)


class CapturePass:
    def __init__(self, *, workers=8):
        need(type(workers) is int and 1 <= workers <= 16, 'bounded capture probe concurrency')
        self.workers = workers
        self.scope = None
        self.accepted = {}
        self.tasks = {}
        self.last_metrics = {}
        self._io = None
        self._io_owner = None

    @staticmethod
    def _record(coordinator, manifest, miner, record):
        value = signed(record, coordinator.identity.id)
        need(value.get('epoch') == manifest['epoch'] and value.get('miner') == miner
             and value.get('manifest_sha256') == sha(canonical(manifest))
             and value.get('assurance') == 'structurally-eligible-unaudited'
             and value.get('fully_audited') is False,
             'original authenticated structural capture only')
        return value

    def _read_client(self, bucket):
        # Dedicated bounded read client (existing Bucket.capture_io_client: 64-connection pool,
        # 1 attempt); any failure simply leaves the read to the original code path.
        if self._io_owner is not bucket:
            maker = getattr(bucket, 'capture_io_client', None)
            self._io = maker(parallel_workers=64) if maker is not None else bucket.client
            self._io_owner = bucket
        return self._io

    def _resolver(self, original, manifest):
        epoch, checkpoint = manifest['epoch'], manifest['checkpoint']['id']
        tasks, lock = self.tasks, threading.Lock()
        def resolve(m, env_id, index):
            need(m['epoch'] == epoch and m['checkpoint']['id'] == checkpoint, 'memo scope is this exact opening')
            key = (env_id, index)
            with lock:
                hit = tasks.get(key)
            if hit is not None:
                return hit
            value = original(m, env_id, index)   # original pinned native resolver
            with lock:
                if len(tasks) < 65536:
                    tasks[key] = value
            return value
        return resolve

    def __call__(self, lifecycle, coordinator, manifest, now):
        scope = (str(coordinator.root.absolute()), sha(canonical(manifest)))
        if scope != self.scope:
            self.scope = scope
            self.accepted.clear()
            self.tasks.clear()
        started = time.time()
        observe = lambda: max(now, int(time.time()))
        miners = lifecycle.services.committed_miners(manifest)
        need(type(miners) is list and len(miners) <= len(manifest['capabilities'])
             and len(set(miners)) == len(miners) and set(miners) <= set(manifest['capabilities']),
             'complete registered commitment listing')
        miners = sorted(miners)
        bound = limits(manifest)
        counters = dict(listed=len(miners), reused=0, full_collect=0, rejected=0,
                        probe_fallback=0, prefetch_hits=0, prefetched_miners=0)
        reader = self._read_client(coordinator.bucket)
        probe_bucket = _PrefetchBucket(coordinator.bucket, _Observations(), counters)
        probe_bucket.client = reader
        name = coordinator.bucket.name

        def probe(miner):
            previous = self.accepted.get(miner)
            if previous is None:
                return False
            try:
                data, received = _submission_get(probe_bucket, paths(manifest, miner), bound.commitment_bytes)
            except SubmissionRejected:
                return False  # Original collect supplies the canonical rejection.
            except Exception:
                counters['probe_fallback'] += 1
                return False  # Transport doubt: the original full collect decides.
            return (manifest['start'] <= received < manifest['deadline'] and received <= observe()
                    and sha(data) == previous['commitment_sha256'])

        observations = _Observations()
        resolve = self._resolver(coordinator.resolve_task, manifest)

        def read(key, limit):
            try:
                response = reader.get_object(Bucket=name, Key=key)
            except Exception:
                return None
            stream = response['Body']
            try:
                size = response['ContentLength']
                if type(size) is not int or not 0 < size <= limit:
                    return None
                data = stream.read(size + 1)
                if len(data) != size:
                    return None
                observations.put(('get', name, key), (size, response['LastModified'], data))
                return data
            except Exception:
                return None
            finally:
                try: stream.close()
                except Exception: pass

        def prefetch(miner):
            keys = [('get', name, paths(manifest, miner))]
            try:
                data = read(paths(manifest, miner), bound.commitment_bytes)
                if data is None:
                    return keys
                groups = json.loads(data)['payload']['groups']   # hint only; original re-validates bytes
                for claim in groups[:4096]:
                    proof_key, document_key = paths(manifest, miner, claim['slot'])
                    keys += [('get', name, document_key), ('head', name, proof_key)]
                    read(document_key, bound.metadata_bytes)
                    try:
                        meta = reader.head_object(Bucket=name, Key=proof_key)
                        observations.put(('head', name, proof_key), (meta['ContentLength'], meta['LastModified']))
                    except Exception:
                        pass
                    # Native task resolution is never run in these threads (tokenizer/session
                    # code is not thread-safe); only the serial original path resolves (memoized).
            except Exception:
                pass
            return keys

        rejected = {}
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='commitment-probe') as pool:
            unchanged_by_miner = dict(zip(miners, pool.map(probe, miners), strict=True))
        todo = []
        for miner in miners:
            if unchanged_by_miner[miner]:
                previous = self.accepted[miner]
                fd = coordinator._lock()
                try:
                    need(not (coordinator.root/'POPULATION.json').exists()
                         and not (coordinator.root/'DISPATCH.json').exists(),
                         'epoch population already sealed')
                    path = coordinator.root/'captured'/(miner+'.json')
                    if (path.is_file() and not path.is_symlink()
                            and path.stat().st_size <= 4*1024**2
                            and sha(path.read_bytes()) == previous['record_sha256']):
                        counters['reused'] += 1
                        continue
                finally:
                    os.close(fd)
            todo.append(miner)
        _install_pure_memo()
        pass_memo = {}
        original_bucket, original_resolve = coordinator.bucket, coordinator.resolve_task
        coordinator.bucket = _PrefetchBucket(original_bucket, observations, counters)
        coordinator.resolve_task = resolve
        window = 2 * self.workers
        try:
            with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='capture-prefetch') as pool:
                futures = {}
                for i, miner in enumerate(todo):
                    for ahead in todo[i:i + window]:
                        if ahead not in futures:
                            futures[ahead] = pool.submit(prefetch, ahead)
                    future = futures.pop(miner)
                    try:
                        keys = future.result()
                        counters['prefetched_miners'] += 1
                    except Exception:
                        keys = []
                    try:
                        counters['full_collect'] += 1
                        self.accepted.pop(miner, None)
                        _active.memo = {'collect': {}, 'pass': pass_memo}
                        try:
                            result = coordinator.collect(miner, now=observe())   # ORIGINAL collect
                        finally:
                            _active.memo = None
                        value = self._record(coordinator, manifest, miner, result)
                        path = coordinator.root/'captured'/(miner+'.json')
                        raw = path.read_bytes()
                        need(json.loads(raw) == result, 'original capture durable local acknowledgement')
                        self.accepted[miner] = dict(commitment_sha256=value['commitment_sha256'],
                                                   record_sha256=sha(raw))
                    except SubmissionRejected as error:
                        rejected[miner] = str(error)[:1024]
                        counters['rejected'] += 1
                    finally:
                        observations.drop(keys)   # unused observations never outlive their miner
                for future in futures.values():
                    future.cancel()
        finally:
            coordinator.bucket, coordinator.resolve_task = original_bucket, original_resolve
        counters['seconds'] = round(time.time() - started, 1)
        counters['tasks_memo'] = len(self.tasks)
        self.last_metrics = counters
        return rejected
