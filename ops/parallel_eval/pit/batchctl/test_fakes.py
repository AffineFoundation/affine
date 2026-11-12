"""Fakes that more than one test file uses. This file holds no test.

``FakeLium``      the Lium command line, in memory. It is called like the ops
                  helper: ``cli('ps', '--format', 'json', timeout=90)`` returns
                  ``(exit code, stdout, stderr)``.
``Ticker``        a clock that moves only when somebody sleeps.
``LockedStore``   the in-memory object store of ``sim.py``, safe for threads.
``ObjectServer``  that store behind signed URLs over HTTP on this machine: one
                  URL reads one key or writes one key, like a presigned R2 URL.
``put_state``     write a model state (parts, index, manifest) into a store.
"""
from __future__ import annotations

import hashlib
import hmac
import http.server
import json
import os
import threading
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import files as F
from .interfaces import NotFound
from .layout import Layout, ObjectRef, StateManifest, sha256_hex
from .sim import RawStore

BOX_HOST = 'box-%d.test'
CONTENT_TYPE = 'application/octet-stream'


class Ticker:
    """A clock that moves only when somebody sleeps or a command takes time."""

    def __init__(self, start: float = 1000.0):
        self.now = start
        self.slept: List[float] = []
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds
            self.slept.append(seconds)


def offer(huid: str, *, gpu: str = 'RTX5090', tier: str = 'secure', price: float = 4.0, ram: Any = 512,
          count: int = 8, free: Optional[int] = None, **more: Any) -> Dict[str, Any]:
    """One row of ``ls``: a node with ``count`` GPUs at ``price`` per GPU hour."""
    row = {
        'huid': huid, 'gpu_type': gpu, 'gpu_count': count,
        'available_gpu_count': count if free is None else free, 'tier': tier,
        'price_per_gpu_hour': price, 'price_per_hour': round(price * count, 4), 'ram_gb': ram,
        'disk_gb': 2000, 'download_mbps': 900, 'gpu_power_limited': False, 'country_code': 'XX',
    }
    row.update(more)
    return {key: value for key, value in row.items() if value is not None}


class FakeLium:
    """The Lium command line, in memory.

    ``script[verb]`` holds what the next calls of a verb do. An entry is:

    * an exception: the call raises it;
    * a tuple ``(code, stdout, stderr)``: the call returns it and does nothing;
    * for ``up`` a word:
        ``ok``             rent the node and print the pod (the default);
        ``no_json``        rent the node, print text that is not JSON;
        ``fail``           rent nothing, exit code 1;
        ``fail_created``   exit code 1, but the pod exists;
        ``slow:<n>``       "still being processed"; the pod shows after ``n`` more ``ps``;
        ``late:<n>``       exit code 1; the pod shows after ``n`` more ``ps``;
        ``raise``          raise ``TimeoutError``, rent nothing;
        ``raise_created``  raise ``TimeoutError``, but the pod exists;
        ``other_name``     rent the node under another name and print that pod;
        ``took:<s>``       like ``ok``, after ``s`` seconds.
    """

    def __init__(self, clock: Optional[Ticker] = None, *,
                 pod_shows: Tuple[str, ...] = ('tier', 'price_per_hour', 'gpu_type', 'gpu_count')):
        self.clock = clock or Ticker()
        self.lock = threading.RLock()
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.pods: Dict[str, Dict[str, Any]] = {}
        self.hidden: List[List[Any]] = []          # [ps calls left, pod]
        self.calls: List[Tuple[str, ...]] = []
        self.timeouts: List[Tuple[str, float]] = []
        self.script: Dict[str, List[Any]] = {'ps': [], 'ls': [], 'up': [], 'rm': []}
        self.pod_shows = pod_shows
        self.explicit: Dict[str, Tuple[str, ...]] = {}   # pod id -> fields that a test set by hand
        self.serial = 0
        self.most = 0
        self.removed: List[str] = []

    # -- set-up ----------------------------------------------------------------

    def add(self, *offers: Dict[str, Any]) -> 'FakeLium':
        for row in offers:
            self.nodes[row['huid']] = dict(row)
        return self

    def pod(self, name: str, *, huid: str = 'other-node', **fields: Any) -> Dict[str, Any]:
        """Put a pod into the account that no ``up`` of this test made."""
        with self.lock:
            row = dict(gpu_type='B300', tier='secure', price_per_hour=40.0, gpu_count=8)
            row.update(fields)
            pod = self._make(name, huid, row)
            self.explicit[pod['id']] = tuple(fields)
            return pod

    def verbs(self, verb: str) -> List[Tuple[str, ...]]:
        return [call for call in self.calls if call[0] == verb]

    def names(self) -> List[str]:
        return sorted(pod['name'] for pod in self.pods.values())

    # -- the command line --------------------------------------------------------

    def __call__(self, *args: str, timeout: float = 180) -> Tuple[int, str, str]:
        with self.lock:
            self.calls.append(tuple(args))
            self.timeouts.append((args[0], timeout))
            script = self.script[args[0]]
            step = script.pop(0) if script else None
            if isinstance(step, BaseException):
                raise step
            if isinstance(step, tuple):
                return step
            return getattr(self, '_' + args[0])(args, step)

    def _make(self, name: str, huid: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        self.serial += 1
        pod = {
            'id': 'pod-%04d' % self.serial, 'name': name, 'huid': huid, 'status': 'RUNNING',
            'ip': BOX_HOST % self.serial, 'ports': {'22': 22},
        }
        pod.update(fields)
        self.pods[pod['id']] = pod
        self.most = max(self.most, len(self.pods))
        return pod

    def _shown(self, pod: Dict[str, Any]) -> Dict[str, Any]:
        shows = ('id', 'name', 'huid', 'status', 'ip', 'ports') + self.pod_shows + self.explicit.get(pod['id'], ())
        return {key: value for key, value in pod.items() if key in shows}

    def _ps(self, args: Tuple[str, ...], step: Any) -> Tuple[int, str, str]:
        assert args == ('ps', '--format', 'json'), args
        for entry in list(self.hidden):
            entry[0] -= 1
            if entry[0] <= 0:
                self.hidden.remove(entry)
                self.pods[entry[1]['id']] = entry[1]
                self.most = max(self.most, len(self.pods))
        return 0, json.dumps([self._shown(pod) for pod in self.pods.values()]), ''

    def _ls(self, args: Tuple[str, ...], step: Any) -> Tuple[int, str, str]:
        assert args[:2] == ('ls', '--gpu') and args[3:] == ('--format', 'json'), args
        rows = [row for row in self.nodes.values() if row.get('gpu_type', args[2]) == args[2]]
        return 0, json.dumps(rows), ''

    def _rent(self, huid: str, name: str, count: int) -> Optional[Dict[str, Any]]:
        node = self.nodes.get(huid)
        if node is None or node.get('available_gpu_count', 0) < count:
            return None
        node['available_gpu_count'] = 0
        return self._make(name, huid, dict(
            gpu_type=node.get('gpu_type'), tier=node.get('tier'), gpu_count=count,
            price_per_hour=round(node['price_per_gpu_hour'] * count, 4),
        ))

    def _up(self, args: Tuple[str, ...], step: Any) -> Tuple[int, str, str]:
        huid, name, count = args[1], args[args.index('--name') + 1], int(args[args.index('-c') + 1])
        mode, _, number = (step or 'ok').partition(':')
        if mode == 'took':
            self.clock.sleep(float(number))
            mode = 'ok'
        if mode in ('fail', 'raise'):
            if mode == 'raise':
                raise TimeoutError('the command line did not answer')
            return 1, '', 'Error: the node refused the rental'
        pod = self._rent(huid, 'someone-else' if mode == 'other_name' else name, count)
        if pod is None:
            return 1, '', 'Error: the node is not available'
        if mode in ('slow', 'late'):
            del self.pods[pod['id']]
            self.hidden.append([int(number), pod])
            text = 'an earlier request is still being processed' if mode == 'slow' else 'Error: gateway timeout'
            return 1, '', text
        if mode == 'raise_created':
            raise TimeoutError('the command line did not answer')
        if mode == 'fail_created':
            return 1, '', 'Error: lost the connection'
        if mode == 'no_json':
            return 0, 'Pod is up. Have fun.', ''
        return 0, json.dumps({'pod': self._shown(pod)}), ''

    def _rm(self, args: Tuple[str, ...], step: Any) -> Tuple[int, str, str]:
        assert len(args) == 3 and args[2] == '-y', args
        pod = self.pods.pop(args[1], None)
        if pod is None:
            return 1, '', 'Error: no such pod'
        self.removed.append(args[1])
        node = self.nodes.get(pod['huid'])
        if node is not None:
            node['available_gpu_count'] = node['gpu_count']
        return 0, 'removed', ''


class LockedStore(RawStore):
    """``RawStore`` with one lock around every call."""

    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.RLock()

    def get(self, key, *, max_bytes):
        with self.lock:
            return super().get(key, max_bytes=max_bytes)

    def head(self, key):
        with self.lock:
            return super().head(key)

    def digest(self, key):
        with self.lock:
            return super().digest(key)

    def put(self, key, data, *, if_none_match=False, if_match=None, who='box'):
        with self.lock:
            return super().put(key, data, if_none_match=if_none_match, if_match=if_match, who=who)

    def copy(self, src, dst, *, src_etag, who='box'):
        with self.lock:
            return super().copy(src, dst, src_etag=src_etag, who=who)

    def delete(self, key, *, who='box'):
        with self.lock:
            return super().delete(key, who=who)

    def list(self, prefix):
        with self.lock:
            return super().list(prefix)


class ObjectServer:
    """A store behind signed URLs. ``get_url`` and ``put_url`` sign one key
    for one method, as ``r2.R2Presigner`` does. Every other request gets 403.

    ``faults[(method, key)]`` holds what the next requests for that key do:

    * ``'500'``      answer 500 and change nothing;
    * ``'403'``      answer 403 and change nothing;
    * ``'cut'``      GET: send half of the bytes, then close the connection;
    * ``'corrupt'``  GET: send the bytes with the first one changed;
    * ``'norange'``  GET: ignore the ``Range`` header.
    """

    def __init__(self, store: Optional[RawStore] = None):
        self.store = store if store is not None else LockedStore()
        self.secret = os.urandom(16)
        self.faults: Dict[Tuple[str, str], List[str]] = {}
        self.log: List[Tuple[str, str, int]] = []
        self.lock = threading.Lock()
        self.before_put = None          # called with (key, data) before a PUT is stored
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def answer(self, method, key, status, body=b'', headers=()):
                with outer.lock:
                    outer.log.append((method, key, status))
                self.send_response(status)
                for name, value in headers:
                    self.send_header(name, value)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                if method != 'HEAD':
                    self.wfile.write(body)

            def key_of(self, method):
                token, _, key = self.path.lstrip('/').partition('/')
                return key if hmac.compare_digest(token, outer.token(method, key)) else None

            def fault(self, method, key):
                with outer.lock:
                    queue = outer.faults.get((method, key))
                    return queue.pop(0) if queue else None

            def do_GET(self):
                key = self.key_of('GET')
                if key is None:
                    return self.answer('GET', self.path, 403)
                fault = self.fault('GET', key)
                if fault in ('500', '403'):
                    return self.answer('GET', key, int(fault))
                try:
                    data = outer.store.get(key, max_bytes=1 << 62).data
                except NotFound:
                    return self.answer('GET', key, 404)
                first = 0
                asked = self.headers.get('Range')
                if asked and fault != 'norange':
                    first = int(asked.split('=')[1].split('-')[0])
                    head = [('Content-Range', 'bytes %d-%d/%d' % (first, len(data) - 1, len(data)))]
                    body, status = data[first:], 206
                else:
                    head, body, status = [], data, 200
                if fault == 'corrupt' and body:
                    body = bytes([body[0] ^ 0xFF]) + body[1:]
                if fault == 'cut':
                    with outer.lock:
                        outer.log.append(('GET', key, -1))
                    self.send_response(status)
                    for name, value in head:
                        self.send_header(name, value)
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body[:len(body) // 2])
                    self.wfile.flush()
                    self.close_connection = True
                    return None
                return self.answer('GET', key, status, body, head)

            def do_PUT(self):
                key = self.key_of('PUT')
                data = self.rfile.read(int(self.headers.get('Content-Length') or 0))
                if key is None or self.headers.get('Content-Type') != CONTENT_TYPE:
                    return self.answer('PUT', self.path, 403)
                fault = self.fault('PUT', key)
                if fault in ('500', '403'):
                    return self.answer('PUT', key, int(fault))
                if outer.before_put is not None:
                    outer.before_put(key, data)
                outer.store.put(key, data)
                return self.answer('PUT', key, 200)

        self.server = http.server.ThreadingHTTPServer(('localhost', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.02), daemon=True, name='object-server')
        self.thread.start()
        self.base = 'http://localhost:%d' % self.server.server_address[1]

    def token(self, method: str, key: str) -> str:
        return hmac.new(self.secret, ('%s\n%s' % (method, key)).encode(), hashlib.sha256).hexdigest()[:32]

    def get_url(self, key: str, expires_s: float = 3600.0) -> str:
        return '%s/%s/%s' % (self.base, self.token('GET', key), key)

    def put_url(self, key: str, expires_s: float = 3600.0) -> str:
        return '%s/%s/%s' % (self.base, self.token('PUT', key), key)

    def seen(self, method: str, key_part: str = '') -> List[Tuple[str, str, int]]:
        with self.lock:
            return [row for row in self.log if row[0] == method and key_part in row[1]]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def put_state(
    store: Any, layout: Layout, state_files: Mapping[str, bytes], *, fp32_master_digest: str,
    checkpoint_id: Optional[str] = None, fp32_entry: Optional[str] = None, parent: Optional[str] = None,
    batch_id: str = 'genesis', chunk: int = 1 << 20, meta: Optional[Mapping[str, Any]] = None,
    write: bool = True,
) -> StateManifest:
    """Cut ``{path: bytes}`` into parts, and write parts, index and manifest
    of that state under its canonical keys. -> the manifest."""
    parts: List[Tuple[ObjectRef, bytes]] = []
    entries: List[F.FileEntry] = []
    for path in sorted(state_files):
        data, role = state_files[path], F.GROUP_ROLE[path.split('/')[0]]
        names = []
        for offset in range(0, len(data), chunk):
            piece = data[offset:offset + chunk]
            ref = ObjectRef(F.slot_name(len(parts)), role, sha256_hex(piece), len(piece))
            parts.append((ref, piece))
            names.append(ref.name)
        entries.append(F.FileEntry(path, role, len(data), sha256_hex(data), tuple(names)))
    index = F.index_body(
        entries, checkpoint_id=checkpoint_id, fp32_entry=fp32_entry, fp32_master_digest=fp32_master_digest,
    )
    whole = dict(meta or {}, fp32_master_digest=fp32_master_digest)
    if checkpoint_id is not None:
        whole[F.META_CHECKPOINT] = checkpoint_id
    if fp32_entry is not None:
        whole[F.META_FP32_ENTRY] = fp32_entry
    manifest = F.manifest_of(
        parent=parent, batch_id=batch_id, parts=[ref for ref, _ in parts], index=index, meta=whole,
    )
    if write:
        for ref, piece in parts:
            store.put(layout.state_object(manifest.state_id, ref.name), piece)
        store.put(layout.state_object(manifest.state_id, F.INDEX_NAME), index)
        store.put(layout.state_manifest(manifest.state_id), manifest.body())
    return manifest


def model_files(seed: str, *, bf16: int = 3, fp32: int = 2, size: int = 300_000) -> Dict[str, bytes]:
    """The files of a small model state: weights, an fp32 master with its
    ``meta.json``, and a descriptor. The bytes depend on ``seed`` only."""
    def blob(name: str, count: int) -> bytes:
        out, block = bytearray(), hashlib.sha256(('%s:%s' % (seed, name)).encode()).digest()
        while len(out) < count:
            block = hashlib.sha256(block).digest()
            out += block
        return bytes(out[:count])

    files: Dict[str, bytes] = {}
    for number in range(1, bf16 + 1):
        name = 'model-%05d-of-%05d.safetensors' % (number, bf16)
        files['bf16/' + name] = blob(name, size + number)
    files['bf16/config.json'] = json.dumps({'seed': seed}).encode()
    weights = {path[5:]: sha256_hex(data) for path, data in files.items()}
    checkpoint = sha256_hex(json.dumps(weights, sort_keys=True).encode())
    entry = sha256_hex(('entry:' + seed).encode())
    master = sha256_hex(('master:' + seed).encode())
    shards = {}
    for number in range(1, fp32 + 1):
        name = 'fp32-master-%05d-of-%05d.safetensors' % (number, fp32)
        files['fp32/' + name] = blob(name, 2 * size + number)
        shards[name] = {'sha256': sha256_hex(files['fp32/' + name]), 'bytes': len(files['fp32/' + name])}
    files['fp32/meta.json'] = json.dumps({
        'version': 'v833f-fp32-head-master-v1', 'bf16_sha256': entry, 'master_digest': master, 'files': shards,
    }, sort_keys=True, indent=1).encode()
    files['aux/descriptor.json'] = json.dumps({'id': checkpoint, 'files': weights}).encode()
    return files


def model_ids(state_files: Mapping[str, bytes]) -> Tuple[str, str, str]:
    """(checkpoint id, fp32 entry, fp32 master digest) of ``model_files``."""
    meta = json.loads(state_files['fp32/meta.json'])
    return json.loads(state_files['aux/descriptor.json'])['id'], meta['bf16_sha256'], meta['master_digest']


def put_model(store: Any, layout: Layout, seed: str, **more: Any) -> Tuple[StateManifest, Dict[str, bytes]]:
    state_files = model_files(seed)
    checkpoint, entry, master = model_ids(state_files)
    manifest = put_state(
        store, layout, state_files, fp32_master_digest=master, checkpoint_id=checkpoint, fp32_entry=entry, **more,
    )
    return manifest, state_files
