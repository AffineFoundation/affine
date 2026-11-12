"""Fakes that more than one test file uses. This file holds no test.

``FakeLium``   the Lium command line, in memory. It is called like the ops
               helper: ``cli('ps', '--format', 'json', timeout=90)`` returns
               ``(exit code, stdout, stderr)``.
``Ticker``     a clock that moves only when somebody sleeps.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional, Tuple

BOX_HOST = 'box-%d.test'


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
