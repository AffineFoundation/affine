"""The Lium provider: rent one whole node by its exact id, list, release.

This module starts no process. It gets the command-line helper of the
existing ops code as a callable (``cli('ps', '--format', 'json', timeout=90)``
returns ``(exit code, stdout, stderr)``), so the Lium key stays where it is
and the command forms are the ones that the ops code uses today:

    ps --format json
    ls --gpu <class> --format json
    up <node id> -c 8 --name <name> --verify-gpus --strict-gpus
       --no-volume-encryption --no-backup -y --json --timeout <s> --ttl <h>
    rm <pod id> -y

Rules of a rental:

* The name is a box name of this run (``guard.box_name``). Nothing else is
  ever rented or removed.
* Only a GPU class of the configured list is rented. The live default list
  of the PIT package is not read and not changed.
* The node is chosen from the offers and rented by its exact id (``huid``):
  a whole node with exactly ``gpu_count`` GPUs, the asked tier, a price at
  or below the ceiling, and host RAM at or above ``min_ram_gb``. An offer
  that does not say how much RAM it has is not rented. (The job on the box
  measures the RAM again before it does any work.)
* At most ``max_boxes`` boxes with a shadow name exist at one time, over all
  shadow runs of this account. The count is exact inside one process.
* Before the ``up`` command is sent, a failure means "no box"
  (``RentUnavailable``). After it was sent, every failure means "a box may
  exist" (``ProviderError``): the controller then looks for the box by name,
  and its sweep removes a box that shows up later.
* A pod is removed by its exact id, and only if its name is a box name of
  this run.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from . import BOX_PREFIX_SHADOW
from ..shadow import ShadowWriteError
from .guard import parse_box_name
from .interfaces import BoxInfo, ProviderError, RentUnavailable

Cli = Callable[..., Tuple[int, str, str]]

PROVIDER_NAME = 'lium'
TIERS = ('secure', 'spot')
# Where an offer can say its host RAM: (key, units per GB). The first key
# that holds a number is used. A dot goes into a nested object.
RAM_FIELDS: Tuple[Tuple[str, float], ...] = (
    ('ram_gb', 1.0), ('memory_gb', 1.0), ('ram_total_gb', 1.0), ('total_ram_gb', 1.0),
    ('host_ram_gb', 1.0), ('system_ram_gb', 1.0),
)
# A RAM figure outside these bounds is a unit error, not a machine.
RAM_SANE_GB = (1.0, 16384.0)
# Words in the error text of ``up`` that mean: the request is still running.
STILL_RUNNING = ('still being processed', 'earlier request')
GPU_KEYS = ('gpu', 'gpu_type', 'gpu_model', 'gpu_name', 'config')
PRICE_KEYS = ('price_per_hour', 'price_h', 'price_usd_h')
_SSH_PORT = re.compile(r'-p\s*(\d{1,5})\b')
_CONFIG_COUNT = re.compile(r'\s*(\d{1,3})\s*[xX\u00d7]')
_SSH_HOST = re.compile(r'root@([A-Za-z0-9.\-:]+)')
# Machines of other work. A node or a pod whose name or node id holds one of
# these words is never rented and never removed, whatever else is true of it.
NEVER_NAMES_DEFAULT: Tuple[str, ...] = (
    'brave-shark-ec', 'noble-wolf-26', 'calm-matrix-56', 'noble-fox-5c', 'gentle-wolf-f6', 'noble-shark-d6',
)


@dataclass(frozen=True)
class LiumConfig:
    run_id: str
    gpus: Tuple[str, ...]
    max_boxes: int = 8
    gpu_count: int = 8
    min_ram_gb: float = 300.0
    min_disk_gb: float = 0.0
    min_download_mbps: float = 0.0
    ram_fields: Tuple[Tuple[str, float], ...] = RAM_FIELDS
    never_names: Tuple[str, ...] = NEVER_NAMES_DEFAULT
    up_flags: Tuple[str, ...] = ('--verify-gpus', '--strict-gpus', '--no-volume-encryption', '--no-backup')
    nodes_per_rent: int = 1
    ls_timeout_s: float = 120.0
    ps_timeout_s: float = 90.0
    rm_timeout_s: float = 180.0
    adopt_wait_s: float = 60.0
    adopt_wait_running_s: float = 300.0
    adopt_poll_s: float = 10.0

    def __post_init__(self) -> None:
        if not self.gpus or any(not isinstance(gpu, str) or not gpu for gpu in self.gpus):
            raise ValueError('gpus must name at least one GPU class')
        if self.max_boxes < 1 or self.gpu_count < 1 or self.nodes_per_rent < 1:
            raise ValueError('max_boxes, gpu_count and nodes_per_rent must be above zero')
        if self.min_ram_gb < 0 or self.min_disk_gb < 0 or self.min_download_mbps < 0:
            raise ValueError('a minimum cannot be below zero')
        if self.min_ram_gb > 0 and not self.ram_fields:
            raise ValueError('a RAM minimum needs the name of a RAM field')
        for field in self.ram_fields:
            if len(field) != 2 or not isinstance(field[0], str) or not field[0] or not field[1] > 0:
                raise ValueError('a RAM field is (key, units per GB)')


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _flat(text: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def gpu_class(row: Mapping[str, Any], classes: Sequence[str]) -> str:
    """The GPU class of a pod or an offer, or '' when the row does not say."""
    for key in GPU_KEYS:
        text = row.get(key)
        if not isinstance(text, str) or not text:
            continue
        flat = _flat(text)
        # The longest name first, so that a short name cannot win inside a long one.
        for name in sorted(classes, key=len, reverse=True):
            token = _flat(name)
            if token in flat or (token.startswith('RTX') and token[3:] in flat):
                return name
        return ''
    return ''


def ram_gb(row: Mapping[str, Any], fields: Sequence[Tuple[str, float]]) -> Optional[float]:
    """The host RAM of an offer in GB, or None if the offer does not say."""
    for key, per_gb in fields:
        value: Any = row
        for part in key.split('.'):
            value = value.get(part) if isinstance(value, Mapping) else None
        number = _number(value)
        if number is None:
            continue
        number /= per_gb
        return number if RAM_SANE_GB[0] <= number <= RAM_SANE_GB[1] else None
    return None


def offer_price_usd_h(offer: Mapping[str, Any], gpu_count: int) -> Optional[float]:
    """The price of the whole node for one hour. If the offer gives two
    figures, the higher one counts."""
    figures = []
    each = _number(offer.get('price_per_gpu_hour'))
    if each is not None:
        figures.append(each * gpu_count)
    whole = _number(offer.get('price_per_hour'))
    if whole is not None:
        figures.append(whole)
    if not figures or min(figures) < 0:
        return None
    return round(max(figures), 4)


def pod_price_usd_h(pod: Mapping[str, Any]) -> Optional[float]:
    for key in PRICE_KEYS:
        number = _number(pod.get(key))
        if number is not None and number >= 0:
            return number
    return None


def ssh_target(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """What the existing ssh helper needs for one pod, or None when the pod
    has no address yet."""
    host = row.get('ip') or row.get('ssh_ip')
    port = row.get('ssh_port') or row.get('port')
    if not port and isinstance(row.get('ports'), Mapping):
        port = row['ports'].get('22')
    command = row.get('ssh_command')
    if isinstance(command, str):
        found_port, found_host = _SSH_PORT.search(command), _SSH_HOST.search(command)
        port = port or (found_port.group(1) if found_port else None)
        host = host or (found_host.group(1) if found_host else None)
    number = _number(port)
    if not isinstance(host, str) or not host or number is None or not 0 < number < 65536:
        return None
    return {
        'id': str(row.get('id') or ''), 'ip': host, 'port': int(number),
        'huid': row.get('huid'), 'name': row.get('name'),
    }


class LiumProvider:
    """``Provider`` on the Lium command line. Safe for many threads."""

    name = PROVIDER_NAME

    def __init__(
        self, cli: Cli, config: LiumConfig, *,
        never: Optional[Callable[[Mapping[str, Any]], bool]] = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._cli = cli
        self.cfg = config
        self._never = never
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock = threading.Lock()
        self._renting: Dict[str, str] = {}             # box name -> node of an ``up`` in flight
        self._rented: Dict[str, Dict[str, Any]] = {}   # pod id -> the offer that this process took
        self._avoid: Set[str] = set()                  # nodes whose ``up`` gave no pod in this process

    # -- the command line ----------------------------------------------------

    def _json(self, what: str, *args: str, timeout: float) -> List[Dict[str, Any]]:
        try:
            code, out, err = self._cli(*args, timeout=timeout)
        except Exception as exc:
            raise ProviderError('%s: %s' % (what, type(exc).__name__)) from None
        if code != 0:
            raise ProviderError('%s failed (%s): %s' % (what, code, (err or out or '')[-200:]))
        try:
            rows = json.loads(out or '')
        except ValueError:
            raise ProviderError('%s: the answer is not JSON' % what) from None
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ProviderError('%s: the answer is not a list of objects' % what)
        return rows

    def pods(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        return self._json('ps', 'ps', '--format', 'json', timeout=timeout or self.cfg.ps_timeout_s)

    def offers(self, gpu: str) -> List[Dict[str, Any]]:
        return self._json('ls', 'ls', '--gpu', gpu, '--format', 'json', timeout=self.cfg.ls_timeout_s)

    # -- what a row means ----------------------------------------------------

    def _forbidden(self, row: Mapping[str, Any]) -> bool:
        for key in ('name', 'huid'):
            text = row.get(key)
            if isinstance(text, str) and any(word and word in text for word in self.cfg.never_names):
                return True
        return bool(self._never is not None and self._never(row))

    def info(self, row: Mapping[str, Any]) -> BoxInfo:
        """One pod as the controller sees it.

        A field that the pod does not show is taken from the offer that this
        process rented, else it stays empty. The controller refuses a box
        whose class, GPU count, tier or price it cannot see.
        """
        box_id = str(row.get('id') or '')
        known = self._rented.get(box_id, {})
        count = _number(row.get('gpu_count'))
        if count is None and isinstance(row.get('config'), str):
            # "8xRTX5090": the count stands before the class.
            found = _CONFIG_COUNT.match(row['config'])
            count = float(found.group(1)) if found else None
        price = pod_price_usd_h(row)
        tier = row.get('tier')
        name = row.get('name')
        return BoxInfo(
            box_id=box_id,
            name=name if isinstance(name, str) else '',
            provider=self.name,
            gpu=gpu_class(row, self.cfg.gpus) or known.get('gpu', ''),
            gpu_count=int(count) if count is not None and count > 0 else int(known.get('gpu_count', 0)),
            tier=tier if tier in TIERS else known.get('tier', ''),
            price_usd_h=price if price is not None else float(known.get('price_usd_h', -1.0)),
        )

    def why_not(self, offer: Mapping[str, Any], *, gpu: str, tier: str, max_price_usd_h: float) -> Optional[str]:
        """The reason that an offer is not rented, or None for a good offer."""
        cfg = self.cfg
        huid = offer.get('huid')
        if not isinstance(huid, str) or not huid:
            return 'no_id'
        if huid in self._avoid or huid in self._renting.values():
            return 'tried_or_in_use'
        if self._forbidden(offer):
            return 'never_list'
        if offer.get('tier') != tier:
            return 'tier'
        if any(isinstance(offer.get(key), str) and offer.get(key) for key in GPU_KEYS) and gpu_class(offer, cfg.gpus) != gpu:
            # The offer names a class and it is not the asked one.
            return 'gpu_class'
        total = _number(offer.get('gpu_count'))
        free = _number(offer.get('available_gpu_count'))
        if total != cfg.gpu_count or free is None or free < cfg.gpu_count:
            return 'not_a_free_whole_node'
        if offer.get('gpu_power_limited') is True:
            return 'power_limited'
        price = offer_price_usd_h(offer, cfg.gpu_count)
        if price is None or price > max_price_usd_h:
            return 'price'
        if cfg.min_ram_gb > 0:
            ram = ram_gb(offer, cfg.ram_fields)
            if ram is None:
                return 'ram_unknown'
            if ram < cfg.min_ram_gb:
                return 'ram'
        if cfg.min_disk_gb > 0 and (_number(offer.get('disk_gb')) or 0.0) < cfg.min_disk_gb:
            return 'disk'
        if cfg.min_download_mbps > 0 and (_number(offer.get('download_mbps')) or 0.0) < cfg.min_download_mbps:
            return 'download'
        return None

    def choose(self, offers: Sequence[Mapping[str, Any]], *, gpu: str, tier: str, max_price_usd_h: float
               ) -> List[Mapping[str, Any]]:
        """The good offers, cheapest first; a faster download breaks a tie."""
        good = [
            offer for offer in offers
            if self.why_not(offer, gpu=gpu, tier=tier, max_price_usd_h=max_price_usd_h) is None
        ]
        return sorted(good, key=lambda offer: (
            offer_price_usd_h(offer, self.cfg.gpu_count), -(_number(offer.get('download_mbps')) or 0.0),
            offer['huid'],
        ))

    def survey(self, *, gpu: str, tier: str, max_price_usd_h: float) -> Dict[str, Any]:
        """For the operator: how many offers there are, how many are good,
        why the others are not, and the keys that an offer has."""
        offers = self.offers(gpu)
        reasons = Counter(
            self.why_not(offer, gpu=gpu, tier=tier, max_price_usd_h=max_price_usd_h) or 'good' for offer in offers
        )
        good = self.choose(offers, gpu=gpu, tier=tier, max_price_usd_h=max_price_usd_h)
        return {
            'gpu': gpu, 'tier': tier, 'offers': len(offers), 'reasons': dict(sorted(reasons.items())),
            'keys': sorted({key for offer in offers for key in offer}),
            'cheapest_usd_h': offer_price_usd_h(good[0], self.cfg.gpu_count) if good else None,
        }

    # -- Provider ------------------------------------------------------------

    def list(self) -> List[BoxInfo]:
        return [self.info(row) for row in self.pods()]

    def shadow_boxes(self, pods: Sequence[Mapping[str, Any]]) -> int:
        """Boxes that count against the cap: every pod with a shadow name, of
        any run, plus the rentals of this process that no pod shows yet."""
        names = {row.get('name') for row in pods}
        listed = sum(1 for name in names if isinstance(name, str) and name.startswith(BOX_PREFIX_SHADOW))
        return listed + sum(1 for name in self._renting if name not in names)

    def rent(
        self, *, name: str, gpu: str, tier: str, gpu_count: int, max_price_usd_h: float,
        ttl_s: float, timeout_s: float,
    ) -> BoxInfo:
        cfg = self.cfg
        if parse_box_name(cfg.run_id, name) is None:
            raise ShadowWriteError('shadow rents no box with a name that this run did not make')
        if any(word and word in name for word in cfg.never_names):
            raise ShadowWriteError('shadow rents no box with a protected name')
        if gpu not in cfg.gpus:
            raise RentUnavailable('GPU class %s is not in the list of this run' % gpu)
        if tier not in TIERS or gpu_count != cfg.gpu_count or ttl_s <= 0 or timeout_s <= 0:
            raise RentUnavailable('no such offer: tier, GPU count, TTL or time limit')
        end = self._monotonic() + timeout_s
        with self._lock:
            if name in self._renting:
                # Another call rents this name right now. Its box may exist.
                raise ProviderError('a rental of %s is in flight' % name)
            try:
                pods = self.pods()
                for row in pods:
                    if row.get('name') == name and row.get('id'):
                        return self.info(row)
                if self.shadow_boxes(pods) >= cfg.max_boxes:
                    raise RentUnavailable('the cap of %d shadow boxes is reached' % cfg.max_boxes)
                nodes = self.choose(self.offers(gpu), gpu=gpu, tier=tier, max_price_usd_h=max_price_usd_h)
            except RentUnavailable:
                raise
            except ProviderError as exc:
                # Nothing was rented yet, so there is no box.
                raise RentUnavailable('before the rental: %s' % exc) from None
            if not nodes:
                raise RentUnavailable('no %s offer for %dx %s within the limits' % (tier, gpu_count, gpu))
            self._renting[name] = nodes[0]['huid']
        try:
            return self._up(name, nodes[:cfg.nodes_per_rent], gpu=gpu, tier=tier, ttl_s=ttl_s, end=end)
        finally:
            with self._lock:
                self._renting.pop(name, None)

    def _up(self, name: str, nodes: Sequence[Mapping[str, Any]], *, gpu: str, tier: str, ttl_s: float,
            end: float) -> BoxInfo:
        cfg = self.cfg
        last = None
        for node in nodes:
            left = end - self._monotonic()
            if left < 90:
                break
            huid = node['huid']
            with self._lock:
                self._renting[name] = huid
            offer = {
                'gpu': gpu, 'tier': tier, 'gpu_count': cfg.gpu_count, 'huid': huid,
                'price_usd_h': offer_price_usd_h(node, cfg.gpu_count),
            }
            # The command gets a little less time than this call has left.
            wait = int(left) - 40
            args = [
                'up', huid, '-c', str(cfg.gpu_count), '--name', name, *cfg.up_flags, '-y', '--json',
                '--timeout', str(wait), '--ttl', '%dh' % max(1, math.ceil(ttl_s / 3600.0)),
            ]
            try:
                code, out, err = self._cli(*args, timeout=wait + 20)
            except Exception as exc:
                code, out, err = -1, '', type(exc).__name__
            pod = self._pod_of_up(out) if code == 0 else None
            if pod is None or pod.get('name') != name or not pod.get('id'):
                # The answer does not show the pod. It can exist anyway.
                text = '%s %s' % (err or '', out or '')
                slow = any(word in text for word in STILL_RUNNING)
                pod = self._find(name, end, cfg.adopt_wait_running_s if slow else cfg.adopt_wait_s)
                if pod is None:
                    with self._lock:
                        self._avoid.add(huid)
                    last = 'up on %s gave no pod (%s): %s' % (huid, code, text.strip()[-200:])
                    continue
            with self._lock:
                self._rented[str(pod['id'])] = offer
            return self.info(pod)
        if last is None:
            # ``up`` was never sent: there is no box.
            raise RentUnavailable('no time left to rent')
        raise ProviderError(last)

    @staticmethod
    def _pod_of_up(out: str) -> Optional[Dict[str, Any]]:
        try:
            raw = json.loads(out)
        except ValueError:
            return None
        pod = raw.get('pod', raw) if isinstance(raw, dict) else None
        return pod if isinstance(pod, dict) else None

    def _find(self, name: str, end: float, wait_s: float) -> Optional[Dict[str, Any]]:
        """Look for a pod with this exact name until ``wait_s`` passed or the
        time of the rental is over. At least one look."""
        end = min(end, self._monotonic() + wait_s)
        while True:
            try:
                for row in self.pods(timeout=max(5.0, min(self.cfg.ps_timeout_s, end - self._monotonic()))):
                    if row.get('name') == name and row.get('id'):
                        return row
            except ProviderError:
                pass
            if self._monotonic() + self.cfg.adopt_poll_s >= end:
                return None
            self._sleep(self.cfg.adopt_poll_s)

    def destroy(self, box_id: str) -> None:
        row = self._row(box_id)
        if row is None:
            return
        if parse_box_name(self.cfg.run_id, row.get('name')) is None or self._forbidden(row):
            # The controller checks the name before it calls. A caller that
            # gets here has a bug: stop, and remove nothing.
            raise ShadowWriteError('shadow removes no pod that this run did not make')
        try:
            code, out, err = self._cli('rm', box_id, '-y', timeout=self.cfg.rm_timeout_s)
        except Exception as exc:
            raise ProviderError('rm: %s' % type(exc).__name__) from None
        if code != 0:
            raise ProviderError('rm failed (%s): %s' % (code, (err or out or '')[-200:]))
        with self._lock:
            self._rented.pop(box_id, None)

    def _row(self, box_id: str) -> Optional[Dict[str, Any]]:
        if not isinstance(box_id, str) or not box_id:
            return None
        for row in self.pods():
            if row.get('id') == box_id:
                return row
        return None

    # -- for the box agent ---------------------------------------------------

    def ssh_target(self, box_id: str) -> Optional[Dict[str, Any]]:
        """The address of one pod of this run, or None if it has none yet."""
        row = self._row(box_id)
        if row is None or parse_box_name(self.cfg.run_id, row.get('name')) is None:
            return None
        return ssh_target(row)


__all__ = [
    'Cli', 'LiumConfig', 'LiumProvider', 'NEVER_NAMES_DEFAULT', 'PROVIDER_NAME', 'RAM_FIELDS', 'TIERS',
    'gpu_class', 'offer_price_usd_h', 'pod_price_usd_h', 'ram_gb', 'ssh_target',
]
