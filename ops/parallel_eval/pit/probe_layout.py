"""Probe serving layout: how vLLM serves the gate probe on one box.

A layout holds the settings that change speed. None of them may change scores:

``gpus_per_replica``        GPUs that one vLLM server (a "replica") uses. 1 is
                            one server per GPU. 2 is tensor parallel 2 (TP2):
                            one server splits the model and its KV cache (the
                            per-token attention state) over two GPUs.
``replicas``                servers on the box. ``None`` uses every GPU.
``gpu_groups``              the GPU ids of each replica. ``None`` takes the
                            GPUs in order (0,1 then 2,3 and so on).
``vllm_version``            the vLLM build that serves.
``vllm_venv``               the Python environment that holds it on the box.
``vllm_requirements``       a lock file (``name==version`` lines) to install
                            that environment from.
``install_vllm``            install the build when the box does not have it.
``runtime_sha256``          digest of every package in the environment. When
                            set, another package set does not serve.
``prefix_caching``          vLLM reuses the KV cache of a prompt start it has
                            already seen.
``max_num_seqs``            sequences one replica decodes at the same time.
``gpu_memory_utilization``  share of each GPU's memory that vLLM may take.
``task_queue``              ``shared``: one queue feeds all replicas, and each
                            request goes to the replica with the fewest
                            requests in flight.
``concurrency_per_replica`` requests the client keeps in flight per replica.
                            ``None`` is ``max_num_seqs``.
``serve_timeout_s``         longest wait for all replicas to answer.
``probe_stall_s``           longest time without a finished sample.
``probe_wall_s``            longest probe. After it the run is closed as
                            partial, and a partial probe never decides.

The gate constants are not settings and no layout can change them: bf16
weights, a KV cache in the model's own format (bf16), a 69632 token window,
loopback only, no image or video input.

There is no FP8 in the probe: no FP8 weights and no FP8 KV cache (Jacob,
2026-11-11). FP8 changes what a server computes, so it is not a speed
setting. No layout has a setting for it, and ``probe_serve.py`` does not
accept a server that runs with an FP8 argument.

This file imports nothing from the repository. ``probe_serve.py`` copies it to
the box and imports it there.
"""
from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

VERSION = 'pit-probe-layout-v1'

BOX_ROOT = '/root/affine-env-probe'
BASE_PORT = 18555
SERVED_MODEL = 'probe'
GATE_DTYPE = 'bfloat16'
GATE_MAX_MODEL_LEN = 69632
LIMIT_MM = '{"image":0,"video":0}'
# vLLM names for the number format of the KV cache and of the weights. The probe keeps both in bf16.
NO_FP8_SETTINGS = ('kv_cache_dtype', 'quantization')

LIVE_VLLM_VERSION = '0.32.0'
LIVE_MAX_NUM_SEQS = 512
LIVE_GMU = 0.90

TASK_QUEUES = ('shared',)
# pe_probe_plan.build_probe_plan never lets the client hold more per replica.
MAX_CLIENT_CONCURRENCY = 512
# podctl.py tracks serve.0.pid .. serve.15.pid.
MAX_REPLICAS = 16
INSTALL_TIMEOUT_S = 1800
GPU_WAIT_S = 300

_VERSION_RE = re.compile(r'^\d+\.\d+\.\d+(\.post\d+)?$')
_HEX64_RE = re.compile(r'^[0-9a-f]{64}$')


class LayoutError(ValueError):
    """A layout that must not serve."""


def _is_int(value: Any) -> bool:
    return type(value) is int


@dataclass(frozen=True)
class ProbeLayout:
    name: str = 'live'
    gpus_per_replica: int = 1
    replicas: Optional[int] = None
    gpu_groups: Optional[Tuple[Tuple[int, ...], ...]] = None
    vllm_version: str = LIVE_VLLM_VERSION
    vllm_venv: Optional[str] = None
    vllm_requirements: Optional[str] = None
    install_vllm: bool = False
    runtime_sha256: Optional[str] = None
    prefix_caching: bool = True
    max_num_seqs: int = LIVE_MAX_NUM_SEQS
    gpu_memory_utilization: float = LIVE_GMU
    task_queue: str = 'shared'
    concurrency_per_replica: Optional[int] = None
    serve_timeout_s: int = 900
    probe_stall_s: int = 1200
    probe_wall_s: int = 5400

    def check(self) -> 'ProbeLayout':
        bad: List[str] = []
        if not isinstance(self.name, str) or not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,63}', self.name):
            bad.append('name must be lower-case letters, digits, dot, dash or underscore')
        if not _is_int(self.gpus_per_replica) or self.gpus_per_replica < 1:
            bad.append('gpus_per_replica must be a whole number, 1 or more')
        if self.replicas is not None and (not _is_int(self.replicas) or not 1 <= self.replicas <= MAX_REPLICAS):
            bad.append('replicas must be 1..%d or null' % MAX_REPLICAS)
        if self.gpu_groups is not None and not bad:
            bad.extend(self._check_groups())
        if not isinstance(self.vllm_version, str) or not _VERSION_RE.match(self.vllm_version):
            bad.append('vllm_version must look like 0.33.0')
        for field in ('vllm_venv', 'vllm_requirements'):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value.startswith('/') or any(c.isspace() for c in value)):
                bad.append('%s must be an absolute path without spaces, or null' % field)
        if self.runtime_sha256 is not None and not (isinstance(self.runtime_sha256, str) and _HEX64_RE.match(self.runtime_sha256)):
            bad.append('runtime_sha256 must be 64 lower-case hex characters or null')
        for field in ('install_vllm', 'prefix_caching'):
            if type(getattr(self, field)) is not bool:
                bad.append('%s must be true or false' % field)
        if not _is_int(self.max_num_seqs) or not 1 <= self.max_num_seqs <= MAX_CLIENT_CONCURRENCY:
            bad.append('max_num_seqs must be 1..%d' % MAX_CLIENT_CONCURRENCY)
        gmu = self.gpu_memory_utilization
        if type(gmu) not in (int, float) or not 0.0 < float(gmu) <= 1.0:
            bad.append('gpu_memory_utilization must be above 0 and at most 1')
        if self.task_queue not in TASK_QUEUES:
            bad.append(
                'task_queue %r is not available: the probe worker has one queue for all replicas (%s)'
                % (self.task_queue, ', '.join(TASK_QUEUES))
            )
        cpr = self.concurrency_per_replica
        if cpr is not None and (not _is_int(cpr) or not 1 <= cpr <= MAX_CLIENT_CONCURRENCY):
            bad.append('concurrency_per_replica must be 1..%d or null' % MAX_CLIENT_CONCURRENCY)
        for field in ('serve_timeout_s', 'probe_stall_s', 'probe_wall_s'):
            value = getattr(self, field)
            if not _is_int(value) or value < 1:
                bad.append('%s must be a whole number of seconds, 1 or more' % field)
        if bad:
            raise LayoutError('probe layout %r: %s' % (self.name, '; '.join(bad)))
        return self

    def _check_groups(self) -> List[str]:
        groups = self.gpu_groups or ()
        if not groups or len(groups) > MAX_REPLICAS:
            return ['gpu_groups must hold 1..%d groups' % MAX_REPLICAS]
        seen: set = set()
        for group in groups:
            if len(group) != self.gpus_per_replica:
                return ['each gpu_groups entry must hold gpus_per_replica (%d) GPU ids' % self.gpus_per_replica]
            for gpu in group:
                if not _is_int(gpu) or gpu < 0:
                    return ['gpu_groups must hold GPU ids (0, 1, ...)']
                if gpu in seen:
                    return ['gpu_groups uses GPU %d twice' % gpu]
                seen.add(gpu)
        if self.replicas is not None and self.replicas != len(groups):
            return ['replicas (%d) must equal the number of gpu_groups (%d)' % (self.replicas, len(groups))]
        return []

    def launcher(self) -> str:
        """``podctl`` when the pinned live serve.sh can start this layout, else ``direct``."""
        live_code = (
            self.gpus_per_replica == 1 and self.replicas is None and self.gpu_groups is None
            and self.prefix_caching and self.vllm_version == LIVE_VLLM_VERSION
            and self.vllm_venv is None
        )
        return 'podctl' if live_code else 'direct'

    def concurrency(self) -> int:
        return min(MAX_CLIENT_CONCURRENCY, self.concurrency_per_replica or self.max_num_seqs)

    def groups(self, gpu_count: int) -> Tuple[Tuple[int, ...], ...]:
        """GPU ids of each replica on a box with ``gpu_count`` GPUs."""
        if not _is_int(gpu_count) or gpu_count < 1:
            raise LayoutError('probe layout %r: the box reports no GPU' % self.name)
        if self.gpu_groups is not None:
            for group in self.gpu_groups:
                for gpu in group:
                    if gpu >= gpu_count:
                        raise LayoutError('probe layout %r: GPU %d is not on a %d GPU box' % (self.name, gpu, gpu_count))
            return tuple(tuple(group) for group in self.gpu_groups)
        per = self.gpus_per_replica
        if self.replicas is None:
            if gpu_count % per:
                raise LayoutError('probe layout %r: %d GPUs do not split into replicas of %d' % (self.name, gpu_count, per))
            count = gpu_count // per
        else:
            count = self.replicas
        if count < 1 or count * per > gpu_count or count > MAX_REPLICAS:
            raise LayoutError('probe layout %r: %d replicas of %d GPUs do not fit %d GPUs' % (self.name, count, per, gpu_count))
        return tuple(tuple(range(i * per, (i + 1) * per)) for i in range(count))

    def gmu_text(self) -> str:
        return '%g' % float(self.gpu_memory_utilization)

    def vllm_args(self, port: int) -> List[str]:
        """Arguments after ``vllm serve <checkpoint dir>``; the shared flags follow the order of the live serve.sh."""
        args = [
            '--host', '127.0.0.1', '--port', str(int(port)), '--served-model-name', SERVED_MODEL,
            '--dtype', GATE_DTYPE, '--max-model-len', str(GATE_MAX_MODEL_LEN),
            '--gpu-memory-utilization', self.gmu_text(),
            '--enable-prefix-caching' if self.prefix_caching else '--no-enable-prefix-caching',
            '--limit-mm-per-prompt', LIMIT_MM,
            '--max-num-seqs', str(self.max_num_seqs),
        ]
        if self.gpus_per_replica > 1:
            args += ['--tensor-parallel-size', str(self.gpus_per_replica)]
        return args

    def serving(self) -> Dict[str, Any]:
        """The settings that decide what the servers compute and how they are fed."""
        doc = self.as_dict()
        for key in ('name', 'install_vllm', 'vllm_requirements', 'serve_timeout_s', 'probe_stall_s', 'probe_wall_s'):
            doc.pop(key)
        return doc

    def is_live(self) -> bool:
        """Every setting is the live one; only the name may differ."""
        return dataclasses.replace(self, name=LIVE.name) == LIVE

    def as_dict(self) -> Dict[str, Any]:
        doc = dataclasses.asdict(self)
        if self.gpu_groups is not None:
            doc['gpu_groups'] = [list(group) for group in self.gpu_groups]
        doc['gpu_memory_utilization'] = float(self.gpu_memory_utilization)
        return doc

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, base: Optional['ProbeLayout'] = None) -> 'ProbeLayout':
        if not isinstance(raw, Mapping):
            raise LayoutError('a probe layout must be a mapping, got %s' % type(raw).__name__)
        fields = {f.name for f in dataclasses.fields(cls)}
        fp8 = sorted(set(raw) & set(NO_FP8_SETTINGS))
        if fp8:
            raise LayoutError('%s is not a probe layout setting: the probe has no FP8, its weights and its KV cache '
                              'stay bf16' % ', '.join(fp8))
        unknown = sorted(set(raw) - fields)
        if unknown:
            raise LayoutError('unknown probe layout setting: %s' % ', '.join(unknown))
        values = dict(raw)
        groups = values.get('gpu_groups')
        if groups is not None:
            if not isinstance(groups, (list, tuple)) or not all(isinstance(g, (list, tuple)) for g in groups):
                raise LayoutError('gpu_groups must be a list of lists of GPU ids')
            values['gpu_groups'] = tuple(tuple(g) for g in groups)
        layout = dataclasses.replace(base, **values) if base is not None else cls(**values)
        return layout.check()


LIVE = ProbeLayout(name='live')
# Real 8x RTX 5090 benches, 2026-11-11: with one GPU per replica only about 296k KV tokens fit per GPU, so the KV cache
# is the limit. TP2 gives a replica the KV room of two GPUs; TP4 was slower (PCIe). max_num_seqs is the middle of the
# measured 96..128 range.
RTX5090_TP2 = ProbeLayout(
    name='rtx5090-tp2', gpus_per_replica=2, vllm_version='0.33.0', install_vllm=True, prefix_caching=False,
    max_num_seqs=112, gpu_memory_utilization=0.94, serve_timeout_s=2400, probe_wall_s=7200,
).check()
# The measured baseline: 2.65 h for 972 tasks x 8 rollouts. It runs the live serve.sh and the live vLLM build.
RTX5090_1GPU = ProbeLayout(
    name='rtx5090-1gpu', max_num_seqs=64, gpu_memory_utilization=0.90, serve_timeout_s=2400, probe_wall_s=12600,
).check()
PRESETS: Dict[str, ProbeLayout] = {layout.name: layout for layout in (LIVE, RTX5090_TP2, RTX5090_1GPU)}
# Shadow mode, per GPU class: the first layout serves; the next ones are tried in order when it cannot start.
SHADOW_DEFAULTS: Dict[str, Tuple[str, ...]] = {'RTX5090': ('rtx5090-tp2', 'rtx5090-1gpu')}


def layout_from(spec: Any) -> ProbeLayout:
    """A preset name, or a mapping of settings with an optional ``preset`` to start from."""
    if isinstance(spec, ProbeLayout):
        return spec.check()
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise LayoutError('unknown probe layout preset %r (have: %s)' % (spec, ', '.join(sorted(PRESETS))))
        return PRESETS[spec]
    if not isinstance(spec, Mapping):
        raise LayoutError('a probe layout is a preset name or a mapping, got %s' % type(spec).__name__)
    values = dict(spec)
    preset = values.pop('preset', None)
    if 'fallback' in values:
        raise LayoutError('fallback belongs to the plan, not to one layout')
    if preset is None:
        return ProbeLayout.from_mapping(values)
    if not isinstance(preset, str):
        raise LayoutError('preset must be a preset name')
    base = layout_from(preset)
    if values and 'name' not in values:
        values['name'] = '%s-custom' % base.name
    return ProbeLayout.from_mapping(values, base=base)


@dataclass(frozen=True)
class ProbePlan:
    """The layout that serves, then the layouts to try in order when it cannot start."""

    layouts: Tuple[ProbeLayout, ...]

    def __post_init__(self) -> None:
        if not self.layouts:
            raise LayoutError('a probe plan needs one layout or more')
        names = [layout.name for layout in self.layouts]
        if len(set(names)) != len(names):
            raise LayoutError('probe plan layouts need different names: %s' % names)

    @property
    def primary(self) -> ProbeLayout:
        return self.layouts[0]

    def is_live(self) -> bool:
        """True when the untouched live serve path is this plan."""
        return len(self.layouts) == 1 and self.primary.is_live()

    def eval_timeout_s(self) -> int:
        """Upper bound for serve plus probe on the box, for the host's wait."""
        install = INSTALL_TIMEOUT_S if any(l.install_vllm and l.launcher() == 'direct' for l in self.layouts) else 0
        serve = sum(l.serve_timeout_s for l in self.layouts)
        return GPU_WAIT_S + install + serve + max(l.probe_wall_s for l in self.layouts) + 600

    def as_dict(self) -> Dict[str, Any]:
        return dict(version=VERSION, layouts=[layout.as_dict() for layout in self.layouts])

    @classmethod
    def from_doc(cls, doc: Any) -> 'ProbePlan':
        if not isinstance(doc, Mapping) or doc.get('version') != VERSION or not isinstance(doc.get('layouts'), list):
            raise LayoutError('not a %s document' % VERSION)
        return cls(tuple(ProbeLayout.from_mapping(raw) for raw in doc['layouts']))


def plan_from(spec: Any) -> ProbePlan:
    """One layout spec, a list of them (first serves), or a mapping with a ``fallback`` key.

    ``fallback`` is a layout spec, a list of specs, or ``null`` for no fallback.
    """
    if isinstance(spec, ProbePlan):
        return spec
    if isinstance(spec, (list, tuple)):
        return ProbePlan(tuple(layout_from(one) for one in spec))
    if isinstance(spec, Mapping) and 'fallback' in spec:
        values = dict(spec)
        fallback = values.pop('fallback')
        more: Sequence[Any] = [] if fallback in (None, False) else (
            list(fallback) if isinstance(fallback, (list, tuple)) else [fallback]
        )
        return ProbePlan((layout_from(values),) + tuple(layout_from(one) for one in more))
    return ProbePlan((layout_from(spec),))


def gpu_class_of(gpu: Any) -> Optional[str]:
    text = str(gpu or '').upper().replace(' ', '')
    for name in ('RTX5090', 'B300', 'B200', 'H200', 'H100'):
        if name in text:
            return name
    return 'RTX5090' if '5090' in text else None


def shadow_plan(gpu: Any, overrides: Optional[Mapping[str, Any]] = None) -> ProbePlan:
    """The plan a shadow box of this GPU class uses.

    ``overrides`` maps a GPU class (or ``default``) to a plan spec. Without an
    entry, RTX5090 serves TP2 with the one-GPU-per-replica fallback and every
    other class serves the live layout.
    """
    overrides = overrides or {}
    cls = gpu_class_of(gpu)
    if cls is not None and cls in overrides:
        return plan_from(overrides[cls])
    if 'default' in overrides:
        return plan_from(overrides['default'])
    return plan_from(list(SHADOW_DEFAULTS.get(cls or '', ('live',))))


def dumps(plan: ProbePlan) -> str:
    return json.dumps(plan.as_dict(), sort_keys=True) + '\n'
