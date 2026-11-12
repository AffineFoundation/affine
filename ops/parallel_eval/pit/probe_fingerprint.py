"""One fingerprint for the hardware class and the layout of a probe.

Terms:

* baseline: the HEAD rollouts of a round. A candidate is judged against them.
* fingerprint: the facts that decide what a probe computes, as one record with a
  hash. It has the GPU class and count, the tensor-parallel size (TP: the number
  of GPUs that one vLLM server uses), the number of servers (replicas), the vLLM
  version, the serving flags and the sampling config.

The pod fingerprint of the serial gate (``gate_service.pod_fingerprint``) has
the GPU serial numbers in it, so it is equal only on one pod. In the parallel
design the baseline pod and the candidate box are different pods. This
fingerprint keeps the same facts without the identity of the pod: it is equal
for two boxes of one class that serve the same layout.

Rule: a candidate is scored only against a baseline that has an equal
fingerprint. A difference, or a fingerprint that is missing or not valid on one
side or on the two sides, gives ``no_decision`` and a request for a new
baseline. The scorer does not run. There is no comparison across hardware or
layouts, and there is no comparison of two probes that do not say what they are.

Where the fingerprint of a document comes from:

* a probe that ``probe_serve.py`` served has a ``serve_layout`` record;
* a baseline has the stamp (``stamp``) that its writer made from the HEAD probe;
* a probe with the live ``serve.sh`` has no record. It gets no decision until
  the box puts a stamp on its result.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Mapping, Optional

from .probe_layout import GATE_DTYPE, GATE_MAX_MODEL_LEN, gpu_class_of

VERSION = 'pit-probe-fingerprint-v1'
KEY = 'probe_fingerprint'
GUARD = 'probe-fingerprint-mismatch'
_GPU_LINE = re.compile(r'^\s*\d+\s+.*(NVIDIA|GeForce|RTX|Tesla)', re.IGNORECASE)
FIELDS = (
    'gpu_class', 'gpu_count', 'tensor_parallel', 'replicas', 'vllm_version', 'dtype', 'max_model_len',
    'prefix_caching', 'max_num_seqs', 'gpu_memory_utilization', 'sampling',
)


class FingerprintError(ValueError):
    """A record does not give a complete and valid fingerprint."""


def _digest(fields: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build(fields: Mapping[str, Any]) -> Dict[str, Any]:
    """The fingerprint of these facts. Each field in ``FIELDS`` is necessary."""
    if not isinstance(fields, Mapping):
        raise FingerprintError('the fingerprint fields must be a mapping')
    missing = [name for name in FIELDS if fields.get(name) is None]
    unknown = sorted(set(fields) - set(FIELDS))
    if missing or unknown:
        raise FingerprintError('fingerprint fields: missing %s, unknown %s' % (missing or 'none', unknown or 'none'))
    clean = json.loads(json.dumps({name: fields[name] for name in FIELDS}, sort_keys=True))
    return dict(version=VERSION, fields=clean, sha256=_digest(clean))


def from_serve_layout(serve: Any) -> Dict[str, Any]:
    """The fingerprint of a probe that ``probe_serve.py`` served, from its ``serve_layout`` record."""
    if not isinstance(serve, Mapping):
        raise FingerprintError('serve_layout must be a mapping')
    gpus = list(serve.get('gpus') or [])
    classes = sorted({gpu_class_of(gpu) or 'unknown' for gpu in gpus})
    if len(classes) != 1 or classes[0] == 'unknown':
        raise FingerprintError('the GPUs of the box must be of one known class, got %s' % (sorted(set(map(str, gpus))) or 'none'))
    layout = serve.get('layout') if isinstance(serve.get('layout'), Mapping) else {}
    replicas = [r for r in (serve.get('replicas') or []) if isinstance(r, Mapping)]
    runtime = serve.get('runtime') if isinstance(serve.get('runtime'), Mapping) else {}
    versions = sorted({str(v) for v in [runtime.get('vllm_version')] + [r.get('vllm_version') for r in replicas] if v})
    if len(versions) != 1:
        raise FingerprintError('the servers must run one vLLM version, got %s' % (versions or 'none'))
    return build(dict(
        gpu_class=classes[0], gpu_count=len(gpus), tensor_parallel=layout.get('gpus_per_replica'),
        replicas=len(replicas) or None, vllm_version=versions[0], dtype=GATE_DTYPE, max_model_len=GATE_MAX_MODEL_LEN,
        prefix_caching=layout.get('prefix_caching'), max_num_seqs=layout.get('max_num_seqs'),
        gpu_memory_utilization=layout.get('gpu_memory_utilization'), sampling=serve.get('sampling'),
    ))


def of(doc: Any) -> Optional[Dict[str, Any]]:
    """The fingerprint of a probe result or of a baseline. None: the document does not declare its hardware and layout."""
    if not isinstance(doc, Mapping):
        return None
    stamped = doc.get(KEY)
    if stamped is not None:
        if not isinstance(stamped, Mapping) or stamped.get('version') != VERSION:
            raise FingerprintError('the stamped fingerprint is not version %s' % VERSION)
        built = build(stamped.get('fields'))
        if stamped.get('sha256') != built['sha256']:
            raise FingerprintError('the stamped fingerprint does not agree with its hash')
        return built
    if doc.get('serve_layout') is not None:
        return from_serve_layout(doc['serve_layout'])
    return None


def stamp(doc: Dict[str, Any], probe_result: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Put the fingerprint of a probe on a document (the baseline writer does this). No fingerprint: an error."""
    found = of(probe_result if probe_result is not None else doc)
    if found is None:
        raise FingerprintError('the probe result does not declare its hardware and layout')
    doc[KEY] = found
    return doc


def differences(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> List[str]:
    out = []
    for name in FIELDS:
        b, c = baseline['fields'][name], candidate['fields'][name]
        if b != c:
            out.append('%s: baseline %s, candidate %s' % (name, json.dumps(b, sort_keys=True)[:160], json.dumps(c, sort_keys=True)[:160]))
    return out


def hardware_note(doc: Any) -> Optional[str]:
    """What a document says about its GPUs, for the operator who reads a no-decision. It is not a proof.

    A probe result of a PIT box has ``eval_gpu``. A baseline of the serial gate has the GPU names of its pod in
    the ``runtime`` lines of its ``settings`` (the output of ``nvidia-smi ... | sort | uniq -c``).
    """
    if not isinstance(doc, Mapping):
        return None
    if doc.get('eval_gpu'):
        return str(doc['eval_gpu'])
    settings = doc.get('settings') if isinstance(doc.get('settings'), Mapping) else {}
    lines = [' '.join(str(line).split()) for line in (settings.get('runtime') or []) if _GPU_LINE.match(str(line))]
    return '; '.join(lines) or None


def _no_decision(problems: List[str], found: Mapping[str, Any], baseline: Any, result: Any) -> Dict[str, Any]:
    return dict(
        decision='no_decision', guard=GUARD, reasons=['%s: %s' % (GUARD, text) for text in problems],
        rebaseline=dict(required=True, reason=GUARD, fingerprint=found.get('candidate')),
        probe_fingerprint=dict(
            baseline=found.get('baseline'), candidate=found.get('candidate'), differences=problems,
            hardware_note=dict(baseline=hardware_note(baseline), candidate=hardware_note(result)),
        ),
    )


def refusal(baseline: Any, result: Any) -> Optional[Dict[str, Any]]:
    """None: the candidate can be scored against this baseline. Else the no-decision to give in place of a score.

    There is one path to a score: each side gives a valid fingerprint, and the two are equal. A side that does not
    declare its hardware and layout gets the same answer as a difference. This includes the case where no side
    declares it (a probe with the live ``serve.sh`` against a baseline of the serial gate): the two can be from
    different hardware, and nothing in the two documents proves that they are not.
    """
    found: Dict[str, Optional[Dict[str, Any]]] = {}
    problems: List[str] = []
    for side, doc in (('baseline', baseline), ('candidate', result)):
        try:
            found[side] = of(doc)
        except FingerprintError as exc:
            found[side] = None
            problems.append('%s: %s' % (side, exc))
            continue
        if found[side] is None:
            problems.append('%s: no hardware and layout fingerprint' % side)
    if not problems:
        problems = differences(found['baseline'], found['candidate'])
    if not problems:
        return None
    return _no_decision(problems, found, baseline, result)


def unchecked(result: Any, decision: Any) -> Dict[str, Any]:
    """The no-decision for a score that the scorer gave when the check did not see a ready baseline.

    The check and the scorer must read the same baseline. If the scorer has a decision and the check had no
    baseline, the fingerprints were not compared. That decision is not used.
    """
    problems = ['the scorer gave %r against a baseline that the fingerprint check did not see' % (decision,)]
    try:
        found = dict(candidate=of(result))
    except FingerprintError as exc:
        found = dict(candidate=None)
        problems.append('candidate: %s' % exc)
    return _no_decision(problems, found, None, result)
