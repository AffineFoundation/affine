"""R2 key layout and strict document formats.

Layout under one root prefix
----------------------------
``HEAD.json``                          the HEAD pointer. The only mutable object.
``links/<link_id>.json``               one chain link. ``link_id`` = sha256 of the body.
``states/<state_id>/manifest.json``    state manifest. ``state_id`` = sha256 of the body.
``states/<state_id>/obj/<name>``       state files. The manifest names their sha256.
``results/<sha256>.json``              batch result. The key is its sha256.
``outcomes/<batch_id>.json``           final outcome of a batch. Written once.
``staging/<batch_id>/<fence>/...``     the only place a box may write.
``RUN.json``                           binds the prefix to one journal. Written once.

A box uploads ``obj/<name>`` files and ``result.json`` under its staging
prefix, then ``COMPLETE.json`` last. The controller reads nothing from a
staging prefix before ``COMPLETE.json`` exists.

Every key outside ``HEAD.json`` and ``staging/`` is derived from content
hashes, and the controller writes it once after it verified the bytes.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from ...durable import canonical

# Used with ``fullmatch`` only: ``$`` would also accept a trailing newline.
HEX64 = re.compile(r'[0-9a-f]{64}')
SAFE_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}')
SAFE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,199}')
META_KEY = re.compile(r'[a-z][a-z0-9_]{0,63}')
# ASCII digits only, and short: ``int`` must never fail on a key from outside.
STAGING_FENCE = re.compile(r'[0-9]{1,18}')

SCHEMA_COMPLETE = 'affine-batchctl-complete-v1'
SCHEMA_STATE = 'affine-batchctl-state-v1'
SCHEMA_LINK = 'affine-batchctl-link-v1'
SCHEMA_HEAD = 'affine-batchctl-head-v1'
SCHEMA_OUTCOME = 'affine-batchctl-outcome-v1'
SCHEMA_HEARTBEAT = 'affine-batchctl-heartbeat-v1'
SCHEMA_RUN = 'affine-batchctl-run-v1'

ROLE_FP32_MASTER = 'fp32_master'
ROLE_BF16_WEIGHTS = 'bf16_weights'
ROLE_AUX = 'aux'
ROLES = (ROLE_FP32_MASTER, ROLE_BF16_WEIGHTS, ROLE_AUX)
# A state without both of these is not a full state and is never recognized.
REQUIRED_ROLES = (ROLE_FP32_MASTER, ROLE_BF16_WEIGHTS)

# Name of the digest the trainer checks before it trains (fail closed).
META_FP32_MASTER_DIGEST = 'fp32_master_digest'

MAX_COMPLETE_BYTES = 1 << 20
MAX_RESULT_BYTES = 64 << 20
MAX_HEAD_BYTES = 1 << 16
MAX_HEARTBEAT_BYTES = 1 << 12
MAX_OBJECTS = 4096
MAX_OBJECT_BYTES = 1 << 44
MAX_META_KEYS = 64

GENESIS_BATCH = 'genesis'
GENESIS_FENCE = 0
RESERVED_BATCH_IDS = frozenset({GENESIS_BATCH})


class PackageInvalid(ValueError):
    """An uploaded package is definitely bad. Not a transient error."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_hex64(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def is_safe_batch_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and SAFE_ID.fullmatch(value) is not None
        and '..' not in value
        and value not in RESERVED_BATCH_IDS
    )


def _need(cond: bool, reason: str) -> None:
    if not cond:
        raise PackageInvalid(reason)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_json(raw: bytes) -> Any:
    """Parse JSON that came from outside. Every failure is a ``ValueError``.

    A document that is nested too deep makes the parser raise
    ``RecursionError``. Bytes from a box must never be able to do more than
    fail the package, so that error is turned into a ``ValueError`` here.
    """
    try:
        return json.loads(raw)
    except RecursionError:
        raise ValueError('json_too_deep') from None


@dataclass(frozen=True)
class ObjectRef:
    name: str
    role: str
    sha256: str
    size: int

    def doc(self) -> Dict[str, Any]:
        return {'name': self.name, 'role': self.role, 'sha256': self.sha256, 'size': self.size}


@dataclass(frozen=True)
class StateManifest:
    """The list of files that make one full model state."""

    parent: Optional[str]
    batch_id: str
    objects: Tuple[ObjectRef, ...]
    meta: Tuple[Tuple[str, Any], ...]

    def doc(self) -> Dict[str, Any]:
        return {
            'schema': SCHEMA_STATE,
            'parent': self.parent,
            'batch_id': self.batch_id,
            'objects': [o.doc() for o in self.objects],
            'meta': dict(self.meta),
        }

    def body(self) -> bytes:
        return canonical(self.doc())

    @property
    def state_id(self) -> str:
        return sha256_hex(self.body())

    @property
    def fp32_master_digest(self) -> str:
        return dict(self.meta)[META_FP32_MASTER_DIGEST]


def parse_manifest(doc: Any, *, batch_id: str, parent: Optional[str]) -> StateManifest:
    """Validate a state manifest. Every rule here fails closed."""
    _need(isinstance(doc, dict), 'manifest_not_object')
    _need(set(doc) == {'schema', 'parent', 'batch_id', 'objects', 'meta'}, 'manifest_keys')
    _need(doc['schema'] == SCHEMA_STATE, 'manifest_schema')
    _need(doc['batch_id'] == batch_id, 'manifest_batch_mismatch')
    _need(doc['parent'] == parent, 'manifest_parent_mismatch')
    objects = doc['objects']
    _need(isinstance(objects, list) and 0 < len(objects) <= MAX_OBJECTS, 'manifest_objects')
    refs = []
    seen = set()
    for item in objects:
        _need(isinstance(item, dict), 'object_not_object')
        _need(set(item) == {'name', 'role', 'sha256', 'size'}, 'object_keys')
        name, role, sha, size = item['name'], item['role'], item['sha256'], item['size']
        _need(isinstance(name, str) and SAFE_NAME.fullmatch(name) is not None and '..' not in name, 'object_name')
        _need(name not in seen, 'object_name_duplicate')
        seen.add(name)
        _need(role in ROLES, 'object_role')
        _need(is_hex64(sha), 'object_sha256')
        _need(_is_int(size) and 0 < size <= MAX_OBJECT_BYTES, 'object_size')
        refs.append(ObjectRef(name, role, sha, size))
    roles = {r.role for r in refs}
    for required in REQUIRED_ROLES:
        _need(required in roles, 'missing_role_%s' % required)
    meta = doc['meta']
    _need(isinstance(meta, dict) and len(meta) <= MAX_META_KEYS, 'manifest_meta')
    for key, value in meta.items():
        _need(isinstance(key, str) and META_KEY.fullmatch(key) is not None, 'meta_key')
        ok = (isinstance(value, str) and len(value) <= 256) or _is_int(value)
        _need(ok, 'meta_value')
    # The next trainer run refuses to start without this digest. A state that
    # does not carry it can never become a base.
    _need(is_hex64(meta.get(META_FP32_MASTER_DIGEST)), 'missing_fp32_master_digest')
    refs.sort(key=lambda r: r.name)
    return StateManifest(parent, batch_id, tuple(refs), tuple(sorted(meta.items())))


@dataclass(frozen=True)
class Package:
    """A parsed ``COMPLETE.json``: what a box claims it uploaded."""

    batch_id: str
    fence: int
    base_state: Optional[str]
    result_sha: str
    result_size: int
    manifest: Optional[StateManifest]
    complete_sha: str


def parse_complete(
    raw: bytes, *, batch_id: str, fence: int, base_state: Optional[str]
) -> Package:
    """Validate ``COMPLETE.json`` against the lease that owns the prefix."""
    _need(len(raw) <= MAX_COMPLETE_BYTES, 'complete_too_large')
    try:
        doc = load_json(raw)
    except ValueError:
        raise PackageInvalid('complete_not_json') from None
    _need(isinstance(doc, dict), 'complete_not_object')
    _need(set(doc) == {'schema', 'batch_id', 'fence', 'base_state', 'result', 'state'}, 'complete_keys')
    _need(doc['schema'] == SCHEMA_COMPLETE, 'complete_schema')
    _need(doc['batch_id'] == batch_id, 'complete_batch_mismatch')
    _need(_is_int(doc['fence']) and doc['fence'] == fence, 'complete_fence_mismatch')
    _need(doc['base_state'] == base_state, 'complete_base_mismatch')
    result = doc['result']
    _need(isinstance(result, dict) and set(result) == {'sha256', 'size'}, 'result_keys')
    _need(is_hex64(result['sha256']), 'result_sha256')
    _need(_is_int(result['size']) and 0 < result['size'] <= MAX_RESULT_BYTES, 'result_size')
    manifest = None
    if doc['state'] is not None:
        manifest = parse_manifest(doc['state'], batch_id=batch_id, parent=base_state)
    return Package(
        batch_id=batch_id,
        fence=fence,
        base_state=base_state,
        result_sha=result['sha256'],
        result_size=result['size'],
        manifest=manifest,
        complete_sha=sha256_hex(raw),
    )


def complete_doc(
    *,
    batch_id: str,
    fence: int,
    base_state: Optional[str],
    result_sha: str,
    result_size: int,
    manifest: Optional[StateManifest],
) -> Dict[str, Any]:
    """The document a box writes last. Shared with the box side and the fakes."""
    return {
        'schema': SCHEMA_COMPLETE,
        'batch_id': batch_id,
        'fence': fence,
        'base_state': base_state,
        'result': {'sha256': result_sha, 'size': result_size},
        'state': manifest.doc() if manifest is not None else None,
    }


def link_doc(
    *,
    seq: int,
    state_id: str,
    parent: Optional[str],
    batch_id: Optional[str],
    result_sha: Optional[str],
    fence: int,
) -> Dict[str, Any]:
    return {
        'schema': SCHEMA_LINK,
        'seq': seq,
        'state': state_id,
        'parent': parent,
        'batch_id': batch_id,
        'result': result_sha,
        'fence': fence,
    }


def link_body(**fields: Any) -> bytes:
    return canonical(link_doc(**fields))


def link_id(**fields: Any) -> str:
    return sha256_hex(link_body(**fields))


def head_body(*, seq: int, link: str, state_id: str) -> bytes:
    """Pointer bytes. A pure function of the apply intent, so retries match."""
    return canonical({'schema': SCHEMA_HEAD, 'seq': seq, 'link': link, 'state': state_id})


@dataclass(frozen=True)
class HeadPointer:
    seq: int
    link: str
    state_id: str


def parse_head(raw: bytes) -> HeadPointer:
    """Parse the pointer. Raises ``ValueError`` on anything unexpected."""
    doc = load_json(raw)
    if not isinstance(doc, dict) or set(doc) != {'schema', 'seq', 'link', 'state'}:
        raise ValueError('head pointer keys')
    if doc['schema'] != SCHEMA_HEAD or not _is_int(doc['seq']) or doc['seq'] < 0:
        raise ValueError('head pointer schema or seq')
    if not is_hex64(doc['link']) or not is_hex64(doc['state']):
        raise ValueError('head pointer ids')
    return HeadPointer(doc['seq'], doc['link'], doc['state'])


def outcome_body(fields: Mapping[str, Any]) -> bytes:
    return canonical(dict(fields, schema=SCHEMA_OUTCOME))


def heartbeat_body(*, fence: int, seq: int, stage: str = '') -> bytes:
    """What a box writes while it works. ``seq`` must grow with each write."""
    return canonical({'schema': SCHEMA_HEARTBEAT, 'fence': fence, 'seq': seq, 'stage': stage})


def parse_heartbeat(raw: bytes, *, fence: int) -> int:
    """Return the heartbeat sequence. Raises ``ValueError`` on anything else."""
    doc = load_json(raw)
    if not isinstance(doc, dict) or doc.get('schema') != SCHEMA_HEARTBEAT:
        raise ValueError('heartbeat schema')
    if doc.get('fence') != fence or not _is_int(doc.get('seq')) or doc['seq'] < 0:
        raise ValueError('heartbeat fence or seq')
    return doc['seq']


def run_body(*, run_id: str, journal_id: str) -> bytes:
    """Binds an R2 prefix to one journal. ``journal_id`` is the hash of the
    first journal record, so a new journal can never take over the prefix."""
    return canonical({'schema': SCHEMA_RUN, 'run_id': run_id, 'journal': journal_id})


class Layout:
    """Key builder for one root prefix."""

    def __init__(self, prefix: str):
        if not prefix or prefix.startswith('/') or not prefix.endswith('/') or '//' in prefix:
            raise ValueError('prefix must be relative and end with one slash')
        self.prefix = prefix

    @property
    def head(self) -> str:
        return self.prefix + 'HEAD.json'

    @property
    def run(self) -> str:
        return self.prefix + 'RUN.json'

    def link(self, link: str) -> str:
        return '%slinks/%s.json' % (self.prefix, link)

    def state_prefix(self, state_id: str) -> str:
        return '%sstates/%s/' % (self.prefix, state_id)

    def state_manifest(self, state_id: str) -> str:
        return self.state_prefix(state_id) + 'manifest.json'

    def state_object(self, state_id: str, name: str) -> str:
        return '%sobj/%s' % (self.state_prefix(state_id), name)

    def result(self, sha: str) -> str:
        return '%sresults/%s.json' % (self.prefix, sha)

    def outcome(self, batch_id: str) -> str:
        return '%soutcomes/%s.json' % (self.prefix, batch_id)

    @property
    def staging_root(self) -> str:
        return self.prefix + 'staging/'

    def staging_prefix(self, batch_id: str, fence: int) -> str:
        return '%s%s/%012d/' % (self.staging_root, batch_id, fence)

    def staging_complete(self, batch_id: str, fence: int) -> str:
        return self.staging_prefix(batch_id, fence) + 'COMPLETE.json'

    def staging_result(self, batch_id: str, fence: int) -> str:
        return self.staging_prefix(batch_id, fence) + 'result.json'

    def staging_heartbeat(self, batch_id: str, fence: int) -> str:
        return self.staging_prefix(batch_id, fence) + 'HEARTBEAT.json'

    def staging_object(self, batch_id: str, fence: int, name: str) -> str:
        return '%sobj/%s' % (self.staging_prefix(batch_id, fence), name)

    def parse_staging_key(self, key: str) -> Optional[Tuple[str, int]]:
        """Return (batch_id, fence) for a staging key, else None."""
        root = self.staging_root
        if not key.startswith(root):
            return None
        parts = key[len(root):].split('/')
        if len(parts) < 3 or STAGING_FENCE.fullmatch(parts[1]) is None:
            return None
        return parts[0], int(parts[1])
