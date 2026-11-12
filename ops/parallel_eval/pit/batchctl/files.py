"""How the files of a model state are stored as objects.

A state manifest (``layout.py``) is a flat list of objects with a role, a
size and a sha256. A model state is a set of files (bf16 weights, the fp32
master, a few small documents), and one file can be larger than one upload
through a signed URL may be. So a file is stored as one or more objects, its
parts, and one more object, ``index.json``, says which parts make which file:

    {"schema": "affine-batchctl-files-v1",
     "checkpoint_id": ..., "fp32_entry": ..., "fp32_master_digest": ...,
     "files": [{"path": "bf16/model-00001-of-00004.safetensors",
                "role": "bf16_weights", "size": ..., "sha256": ...,
                "parts": ["p-000000", "p-000001"]}, ...]}

``path`` is ``<group>/<file name>``: ``bf16/`` the weights that are served
and trained, ``fp32/`` the fp32 master with its ``meta.json``, ``aux/``
small documents. A part has the role of its file. A file of size zero has no
part, because an object of a manifest is never empty.

The manifest holds the sha256 of ``index.json`` and of every part, and the
index holds the sha256 of every file. So the id of a state (the sha256 of its
manifest) fixes every byte of every file.

The box side (``box/job.py``) reads and writes the same format with its own
code, because that file must stand alone. ``test_box_job.py`` compares the two.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ...durable import canonical
from .layout import (
    MAX_OBJECTS, META_FP32_MASTER_DIGEST, ROLE_AUX, ROLE_BF16_WEIGHTS, ROLE_FP32_MASTER, ROLES, ObjectRef,
    StateManifest, is_hex64, load_json, sha256_hex,
)

FILES_SCHEMA = 'affine-batchctl-files-v1'
INDEX_NAME = 'index.json'
SLOT_NAME = 'p-%06d'
MAX_INDEX_BYTES = 8 << 20
GROUP_BF16, GROUP_FP32, GROUP_AUX = 'bf16', 'fp32', 'aux'
GROUP_ROLE = {GROUP_BF16: ROLE_BF16_WEIGHTS, GROUP_FP32: ROLE_FP32_MASTER, GROUP_AUX: ROLE_AUX}
DESCRIPTOR_PATH = 'aux/descriptor.json'
META_CHECKPOINT = 'checkpoint_id'
META_FP32_ENTRY = 'fp32_entry'
META_FORMAT = 'format'

PATH = re.compile(r'([a-z0-9]+/)?[A-Za-z0-9][A-Za-z0-9._-]{0,199}')


class IndexInvalid(ValueError):
    """The file index of a state is not valid."""


@dataclass(frozen=True)
class FileEntry:
    path: str
    role: str
    size: int
    sha256: str
    parts: Tuple[str, ...]

    def doc(self) -> Dict[str, Any]:
        return {
            'path': self.path, 'role': self.role, 'size': self.size, 'sha256': self.sha256,
            'parts': list(self.parts),
        }


@dataclass(frozen=True)
class Index:
    files: Tuple[FileEntry, ...]
    checkpoint_id: Optional[str]
    fp32_entry: Optional[str]
    fp32_master_digest: str

    def file(self, path: str) -> Optional[FileEntry]:
        return next((entry for entry in self.files if entry.path == path), None)

    def group(self, group: str) -> Tuple[FileEntry, ...]:
        return tuple(entry for entry in self.files if entry.path.startswith(group + '/'))


def is_path(value: Any) -> bool:
    return isinstance(value, str) and PATH.fullmatch(value) is not None and '..' not in value


def slot_name(number: int) -> str:
    return SLOT_NAME % number


def index_body(files: Sequence[FileEntry], *, checkpoint_id: Optional[str], fp32_entry: Optional[str],
               fp32_master_digest: str) -> bytes:
    return canonical({
        'schema': FILES_SCHEMA, 'files': [entry.doc() for entry in sorted(files, key=lambda entry: entry.path)],
        'checkpoint_id': checkpoint_id, 'fp32_entry': fp32_entry, 'fp32_master_digest': fp32_master_digest,
    })


def parse_index(raw: bytes, objects: Mapping[str, ObjectRef]) -> Index:
    """Validate ``index.json`` against the objects of its manifest.

    Every part must be an object of the manifest, used by one file only, with
    the role of that file, and the sizes of the parts must add up to the size
    of the file. Every object of the manifest but the index is such a part.
    """
    def need(cond: bool, why: str) -> None:
        if not cond:
            raise IndexInvalid(why)

    need(len(raw) <= MAX_INDEX_BYTES, 'index_too_large')
    try:
        doc = load_json(raw)
    except ValueError:
        raise IndexInvalid('index_not_json') from None
    need(isinstance(doc, dict), 'index_not_object')
    need(set(doc) == {'schema', 'files', 'checkpoint_id', 'fp32_entry', 'fp32_master_digest'}, 'index_keys')
    need(doc['schema'] == FILES_SCHEMA, 'index_schema')
    need(is_hex64(doc['fp32_master_digest']), 'index_fp32_master_digest')
    for key in ('checkpoint_id', 'fp32_entry'):
        need(doc[key] is None or (isinstance(doc[key], str) and 0 < len(doc[key]) <= 256), 'index_' + key)
    need(isinstance(doc['files'], list) and doc['files'], 'index_files')
    files: List[FileEntry] = []
    paths, used = set(), set()
    for item in doc['files']:
        need(isinstance(item, dict) and set(item) == {'path', 'role', 'size', 'sha256', 'parts'}, 'file_keys')
        path, role, size, sha, parts = item['path'], item['role'], item['size'], item['sha256'], item['parts']
        need(is_path(path) and path not in paths, 'file_path')
        paths.add(path)
        need(role in ROLES, 'file_role')
        need(isinstance(size, int) and not isinstance(size, bool) and size >= 0, 'file_size')
        need(is_hex64(sha), 'file_sha256')
        need(isinstance(parts, list) and all(isinstance(name, str) for name in parts), 'file_parts')
        total = 0
        for name in parts:
            need(name in objects and name not in used and name != INDEX_NAME, 'file_part')
            need(objects[name].role == role, 'file_part_role')
            used.add(name)
            total += objects[name].size
        need(total == size, 'file_parts_size')
        files.append(FileEntry(path, role, size, sha, tuple(parts)))
    need(set(objects) - used <= {INDEX_NAME}, 'object_without_file')
    return Index(tuple(files), doc['checkpoint_id'], doc['fp32_entry'], doc['fp32_master_digest'])


def manifest_of(
    *, parent: Optional[str], batch_id: str, parts: Sequence[ObjectRef], index: bytes, meta: Mapping[str, Any],
) -> StateManifest:
    """The manifest of a state from its parts and its index."""
    objects = list(parts) + [ObjectRef(INDEX_NAME, ROLE_AUX, sha256_hex(index), len(index))]
    if len(objects) > MAX_OBJECTS:
        raise IndexInvalid('too_many_objects')
    if not is_hex64(meta.get(META_FP32_MASTER_DIGEST)):
        raise IndexInvalid('missing_fp32_master_digest')
    return StateManifest(
        parent, batch_id, tuple(sorted(objects, key=lambda ref: ref.name)), tuple(sorted(meta.items())),
    )


def slots_for(manifest: StateManifest, chunk_bytes: int, *, spare: float = 0.25, more: int = 16) -> List[str]:
    """Names of the upload slots that a box gets for a state of about the
    size of ``manifest`` (its base): one slot per ``chunk_bytes``, one more
    per object for the last part of each file, and some spare."""
    if chunk_bytes <= 0:
        raise ValueError('chunk_bytes must be above zero')
    count = sum(-(-ref.size // chunk_bytes) for ref in manifest.objects) + len(manifest.objects)
    count = min(MAX_OBJECTS - 1, int(count * (1.0 + spare)) + more)
    return [slot_name(number) for number in range(count)]


__all__ = [
    'DESCRIPTOR_PATH', 'FILES_SCHEMA', 'FileEntry', 'GROUP_AUX', 'GROUP_BF16', 'GROUP_FP32', 'GROUP_ROLE',
    'INDEX_NAME', 'Index', 'IndexInvalid', 'MAX_INDEX_BYTES', 'META_CHECKPOINT', 'META_FORMAT', 'META_FP32_ENTRY',
    'index_body', 'is_path', 'manifest_of', 'parse_index', 'slot_name', 'slots_for',
]
