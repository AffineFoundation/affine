"""Make the first state ("genesis") of a run from the live model HEAD.

Terms
-----
* Live HEAD: the checkpoint that the validator host uses now. Its id is ``H``.
* Descriptor: the document that names the bf16 weight files of ``H`` and the sha256 of each file.
* Master index: the document that names the shards of the fp32 master of ``H`` with the sha256, the
  size and the R2 key of each shard. The index of the live publisher names ``meta.json`` by a pointer
  (``meta_json.r2_key``) and has no file entry for it. An index of the PIT pipeline names it as a file too.
* ``meta.json``: the file of the fp32 master that the trainer on the box reads. It names the master
  digest, the entry (``bf16_sha256``) and each shard with its sha256 and its size.
* Genesis state: the first state of a run. Its manifest has the batch id ``genesis`` and no parent.
* Claim: a sha256 that a document names. This module does not download a file to check a claim.

What the seed does
------------------
1. It reads ``meta.json`` of the live master and checks it against the master index (see the rules).
2. It reads the size and the etag of every live file (``head`` only).
3. It plans the state: one object for each file, then ``index.json``, then the manifest (``files.py``).
4. It copies every live file to the prefix of the run. R2 makes each copy on the server side.
5. It writes the descriptor and ``index.json``. It writes the manifest last, and only if its key is free.

The files of the state are the files of a trained state (``box/steps.py``): ``bf16/<name>`` for each
weight file, ``fp32/<name>`` for each file of the fp32 master, and ``aux/descriptor.json``.

Rules
-----
* The live side is read-only. No live key is written, changed or deleted. A live key is used for
  ``head`` and as the source of a server-side copy, and for nothing else.
* Every key that is written is under the prefix of the run (``guard.check_key``). A live key that
  is under that prefix is refused, so a write can never change a live key.
* The seed is idempotent. After an error, run it again. An object that is complete is not copied again.
* The sha256 of a copied object is a claim. Its size comes from R2. The controller hashes every object
  again before it uses the state (``effects._verify_state``), so a wrong claim fails closed later.
* ``meta.json`` is not a claim: the seed reads its bytes. They must name the master digest and the entry
  of the master index, and the same shards with the same sha256 and size. If the index names
  ``meta.json`` as a file, the bytes must have that sha256 and that size. The copy is made only while
  the live key still has the bytes that were checked. Any difference stops the seed before a write.

Command line::

    python -m ops.parallel_eval.pit.batchctl.genesis --run-id R --bucket-config FILE --life-root DIR
        --master-index-dir DIR [--checkpoint H] [--ckpt-prefix P] [--source-bucket B]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ...durable import canonical
from . import guard
from .files import (
    DESCRIPTOR_PATH, GROUP_BF16, GROUP_FP32, INDEX_NAME, META_CHECKPOINT, META_FP32_ENTRY, FileEntry, index_body,
    manifest_of, parse_index, slot_name,
)
from .interfaces import NotFound, PreconditionFailed, StoreError, TooLarge
from .layout import (
    GENESIS_BATCH, MAX_COMPLETE_BYTES, MAX_OBJECT_BYTES, META_FP32_MASTER_DIGEST, ROLE_AUX, ROLE_BF16_WEIGHTS,
    ROLE_FP32_MASTER, SAFE_NAME, Layout, ObjectRef, StateManifest, is_hex64, load_json, parse_manifest, sha256_hex,
)
from .r2 import R2Store, ReadOnlyStore, client_from_bucket_config

DEFAULT_CKPT_PREFIX = 'mimo/affine-mimo9b-grpo-20261013/checkpoints'
CURRENT_NAME = 'CURRENT.json'
JOB_NAME = 'TRAINING-JOB.json'
JOB_CHECKPOINT = ('payload', 'manifest', 'payload', 'checkpoint')
MASTER_META = 'meta.json'
# The live file names each tensor of the master: about 100 KiB. Well under this limit.
MAX_META_BYTES = 16 << 20
MAX_KEY_CHARS = 1024
# Used with ``fullmatch`` only. Short, so that ``int`` never fails on a directory name.
EPOCH_DIR = re.compile(r'[0-9]{1,18}')


class GenesisError(RuntimeError):
    """The first state cannot be made from these facts. A second run does not help."""


@dataclass(frozen=True)
class LiveHead:
    """The facts of the live HEAD that the first state is made from."""

    checkpoint_id: str
    weights: Tuple[Tuple[str, str], ...]            # (file name, sha256), sorted
    entry: str                                      # bf16_sha256 of the master index
    master_digest: str
    # (file name, sha256, bytes, r2 key), sorted: the shards, and ``meta.json`` if the index names it as a file.
    master: Tuple[Tuple[str, str, int, str], ...]
    meta_key: str                                   # the live R2 key of ``meta.json`` of the master


@dataclass(frozen=True)
class Plan:
    """The first state and the work that puts it in R2."""

    manifest: StateManifest
    index: bytes
    copies: Tuple[Tuple[str, str, int], ...]   # (live source key, object name, size)
    local: Tuple[Tuple[str, bytes], ...]       # (object name, bytes): the descriptor part and index.json


@dataclass(frozen=True)
class _Source:
    """One live file that the seed copies."""

    path: str
    role: str
    sha256: str
    key: str
    named_size: Optional[int]   # the size that the master index names; a descriptor names none


def _need(cond: bool, why: str) -> None:
    if not cond:
        raise GenesisError(why)


def _is_name(value: Any) -> bool:
    return isinstance(value, str) and SAFE_NAME.fullmatch(value) is not None and '..' not in value


def _is_size(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value <= MAX_OBJECT_BYTES


def _is_key(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= MAX_KEY_CHARS and not value.startswith('/')


# ----------------------------------------------------------------------------
# The live HEAD
# ----------------------------------------------------------------------------

def _weights(files: Any) -> Tuple[Tuple[str, str], ...]:
    _need(isinstance(files, Mapping) and len(files) > 0, 'the descriptor names no files')
    out = []
    for name, value in files.items():
        sha = value.get('sha256') if isinstance(value, Mapping) else value
        _need(_is_name(name) and is_hex64(sha), 'the descriptor has a file entry that is not valid: %.80r' % (name,))
        out.append((name, sha))
    return tuple(sorted(out))


def _master(files: Any) -> Tuple[Tuple[str, str, int, str], ...]:
    _need(isinstance(files, Mapping), 'the master index names no files')
    out = []
    for name, info in files.items():
        ok = (
            _is_name(name) and isinstance(info, Mapping) and is_hex64(info.get('sha256'))
            and _is_size(info.get('bytes')) and _is_key(info.get('r2_key'))
        )
        _need(ok, 'the master index has a file entry that is not valid: %.80r' % (name,))
        out.append((name, info['sha256'], info['bytes'], info['r2_key']))
    _need(any(name != MASTER_META for name, _sha, _size, _key in out), 'the master index names no shard')
    return tuple(sorted(out))


def _meta_key(master_index: Mapping[str, Any], master: Sequence[Tuple[str, str, int, str]]) -> str:
    """The live R2 key of ``meta.json`` of the master.

    The index of the live publisher has a pointer: ``meta_json.r2_key``. An index of the PIT pipeline
    also names ``meta.json`` as a file. With neither, ``meta.json`` is in the directory of the shards.
    """
    keys = {name: key for name, _sha, _size, key in master}
    pointer = master_index.get('meta_json')
    pointed = pointer.get('r2_key') if isinstance(pointer, Mapping) else None
    if pointed is not None:
        _need(_is_key(pointed) and pointed.rpartition('/')[2] == MASTER_META,
              'meta_json of the master index does not point to a meta.json')
        _need(keys.get(MASTER_META, pointed) == pointed, 'the master index names two keys for meta.json')
        return pointed
    if MASTER_META in keys:
        return keys[MASTER_META]
    folders = {key.rpartition('/')[0] for key in keys.values()}
    _need(len(folders) == 1 and '' not in folders,
          'the master index names no meta.json, and its shards are not in one directory')
    return '%s/%s' % (folders.pop(), MASTER_META)


def live_head(descriptor: Mapping[str, Any], master_index: Mapping[str, Any]) -> LiveHead:
    """Validate the two documents of the live HEAD. This function reads nothing."""
    _need(isinstance(descriptor, Mapping) and is_hex64(descriptor.get('id')), 'the descriptor names no checkpoint id')
    _need(isinstance(master_index, Mapping), 'the master index is not an object')
    entry, digest = master_index.get('bf16_sha256'), master_index.get('master_digest')
    _need(is_hex64(entry), 'the master index names no bf16_sha256')
    _need(is_hex64(digest), 'the master index names no master_digest')
    # ``meta_json`` is a pointer to ``meta.json``. If it also names a digest or an entry, that must be
    # the one of the index. The seed checks the bytes of ``meta.json`` itself (``_with_meta``).
    meta = master_index.get('meta_json', {})
    agrees = (
        isinstance(meta, Mapping) and meta.get('master_digest', digest) == digest
        and meta.get('bf16_sha256', entry) == entry
    )
    _need(agrees, 'meta_json of the master index names another master')
    master = _master(master_index.get('files'))
    return LiveHead(
        checkpoint_id=descriptor['id'], weights=_weights(descriptor.get('files')), entry=entry,
        master_digest=digest, master=master, meta_key=_meta_key(master_index, master),
    )


def _document(path: Path) -> Any:
    """The JSON document of a local file, or None when the file cannot be read."""
    try:
        return load_json(path.read_bytes())
    except (OSError, ValueError):
        return None


def _block(doc: Any, *keys: str) -> Optional[Mapping[str, Any]]:
    """The object at ``keys`` inside ``doc``, or None."""
    for key in keys:
        doc = doc.get(key) if isinstance(doc, Mapping) else None
    return doc if isinstance(doc, Mapping) else None


def _job_checkpoint(life_root: Path, checkpoint_id: str) -> Optional[Mapping[str, Any]]:
    """The checkpoint block of the training job of the newest epoch that names ``checkpoint_id``."""
    try:
        epochs = sorted(
            (int(item.name), item) for item in (life_root / 'epochs').iterdir() if EPOCH_DIR.fullmatch(item.name)
        )
    except OSError:
        return None
    for _number, directory in reversed(epochs):
        block = _block(_document(directory / JOB_NAME), *JOB_CHECKPOINT)
        if block is not None and block.get('id') == checkpoint_id:
            return block
    return None


def read_live_head(life_root: Path, master_index_dir: Path, *, checkpoint: Optional[str] = None) -> LiveHead:
    """Read the live HEAD from the files of the validator host. Read-only.

    ``checkpoint``: the id of the HEAD. The default is the id that ``CURRENT.json`` names.
    """
    life_root = Path(life_root)
    current = _block(_document(life_root / CURRENT_NAME), 'checkpoint')
    checkpoint_id = checkpoint or (current or {}).get('id')
    _need(is_hex64(checkpoint_id), 'no checkpoint id: %s names none and none was given' % CURRENT_NAME)
    if current is not None and current.get('id') == checkpoint_id and 'files' in current:
        descriptor: Optional[Mapping[str, Any]] = current
    else:
        descriptor = _job_checkpoint(life_root, checkpoint_id)
    _need(descriptor is not None, 'no descriptor of checkpoint %s under %s' % (checkpoint_id, life_root))
    master_index = _document(Path(master_index_dir) / (checkpoint_id + '.json'))
    _need(master_index is not None,
          'no fp32 master index of checkpoint %s under %s' % (checkpoint_id, master_index_dir))
    return live_head(descriptor, master_index)


# ----------------------------------------------------------------------------
# The plan
# ----------------------------------------------------------------------------

def _sources(head: LiveHead, ckpt_prefix: str) -> List[_Source]:
    _need(_is_key(ckpt_prefix), 'the checkpoint prefix is not valid')
    model = '%s/%s/model/' % (ckpt_prefix.rstrip('/'), head.checkpoint_id)
    out = [
        _Source('%s/%s' % (GROUP_BF16, name), ROLE_BF16_WEIGHTS, sha, model + name, None)
        for name, sha in head.weights
    ]
    out += [
        _Source('%s/%s' % (GROUP_FP32, name), ROLE_FP32_MASTER, sha, key, size)
        for name, sha, size, key in head.master
    ]
    return out


def _descriptor_body(head: LiveHead) -> bytes:
    return canonical({'id': head.checkpoint_id, 'files': dict(head.weights)})


def _accepted(manifest: StateManifest, index: bytes) -> None:
    """The controller parses a state with these two functions. A state that they refuse is never written."""
    try:
        parse_manifest(manifest.doc(), batch_id=GENESIS_BATCH, parent=None)
        parse_index(index, {ref.name: ref for ref in manifest.objects})
    except ValueError as exc:
        raise GenesisError('the planned state is not valid: %s' % exc) from None


def plan(head: LiveHead, sizes: Mapping[str, int], *, ckpt_prefix: str = DEFAULT_CKPT_PREFIX) -> Plan:
    """The first state from the live HEAD and the sizes of its live keys.

    Pure: no I/O. The same input gives the same state id. ``sizes`` maps a live source key to its size.
    """
    # The trainer on the box reads ``meta.json``. A state without it trains nothing.
    _need(any(name == MASTER_META for name, _sha, _size, _key in head.master),
          'the fp32 master has no meta.json: the seed reads it from its live key')
    descriptor = _descriptor_body(head)
    rows: List[Tuple[str, str, str, int, Optional[str]]] = [
        (DESCRIPTOR_PATH, ROLE_AUX, sha256_hex(descriptor), len(descriptor), None),
    ]
    for source in _sources(head, ckpt_prefix):
        size = sizes.get(source.key)
        _need(size is not None, 'the live key %s has no known size' % source.key)
        _need(_is_size(size), 'the live key %s is empty or too large' % source.key)
        _need(source.named_size in (None, size),
              'the live key %s has %d bytes, the master index names %s' % (source.key, size, source.named_size))
        rows.append((source.path, source.role, source.sha256, size, source.key))
    entries: List[FileEntry] = []
    parts: List[ObjectRef] = []
    copies: List[Tuple[str, str, int]] = []
    local: List[Tuple[str, bytes]] = []
    for number, (path, role, sha, size, key) in enumerate(sorted(rows, key=lambda row: row[0])):
        slot = slot_name(number)
        entries.append(FileEntry(path, role, size, sha, (slot,)))
        parts.append(ObjectRef(slot, role, sha, size))
        if key is None:
            local.append((slot, descriptor))
        else:
            copies.append((key, slot, size))
    index = index_body(
        entries, checkpoint_id=head.checkpoint_id, fp32_entry=head.entry, fp32_master_digest=head.master_digest,
    )
    meta: Dict[str, Any] = {
        META_FP32_MASTER_DIGEST: head.master_digest, META_CHECKPOINT: head.checkpoint_id, META_FP32_ENTRY: head.entry,
    }
    try:
        manifest = manifest_of(parent=None, batch_id=GENESIS_BATCH, parts=parts, index=index, meta=meta)
    except ValueError as exc:
        raise GenesisError('the planned state is not valid: %s' % exc) from None
    _accepted(manifest, index)
    local.append((INDEX_NAME, index))
    return Plan(manifest, index, tuple(copies), tuple(local))


# ----------------------------------------------------------------------------
# The seed
# ----------------------------------------------------------------------------

def _layout(prefix: str) -> Layout:
    """The key builder of the run. A prefix that the guard refuses stops the seed before the first request."""
    guard.check_key(prefix, '%sstates/%s/manifest.json' % (prefix, '0' * 64))
    return Layout(prefix)


def _live(store: Any, src_bucket: Optional[str]) -> ReadOnlyStore:
    """The live side: a store that can only read."""
    source = store if src_bucket in (None, store.bucket) else R2Store(store._c, src_bucket, store.cfg)
    return ReadOnlyStore(source)


def _with_meta(live: ReadOnlyStore, head: LiveHead) -> Tuple[LiveHead, str]:
    """Read ``meta.json`` of the live master and check it against the master index.

    Returns the head with ``meta.json`` as a file of the master, and the etag of the bytes that were
    checked. Each difference is a ``GenesisError``: the seed then writes nothing.
    """
    try:
        blob = live.get(head.meta_key, max_bytes=MAX_META_BYTES)
    except NotFound:
        raise GenesisError('the live key %s does not exist in the bucket %s' % (head.meta_key, live.bucket)) from None
    except TooLarge:
        raise GenesisError('the live key %s is too large for a meta.json' % head.meta_key) from None
    try:
        meta = load_json(blob.data)
    except ValueError:
        meta = None
    _need(isinstance(meta, Mapping), 'meta.json of the live master is not a JSON object')
    _need(meta.get('master_digest') == head.master_digest, 'meta.json of the live master names another master digest')
    _need(meta.get('bf16_sha256') == head.entry, 'meta.json of the live master names another bf16_sha256')
    files = meta.get('files')
    _need(isinstance(files, Mapping), 'meta.json of the live master names no files')
    in_meta = {
        name: (info.get('sha256'), info.get('bytes')) if isinstance(info, Mapping) else None
        for name, info in files.items()
    }
    shards = [row for row in head.master if row[0] != MASTER_META]
    _need(in_meta == {name: (sha, size) for name, sha, size, _key in shards},
          'meta.json of the live master and the master index do not name the same shards')
    row = (MASTER_META, sha256_hex(blob.data), len(blob.data), head.meta_key)
    _need(all(named == row for named in head.master if named[0] == MASTER_META),
          'meta.json of the live master does not have the sha256 and the size that the master index names')
    return replace(head, master=tuple(sorted(shards + [row]))), blob.etag


def _live_infos(live: ReadOnlyStore, keys: Sequence[str]) -> Dict[str, Any]:
    """Size and etag of every live key."""
    infos: Dict[str, Any] = {}
    for key in keys:
        info = live.head(key)
        _need(info is not None, 'the live key %s does not exist in the bucket %s' % (key, live.bucket))
        infos[key] = info
    return infos


def _stored(store: Any, key: str) -> Optional[bytes]:
    try:
        return store.get(key, max_bytes=MAX_COMPLETE_BYTES).data
    except NotFound:
        return None


def _copy(store: Any, src: str, dst: str, size: int, *, src_etag: str, src_bucket: Optional[str],
          log: Callable[[str], None]) -> None:
    have = store.head(dst)
    if have is not None and have.size == size:
        log('kept %s (%d bytes)' % (dst, size))
        return
    log('copy %s to %s (%d bytes)' % (src, dst, size))
    store.copy(src, dst, src_etag=src_etag, src_bucket=src_bucket)
    made = store.head(dst)
    _need(made is not None and made.size == size, 'the copy %s does not have the %d bytes of its source' % (dst, size))


def _put_manifest(store: Any, key: str, body: bytes) -> None:
    """Create the manifest. A manifest that is there already must have the same bytes."""
    try:
        store.put(key, body, if_none_match=True)
    except PreconditionFailed:
        _need(_stored(store, key) == body, 'the manifest key %s holds other bytes' % key)


def _seed(store: Any, head: LiveHead, *, prefix: str, ckpt_prefix: str, src_bucket: Optional[str],
          log: Callable[[str], None]) -> Plan:
    lay = _layout(prefix)
    live_keys = [head.meta_key] + [source.key for source in _sources(head, ckpt_prefix)]
    _need(not any(key.startswith(prefix) for key in live_keys), 'a live key is under the prefix of the run')
    live = _live(store, src_bucket)
    head, meta_etag = _with_meta(live, head)
    infos = _live_infos(live, [source.key for source in _sources(head, ckpt_prefix)])
    # ``meta.json`` is copied only while its live key has the bytes that were checked.
    etags = dict({key: info.etag for key, info in infos.items()}, **{head.meta_key: meta_etag})
    made = plan(head, {key: info.size for key, info in infos.items()}, ckpt_prefix=ckpt_prefix)
    state_id, body = made.manifest.state_id, made.manifest.body()
    names = [name for _src, name, _size in made.copies] + [name for name, _data in made.local]
    keys = {name: guard.check_key(prefix, lay.state_object(state_id, name)) for name in names}
    manifest_key = guard.check_key(prefix, lay.state_manifest(state_id))

    stored = _stored(store, manifest_key)
    if stored == body:
        log('the state %s is complete already' % state_id)
        return made
    _need(stored is None, 'the manifest key %s holds other bytes' % manifest_key)
    for src, name, size in made.copies:
        _copy(store, src, keys[name], size, src_etag=etags[src], src_bucket=src_bucket, log=log)
    for name, data in made.local:
        store.put(keys[name], data)
    # Last: the manifest says that the state is complete.
    _put_manifest(store, manifest_key, body)
    log('the state %s is complete' % state_id)
    return made


def seed(store: Any, head: LiveHead, *, prefix: str, ckpt_prefix: str = DEFAULT_CKPT_PREFIX,
         src_bucket: Optional[str] = None, log: Callable[[str], None] = lambda text: None) -> str:
    """Put the first state of a run in R2 and return its state id.

    ``store`` is an ``r2.R2Store`` on the bucket of the run. The live keys are in the same bucket,
    or in ``src_bucket``. ``prefix`` is the prefix of the run. The guard refuses a prefix that is
    not the prefix of a run (``ShadowWriteError``) before the first request.

    ``GenesisError``: the facts are not valid, and a second run does not help. A ``StoreError`` of
    the store is passed on: run the seed again.
    """
    made = _seed(store, head, prefix=prefix, ckpt_prefix=ckpt_prefix, src_bucket=src_bucket, log=log)
    return made.manifest.state_id


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def _options(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='make the first state of a run from the live HEAD (read-only copy)')
    parser.add_argument('--run-id', required=True, help='4 to 32 characters, a-z and 0-9; names the R2 prefix')
    parser.add_argument('--bucket-config', required=True, help='file with the R2 endpoint, bucket and keys')
    parser.add_argument('--life-root', required=True, help='directory with CURRENT.json and epochs/')
    parser.add_argument('--master-index-dir', required=True, help='directory with <checkpoint id>.json')
    parser.add_argument('--checkpoint', default=None, help='checkpoint id; the default is the id of CURRENT.json')
    parser.add_argument('--ckpt-prefix', default=DEFAULT_CKPT_PREFIX, help='R2 prefix of the live checkpoints')
    parser.add_argument('--source-bucket', default=None, help='bucket of the live keys, if it is another bucket')
    return parser.parse_args(argv)


def _store(path: Path) -> R2Store:
    try:
        client, bucket = client_from_bucket_config(path)
    except (OSError, KeyError, ValueError, TypeError, AttributeError) as exc:
        # Only the type of the error: the text of an error must never show a part of the config.
        raise GenesisError('cannot use the bucket config: %s' % type(exc).__name__) from None
    store = R2Store(client, bucket)
    store.check_bucket()
    return store


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _options(argv)
    try:
        prefix = guard.shadow_prefix(args.run_id)
        head = read_live_head(Path(args.life_root), Path(args.master_index_dir), checkpoint=args.checkpoint)
        made = _seed(
            _store(Path(args.bucket_config)), head, prefix=prefix, ckpt_prefix=args.ckpt_prefix,
            src_bucket=args.source_bucket, log=_say,
        )
    except (GenesisError, StoreError, guard.ShadowWriteError) as exc:
        print(json.dumps({'error': '%s: %s' % (type(exc).__name__, exc)}))
        return 1
    objects = made.manifest.objects
    print(json.dumps({
        'genesis_state': made.manifest.state_id, 'checkpoint': head.checkpoint_id,
        'fp32_master_digest': head.master_digest, 'objects': len(objects),
        'bytes': sum(ref.size for ref in objects),
    }))
    return 0


__all__ = [
    'DEFAULT_CKPT_PREFIX', 'GenesisError', 'LiveHead', 'Plan', 'live_head', 'main', 'plan', 'read_live_head', 'seed',
]


if __name__ == '__main__':
    sys.exit(main())
