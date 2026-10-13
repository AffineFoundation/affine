"""Reproducible source packaging for the prospective MiMo cutover.

Building a release is local preparation. It neither publishes discovery nor
changes an epoch, a service, a payout assessment, or an optimizer lineage.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

MODEL_ID = 'XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B'
MODEL_REVISION = '2367e865d009c13ac81713a2878291d33ab28177'
VERL_REVISION = 'a2ad9f6160b03ff2d47e59832bfb6b289f37c917'
VERSION = 'affine-mimo-release-v1'
SOURCE_ROOTS = ('subnet', 'vendor/mimo_verl_pinned')
# EnvironmentSpec authenticates the complete native source directories, including
# grader assets and package metadata. Omitting non-Python files changes its hash
# on the miner even when the Python implementation is identical.
ENVIRONMENT_ROOTS = ('subnet/vendor/legacy/rollouts/envs',
                     'subnet/vendor/research/environments')
FORBIDDEN_PARTS = frozenset({'state', '.git', '.venv', '__pycache__', 'private',
                             'credentials', 'wallets', 'hotkeys', 'coldkeys'})
MAX_FILE_BYTES = 16 * 1024**2


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def source_bundle(record, *, url=None):
    """Return the miner-download identity, which hashes ARCHIVE bytes.

    ``source_sha256`` separately identifies the uncompressed file inventory.
    It must never be used as the download SHA: the signed-source launcher
    authenticates the gzip archive before extraction.
    """
    expected = record.get('archive_sha256')
    size = record.get('archive_bytes')
    if (record.get('version') != VERSION or type(expected) is not str
            or len(expected) != 64 or not set(expected) <= set('0123456789abcdef')
            or type(size) is not int or not 0 < size <= 32*1024**2
            or digest(record.get('source')) != record.get('source_sha256')):
        raise ValueError('verified release archive/inventory identities required')
    result = dict(sha256=expected, size=size)
    if url is not None:
        from .backend_jobs import r2_url
        result['url'] = r2_url(url, 'GET')
    return result


def _path(name):
    path = PurePosixPath(name)
    if (not isinstance(name, str) or not name or path.is_absolute()
            or str(path) != name or '..' in path.parts or '\\' in name
            or set(path.parts) & FORBIDDEN_PARTS
            or any(x in path.name.lower() for x in ('.private', '.seed', '.env', '.pem', '.key'))):
        raise ValueError('release source path is not public code')
    return path


def task_assets(settings):
    """Only operator-declared, content-pinned public task snapshots are assets."""
    assets = settings.get('task_assets', {})
    if type(assets) is not dict or len(assets) > 256:
        raise ValueError('bounded explicit task asset inventory required')
    result = {}
    for name, row in assets.items():
        path = _path(name)
        if len(path.parts) < 2 or path.parts[0] != 'assets':
            raise ValueError('task asset must use its explicit assets/ path')
        if (type(row) is not dict or set(row) != {'sha256', 'bytes'}
                or type(row['bytes']) is not int or not 0 < row['bytes'] <= MAX_FILE_BYTES
                or type(row['sha256']) is not str or len(row['sha256']) != 64
                or not set(row['sha256']) <= set('0123456789abcdef')):
            raise ValueError('exact task asset SHA256 and bounded byte count required')
        result[name] = dict(row)
    return result


def source_files(root, *, extra=()):
    """Inventory public code, never recursively archive the working directory."""
    root = Path(root).resolve(strict=True)
    names = set(extra)
    for dirname in SOURCE_ROOTS:
        directory = root / dirname
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError('required release source directory missing: ' + dirname)
        for path in directory.rglob('*'):
            if path.is_symlink():
                raise ValueError('symlink in release source: ' + str(path.relative_to(root)))
            if path.is_file() and (path.suffix in ('.py', '.json', '.jinja') or path.name.startswith('LICENSE')):
                if '__pycache__' not in path.parts:
                    names.add(path.relative_to(root).as_posix())
    for dirname in ENVIRONMENT_ROOTS:
        directory = root / dirname
        if directory.is_symlink():
            raise ValueError('symlink in environment source: ' + dirname)
        for path in directory.rglob('*'):
            if path.is_symlink():
                raise ValueError('symlink in environment source: ' + str(path.relative_to(root)))
            if path.is_file() and path.suffix != '.pyc' and '__pycache__' not in path.parts:
                names.add(path.relative_to(root).as_posix())
    result = {}
    for name in sorted(names):
        _path(name)
        path = root / name
        if not path.is_file() or path.is_symlink() or path.resolve() != path:
            raise ValueError('missing or escaping release file: ' + name)
        raw = path.read_bytes()
        if len(raw) > MAX_FILE_BYTES:
            raise ValueError('model/data objects do not belong in the source bundle')
        result[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return result


def build(root, destination, *, settings, extra=()):
    """Produce byte-reproducible archive and immutable local release record.

    The record explicitly says prepared, not activated. Source is read twice to
    catch a concurrent agent editing a file between inventory and compression.
    """
    root = Path(root).resolve(strict=True)
    destination = Path(destination)
    assets = task_assets(settings)
    inventory = source_files(root, extra=tuple(extra) + tuple(assets))
    if {n: r for n, r in inventory.items() if n.startswith('assets/')} != assets:
        raise ValueError('task assets differ from explicit pinned release settings')
    if settings.get('model') != {'id': MODEL_ID, 'revision': MODEL_REVISION}:
        raise ValueError('selected immutable MiMo model revision required')
    if settings.get('reference_revision') != VERL_REVISION:
        raise ValueError('pinned MiMo reference revision required')
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode='wb', mtime=0, filename='') as compressed:
        with tarfile.open(fileobj=compressed, mode='w|') as archive:
            for name, row in inventory.items():
                raw = (root / name).read_bytes()
                if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
                    raise ValueError('source changed during release preparation')
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(raw), 0o644, 0
                archive.addfile(info, io.BytesIO(raw))
    archive_bytes = output.getvalue()
    record = dict(version=VERSION, status='prepared-not-activated', settings=settings,
                  source=inventory, source_sha256=digest(inventory),
                  archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
                  archive_bytes=len(archive_bytes))
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Refuse replacing a previously reviewed release, even with the same label.
    if destination.exists():
        raise FileExistsError(destination)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + '.', dir=destination.parent))
    try:
        for name, raw in [('source.tar.gz', archive_bytes), ('RELEASE.json', canonical(record))]:
            with (temporary / name).open('xb') as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            import shutil
            shutil.rmtree(temporary)
    return record


def verify(directory):
    """Verify closure without extracting or importing untrusted archive files."""
    directory = Path(directory)
    record = json.loads((directory / 'RELEASE.json').read_bytes())
    raw = (directory / 'source.tar.gz').read_bytes()
    if (record.get('version') != VERSION or len(raw) != record['archive_bytes']
            or hashlib.sha256(raw).hexdigest() != record['archive_sha256']
            or digest(record['source']) != record['source_sha256']):
        raise ValueError('release digest mismatch')
    assets = task_assets(record.get('settings', {}))
    if {n: r for n, r in record['source'].items() if n.startswith('assets/')} != assets:
        raise ValueError('release task assets differ from pinned settings')
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        members = archive.getmembers()
        if len(members) != len(record['source']) or {m.name for m in members} != set(record['source']):
            raise ValueError('release source membership mismatch')
        for member in members:
            _path(member.name)
            expected = record['source'][member.name]
            if not member.isfile() or member.size != expected['bytes'] or member.size > MAX_FILE_BYTES:
                raise ValueError('release member type/size mismatch')
            body = archive.extractfile(member).read()
            if hashlib.sha256(body).hexdigest() != expected['sha256']:
                raise ValueError('release source content mismatch')
    return record
