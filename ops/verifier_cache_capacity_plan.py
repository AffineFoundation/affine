"""Read-only proof for reusing protected same-host checkpoint replicas.

This is an operator preflight, not retirement or permission to mutate a cache.
It hashes complete explicitly named replicas and reports an upper bound on
space recoverable by replacing a separate single-link replica with hardlinks.
Protected checkpoint identities and every logical path remain in the plan.
No filesystem mutation, remote commands, GPU imports or credential handling.
"""
import hashlib
import json
import re
import stat
from pathlib import Path


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _snapshot(root, files):
    root = Path(root)
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise ValueError('ordinary absolute cache directory required')
    if {p.name for p in root.iterdir()} != set(files):
        raise ValueError('exact checkpoint membership required')
    result = {}
    for name, expected in files.items():
        path = root / name
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected['size']:
            raise ValueError('ordinary checkpoint object with exact size required')
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(block)
        after = path.lstat()
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink', 'st_blocks')
        if any(getattr(before, field) != getattr(after, field) for field in fields):
            raise ValueError('checkpoint changed during hashing')
        if hasher.hexdigest() != expected['sha256']:
            raise ValueError('checkpoint byte hash mismatch')
        result[name] = dict(device=before.st_dev, inode=before.st_ino,
                            size=before.st_size, allocated_bytes=before.st_blocks * 512,
                            links=before.st_nlink, mtime_ns=before.st_mtime_ns,
                            ctime_ns=before.st_ctime_ns, sha256=hasher.hexdigest())
    return result


def plan_reuse(checkpoint, files, keeper, replica, protected, *, free_bytes,
               artifact_bytes, reserve_bytes):
    """Prove duplicate bytes, never authorize cleanup or assume runtime idle.

    A single-link destination is required for a guaranteed reclaim estimate.
    Alias clusters with unknown other links intentionally yield no estimate.
    A separate process/FD/mmap/GPU barrier and authenticated R2/runtime receipts
    are required before any future operator replacement mechanism may act.
    """
    if not isinstance(checkpoint, str) or not re.fullmatch('[0-9a-f]{64}', checkpoint):
        raise ValueError('checkpoint identity required')
    if not isinstance(files, dict) or not 1 <= len(files) <= 32:
        raise ValueError('complete file map required')
    for name, meta in files.items():
        if (not re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_.-]*', name)
                or not isinstance(meta, dict) or type(meta.get('size')) is not int
                or meta['size'] <= 0 or not isinstance(meta.get('sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', meta['sha256'])):
            raise ValueError('safe exact file metadata required')
    if 'config.json' not in files or not any(n.endswith('.safetensors') for n in files):
        raise ValueError('model file map required')
    if canonical_digest({n: m['sha256'] for n, m in files.items()}) != checkpoint:
        raise ValueError('canonical checkpoint identity mismatch')
    if (not isinstance(protected, list) or checkpoint not in protected
            or any(not isinstance(c, str) or not re.fullmatch('[0-9a-f]{64}', c) for c in protected)):
        raise ValueError('preserved current/pending checkpoint protection required')
    if any(type(v) is not int or v < 0 for v in (free_bytes, artifact_bytes, reserve_bytes)):
        raise ValueError('explicit nonnegative integer capacity budget required')
    if Path(keeper) == Path(replica):
        raise ValueError('distinct logical cache paths required')
    first = _snapshot(keeper, files)
    second = _snapshot(replica, files)
    eligible = True
    reclaim = 0
    for name in files:
        a, b = first[name], second[name]
        if a['device'] != b['device']:
            raise ValueError('same filesystem required')
        if a['inode'] == b['inode']:
            continue
        if b['links'] != 1:
            eligible = False
        else:
            reclaim += b['allocated_bytes']
    if not eligible:
        reclaim = 0
    # Detect replacement/inventory changes anywhere during the second full hash.
    for root, snapshot in ((Path(keeper), first), (Path(replica), second)):
        if {p.name for p in root.iterdir()} != set(files):
            raise ValueError('membership changed during preflight')
        for name, row in snapshot.items():
            current = (root / name).lstat()
            if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns,
                current.st_ctime_ns, current.st_nlink) != (row['device'], row['inode'],
                row['size'], row['mtime_ns'], row['ctime_ns'], row['links']):
                raise ValueError('cache changed during preflight')
    return dict(kind='read-only-protected-checkpoint-reuse-proof-v1', checkpoint=checkpoint,
                keeper=str(keeper), replica=str(replica), protected_checkpoints=list(protected),
                keeper_files=first, replica_files=second, duplicate_bytes_verified=True,
                single_link_replacement_candidate=eligible,
                potential_reclaimed_bytes=reclaim, free_bytes=free_bytes,
                required_free_bytes=artifact_bytes + reserve_bytes,
                sufficient_capacity_now=free_bytes >= artifact_bytes + reserve_bytes,
                sufficient_capacity_after_prospective_reuse=free_bytes + reclaim >= artifact_bytes + reserve_bytes,
                mutation_authorized=False, runtime_idle_verified=False,
                archive_authenticated=False, logical_paths_must_be_preserved=True)
