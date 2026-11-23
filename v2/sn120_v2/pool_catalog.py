"""Public, verified pool artifacts for replication without a git checkout."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from . import layout
from .canonical import sha256_hex
from .envs.math_pool import MathPool
from .envs.reserve import load_reserve
from .errors import VerifyError
from .pool_runtime import load_active
from .store import LocalStore


def _mirror(contract: Any, key: str, store: LocalStore) -> None:
    """Read public bytes through the existing header-sending downloader."""
    if not store.exists(key):
        from .publisher import fetch
        from urllib.parse import unquote, urlparse
        import urllib.request

        location = contract.url(key)
        if location.startswith('file://'):
            raw = Path(unquote(urlparse(location).path)).read_bytes()
        else:
            request = urllib.request.Request(location, headers={'User-Agent': 'sn120-v2-pool-replicate', 'Cache-Control': 'no-cache'})
            with urllib.request.urlopen(request, timeout=30) as reply:
                raw = reply.read()
        # Each content-addressed object is verified below; manifest, active-list
        # and rotation hashes bind the complete catalog.
        store.put(key, raw)


def public_catalog(contract: Any, work: Path, view: Mapping[str, Any], base: MathPool):
    """Read active list + mirrored reserve, verify sha256, return catalog.

    The cache is scoped by round and manifest. A later manifest may replace
    the public withdrawn pointer without mixing retired/reserve files from an
    earlier round.
    """
    block = view['pool']
    store = LocalStore(Path(work) / 'public-pool' / str(view['round_id']) / block['reserve_manifest_sha'])
    _mirror(contract, layout.active_pool(block['active_sha']), store)
    active = load_active(store, block['active_sha'])
    if len(active) != block['size']:
        raise VerifyError('round active set size differs from its public pool block')
    _mirror(contract, layout.RESERVE_MANIFEST, store)
    _mirror(contract, layout.RESERVE_WITHDRAWN, store)
    root = store.root / 'reserve'
    (root / 'shards').mkdir(parents=True, exist_ok=True)
    manifest = store.get(layout.RESERVE_MANIFEST)
    (root / 'MANIFEST.jsonl').write_bytes(manifest)
    (root / 'withdrawn.jsonl').write_bytes(store.get(layout.RESERVE_WITHDRAWN))
    if sha256_hex(manifest) != block['reserve_manifest_sha']:
        raise VerifyError('mirrored reserve manifest differs from the round')
    rows = [json.loads(line) for line in manifest.splitlines()]
    for row in rows[:-1]:
        name, sha = row['shard'], row['sha256']
        key = layout.reserve_shard(name, sha)
        _mirror(contract, key, store)
        (root / 'shards' / name).write_bytes(store.get(key))
    _mirror(contract, layout.RETIRED_POOL, store)
    _mirror(contract, layout.RETIRED_TASKS, store)
    (root / 'retired.json').write_bytes(store.get(layout.RETIRED_POOL))
    (root / 'retired_tasks.json').write_bytes(store.get(layout.RETIRED_TASKS))
    _, catalog = load_reserve(root, with_catalog=True, allow_retired=block.get('rule_version') == 'affine-pool-rotation-v2')
    return catalog, active


def public_rotation_inputs(contract: Any, work: Path, view: Mapping[str, Any]):
    """Get the published rotation and its source snapshot into a LocalStore."""
    store = LocalStore(Path(work) / 'public-pool' / str(view['round_id']))
    for key in (layout.pool_rotation(view['round_id']), layout.irt_snapshot(view['irt_snapshot_id'])):
        _mirror(contract, key, store)
    rotation = json.loads(store.get(layout.pool_rotation(view['round_id'])))
    if rotation.get('rule_version') == 'affine-pool-rotation-v2':
        _mirror(contract, layout.pool_ranking(view['round_id']), store)
    if rotation.get('prev_state_sha'):
        _mirror(contract, layout.pool_rotation_state(rotation['prev_state_sha']), store)
    if rotation.get('initial_active_sha'):
        _mirror(contract, layout.active_pool(rotation['initial_active_sha']), store)
    source = rotation.get('source_snapshot_id')
    if source:
        _mirror(contract, layout.irt_snapshot(source), store)
        if rotation.get('prev_state_sha'):
            previous = json.loads(store.get(layout.pool_rotation_state(rotation['prev_state_sha'])))
            snapshot = json.loads(store.get(layout.irt_snapshot(source)))
            if set(snapshot['tasks']) != set(previous['active_ids']):
                # The previous close proves uncovered rows were unscored; a
                # normally scored rotation still refuses a source mismatch.
                _mirror(contract, layout.round_file(previous['round_id']), store)
    return store, rotation
