"""The live reserve cache never becomes a checkout write or a mirror rollback."""
from __future__ import annotations

import json
import shutil
from argparse import Namespace
from pathlib import Path

import pytest

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.envs.math_pool import load_pool
from sn120_v2.envs.reserve import MANIFEST, WITHDRAWN, empty_manifest, load_reserve, reserve_dir
from sn120_v2.envs.reserve.__main__ import ingest
from sn120_v2.errors import VerifyError
from sn120_v2.params import Params
from sn120_v2.pool_runtime import PoolRuntime, mirror_reserve
from sn120_v2.state import State
from sn120_v2.store import LocalStore


def make_reserve(root: Path, *, prefix: str = "simnew", number: int = 0) -> dict:
    (root / "shards").mkdir(parents=True, exist_ok=True)
    (root / WITHDRAWN).write_bytes(b"")
    row = {"id": f"{prefix}:{number}", "env": "affine_math", "problem": f"live state problem {prefix} {number}", "answer": str(number),
           "source": f"sim@rev:{number}", "license": "CC0-1.0", "added_at": "2026-11-18",
           "audit": {"verdict": "keep", "method": "sim", "ref": "sim"}}
    raw = canonical_json(row) + b"\n"
    name = f"{number + 1:04d}-sim.jsonl"
    (root / "shards" / name).write_bytes(raw)
    line = {"shard": name, "sha256": sha256_hex(raw), "count": 1, "added_at": "2026-11-18"}
    (root / MANIFEST).write_bytes(b"".join(canonical_json(row) + b"\n" for row in
                                          [line, {"withdrawn": {"sha256": sha256_hex(b""), "count": 0}}]))
    return line


def runtime(store: LocalStore, root: Path, *, seed: Path | None = None) -> PoolRuntime:
    params = Params(rotation_enabled=True, rotation_active_size=2)
    return PoolRuntime(store, params, State(params=params), root=root, base=load_pool(), initial_ids=("math:0", "math:1"), seed=seed)


def test_seed_migration_and_round_open_leave_the_packaged_checkout_untouched(tmp_path):
    seed = tmp_path / "checkout" / "reserve"
    make_reserve(seed)
    before = {path.relative_to(seed): path.read_bytes() for path in seed.rglob("*") if path.is_file()}
    store = LocalStore(tmp_path / "store")
    live = tmp_path / "state" / "reserve"
    pool = runtime(store, live, seed=seed)
    pool._publish_retired_local(type("Decision", (), {"retired_archive": ()})())
    mirror_reserve(store, live)
    assert (live / MANIFEST).read_bytes() == (seed / MANIFEST).read_bytes()
    assert before == {path.relative_to(seed): path.read_bytes() for path in seed.rglob("*") if path.is_file()}

    extension = make_reserve(tmp_path / "ingest", prefix="simnext", number=1)
    (live / "shards" / extension["shard"]).write_bytes((tmp_path / "ingest" / "shards" / extension["shard"]).read_bytes())
    rows = [json.loads(line) for line in (live / MANIFEST).read_bytes().splitlines()]
    rows[-1:-1] = [extension]
    (live / MANIFEST).write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    # A reset of packaged MANIFEST.jsonl cannot remove the local extension.
    (seed / MANIFEST).write_bytes(empty_manifest())
    restored = runtime(store, live, seed=seed)
    assert set(restored.reserve.queue) == {"simnew:0", "simnext:1"}
    assert restored.reserve.manifest_sha == sha256_hex((live / MANIFEST).read_bytes())


def test_migrates_a_verified_dirty_checkout_extension(tmp_path):
    legacy = tmp_path / "old-live"
    make_reserve(legacy)
    mirror = LocalStore(tmp_path / "store")
    mirror_reserve(mirror, legacy)
    extension = make_reserve(tmp_path / "extension", prefix="simnext", number=1)
    (legacy / "shards" / extension["shard"]).write_bytes((tmp_path / "extension" / "shards" / extension["shard"]).read_bytes())
    rows = [json.loads(line) for line in (legacy / MANIFEST).read_bytes().splitlines()]
    rows[-1:-1] = [extension]
    (legacy / MANIFEST).write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    live = tmp_path / "state" / "reserve"
    restored = runtime(mirror, live, seed=legacy)
    assert set(restored.reserve.queue) == {"simnew:0", "simnext:1"}
    assert (live / MANIFEST).read_bytes() == (legacy / MANIFEST).read_bytes()


def test_restore_only_writes_live_state_and_preserves_a_public_prefix_extension(tmp_path):
    seed = reserve_dir()
    before = {path.relative_to(seed): path.read_bytes() for path in seed.rglob("*") if path.is_file()}
    store = LocalStore(tmp_path / "store")
    live = tmp_path / "state" / "reserve"
    make_reserve(live)
    runtime(store, live)
    mirror_reserve(store, live)
    public = (live / MANIFEST).read_bytes()
    extension = make_reserve(live, prefix="simnext", number=1)
    manifest = [json.loads(line) for line in public.splitlines()]
    manifest[-1:-1] = [extension]
    extended = b"".join(canonical_json(row) + b"\n" for row in manifest)
    (live / MANIFEST).write_bytes(extended)

    restored = runtime(store, live)

    assert (live / MANIFEST).read_bytes() == extended
    assert restored.reserve.manifest_sha == sha256_hex(extended)
    assert set(restored.reserve.queue) == {"simnew:0", "simnext:1"}
    assert before == {path.relative_to(seed): path.read_bytes() for path in seed.rglob("*") if path.is_file()}


def test_empty_packaged_seed_is_filled_and_survives_a_checkout_reset(tmp_path):
    store = LocalStore(tmp_path / "store")
    live = tmp_path / "state" / "reserve"
    make_reserve(live)
    mirror_reserve(store, live)
    expected = (live / MANIFEST).read_bytes()
    seed = tmp_path / "fresh-seed"
    (seed / "shards").mkdir(parents=True)
    (seed / MANIFEST).write_bytes(empty_manifest())
    (seed / WITHDRAWN).write_bytes(b"")
    shutil.rmtree(live)

    restored = runtime(store, live, seed=seed)
    assert (live / MANIFEST).read_bytes() == expected
    assert restored.reserve.queue == ("simnew:0",)

    # A checkout reset may replace the immutable seed; the live state is untouched.
    (seed / MANIFEST).write_bytes(empty_manifest())
    (seed / WITHDRAWN).write_bytes(b"")
    for shard in (seed / "shards").iterdir():
        shard.unlink()
    assert (live / MANIFEST).read_bytes() == expected
    assert runtime(store, live, seed=seed).reserve.manifest_sha == sha256_hex(expected)
    withdrawn = canonical_json({"id": "simnew:0", "reason": "stale packaged withdrawal", "added_at": "2026-11-18"}) + b"\n"
    (seed / WITHDRAWN).write_bytes(withdrawn)
    (seed / MANIFEST).write_bytes(canonical_json({"withdrawn": {"sha256": sha256_hex(withdrawn), "count": 1}}) + b"\n")
    shutil.rmtree(live)
    assert runtime(store, live, seed=seed).reserve.manifest_sha == sha256_hex(expected)
    assert (live / WITHDRAWN).read_bytes() == store.get(layout.RESERVE_WITHDRAWN)


def test_divergent_local_history_fails_before_it_is_replaced(tmp_path):
    store = LocalStore(tmp_path / "store")
    live = tmp_path / "state" / "reserve"
    make_reserve(live)
    mirror_reserve(store, live)
    divergent = tmp_path / "divergent"
    make_reserve(divergent, prefix="other")
    before = (divergent / MANIFEST).read_bytes()

    with pytest.raises(VerifyError, match="does not extend"):
        runtime(store, divergent)
    assert (divergent / MANIFEST).read_bytes() == before


def test_cli_ingest_writes_only_the_live_state_root(tmp_path):
    seed = reserve_dir()
    before = {path.relative_to(seed): path.read_bytes() for path in seed.rglob("*") if path.is_file()}
    source = tmp_path / "input.jsonl"
    source.write_text('{"problem":"live CLI reserve problem","answer":"1"}\n')
    args = Namespace(input=str(source), source="testset@deadbeef", license="CC0-1.0", prefix="omni",
                     audit_method="unit-test", audit_ref="v2/tests/test_reserve_live_state.py", reject_mc=False,
                     env="affine_math", readmit=None, state_dir=tmp_path / "state")
    report = ingest(args)
    assert report["ingested"] == 1
    root = tmp_path / "state" / "reserve"
    assert load_reserve(root)[0].shards[-1].shard == report["shard"]
    assert (root / "shards" / report["shard"]).is_file()
    assert before == {path.relative_to(seed): path.read_bytes() for path in seed.rglob("*") if path.is_file()}
