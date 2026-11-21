"""Live reserve deployment, reload and Generator epoch ingestion."""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.envs.reserve import MANIFEST, WITHDRAWN, empty_manifest, load_reserve, prepare_live_reserve
from sn120_v2.envs.reserve.generator import ingest_generator
from sn120_v2.envs.reserve.__main__ import main
from sn120_v2.errors import V2Error, VerifyError
from sn120_v2.pool_runtime import mirror_reserve
from sn120_v2.store import LocalStore


def write_intake_done(args: Namespace, count: int | None = None) -> None:
    raw = Path(args.input).read_bytes()
    Path(args.input).with_suffix(".done").write_text(json.dumps({
        "epoch": args.epoch, "lines": len(raw.splitlines()) if count is None else count,
        "sha256": sha256_hex(raw), "format": "generator-intake-v2"}))


def reserve(root: Path, count: int = 1, prefix: str = "seed", start: int = 0) -> None:
    (root / "shards").mkdir(parents=True, exist_ok=True)
    (root / WITHDRAWN).write_bytes(b"")
    manifest = [json.loads(line) for line in (root / MANIFEST).read_bytes().splitlines()] if (root / MANIFEST).exists() else [{"withdrawn": {"sha256": sha256_hex(b""), "count": 0}}]
    for number in range(start, start + count):
        row = {"id": f"{prefix}:{number}", "env": "affine_math", "problem": f"Unique reserve problem {prefix} {number}", "answer": str(number),
               "source": f"test@rev:{number}", "license": "CC0-1.0", "added_at": "2026-11-20",
               "audit": {"verdict": "keep", "method": "unit", "ref": "test"}}
        raw = canonical_json(row) + b"\n"
        name = f"{number + 1:04d}-{prefix}.jsonl"
        (root / "shards" / name).write_bytes(raw)
        manifest[-1:-1] = [{"shard": name, "sha256": sha256_hex(raw), "count": 1, "added_at": "2026-11-20"}]
    if not manifest:
        manifest = [{"withdrawn": {"sha256": sha256_hex(b""), "count": 0}}]
    (root / MANIFEST).write_bytes(b"".join(canonical_json(row) + b"\n" for row in manifest))


def test_live_seed_prefix_extension_divergence_and_seed_immutable(tmp_path, monkeypatch):
    seed, live = tmp_path / "seed", tmp_path / "live"
    reserve(seed, 2)
    before = {path.relative_to(seed): path.read_bytes() for path in seed.rglob("*") if path.is_file()}
    monkeypatch.setenv("SN120_RESERVE_DIR", str(live))
    prepare_live_reserve(live, seed)
    assert (live / MANIFEST).read_bytes() == (seed / MANIFEST).read_bytes()
    assert {path.relative_to(seed): path.read_bytes() for path in seed.rglob("*") if path.is_file()} == before
    reserve(live, 1, "live", 2)
    reserve(seed, 1, "future", 2)
    with pytest.raises(VerifyError, match="diverges"):
        prepare_live_reserve(live, seed)
    # A strict older live prefix gets the future packaged suffix.
    live2 = tmp_path / "older"
    live2.mkdir()
    (live2 / MANIFEST).write_bytes(b"\n".join((seed / MANIFEST).read_bytes().splitlines()[:1] + [(seed / MANIFEST).read_bytes().splitlines()[-1]]) + b"\n")
    (live2 / WITHDRAWN).write_bytes(b"")
    monkeypatch.setenv("SN120_RESERVE_DIR", str(live2))
    prepare_live_reserve(live2, seed)
    assert (live2 / MANIFEST).read_bytes() == (seed / MANIFEST).read_bytes()
    diverged = tmp_path / "diverged"
    reserve(diverged, 1, "other")
    monkeypatch.setenv("SN120_RESERVE_DIR", str(diverged))
    with pytest.raises(VerifyError, match="diverges"):
        prepare_live_reserve(diverged, seed)


def test_reload_at_open_uses_extension_and_keeps_old_on_bad_candidate(tmp_path, monkeypatch):
    from test_rotation_controller import rotated_world, world_pool
    from sn120_v2.irt import IRTSampler
    world, pool = rotated_world(tmp_path, reserve_count=2, rotation_rule="info")
    root = tmp_path / "reserve"
    old_sha = pool.reserve.manifest_sha
    extra = tmp_path / "extra"
    reserve(extra, 1, "gen", 2)
    (root / "shards" / "0003-gen.jsonl").write_bytes((extra / "shards" / "0003-gen.jsonl").read_bytes())
    lines = (root / MANIFEST).read_bytes().splitlines()
    lines[-1:-1] = [canonical_json({"shard": "0003-gen.jsonl", "sha256": sha256_hex((root / "shards" / "0003-gen.jsonl").read_bytes()),
                                    "count": 1, "added_at": "2026-11-20"})]
    (root / MANIFEST).write_bytes(b"\n".join(lines) + b"\n")
    snapshot = world.controller.sampler.public_snapshot()
    decision, sampler = pool.open("r000001", snapshot, world.controller.state)
    assert pool.reserve.manifest_sha != old_sha
    assert pool.task("gen:2").answer == "2"
    pool.commit(decision)
    (root / MANIFEST).write_bytes(empty_manifest())
    pool._reload_at_open()
    assert pool.reserve.manifest_sha == decision.reserve_manifest_sha
    (root / MANIFEST).write_bytes(b"\n".join(lines) + b"\n")
    draw = sampler.draw(world.controller.state.head, "r1" * 32, 1)
    boundary = sampler.update(world.controller.state.head, draw, {"null": ["F"]})
    world.store.put(layout.irt_snapshot(boundary), canonical_json(sampler.public_snapshot()))
    world.controller.state.snapshot_id = boundary
    # Give the synthetic boundary a normal scored close record; an old
    # no_decision rotation now deliberately requires its actual close record.
    world.controller.state.rounds.append(type("Round", (), {"round_id": decision.round_id, "outcome": "null"})())
    second, _ = pool.open("r000002", sampler.public_snapshot(), world.controller.state)
    assert second.active_ids
    pool.commit(second)
    previous = pool.reserve
    previous_catalog = pool.catalog
    (root / "retired.json").write_bytes(b"[]")
    (root / "shards" / "0001-sim.jsonl").write_bytes(b"{}\n")
    pool._reload_at_open()
    assert pool.catalog is previous_catalog and pool.catalog.task("gen:2").answer == "2"
    assert pool.reserve == previous and pool.catalog.task("gen:2").answer == "2"
    pool.open("r000003", sampler.public_snapshot(), world.controller.state)


def input_rows() -> list[dict]:
    return [
        {"problem": "Unique generator compute 40 + 2", "answer": "42", "source_id": "a", "prior_p": .5},
        {"problem": "  UNIQUE GENERATOR COMPUTE 40 + 2  ", "answer": "42", "source_id": "b", "prior_p": .5},
        {"problem": "Prove that this unique object has positive length", "answer": "1", "source_id": "c", "prior_p": .5},
        {"problem": "Choose: \\textbf{(A)} 1 \\qquad \\textbf{(B)} 2", "answer": "1", "source_id": "d", "prior_p": .5},
        {"problem": "A row absent from prior", "answer": "1", "source_id": "e", "prior_p": .5},
        {"problem": "A restricted prior row", "answer": "1", "source_id": "f", "prior_p": .5},
    ]


def generator_args(tmp_path: Path, rows: list[dict], epoch: str = "20261120T0400Z") -> Namespace:
    source = tmp_path / "input.jsonl"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"schema": "affine-generator-results-v1", "epoch": epoch,
                                  "counts": {"accepted": len(rows)}, "king": "a" * 64,
                                  "pool": "sn120-math-prior-v1:45b793b4d00d1f7225d34fd35e58018612a0ded00d96a9c23b750cfb18994115"}))
    keys = [sha256_hex(__import__("sn120_v2.envs.reserve", fromlist=["normalise"]).normalise(row["problem"]).encode()) for row in rows]
    index = {key: {"source": "dataset@rev", "license": "mit"} for key in keys}
    if len(keys) > 4:
        index.pop(keys[4])
    if len(keys) > 5:
        index[keys[5]] = {"source": "dataset@rev", "license": "cc-by-nc-4.0"}
    path = tmp_path / "prior.json"
    path.write_text(json.dumps({"schema": "sn120-math-prior-index-v1", "pool": "sn120-math-prior-v1:45b793b4d00d1f7225d34fd35e58018612a0ded00d96a9c23b750cfb18994115",
                               "rows": len(index), "problems": index}))
    (tmp_path / "active.json").write_text("[]")
    return Namespace(input=str(source), epoch=epoch, results=str(results), done=None, prior_index=str(path), active=str(tmp_path / "active.json"), public_url="https://affine.io", reject_mc=True)


def test_generator_epoch_ledger_filters_idempotence_and_atomic_write(tmp_path, monkeypatch):
    seed, root = tmp_path / "seed", tmp_path / "reserve"
    reserve(seed, 0)
    prepare_live_reserve(root, seed)
    monkeypatch.setenv("SN120_RESERVE_DIR", str(root))
    args = generator_args(tmp_path, input_rows())
    report = ingest_generator(args)
    assert report["rows"] == 6 and report["ingested"] == 1
    assert report["rejected"] == {"duplicate_same_file": 1, "proof": 1, "not_in_prior": 1, "mc": 1, "license": 1}
    assert report["shards"][0].endswith("-mit.jsonl")
    row = json.loads((root / "shards" / report["shards"][0]).read_text())
    assert row["id"] == "gen:0" and row["prior_p"] == .5
    assert row["audit"]["verdict"] == "keep" and "prior_source=dataset@rev" in row["source"]
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert ingest_generator(args)["already_ingested"]
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    Path(args.input).write_text(Path(args.input).read_text().replace('"answer": "42"', '"answer": "43"', 1))
    with pytest.raises(Exception) as changed:
        ingest_generator(args)
    assert type(changed.value).__name__ == "InputChanged"
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    store = LocalStore(tmp_path / "store")
    mirror_reserve(store, root)
    assert store.get(layout.RESERVE_MANIFEST) == (root / MANIFEST).read_bytes()
    for shard in load_reserve(root)[0].shards:
        assert store.get(layout.reserve_shard(shard.shard, shard.sha256)) == (root / "shards" / shard.shard).read_bytes()


def test_generator_incomplete_and_zero_kept_epoch(tmp_path, monkeypatch):
    seed, root = tmp_path / "seed", tmp_path / "reserve"
    reserve(seed, 0)
    monkeypatch.setenv("SN120_RESERVE_DIR", str(root))
    prepare_live_reserve(root, seed)
    args = generator_args(tmp_path, input_rows()[:1])
    Path(args.results).write_text(json.dumps(json.loads(Path(args.results).read_text()) | {"counts": {"accepted": 2}}))
    with pytest.raises(Exception) as incomplete:
        ingest_generator(args)
    assert type(incomplete.value).__name__ == "Incomplete"
    assert not (root / "generator-ingested.jsonl").exists()
    args = generator_args(tmp_path, input_rows()[2:3], "20261120T0800Z")
    before = (root / MANIFEST).read_bytes()
    report = ingest_generator(args)
    assert report["ingested"] == 0 and report["rows"] == 1
    assert (root / MANIFEST).read_bytes() == before
    assert len((root / "generator-ingested.jsonl").read_text().splitlines()) == 1


def test_generator_cli_exit_3_and_4_and_dotted_license_shard(tmp_path, monkeypatch, capsys):
    seed, root = tmp_path / "seed", tmp_path / "reserve"
    reserve(seed, 0)
    monkeypatch.setenv("SN120_RESERVE_DIR", str(root))
    prepare_live_reserve(root, seed)
    row = {"problem": "Unique generator arithmetic", "answer": "2", "source_id": "a", "prior_p": .5,
           "source": "dataset@rev", "license": "apache-2.0"}
    args = generator_args(tmp_path, [row])
    argv = ["ingest-generator", args.input, "--epoch", args.epoch, "--results", args.results, "--active", args.active]
    assert main(argv) == 0
    capsys.readouterr()
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    shard = next((root / "shards").glob("*apache*"))
    assert "apache-2-0.jsonl" in shard.name
    Path(args.input).write_text(Path(args.input).read_text().replace('"answer": "2"', '"answer": "3"'))
    assert main(argv) == 4
    Path(args.input).write_text(Path(args.input).read_text().splitlines()[0])
    assert main(argv) == 3
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    capsys.readouterr()


def test_live_intake_v2_first_rows_preserve_row_king_without_results(tmp_path, monkeypatch):
    seed, root = tmp_path / "seed", tmp_path / "reserve"
    reserve(seed, 0)
    prepare_live_reserve(root, seed)
    monkeypatch.setenv("SN120_RESERVE_DIR", str(root))
    fixture = Path(__file__).parent / "fixtures/generator-intake-v2-20261120T0400Z.jsonl"
    rows = [json.loads(line) for line in fixture.read_bytes().splitlines()]
    args = generator_args(tmp_path, rows)
    Path(args.input).write_bytes(fixture.read_bytes())
    Path(args.input).with_suffix(".done").write_bytes(fixture.with_suffix(".done").read_bytes())
    Path(args.results).unlink()  # Production ingest-ready supplies no --results.
    report = ingest_generator(args)
    assert report["rows"] == 4 and report["ingested"] >= 1
    assert "schema" not in report["rejected"] and report["rejected"]["proof"] == 1
    king = "56bae61bb4fde8dc786567e1e833ff5d53f5203f537a32ff2d2ee1fbdfc2e810"
    assert report["king"] == king and report["king_revisions"] == [king]
    stored = [json.loads(line) for name in report["shards"] for line in (root / "shards" / name).read_bytes().splitlines()]
    assert len(stored) == report["ingested"]
    assert all(row["audit"]["ref"].endswith("#king=" + king) for row in stored)
    assert all(f"frozen king {king[:12]} " in row["audit"]["method"] for row in stored)
    ledger = json.loads((root / "generator-ingested.jsonl").read_text())
    assert ledger["king"] == king and ledger["input_sha256"] == sha256_hex(fixture.read_bytes())


def test_king_precedence_and_revision_format(tmp_path, monkeypatch):
    seed, root = tmp_path / "seed", tmp_path / "reserve"
    reserve(seed, 0)
    prepare_live_reserve(root, seed)
    monkeypatch.setenv("SN120_RESERVE_DIR", str(root))
    row = {"problem": "Unique king priority compute 10 + 2", "answer": "12", "task_id": "king-priority",
           "license": "mit", "prior_p": .5, "king": "d" * 64, "prior_king": "e" * 64, "king_revision": "f" * 64}
    args = generator_args(tmp_path, [row])
    Path(args.results).unlink()
    write_intake_done(args)
    report = ingest_generator(args)
    stored = json.loads((root / "shards" / report["shards"][0]).read_text())
    assert stored["audit"]["ref"].endswith("#king=" + "d" * 64)
    assert report["king"] == "d" * 64
    for number, (key, value) in enumerate((("king", ""), ("prior_king", None), ("king_revision", "A" * 64), ("king", "f" * 63), ("prior_king", "f" * 63))):
        bad = {key: value, "problem": f"Invalid king row {number}", "answer": "1", "source_id": "bad", "prior_p": .5}
        if number == 4:
            bad["king"] = "d" * 64  # Lower-priority fields are validated too.
        args = generator_args(tmp_path / f"bad{number}", [bad], f"20261120T{800 + number:04d}Z")
        Path(args.results).unlink()
        write_intake_done(args)
        with pytest.raises(V2Error, match="nonempty lowercase 64-hex"):
            ingest_generator(args)
    assert len((root / "generator-ingested.jsonl").read_text().splitlines()) == 1


def test_publish_refuses_manifest_different_from_decision(tmp_path):
    from test_rotation_controller import rotated_world
    world, pool = rotated_world(tmp_path)
    decision, _ = pool.open("r000001", world.controller.sampler.public_snapshot(), world.controller.state)
    (tmp_path / "reserve" / MANIFEST).write_bytes(empty_manifest())
    with pytest.raises(VerifyError, match="between rotation decision and publication"):
        pool.publish_open(decision)
    assert not world.store.exists(layout.RESERVE_MANIFEST)


def test_manifest_commit_without_ledger_recovers_without_replacing_shards(tmp_path, monkeypatch):
    seed, root = tmp_path / "seed", tmp_path / "reserve"
    reserve(seed, 0)
    monkeypatch.setenv("SN120_RESERVE_DIR", str(root))
    prepare_live_reserve(root, seed)
    args = generator_args(tmp_path, input_rows())
    first = ingest_generator(args)
    (root / "generator-ingested.jsonl").unlink()
    before = (root / MANIFEST).read_bytes()
    shards = {path.name: path.read_bytes() for path in (root / "shards").glob("*generator*.jsonl")}
    recovered = ingest_generator(args)
    assert recovered["already_ingested"] and recovered["ingested"] == 0
    assert (root / MANIFEST).read_bytes() == before
    assert shards == {path.name: path.read_bytes() for path in (root / "shards").glob("*generator*.jsonl")}
    assert json.loads((root / "generator-ingested.jsonl").read_text())["shards"] == first["shards"]
    assert ingest_generator(args)["already_ingested"]


def test_done_json_plain_and_new_row_fields(tmp_path, monkeypatch):
    seed, root = tmp_path / "seed", tmp_path / "reserve"
    reserve(seed, 0)
    monkeypatch.setenv("SN120_RESERVE_DIR", str(root))
    prepare_live_reserve(root, seed)
    king = "b" * 64
    row = {"problem": "New format compute 2 + 2", "answer": "4", "task_id": "stable-task-1",
           "license": "cc-by-nc-4.0", "prior_king": king, "prior_p": .5}
    args = generator_args(tmp_path, [row])
    Path(args.results).unlink()
    Path(args.input).with_suffix(".done").write_text(json.dumps({"rows": 1, "sha256": sha256_hex(Path(args.input).read_bytes()), "epoch": args.epoch}))
    rejected = ingest_generator(args)
    assert rejected["rejected"] == {"license": 1} and rejected["ingested"] == 0
    row["license"] = "mit"
    Path(args.input).write_text(json.dumps(row) + "\n")
    Path(args.input).with_suffix(".done").write_text(f"1 {sha256_hex(Path(args.input).read_bytes())}\n")
    args = generator_args(tmp_path / "accepted", [row], "20261120T0800Z")
    Path(args.results).unlink()
    Path(args.input).with_suffix(".done").write_text(f"1 {sha256_hex(Path(args.input).read_bytes())}\n")
    report = ingest_generator(args)
    stored = json.loads(next((root / "shards").glob("*generator*.jsonl")).read_text())
    assert stored["source"].startswith(f"sn120-generator:{args.epoch}:stable-task-1;")
    assert stored["audit"]["method"].find(king[:12]) >= 0 and stored["audit"]["ref"].endswith(king)
    assert stored["source"].endswith("prior_source=dataset@rev")

    intake = tmp_path / "intake"
    args = generator_args(intake, [row], "20261120T1200Z")
    Path(args.results).unlink()
    Path(args.input).with_suffix(".done").write_text(json.dumps({
        "format": "generator-intake-v2", "epoch": args.epoch, "lines": 1,
        "sha256": sha256_hex(Path(args.input).read_bytes())}))
    report = ingest_generator(args)
    assert report["rows"] == 1 and report["done_sha256"] == sha256_hex(Path(args.input).read_bytes())
    Path(args.input).with_suffix(".done").write_text(json.dumps({
        "epoch": args.epoch, "lines": 1, "sha256": sha256_hex(Path(args.input).read_bytes())}))
    assert ingest_generator(args)["already_ingested"]


def test_done_mismatch_and_backfill_subset_and_new_problem(tmp_path, monkeypatch):
    seed, root = tmp_path / "seed", tmp_path / "reserve"
    reserve(seed, 0)
    monkeypatch.setenv("SN120_RESERVE_DIR", str(root))
    prepare_live_reserve(root, seed)
    king = "c" * 64
    rows = [{"problem": "Backfill compute 3 + 3", "answer": "6", "task_id": "task-1", "license": "mit", "prior_king": king, "prior_p": .5}]
    args = generator_args(tmp_path, rows)
    Path(args.results).unlink()
    Path(args.input).with_suffix(".done").write_text("2 " + "0" * 64)
    with pytest.raises(Exception) as incomplete:
        ingest_generator(args)
    assert type(incomplete.value).__name__ == "Incomplete"
    assert not (root / "generator-ingested.jsonl").exists()
    Path(args.input).with_suffix(".done").write_text(f"1 {sha256_hex(Path(args.input).read_bytes())}")
    assert ingest_generator(args)["ingested"] == 1
    original_manifest = (root / MANIFEST).read_bytes()
    rows[0].update(source_id="legacy-source", task_id="stable-task-1", king_revision=king)
    Path(args.input).write_text(json.dumps(rows[0], sort_keys=True) + "\n")
    Path(args.input).with_suffix(".done").write_text(f"1 {sha256_hex(Path(args.input).read_bytes())}")
    subset = ingest_generator(args)
    assert subset["already_ingested"] and subset["message"] == "backfill_seen"
    assert (root / MANIFEST).read_bytes() == original_manifest
    assert len((root / "generator-ingested.jsonl").read_text().splitlines()) == 1
    new = {"problem": "Backfill compute 5 + 5", "answer": "10", "task_id": "task-2", "license": "mit", "prior_king": king, "prior_p": .5}
    Path(args.input).write_text("\n".join(json.dumps(row) for row in [rows[0], new]) + "\n")
    Path(args.input).with_suffix(".done").write_text(f"2 {sha256_hex(Path(args.input).read_bytes())}")
    result = ingest_generator(args)
    assert result["ingested"] == 1 and result["backfill_shards"][0].find("-b1-") >= 0
    assert json.loads((root / "generator-ingested.jsonl").read_text())["shards"] == result["shards"]
    assert len((root / "generator-ingested.jsonl").read_text().splitlines()) == 1
