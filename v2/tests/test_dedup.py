"""Pure duplicate evidence, low-rank cosine and training-seed boundaries."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from sn120_v2.canonical import canonical_json, manifest_digest, sha256_hex
from sn120_v2.dedup import (
    STAGE_EXACT,
    STAGE_KING_EXACT,
    STAGE_KING_ROLLOUT,
    STAGE_ROLLOUT,
    LoraMatrices,
    cluster_entries,
    cluster_past_kings,
    cosine,
    normalized_rollouts,
    rollout_overlap,
    stage_b,
    dryrun,
)
from sn120_v2.params import Params, Recipe
from sn120_v2.sampling import train_seed
from sn120_v2.types import DUPLICATE, LoRAArtifact, TrainReport


def raw_rollouts(*pairs: tuple[str, str], reward: float = 0.0) -> bytes:
    groups: dict[str, list[dict]] = {}
    for prompt, output in pairs:
        groups.setdefault(prompt, []).append({"output": output, "reward": reward})
    return json.dumps({"groups": [{"prompt": prompt, "rollouts": rows} for prompt, rows in groups.items()]}).encode()


def rollout_set(*labels: str) -> frozenset[str]:
    found = normalized_rollouts(raw_rollouts(*(("p", label) for label in labels)))
    assert found is not None
    return found


def entry(build, position: int, label: str | None = None):
    made = build.entry(position)
    if label is None:
        return made
    return replace(made, batch=replace(made.batch, batch_sha256=build.h(label)))


def entries_with_hashes(build, count: int, label: str | None = None):
    # A non-None label intentionally gives every entry the same exact batch.
    return [entry(build, position, label) for position in range(count)]


def first_seen(entries, *times: int) -> dict[str, int]:
    if not times:
        times = tuple(range(100, 100 + len(entries)))
    return {made.batch.batch_sha256: seen for made, seen in zip(entries, times)}


def stage_a(build, entries, sets, params=None, times=None):
    return cluster_entries(entries, times or first_seen(entries), sets, params or build.params())


# ---------------------------------------------------------------- Stage A


def test_exact_batch_hash_wins_even_when_rollout_evidence_is_unavailable(build):
    entries = entries_with_hashes(build, 3, "same")
    kept, dropped, clusters = stage_a(build, entries, {})
    assert [made.queue_pos for made in kept] == [0]
    assert [row.queue_pos for row in dropped] == [1, 2]
    assert all(row.status == DUPLICATE and row.charged for row in dropped)
    assert clusters[0]["keeper"]["queue_pos"] == 0
    assert [row["stage"] for row in clusters[0]["dropped"]] == [STAGE_EXACT, STAGE_EXACT]


def test_rollout_subset_reaches_overlap_one_and_is_duplicate(build):
    entries = entries_with_hashes(build, 2)
    sets = {
        entries[0].batch.batch_sha256: rollout_set("a", "b", "c", "d"),
        entries[1].batch.batch_sha256: rollout_set("b", "d"),
    }
    kept, dropped, clusters = stage_a(build, entries, sets)
    assert [made.queue_pos for made in kept] == [0]
    assert [row.queue_pos for row in dropped] == [1]
    assert clusters[0]["dropped"][0]["stage"] == STAGE_ROLLOUT
    assert clusters[0]["dropped"][0]["similarity"] == 1.0


def test_whitespace_and_rewards_do_not_change_rollout_identity(build):
    plain = raw_rollouts(("  one\n prompt ", "  alpha\t beta\n"), ("two", "gamma"), reward=0.0)
    changed = raw_rollouts(("two", "  gamma  "), ("one prompt", "alpha beta"), reward=1.0)
    assert normalized_rollouts(plain) == normalized_rollouts(changed)
    entries = entries_with_hashes(build, 2)
    sets = {made.batch.batch_sha256: value for made, value in zip(entries, (normalized_rollouts(plain), normalized_rollouts(changed)))}
    kept, dropped, clusters = stage_a(build, entries, sets)
    assert len(kept) == 1 and len(dropped) == 1
    assert clusters[0]["dropped"][0]["similarity"] == 1.0


def test_honest_disjoint_rollouts_are_kept(build):
    entries = entries_with_hashes(build, 2)
    sets = {
        entries[0].batch.batch_sha256: rollout_set("a", "b"),
        entries[1].batch.batch_sha256: rollout_set("c", "d"),
    }
    kept, dropped, clusters = stage_a(build, entries, sets)
    assert kept == entries and dropped == [] and clusters == []
    assert rollout_overlap(sets[entries[0].batch.batch_sha256], sets[entries[1].batch.batch_sha256]) == 0.0


def test_overlap_strictly_below_threshold_is_kept(build):
    entries = entries_with_hashes(build, 2)
    left, right = rollout_set("a", "b", "c"), rollout_set("a", "d", "e")
    assert rollout_overlap(left, right) == pytest.approx(1 / 3)
    kept, dropped, clusters = stage_a(build, entries, dict(zip((made.batch.batch_sha256 for made in entries), (left, right))))
    assert kept == entries and dropped == [] and clusters == []


@pytest.mark.parametrize("raw", [
    b"not JSON",
    b"[]",
    b"{}",
    b'{"groups": []}',
    b'{"groups": [{"prompt": "p", "rollouts": []}]}',
    b'{"groups": [{"prompt": 1, "rollouts": []}]}',
    b'{"groups": [{"prompt": "p", "rollouts": [{"output": 1}]}]}',
])
def test_unparsable_rollout_evidence_is_unknown_not_empty(raw):
    assert normalized_rollouts(raw) is None
    assert rollout_overlap(None, frozenset({"known"})) is None
    assert rollout_overlap(frozenset(), frozenset({"known"})) is None


def test_unparsable_rollouts_fail_open_in_stage_a(build):
    entries = entries_with_hashes(build, 2)
    kept, dropped, clusters = stage_a(build, entries, {entries[0].batch.batch_sha256: None})
    assert kept == entries and dropped == [] and clusters == []


def test_past_king_exact_and_rollout_copy_lose_regardless_of_first_seen(build):
    entries = entries_with_hashes(build, 2)
    king_entry, copied = entries
    copied = replace(copied, batch=replace(copied.batch, batch_sha256=build.h("king-batch")))
    other = entries_with_hashes(build, 1)[0]
    other = replace(other, queue_pos=2, batch=replace(other.batch, batch_sha256=build.h("honest-batch")))
    king = {
        "uid": king_entry.reg.uid,
        "hotkey": king_entry.reg.hotkey,
        "batch_sha256": copied.batch.batch_sha256,
        "first_seen_unix": 1,
        "rollouts": None,
    }
    kept, dropped, clusters = cluster_past_kings(
        [copied, other], {copied.batch.batch_sha256: 100, other.batch.batch_sha256: 50}, {}, [king], build.params()
    )
    assert [made.queue_pos for made in kept] == [2]
    assert [row.queue_pos for row in dropped] == [1]
    assert "past king UID" in dropped[0].reason
    assert clusters[0]["dropped"][0]["stage"] == STAGE_KING_EXACT

    king["rollouts"] = rollout_set("a", "b", "c")
    copied = replace(copied, batch=replace(copied.batch, batch_sha256=build.h("different-batch")))
    sets = {copied.batch.batch_sha256: rollout_set("b", "c")}
    kept, dropped, clusters = cluster_past_kings(
        [copied, other], {copied.batch.batch_sha256: 100, other.batch.batch_sha256: 50}, sets, [king], build.params()
    )
    assert [made.queue_pos for made in kept] == [2]
    assert clusters[0]["dropped"][0]["stage"] == STAGE_KING_ROLLOUT
    assert clusters[0]["dropped"][0]["similarity"] == 1.0


def test_greedy_first_seen_order_does_not_compare_through_a_dropped_entry(build):
    entries = entries_with_hashes(build, 3)
    a, b, c = (made.batch.batch_sha256 for made in entries)
    # B shares half of A; C shares two thirds of B but nothing with A.
    sets = {a: rollout_set("a", "b", "c", "d"), b: rollout_set("a", "b", "e", "f"), c: rollout_set("e", "f", "z")}
    kept, dropped, clusters = stage_a(build, entries, sets)
    assert [made.queue_pos for made in kept] == [0, 2]
    assert [row.queue_pos for row in dropped] == [1]
    assert len(clusters) == 1 and clusters[0]["keeper"]["queue_pos"] == 0
    assert clusters[0]["dropped"][0]["similarity"] == 0.5


def test_half_mixture_of_two_honest_batches_drops_only_the_mixture(build):
    entries = entries_with_hashes(build, 3)
    hashes = [made.batch.batch_sha256 for made in entries]
    sets = {
        hashes[0]: rollout_set("a", "b", "c", "d"),
        hashes[1]: rollout_set("e", "f", "g", "h"),
        hashes[2]: rollout_set("a", "b", "e", "f"),
    }
    kept, dropped, clusters = stage_a(build, entries, sets)
    assert [made.queue_pos for made in kept] == [0, 1]
    assert [row.queue_pos for row in dropped] == [2]
    assert clusters[0]["dropped"][0]["similarity"] == 0.5


def test_dryrun_reports_exact_duplicate_cluster_from_distinct_hotkeys(tmp_path):
    raw = json.dumps({"version": "affine-batch-v3", "hotkey": "miner",
                      "groups": [{"prompt": "p", "rollouts": [{"output": "a", "reward": 1},
                                                              {"output": "b", "reward": 0}]}]}).encode()
    paths = []
    for index in range(2):
        path = tmp_path / f"batch-{index}.json"
        path.write_bytes(raw)
        paths.append(path)
    report = dryrun(paths, Params())
    assert len(report["exact_duplicate_clusters"]) == 1
    assert report["stage_a"]["decisions"][0]["queue_pos"] == 1
    assert report["stage_a"]["decisions"][0]["status"] == "duplicate"


def test_first_seen_tie_uses_queue_position_not_input_order(build):
    entries = list(reversed(entries_with_hashes(build, 3, "copy")))
    kept, dropped, _ = stage_a(build, entries, {}, times={made.batch.batch_sha256: 100 for made in entries})
    assert [made.queue_pos for made in kept] == [0]
    assert [row.queue_pos for row in dropped] == [1, 2]


def test_empty_single_and_all_duplicate_stage_a_inputs(build):
    params = build.params()
    assert cluster_entries([], {}, {}, params) == ([], [], [])
    single = entries_with_hashes(build, 1)
    assert cluster_entries(single, first_seen(single), {}, params) == (single, [], [])
    copies = entries_with_hashes(build, 4, "all-copies")
    kept, dropped, clusters = stage_a(build, copies, {})
    assert len(kept) == 1 and kept[0].queue_pos == 0
    assert [row.queue_pos for row in dropped] == [1, 2, 3]
    assert len(clusters) == 1 and len(clusters[0]["dropped"]) == 3


# ---------------------------------------------------------------- Synthetic safetensors / Stage B


def lora_bytes(a_values: list[list[float]], b_values: list[list[float]]) -> bytes:
    values = [*a_values, *b_values]
    flat = [value for row in values for value in row]
    b_offset = 4 * sum(len(row) for row in a_values)
    header = json.dumps({
        "layer.lora_A.weight": {"dtype": "F32", "shape": [len(a_values), len(a_values[0])], "data_offsets": [0, b_offset]},
        "layer.lora_B.weight": {"dtype": "F32", "shape": [len(b_values), len(b_values[0])], "data_offsets": [b_offset, 4 * len(flat)]},
    }, separators=(",", ":")).encode()
    return struct.pack("<Q", len(header)) + header + struct.pack(f"<{len(flat)}f", *flat)


def combined_lora_bytes(value: float) -> bytes:
    # rank=1: A and B are both 1x1. The fixture recipe below pins that rank.
    return lora_bytes([[value]], [[value]])


def lora_artifact(build, made, params, raw: bytes, seed: int):
    files = {"adapter_config.json": build.h("config"), "adapter_model.safetensors": sha256_hex(raw)}
    return LoRAArtifact(
        lora_hash=manifest_digest(files), files=files,
        sizes={"adapter_config.json": 1, "adapter_model.safetensors": len(raw)},
        prefix=f"pending/rounds/r000001/r000001-l0/loras/{made.queue_pos:03d}",
        head_version_id="pure-stage-b", batch_sha256=made.batch.batch_sha256,
        recipe_id="pure-recipe", rank=1, alpha=1.0, target_modules=params.recipe.target_modules,
        report=TrainReport(seed=seed, steps=1, loss=0.5, step_norm=0.1, tokens=100, wall_s=1.0,
                           gpu_class="cpu", image_digest=build.image),
    )


def run_stage_b(build, raws: list[bytes | Exception | None], seeds: list[int] | None = None, excluded=()):
    params = build.params(recipe=Recipe(image_digests=(build.image,), rank=1, alpha=1.0))
    entries = entries_with_hashes(build, len(raws))
    seeds = seeds or list(range(100, 100 + len(raws)))
    artifacts = {made.queue_pos: lora_artifact(build, made, params, b"" if not isinstance(raw, bytes) else raw, seed)
                 for made, raw, seed in zip(entries, raws, seeds)}
    files = {}
    for made, raw in zip(entries, raws):
        if isinstance(raw, bytes):
            files[f"{artifacts[made.queue_pos].prefix}/adapter_model.safetensors"] = raw
    def get_file(key: str) -> bytes:
        found = files[key]
        if isinstance(found, Exception):
            raise found
        return found
    logs: list[str] = []
    dropped, clusters = stage_b(entries, artifacts, first_seen(entries), params, get_file,
                                dict(zip((made.queue_pos for made in entries), seeds)), excluded=excluded, log=logs.append)
    return dropped, clusters, logs


def test_synthetic_safetensors_scaled_lora_cosine_and_stage_b(build):
    params = build.params(recipe=Recipe(image_digests=(build.image,), rank=1, alpha=1.0))
    made = entries_with_hashes(build, 1)[0]
    # A=B=2 and alpha/rank=1: delta-W=4, Frobenius norm=4.
    artifact = lora_artifact(build, made, params, combined_lora_bytes(2.0), 100)
    matrix = LoraMatrices.read(artifact, lambda key: combined_lora_bytes(2.0))
    assert matrix.norm() == pytest.approx(4.0)
    assert cosine(matrix, matrix) == pytest.approx(1.0)

    dropped, clusters, _ = run_stage_b(build, [combined_lora_bytes(2.0), combined_lora_bytes(3.0)])
    assert [row.queue_pos for row in dropped] == [1]
    assert clusters[0]["dropped"][0]["similarity"] == pytest.approx(1.0)

    # Orthogonal nonzero 2D updates have cosine zero and remain independent.
    orthogonal = lora_bytes([[1.0, 0.0]], [[0.0], [2.0]])
    base = LoraMatrices.read(
        lora_artifact(build, made, params, lora_bytes([[1.0, 0.0]], [[2.0], [0.0]]), 100),
        lambda key: lora_bytes([[1.0, 0.0]], [[2.0], [0.0]]),
    )
    assert cosine(base, LoraMatrices.read(
        lora_artifact(build, made, params, orthogonal, 100), lambda key: orthogonal
    )) == pytest.approx(0.0)


def test_rectangular_lora_math_matches_an_explicit_delta_w(build):
    # rank=2, inputs=3, outputs=2; alpha/rank=2. This catches transposed
    # Gram products and zip-truncated fallbacks, not only scalar fixtures.
    a = [[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]]
    b = [[2.0, 0.0], [-1.0, 3.0]]
    raw = lora_bytes(a, b)
    artifact = lora_artifact(build, entries_with_hashes(build, 1)[0],
                             build.params(recipe=Recipe(image_digests=(build.image,), rank=2, alpha=4.0)), raw, 100)
    artifact = replace(artifact, rank=2, alpha=4.0)
    matrix = LoraMatrices.read(artifact, lambda key: raw)
    np = pytest.importorskip("numpy")
    delta_w = 2.0 * np.asarray(b, dtype=np.float64) @ np.asarray(a, dtype=np.float64)
    expected = float(np.sum(delta_w * delta_w, dtype=np.float64))
    assert matrix.norm() ** 2 == pytest.approx(expected)
    assert cosine(matrix, matrix) == pytest.approx(1.0)


def test_no_numpy_fallback_matches_rectangular_lora_math(build, monkeypatch):
    a = [[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]]
    b = [[2.0, 0.0], [-1.0, 3.0]]
    raw = lora_bytes(a, b)
    artifact = lora_artifact(build, entries_with_hashes(build, 1)[0],
                             build.params(recipe=Recipe(image_digests=(build.image,), rank=2, alpha=4.0)), raw, 100)
    artifact = replace(artifact, rank=2, alpha=4.0)
    import builtins
    real_import = builtins.__import__

    def without_numpy(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("test the pure-Python arithmetic")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_numpy)
    matrix = LoraMatrices.read(artifact, lambda key: raw)
    assert matrix.norm() ** 2 == pytest.approx(361.0)
    assert cosine(matrix, matrix) == pytest.approx(1.0)


def test_stage_b_does_not_compare_through_a_dropped_lora(build):
    # B has cosine .5 with A; C has cosine .447 with A, enough to drop
    # through B if B were retained as an anchor. Only B may drop.
    dropped, clusters, _ = run_stage_b(build, [
        lora_bytes([[2.0, 1.0]], [[1.0], [0.0]]),
        lora_bytes([[1.0, 0.0]], [[1.0], [2.2 ** 0.5]]),
        lora_bytes([[0.0, 1.0]], [[0.0], [1.0]]),
    ], seeds=[100, 100, 100])
    assert [row.queue_pos for row in dropped] == [1]
    assert clusters[0]["dropped"][0]["similarity"] == pytest.approx(0.5)


@pytest.mark.parametrize("bad,log_category", [
    (float("nan"), "cannot be compared"),
    (float("inf"), "cannot be compared"),
])
def test_stage_b_fails_open_for_nonfinite_updates(build, bad, log_category):
    dropped, clusters, logs = run_stage_b(build, [combined_lora_bytes(2.0), combined_lora_bytes(bad)])
    assert dropped == [] and clusters == []
    assert any("queue_pos 1" in line and log_category in line for line in logs)


def test_stage_b_fails_open_for_an_all_zero_update(build):
    dropped, clusters, logs = run_stage_b(build, [combined_lora_bytes(2.0), combined_lora_bytes(0.0)])
    assert dropped == [] and clusters == []
    assert any("queue_pos 1" in line and "unusable norm" in line for line in logs)


def test_zero_norm_in_one_module_does_not_hide_a_nonzero_global_update(build):
    def two_modules(first: float) -> bytes:
        tensors = {
            "zero.lora_A.weight": ([[0.0]], [[0.0]]),
            "zero.lora_B.weight": ([[0.0]], [[0.0]]),
            "signal.lora_A.weight": ([[first]], [[first]]),
            "signal.lora_B.weight": ([[first]], [[first]]),
        }
        header, values, offset = {}, [], 0
        for name, (a, b) in tensors.items():
            matrix = a if name.endswith("lora_A.weight") else b
            flat = [x for row in matrix for x in row]
            header[name] = {"dtype": "F32", "shape": [len(matrix), len(matrix[0])],
                            "data_offsets": [offset, offset + 4 * len(flat)]}
            values.extend(flat)
            offset += 4 * len(flat)
        encoded = json.dumps(header, separators=(",", ":")).encode()
        return struct.pack("<Q", len(encoded)) + encoded + struct.pack(f"<{len(values)}f", *values)

    dropped, clusters, logs = run_stage_b(build, [two_modules(2.0), two_modules(3.0)], seeds=[100, 100])
    assert [row.queue_pos for row in dropped] == [1]
    assert clusters[0]["dropped"][0]["similarity"] == pytest.approx(1.0)


@pytest.mark.parametrize("problem", ["missing", "unreadable", "truncated"])
def test_stage_b_fails_open_for_missing_unreadable_and_malformed_weights(build, problem):
    def get_file(key: str) -> bytes:
        if key.endswith("/001/adapter_model.safetensors"):
            if problem == "missing":
                raise KeyError(key)
            if problem == "unreadable":
                raise OSError("simulated read failure")
            return b"not a safetensors file"
        return combined_lora_bytes(2.0)
    params = build.params(recipe=Recipe(image_digests=(build.image,), rank=1, alpha=1.0))
    entries = entries_with_hashes(build, 2)
    raw = combined_lora_bytes(2.0)
    artifacts = {made.queue_pos: lora_artifact(build, made, params, raw, 100 + made.queue_pos) for made in entries}
    logs: list[str] = []
    dropped, clusters = stage_b(entries, artifacts, first_seen(entries), params, get_file, {0: 100, 1: 101}, log=logs.append)
    assert dropped == [] and clusters == []
    assert any("queue_pos 1" in line and "cannot be compared" in line for line in logs)


def test_stage_b_excludes_null_candidates_and_checks_report_against_request_seed(build):
    null, other, _ = run_stage_b(build, [combined_lora_bytes(2.0)] * 3, excluded=(0,))
    assert [row.queue_pos for row in null] == [2] and len(other) == 1 and other[0]["keeper"]["queue_pos"] == 1
    wrong_seed = run_stage_b(build, [combined_lora_bytes(2.0), combined_lora_bytes(2.0)], seeds=[100, 200])[0]
    assert [row.queue_pos for row in wrong_seed] == [1], "own-seed cosine one crosses the 0.002 floor"
    entries = entries_with_hashes(build, 2)
    params = build.params(recipe=Recipe(image_digests=(build.image,), rank=1, alpha=1.0))
    raw = combined_lora_bytes(2.0)
    artifacts = {entry.queue_pos: lora_artifact(build, entry, params, raw, 100 + entry.queue_pos) for entry in entries}
    logs = []
    dropped, clusters = stage_b(entries, artifacts, first_seen(entries), params,
                                lambda key: raw, {0: 100, 1: 200}, log=logs.append)
    assert dropped == [] and clusters == []
    assert any("seed verification failed" in line for line in logs)


def test_stage_b_honors_disabled_legacy_params_without_reading_weights(build):
    dropped, clusters, _ = run_stage_b(build, [combined_lora_bytes(2.0)] * 2, excluded=())
    assert dropped and clusters, "the enabled fixture is genuinely comparable"

    def unexpected_read(key):
        raise AssertionError(f"disabled dedup read {key}")

    entries = entries_with_hashes(build, 2)
    assert stage_b(entries, {}, first_seen(entries), build.params(dedup_enabled=False), unexpected_read, {}) == ([], [])


def test_stage_b_whole_comparison_exception_fails_open(build, monkeypatch):
    import sn120_v2.dedup as module
    monkeypatch.setattr(module, "_cluster_loaded", lambda *args: (_ for _ in ()).throw(OSError("comparison failed")))
    dropped, clusters, logs = run_stage_b(build, [combined_lora_bytes(2.0), combined_lora_bytes(2.0)])
    assert dropped == [] and clusters == []
    assert any("failed open with no drops" in line for line in logs)


# ---------------------------------------------------------------- Round-wide and legacy training seeds


def test_dryrun_synthetic_batches_and_loras_report_all_rules_without_writes(tmp_path):
    def frozen(hotkey, prompt, outputs, train=None):
        return json.dumps({
            "version": "affine-batch-v3", "hotkey": hotkey, "train": train or {},
            "groups": [{"prompt": prompt, "rollouts": [{"output": output, "reward": float(index % 2)}
                                                        for index, output in enumerate(outputs)]}],
        }).encode()

    paths = []
    for index, (prompt, outputs, train) in enumerate([
        ("p0", ["a", "b"], None),
        ("p0", ["c", "d"], {"global_norm_cap": 0.01}),
        ("p0", ["e", "f"], None),
    ]):
        path = tmp_path / f"batch-{index}.json"
        path.write_bytes(frozen(f"hk{index}", prompt, outputs, train))
        paths.append(path)
    dirs = []
    for index, norm in enumerate((float("nan"), 0.1, 0.02)):
        directory = tmp_path / f"lora-{index}"
        directory.mkdir()
        raw = combined_lora_bytes(2.0 + index)
        (directory / "adapter_model.safetensors").write_bytes(raw)
        (directory / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1.0}))
        report = TrainReport(seed=100 + index, steps=1, loss=0.5, step_norm=norm,
                             tokens=10, wall_s=0.1, gpu_class="cpu", image_digest="sim-image")
        (directory / "train_report.json").write_bytes(report.to_json())
        dirs.append(directory)
    params = Params(recipe=Recipe(image_digests=("sim-image",), rank=1, alpha=1.0))
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    report = dryrun(paths, params, first_seen_unix=(10, 11, 12), lora_dirs=dirs)
    assert report["read_only"] is True
    assert [row["nonzero_advantage_rollouts"] for row in report["entries"]] == [2, 2, 2]
    assert all(row["would_fail_min_rollouts"] for row in report["entries"])
    assert [row["static_null_update"] for row in report["entries"]] == [False, True, False]
    assert report["entries"][1]["admission"]["rule"] == "null_update"
    assert report["entries"][1]["nonzero_advantage_rollouts"] == 2
    assert [row["would_be_null_update"] for row in report["entries"]] == [True, False, False]
    assert report["entries"][0]["step_norm_nonfinite"] is True
    assert report["stage_b"]["available"] is True
    assert [row["queue_pos"] for row in report["stage_b"]["decisions"]] == [2]
    assert report["stage_b"]["clusters"][0]["keeper"]["queue_pos"] == 1
    assert {path: path.read_bytes() for path in before} == before


def test_dryrun_stage_b_cannot_use_a_stage_a_duplicate_as_keeper(tmp_path):
    # A and B have identical batch bytes; B and C have identical LoRAs. Only A may anchor Stage B.
    def batch(hotkey, output):
        return json.dumps({"version": "affine-batch-v3", "hotkey": hotkey,
                           "groups": [{"prompt": "p", "rollouts": [{"output": f"{output}-a", "reward": 1},
                                                                   {"output": f"{output}-b", "reward": 0}]}]}).encode()

    paths, dirs = [], []
    for pos in range(3):
        path = tmp_path / f"batch-{pos}.json"
        path.write_bytes(batch("same" if pos < 2 else f"other-{pos}", "shared" if pos < 2 else "independent"))
        paths.append(path)
        directory = tmp_path / f"lora-{pos}"
        directory.mkdir()
        (directory / "adapter_model.safetensors").write_bytes(
            combined_lora_bytes(2.0) if pos < 2 else lora_bytes([[1.0, 0.0]], [[0.0], [1.0]])
        )
        (directory / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1.0}))
        (directory / "train_report.json").write_bytes(TrainReport(100 + pos, 1, 0.5, 0.1, 10, 0.1, "cpu", "sim").to_json())
        dirs.append(directory)
    report = dryrun(paths, Params(recipe=Recipe(image_digests=("sim",), rank=1, alpha=1.0)), lora_dirs=dirs)
    assert [row["queue_pos"] for row in report["stage_a"]["decisions"]] == [1]
    assert report["stage_b"]["decisions"] == [] and report["stage_b"]["clusters"] == []
    assert [row["queue_pos"] for row in report["exact_duplicate_clusters"][0]["dropped"]] == [1]


def test_dryrun_missing_lora_fails_open_without_store_or_network(tmp_path):
    path = tmp_path / "batch.json"
    path.write_bytes(json.dumps({"version": "affine-batch-v3", "hotkey": "miner",
                                 "groups": [{"prompt": "p", "rollouts": [{"output": "a", "reward": 1},
                                                                         {"output": "b", "reward": 0}]}]}).encode())
    missing = tmp_path / "absent"
    before = set(tmp_path.iterdir())
    report = dryrun([path], Params(), lora_dirs=[missing])
    assert "lora_error" in report["entries"][0]
    assert report["stage_b"] == {"decisions": [], "clusters": [], "available": True}
    assert set(tmp_path.iterdir()) == before


# ---------------------------------------------------------------- Round-wide and legacy training seeds


def test_round_wide_train_seed_golden_is_independent_of_batch_hash():
    # Independently: int(sha256(b"seed-0|train").hexdigest(), 16) >> 193.
    expected = int(hashlib.sha256(b"seed-0|train").hexdigest(), 16) >> 193
    assert expected == 5463705811908494762
    assert train_seed("seed-0", "a" * 64) == expected
    assert train_seed("seed-0", "b" * 64, "round") == expected


def test_legacy_batch_and_auto_seed_goldens_and_hash_behavior():
    for batch, expected in (("a" * 64, 6703626329271231394), ("b" * 64, 5301982802906811973)):
        text = f"seed-0|train|{batch}".encode()
        assert int(hashlib.sha256(text).hexdigest(), 16) >> 193 == expected
        assert train_seed("seed-0", batch, "batch") == expected
        assert train_seed("seed-0", batch, "auto") == expected
    assert train_seed("seed-0", "a" * 64, "batch") != train_seed("seed-0", "b" * 64, "batch")


def test_legacy_params_without_seed_scope_keep_hash_and_batch_behavior(build):
    data = build.params(train_seed_scope="batch").to_dict()
    del data["train_seed_scope"]
    recorded_hash = sha256_hex(canonical_json(data))
    restored = Params.from_dict(data)
    assert restored.train_seed_scope == "batch"
    assert restored.params_hash == recorded_hash
    assert restored.to_dict() == data
    assert train_seed("seed-0", "a" * 64, restored.train_seed_scope) == 6703626329271231394
