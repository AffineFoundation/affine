"""Global-prior ingest: verified parquet, deterministic priors and exact N."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sn120_v2.canonical import sha256_hex
from sn120_v2.envs.reserve import UNSCREENED_METHOD, candidate_metadata, load_reserve, reserve_dir
from sn120_v2.envs.reserve.prior import prior_p
from sn120_v2.errors import V2Error


def _release(tmp_path: Path, rows: list[dict]) -> Path:
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    root = tmp_path / "published"
    root.mkdir()
    columns = ["id", "problem", "answer", "answer_type", "source", "source_id", "license",
               "subsource", "topic", "difficulty_meta", "also_in", "in_sn120_pool"]
    full = []
    for index, row in enumerate(rows):
        full.append(dict(
            id=f"prior-{index}", problem=f"UNIQUE-PRIOR-SYNTHETIC-{index}: compute {index + 2} + 2",
            answer=str(index + 4), answer_type="integer", source="SynthLabsAI/Big-Math-RL-Verified@c75d2f11",
            source_id=str(index), license="apache-2.0", subsource="test", topic="",
            difficulty_meta="{}", also_in="[]", in_sn120_pool="",
        ) | row)
    path = root / "prior-00-of-08.parquet"
    pq.write_table(pa.Table.from_pylist(full, schema=pa.schema([(name, pa.string()) for name in columns])), path, compression="zstd")
    (root / "MANIFEST.json").write_text(json.dumps({
        "name": "sn120-math-prior", "version": "v1", "rows": len(full), "columns": columns,
        "shards": [{"file": path.name, "rows": len(full), "sha256": sha256_hex(path.read_bytes())}],
    }))
    return root


def _args(published: Path, n: int = 4, **extra) -> list[str]:
    return ["ingest-prior", "--prior-dir", str(published), "--n", str(n), "--seed", "synthetic",
            "--added-at", "2026-11-19", *(word for key, value in extra.items() for word in ("--" + key.replace("_", "-"), str(value)))]


def test_prior_mapping_is_documented_and_bounded():
    assert prior_p({"pass_72b_tir": .375}) == .375
    assert prior_p({"llama8b_solve": .5}) == .5
    assert prior_p({"gptoss_acc": {"reason_low_no_tool": 0, "reason_high_with_tool": 1}}) == .5
    assert prior_p({"r1_correct": 6}) == .75
    assert prior_p({"polaris_diff": "7/8"}) == .125
    assert prior_p({"difficulty": 8.5}) is None
    assert prior_p({"pass_rate": 3}) is None
    assert prior_p({"r1_correct": 9}) is None


def test_ingest_prior_synthetic_exact_n_filters_metadata_and_reruns(tmp_path, monkeypatch, capsys):
    from test_reserve_ingest import setup_reserve
    from sn120_v2.envs.reserve.__main__ import main
    root = setup_reserve(tmp_path, monkeypatch)
    rows = [
        {"difficulty_meta": json.dumps({"llama8b_solve": .5})},
        {"source": "nvidia/Nemotron-Math-v2@8e793210", "difficulty_meta": json.dumps({"gptoss_acc": {"a": .5, "b": .25}})},
        {"source": "open-r1/OpenR1-Math-220k@e4e141ec", "difficulty_meta": json.dumps({"r1_correct": 4})},
        {"source": "POLARIS-Project/Polaris-Dataset-53K@296f8e34", "difficulty_meta": json.dumps({"polaris_diff": "4/8"})},
        {"difficulty_meta": json.dumps({"key_from_majority": "True", "gptoss_acc": {"a": .5}})},
        {"difficulty_meta": json.dumps({"llama8b_solve": .99})},
        {"in_sn120_pool": "math:3", "difficulty_meta": json.dumps({"llama8b_solve": .5})},
    ]
    published = _release(tmp_path, rows)
    assert main(_args(published, 4, no_signal_share=.25)) == 0
    report = json.loads(capsys.readouterr().out)
    reserve, _ = load_reserve(root)
    assert len(reserve.queue) == 4 and report["ingested"] == 4
    lines = [json.loads(line) for shard in reserve.shards for line in (root / "shards" / shard.shard).read_bytes().splitlines()]
    assert all(row["audit"] == {"method": UNSCREENED_METHOD, "verdict": "unscreened"} for row in lines)
    assert sum("prior_p" not in row for row in lines) <= 1
    assert all("prior_p" not in row or .15 <= row["prior_p"] <= .85 for row in lines)
    assert all(row["id"].split(":")[0] in {"bigmath", "nemoprior", "openr1prior", "polarisprior"} for row in lines)
    assert all(not row["source"].endswith(":4") and not row["source"].endswith(":5") and not row["source"].endswith(":6") for row in lines)
    original = {path.name: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    metadata = candidate_metadata(root, reserve)
    assert all(metadata[task]["unscreened"] for task in reserve.queue)
    assert reserve.audit_verdicts.keys() == set(reserve.queue)
    assert main(["verify"]) == 0
    capsys.readouterr()
    # An empty additional selection leaves all committed bytes unchanged.
    assert main(_args(published, 1, min_p=0, max_p=0, no_signal_share=0)) == 2
    assert "only 0 eligible rows" in capsys.readouterr().err
    assert {path.name: path.read_bytes() for path in root.rglob("*") if path.is_file()} == original


def test_ingest_prior_replay_is_byte_stable_and_source_cap_is_global(tmp_path, monkeypatch, capsys):
    from test_reserve_ingest import setup_reserve
    from sn120_v2.envs.reserve.__main__ import main
    root = setup_reserve(tmp_path, monkeypatch)
    rows = [{"source": "SynthLabsAI/Big-Math-RL-Verified@c75d2f11"} for _ in range(12)]
    rows += [{"source": "nvidia/OpenMathReasoning@d3d08664"} for _ in range(12)]
    published = _release(tmp_path, rows)
    assert main(_args(published, 4, no_signal_share=1)) == 0
    capsys.readouterr()
    first = {path.name: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    counts = {}
    for shard in load_reserve(root)[0].shards:
        for row in map(json.loads, (root / "shards" / shard.shard).read_bytes().splitlines()):
            source = row["source"].split("@")[0]
            counts[source] = counts.get(source, 0) + 1
    assert counts == {"SynthLabsAI/Big-Math-RL-Verified": 2, "nvidia/OpenMathReasoning": 2}
    other = tmp_path / "other-reserve"
    import shutil
    shutil.copytree(root, other)
    (other / "MANIFEST.jsonl").write_bytes(b'{"withdrawn":{"count":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}}\n')
    for path in (other / "shards").iterdir():
        path.unlink()
    monkeypatch.setenv("SN120_RESERVE_DIR", str(other))
    assert main(_args(published, 4, no_signal_share=1)) == 0
    capsys.readouterr()
    assert first == {path.name: path.read_bytes() for path in other.rglob("*") if path.is_file()}


def test_ingest_prior_refuses_manifest_tamper(tmp_path, monkeypatch):
    from test_reserve_ingest import setup_reserve
    setup_reserve(tmp_path, monkeypatch)
    published = _release(tmp_path, [{}])
    path = published / "prior-00-of-08.parquet"
    raw = path.read_bytes()
    path.write_bytes(b"tampered" + raw[8:])
    from sn120_v2.envs.reserve.prior import verify_release
    with pytest.raises(V2Error, match="sha256"):
        verify_release(published)
