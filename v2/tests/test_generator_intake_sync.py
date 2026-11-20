"""Local fake-fetcher tests for the stdlib box-side intake sync tool."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "generator_intake_sync", Path(__file__).parents[1] / "ops" / "generator_intake_sync.py"
)
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)
EP = "20261120T0000Z"
INTAKE = [
    {"problem": "π = ?", "answer": "3.14", "source_id": "math:a", "prior_p": 0.5},
    {"problem": "1+1", "answer": "2", "source_id": "math:b", "prior_p": 1},
]
ACCEPTED = [
    {"task_id": "a" * 20, "king": "frozen-head-v1"},
    {"task_id": "b" * 20, "king": "frozen-head-v1"},
]
MAP = [
    {"id": "a" * 20, "source_id": "math:a", "license": "MIT"},
    {"id": "b" * 20, "source_id": "math:b", "license": None},
]


def lines(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def fetcher(intake=None, accepted=None, *, checksum=None):
    intake = lines(INTAKE if intake is None else intake)
    accepted = lines(ACCEPTED if accepted is None else accepted)
    documents = {
        f"/epochs/{EP}/intake.jsonl": intake,
        f"/epochs/{EP}/accepted.jsonl": accepted,
        f"/epochs/{EP}/intake.sha256": f"{hashlib.sha256(intake).hexdigest() if checksum is None else checksum}  intake.jsonl\n".encode(),
        "/state.json": json.dumps({"published_epochs": [{"epoch": EP}]}).encode(),
    }
    calls = []

    def fetch(url):
        calls.append(url)
        key = url.removeprefix("https://affine.io/generator")
        if key not in documents:
            raise FileNotFoundError(url)
        return documents[key]

    fetch.calls = calls
    return fetch


def pool_map(tmp_path, rows=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "pool.jsonl.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(lines(MAP if rows is None else rows))
    return str(path)


def run(tmp_path, fetch=None, **kwargs):
    mapping = kwargs.pop("pool_map", None)
    if mapping is None:
        mapping = pool_map(tmp_path)
    return sync.sync_epoch(
        EP, out=str(tmp_path), fetch=fetch or fetcher(),
        pool_map=mapping, **kwargs,
    )


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_v2_schema_order_values_done_and_idempotent_bytes(tmp_path):
    fetch = fetcher()
    assert run(tmp_path, fetch) == "wrote 2"
    final = tmp_path / f"accepted-{EP}.jsonl"
    done = tmp_path / f"accepted-{EP}.done"
    raw = final.read_bytes()
    rows = read(final)
    assert [list(row) for row in rows] == [list(sync.FIELDS)] * 2
    assert rows == [
        {**INTAKE[0], "task_id": "a" * 20, "license": "MIT", "king": "frozen-head-v1"},
        {**INTAKE[1], "task_id": "b" * 20, "license": None, "king": "frozen-head-v1"},
    ]
    assert "π = ?" in raw.decode()
    assert json.loads(done.read_text()) == {
        "epoch": EP, "lines": 2, "sha256": hashlib.sha256(raw).hexdigest(), "format": "generator-intake-v2",
    }
    assert done.read_bytes().endswith(b"\n")
    before = final.stat().st_mtime_ns, done.stat().st_mtime_ns
    calls = len(fetch.calls)
    assert run(tmp_path, fetch) == "exists"
    assert final.read_bytes() == raw
    assert (final.stat().st_mtime_ns, done.stat().st_mtime_ns) == before
    assert len(fetch.calls) == calls


def test_crash_between_jsonl_and_done_repairs_done_only(tmp_path, monkeypatch):
    real_write_done = sync.write_done
    monkeypatch.setattr(sync, "write_done", lambda *args: None)
    assert run(tmp_path) == "wrote 2"
    final = tmp_path / f"accepted-{EP}.jsonl"
    before = final.read_bytes(), final.stat().st_mtime_ns
    monkeypatch.setattr(sync, "write_done", real_write_done)
    assert run(tmp_path) == "done repaired"
    assert (final.read_bytes(), final.stat().st_mtime_ns) == before
    assert sync.valid_done(EP, str(final), str(tmp_path / f"accepted-{EP}.done"))
    assert run(tmp_path) == "exists"


def test_invalid_done_is_repaired_without_rewriting_v2_jsonl(tmp_path):
    run(tmp_path)
    final = tmp_path / f"accepted-{EP}.jsonl"
    before = final.read_bytes(), final.stat().st_mtime_ns
    (tmp_path / f"accepted-{EP}.done").write_text('{"lines": 999}\n')
    assert run(tmp_path) == "done repaired"
    assert (final.read_bytes(), final.stat().st_mtime_ns) == before


def test_legacy_backfill_upgrades_and_preserves_existing_four_values(tmp_path):
    final = tmp_path / f"accepted-{EP}.jsonl"
    original = lines(INTAKE)
    final.write_bytes(original)
    assert run(tmp_path, backfill=True) == "wrote 2"
    rows = read(final)
    assert [{key: row[key] for key in sync.KEYS} for row in rows] == INTAKE
    assert [row["license"] for row in rows] == ["MIT", None]
    assert json.loads((tmp_path / f"accepted-{EP}.done").read_text())["lines"] == 2


def test_backfill_preserves_original_clean_serialization(tmp_path):
    original_clean = '\n'.join([
        '{"problem": "What is 1?", "answer": "1", "source_id": "prior:1", "prior_p": 0.5}',
        '{"problem": "π", "answer": "3.14", "source_id": "prior:2", "prior_p": 1}',
    ]) + '\n'
    final = tmp_path / f"accepted-{EP}.jsonl"
    final.write_text(original_clean, encoding="utf-8")
    before = [json.loads(line) for line in original_clean.splitlines()]
    accepted = [{"task_id": "a" * 20, "king": "head"}, {"task_id": "b" * 20, "king": "head"}]
    assert run(tmp_path, fetcher(intake=before, accepted=accepted), pool_map=pool_map(tmp_path, [{"id": "a" * 20, "source_id": "prior:1", "license": "MIT"}, {"id": "b" * 20, "source_id": "prior:2", "license": None}]), backfill=True) == "wrote 2"
    after = read(final)
    for old, new in zip(before, after):
        assert list(new)[:4] == list(sync.KEYS)
        for key in sync.KEYS:
            assert new[key] == old[key]
    assert final.read_text(encoding="utf-8").splitlines()[0].startswith('{"problem": "What is 1?", "answer": "1", "source_id": "prior:1", "prior_p": 0.5, "task_id":')


def test_nonfinite_prior_p_writes_nothing(tmp_path):
    body = b'{"problem":"p","answer":"a","source_id":"s","prior_p":NaN}\n'
    documents = {
        f"/epochs/{EP}/intake.jsonl": body,
        f"/epochs/{EP}/accepted.jsonl": lines(ACCEPTED[:1]),
        f"/epochs/{EP}/intake.sha256": hashlib.sha256(body).hexdigest().encode(),
    }
    with pytest.raises(AssertionError):
        sync.sync_epoch(EP, out=str(tmp_path), fetch=lambda url: documents[url.removeprefix("https://affine.io/generator")], pool_map=str(tmp_path / "missing"))
    assert not (tmp_path / f"accepted-{EP}.jsonl").exists()
    assert not (tmp_path / f"accepted-{EP}.done").exists()


def test_legacy_without_backfill_is_untouched(tmp_path):
    final = tmp_path / f"accepted-{EP}.jsonl"
    original = lines(INTAKE)
    final.write_bytes(original)
    fetch = fetcher()
    assert run(tmp_path, fetch) == "legacy (run --backfill)"
    assert final.read_bytes() == original
    assert not (tmp_path / f"accepted-{EP}.done").exists()
    assert fetch.calls == []


@pytest.mark.parametrize("change", ["count", "source"])
def test_mismatches_write_nothing(tmp_path, change, capsys):
    accepted = list(ACCEPTED)
    mapping = MAP
    if change == "count":
        accepted = accepted[:1]
    else:
        mapping = [{**MAP[0], "source_id": "different"}, MAP[1]]
    assert sync.main(
        ["--epoch", EP, "--out", str(tmp_path), "--pool-map", pool_map(tmp_path, mapping)],
        fetch=fetcher(accepted=accepted),
    ) == 0  # A bad epoch does not prevent later epochs from syncing.
    assert "error ValueError" in capsys.readouterr().out
    assert not (tmp_path / f"accepted-{EP}.jsonl").exists()
    assert not (tmp_path / f"accepted-{EP}.done").exists()


def test_sha_mismatch_writes_nothing(tmp_path):
    assert run(tmp_path, fetcher(checksum="0" * 64)) == "sha-mismatch (skip; retry next run)"
    assert not (tmp_path / f"accepted-{EP}.jsonl").exists()
    assert not (tmp_path / f"accepted-{EP}.done").exists()


def test_legacy_sha_mismatch_does_not_backfill(tmp_path):
    final = tmp_path / f"accepted-{EP}.jsonl"
    original = lines(INTAKE)
    final.write_bytes(original)
    assert run(tmp_path, fetcher(checksum="0" * 64), backfill=True) == "sha-mismatch (skip; retry next run)"
    assert final.read_bytes() == original
    assert not (tmp_path / f"accepted-{EP}.done").exists()


def test_v2_done_repair_makes_no_network_requests(tmp_path):
    run(tmp_path)
    (tmp_path / f"accepted-{EP}.done").unlink()
    def unavailable(url):
        raise AssertionError("valid v2 file should not need network")
    assert run(tmp_path, unavailable) == "done repaired"
    assert run(tmp_path, unavailable) == "exists"


def test_missing_map_means_null_licenses(tmp_path):
    assert sync.sync_epoch(EP, out=str(tmp_path), fetch=fetcher(), pool_map=str(tmp_path / "missing.gz")) == "wrote 2"
    assert [row["license"] for row in read(tmp_path / f"accepted-{EP}.jsonl")] == [None, None]


def test_future_intake_fields_are_used_and_task_id_cross_checked(tmp_path):
    intake = [{**row, "task_id": accepted["task_id"], "license": "future-license", "king": "future-king"} for row, accepted in zip(INTAKE, ACCEPTED)]
    assert run(tmp_path, fetcher(intake=intake)) == "wrote 2"
    for row in read(tmp_path / f"accepted-{EP}.jsonl"):
        assert row["license"] == "future-license" and row["king"] == "future-king"
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="task_id mismatch"):
        sync.sync_epoch(EP, out=str(other), fetch=fetcher(intake=intake[:1] + [{**intake[1], "task_id": "c" * 20}]), pool_map=pool_map(other))


def test_empty_intake_writes_empty_file_and_zero_done(tmp_path):
    assert run(tmp_path, fetcher(intake=[], accepted=[])) == "wrote 0"
    final = tmp_path / f"accepted-{EP}.jsonl"
    assert final.read_bytes() == b""
    assert json.loads((tmp_path / f"accepted-{EP}.done").read_text())["lines"] == 0


def test_main_epoch_and_state_paths(tmp_path):
    fetch = fetcher()
    assert sync.main(["--epoch", EP, "--out", str(tmp_path), "--pool-map", pool_map(tmp_path)], fetch=fetch) == 0
    other = tmp_path / "state"
    assert sync.main(["--out", str(other), "--pool-map", pool_map(other)], fetch=fetch) == 0
    assert (other / f"accepted-{EP}.done").is_file()
    assert sync.main(["--epoch", "../bad", "--out", str(tmp_path)], fetch=fetch) == 2
