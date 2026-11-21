"""Generator-feed deployment safeguards."""
from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import pytest

from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.envs.reserve import MANIFEST, WITHDRAWN, prepare_live_reserve


def reserve(root: Path, count: int = 1) -> None:
    """Build a minimal packaged seed without importing another test module."""
    (root / "shards").mkdir(parents=True, exist_ok=True)
    (root / WITHDRAWN).write_bytes(b"")
    manifest = []
    for number in range(count):
        row = {"id": f"seed:{number}", "env": "affine_math", "problem": f"Unique reserve problem seed {number}", "answer": str(number),
               "source": f"test@rev:{number}", "license": "CC0-1.0", "added_at": "2026-11-20",
               "audit": {"verdict": "keep", "method": "unit", "ref": "test"}}
        raw = canonical_json(row) + b"\n"
        name = f"{number + 1:04d}-seed.jsonl"
        (root / "shards" / name).write_bytes(raw)
        manifest.append({"shard": name, "sha256": sha256_hex(raw), "count": 1, "added_at": "2026-11-20"})
    manifest.append({"withdrawn": {"sha256": sha256_hex(b""), "count": 0}})
    (root / MANIFEST).write_bytes(b"".join(canonical_json(row) + b"\n" for row in manifest))


@pytest.fixture
def genfeed_operator(tmp_path, monkeypatch):
    count = itertools.count()
    def load(live: Path, inbox: Path):
        kit = Path(__file__).parents[1] / "tools/genfeed-deploy"

        target = tmp_path / "kit"
        target.mkdir(exist_ok=True)
        (target / "expect.json").write_text((kit / "expect.json").read_text())
        spec = importlib.util.spec_from_file_location(f"genfeed_test_operator_{next(count)}", kit / "ol_operator.py")
        operator = importlib.util.module_from_spec(spec)
        for name, value in {"R2_HERE": target, "R2_V2": tmp_path / "v2", "R2_RESERVE_DIR": live,
                            "R2_GENERATOR_INBOX": inbox, "R2_PY": "python3"}.items():
            monkeypatch.setenv(name, str(value))
        spec.loader.exec_module(operator)
        return operator
    return load



def test_ingest_ready_fails_on_zero_kept_or_no_complete_epoch(tmp_path, genfeed_operator, capsys):
    from types import SimpleNamespace
    live, inbox = tmp_path / "live", tmp_path / "inbox"
    live.mkdir(), inbox.mkdir()
    operator = genfeed_operator(live, inbox)
    epoch = "20261120T0400Z"
    (inbox / f"accepted-{epoch}.jsonl").write_bytes(b"{}\n")
    (inbox / f"accepted-{epoch}.done").write_text("{}")
    (live / "generator-ingested.jsonl").write_text(json.dumps({"epoch": "20261119T2300Z", "ingested": 0}) + "\n")
    ledger = live / "generator-ingested.jsonl"
    prior = ledger.read_bytes()
    def zero_kept(*args, **kwargs):
        ledger.write_bytes(prior + json.dumps({"epoch": epoch, "ingested": 0}).encode() + b"\n")
        return SimpleNamespace(returncode=0, stdout="nothing kept")
    operator.sh = zero_kept
    with pytest.raises(SystemExit, match="kept 0 tasks"):
        operator.ingest_ready()
    assert "INGEST READY DONE" not in capsys.readouterr().out
    (live / "generator-ingested.jsonl").unlink()
    operator.sh = lambda *args, **kwargs: SimpleNamespace(returncode=3, stdout="incomplete", stderr="not complete")
    with pytest.raises(SystemExit, match="no complete epoch was ingested"):
        operator.ingest_ready()
    assert "INGEST READY DONE" not in capsys.readouterr().out
    operator.sh = lambda *args, **kwargs: (
        (live / "generator-ingested.jsonl").write_text(json.dumps({"epoch": epoch, "ingested": 1}) + "\n"),
        SimpleNamespace(returncode=0, stdout="kept one"))[1]
    operator.ingest_ready()
    assert "INGEST READY DONE" in capsys.readouterr().out


def test_ingest_ready_real_done_without_results(tmp_path, genfeed_operator):
    from types import SimpleNamespace
    live, inbox = tmp_path / "live", tmp_path / "inbox"
    live.mkdir(), inbox.mkdir()
    operator = genfeed_operator(live, inbox)
    fixture = Path(__file__).parent / "fixtures/generator-intake-v2-20261120T0400Z"
    epoch = "20261120T0400Z"
    for suffix in (".jsonl", ".done"):
        (inbox / f"accepted-{epoch}{suffix}").write_bytes(fixture.with_suffix(suffix).read_bytes())
    calls = []
    def successful_ingest(args, **kwargs):
        calls.append(args)
        (live / "generator-ingested.jsonl").write_text(json.dumps({"epoch": epoch, "ingested": 3}) + "\n")
        return SimpleNamespace(returncode=0, stdout="kept three")
    operator.sh = successful_ingest
    operator.ingest_ready()
    assert len(calls) == 1 and "--results" not in calls[0]
    assert Path(calls[0][calls[0].index("ingest-generator") + 1]) == inbox / f"accepted-{epoch}.jsonl"


def test_dirty_leftover_live_is_backed_up_and_cleanly_reseeded(tmp_path, genfeed_operator):
    seed, live = tmp_path / "packaged", tmp_path / "live"
    reserve(seed, 1)
    operator = genfeed_operator(live, tmp_path / "inbox")
    operator.RESERVE = seed
    operator.EXPECT = operator.EXPECT | {"new_manifest_sha256": sha256_hex((seed / MANIFEST).read_bytes())}
    prepare_live_reserve(live, seed)
    assert operator.preserve_dirty_live() is None
    failed_ledger = {"epoch": "20261120T0400Z", "ingested": 0, "input_sha256": "8ae473e2"}
    (live / "generator-ingested.jsonl").write_text(json.dumps(failed_ledger) + "\n")
    (live / "retired.json").write_text("[]")
    packaged_before = {path.relative_to(seed): path.read_bytes() for path in seed.rglob("*") if path.is_file()}
    backup = operator.preserve_dirty_live()
    assert backup is not None and ".backup-" in backup.name and not live.exists()
    assert json.loads((backup / "generator-ingested.jsonl").read_text()) == failed_ledger
    assert (backup / "retired.json").read_text() == "[]"
    prepare_live_reserve(live, seed)
    assert (live / MANIFEST).read_bytes() == (seed / MANIFEST).read_bytes()
    assert not (live / "generator-ingested.jsonl").exists()
    assert {path.relative_to(seed): path.read_bytes() for path in seed.rglob("*") if path.is_file()} == packaged_before
    prepare_live_reserve(live, seed)
    (live / "shards" / "leftover.jsonl.rejected.json").write_text("{}\n")
    backup = operator.preserve_dirty_live()
    assert backup is not None and (backup / "shards/leftover.jsonl.rejected.json").is_file()
    prepare_live_reserve(live, seed)
    (live / "leftover.txt").write_text("leftover\n")
    backup = operator.preserve_dirty_live()
    assert backup is not None and (backup / "leftover.txt").read_text() == "leftover\n"
    operator.LIVE = seed
    with pytest.raises(SystemExit, match="packaged reserve"):
        operator.preserve_dirty_live()


def test_apply_reseeds_failed_leftover_live_before_ingest(tmp_path, genfeed_operator):
    from types import SimpleNamespace
    seed, live = tmp_path / "packaged", tmp_path / "reserve-live"
    reserve(seed, 0)
    operator = genfeed_operator(live, tmp_path / "inbox")
    operator.RESERVE = seed
    operator.V2 = tmp_path / "v2"
    operator.PATH_FILE = tmp_path / "reserve-live.path"
    operator.DRY = True
    packaged = operator.V2 / "sn120_v2/envs/reserve"
    packaged.mkdir(parents=True)
    (packaged / MANIFEST).write_bytes((seed / MANIFEST).read_bytes())
    operator.EXPECT = operator.EXPECT | {"new_manifest_sha256": sha256_hex((seed / MANIFEST).read_bytes()),
                                          "old_commit": "old", "new_commit": "new", "old_hash": "params", "new_hash": "params"}
    prepare_live_reserve(live, seed)
    (live / "generator-ingested.jsonl").write_text(json.dumps({"epoch": "20261120T0400Z", "ingested": 0, "input_sha256": "8ae473e2"}) + "\n")
    (live / "retired.json").write_text("[]")
    (live / "retired_tasks.json").write_text("[]")
    packaged_before = {path.relative_to(seed): path.read_bytes() for path in seed.rglob("*") if path.is_file()}
    calls = []
    current_head = ["old"]
    operator.preflight = lambda: {"params_hash": "params"}
    operator.head = lambda: current_head[0]
    operator.digest = lambda: operator.EXPECT["digest"]
    operator.verify_prior_index = lambda: None
    operator.pub_get = lambda key: (seed / MANIFEST).read_bytes()
    operator.record_target = lambda state: {"commit": "old"}
    operator.checkout = lambda commit: current_head.__setitem__(0, commit)
    operator.pool_ok = lambda: (True, "")
    operator.params_hash = lambda path: "params"
    operator.pinned_digest = lambda: operator.EXPECT["digest"]
    def fake_sh(args, **kwargs):
        calls.append(args)
        if args[:3] == ["git", "cat-file", "-t"]:
            return SimpleNamespace(stdout="commit\n", returncode=0)
        if args[:2] == ["python3", "-c"] and "prepare_live_reserve" in args[2]:
            prepare_live_reserve(live, seed)
        if args[:2] == ["python3", "-c"] and "load_params" in args[2]:
            return SimpleNamespace(stdout="params\n")
        return SimpleNamespace(stdout="new\n", returncode=0)
    operator.sh = fake_sh
    operator.reserve_verify = lambda: (True, "")
    operator.apply()
    backups = list(tmp_path.glob("reserve-live.backup-*"))
    assert len(backups) == 1 and (backups[0] / "generator-ingested.jsonl").exists()
    assert (live / MANIFEST).read_bytes() == (seed / MANIFEST).read_bytes()
    assert not (live / "generator-ingested.jsonl").exists() and not (live / "retired.json").exists()
    assert {path.relative_to(seed): path.read_bytes() for path in seed.rglob("*") if path.is_file()} == packaged_before
