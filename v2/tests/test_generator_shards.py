"""Multi-pod shard scoring and merge, entirely with fakes and local files."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from sn120_v2.envs.math_grader import Grade
from sn120_v2.errors import InfraError
from sn120_v2.generator.constants import POOL_ID
from sn120_v2.generator.epoch import epoch_deadline
from sn120_v2.generator.fake_vllm import FakeFarm, FakePool
from sn120_v2.generator.remote import ScorerRemote, SSHConfig, SSHRunner
from sn120_v2.generator.runner import run_job
from sn120_v2.generator.scoring import ScoringJob, TaskSpec, rollout_seed


def job(count=6):
    return ScoringJob(
        "20261120T0000Z",
        tuple(TaskSpec(f"{i + 1:020x}", f"owner-{i % 2}") for i in range(count)),
        "a" * 64,
        "shard-secret",
        time.time() + 120,
        POOL_ID,
    )


def grades():
    class Grader:
        def grade_many(self, jobs):
            return [Grade("P", 1.0, False, "\\boxed{1}", "fake") if "\\boxed{1}" in output and not capped else Grade("F", 0.0, False, "\\boxed{2}", "fake") for _, output, capped in jobs]
    return Grader()


def test_two_shard_results_cover_job_and_k1_is_unchanged(tmp_path):
    current = job()
    with FakeFarm(first_port=0, mode="mixed") as farm:
        for task in current.tasks:
            for sample in range(32):
                farm.seeds[rollout_seed(current.seed_secret, task.task_id, sample)] = (task.task_id, sample)
        full = run_job(current, tmp_path / "full", pool=FakePool(), grader=grades(), urls=farm.urls, clients_per_server=4)
    # The fake farm above is closed only after the run; use independent farms for shards.
    shard_documents = []
    for index in range(2):
        with FakeFarm(first_port=0, mode="mixed") as farm:
            for task in current.tasks:
                for sample in range(32):
                    farm.seeds[rollout_seed(current.seed_secret, task.task_id, sample)] = (task.task_id, sample)
            shard_documents.append(run_job(current, tmp_path / f"shard-{index}", pool=FakePool(), grader=grades(), urls=farm.urls, clients_per_server=4, shard=(index, 2)))
    assert set(full) == set(shard_documents[0]) - {"shard"}
    merged_ids = {row["id"]: row for row in shard_documents[0]["tasks"] + shard_documents[1]["tasks"]}
    assert [row["id"] for row in full["tasks"]] == [task.task_id for task in current.tasks]
    assert set(merged_ids) == {task.task_id for task in current.tasks}
    assert [row["id"] for row in shard_documents[0]["tasks"]] == [current.tasks[i].task_id for i in range(0, len(current.tasks), 2)]
    assert [row["id"] for row in shard_documents[1]["tasks"]] == [current.tasks[i].task_id for i in range(1, len(current.tasks), 2)]
    assert all(document["shard"]["count"] == 2 for document in shard_documents)
    assert all("shard" not in document for document in [full])


def test_merge_and_reject_mismatch_overlap(tmp_path):
    current = job(2)
    (tmp_path / "key").write_text("fake")
    config = SSHConfig("primary", 22, tmp_path / "key", tmp_path / "work")
    remote = ScorerRemote(config)
    (remote.work_dir / "jobs" / current.epoch).mkdir(parents=True)
    (remote.work_dir / "jobs" / current.epoch / "job.json").write_text(json.dumps(current.to_dict()))
    primary = remote.work_dir / "epochs" / current.epoch / "out-shard-0"
    helper = remote.work_dir / "epochs" / current.epoch / "out-shard-1"
    primary.mkdir(parents=True)
    helper.mkdir(parents=True)
    def document(index):
        return {
            "epoch": current.epoch, "head_version": current.head_version,
            "seed_commitment": "a" * 64, "pool_pin": current.pool_pin,
            "shard": {"index": index, "count": 2, "tasks": 1},
            "tasks": [{"id": current.tasks[index].task_id, "s": 4, "n": 8, "decision": "accept"}],
            "timings": {"wall_s": 1}, "throughput": {"rollouts": 8}, "servers": [], "hotkeys": {"owner": {"submitted": index + 1, "accepted": 1}},
        }
    for path, index in ((primary, 0), (helper, 1)):
        (path / "results.json").write_text(json.dumps(document(index)))
        (path / "rollouts").mkdir()
        (path / "rollouts" / f"{current.tasks[index].task_id}.jsonl").write_text("{}\n")
    merged = remote._merge_shards(current.epoch, primary, [helper])
    results = json.loads((merged / "results.json").read_text())
    assert [row["id"] for row in results["tasks"]] == [task.task_id for task in current.tasks]
    assert "shard" not in results
    assert results["throughput"] == [{"rollouts": 8}, {"rollouts": 8}]
    assert results["hotkeys"] == {"owner": {"submitted": 3, "accepted": 2}}
    (primary / "rollouts" / f"{current.tasks[1].task_id}.jsonl").write_text("{}\n")
    with pytest.raises(InfraError, match="overlap"):
        remote._merge_shards(current.epoch, primary, [helper])
    (primary / "rollouts" / f"{current.tasks[1].task_id}.jsonl").unlink()
    bad = json.loads((helper / "results.json").read_text())
    bad["head_version"] = "b" * 64
    (helper / "results.json").write_text(json.dumps(bad))
    with pytest.raises(InfraError, match="identity"):
        remote._merge_shards(current.epoch, primary, [helper])


def test_repeated_dispatch_launches_primary_and_helper_once(tmp_path):
    from test_generator_remote import EPOCH, FakeRemote
    key = tmp_path / "key"
    key.write_text("fake")
    config = SSHConfig("primary", 22, key, tmp_path / "work")
    primary = FakeRemote(config)
    helper = FakeRemote(config)
    helper.destination = "root@helper"
    remote = ScorerRemote(config, runner=primary, helpers=[helper])
    epoch = EPOCH
    job_path = remote.work_dir / "jobs" / epoch / "job.json"
    job_path.parent.mkdir(parents=True)
    job_path.write_text(json.dumps({"epoch": epoch, "head_version": "a" * 64, "tasks": []}))
    for _ in range(5):
        remote.dispatch(epoch)
    assert sum("nohup" in command for _, command in primary.calls) == 1
    assert sum("nohup" in command for _, command in helper.calls) == 1
    assert sum("ensure_head" in command for _, command in primary.calls) == 1
    assert sum("ensure_head" in command for _, command in helper.calls) == 1


def test_delayed_helper_never_creates_partial_output(tmp_path):
    from test_generator_remote import EPOCH, FakeRemote, public_archive
    key = tmp_path / "key"
    key.write_text("fake")
    config = SSHConfig("primary", 22, key, tmp_path / "work")
    primary = FakeRemote(config)
    helper = FakeRemote(config)
    helper.destination = "root@helper"
    remote = ScorerRemote(config, runner=primary, helpers=[helper])
    epoch = EPOCH
    job_path = remote.work_dir / "jobs" / epoch / "job.json"
    job_path.parent.mkdir(parents=True)
    job_path.write_text(json.dumps({"epoch": epoch, "head_version": "a" * 64, "tasks": []}))
    remote.dispatch(epoch)
    primary.markers["done.json"] = '{"code":0}'
    primary.archive = public_archive()
    helper.archive = public_archive()
    assert remote.collect(epoch).state == "running"
    assert not (remote.work_dir / "epochs" / epoch / "out").exists()
    helper.markers["done.json"] = '{"code":0}'
    with pytest.raises(InfraError):
        remote.collect(epoch)
    assert not (remote.work_dir / "epochs" / epoch / "out").exists()


def test_helper_config_and_dispatch_idempotency(tmp_path):
    key = tmp_path / "key"
    key.write_text("fake")
    config = {
        "host": "primary", "port": 22, "key_path": str(key),
        "helpers": [{"host": "helper", "port": 23, "user": "root", "key_path": str(key)}],
    }
    remote = ScorerRemote.from_config(config, tmp_path / "work")
    assert len(remote.helpers) == 1 and remote.helpers[0].destination == "root@helper"
    calls = []
    class Runner:
        destination = "root@helper"
        def ssh(self, command, *, timeout=None):
            calls.append(command)
            return type("Reply", (), {"stdout": ""})()
        def scp(self, source, destination, *, timeout=None):
            calls.append("copy")
    remote.helpers = (Runner(),)
    epoch = "20261120T0000Z"
    job_path = remote.work_dir / "jobs" / epoch / "job.json"
    job_path.parent.mkdir(parents=True)
    job_path.write_text(json.dumps({"epoch": epoch, "head_version": "a" * 64, "tasks": []}))
    state = remote.work_dir / "jobs" / epoch / "remote-state.json"
    remote._dispatch_helpers(epoch, job_path, state, "a" * 64, {})
    before = list(calls)
    remote._dispatch_helpers(epoch, job_path, state, "a" * 64, {})
    assert calls == before
    assert any("--shard 1/2" in command for command in before)
    assert any("ensure_head" in command for command in before)
