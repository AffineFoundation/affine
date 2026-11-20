"""Multi-pod shard scoring and merge, entirely with fakes and local files."""
from __future__ import annotations

import json
import shutil
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


def job(count=6, *, deadline=None):
    return ScoringJob(
        "20261120T0000Z",
        tuple(TaskSpec(f"{i + 1:020x}", f"owner-{i % 2}") for i in range(count)),
        "a" * 64,
        "shard-secret",
        time.time() + 120 if deadline is None else deadline,
        POOL_ID,
    )


def transport_job(count=6):
    """A timestamp-valid scoring job, for transport tests that do not score."""
    return ScoringJob(
        "20991231T0000Z",
        tuple(TaskSpec(f"{i + 1:020x}", f"owner-{i % 2}") for i in range(count)),
        "a" * 64,
        "shard-secret",
        4102444800.0,
        POOL_ID,
    )
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
            "tasks": [{"id": current.tasks[index].task_id, "hotkey": current.tasks[index].hotkey, "s": 4, "n": 8, "decision": "accept"}],
            "timings": {"wall_s": 1}, "throughput": {"rollouts": 8}, "servers": [], "hotkeys": {"owner": {"submitted": index + 1, "accepted": 1}},
        }
    for path, index in ((primary, 0), (helper, 1)):
        (path / "results.json").write_text(json.dumps(document(index)))
        (path / "rollouts").mkdir()
        (path / "rollouts" / f"{current.tasks[index].task_id}.jsonl").write_text("{}\n")
    merged = remote._merge_shards(current.epoch, primary, [(1, helper)])
    results = json.loads((merged / "results.json").read_text())
    assert [row["id"] for row in results["tasks"]] == [task.task_id for task in current.tasks]
    assert "shard" not in results
    assert results["throughput"] == {
        "rollouts": 16, "new_rollouts": 0, "requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "generate_seconds": 0, "grade_seconds": 0, "rollouts_per_s": 0.0, "completion_tokens_per_s": 0.0,
        "clients_per_server": 0, "servers": 0,
    }
    assert results["hotkeys"] == {
        "owner-0": {"submitted": 1, "eligible": 1, "scored": 1, "accepted": 1, "rejected": 0, "unscored": 0, "rollouts": 8, "passes": 4},
        "owner-1": {"submitted": 1, "eligible": 1, "scored": 1, "accepted": 1, "rejected": 0, "unscored": 0, "rollouts": 8, "passes": 4},
    }
    shutil.rmtree(merged)
    (primary / "rollouts" / f"{current.tasks[1].task_id}.jsonl").write_text("{}\n")
    with pytest.raises(InfraError, match="overlap"):
        remote._merge_shards(current.epoch, primary, [(1, helper)])
    (primary / "rollouts" / f"{current.tasks[1].task_id}.jsonl").unlink()
    shutil.rmtree(remote.work_dir / "epochs" / current.epoch / ".merged-staging", ignore_errors=True)
    bad = json.loads((helper / "results.json").read_text())
    bad["head_version"] = "b" * 64
    (helper / "results.json").write_text(json.dumps(bad))
    with pytest.raises(InfraError, match="identity"):
        remote._merge_shards(current.epoch, primary, [(1, helper)])


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
    job_path.write_text(json.dumps(transport_job().to_dict()))
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
    job_path.write_text(json.dumps(transport_job().to_dict()))
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


def test_epoch_helper_set_is_pinned_at_primary_launch(tmp_path):
    from test_generator_remote import EPOCH, FakeRemote, public_archive
    key = tmp_path / "key"
    key.write_text("fake")
    config = SSHConfig("primary", 22, key, tmp_path / "work")
    primary = FakeRemote(config)
    remote = ScorerRemote(config, runner=primary)
    epoch = EPOCH
    job_path = remote.work_dir / "jobs" / epoch / "job.json"
    job_path.parent.mkdir(parents=True)
    job_path.write_text(json.dumps(transport_job().to_dict()))
    remote.dispatch(epoch)
    state = json.loads((remote.work_dir / "jobs" / epoch / "remote-state.json").read_text())
    assert state["shard_count"] == 1 and state["helper_identities"] == []
    helper = FakeRemote(config)
    helper.destination = "root@new-helper"
    remote.helpers = (helper,)
    remote.dispatch(epoch)
    assert not helper.calls
    primary.markers["done.json"] = '{"code":0}'
    primary.archive = public_archive()
    assert remote.collect(epoch).complete


def test_legacy_state_without_shard_count_never_calls_helpers(tmp_path):
    from test_generator_remote import EPOCH, FakeRemote, public_archive
    key = tmp_path / "key"
    key.write_text("fake")
    config = SSHConfig("primary", 22, key, tmp_path / "work")
    epoch = EPOCH
    job_path = tmp_path / "work" / "jobs" / epoch / "job.json"
    job_path.parent.mkdir(parents=True)
    job_path.write_text(json.dumps(transport_job().to_dict()))
    state_path = tmp_path / "work" / "jobs" / epoch / "remote-state.json"
    state_path.write_text(json.dumps({
        "epoch": epoch, "state": "dispatched", "launch_attempted": True,
        "job_sha256": __import__("sn120_v2.canonical", fromlist=["sha256_hex"]).sha256_hex(job_path.read_bytes()), "attempts": 1,
    }))
    primary = FakeRemote(config)
    helper = FakeRemote(config)
    helper.destination = "root@legacy-helper"
    remote = ScorerRemote(config, runner=primary, helpers=[helper])
    remote.dispatch(epoch)
    assert helper.calls == []
    primary.markers["done.json"] = '{"code":0}'
    primary.archive = public_archive()
    assert remote.collect(epoch).complete


def test_k1_sticky_states_return_without_any_ssh_or_scp(tmp_path):
    from test_generator_remote import EPOCH, FakeRemote
    for state_name in ("running", "complete", "failed"):
        for with_helpers in (False, True):
            key = tmp_path / f"{state_name}-{with_helpers}-key"
            key.write_text("fake")
            config = SSHConfig("primary", 22, key, tmp_path / f"{state_name}-{with_helpers}-work")
            primary = FakeRemote(config)
            helper = FakeRemote(config)
            remote = ScorerRemote(config, runner=primary, helpers=[helper] if with_helpers else [])
            epoch = EPOCH
            job_path = remote.work_dir / "jobs" / epoch / "job.json"
            job_path.parent.mkdir(parents=True)
            job_path.write_text(json.dumps(transport_job().to_dict()))
            if state_name == "complete":
                result_path = remote.work_dir / "epochs" / epoch / "out" / "results.json"
                result_path.parent.mkdir(parents=True)
                result_path.write_text("{}")
            state = remote.work_dir / "jobs" / epoch / "remote-state.json"
            state.write_text(json.dumps({
                "epoch": epoch, "state": state_name, "launch_attempted": True,
                "job_sha256": __import__("sn120_v2.canonical", fromlist=["sha256_hex"]).sha256_hex(job_path.read_bytes()), "attempts": 1,
            }))
            result = remote.dispatch(epoch)
            expected = state_name
            assert result.state == expected
            assert primary.calls == [] and helper.calls == []
            assert primary.copies == [] and helper.copies == []


def test_launch_matrix_preserves_primary_for_every_k_and_state(tmp_path):
    from test_generator_remote import EPOCH, FakeRemote
    for state_name in ("preparing", "dispatched", "running", "collecting", "complete", "failed"):
        for launched in (False, True):
            for count in (1, 2):
                key = tmp_path / f"{state_name}-{launched}-{count}-key"
                key.write_text("fake")
                config = SSHConfig("primary", 22, key, tmp_path / f"{state_name}-{launched}-{count}-work")
                primary = FakeRemote(config)
                helper = FakeRemote(config)
                helper.config = config
                remote = ScorerRemote(config, runner=primary, helpers=[helper] if count == 2 else [])
                epoch = EPOCH
                job_path = remote.work_dir / "jobs" / epoch / "job.json"
                job_path.parent.mkdir(parents=True)
                job_path.write_text(json.dumps(transport_job().to_dict()))
                if state_name == "complete":
                    result_path = remote.work_dir / "epochs" / epoch / "out" / "results.json"
                    result_path.parent.mkdir(parents=True)
                    result_path.write_text("{}")
                state = remote.work_dir / "jobs" / epoch / "remote-state.json"
                state.write_text(json.dumps({
                    "epoch": epoch, "state": state_name, "launch_attempted": launched,
                    "shard_count": count,
                    "helper_identities": [{"host": config.host, "port": config.port}] if count == 2 else [],
                    "job_sha256": __import__("sn120_v2.canonical", fromlist=["sha256_hex"]).sha256_hex(job_path.read_bytes()), "attempts": 1,
                }))
                remote.dispatch(epoch)
                if launched:
                    assert sum("nohup" in command for _, command in primary.calls) == 0
                    assert primary.copies == []
                if count == 2:
                    assert sum("nohup" in command for _, command in helper.calls) <= 1


def test_ten_tick_k2_launch_and_merge_once(tmp_path):
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
    job_path.write_text(json.dumps(transport_job().to_dict()))
    for _ in range(10):
        remote.dispatch(epoch)
    assert sum("nohup" in command for _, command in primary.calls) == 1
    assert sum("nohup" in command for _, command in helper.calls) == 1
    primary.markers["done.json"] = '{"code":0}'
    helper.markers["done.json"] = '{"code":0}'
    primary.archive = public_archive()
    helper.archive = public_archive()
    with pytest.raises(InfraError):
        remote.collect(epoch)
    assert not (remote.work_dir / "epochs" / epoch / "out").exists()


def test_helper_failure_deadline_synthesizes_time_budget_without_rollouts(tmp_path, monkeypatch):
    current = job(2)
    (tmp_path / "key").write_text("fake")
    config = SSHConfig("primary", 22, tmp_path / "key", tmp_path / "work")
    remote = ScorerRemote(config)
    monkeypatch.setattr("sn120_v2.generator.remote.time.time", lambda: current.budget_deadline + 1)
    (remote.work_dir / "jobs" / current.epoch).mkdir(parents=True)
    (remote.work_dir / "jobs" / current.epoch / "job.json").write_text(json.dumps(current.to_dict()))
    (remote.work_dir / "jobs" / current.epoch / "remote-state.json").write_text(json.dumps({"shard_count": 2, "results_grace_s": 0}))
    primary = remote.work_dir / "epochs" / current.epoch / "out-shard-0"
    primary.mkdir(parents=True)
    (primary / "results.json").write_text(json.dumps({
        "epoch": current.epoch, "head_version": current.head_version, "seed_commitment": "a" * 64, "budget_deadline": current.budget_deadline,
        "pool_pin": current.pool_pin, "shard": {"index": 0, "count": 2, "tasks": 1},
        "tasks": [{"id": current.tasks[0].task_id, "hotkey": current.tasks[0].hotkey, "s": 4, "n": 8, "q": 0.5, "decision": "accept"}],
        "timings": {}, "throughput": {}, "servers": [], "hotkeys": {"owner": {"submitted": 1, "accepted": 1}},
    }))
    (primary / "rollouts").mkdir()
    (primary / "rollouts" / f"{current.tasks[0].task_id}.jsonl").write_text("{}\n")
    merged = remote._merge_shards(current.epoch, primary, [])
    document = json.loads((merged / "results.json").read_text())
    assert document["tasks"][0]["decision"] == "accept"
    assert document["tasks"][1]["decision"] == "unscored"
    assert document["tasks"][1]["unscored_reason"] == "time_budget" and document["tasks"][1]["n"] == 0
    assert not (merged / "rollouts" / f"{current.tasks[1].task_id}.jsonl").exists()


def test_outcome_matrix_launch_and_helper_failure(tmp_path):
    from test_generator_remote import EPOCH, FakeRemote
    for state_name in ("preparing", "running", "collecting", "complete", "failed"):
        for count in (1, 2):
            for outcome in ("ok", "dead", "unreachable"):
                key = tmp_path / f"{state_name}-{count}-{outcome}-key"
                key.write_text("fake")
                config = SSHConfig("primary", 22, key, tmp_path / f"{state_name}-{count}-{outcome}-work")
                primary = FakeRemote(config)
                helper = FakeRemote(config)
                remote = ScorerRemote(config, runner=primary, helpers=[helper] if count == 2 else [])
                epoch = EPOCH
                job_path = remote.work_dir / "jobs" / epoch / "job.json"
                job_path.parent.mkdir(parents=True)
                job_path.write_text(json.dumps(transport_job().to_dict()))
                state = remote.work_dir / "jobs" / epoch / "remote-state.json"
                digest = __import__("sn120_v2.canonical", fromlist=["sha256_hex"]).sha256_hex(job_path.read_bytes())
                state.write_text(json.dumps({
                    "epoch": epoch, "state": state_name, "launch_attempted": state_name != "preparing",
                    "shard_count": count, "helper_identities": [{"host": config.host, "port": config.port}] if count == 2 else [],
                    "job_sha256": digest, "attempts": 1,
                    "helpers": [{"epoch": epoch, "state": "dispatched", "shard": 1, "launch_attempted": True, "job_sha256": digest}]
                    if count == 2 and state_name != "preparing" else [],
                }))
                if outcome == "unreachable":
                    helper.copy_error = InfraError("unreachable")
                if state_name == "complete":
                    result_path = remote.work_dir / "epochs" / epoch / "out" / "results.json"
                    result_path.parent.mkdir(parents=True)
                    result_path.write_text("{}")
                if state_name == "preparing" and count == 2 and outcome == "unreachable":
                    assert remote.dispatch(epoch).state == "dispatched"
                    assert sum("nohup" in command for _, command in helper.calls) == 0
                else:
                    remote.dispatch(epoch)
                if state_name != "preparing":
                    assert sum("nohup" in command for _, command in primary.calls) == 0
                    assert primary.copies == []
                assert sum("nohup" in command for _, command in primary.calls) <= 1
                assert sum("nohup" in command for _, command in helper.calls) <= 1


def test_launch_matrix_helper_launch_uncertainty_never_relaunches_primary(tmp_path):
    from test_generator_remote import EPOCH, FakeRemote
    key = tmp_path / "key"
    key.write_text("fake")
    config = SSHConfig("primary", 22, key, tmp_path / "work")
    primary, helper = FakeRemote(config), FakeRemote(config)
    helper.launch_error = InfraError("launch outcome unknown")
    remote = ScorerRemote(config, runner=primary, helpers=[helper])
    job_path = remote.work_dir / "jobs" / EPOCH / "job.json"
    job_path.parent.mkdir(parents=True)
    job_path.write_text(json.dumps(transport_job().to_dict()))
    with pytest.raises(InfraError, match="unknown"):
        remote.dispatch(EPOCH)
    state = json.loads((remote.work_dir / "jobs" / EPOCH / "remote-state.json").read_text())
    state["launch_attempted"] = True
    (remote.work_dir / "jobs" / EPOCH / "remote-state.json").write_text(json.dumps(state))
    resumed_primary, resumed_helper = FakeRemote(config), FakeRemote(config)
    remote = ScorerRemote(config, runner=resumed_primary, helpers=[resumed_helper])
    assert remote.dispatch(EPOCH).state == "dispatched"
    assert sum("nohup" in command for _, command in resumed_primary.calls) == 0
    assert sum("nohup" in command for _, command in resumed_helper.calls) == 1
    before = list(resumed_primary.calls), list(resumed_helper.calls)
    assert remote.dispatch(EPOCH).state == "dispatched"
    assert (resumed_primary.calls, resumed_helper.calls) == before

def test_unreachable_helper_polls_until_grace_then_publishes_time_budget(tmp_path, monkeypatch):
    from test_generator_remote import FakeRemote, archive
    epoch = "20261120T0000Z"
    current = ScoringJob(epoch, (TaskSpec("0" * 20, "owner-0"), TaskSpec("1" * 20, "owner-1")), "a" * 64, "secret", epoch_deadline(epoch).timestamp() + 13500, POOL_ID)
    key = tmp_path / "key"
    key.write_text("fake")
    config = SSHConfig("primary", 22, key, tmp_path / "work")
    primary, helper = FakeRemote(config), FakeRemote(config)
    remote = ScorerRemote(config, runner=primary, helpers=[helper])
    job_path = remote.work_dir / "jobs" / epoch / "job.json"
    job_path.parent.mkdir(parents=True)
    job_path.write_text(json.dumps(current.to_dict()))
    remote.dispatch(epoch)
    primary.markers["done.json"] = '{"code":0}'
    document = {
        "epoch": epoch, "head_version": current.head_version, "seed_commitment": "a" * 64, "pool_pin": current.pool_pin,
        "budget_deadline": current.budget_deadline, "shard": {"index": 0, "count": 2, "tasks": 1},
        "tasks": [{"id": "0" * 20, "hotkey": "owner-0", "s": 4, "n": 8, "q": 0.5, "decision": "accept"}],
        "timings": {"wall_s": 12}, "throughput": {"rollouts": 8, "servers": 8}, "servers": [], "hotkeys": {},
    }
    primary.archive = archive({"./results.json": json.dumps(document).encode(), f"./rollouts/{'0' * 20}.jsonl": b"{}\n"})
    helper.markers["done.json"] = "not-json"
    clock = [current.budget_deadline + 2699]
    monkeypatch.setattr("sn120_v2.generator.remote.time.time", lambda: clock[0])
    remote.collect(epoch)
    assert remote.collect(epoch).state == "running"
    assert not (remote.work_dir / "epochs" / epoch / "out").exists()
    assert not (remote.work_dir / "epochs" / epoch / "out").exists()
    clock[0] += 2
    assert remote.collect(epoch).complete
    results = json.loads((remote.work_dir / "epochs" / epoch / "out" / "results.json").read_text())
    assert results["tasks"][1]["unscored_reason"] == "time_budget"
    assert not any(kind == "tar" for kind, _ in helper.calls)


def test_fresh_unreachable_helper_degrades_to_k1_without_any_launch(tmp_path):
    from test_generator_remote import EPOCH, FakeRemote
    key = tmp_path / "key"
    key.write_text("fake")
    config = SSHConfig("primary", 22, key, tmp_path / "work")
    primary, helper = FakeRemote(config), FakeRemote(config)
    remote = ScorerRemote(config, runner=primary, helpers=[helper])
    job_path = remote.work_dir / "jobs" / EPOCH / "job.json"
    job_path.parent.mkdir(parents=True)
    job_path.write_text(json.dumps(transport_job().to_dict()))
    helper.copy_error = InfraError("unreachable")
    assert remote.dispatch(EPOCH).state == "dispatched"
    launches = [command for _, command in primary.calls + helper.calls if "nohup" in command]
    assert len(launches) == 1 and "--shard" not in launches[0]
    assert sum("nohup" in command for _, command in helper.calls) == 0
    assert remote.dispatch(EPOCH).state == "dispatched"
    assert [command for _, command in primary.calls + helper.calls if "nohup" in command] == launches
    state = json.loads((remote.work_dir / "jobs" / EPOCH / "remote-state.json").read_text())
    assert state["shard_count"] == 1 and state["helper_identities"] == [] and state["launch_attempted"]


def test_dead_helper_publishes_owned_shard_as_time_budget(tmp_path, monkeypatch):
    from test_generator_remote import FakeRemote
    epoch = "20261120T0000Z"
    current = ScoringJob(epoch, (TaskSpec("0" * 20, "owner-0"), TaskSpec("1" * 20, "owner-1")), "a" * 64, "secret", epoch_deadline(epoch).timestamp() + 13500, POOL_ID)
    key = tmp_path / "key"
    key.write_text("fake")
    config = SSHConfig("primary", 22, key, tmp_path / "work")
    primary, helper = FakeRemote(config), FakeRemote(config)
    remote = ScorerRemote(config, runner=primary, helpers=[helper])
    job_path = remote.work_dir / "jobs" / epoch / "job.json"
    job_path.parent.mkdir(parents=True)
    job_path.write_text(json.dumps(current.to_dict()))
    now = current.budget_deadline + 2701
    monkeypatch.setattr("sn120_v2.generator.remote.time.time", lambda: now)
    remote.dispatch(epoch)
    primary.markers["done.json"] = '{"code":0}'
    document = {
        "epoch": epoch, "head_version": current.head_version, "seed_commitment": "a" * 64, "pool_pin": current.pool_pin,
        "shard": {"index": 0, "count": 2, "tasks": 1},
        "tasks": [{"id": "0" * 20, "hotkey": "owner-0", "s": 4, "n": 8, "q": 0.5, "decision": "accept"}],
        "timings": {"wall_s": 12}, "throughput": {"rollouts": 8, "servers": 8}, "servers": [], "hotkeys": {},
    }
    from test_generator_remote import archive
    primary.archive = archive({"./results.json": json.dumps(document).encode(), f"./rollouts/{'0' * 20}.jsonl": b"{}\n"})
    status = remote.collect(epoch)
    assert status.complete
    assert not any(kind == "tar" for kind, _ in helper.calls)
    results = json.loads((status.out_dir / "results.json").read_text())
    assert results["tasks"][1] == {"id": "1" * 20, "hotkey": "owner-1", "s": 0, "n": 0, "q": None, "decision": "unscored", "unscored_reason": "time_budget", "reason": "helper did not complete before the scoring budget grace"}
    assert results["hotkeys"]["owner-1"]["submitted"] == 1
    assert results["hotkeys"]["owner-1"]["unscored"] == 1
    assert results["hotkeys"]["owner-1"]["scored"] == 0
    assert not (status.out_dir / "rollouts" / f"{'1' * 20}.jsonl").exists()
    before = list(primary.calls), list(helper.calls)
    assert remote.collect(epoch).complete
    assert (primary.calls, helper.calls) == before


def test_live_k1_state_is_unchanged_and_zero_launch_calls(tmp_path):
    from test_generator_remote import EPOCH, FakeRemote, public_archive
    key = tmp_path / "key"
    key.write_text("fake")
    config = SSHConfig("primary", 22, key, tmp_path / "work")
    epoch = EPOCH
    job_path = tmp_path / "work" / "jobs" / epoch / "job.json"
    job_path.parent.mkdir(parents=True)
    job_path.write_text(json.dumps(transport_job().to_dict()))
    state_path = tmp_path / "work" / "jobs" / epoch / "remote-state.json"
    state_path.write_text(json.dumps({
        "epoch": epoch, "state": "running", "launch_attempted": True,
        "job_sha256": __import__("sn120_v2.canonical", fromlist=["sha256_hex"]).sha256_hex(job_path.read_bytes()), "attempts": 1,
    }))
    before_state = state_path.read_bytes()
    primary = FakeRemote(config)
    remote = ScorerRemote(config, runner=primary)
    assert remote.dispatch(epoch).state == "running"
    assert state_path.read_bytes() == before_state
    assert primary.calls == [] and primary.copies == []
    primary.markers["done.json"] = '{"code":0}'
    primary.archive = public_archive()
    assert remote.collect(epoch).complete
    assert [kind for kind, _ in primary.calls] == ["ssh", "ssh", "tar"]


def test_merged_helper_timeout_passes_real_publication_privacy_verifier(tmp_path):
    from sn120_v2.generator.decisions import decision
    from sn120_v2.generator.epoch import epoch_start
    from sn120_v2.generator.publish import publication
    from test_generator_publish import Pool
    epoch = "20261120T0000Z"
    secret = "private merge test seed"
    commitment = __import__("sn120_v2.canonical", fromlist=["sha256_hex"]).sha256_hex(secret.encode())
    task_ids = ["0" * 20, "1" * 20]
    current = ScoringJob(epoch, tuple(TaskSpec(task_id, f"owner-{index}") for index, task_id in enumerate(task_ids)), "a" * 64, secret, epoch_deadline(epoch).timestamp() + 13500, POOL_ID)
    key = tmp_path / "key"
    key.write_text("fake")
    remote = ScorerRemote(SSHConfig("primary", 22, key, tmp_path / "work"))
    job_dir = remote.work_dir / "jobs" / epoch
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(json.dumps(current.to_dict()))
    (job_dir / "remote-state.json").write_text(json.dumps({"shard_count": 2, "results_grace_s": 2700}))
    primary = remote.work_dir / "epochs" / epoch / "out-shard-0"
    (primary / "rollouts").mkdir(parents=True)
    (primary / "results.json").write_text(json.dumps({
        "epoch": epoch, "head_version": current.head_version, "seed_commitment": commitment, "pool_pin": current.pool_pin,
        "shard": {"index": 0, "count": 2, "tasks": 1},
        "tasks": [{"id": task_ids[0], "hotkey": "owner-0", "s": 4, "n": 8, "q": decision(4, 8).q, "decision": "accept"}],
        "timings": {}, "throughput": {}, "servers": [], "hotkeys": {},
    }))
    (primary / "rollouts" / f"{task_ids[0]}.jsonl").write_text(json.dumps({"text": "accepted full text", "seed": 0, "pass": True}) + "\n")
    start = epoch_start(epoch).timestamp()
    evidence = {
        "epoch": epoch, "pool": POOL_ID, "king": current.head_version, "seed_commitment": commitment,
        "tasks": [{"task_id": task_id, "hotkey": f"owner-{index}", "uid": index + 1, "uploaded_at": start + index + 1, "file_sha256": __import__("sn120_v2.canonical", fromlist=["sha256_hex"]).sha256_hex(task_id.encode()), "head_version": current.head_version} for index, task_id in enumerate(task_ids)],
    }
    merged = remote._merge_shards(epoch, primary, [])
    plan = publication(evidence, json.loads((merged / "results.json").read_text()), out_dir=merged, secret=secret, history=[], registered={1: "owner-0", 2: "owner-1"}, now=epoch_deadline(epoch).timestamp() + 60, pool_rows=Pool(), frozen_at=epoch_deadline(epoch).timestamp())
    accepted, unscored = plan["results"]["tasks"]
    assert accepted["task_id"] == task_ids[0]
    assert set(unscored) == {"task_sha256", "decision", "n", "s", "unscored_reason"}
    assert unscored["decision"] == "unscored" and unscored["n"] == 0
    assert set(plan["rollouts"]) == {task_ids[0]}
    assert task_ids[1].encode() not in json.dumps(plan["results"]).encode()


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
        config = type("Config", (), {"host": "helper", "port": 23})()
        def ssh(self, command, *, timeout=None):
            calls.append(command)
            return type("Reply", (), {"stdout": ""})()
        def scp(self, source, destination, *, timeout=None):
            calls.append("copy")
    remote.helpers = (Runner(),)
    epoch = "20261120T0000Z"
    job_path = remote.work_dir / "jobs" / epoch / "job.json"
    job_path.parent.mkdir(parents=True)
    job_path.write_text(json.dumps(transport_job().to_dict()))
    state = remote.work_dir / "jobs" / epoch / "remote-state.json"
    records = remote._dispatch_helpers(epoch, job_path, state, "a" * 64, {"shard_count": 2, "helper_identities": [{"host": "helper", "port": 23}]})
    before = list(calls)
    records = remote._dispatch_helpers(epoch, job_path, state, "a" * 64, {"shard_count": 2, "helper_identities": [{"host": "helper", "port": 23}], "launch_attempted": True, "helpers": records})
    assert records[0]["launch_attempted"]
    assert calls == before
    assert any("--shard 1/2" in command for command in before)
    assert any("ensure_head" in command for command in before)
