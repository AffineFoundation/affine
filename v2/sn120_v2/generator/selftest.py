"""CLI selftest: real HTTP, real math GradePool, deterministic fake model."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sn120_v2.envs.math_grader import GradePool

from .constants import POOL_ID
from .epoch import epoch_deadline
from .fake_vllm import FakeFarm, FakePool
from .runner import run_job
from .scoring import JOB_VERSION, MAX_SAMPLES, ScoringJob, TaskSpec, rollout_seed


def selftest(*, root: str | None = None, verbose: bool = False, first_port: int = 0) -> int:
    temporary = None if root else tempfile.mkdtemp(prefix="affine-generator-selftest-")
    directory = Path(root) if root else Path(temporary)
    if verbose:
        print(f"selftest directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    farm = FakeFarm(mode="mixed", first_port=first_port).start()
    grader = GradePool(workers=4, timeout_s=30.0)
    try:
        epoch = "20261120T0000Z"
        deadline = epoch_deadline(epoch).timestamp() + 3 * 3600 + 45 * 60
        for name, count in (("complete", 10), ("cutoff", 20)):
            job = ScoringJob(
                version=JOB_VERSION,
                epoch=epoch,
                tasks=tuple(TaskSpec(f"{index + (0 if name == 'complete' else 100):020x}", f"owner-{index % 3}") for index in range(count)),
                head_version="a" * 64,
                seed_secret="selftest-only-secret",
                budget_deadline=deadline,
                pool_pin=POOL_ID,
            )
            for task in job.tasks:
                for sample in range(MAX_SAMPLES):
                    farm.seeds[rollout_seed(job.seed_secret, task.task_id, sample)] = (task.task_id, sample)
            out = directory / name / "out"
            out.mkdir(parents=True, exist_ok=True)
            (out.parent / "job.json").write_text(json.dumps(job.to_dict()))
            loaded = ScoringJob.load(out.parent / "job.json")
            before = len(farm.requests)
            if name == "cutoff":
                clock = [deadline - 1.0]

                def now() -> float:
                    return clock[0]

                # Advance time only as requests complete; all durable responses
                # are graded, while everything under eight remains unscored.
                original = farm.record

                def advance(body: dict) -> None:
                    original(body)
                    clock[0] += 0.25

                farm.record = advance
                result = run_job(loaded, out, pool=FakePool(), grader=grader, urls=farm.urls, clock=now, clients_per_server=4)
                assert result["timings"]["budget_exhausted"]
                assert any(row["decision"] == "unscored" for row in result["tasks"])
            else:
                result = run_job(loaded, out, pool=FakePool(), grader=grader, urls=farm.urls, clients_per_server=4)
            assert (out / "DONE").is_file()
            assert (out / "results.json").is_file()
            assert result["seed_secret"] == loaded.seed_secret
            assert result["seed_commitment"]
            for row in result["tasks"]:
                rollouts = [json.loads(line) for line in (out / "rollouts" / f"{row['id']}.jsonl").read_text().splitlines()]
                assert len(rollouts) == row["n"]
                assert all(item["messages"][0]["content"] for item in rollouts)
                assert all(item["grade"]["code"] in ("P", "F", "C", "E") for item in rollouts)
                assert row["s"] == sum(item["grade"]["passed"] for item in rollouts)
            resumed = run_job(loaded, out, pool=FakePool(), grader=grader, urls=farm.urls, clients_per_server=4)
            assert resumed == result
            generated = len(farm.requests) - before
            print(f"selftest {name}: {sum(row['n'] for row in result['tasks'])} rollouts, {generated} requests, resume made no new requests")
        print("Generator scorer selftest passed: real GradePool, 8 loopback servers, finalization, cutoff and resume")
        return 0
    finally:
        grader.close()
        farm.close()
        if root is None:
            import shutil

            shutil.rmtree(directory)
