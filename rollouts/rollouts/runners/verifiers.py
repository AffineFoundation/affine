"""Verifiers-v1 runner: one batch through `uv run eval` (local docker),
traces parsed into envelopes. Port of the prime-lane batch path with the
policy matrix wired in: harness and endpoint chain come from the Policy,
not hardcoded config.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from datagen.providers import looks_like_provider_failure

from rollouts.adapters.verifiers import envelopes_from_traces
from rollouts.catalog import VERIFIERS_IMAGE_PREFIXES
from rollouts.config import RolloutsConfig
from rollouts.registry import Source
from rollouts.runners.base import (
    BatchResult,
    EndpointHealth,
    prune_images,
    run_streamed,
)
from rollouts.schema import (
    Endpoint,
    Policy,
    PolicyStamp,
    trace_error_type,
    trace_reward_score,
    trace_stats,
)

log = logging.getLogger("rollouts.runners.verifiers")


def eval_cmd(cfg: RolloutsConfig, source: Source, endpoint: Endpoint,
             harness: str, batch: list[dict], run_dir: Path) -> list[str]:
    uids = [r["uid"] for r in batch]
    cmd = [
        "nice", "-n", "10", "uv", "run", "eval", source.taskset_id,
        "-n", str(len(uids)),
        "-m", endpoint.model,
        "--client.base-url", endpoint.base_url,
        "--client.api-key-var", endpoint.key_env,
        "--env.agent.harness.id", harness,
        "--env.agent.runtime.type", "docker",
        "--env.agent.max-turns", str(cfg.max_turns),
        "--env.agent.timeout.setup", "1800",
        "--env.agent.timeout.rollout", str(cfg.rollout_timeout_s),
        "--env.agent.timeout.scoring", "1800",
        "--push", "False", "--rich", "False",
        "-c", str(min(cfg.max_containers, len(uids))),
        "-o", str(run_dir),
    ]
    if source.select == "tasks":
        # Harbor filters on the task directory basename where flagged;
        # traces still key on the full TaskData.name.
        task_ids = ([u.rsplit("/", 1)[-1] for u in uids]
                    if source.task_id_basename else uids)
        cmd.extend(["--env.taskset.tasks", *task_ids])
    else:
        # Rows may carry a raw instance_id distinct from the uid (swesmith
        # prefixes uids with the language shard key); the HF filter matches
        # the raw id.
        filter_ids = [r.get("instance_id") or r["uid"] for r in batch]
        id_set = "{" + ",".join(repr(u) for u in filter_ids) + "}"
        filter_expr = f"lambda row: row[{source.uid_field!r}] in {id_set}"
        if source.pass_dataset:
            cmd.extend([
                "--env.taskset.dataset-name", source.dataset,
                "--env.taskset.split", source.split,
            ])
        elif source.split:
            cmd.extend(["--env.taskset.split", source.split])
        cmd.extend(["--env.taskset.filter-fn", filter_expr])
    cmd.extend(source.extra_flags)
    return cmd


def build_local_images(batch: list[dict]) -> tuple[list[dict], list[str]]:
    """Local-build per-task images (terminal_lego); (ok_batch, failed_uids)."""
    ok: list[dict] = []
    failed: list[str] = []
    for row in batch:
        uid = row["uid"]
        image = row.get("image") or ""
        task_dir = Path(row.get("task_dir") or "")
        dockerfile = task_dir / "environment" / "Dockerfile"
        if not image or not dockerfile.is_file():
            log.warning("%s: missing image/Dockerfile", uid)
            failed.append(uid)
            continue
        probe = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, timeout=60)
        if probe.returncode == 0:
            ok.append(row)
            continue
        log.info("docker build %s <- %s", image, dockerfile)
        proc = subprocess.run(
            ["docker", "build", "-t", image, "-f", str(dockerfile),
             str(dockerfile.parent)],
            capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            tail = (proc.stdout or "")[-500:] + (proc.stderr or "")[-500:]
            log.error("docker build failed for %s: %s", uid, tail)
            failed.append(uid)
            continue
        ok.append(row)
    return ok, failed


def reap_containers() -> None:
    """Remove leftover containers from verifiers image namespaces only —
    mini_swe's swerebench/sweb.eval containers never match."""
    try:
        out = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.ID}} {{.Image}}"],
            capture_output=True, text=True, timeout=60).stdout
        stale = [line.split()[0] for line in out.splitlines()
                 if len(line.split()) == 2
                 and line.split()[1].startswith(VERIFIERS_IMAGE_PREFIXES)]
        if stale:
            subprocess.run(["docker", "rm", "-f", *stale],
                           capture_output=True, timeout=120)
            log.info("reaped %d leftover container(s)", len(stale))
    except Exception:
        log.warning("container reap failed", exc_info=True)


def _per_task_rows(envelopes: list[dict]) -> list[dict]:
    rows = []
    for env in envelopes:
        trace = env["trace"]
        score = trace_reward_score(trace)
        rows.append({
            "uid": env["task"]["uid"],
            "resolved": score,
            "stop": trace.get("stop_condition"),
            "error": trace_error_type(trace),
            **trace_stats(trace),
        })
    return rows


def _batch_suspect(per_task: list[dict], produced_traces: bool) -> bool:
    """A batch with no traces, or where nothing resolved and most rollouts
    errored with provider-failure signatures, is retried on the fallback
    endpoint."""
    if not produced_traces or not per_task:
        return True
    errored = [r for r in per_task if r.get("error")]
    if any(r["resolved"] == 1.0 for r in per_task):
        return False
    if len(errored) < max(1, len(per_task) // 2):
        return False
    blob = " ".join(str(r.get("error") or "") + str(r.get("stop") or "")
                    for r in errored)
    return looks_like_provider_failure(blob) or len(errored) == len(per_task)


class VerifiersRunner:
    def __init__(self, cfg: RolloutsConfig, health: EndpointHealth,
                 env: dict):
        self.cfg = cfg
        self.health = health
        self.env = env

    def run_batch(self, source: Source, policy: Policy, batch: list[dict],
                  run_dir: Path) -> BatchResult:
        result = BatchResult()
        reap_containers()

        if source.local_docker_build:
            batch, build_failed = build_local_images(batch)
            for uid in build_failed:
                result.per_task.append({
                    "uid": uid, "resolved": None, "stop": None,
                    "error": "docker_build_failed"})
            if not batch:
                log.error("all local docker builds failed; skipping batch")
                return result

        meta_by_uid = {r["uid"]: r for r in batch}
        endpoints = self.health.ordered(policy, self.env)
        for attempt, endpoint in enumerate(endpoints):
            attempt_dir = run_dir / endpoint.name
            attempt_dir.mkdir(parents=True, exist_ok=True)
            env = dict(self.env)
            env["PATH"] = f"{Path.home()}/.local/bin:" + env.get("PATH", "")
            code, out = run_streamed(
                eval_cmd(self.cfg, source, endpoint, policy.harness, batch,
                         attempt_dir),
                env, self.cfg.batch_timeout_s, cwd=self.cfg.verifiers_dir)
            if code != 0:
                log.error("eval exited %s; tail:\n%s", code, out[-2000:])
            stamp = PolicyStamp(policy_id=policy.id, model=endpoint.label,
                                harness=policy.harness,
                                endpoint=endpoint.name)
            traces_path = attempt_dir / "traces.jsonl"
            envelopes, _ = envelopes_from_traces(
                traces_path, source=source.name, env_id=source.taskset_id,
                meta_by_uid=meta_by_uid, policy=stamp)
            per_task = _per_task_rows(envelopes)
            suspect = code != 0 or _batch_suspect(per_task,
                                                  traces_path.exists())
            log.info("batch via %s: exit=%s tasks=%d resolved=%d%s",
                     endpoint.name, code, len(per_task),
                     sum(1 for r in per_task if r["resolved"] == 1.0),
                     " [provider-suspect]" if suspect else "")
            if suspect:
                self.health.strike(endpoint.name, f"eval exit {code}")
            else:
                self.health.mark_ok(endpoint.name)
            if not suspect or attempt == len(endpoints) - 1:
                result.envelopes = envelopes
                result.per_task.extend(per_task)
                result.endpoint = endpoint
                result.produced_output = traces_path.exists()
                break
            log.warning("retrying batch on fallback endpoint")

        if self.cfg.prune_images:
            prune_images([r.get("image") or "" for r in batch])
        return result
