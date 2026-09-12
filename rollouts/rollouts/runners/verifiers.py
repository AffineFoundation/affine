"""Verifiers-v1 runner: one batch through `uv run eval` (local docker),
traces parsed into envelopes. Port of the prime-lane batch path with the
policy matrix wired in: harness and endpoint chain come from the Policy,
not hardcoded config.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from datagen.providers import looks_like_provider_failure

from rollouts import loopguard
from rollouts.adapters.verifiers import (
    NO_VISIBLE_REPLY_STOP,
    envelopes_from_traces,
    mark_no_visible_reply,
)
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

# The mini-swe-agent harnesses install `mini-swe-agent==2.4.6` +
# `litellm[proxy]` (unpinned) into every task container with a PEP 723 uv
# script. litellm >= 1.98.0 (2026-08-22) no longer imports on Python 3.10
# (`typing.NotRequired`) although it still declares `>= 3.10`, and on a
# 3.10 image (r2e_gym, others) uv picks the image's interpreter, so every
# such rollout died at start-up as "Unknown model class: litellm_textbased"
# (600 r2e_gym + ~130 other king_textbased rollouts, 2026-09-10/11).
# UV_PYTHON makes uv fetch a managed 3.12 for the script env instead
# (~+15 s per container). Harness env vars ride `--env.agent.harness.env.*`.
MINI_SWE_HARNESSES = ("mini_swe_agent", "mini_swe_textbased")
MINI_SWE_HARNESS_ENV = {"UV_PYTHON": "3.12"}
# Harnesses whose "agent completed" may hide a final reply with no visible
# text (adapters.verifiers.mark_no_visible_reply). pi ends the agent when its
# last tool completes even if the model then says nothing.
NO_VISIBLE_REPLY_HARNESSES = ("pi",)


def eval_cmd(cfg: RolloutsConfig, source: Source, endpoint: Endpoint,
             harness: str, batch: list[dict], run_dir: Path,
             sampling: dict | None = None, runtime: str = "docker",
             ) -> list[str]:
    uids = [r["uid"] for r in batch]
    sampling = sampling or {}
    cmd = [
        "nice", "-n", "10", "uv", "run", "eval", source.taskset_id,
        "-n", str(len(uids)),
        "-m", endpoint.model,
        "--client.base-url", endpoint.base_url,
        "--client.api-key-var", endpoint.key_env,
        "--env.agent.harness.id", harness,
    ]
    if harness in MINI_SWE_HARNESSES:
        for key, value in MINI_SWE_HARNESS_ENV.items():
            cmd.extend([f"--env.agent.harness.env.{key}", value])
    cmd += [
        "--env.agent.runtime.type", runtime,
        "--env.agent.max-turns", str(cfg.max_turns),
        "--env.agent.timeout.setup", "1800",
        "--env.agent.timeout.rollout", str(cfg.rollout_timeout_s),
        "--env.agent.timeout.scoring", "1800",
        "--push", "False", "--rich", "False",
        "-c", str(min(cfg.max_containers, source.max_concurrency or cfg.max_containers,
                      len(uids))),
        "-o", str(run_dir),
    ]
    # Policy sampling rides the v1 eval CLI's dotted SamplingConfig; unset
    # keys keep the eval defaults, which every pre-policy batch ran with.
    if "temperature" in sampling:
        cmd.extend(["--sampling.temperature", str(sampling["temperature"])])
    if "max_tokens" in sampling:
        cmd.extend(["--sampling.max-tokens", str(sampling["max_tokens"])])
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


# Container ownership (2026-09-12). Every container the eval subprocess
# creates is stamped by the `dockerwrap/docker` shim with
# `rollouts.supervisor=<pid>@<boot id>` and `rollouts.batch=<run tag>`. The
# per-batch reaper removes only containers whose supervisor is this process
# or is gone (crashed supervisor -> orphans); containers of another live
# supervisor or without the label (a one-off `uv run eval`, another agent's
# experiment) are left alone. Before this, the reaper removed every
# container in the verifiers image namespaces — two workers' one-offs on a
# pod and the live supervisor were killing each other's rollouts (exit 137).
DOCKERWRAP_DIR = str(Path(__file__).resolve().parent.parent / "dockerwrap")
SUPERVISOR_LABEL = "rollouts.supervisor"
BATCH_LABEL = "rollouts.batch"


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()[:8]
    except OSError:
        return "noboot"


def supervisor_id(pid: int | None = None) -> str:
    """`<pid>@<boot id>`: a pid alone could be reused after a reboot."""
    return f"{pid if pid is not None else os.getpid()}@{_boot_id()}"


def supervisor_alive(label: str) -> bool:
    pid_s, _, boot = label.partition("@")
    if boot != _boot_id():
        return False
    try:
        os.kill(int(pid_s), 0)
    except (ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def reap_containers(owner: str | None = None) -> None:
    """Remove this supervisor's leftover containers and any orphan whose
    supervisor no longer runs. Unlabeled and other live supervisors'
    containers are untouched."""
    owner = owner or supervisor_id()
    try:
        out = subprocess.run(
            ["docker", "ps", "-a", "--format",
             '{{.ID}}\t{{.Label "' + SUPERVISOR_LABEL + '"}}'],
            capture_output=True, text=True, timeout=60).stdout
        stale = []
        for line in out.splitlines():
            cid, _, label = line.partition("\t")
            label = label.strip()
            if label and (label == owner or not supervisor_alive(label)):
                stale.append(cid)
        if stale:
            subprocess.run(["docker", "rm", "-f", *stale],
                           capture_output=True, timeout=120)
            log.info("reaped %d leftover container(s) owned by this or a "
                     "dead supervisor", len(stale))
    except Exception:
        log.warning("container reap failed", exc_info=True)


def reap_all_verifiers_containers() -> None:
    """The pre-2026-09-12 reaper: every container in the verifiers image
    namespaces, whoever created it (mini_swe's swerebench/sweb.eval
    containers never match). Explicit `rollouts.run --reap-all` only — for
    a pod start where unlabeled orphans must be cleared; never per batch."""
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
            log.info("reaped %d verifiers container(s) (--reap-all)", len(stale))
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
    # Where the agent's harness process lives. Shell agents need a per-task
    # container; the chat runner below runs harness `null` in a subprocess.
    RUNTIME = "docker"

    def __init__(self, cfg: RolloutsConfig, health: EndpointHealth,
                 env: dict):
        self.cfg = cfg
        self.health = health
        self.env = env

    def run_batch(self, source: Source, policy: Policy, batch: list[dict],
                  run_dir: Path) -> BatchResult:
        result = BatchResult()
        if self.RUNTIME == "docker":
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
            if self.RUNTIME == "docker":
                # dockerwrap/docker stamps ownership labels on every
                # container this eval creates (see reap_containers).
                env["PATH"] = DOCKERWRAP_DIR + ":" + env["PATH"]
                env["ROLLOUTS_SUPERVISOR"] = supervisor_id()
                env["ROLLOUTS_BATCH"] = run_dir.name
            if policy.loop_guard_repeats > 0:
                # rollouts.loopguard: sitecustomize installs the `loop_guard`
                # @stop in the eval process; the threshold rides the env.
                env[loopguard.ENV_REPEATS] = str(policy.loop_guard_repeats)
                env["PYTHONPATH"] = loopguard.SITE_DIR + (
                    ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            code, out = run_streamed(
                eval_cmd(self.cfg, source, endpoint, policy.harness, batch,
                         attempt_dir, policy.sampling, runtime=self.RUNTIME),
                env, self.cfg.batch_timeout_s, cwd=self.cfg.verifiers_dir)
            if code != 0:
                log.error("eval exited %s; tail:\n%s", code, out[-2000:])
            stamp = PolicyStamp(policy_id=policy.id, model=endpoint.label,
                                harness=policy.harness,
                                endpoint=endpoint.name,
                                action_kind=policy.action_kind)
            traces_path = attempt_dir / "traces.jsonl"
            envelopes, _ = envelopes_from_traces(
                traces_path, source=source.name, env_id=source.taskset_id,
                meta_by_uid=meta_by_uid, policy=stamp)
            if policy.harness in NO_VISIBLE_REPLY_HARNESSES:
                n_silent = sum(mark_no_visible_reply(e["trace"]) for e in envelopes)
                if n_silent:
                    log.info("%d rollout(s) finished without a visible reply "
                             "-> stop_condition=%s", n_silent, NO_VISIBLE_REPLY_STOP)
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

        if self.cfg.prune_images and self.RUNTIME == "docker":
            prune_images([r.get("image") or "" for r in batch])
        return result


class VerifiersChatRunner(VerifiersRunner):
    """Same eval CLI, no container: for tasksets whose agent is a plain chat
    loop (harness `null` — math answers, native tool calling over MCP).
    Task images, docker reaping and image pruning do not apply."""

    RUNTIME = "subprocess"
