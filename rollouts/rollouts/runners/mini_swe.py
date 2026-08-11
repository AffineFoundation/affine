"""mini-swe-agent runner: one batch of swebench-format instances through
mini-extra rollouts, optional swebench-harness verdict as telemetry, and
trajectories adapted into trace envelopes.

Port of the legacy datagen loop's generation + eval phases, minus the
multi-provider wave sharding: the scheduler already picked one policy, so a
batch runs on the policy's first healthy endpoint and retries whole on the
next one when the endpoint (not the instances) failed.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import time
from pathlib import Path

import yaml

from datasets import load_dataset

from datagen.agentrun import (
    classify_traj,
    merge_predictions,
    parse_resolved_ids,
    reap_containers_for_images,
    reap_stale_containers,
)
from datagen.instances import materialize_subset

from rollouts.adapters.mini_swe import envelope_from_traj
from rollouts.config import RolloutsConfig
from rollouts.registry import Source
from rollouts.runners.base import (
    BatchResult,
    EndpointHealth,
    prune_images,
    run_streamed,
)
from rollouts.schema import Endpoint, Policy, PolicyStamp

log = logging.getLogger("rollouts.runners.mini_swe")

# The corpus contract requires actions as closed ```bash blocks; the
# packaged mini v2 config fences actions as ```mswea_bash_command instead.
ACTION_REGEX = r"```bash\s*\n(.*?)\n```"

_DATASET_CACHE: dict[tuple[str, str], dict[str, dict]] = {}


def _dataset_rows(dataset: str, split: str) -> dict[str, dict]:
    """instance_id -> full swebench row (disk-cached HF load, memoized)."""
    key = (dataset, split)
    if key not in _DATASET_CACHE:
        rows = load_dataset(dataset, split=split)
        _DATASET_CACHE[key] = {r["instance_id"]: dict(r) for r in rows}
    return _DATASET_CACHE[key]


def write_agent_config(cfg: RolloutsConfig, policy: Policy,
                       endpoint: Endpoint, path: Path) -> None:
    """Derive the agent config from mini-swe-agent's packaged swebench
    config (backticks variant — the text-action format the corpus uses).
    The API key is injected via the subprocess environment, never written
    here (mini-swe-agent serializes this config into every trajectory)."""
    spec = importlib.util.find_spec("minisweagent")
    if spec is None or not spec.origin:
        raise RuntimeError("mini-swe-agent is not installed")
    base = (Path(spec.origin).parent / "config" / "benchmarks"
            / "swebench_backticks.yaml")
    data = yaml.safe_load(
        base.read_text().replace("mswea_bash_command", "bash"))
    data["agent"]["step_limit"] = cfg.step_limit
    data["agent"]["cost_limit"] = cfg.cost_limit
    model = data.setdefault("model", {})
    model["model_name"] = endpoint.litellm
    model["model_class"] = "litellm_textbased"
    model["action_regex"] = ACTION_REGEX
    model["cost_tracking"] = "ignore_errors"
    kwargs = model.setdefault("model_kwargs", {})
    kwargs.update({
        "temperature": policy.sampling.get("temperature", 0.7),
        "num_retries": 2,
        "drop_params": True,
    })
    if not endpoint.litellm_model:
        # OpenAI-compatible route: point litellm at the endpoint.
        kwargs["api_base"] = endpoint.base_url
    env_cfg = data.setdefault("environment", {})
    env_cfg["pull_timeout"] = 600
    env_cfg["container_timeout"] = "4h"
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _subprocess_env(base_env: dict, endpoint: Endpoint) -> dict:
    env = dict(base_env)
    env["MSWEA_COST_TRACKING"] = "ignore_errors"
    key_var = ("OPENROUTER_API_KEY"
               if endpoint.litellm.startswith("openrouter/")
               else "OPENAI_API_KEY")
    env[key_var] = base_env[endpoint.key_env]
    return env


class MiniSweRunner:
    def __init__(self, cfg: RolloutsConfig, health: EndpointHealth,
                 env: dict):
        self.cfg = cfg
        self.health = health
        self.env = env

    # -- generation --------------------------------------------------------------

    def _agent_attempt(self, source: Source, policy: Policy,
                       endpoint: Endpoint, rows: list[dict],
                       gdir: Path) -> dict[str, tuple[str, str]]:
        """One endpoint's attempt at the whole batch; iid -> (kind, detail)
        from classify_traj."""
        subset_dir = materialize_subset(rows, gdir / "subset")
        agent_cfg = gdir / "agent.yaml"
        write_agent_config(self.cfg, policy, endpoint, agent_cfg)
        preds_dir = gdir / "preds"
        preds_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "mini-extra", "swebench",
            "--model", endpoint.litellm,
            "--config", str(agent_cfg),
            "--subset", str(subset_dir),
            "--split", "test",
            "--workers", str(max(1, min(self.cfg.max_containers, len(rows)))),
            "--output", str(preds_dir),
            "--environment-class", "docker",
        ]
        code, out = run_streamed(
            cmd, _subprocess_env(self.env, endpoint),
            self.cfg.batch_timeout_s, cwd=gdir)
        if code == -2:
            # Killing the process group orphans its containers; reap only
            # this batch's, by instance image. Finished trajectories still
            # classify normally (salvage) — only stragglers requeue.
            reap_containers_for_images(
                [r.get("docker_image") or r.get("image_name") or ""
                 for r in rows])
        elif code != 0:
            log.error("agent exit %s: %s", code, (out or "")[-1500:])
            return {r["instance_id"]: ("provider", f"agent exit {code}")
                    for r in rows}
        results = {}
        for r in rows:
            iid = r["instance_id"]
            results[iid] = classify_traj(
                preds_dir / iid / f"{iid}.traj.json")
        return results

    # -- swebench verdict telemetry ------------------------------------------------

    def _swe_eval(self, source: Source, eval_ids: list[str],
                  preds_jsonl: Path, run_dir: Path) -> set[str] | None:
        """Resolved ids, or None on persistent eval failure."""
        for attempt in range(2):
            run_id = f"rollouts-{int(time.time())}"
            cmd = [
                "python", "-m", "swebench.harness.run_evaluation",
                "--dataset_name", source.dataset,
                "--split", source.split,
                "--predictions_path", str(preds_jsonl),
                "--instance_ids", *eval_ids,
                "--max_workers", str(max(1, min(self.cfg.eval_workers, 24))),
                "--run_id", run_id,
                "--namespace", source.swe_namespace,
                "--cache_level", "instance",
            ]
            env = dict(self.env)
            env["MSWEA_COST_TRACKING"] = "ignore_errors"
            code, out = run_streamed(cmd, env, self.cfg.eval_timeout_s,
                                     cwd=run_dir)
            if code == 0:
                got = parse_resolved_ids(run_dir, run_id)
                if got is not None:
                    return set(got)
                log.error("eval finished but no resolved_ids report "
                          "(attempt %d)", attempt + 1)
            else:
                log.error("eval failed (exit %s, attempt %d): %s", code,
                          attempt + 1, (out or "")[-2000:])
            time.sleep(30)
        return None

    # -- batch --------------------------------------------------------------------

    def run_batch(self, source: Source, policy: Policy, batch: list[dict],
                  run_dir: Path) -> BatchResult:
        result = BatchResult()
        reap_stale_containers()
        all_rows = _dataset_rows(source.dataset, source.split)
        meta_by_uid = {r["uid"]: r for r in batch}
        rows = []
        for r in batch:
            full = all_rows.get(r["uid"])
            if full is None:
                result.per_task.append({
                    "uid": r["uid"], "resolved": None, "stop": None,
                    "error": "instance missing from dataset"})
            else:
                rows.append(full)
        if not rows:
            return result

        endpoints = self.health.ordered(policy, self.env)
        results: dict[str, tuple[str, str]] = {}
        used: tuple[Endpoint, Path] | None = None
        for attempt, endpoint in enumerate(endpoints):
            gdir = run_dir / endpoint.name
            gdir.mkdir(parents=True, exist_ok=True)
            results = self._agent_attempt(source, policy, endpoint, rows,
                                          gdir)
            n_attempted = sum(1 for kind, _ in results.values()
                              if kind == "attempted")
            provider_broken = all(kind == "provider"
                                  for kind, _ in results.values())
            log.info("batch via %s: %d/%d attempted%s", endpoint.name,
                     n_attempted, len(rows),
                     " [provider-suspect]" if provider_broken else "")
            if provider_broken:
                self.health.strike(endpoint.name,
                                   next(iter(results.values()))[1])
            else:
                self.health.mark_ok(endpoint.name)
            if not provider_broken or attempt == len(endpoints) - 1:
                used = (endpoint, gdir / "preds")
                break
            log.warning("retrying batch on fallback endpoint")

        endpoint, preds_dir = used
        attempted = {iid: detail for iid, (kind, detail) in results.items()
                     if kind == "attempted"}
        for iid, (kind, detail) in results.items():
            if kind == "error":
                result.per_task.append({
                    "uid": iid, "resolved": None, "stop": None,
                    "error": detail})
            # provider-kind on the final endpoint stays unmarked -> the
            # task is re-selected later (zero-yield cooldown bounds loops).

        # swebench verdict telemetry: only instances that submitted a patch
        # can resolve; a persistent eval failure downgrades the verdict to
        # unscored (None) rather than discarding the trajectories.
        resolved: set[str] | None = set()
        eval_ran = False
        if attempted and source.swe_eval:
            preds_jsonl = merge_predictions([preds_dir],
                                            run_dir / "predictions.jsonl")
            with_patch: set[str] = set()
            if preds_jsonl is not None:
                for line in open(preds_jsonl, encoding="utf-8"):
                    row = json.loads(line)
                    if row.get("model_patch"):
                        with_patch.add(row["instance_id"])
            eval_ids = sorted(set(attempted) & with_patch)
            if eval_ids:
                got = self._swe_eval(source, eval_ids, preds_jsonl, run_dir)
                if got is None:
                    resolved = None
                else:
                    resolved = got
                    eval_ran = True
            else:
                eval_ran = True   # nothing had a patch: all unresolved

        stamp = PolicyStamp(policy_id=policy.id, model=endpoint.label,
                            harness=policy.harness, endpoint=endpoint.name)
        for iid, detail in attempted.items():
            traj_path = preds_dir / iid / f"{iid}.traj.json"
            verdict: float | None = None
            if eval_ran and resolved is not None:
                verdict = 1.0 if iid in resolved else 0.0
            envelope = envelope_from_traj(
                traj_path, source=source.name, env_id=source.dataset,
                task=meta_by_uid[iid], policy=stamp, resolved=verdict)
            if envelope is None:
                result.per_task.append({
                    "uid": iid, "resolved": None, "stop": None,
                    "error": "unreadable trajectory"})
                continue
            result.envelopes.append(envelope)
            result.per_task.append({
                "uid": iid, "resolved": verdict, "stop": detail,
                "error": None})
        result.endpoint = endpoint
        result.produced_output = bool(attempted)

        if self.cfg.prune_images:
            prune_images([r.get("docker_image") or r.get("image_name") or ""
                          for r in rows])
        return result
