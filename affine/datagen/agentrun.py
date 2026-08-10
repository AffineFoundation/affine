"""mini-swe-agent rollouts + swebench harness evaluation for one batch.

Adapted from evalsrv/swerunner.py (deployed in production, deliberately left
untouched): same continuous stdout drain, container reaping, and report
parsing — retargeted at a pool of API providers and per-batch subsets.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import yaml

from .config import DatagenConfig
from .providers import Provider, looks_like_provider_failure

log = logging.getLogger("datagen.agent")

# The corpus contract (affine/scripts/corpus_push.py) requires actions as
# closed ```bash blocks; the packaged mini v2 config fences actions as
# ```mswea_bash_command instead.
ACTION_REGEX = r"```bash\s*\n(.*?)\n```"

# Agent exit statuses that mean the instance was genuinely attempted (the
# rollout ran to a terminal state); anything else is an exception class name.
ATTEMPTED_EXITS = {"Submitted", "LimitsExceeded", "TimeExceeded",
                   "RepeatedFormatError"}


def write_agent_config(cfg: DatagenConfig, provider: Provider,
                       path: Path) -> None:
    """Derive the agent config from mini-swe-agent's packaged swebench config.

    The backticks variant is the text-action format our corpus uses (native
    tool calls are NOT wanted), and its templates use the ``{{task}}``
    variable the v2 runner actually provides. Every mswea_bash_command fence
    is rewritten to plain ``bash`` (with action_regex to match) so
    trajectories are generated natively in the corpus action format instead
    of being rewritten post-hoc.

    The provider's api_base goes into model_kwargs (LitellmModel passes
    model_kwargs straight through to litellm.completion); its API key does
    NOT — mini-swe-agent serializes the model config into every trajectory
    file, so the key is injected via the subprocess environment instead
    (Provider.subprocess_env).
    """
    spec = importlib.util.find_spec("minisweagent")
    if spec is None or not spec.origin:
        raise RuntimeError("mini-swe-agent is not installed")
    base = (Path(spec.origin).parent / "config" / "benchmarks"
            / "swebench_backticks.yaml")
    data = yaml.safe_load(base.read_text().replace("mswea_bash_command", "bash"))
    data["agent"]["step_limit"] = cfg.step_limit
    data["agent"]["cost_limit"] = cfg.cost_limit
    model = data.setdefault("model", {})
    model["model_name"] = provider.model
    model["model_class"] = "litellm_textbased"
    model["action_regex"] = ACTION_REGEX
    model["cost_tracking"] = "ignore_errors"
    kwargs = model.setdefault("model_kwargs", {})
    kwargs.update({
        "temperature": cfg.temperature,
        # Bound retry storms: a broken instance/endpoint should fail its
        # batch slot in minutes, not ride litellm backoff into the timeout.
        "num_retries": 2,
        "drop_params": True,
    })
    if provider.api_base:
        kwargs["api_base"] = provider.api_base
    env_cfg = data.setdefault("environment", {})
    # The packaged default (120s) killed every cold first-ever image pull on
    # the datagen box; prefetch usually makes pulls no-ops, this covers the
    # first batch and prefetch misses.
    env_cfg["pull_timeout"] = 600
    # Containers self-expire (`sleep <container_timeout>` + --rm). 2h default
    # was shorter than a slow lane's rollout; a mid-run expiry strands the
    # agent with exec errors.
    env_cfg["container_timeout"] = "4h"
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def classify_traj(traj_path: Path) -> tuple[str, str]:
    """Classify one instance's rollout: (kind, detail).

    kind: "attempted" (real terminal state, eligible for eval),
          "provider" (rate-limit/capacity/auth — reroute to another provider),
          "error" (instance-level failure, e.g. docker pull broke).
    """
    if not traj_path.exists():
        return "provider", "no trajectory file"
    try:
        info = json.loads(traj_path.read_text()).get("info", {})
    except Exception:
        return "error", "unreadable trajectory"
    status = str(info.get("exit_status") or "")
    if status in ATTEMPTED_EXITS:
        return "attempted", status
    if not status:
        # Trajectory snapshot of a run that never reached a terminal state —
        # the agent process was killed mid-rollout (group timeout/crash).
        return "provider", "incomplete trajectory"
    blob = " ".join(str(info.get(k) or "")
                    for k in ("exit_status", "exception_str", "traceback"))
    if looks_like_provider_failure(blob):
        return "provider", status
    return "error", status


def traj_usage(traj_path: Path) -> dict:
    """Sum token usage over a trajectory's assistant turns (litellm response
    usage is persisted in each message's extra). Basis for per-lane $/yield."""
    out = {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0,
           "n_msgs": 0}
    try:
        data = json.loads(traj_path.read_text())
    except Exception:
        return out
    msgs = data.get("messages", [])
    out["n_msgs"] = len(msgs)
    for m in msgs:
        usage = (m.get("extra", {}).get("response", {}) or {}).get("usage") or {}
        if usage:
            out["n_calls"] += 1
            out["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            out["completion_tokens"] += int(usage.get("completion_tokens") or 0)
    return out


def reap_stale_containers() -> None:
    """Remove leftover mini-swe-agent containers from aborted/crashed runs.

    Runs before the agent phase starts, when none of ours should exist —
    orphans keep holding CPU/RAM and skew the next batch."""
    try:
        subprocess.run(
            "docker ps -aq --filter name=minisweagent- | xargs -r docker rm -f",
            shell=True, capture_output=True, timeout=180)
    except Exception:
        log.warning("stale container reap failed", exc_info=True)


def row_image(row: dict) -> str:
    return str(row.get("docker_image") or row.get("image_name") or "")


def prune_images(images: list[str]) -> None:
    """Drop exactly the finished batch's images. Never prune by namespace
    glob: with pipelining, the next batch's images are prefetched while this
    batch evaluates, and a glob would delete them."""
    for image in images:
        if not image:
            continue
        try:
            subprocess.run(["docker", "rmi", "-f", image],
                           capture_output=True, timeout=120)
        except Exception:
            log.warning("prune of %s failed", image, exc_info=True)


def reap_containers_for_images(images: list[str]) -> None:
    """Kill containers spawned from the given instance images. Used after a
    provider group times out: killing the mini-extra process group leaves its
    docker containers running (observed live: a 2h-old orphan), and reaping
    ALL minisweagent-* here would murder the other groups mid-wave."""
    for image in images:
        if not image:
            continue
        try:
            subprocess.run(
                f"docker ps -aq --filter ancestor={image} | xargs -r docker rm -f",
                shell=True, capture_output=True, timeout=120)
        except Exception:
            log.warning("container reap for %s failed", image, exc_info=True)


def prefetch_images(rows: list[dict], workers: int = 3) -> None:
    """Pull the given instances' docker images (next-batch prefetch). Batch
    selection is deterministic, so the next batch is knowable in advance;
    pulling during the current batch's generation kills the cold-start
    timeout class entirely. Best-effort — a failed pull just means the agent
    phase pulls it itself under its own (generous) pull_timeout."""
    def pull(image: str) -> None:
        try:
            subprocess.run(["docker", "pull", "-q", image],
                           capture_output=True, timeout=900)
        except Exception:
            log.info("prefetch of %s failed (agent phase will retry)", image)

    images = [img for img in (row_image(r) for r in rows) if img]
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(pull, images))
    log.info("prefetched %d images in %.0fs", len(images), time.time() - t0)


def _run(cmd: list[str], env: dict, timeout_s: int, cwd: Path) -> tuple[int, str]:
    log.info("run: %s (cwd=%s)", " ".join(cmd), cwd)
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(cwd), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, start_new_session=True)
    t0 = time.time()
    chunks: list[str] = []

    # Drain stdout continuously. Reading only after exit deadlocks the run:
    # once the harness prints more than the pipe buffer (~64KB), its write
    # blocks inside rich's console handler while holding the logging lock,
    # every worker thread queues on that lock, and the run freezes until the
    # timeout (observed live on the bench pod — see evalsrv/swerunner.py).
    def _drain() -> None:
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                chunks.append(line)
        except Exception:
            pass  # EOF/decode races on kill are fine — output is best-effort

    reader = threading.Thread(target=_drain, daemon=True, name="datagen-drain")
    reader.start()

    while True:
        if proc.poll() is not None:
            break
        if time.time() - t0 > timeout_s:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                log.warning("could not kill subprocess", exc_info=True)
            return -2, f"timeout after {timeout_s}s"
        time.sleep(2)
    reader.join(timeout=30)
    return proc.returncode, "".join(chunks)


def agent_cmd(provider: Provider, subset_dir: Path, agent_cfg: Path,
              preds_dir: Path, n_instances: int) -> list[str]:
    return [
        "mini-extra", "swebench",
        "--model", provider.model,
        "--config", str(agent_cfg),
        "--subset", str(subset_dir),
        "--split", "test",
        "--workers", str(max(1, min(provider.concurrency, n_instances))),
        "--output", str(preds_dir),
        "--environment-class", "docker",
    ]


def eval_cmd(cfg: DatagenConfig, dataset: str, split: str, preds_jsonl: Path,
             instance_ids: list[str], run_id: str) -> list[str]:
    return [
        "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset,
        "--split", split,
        "--predictions_path", str(preds_jsonl),
        "--instance_ids", *instance_ids,
        "--max_workers", str(max(1, min(cfg.eval_workers, 24))),
        "--run_id", run_id,
        "--namespace", cfg.namespace,
        "--cache_level", "instance",
    ]


def run_agent_phase(cfg: DatagenConfig, provider: Provider, subset_dir: Path,
                    agent_cfg: Path, preds_dir: Path, run_dir: Path,
                    n_instances: int) -> tuple[int, str]:
    return _run(agent_cmd(provider, subset_dir, agent_cfg, preds_dir,
                          n_instances),
                provider.subprocess_env(), cfg.agent_timeout_s, cwd=run_dir)


def merge_predictions(preds_dirs: list[Path], out_path: Path) -> Path | None:
    """Merge per-provider preds.json files -> one jsonl the harness accepts.
    A non-empty patch beats an empty one (a failed first attempt writes an
    empty entry that a requeued run may have filled elsewhere)."""
    merged: dict[str, dict] = {}
    for preds_dir in preds_dirs:
        cands = ([preds_dir / "preds.json"]
                 if (preds_dir / "preds.json").exists()
                 else sorted(preds_dir.rglob("preds.json"),
                             key=lambda p: p.stat().st_mtime, reverse=True)[:1])
        for preds_json in cands:
            raw = json.loads(preds_json.read_text())
            rows = raw.values() if isinstance(raw, dict) else raw
            for row in rows:
                iid = row.get("instance_id")
                if not iid:
                    continue
                if iid not in merged or (row.get("model_patch")
                                         and not merged[iid].get("model_patch")):
                    merged[iid] = row
    if not merged:
        return None
    with open(out_path, "w", encoding="utf-8") as f:
        for row in merged.values():
            f.write(json.dumps(row) + "\n")
    return out_path


def run_eval_phase(cfg: DatagenConfig, dataset: str, split: str,
                   preds_jsonl: Path, instance_ids: list[str], run_id: str,
                   run_dir: Path) -> tuple[int, str]:
    env = dict(os.environ)
    env["MSWEA_COST_TRACKING"] = "ignore_errors"
    return _run(eval_cmd(cfg, dataset, split, preds_jsonl, instance_ids,
                         run_id),
                env, cfg.eval_timeout_s, cwd=run_dir)


def parse_resolved_ids(run_dir: Path, run_id: str) -> list[str] | None:
    """Find the model-level report for run_id and return its resolved_ids.

    run_evaluation writes `<model>.<run_id>.json` into its CWD; the files
    under logs/run_evaluation/<run_id>/ are per-instance reports without a
    resolved_ids key. None (report missing) is distinct from [] (nothing
    resolved) — only a report that declares resolved ids may be trusted."""
    cands = list(run_dir.glob(f"*.{run_id}.json"))
    log_dir = run_dir / "logs" / "run_evaluation" / run_id
    if log_dir.is_dir():
        cands += list(log_dir.rglob("*.json"))
    for path in sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("resolved_ids"), list):
            return [str(i) for i in data["resolved_ids"]]
    return None
