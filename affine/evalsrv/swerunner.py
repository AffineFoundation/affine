"""Advisory SWE-rebench lite suite against a served miner model.

Pins a small fixed instance list (evalsrv/data/swe_rebench_lite_ids.json),
runs mini-SWE-agent against the local OpenAI-compatible vLLM endpoint, then
scores patches with the SWE-rebench fork harness. Advisory only — never S*.
"""

from __future__ import annotations

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

log = logging.getLogger("evalsrv.swe")

SUITE_NAME = "swe_rebench_lite"
IDS_PATH = Path(__file__).resolve().parent / "data" / "swe_rebench_lite_ids.json"
BENCH_DATA_DIR = Path(os.environ.get("AFFINE_BENCH_DIR", "/root/bench"))


def _load_pin() -> dict:
    return json.loads(IDS_PATH.read_text())


def _prepare_subset(pin: dict, out_dir: Path) -> Path:
    """Materialize the pinned instances as a local HF dataset directory."""
    from datasets import Dataset, DatasetDict, load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".ready"
    want = list(pin["instance_ids"])
    if marker.exists():
        try:
            got = json.loads(marker.read_text()).get("instance_ids", [])
            if got == want:
                return out_dir
        except Exception:
            pass

    ds = load_dataset(pin["dataset"], split=pin.get("split", "test"))
    want_set = set(want)
    rows = [r for r in ds if r["instance_id"] in want_set]
    found = {r["instance_id"] for r in rows}
    missing = [i for i in want if i not in found]
    if missing:
        raise RuntimeError(f"pinned instances missing from dataset: {missing[:5]}")
    # Preserve pin order; wrap as DatasetDict so mini-extra --split test works.
    by_id = {r["instance_id"]: r for r in rows}
    ordered = [by_id[i] for i in want]
    DatasetDict({"test": Dataset.from_list(ordered)}).save_to_disk(str(out_dir))
    marker.write_text(json.dumps({"instance_ids": want}))
    return out_dir


def _write_agent_config(model_repo: str, model_port: int, path: Path) -> None:
    """Derive the agent config from mini-swe-agent's packaged swebench config.

    The backticks variant is required: miner models are served by plain vLLM
    chat completions (no tool-call parser), and its templates use the
    ``{{task}}`` variable the installed runner actually provides. A hand-rolled
    template previously referenced ``{{problem_statement}}``, which mini-swe-
    agent v2 never exposes — every task failed at step 0 with api_calls=0.
    Only the model section and limits are overridden here so the config tracks
    whatever mini-swe-agent version bootstrap installs.
    """
    spec = importlib.util.find_spec("minisweagent")
    if spec is None or not spec.origin:
        raise RuntimeError("mini-swe-agent is not installed")
    base = (Path(spec.origin).parent / "config" / "benchmarks"
            / "swebench_backticks.yaml")
    cfg = yaml.safe_load(base.read_text())
    cfg["agent"]["step_limit"] = 50
    cfg["agent"]["cost_limit"] = 0  # local vLLM is free; 0 disables the check
    model = cfg.setdefault("model", {})
    model["model_name"] = f"openai/{model_repo}"
    # v2's default LitellmModel sends native tool calls (tools=[BASH_TOOL]);
    # plain vLLM rejects those with 400 unless launched with a tool parser.
    # The backticks templates expect the text-action model class instead.
    model["model_class"] = "litellm_textbased"
    model["cost_tracking"] = "ignore_errors"
    kwargs = model.setdefault("model_kwargs", {})
    kwargs.update({
        "api_base": f"http://127.0.0.1:{model_port}/v1",
        "api_key": "local",
        "temperature": 0.0,
        # Bound retry storms: a model that rejects requests outright should
        # fail its 25 tasks in minutes, not ride litellm backoff into the
        # suite-level timeout with zero trajectories.
        "num_retries": 2,
    })
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _reap_stale_containers() -> None:
    """Remove leftover mini-swe-agent containers from aborted/crashed runs.

    Aborts SIGTERM the harness process group but orphan its docker containers,
    which keep holding CPU/RAM and skew the next run. Only containers named
    minisweagent-* are touched (never swebench eval containers mid-run: reaping
    happens before the agent phase starts, when none of ours should exist)."""
    try:
        subprocess.run(
            "docker ps -aq --filter name=minisweagent- | xargs -r docker rm -f",
            shell=True, capture_output=True, timeout=180)
    except Exception:
        log.warning("stale container reap failed", exc_info=True)


def _run(cmd: list[str], env: dict, timeout_s: int,
         abort_event: threading.Event | None) -> tuple[int, str]:
    log.info("swe: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    t0 = time.time()
    chunks: list[str] = []

    # Drain stdout continuously. Reading only after exit deadlocks the whole
    # suite: once the harness prints more than the pipe buffer (~64KB), its
    # write blocks *inside rich's console handler while holding the logging
    # lock*, every worker thread queues on that lock, and the run freezes
    # until the suite timeout (observed: king bench stalled 1h at 16/25
    # trajectories; af-k1 and Tok331102 rode the same deadlock to 7200s).
    def _drain() -> None:
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                chunks.append(line)
        except Exception:
            pass  # EOF/decode races on kill are fine — output is best-effort

    reader = threading.Thread(target=_drain, daemon=True, name="swe-drain")
    reader.start()

    def _kill():
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            log.warning("could not kill swe subprocess", exc_info=True)

    while True:
        if proc.poll() is not None:
            break
        if abort_event is not None and abort_event.is_set():
            _kill()
            return -1, "aborted"
        if time.time() - t0 > timeout_s:
            _kill()
            return -2, f"timeout after {timeout_s}s"
        time.sleep(2)
    reader.join(timeout=30)
    return proc.returncode, "".join(chunks)


def _parse_resolve_rate(run_id: str, n_instances: int) -> tuple[float | None, int]:
    """Find the swebench model-level report for run_id → (rate, n_resolved).

    run_evaluation writes `<model>.<run_id>.json` into its CWD; the files
    under logs/run_evaluation/<run_id>/ are per-instance reports that carry
    no `resolved_ids` key. Only a report that actually declares resolved
    counts may produce a rate — treating "key absent" as 0 resolved once
    turned a genuine 5/25 run into a published 0.0."""
    cands = list(Path.cwd().glob(f"*.{run_id}.json"))
    log_dir = Path("logs") / "run_evaluation" / run_id
    if log_dir.is_dir():
        cands += list(log_dir.rglob("*.json"))
    for path in sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if isinstance(data.get("resolved_ids"), list):
            n_ok = len(data["resolved_ids"])
            total = int(data.get("total_instances") or n_instances) or 1
            return n_ok / total, n_ok
        if "resolved_instances" in data and "total_instances" in data:
            n_ok = int(data["resolved_instances"])
            total = int(data["total_instances"]) or n_instances or 1
            return n_ok / total, n_ok
    return None, 0


def run_swe_lite(model_repo: str, model_port: int, *,
                 workers: int = 2, timeout_s: int = 14400,
                 abort_event: threading.Event | None = None) -> dict:
    """Run the pinned swe_rebench_lite suite. Returns a result dict."""
    pin = _load_pin()
    n = len(pin["instance_ids"])
    run_root = BENCH_DATA_DIR / f"swe-{int(time.time())}"
    run_root.mkdir(parents=True, exist_ok=True)
    subset_dir = BENCH_DATA_DIR / "swe_lite_dataset"
    agent_cfg = run_root / "agent.yaml"
    preds_dir = run_root / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    _reap_stale_containers()
    try:
        _prepare_subset(pin, subset_dir)
        _write_agent_config(model_repo, model_port, agent_cfg)
    except Exception as e:
        return {"ok": False, "suite": SUITE_NAME, "error": f"prepare: {e}"}

    env = dict(os.environ)
    env["MSWEA_COST_TRACKING"] = "ignore_errors"
    env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY") or "local"

    agent_cmd = [
        "mini-extra", "swebench",
        "--model", f"openai/{model_repo}",
        "--config", str(agent_cfg),
        "--subset", str(subset_dir),
        "--split", "test",
        "--workers", str(max(1, workers)),
        "--output", str(preds_dir),
        "--environment-class", "docker",
    ]
    code, out = _run(agent_cmd, env, timeout_s=timeout_s // 2,
                     abort_event=abort_event)
    if code == -1:
        return {"ok": False, "suite": SUITE_NAME, "aborted": True,
                "error": "aborted to free GPUs"}
    if code == -2:
        return {"ok": False, "suite": SUITE_NAME, "error": out}
    if code != 0:
        return {"ok": False, "suite": SUITE_NAME,
                "wall_time_s": round(time.time() - t0, 1),
                "error": (out or "")[-2000:] or f"agent exit {code}"}

    preds_json = preds_dir / "preds.json"
    if not preds_json.exists():
        # Newer mini may nest preds; take the newest preds.json.
        cands = sorted(preds_dir.rglob("preds.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            return {"ok": False, "suite": SUITE_NAME,
                    "wall_time_s": round(time.time() - t0, 1),
                    "error": "no preds.json from mini-swe-agent"}
        preds_json = cands[0]

    # Convert dict preds.json → jsonl for swebench harness when needed.
    preds_jsonl = run_root / "predictions.jsonl"
    try:
        raw = json.loads(preds_json.read_text())
        if isinstance(raw, dict):
            with open(preds_jsonl, "w") as f:
                for iid, row in raw.items():
                    if isinstance(row, dict):
                        rec = dict(row)
                        rec.setdefault("instance_id", iid)
                        f.write(json.dumps(rec) + "\n")
        elif isinstance(raw, list):
            with open(preds_jsonl, "w") as f:
                for row in raw:
                    f.write(json.dumps(row) + "\n")
        else:
            preds_jsonl = preds_json
    except Exception:
        preds_jsonl = preds_json

    run_id = f"affine-{int(time.time())}"
    eval_cmd = [
        "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", pin["dataset"],
        "--split", pin.get("split", "test"),
        "--predictions_path", str(preds_jsonl),
        "--instance_ids", *pin["instance_ids"],
        "--max_workers", str(max(1, min(workers, 8))),
        "--run_id", run_id,
        "--namespace", pin.get("namespace", "swerebench"),
        "--cache_level", "instance",
    ]
    remaining = max(300, timeout_s - int(time.time() - t0))
    code, out = _run(eval_cmd, env, timeout_s=remaining, abort_event=abort_event)
    if code == -1:
        return {"ok": False, "suite": SUITE_NAME, "aborted": True,
                "error": "aborted during evaluation"}
    if code != 0:
        return {"ok": False, "suite": SUITE_NAME,
                "wall_time_s": round(time.time() - t0, 1),
                "error": (out or "")[-2000:] or f"eval exit {code}",
                "preds_path": str(preds_json)}

    rate, n_ok = _parse_resolve_rate(run_id, n)
    if rate is None:
        return {"ok": False, "suite": SUITE_NAME,
                "wall_time_s": round(time.time() - t0, 1),
                "error": "evaluation finished but resolve rate not found",
                "preds_path": str(preds_json)}

    return {
        "ok": True,
        "suite": SUITE_NAME,
        "score": round(rate, 4),
        "n_resolved": n_ok,
        "n_instances": n,
        "wall_time_s": round(time.time() - t0, 1),
        "preds_path": str(preds_json),
        "run_id": run_id,
    }
