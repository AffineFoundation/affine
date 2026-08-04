"""Advisory SWE-rebench lite suite against a served miner model.

Pins a small fixed instance list (evalsrv/data/swe_rebench_lite_ids.json),
runs mini-SWE-agent against the local OpenAI-compatible vLLM endpoint, then
scores patches with the SWE-rebench fork harness. Advisory only — never S*.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("evalsrv.swe")

SUITE_NAME = "swe_rebench_lite"
IDS_PATH = Path(__file__).resolve().parent / "data" / "swe_rebench_lite_ids.json"
AGENT_CFG_TEMPLATE = Path(__file__).resolve().parent / "swebench_agent.yaml"
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
    raw = AGENT_CFG_TEMPLATE.read_text()
    raw = raw.replace("openai/PLACEHOLDER", f"openai/{model_repo}")
    raw = re.sub(
        r'api_base:\s*"[^"]*"',
        f'api_base: "http://127.0.0.1:{model_port}/v1"',
        raw,
        count=1,
    )
    path.write_text(raw)


def _run(cmd: list[str], env: dict, timeout_s: int,
         abort_event: threading.Event | None) -> tuple[int, str]:
    log.info("swe: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    t0 = time.time()
    chunks: list[str] = []

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
    out = proc.stdout.read() if proc.stdout else ""
    chunks.append(out)
    return proc.returncode, "".join(chunks)


def _parse_resolve_rate(report_dir: Path, n_instances: int) -> tuple[float | None, int]:
    """Find swebench report JSON and return (resolve_rate, n_resolved)."""
    reports = sorted(report_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime,
                     reverse=True)
    for path in reports:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        resolved = data.get("resolved_ids") or data.get("resolved") or []
        if isinstance(resolved, list):
            n_ok = len(resolved)
            return (n_ok / n_instances if n_instances else 0.0), n_ok
        if "resolved_instances" in data and "total_instances" in data:
            n_ok = int(data["resolved_instances"])
            total = int(data["total_instances"]) or n_instances
            return n_ok / total, n_ok
    # Fallback: preds.json presence without report → unknown score.
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
        "--max_workers", str(max(1, min(workers, 4))),
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

    rate, n_ok = _parse_resolve_rate(Path("logs") / "run_evaluation" / run_id, n)
    if rate is None:
        # Also search under run_root / cwd.
        rate, n_ok = _parse_resolve_rate(Path.cwd(), n)
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
