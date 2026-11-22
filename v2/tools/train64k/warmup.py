#!/usr/bin/env python3
"""Pre-tune long-row FLA/Triton kernels with the real trainer child.

One child runs per selected GPU with the same environment as production. Jobs
and adapters live only under ``<work-dir>/warmup`` and are deleted afterward;
nothing is uploaded and no chain or production storage is accessed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

# Invoke from the repository root with ``PYTHONPATH=v2``; the tool lives below
# v2/tools, not in the importable package.
V2 = Path(__file__).resolve().parents[2]

from sn120_v2.gpu.batch import training_rows
from sn120_v2.gpu.procs import Child
from sn120_v2.gpu.trainer import job_env
from sn120_v2.params import Params, load_params


def lengths(minimum: int = 32768, maximum: int = 69632, stride: int = 2048) -> list[int]:
    """Inclusive, ascending target output lengths."""
    if not 1 <= stride or not 1 <= minimum <= maximum:
        raise ValueError("lengths need 1 <= min <= max and a positive stride")
    return list(range(minimum, maximum + 1, stride))


def l2norm_class(total_tokens: int) -> int:
    """The q/k FLA l2norm NB class: NB = cdiv(T * 16, 65536)."""
    if total_tokens < 0:
        raise ValueError("a token count cannot be negative")
    return (total_tokens * 16 + 65535) // 65536


def repeated_text(tokenizer: Any, target_tokens: int) -> str:
    """Plain repeated text that tokenizes to exactly ``target_tokens`` ids."""
    if target_tokens < 1:
        raise ValueError("a warm-up output needs at least one token")
    words = ["warm"] * target_tokens
    text = " ".join(words)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != target_tokens:
        raise ValueError(f"repeated text has {len(ids)} tokens, expected {target_tokens}")
    return text


def warmup_rows(tokenizer: Any, target_lengths: Sequence[int], prompt_tokens: int | None = None) -> list[dict]:
    """Two mixed-reward rows per length, each with exactly the requested output tokens.

    With no explicit prompt count, use the production chat-template rendering.
    Tests may pass a small count to keep their synthetic tokenizer simple.
    """
    if not target_lengths or (prompt_tokens is not None and not 1 <= prompt_tokens < min(target_lengths)):
        raise ValueError("prompt_tokens must be positive and below every target length")
    groups = []
    for length in target_lengths:
        prompt = f"Warm up the long-row kernels for length {length}."
        if prompt_tokens is None:
            from sn120_v2.gpu.batch import render_prompt_ids
            n_prompt = len(render_prompt_ids(tokenizer, prompt))
        else:
            n_prompt = prompt_tokens
        if not 1 <= n_prompt < length:
            raise ValueError(f"prompt has {n_prompt} tokens, outside target length {length}")
        text = repeated_text(tokenizer, length - n_prompt)
        groups.append((prompt, ((text, 1.0), (text, 0.0))))
    rows = training_rows(groups)
    assert [row["advantage"] for row in rows] == [0.5, -0.5] * len(target_lengths)
    return rows


def compute_processes(gpu_report: str, process_report: str, selected: Sequence[int]) -> dict[int, list[str]]:
    """Parse index/UUID and compute-app reports, failing closed on bad input."""
    by_uuid = {}
    for line in gpu_report.splitlines():
        if line.strip():
            parts = [part.strip() for part in line.split(",", 1)]
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1]:
                raise ValueError(f"cannot read nvidia-smi GPU row: {line!r}")
            by_uuid[parts[1]] = int(parts[0])
    active: dict[int, list[str]] = {}
    for line in process_report.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4 or parts[0] not in by_uuid or not parts[1].isdigit():
            raise ValueError(f"cannot read nvidia-smi compute-process row: {line!r}")
        gpu = by_uuid[parts[0]]
        if gpu in selected:
            active.setdefault(gpu, []).append(",".join(parts[1:]))
    return active


def active_gpus(selected: Sequence[int]) -> dict[int, list[str]]:
    """Return compute processes on selected GPUs; any query failure refuses."""
    argv = [
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu_rows = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        report = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot prove the selected GPUs are idle: {exc}") from exc
    return compute_processes(gpu_rows, report, selected)


def memory_used_mib(gpu: int) -> float | None:
    """One driver memory reading, or None when nvidia-smi cannot supply it."""
    try:
        report = subprocess.run(
            ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        return float(report)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def make_job(base_dir: Path, rows: Sequence[dict], params: Params, job_dir: Path) -> None:
    job_dir.mkdir(parents=True)
    (job_dir / "lora").mkdir()
    job = {
        "base_dir": str(base_dir), "base_hash": "train64k-warmup-discard",
        "dtype": params.dtype, "recipe": params.recipe.to_dict(), "seed": 64000,
        "rows": list(rows),
    }
    (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")


def run_gpu(gpu: int, base_dir: Path, work_dir: Path, params: Params, target_lengths: Sequence[int]) -> dict:
    """Run one real trainer child per GPU, then remove its jobs and adapters."""
    import transformers
    from sn120_v2.gpu.train_job import encode

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    # Build one group at a time so peak temporary Python memory stays bounded.
    root = work_dir / "warmup"
    started = time.monotonic()
    classes: set[int] = set()
    forward_tokens = 0
    job_dir = root / f"gpu{gpu}"
    shutil.rmtree(job_dir, ignore_errors=True)
    try:
        rows = warmup_rows(tokenizer, target_lengths)
        recipe = params.recipe
        encoded, dropped = encode(tokenizer, rows, recipe)
        if dropped or len(encoded) != 2 * len(target_lengths):
            raise RuntimeError(f"warm-up encoding dropped {dropped} rows")
        make_job(base_dir, rows, params, job_dir)
        env = job_env([gpu], work_dir)
        if env["TRITON_CACHE_DIR"] != str(work_dir / "triton" / f"gpu{gpu}"):
            raise AssertionError("trainer job_env did not select the per-GPU Triton cache")
        env["PYTHONUNBUFFERED"] = "1"
        argv = [sys.executable, "-m", "sn120_v2.gpu.train_job", str(job_dir)]
        proc = Child(argv, env, job_dir / "job.log")
        deadline = time.monotonic() + params.lease_timeout_s
        baseline = memory_used_mib(gpu)
        peak = baseline
        try:
            code = None
            while code is None and time.monotonic() < deadline:
                code = proc.wait(1.0)
                reading = memory_used_mib(gpu)
                if reading is not None:
                    peak = reading if peak is None else max(peak, reading)
            if code is None:
                proc.stop()
                raise RuntimeError(f"GPU {gpu}: warm-up exceeded the {params.lease_timeout_s}s lease")
        except BaseException:
            if proc.alive():
                proc.stop()
            raise
        if code != 0:
            raise RuntimeError(f"GPU {gpu}: trainer child exited {code}; log: {job_dir / 'job.log'}")
        report = json.loads((job_dir / "report.json").read_text(encoding="utf-8"))
        if report.get("status") != "ok":
            raise RuntimeError(f"GPU {gpu}: {report}; log: {job_dir / 'job.log'}")
        for row in encoded:
            classes.add(l2norm_class(len(row["prompt_ids"]) + len(row["tokens"])))
        forward_tokens = int(report.get("forward_tokens", 0))
        return {
            "gpu": gpu, "wall_s": round(time.monotonic() - started, 1),
            "triton_cache_dir": env["TRITON_CACHE_DIR"], "lengths": list(target_lengths),
            "l2norm_classes": sorted(classes), "forward_tokens": forward_tokens,
            "peak_memory_mib": peak, "adapter": "discarded", "uploaded": False,
        }
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def visible_gpus(environ: Mapping[str, str] = os.environ) -> list[int]:
    text = environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if text:
        try:
            return [int(part.strip()) for part in text.split(",")]
        except ValueError as exc:
            raise ValueError("CUDA_VISIBLE_DEVICES must contain GPU indexes") from exc
    try:
        report = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        return [int(line.strip()) for line in report.splitlines() if line.strip()]
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot list visible GPUs: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True, help="resident head directory used by the box")
    parser.add_argument("--work-dir", type=Path, required=True, help="the box work_dir")
    parser.add_argument("--gpus", default="", help="comma-separated GPU indexes; default: all visible GPUs")
    parser.add_argument("--min", type=int, default=32768)
    parser.add_argument("--max", type=int, default=69632)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--params", type=Path, default=V2 / "ops" / "train64k_params.json")
    args = parser.parse_args(argv)
    target_lengths = lengths(args.min, args.max, args.stride)
    params = load_params(args.params)
    selected = [int(part.strip()) for part in args.gpus.split(",") if part.strip()] if args.gpus else visible_gpus()
    if not selected or len(set(selected)) != len(selected) or min(selected) < 0:
        parser.error("--gpus must contain distinct nonnegative GPU indexes")
    conflicts = active_gpus(selected)
    if conflicts:
        for gpu, processes in sorted(conflicts.items()):
            print(f"REFUSE: GPU {gpu} has active compute process(es): {processes}", file=sys.stderr, flush=True)
        return 2
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "warmup_start", "gpus": selected, "lengths": target_lengths,
                      "params_hash": params.params_hash, "adapter": "discard", "upload": False}, sort_keys=True), flush=True)
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as pool:
        jobs = {pool.submit(run_gpu, gpu, args.base_dir.resolve(), work_dir, params, target_lengths): gpu for gpu in selected}
        for future in concurrent.futures.as_completed(jobs):
            gpu = jobs[future]
            try:
                print(json.dumps({"event": "warmup_gpu", **future.result()}, sort_keys=True), flush=True)
            except Exception as exc:  # every other GPU still gets its cleanup
                errors.append((gpu, exc))
                print(f"ERROR: GPU {gpu}: {exc}", file=sys.stderr, flush=True)
    if errors:
        return 1
    print(json.dumps({"event": "warmup_done", "gpus": selected, "temporary_jobs_deleted": True,
                      "adapters_discarded": True, "uploaded": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
