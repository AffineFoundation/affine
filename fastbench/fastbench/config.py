"""Env-driven configuration. Every knob has a sane default for the target
box (8x B200/B300, ~200 vCPU, multi-TB NVMe); override via FASTBENCH_* env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Config:
    # -- service -------------------------------------------------------------
    port: int = field(default_factory=lambda: _int("FASTBENCH_PORT", 9100))
    token: str = field(default_factory=lambda: os.environ.get("FASTBENCH_TOKEN", ""))
    data_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("FASTBENCH_DATA_DIR", "/root/fastbench_data")))
    # -- serving (vLLM) --------------------------------------------------------
    vllm_port: int = field(default_factory=lambda: _int("FASTBENCH_VLLM_PORT", 8100))
    # Wall time is set by the slowest single rollout, whose steps are strictly
    # sequential, so per-step latency beats aggregate throughput. Measured on
    # 8x B200 with Qwen3.6-35B-A3B at n=25: TP=1/DP=8 -> 526s of rollouts,
    # TP=4/DP=2 -> 282s. Hence TP=4 rather than the throughput-optimal TP=1.
    tp: int = field(default_factory=lambda: _int("FASTBENCH_TP", 4))
    # 0 = auto: use every visible GPU (dp = n_gpus // tp).
    dp: int = field(default_factory=lambda: _int("FASTBENCH_DP", 0))
    max_model_len: int = field(default_factory=lambda: _int(
        "FASTBENCH_MAX_MODEL_LEN", 65536))
    gpu_memory_utilization: float = field(default_factory=lambda: _float(
        "FASTBENCH_GPU_UTIL", 0.85))
    max_num_seqs: int = field(default_factory=lambda: _int(
        "FASTBENCH_MAX_NUM_SEQS", 256))
    max_num_batched_tokens: int = field(default_factory=lambda: _int(
        "FASTBENCH_MAX_BATCHED_TOKENS", 32768))
    # Targeted Blackwell mitigations (measured on this box, 2026-08-22):
    #  * moe-backend auto picks flashinfer_trtllm whose fused_moe_trtllm_sm100
    #    JIT fails against the pip cu13 toolchain ("CUDA compiler and CUDA
    #    toolkit headers are incompatible") -> force triton.
    #  * gdn_prefill auto JITs on Qwen3.6 GDN models -> force triton.
    #  * Do NOT force the attention backend: FLASH_ATTN rejects head_dim=256
    #    on SM100 (AssertionError), auto attention works.
    # Empty string disables an override.
    moe_backend: str = field(default_factory=lambda: os.environ.get(
        "FASTBENCH_MOE_BACKEND", "triton"))
    gdn_prefill_backend: str = field(default_factory=lambda: os.environ.get(
        "FASTBENCH_GDN_PREFILL", "triton"))
    # Extra raw args appended to `vllm serve` (space-separated).
    vllm_extra_args: str = field(default_factory=lambda: os.environ.get(
        "FASTBENCH_VLLM_EXTRA_ARGS", ""))
    model_load_timeout_s: int = field(default_factory=lambda: _int(
        "FASTBENCH_MODEL_LOAD_TIMEOUT_S", 3600))
    # How many model snapshots to keep in the HF cache (LRU; the currently
    # served one is always kept).
    keep_snapshots: int = field(default_factory=lambda: _int(
        "FASTBENCH_KEEP_SNAPSHOTS", 3))
    # -- rollouts (mini-SWE-agent) ---------------------------------------------
    # Hard cap on concurrent agent rollouts (docker/CPU-bound). The effective
    # worker count is min(n, cap).
    agent_workers_cap: int = field(default_factory=lambda: _int(
        "FASTBENCH_AGENT_WORKERS_CAP", 150))
    # At 50 steps, 15 of 25 tasks had edited files but ran out of steps before
    # submitting, so real work scored zero. 75 buys those tasks a chance to
    # finish, at roughly 3 extra minutes on an n=25 warm run.
    step_limit: int = field(default_factory=lambda: _int(
        "FASTBENCH_STEP_LIMIT", 75))
    # Reasoning models break the text-action agent: their <think> block lands
    # in `content`, so the parser sees prose full of ``` fences and aborts with
    # RepeatedFormatError (measured: 16/25 tasks on Qwen3.6-35B-A3B).
    # Thinking off also cuts tokens per step, which is the rollout bottleneck.
    disable_thinking: bool = field(default_factory=lambda: os.environ.get(
        "FASTBENCH_DISABLE_THINKING", "1") == "1")
    # Alternative to the above: let vLLM split reasoning into its own field
    # (e.g. "qwen3", "deepseek_r1"). Model-specific, so off by default.
    reasoning_parser: str = field(default_factory=lambda: os.environ.get(
        "FASTBENCH_REASONING_PARSER", ""))
    max_tokens: int = field(default_factory=lambda: _int(
        "FASTBENCH_MAX_TOKENS", 8192))
    # Kill a rollout that repeats one identical action this many times: a
    # collapsed checkpoint otherwise spends its whole step budget looping (one
    # merge repeated a single invented command 58 times in a row). 0 disables.
    repeat_limit: int = field(default_factory=lambda: _int(
        "FASTBENCH_REPEAT_LIMIT", 8))
    # -- eval (swebench harness) -------------------------------------------------
    # Max docker workers inside one eval batch subprocess.
    eval_workers_cap: int = field(default_factory=lambda: _int(
        "FASTBENCH_EVAL_WORKERS_CAP", 24))
    # Concurrent eval batch subprocesses (streaming evals overlap rollouts).
    # At n=150 four slots left a 231s tail (28% of the wall) draining the last
    # rollouts; the box has 208 vCPU, so the tail was scheduling, not CPU.
    eval_parallel: int = field(default_factory=lambda: _int(
        "FASTBENCH_EVAL_PARALLEL", 8))
    # Per-instance test timeout handed to the harness. Its own default is
    # 1800s, so one hanging test suite can add half an hour to a run that is
    # otherwise minutes long (observed: a 3-task bench stuck 15min on one
    # container). A suite that cannot finish inside this is a failed task.
    eval_instance_timeout_s: int = field(default_factory=lambda: _int(
        "FASTBENCH_EVAL_INSTANCE_TIMEOUT_S", 600))
    # Hard ceiling per eval batch, in case the harness ignores its own timeout.
    eval_batch_timeout_s: float = field(default_factory=lambda: _float(
        "FASTBENCH_EVAL_BATCH_TIMEOUT_S", 900.0))
    # Batch accumulation: each batch pays a fixed harness startup, so prefer
    # a few instances per launch once the pipe is primed.
    eval_batch_min: int = field(default_factory=lambda: _int(
        "FASTBENCH_EVAL_BATCH_MIN", 4))
    eval_batch_max_wait_s: float = field(default_factory=lambda: _float(
        "FASTBENCH_EVAL_BATCH_MAX_WAIT_S", 90.0))
    # -- job ---------------------------------------------------------------------
    job_timeout_s: int = field(default_factory=lambda: _int(
        "FASTBENCH_JOB_TIMEOUT_S", 7200))
    queue_max: int = field(default_factory=lambda: _int(
        "FASTBENCH_QUEUE_MAX", 20))
    history_keep_artifacts: int = field(default_factory=lambda: _int(
        "FASTBENCH_KEEP_ARTIFACTS", 50))

    @property
    def bench_root(self) -> Path:
        return self.data_dir / "runs"

    @property
    def history_path(self) -> Path:
        return self.data_dir / "history.jsonl"


def load_config() -> Config:
    return Config()
