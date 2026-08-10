"""Load the affine.toml chain contract + environment secrets.

`affine.toml` is the public contract (what miners can rely on); env vars are
operator secrets (doppler-provided). Every module reads config through
`load_config()` — nothing else parses the toml or reaches into os.environ for
contract values.

Sections are parsed once into typed dataclasses so call sites use
`cfg.duel.k_sigma` (already the right type) instead of
`float(cfg.duel["k_sigma"])`. The raw dict is still exposed as `cfg.raw` for
the eval-server side (engine/dueling), which is handed the whole dict and
reads it positionally.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Secrets:
    """Operator credentials pulled from the environment at startup."""

    hf_token: str = ""
    hippius_access_key: str = ""
    hippius_secret_key: str = ""
    lium_api_key: str = ""
    targon_api_key: str = ""
    openrouter_api_key: str = ""
    # Shared secret the validator sends and evalsrv requires on every request.
    eval_token: str = ""
    # TaoMarketCap public API key (dash market stream). Header: Authorization: <key>.
    taomarketcap: str = ""

    @classmethod
    def from_env(cls) -> "Secrets":
        return cls(
            hf_token=os.environ.get("HF_TOKEN", ""),
            hippius_access_key=os.environ.get("HIPPIUS_ACCESS_KEY", ""),
            hippius_secret_key=os.environ.get("HIPPIUS_SECRET_KEY", ""),
            lium_api_key=os.environ.get("LIUM_API_KEY", ""),
            targon_api_key=os.environ.get("TARGON_API_KEY", ""),
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            eval_token=os.environ.get("AFFINE_EVAL_TOKEN", ""),
            # Doppler stores TMC_API_KEY; older docs used TAOMARKETCAP.
            taomarketcap=(
                os.environ.get("TAOMARKETCAP")
                or os.environ.get("TMC_API_KEY")
                or ""
            ),
        )


@dataclass(frozen=True)
class SubmissionCfg:
    reveal_prefix: str
    repo_pattern: str
    coldkey_prefix_len: int
    coldkey_suffix_len: int
    max_model_size_gb: float
    max_total_repo_gb: float
    allow_python_files: bool
    allow_auto_map: bool
    max_repo_files: int
    max_config_bytes: int


@dataclass(frozen=True)
class TeacherCfg:
    repo: str
    port: int
    tp: int
    gpus: str
    # When set (OpenAI-compatible base incl. /v1), evalsrv probes this remote
    # endpoint and skips launching a local teacher vLLM.
    base_url: str = ""


@dataclass(frozen=True)
class DuelCfg:
    # Scoring contract (Reason v3): n_turns + k_sigma. That's all of it.
    n_turns: int
    k_sigma: float
    # Sampling / ops knobs (no effect on scoring semantics).
    n_teacher_samples: int
    n_miner_samples: int
    temperature: float
    max_thought_tokens: int
    max_action_tokens: int
    concurrency: int
    timeout_s: int


@dataclass(frozen=True)
class DatasetCfg:
    corpus_base_url: str
    manifest_key: str
    refresh_interval_s: int = 3600


@dataclass(frozen=True)
class BenchCfg:
    enabled: bool
    suites: list[str]
    num_trials: int
    max_concurrency: int
    user_llm: str
    policy: str


@dataclass(frozen=True)
class EvalMachineCfg:
    provider_order: list[str]
    gpu_type: str
    gpu_types: list[str]
    gpu_count: int
    max_price_per_hour: float
    port: int
    health_interval_s: int
    unhealthy_threshold: int
    provision_timeout_s: int
    lium_template_id: str
    targon_resource: str
    targon_image: str


# Alias: bench pods reuse the same machine knobs as the duel eval machine.
BenchMachineCfg = EvalMachineCfg


@dataclass(frozen=True)
class ValidatorCfg:
    poll_interval_s: int
    weight_interval_s: int
    state_dir: str
    tick_warn_after_s: int
    tick_restart_after_s: int
    stream_idle_warn_s: int
    stream_idle_timeout_s: int
    max_transient_eval_retries: int
    max_infra_front_requeues: int
    dashboard_flush_min_interval_s: float
    metagraph_max_age_s: int
    jobs_retention: int


@dataclass(frozen=True)
class Config:
    """Typed view over affine.toml."""

    raw: dict = field(repr=False)
    secrets: Secrets = field(repr=False)
    submission: SubmissionCfg = field(repr=False)
    teacher: TeacherCfg = field(repr=False)
    duel: DuelCfg = field(repr=False)
    dataset: DatasetCfg = field(repr=False)
    bench: BenchCfg = field(repr=False)
    eval_machine: EvalMachineCfg = field(repr=False)
    bench_machine: EvalMachineCfg = field(repr=False)
    validator: ValidatorCfg = field(repr=False)

    # -- promoted scalar values ---------------------------------------------
    @property
    def netuid(self) -> int:
        return int(self.raw["subnet"]["netuid"])

    @property
    def network(self) -> str:
        return self.raw["subnet"]["network"]

    @property
    def subnet_name(self) -> str:
        return self.raw["subnet"]["name"]

    @property
    def burn_uid(self) -> int:
        return int(self.raw["subnet"]["burn_uid"])

    @property
    def king_chain_size(self) -> int:
        return int(self.raw["subnet"]["king_chain_size"])

    @property
    def min_submission_block(self) -> int:
        return int(self.raw["subnet"]["min_submission_block"])

    @property
    def weight_version_key(self) -> int:
        return int(self.raw["subnet"].get("weight_version_key", 0))

    @property
    def wallet_name(self) -> str:
        return self.raw["wallet"]["name"]

    @property
    def wallet_hotkey(self) -> str:
        return self.raw["wallet"]["hotkey"]

    # -- sections still consumed as dicts (single-module) -------------------
    @property
    def miner_serving(self) -> dict:
        return self.raw["miner_serving"]

    @property
    def seed_king(self) -> dict:
        return self.raw["seed_king"]

    @property
    def hippius(self) -> dict:
        return self.raw["hippius"]

    @property
    def dashboard(self) -> dict:
        """Hot-path dash-api knobs (host/port/public URL). Optional section."""
        return self.raw.get("dashboard") or {
            "api_host": "127.0.0.1",
            "api_port": 8787,
            "public_host": "localhost",
            "public_base_url": "https://localhost:8443",
        }

    @property
    def state_dir(self) -> Path:
        d = Path(self.validator.state_dir)
        if not d.is_absolute():
            d = _repo_root() / d
        d.mkdir(parents=True, exist_ok=True)
        return d


def _submission(raw: dict) -> SubmissionCfg:
    s = raw["submission"]
    return SubmissionCfg(
        reveal_prefix=str(s["reveal_prefix"]),
        repo_pattern=str(s["repo_pattern"]),
        coldkey_prefix_len=int(s["coldkey_prefix_len"]),
        coldkey_suffix_len=int(s["coldkey_suffix_len"]),
        max_model_size_gb=float(s["max_model_size_gb"]),
        max_total_repo_gb=float(s["max_total_repo_gb"]),
        allow_python_files=bool(s["allow_python_files"]),
        allow_auto_map=bool(s["allow_auto_map"]),
        max_repo_files=int(s["max_repo_files"]),
        max_config_bytes=int(s["max_config_bytes"]),
    )


def _duel(raw: dict) -> DuelCfg:
    d = raw["duel"]
    return DuelCfg(
        n_turns=int(d["n_turns"]), k_sigma=float(d["k_sigma"]),
        n_teacher_samples=int(d["n_teacher_samples"]),
        n_miner_samples=int(d["n_miner_samples"]),
        temperature=float(d["temperature"]),
        max_thought_tokens=int(d["max_thought_tokens"]),
        max_action_tokens=int(d["max_action_tokens"]),
        concurrency=int(d["concurrency"]), timeout_s=int(d["timeout_s"]),
    )


def _machine_cfg(section: dict) -> EvalMachineCfg:
    gpu_type = str(section["gpu_type"])
    gpu_types = list(section["gpu_types"]) if section.get("gpu_types") else [gpu_type]
    return EvalMachineCfg(
        provider_order=list(section["provider_order"]), gpu_type=gpu_type,
        gpu_types=gpu_types, gpu_count=int(section["gpu_count"]),
        max_price_per_hour=float(section["max_price_per_hour"]),
        port=int(section["port"]),
        health_interval_s=int(section["health_interval_s"]),
        unhealthy_threshold=int(section["unhealthy_threshold"]),
        provision_timeout_s=int(section["provision_timeout_s"]),
        lium_template_id=str(section.get("lium_template_id", "")),
        targon_resource=str(section["targon_resource"]),
        targon_image=str(section["targon_image"]),
    )


def _eval_machine(raw: dict) -> EvalMachineCfg:
    return _machine_cfg(raw["eval_machine"])


def _bench_machine(raw: dict) -> EvalMachineCfg:
    return _machine_cfg(raw["bench_machine"])


def _validator(raw: dict) -> ValidatorCfg:
    v = raw["validator"]
    return ValidatorCfg(
        poll_interval_s=int(v["poll_interval_s"]),
        weight_interval_s=int(v["weight_interval_s"]),
        state_dir=str(v["state_dir"]),
        tick_warn_after_s=int(v["tick_warn_after_s"]),
        tick_restart_after_s=int(v["tick_restart_after_s"]),
        stream_idle_warn_s=int(v["stream_idle_warn_s"]),
        stream_idle_timeout_s=int(v["stream_idle_timeout_s"]),
        max_transient_eval_retries=int(v["max_transient_eval_retries"]),
        max_infra_front_requeues=int(v.get("max_infra_front_requeues", 5)),
        dashboard_flush_min_interval_s=float(v["dashboard_flush_min_interval_s"]),
        metagraph_max_age_s=int(v["metagraph_max_age_s"]),
        jobs_retention=int(v["jobs_retention"]),
    )


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else _repo_root() / "affine.toml"
    with open(p, "rb") as f:
        raw = tomllib.load(f)
    t = raw["teacher"]
    ds = raw["dataset"]
    b = raw["bench"]
    return Config(
        raw=raw,
        secrets=Secrets.from_env(),
        submission=_submission(raw),
        teacher=TeacherCfg(repo=str(t["repo"]), port=int(t["port"]),
                           tp=int(t["tp"]), gpus=str(t["gpus"]),
                           base_url=str(t.get("base_url") or "").rstrip("/")),
        duel=_duel(raw),
        dataset=DatasetCfg(corpus_base_url=str(ds["corpus_base_url"]).rstrip("/"),
                           manifest_key=str(ds["manifest_key"]),
                           refresh_interval_s=int(ds.get("refresh_interval_s", 3600))),
        bench=BenchCfg(enabled=bool(b["enabled"]), suites=list(b["suites"]),
                       num_trials=int(b["num_trials"]),
                       max_concurrency=int(b["max_concurrency"]),
                       user_llm=str(b["user_llm"]), policy=str(b["policy"])),
        eval_machine=_eval_machine(raw),
        bench_machine=_bench_machine(raw),
        validator=_validator(raw),
    )
