#!/usr/bin/env python3
"""Memory-optimised wrapper around the reference Affine simulator.

This script reuses all scenario generation and scoring logic from
``scripts.irt_simulator`` to remain numerically identical to the repo's
baseline while streaming outputs to keep memory usage manageable for very
large runs.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import re
import sys
import types
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent


def _build_affine_stub() -> types.ModuleType:
    source_path = _THIS_DIR.parent / "affine" / "__init__.py"
    text = source_path.read_text()
    try:
        start = text.index("def _sigmoid")
        end = text.index("# Central env registry")
    except ValueError as exc:  # pragma: no cover - developer safety
        raise RuntimeError("Unable to locate OnlineIRT2PL definition in affine source") from exc

    snippet = text[start:end]
    namespace: Dict[str, Any] = {
        "math": math,
        "dataclass": dataclass,
        "Iterable": Iterable,
        "Tuple": Tuple,
        "Dict": Dict,
        "Optional": Optional,
    }
    exec(snippet, namespace)

    module = types.ModuleType("affine_stub")
    for name in [
        "_sigmoid",
        "_clamp",
        "_EnvParams",
        "_ChallengeParams",
        "_AbilityParams",
        "OnlineIRT2PL",
    ]:
        module.__dict__[name] = namespace[name]

    const_defaults = {
        "EPS_FLOOR": 0.005,
        "Z_NOT_WORSE": 1.28,
        "EPS_WIN": 0.008,
        "Z_WIN": 0.5,
        "ELIG": 0.03,
    }
    for const_name in const_defaults:
        pattern = re.compile(rf"^{const_name}\\s*=\\s*([0-9eE+\-.]+)\\s*$", re.MULTILINE)
        match = pattern.search(text)
        if match:
            module.__dict__[const_name] = float(match.group(1))
        else:
            module.__dict__[const_name] = const_defaults[const_name]

    return module


try:
    import affine as _affine_mod  # type: ignore
except Exception:
    _affine_mod = _build_affine_stub()
    sys.modules["affine"] = _affine_mod
else:
    sys.modules["affine"] = _affine_mod

_BASELINE_PATH = _THIS_DIR / "irt_simulator.py"
_SPEC = importlib.util.spec_from_file_location("sim2_baseline", _BASELINE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - safety guard
    raise RuntimeError(f"Unable to load baseline simulator from {_BASELINE_PATH}")
baseline = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = baseline
_SPEC.loader.exec_module(baseline)  # type: ignore[arg-type]

ScenarioConfig = baseline.ScenarioConfig


class _CategoricalEncoder:
    """Utility that assigns stable integer codes to string categories."""

    __slots__ = ("_mapping", "_categories")

    def __init__(self) -> None:
        self._mapping: Dict[str, int] = {}
        self._categories: List[str] = []

    def encode(self, key: str) -> int:
        try:
            return self._mapping[key]
        except KeyError:
            idx = len(self._categories)
            self._mapping[key] = idx
            self._categories.append(key)
            return idx

    @property
    def categories(self) -> List[str]:
        return self._categories


class _MetricsBuffer:
    """Columnar accumulator for per-seed metric outputs."""

    __slots__ = (
        "_total_jobs",
        "_metric_order",
        "_model_order",
        "_rows_per_job",
        "_scenario_encoder",
        "_base_encoder",
        "_metric_encoder",
        "_model_encoder",
        "_scenario_codes",
        "_base_codes",
        "_metric_codes",
        "_model_codes",
        "_seed_values",
        "_metric_template",
        "_model_template",
        "_values",
        "_index",
    )

    def __init__(self, total_jobs: int) -> None:
        self._total_jobs = total_jobs
        self._metric_order: List[str] = []
        self._model_order: Dict[str, List[str]] = {}
        self._rows_per_job: Optional[int] = None
        self._scenario_encoder = _CategoricalEncoder()
        self._base_encoder = _CategoricalEncoder()
        self._metric_encoder = _CategoricalEncoder()
        self._model_encoder = _CategoricalEncoder()
        self._scenario_codes: Optional[np.ndarray] = None
        self._base_codes: Optional[np.ndarray] = None
        self._metric_codes: Optional[np.ndarray] = None
        self._model_codes: Optional[np.ndarray] = None
        self._seed_values: Optional[np.ndarray] = None
        self._metric_template: Optional[np.ndarray] = None
        self._model_template: Optional[np.ndarray] = None
        self._values: Optional[np.ndarray] = None
        self._index = 0

    def _ensure_schema(self, metrics_map: Mapping[str, Mapping[str, float]]) -> None:
        if self._rows_per_job is not None:
            return

        for metric_name, models in metrics_map.items():
            self._metric_order.append(metric_name)
            self._model_order[metric_name] = list(models.keys())

        self._rows_per_job = sum(len(models) for models in self._model_order.values())
        if self._rows_per_job <= 0:
            self._rows_per_job = 0
            self._scenario_codes = np.empty(0, dtype=np.int32)
            self._base_codes = np.empty(0, dtype=np.int32)
            self._metric_codes = np.empty(0, dtype=np.int16)
            self._model_codes = np.empty(0, dtype=np.int16)
            self._seed_values = np.empty(0, dtype=np.int64)
            self._metric_template = np.empty(0, dtype=np.int16)
            self._model_template = np.empty(0, dtype=np.int16)
            self._values = np.empty(0, dtype=np.float64)
            return

        capacity = self._total_jobs * self._rows_per_job
        self._scenario_codes = np.empty(capacity, dtype=np.int32)
        self._base_codes = np.empty(capacity, dtype=np.int32)
        self._metric_codes = np.empty(capacity, dtype=np.int16)
        self._model_codes = np.empty(capacity, dtype=np.int16)
        self._seed_values = np.empty(capacity, dtype=np.int64)
        self._metric_template = np.empty(self._rows_per_job, dtype=np.int16)
        self._model_template = np.empty(self._rows_per_job, dtype=np.int16)
        self._values = np.empty(capacity, dtype=np.float64)

        template_pos = 0
        for metric_name in self._metric_order:
            metric_code = self._metric_encoder.encode(metric_name)
            for model_name in self._model_order[metric_name]:
                self._metric_template[template_pos] = metric_code
                self._model_template[template_pos] = self._model_encoder.encode(model_name)
                template_pos += 1

    def append(self, label: str, base_name: str, seed: int, metrics_map: Mapping[str, Mapping[str, float]]) -> None:
        self._ensure_schema(metrics_map)
        if not self._rows_per_job:
            return

        assert self._scenario_codes is not None
        assert self._base_codes is not None
        assert self._metric_codes is not None
        assert self._model_codes is not None
        assert self._seed_values is not None
        assert self._metric_template is not None
        assert self._model_template is not None
        assert self._values is not None

        start = self._index
        end = start + self._rows_per_job

        scenario_code = self._scenario_encoder.encode(label)
        base_code = self._base_encoder.encode(base_name)

        self._scenario_codes[start:end] = scenario_code
        self._base_codes[start:end] = base_code
        self._metric_codes[start:end] = self._metric_template
        self._model_codes[start:end] = self._model_template
        self._seed_values[start:end] = seed

        pos = start
        for metric_name in self._metric_order:
            model_dict = metrics_map.get(metric_name, {})
            for model_name in self._model_order[metric_name]:
                self._values[pos] = float(model_dict.get(model_name, 0.0))
                pos += 1

        self._index = end

    def to_dataframe(self) -> pd.DataFrame:
        if not self._index:
            return pd.DataFrame(columns=["scenario", "scenario_base", "metric", "model", "value", "seed"])

        assert self._scenario_codes is not None
        assert self._base_codes is not None
        assert self._metric_codes is not None
        assert self._model_codes is not None
        assert self._seed_values is not None
        assert self._values is not None

        length = self._index
        scenario = pd.Categorical.from_codes(
            self._scenario_codes[:length],
            categories=self._scenario_encoder.categories,
        )
        scenario_base = pd.Categorical.from_codes(
            self._base_codes[:length],
            categories=self._base_encoder.categories,
        )
        metric = pd.Categorical.from_codes(
            self._metric_codes[:length],
            categories=self._metric_encoder.categories,
        )
        model = pd.Categorical.from_codes(
            self._model_codes[:length],
            categories=self._model_encoder.categories,
        )

        data = {
            "scenario": scenario,
            "scenario_base": scenario_base,
            "metric": metric,
            "model": model,
            "value": self._values[:length],
            "seed": self._seed_values[:length],
        }

        return pd.DataFrame(data)


class _DynamicBuffer:
    """Resizable columnar buffer for sparsely-emitted metric families."""

    __slots__ = (
        "_capacity",
        "_index",
        "_scenario_codes",
        "_base_codes",
        "_metric_codes",
        "_model_codes",
        "_seed_values",
        "_values",
        "_scenario_encoder",
        "_base_encoder",
        "_metric_encoder",
        "_model_encoder",
    )

    def __init__(self, initial_capacity: int = 1024) -> None:
        self._capacity = initial_capacity
        self._index = 0
        self._scenario_codes = np.empty(initial_capacity, dtype=np.int32)
        self._base_codes = np.empty(initial_capacity, dtype=np.int32)
        self._metric_codes = np.empty(initial_capacity, dtype=np.int16)
        self._model_codes = np.empty(initial_capacity, dtype=np.int16)
        self._seed_values = np.empty(initial_capacity, dtype=np.int64)
        self._values = np.empty(initial_capacity, dtype=np.float64)
        self._scenario_encoder = _CategoricalEncoder()
        self._base_encoder = _CategoricalEncoder()
        self._metric_encoder = _CategoricalEncoder()
        self._model_encoder = _CategoricalEncoder()

    def _ensure_capacity(self, extra: int = 1) -> None:
        needed = self._index + extra
        if needed <= self._capacity:
            return
        new_capacity = max(needed, self._capacity * 2)

        def _grow(arr: np.ndarray, dtype) -> np.ndarray:
            new_arr = np.empty(new_capacity, dtype=dtype)
            new_arr[: self._index] = arr[: self._index]
            return new_arr

        self._scenario_codes = _grow(self._scenario_codes, self._scenario_codes.dtype)
        self._base_codes = _grow(self._base_codes, self._base_codes.dtype)
        self._metric_codes = _grow(self._metric_codes, self._metric_codes.dtype)
        self._model_codes = _grow(self._model_codes, self._model_codes.dtype)
        self._seed_values = _grow(self._seed_values, self._seed_values.dtype)
        self._values = _grow(self._values, self._values.dtype)
        self._capacity = new_capacity

    def append(self, label: str, base_name: str, seed: int, metric_name: str, model_name: str, value: float) -> None:
        self._ensure_capacity(1)
        scenario_code = self._scenario_encoder.encode(label)
        base_code = self._base_encoder.encode(base_name)
        metric_code = self._metric_encoder.encode(metric_name)
        model_code = self._model_encoder.encode(model_name)

        idx = self._index
        self._scenario_codes[idx] = scenario_code
        self._base_codes[idx] = base_code
        self._metric_codes[idx] = metric_code
        self._model_codes[idx] = model_code
        self._seed_values[idx] = seed
        self._values[idx] = float(value)
        self._index += 1

    def to_dataframe(self) -> pd.DataFrame:
        if not self._index:
            return pd.DataFrame(columns=["scenario", "scenario_base", "metric", "model", "value", "seed"])

        length = self._index
        data = {
            "scenario": pd.Categorical.from_codes(
                self._scenario_codes[:length],
                categories=self._scenario_encoder.categories,
            ),
            "scenario_base": pd.Categorical.from_codes(
                self._base_codes[:length],
                categories=self._base_encoder.categories,
            ),
            "metric": pd.Categorical.from_codes(
                self._metric_codes[:length],
                categories=self._metric_encoder.categories,
            ),
            "model": pd.Categorical.from_codes(
                self._model_codes[:length],
                categories=self._model_encoder.categories,
            ),
            "value": self._values[:length],
            "seed": self._seed_values[:length],
        }
        return pd.DataFrame(data)


class _GridAggregator:
    """Streaming aggregator for grid search deltas."""

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: Dict[str, Dict[Tuple[float, float], Dict[str, Any]]] = {}

    def update(
        self,
        base_name: str,
        grid_meta: Mapping[str, float],
        metrics_map: Mapping[str, Mapping[str, float]],
    ) -> None:
        noise = float(grid_meta.get("noise", 0.0))
        samples = float(grid_meta.get("samples", 0.0))
        bucket = self._data.setdefault(base_name, {}).setdefault((noise, samples), {"count": 0, "metrics": {}})
        bucket["count"] += 1
        metric_bucket: Dict[str, float] = bucket["metrics"]  # type: ignore[assignment]
        for metric_name, model_values in metrics_map.items():
            delta = float(model_values.get("irt", 0.0) - model_values.get("baseline", 0.0))
            metric_bucket[metric_name] = metric_bucket.get(metric_name, 0.0) + delta

    def to_dict(self) -> Dict[str, List[Dict[str, float]]]:
        if not self._data:
            return {}

        out: Dict[str, List[Dict[str, float]]] = {}
        for base_name, combos in self._data.items():
            rows: List[Dict[str, float]] = []
            for (noise, samples), payload in combos.items():
                count = max(1, int(payload["count"]))
                metric_bucket: Mapping[str, float] = payload["metrics"]
                row: Dict[str, float] = {"noise": noise, "samples": samples}
                for metric_name, total in metric_bucket.items():
                    row[f"{metric_name}_delta"] = float(total) / float(count)
                rows.append(row)
            out[base_name] = rows
        return out


def simulate(
        scenario_name: str,
        seed: int,
        *,
        scenario_label: Optional[str] = None,
        interaction_scale: float = 1.0,
        return_details: bool = False,
        config_override: Optional[ScenarioConfig] = None,
        grid_meta: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Delegate to the baseline simulator, trimming heavy payloads when possible."""

    result = baseline.simulate(
        scenario_name,
        seed,
        scenario_label=scenario_label,
        interaction_scale=interaction_scale,
        return_details=return_details,
        config_override=config_override,
        grid_meta=grid_meta,
    )

    if return_details:
        return result

    trimmed = {
        "scenario": result["scenario"],
        "scenario_base": result["scenario_base"],
        "description": result.get("description"),
        "metrics": result["metrics"],
        "special": result.get("special", {}),
    }
    if "grid_meta" in result:
        trimmed["grid_meta"] = result["grid_meta"]
    return trimmed


def _simulate_worker(
        args: Tuple[str, str, int, float, bool, Optional[ScenarioConfig], Optional[Dict[str, float]]]
) -> Tuple[str, str, int, bool, Dict[str, Any]]:
    base_name, label, seed, interaction_scale, return_details, override, grid_meta = args
    result = simulate(
        base_name,
        seed,
        scenario_label=label,
        interaction_scale=interaction_scale,
        return_details=return_details,
        config_override=override,
        grid_meta=grid_meta,
    )
    return label, base_name, seed, return_details, result


def run_all_scenarios(
        seeds: Iterable[int],
        *,
        scenario_names: Optional[List[str]] = None,
        replicas: int = 1,
        interaction_scale: float = 1.0,
        random_scenarios: int = 0,
        workers: Optional[int] = None,
        backend: str = "process",
        variations_per_scenario: int = 0,
        grid_size: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, List[Dict[str, float]]]]:
    default_order = [
        "balanced",
        "env_imbalance",
        "cold_start",
        "drift",
        "adversarial_gamer",
        "random_noise",
        "perfect_vs_failure",
        "zero_discrimination",
        "tiny_data",
    ]
    if scenario_names:
        if any(name.lower() == "all" for name in scenario_names):
            scenario_names = default_order
        else:
            unknown = sorted(set(scenario_names) - set(default_order))
            if unknown:
                raise ValueError(f"Unknown scenario names: {unknown}")
    else:
        scenario_names = default_order

    rng = random.Random(12345)
    job_plan: List[Tuple[str, str]] = []
    scenario_overrides: Dict[str, ScenarioConfig] = {}
    grid_meta_map: Dict[str, Dict[str, float]] = {}
    for scenario in scenario_names:
        for rep in range(replicas):
            label = scenario if replicas == 1 else f"{scenario}#rep{rep:02d}"
            job_plan.append((scenario, label))
        if variations_per_scenario > 0:
            for cfg in baseline._make_variations(scenario, variations_per_scenario, rng):
                scenario_overrides[cfg.name] = cfg
                job_plan.append((scenario, cfg.name))
        if grid_size > 0:
            noise_levels = baseline._logspace(0.1, 10.0, grid_size)
            sample_levels = baseline._logspace(10.0, 10_000.0, grid_size)
            for ni, noise_mul in enumerate(noise_levels):
                for si, sample_target in enumerate(sample_levels):
                    cfg = baseline._build_named_scenario(scenario, rng)
                    cfg.name = f"{scenario}~grid{ni:02d}_{si:02d}"
                    cfg.interactions = max(10, int(round(sample_target)))
                    cfg.challenge_pool = max(50, int(round(math.sqrt(cfg.interactions) * 5)))
                    cfg.challenge_bias_std *= noise_mul
                    cfg.global_noise = min(
                        0.95,
                        max(0.0, cfg.global_noise * noise_mul if cfg.global_noise > 0 else 0.05 * noise_mul),
                    )
                    for profile in cfg.miners.values():
                        profile.noise = min(
                            0.95,
                            max(0.0, profile.noise * noise_mul if profile.noise > 0 else 0.05 * noise_mul),
                        )
                    scenario_overrides[cfg.name] = cfg
                    grid_meta_map[cfg.name] = {
                        "noise": float(noise_mul),
                        "samples": float(cfg.interactions),
                    }
                    job_plan.append((scenario, cfg.name))
    for idx in range(random_scenarios):
        base = f"random_{idx:02d}"
        job_plan.append((base, base))

    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("At least one seed is required")

    scenario_target_counts = {label: len(seed_list) for _, label in job_plan}
    scenario_progress = {label: 0 for _, label in job_plan}

    first_seed = seed_list[0]
    job_specs: List[Tuple[str, str, Optional[ScenarioConfig], Optional[Dict[str, float]], float]] = []
    for base_name, label in job_plan:
        override = scenario_overrides.get(label)
        grid_meta = grid_meta_map.get(label)
        job_scale = 1.0 if grid_meta else interaction_scale
        job_specs.append((base_name, label, override, grid_meta, job_scale))

    total_jobs = len(job_plan) * len(seed_list)
    completed = 0

    metrics_buffer = _MetricsBuffer(total_jobs)
    special_buffer = _DynamicBuffer(max(1024, len(job_plan) * 4))
    sample_details: Dict[str, Any] = {}
    grid_aggregator = _GridAggregator()

    max_workers = max(1, workers or (os.cpu_count() or 1))

    def process_result(label: str, base_name: str, seed: int, return_details: bool, result: Dict[str, Any]) -> None:
        nonlocal completed
        completed += 1
        scenario_progress[label] += 1

        metrics_map = result["metrics"]
        metrics_buffer.append(label, base_name, seed, metrics_map)

        special_map = result.get("special", {})
        for special_name, models in special_map.items():
            for model_name, value in models.items():
                special_buffer.append(label, base_name, seed, special_name, model_name, value)

        grid_meta = result.get("grid_meta")
        if grid_meta:
            grid_aggregator.update(base_name, grid_meta, metrics_map)

        if return_details and label not in sample_details:
            sample_details[label] = result

        if scenario_progress[label] == scenario_target_counts[label]:
            print(
                f"[simulate] Completed scenario {label} with {scenario_target_counts[label]} seeds "
                f"({completed}/{total_jobs})"
            )

    backend_lower = (backend or "process").lower()
    if backend_lower not in {"process", "thread", "none"}:
        raise ValueError(f"Unknown backend: {backend}")

    executor_cls = None
    if backend_lower == "process":
        executor_cls = ProcessPoolExecutor if max_workers > 1 else None
    elif backend_lower == "thread":
        executor_cls = ThreadPoolExecutor if max_workers > 1 else None

    def job_iterator() -> Iterable[Tuple[str, str, int, float, bool, Optional[ScenarioConfig], Optional[Dict[str, float]]]]:
        for base_name, label, override, grid_meta, job_scale in job_specs:
            for seed in seed_list:
                yield (base_name, label, seed, job_scale, seed == first_seed, override, grid_meta)

    def _execute(executor_class) -> None:
        if executor_class is None:
            for job in job_iterator():
                label, base_name, seed, return_details, result = _simulate_worker(job)
                process_result(label, base_name, seed, return_details, result)
        else:
            chunksize = 1
            if max_workers > 1 and total_jobs > max_workers:
                chunksize = max(1, min(64, total_jobs // (max_workers * 4)))
            with executor_class(max_workers=max_workers) as pool:
                for label, base_name, seed, return_details, result in pool.map(
                    _simulate_worker,
                    job_iterator(),
                    chunksize=chunksize,
                ):
                    process_result(label, base_name, seed, return_details, result)

    try:
        _execute(executor_cls)
    except PermissionError:
        if backend_lower == "process":
            print("Process pool not permitted; falling back to thread backend.")
            if completed:
                metrics_buffer = _MetricsBuffer(total_jobs)
                special_buffer = _DynamicBuffer(max(1024, len(job_plan) * 4))
                sample_details = {}
                grid_aggregator = _GridAggregator()
                scenario_progress = {label: 0 for _, label in job_plan}
                completed = 0
            fallback_cls = ThreadPoolExecutor if max_workers > 1 else None
            _execute(fallback_cls)
        else:
            raise

    metrics_df = metrics_buffer.to_dataframe()
    special_df = special_buffer.to_dataframe()
    grid_records = grid_aggregator.to_dict()
    return metrics_df, special_df, sample_details, grid_records


def main(argv: Optional[List[str]] = None) -> None:
    args = baseline.parse_args(argv)
    grid_data: Dict[str, List[Dict[str, float]]] = {}

    if args.load_prefix:
        base_path = Path(args.load_prefix)
        if base_path.suffix:
            base_path = base_path.with_suffix("")
        metrics_path = base_path.parent / f"{base_path.name}_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics CSV not found: {metrics_path}")
        metrics_df = pd.read_csv(metrics_path)

        special_path = base_path.parent / f"{base_path.name}_special.csv"
        special_df = pd.read_csv(special_path) if special_path.exists() else pd.DataFrame()

        grid_path = base_path.parent / f"{base_path.name}_grid.json"
        if grid_path.exists():
            grid_data = json.loads(grid_path.read_text())
        else:
            grid_data = {}

        sample_details = {}
        if not grid_data:
            grid_data = baseline._reconstruct_grid_from_metrics(metrics_df)
    else:
        seeds = range(args.seeds)
        metrics_df, special_df, sample_details, grid_data = run_all_scenarios(
            seeds,
            scenario_names=args.scenarios,
            replicas=args.replicas,
            interaction_scale=args.interaction_scale,
            random_scenarios=args.random_scenarios,
            workers=args.workers,
            backend=args.backend,
            variations_per_scenario=args.variations,
            grid_size=args.grid_size,
        )

    summary = (
        metrics_df.groupby(["scenario_base", "metric", "model"])  # type: ignore[arg-type]
        .agg(mean=("value", "mean"), std=("value", "std"))
        .reset_index()
    )
    print("=== Metric Summary (mean ± std across seeds/replicas) ===")
    for scenario in summary["scenario_base"].unique():
        print(f"\nScenario: {scenario}")
        sub = summary[summary["scenario_base"] == scenario]
        for metric in sub["metric"].unique():
            sub_metric = sub[sub["metric"] == metric]
            line = ", ".join(
                f"{row['model']}: {row['mean']:.3f} ± {row['std']:.3f}" for _, row in sub_metric.iterrows()
            )
            print(f"  {metric}: {line}")

    diff_summary = (
        summary.pivot_table(index=["scenario_base", "metric"], columns="model", values="mean")
        .reset_index()
    )
    diff_summary["delta"] = diff_summary["irt"] - diff_summary["baseline"]
    print("\n=== Mean Improvement (IRT - Baseline) ===")
    for scenario in diff_summary["scenario_base"].unique():
        rows = diff_summary[diff_summary["scenario_base"] == scenario]
        deltas = ", ".join(
            f"{row['metric']}: {row['delta']:+.4f}" for _, row in rows.iterrows()
        )
        print(f"  {scenario}: {deltas}")

    if not special_df.empty:
        special_summary = (
            special_df.groupby(["scenario_base", "metric", "model"])  # type: ignore[arg-type]
            .agg(mean=("value", "mean"), std=("value", "std"))
            .reset_index()
        )
        print("\n=== Special-case Diagnostics ===")
        for scenario in special_summary["scenario_base"].unique():
            print(f"\nScenario: {scenario}")
            sub = special_summary[special_summary["scenario_base"] == scenario]
            for metric in sub["metric"].unique():
                sub_metric = sub[sub["metric"] == metric]
                line = ", ".join(
                    f"{row['model']}: {row['mean']:.3f} ± {row['std']:.3f}" for _, row in sub_metric.iterrows()
                )
                print(f"  {metric}: {line}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.load_prefix:
        metrics_path = output_dir / f"{args.csv_prefix}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics CSV to {metrics_path}")
        if not special_df.empty:
            special_path = output_dir / f"{args.csv_prefix}_special.csv"
            special_df.to_csv(special_path, index=False)
            print(f"Saved special metrics CSV to {special_path}")
        if grid_data:
            grid_path = output_dir / f"{args.csv_prefix}_grid.json"
            grid_path.write_text(json.dumps(grid_data, indent=2))
            print(f"Saved grid metadata to {grid_path}")

    if args.dump_details:
        dump_path = Path(args.dump_details)
        payload = {
            key: {
                "scenario": value["scenario"],
                "scenario_base": value.get("scenario_base", key),
                "description": value.get("description"),
                "metrics": value.get("metrics"),
                "special": value.get("special"),
                "coverage": value.get("coverage"),
            }
            for key, value in sample_details.items()
        }
        dump_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved scenario detail snapshot to {dump_path}")

    if not args.no_plot:
        baseline.build_visualization(
            metrics_df,
            sample_details,
            output_dir / "irt_simulation_overview.png",
            grid_data=grid_data,
        )


if __name__ == "__main__":
    main()
