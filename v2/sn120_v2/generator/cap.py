"""Per-epoch submission capacity, frozen at the first link issuance for that window."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping

from ..errors import ConfigError
from .common import atomic_json, read_json, utc_text
from .constants import MAX_TASKS_PER_SUBMISSION
from .epoch import epoch_id

SCORING_BUDGET_S = 3 * 3600 + 45 * 60
FORMULA = "floor(rollouts_per_pod_s * pods * (scoring_budget_s - king_swap_s) * (1 - safety_margin) / samples_per_task); cap = clamp(floor(capacity_tasks / active_hotkeys), min_cap, max_cap)"
DEFAULTS = {
    "scoring_budget_s": SCORING_BUDGET_S,
    "king_swap_s": 900,
    "safety_margin": 0.25,
    "samples_per_task": 8,
    "min_cap": 1,
    "max_cap": MAX_TASKS_PER_SUBMISSION,
}


def pods_from_helpers(ssh: Mapping[str, Any] | None) -> int:
    helpers = (ssh or {}).get("helpers", [])
    if not isinstance(helpers, list):
        raise ConfigError("Generator ssh.helpers must be a list")
    return 1 + len(helpers)


def cap_inputs(config: Mapping[str, Any] | None, *, ssh: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise ConfigError("Generator submission_cap must be an object")
    allowed = {"first_epoch", "rollouts_per_pod_s", "pods", *DEFAULTS}
    unknown = set(config) - allowed
    if unknown:
        raise ConfigError(f"unknown Generator submission_cap fields: {', '.join(sorted(unknown))}")
    if "first_epoch" not in config or "rollouts_per_pod_s" not in config:
        raise ConfigError("Generator submission_cap needs first_epoch and rollouts_per_pod_s")
    values = {**DEFAULTS, **config}
    if "pods" not in config:
        values["pods"] = pods_from_helpers(ssh)
    values["first_epoch"] = epoch_id(values["first_epoch"])
    for key in ("rollouts_per_pod_s", "safety_margin"):
        if type(values[key]) not in (int, float) or not math.isfinite(values[key]):
            raise ConfigError(f"Generator submission_cap {key} must be finite")
    if values["rollouts_per_pod_s"] <= 0 or not 0 <= values["safety_margin"] < 1:
        raise ConfigError("Generator submission_cap requires positive throughput and 0 <= safety_margin < 1")
    for key in ("pods", "scoring_budget_s", "king_swap_s", "samples_per_task", "min_cap", "max_cap"):
        if type(values[key]) is not int or values[key] <= 0:
            raise ConfigError(f"Generator submission_cap {key} must be a positive integer")
    if values["scoring_budget_s"] <= values["king_swap_s"]:
        raise ConfigError("Generator scoring budget must exceed king swap time")
    if values["min_cap"] > values["max_cap"] or values["max_cap"] > MAX_TASKS_PER_SUBMISSION:
        raise ConfigError("Generator submission_cap clamps are invalid")
    return values


def capacity_tasks(inputs: Mapping[str, Any]) -> int:
    return math.floor(
        inputs["rollouts_per_pod_s"] * inputs["pods"]
        * (inputs["scoring_budget_s"] - inputs["king_swap_s"])
        * (1 - inputs["safety_margin"]) / inputs["samples_per_task"]
    )


def cap_for(inputs: Mapping[str, Any], active_hotkeys: int) -> tuple[int, int]:
    if type(active_hotkeys) is not int or active_hotkeys < 1:
        raise ConfigError("Generator submission cap requires at least one active hotkey")
    capacity = capacity_tasks(inputs)
    cap = min(inputs["max_cap"], max(inputs["min_cap"], capacity // active_hotkeys))
    return cap, capacity


def rejection_reason(count: int, epoch: str, cap: int) -> str:
    return f"submission has {count} task ids; the per-hotkey cap for {epoch} is {cap}"


def frozen(work_dir: Path, epoch: str) -> dict[str, Any] | None:
    path = work_dir / "caps" / f"{epoch_id(epoch)}.json"
    return read_json(path) if path.is_file() else None


def freeze(
    work_dir: Path, epoch: str, inputs: Mapping[str, Any] | None, active_hotkeys: int, *, now: float | None = None
) -> dict[str, Any] | None:
    """Read an immutable cap or create it with the first actual issuance count."""
    epoch = epoch_id(epoch)
    existing = frozen(work_dir, epoch)
    if existing is not None:
        return existing
    if inputs is None or epoch < inputs["first_epoch"]:
        return None
    cap, capacity = cap_for(inputs, active_hotkeys)
    timestamp = time.time() if now is None else now
    record = {
        "epoch": epoch, "cap": cap, "capacity_tasks": capacity, "active_hotkeys": active_hotkeys,
        "inputs": dict(inputs), "formula": FORMULA, "frozen_at": utc_text(timestamp),
    }
    atomic_json(work_dir / "caps" / f"{epoch}.json", record, private=True)
    return record


def public(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "epoch": record["epoch"], "cap": record["cap"], "capacity_tasks": record["capacity_tasks"],
        "active_hotkeys": record["active_hotkeys"], "formula": record["formula"], "inputs": dict(record["inputs"]),
    }
