"""Online IRT task sampler: the ``TaskSampler`` seam.

Model. Each task has a difficulty and each model version has an ability, on
one logit scale (a Rasch model):

    P(model passes task) = 1 / (1 + exp(difficulty - ability))

Draw. Each pool task has a whole-number sampling mass in the snapshot. With
the Fisher rule, a task gets mass in proportion to the Fisher information
``p * (1 - p)`` of its predicted pass rate under the Rasch model. The
sqrt-Fisher rule uses ``sqrt(p * (1 - p))``. Both mix their preference with
a uniform share over every pool task, then round to integer masses. With
the band rule, a task gets more
mass when the head's predicted pass rate is inside the middle band, and
``explore_frac`` of all mass is shared by the tasks seen in fewer than
``explore_below`` rounds. Every task keeps a mass above zero: there is no
hidden or excluded task. The draw takes exactly ``n_tasks`` tasks with
inclusion probability in proportion to mass (systematic sampling on a
seed-ordered pool). It uses only the published masses, SHA-256 and
whole-number arithmetic, so anyone can rebuild it from the snapshot and the
round seed with ``draw_from_snapshot``.

Update. After a round, each model's ability is fitted from its own rollouts
and each drawn task's difficulty is refitted from every model's rollout on
it. Difficulties are keyed by task only, so they stay valid when the head
changes. A new head starts from the ability its winning LoRA showed.

The snapshot returned by ``public_snapshot`` is also the whole state:
``IRTSampler(pool, params, snapshot=...)`` continues exactly where it was.
This module does no I/O. The controller stores the snapshot.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

from .canonical import digest, sha256_hex
from .errors import ContractError, VerifyError
from .params import Params
from .sampling import draw_rank
from .types import CAP, FAIL, NULL_MODEL, PASS, Draw, HeadVersion, check_outcomes

SCHEMA = "affine-irt-v1"
MODEL = "rasch: P(pass) = 1 / (1 + exp(difficulty - ability))"
BODY_KEYS = ("schema", "model", "head_version_id", "rounds", "updates", "config", "heads", "loras", "tasks")
TASK_COLUMNS = ("difficulty", "information", "rounds_seen", "sampling_mass")
DRAW_RULE = (
    "pi(task) = min(1, n_tasks * mass / total mass), fixed size n_tasks, systematic sampling on the pool "
    "ordered by sha256('<round_seed>|irt-order|<task_id>'), start = sha256('<round_seed>|irt-start') mod "
    "total mass; published weight = 1 / pi; evaluation order by sha256('<round_seed>|draw|<task_id>')"
)

MASS_SCALE = 10**12  # masses are whole numbers on this scale, so the draw needs no float
LIMIT = 8.0  # abilities and difficulties stay inside [-LIMIT, LIMIT], so nothing can become infinite
PRIOR_PRECISION = 0.25  # a new task or model starts as Normal(mean, sd 2) in logits
BAND_FLOOR = 0.25  # relative mass of a task far outside the middle band; keeps weights within 4x
EXPLORE_BELOW = 3  # a task seen in fewer rounds than this is "barely seen"
SAME_SEED_CORRELATION = 0.5  # placeholder: assumed correlation of two models' results on one task and seed
SWEEPS = 3  # ability and difficulty fits alternate this many times per round
NEWTON_STEPS = 40
KEEP_UPDATES = 16  # how many recent update stamps are kept to make a repeated update a no-op
KEEP_HEADS = 64  # how many head versions keep an ability record
DECIMALS = 6

FISHER_MASS_RULE = (
    "q(task) = p * (1 - p), p = predicted_pass(head_ability, difficulty); "
    f"mass = max(1, round(((1 - uniform_frac) * q / sum(q) + uniform_frac / pool_size) * {MASS_SCALE})); "
    f"the uniform share applies to every pool task, and every mass is an integer above zero; MASS_SCALE = {MASS_SCALE}"
)
SQRT_FISHER_MASS_RULE = (
    "q(task) = sqrt(p * (1 - p)), p = predicted_pass(head_ability, difficulty); "
    f"mass = max(1, round(((1 - uniform_frac) * q / sum(q) + uniform_frac / pool_size) * {MASS_SCALE})); "
    f"the uniform share applies to every pool task, and every mass is an integer above zero; MASS_SCALE = {MASS_SCALE}"
)
MASS_RULES = {
    "band": (
        "band_preference(predicted_pass(head_ability, difficulty), middle_band, band_floor); "
        "explore_frac of the mass is shared by tasks seen in fewer than explore_below rounds; "
        f"every mass is an integer above zero; MASS_SCALE = {MASS_SCALE}"
    ),
    "fisher": FISHER_MASS_RULE,
    "sqrt_fisher": SQRT_FISHER_MASS_RULE,
}


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def _hash(round_seed: str, *parts: str) -> str:
    return sha256_hex("|".join((round_seed, *parts)).encode("utf-8"))


def predicted_pass(ability: float, difficulty: float) -> float:
    return 1.0 / (1.0 + math.exp(difficulty - ability))


def band_preference(pass_rate: float, band: Sequence[float], floor: float) -> float:
    """1.0 inside the band, falling in a straight line to ``floor`` at pass rate 0 or 1."""
    low, high = band
    if pass_rate < low:
        inside = pass_rate / low
    elif pass_rate > high:
        inside = (1.0 - pass_rate) / (1.0 - high)
    else:
        inside = 1.0
    return floor + (1.0 - floor) * inside


def fisher_preference(pass_rate: float) -> float:
    """Fisher information of one binary outcome under the Rasch model."""
    return pass_rate * (1.0 - pass_rate)


def fisher_masses(tasks: Mapping[str, Sequence[float]], head_ability: float, uniform_frac: float, sqrt: bool = False) -> dict[str, int]:
    """Integer masses for Fisher (unchanged) or sqrt-Fisher preference."""
    information = {task: fisher_preference(predicted_pass(head_ability, row[0])) for task, row in tasks.items()}
    if sqrt:
        information = {task: math.sqrt(q) for task, q in information.items()}
    total = math.fsum(information.values())
    return {
        task: max(1, round(((1.0 - uniform_frac) * q / total + uniform_frac / len(tasks)) * MASS_SCALE))
        for task, q in information.items()
    }


def band_masses(tasks: Mapping[str, Sequence[float]], head_ability: float, config: Mapping[str, Any]) -> dict[str, int]:
    """Whole-number mass of each task: middle-band preference plus the exploration share."""
    prefer = {
        task: band_preference(predicted_pass(head_ability, row[0]), config["middle_band"], config["band_floor"])
        for task, row in tasks.items()
    }
    fresh = sum(row[2] < config["explore_below"] for row in tasks.values())
    share = config["explore_frac"] if fresh else 0.0
    total = math.fsum(prefer.values())
    masses = {}
    for task, row in tasks.items():
        part = (1.0 - share) * prefer[task] / total
        if row[2] < config["explore_below"]:
            part += share / fresh
        masses[task] = max(1, round(part * MASS_SCALE))
    return masses


def sampling_masses(tasks: Mapping[str, Sequence[float]], head_ability: float, config: Mapping[str, Any]) -> dict[str, int]:
    """Whole-number masses under the configured draw rule."""
    if config.get("draw_rule") in ("fisher", "sqrt_fisher"):
        return fisher_masses(tasks, head_ability, config["uniform_frac"], sqrt=config["draw_rule"] == "sqrt_fisher")
    return band_masses(tasks, head_ability, config)


def draw_from_masses(masses: Mapping[str, int], snapshot_id: str, round_seed: str, n_tasks: int) -> Draw:
    """Fixed-size draw with inclusion probability min(1, n * mass / total). Whole numbers and SHA-256 only."""
    _need(n_tasks >= 1 and bool(masses), "a draw needs a pool and at least one task")
    _need(all(type(m) is int and m >= 1 for m in masses.values()), "every sampling mass must be a whole number above zero")
    take = min(n_tasks, len(masses))
    by_mass = sorted(masses, key=lambda task: (-masses[task], task))
    total = sum(masses.values())
    weights: dict[str, float] = {}
    sure = 0
    # A task whose share reaches one whole slot is taken for certain; the rest share the other slots.
    while sure < take and (take - sure) * masses[by_mass[sure]] >= total:
        weights[by_mass[sure]] = 1.0
        total -= masses[by_mass[sure]]
        sure += 1
    slots = take - sure
    if slots:
        # One point every ``total`` units along the seed-ordered pool; a task is drawn when a point falls on it.
        start = int(_hash(round_seed, "irt-start"), 16) % total
        edge = 0
        for task in sorted(by_mass[sure:], key=lambda task: _hash(round_seed, "irt-order", task)):
            points_before = -((start - edge) // total)
            edge += slots * masses[task]
            if -((start - edge) // total) > points_before:
                weights[task] = total / (slots * masses[task])
    chosen = sorted(weights, key=lambda task: draw_rank(round_seed, task))
    return Draw(snapshot_id, round_seed, tuple(chosen), tuple(weights[task] for task in chosen))


def snapshot_digest(snapshot: Mapping[str, Any]) -> str:
    return "irt-" + digest({key: snapshot[key] for key in BODY_KEYS} | ({"rotation": snapshot["rotation"]} if "rotation" in snapshot else {}))


def verify_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Fail closed unless the snapshot is of this schema and its id is the hash of its content."""
    if snapshot.get("schema") != SCHEMA or any(key not in snapshot for key in BODY_KEYS):
        raise VerifyError("not an affine-irt-v1 snapshot")
    if snapshot.get("snapshot_id") != snapshot_digest(snapshot):
        raise VerifyError("IRT snapshot id does not match its content")


def draw_from_snapshot(snapshot: Mapping[str, Any], round_seed: str, n_tasks: int) -> Draw:
    """Rebuild a round's draw from a published snapshot and the public round seed."""
    verify_snapshot(snapshot)
    masses = {task: row[3] for task, row in snapshot["tasks"].items()}
    return draw_from_masses(masses, snapshot["snapshot_id"], round_seed, n_tasks)


def fit(sign: int, mean: float, precision: float, rows: Sequence[tuple[float, float, float]]) -> tuple[float, float]:
    """Most likely value of one ability (sign +1) or one difficulty (sign -1), and its information.

    ``rows`` holds (the other side's value, passed 0 or 1, weight). The start
    is Normal(mean, 1 / precision). Newton steps on a concave function, with
    the step and the value bounded, so the result is always finite.
    """
    value, curve = mean, precision
    for _ in range(NEWTON_STEPS):
        slope, curve = -precision * (value - mean), precision
        for other, passed, weight in rows:
            p = predicted_pass(value, other) if sign > 0 else predicted_pass(other, value)
            slope += sign * weight * (passed - p)
            curve += weight * p * (1.0 - p)
        moved = max(-LIMIT, min(LIMIT, value + max(-1.0, min(1.0, slope / curve))))
        done, value = abs(moved - value) < 1e-9, moved
        if done:
            break
    return round(value, DECIMALS), round(curve, DECIMALS)


class IRTSampler:
    """``TaskSampler`` with a shared IRT table. Only the controller calls ``update``."""

    def __init__(
        self,
        pool: Sequence[str],
        params: Params | None = None,
        snapshot: Mapping[str, Any] | None = None,
        *,
        initial: Mapping[str, float | Sequence[float]] | None = None,
        rotation: Mapping[str, Any] | None = None,
    ) -> None:
        _need(bool(pool) and len(set(pool)) == len(pool), "the task pool must be non-empty with no repeated task")
        params = params or Params()
        if params.irt_draw_rule in ("fisher", "sqrt_fisher"):
            config = {
                "draw_rule": params.irt_draw_rule,
                "uniform_frac": float(params.irt_explore_frac),
                "prior_precision": PRIOR_PRECISION,
                "same_seed_correlation": SAME_SEED_CORRELATION,
            }
        else:
            config = {
                "middle_band": [float(edge) for edge in params.irt_middle_band],  # always decimals: one JSON form
                "explore_frac": float(params.irt_explore_frac),
                "explore_below": EXPLORE_BELOW,
                "band_floor": BAND_FLOOR,
                "prior_precision": PRIOR_PRECISION,
                "same_seed_correlation": SAME_SEED_CORRELATION,
            }
        self._last_draw: dict[str, Any] | None = None
        self._rotation = copy.deepcopy(rotation)
        if snapshot is not None:
            verify_snapshot(snapshot)
            body = copy.deepcopy({key: snapshot[key] for key in BODY_KEYS})
            # Rotation is a body field only when enabled, so snapshot_id seals
            # it. Old snapshots, and every off-mode snapshot, have no such key.
            saved_rotation = body.get("rotation")
            self._rotation = saved_rotation if saved_rotation is not None else self._rotation
            if body["config"] == config and set(body["tasks"]) == set(pool) and (rotation is None or rotation == saved_rotation):
                self._body, self._id = body, snapshot["snapshot_id"]
                return
        else:
            body = {"schema": SCHEMA, "model": MODEL, "head_version_id": "", "rounds": 0, "updates": [], "heads": {}, "loras": {}, "tasks": {}}
        # A normal new pool task starts at the old prior. An admitted task may
        # explicitly start at head ability, giving p=.5 and maximum Fisher q.
        # A returning task supplies its full archived row, not a fresh prior.
        def start_row(task: str) -> Sequence[float]:
            if initial and task in initial:
                row = initial[task]
                if isinstance(row, (int, float)):
                    return (round(row, DECIMALS), PRIOR_PRECISION, 0)
                return (round(float(row[0]), DECIMALS), float(row[1]), int(row[2]))
            return (0.0, PRIOR_PRECISION, 0)

        body["tasks"] = {
            task: list(body["tasks"].get(task, start_row(task)))[:3]
            for task in sorted(pool)
        }
        if rotation is not None:
            body["rotation"] = copy.deepcopy(rotation)
        else:
            body.pop("rotation", None)
        body["config"] = config
        self._commit(body)

    def _commit(self, body: dict[str, Any]) -> None:
        """Set the masses for the tagged head, then swap in the new state in one step."""
        ability = body["heads"].get(body["head_version_id"], {}).get("ability", 0.0)
        masses = sampling_masses(body["tasks"], ability, body["config"])
        body["tasks"] = {task: [*row[:3], masses[task]] for task, row in body["tasks"].items()}
        self._body, self._id = body, snapshot_digest(body)
        self._last_draw = None

    def snapshot_id(self) -> str:
        return self._id

    def draw(self, head: HeadVersion, round_seed: str, n_tasks: int) -> Draw:
        """A pure function of the snapshot and the seed. It never changes the table.

        The masses favour the head named in the snapshot. A draw for another
        head still uses them: the weights stay exact, only less well aimed.
        """
        masses = {task: row[3] for task, row in self._body["tasks"].items()}
        draw = draw_from_masses(masses, self._id, round_seed, n_tasks)
        self._last_draw = {"round_seed": round_seed, "n_tasks": len(draw.task_ids), "for_head_version_id": head.version_id}
        return draw

    def update(
        self,
        head: HeadVersion,
        draw: Draw,
        outcomes: Mapping[str, Sequence[str]],
        *,
        winner_lora_hash: str = "",
        new_head_version_id: str = "",
    ) -> str:
        n = len(draw.task_ids)
        for key, texts in outcomes.items():
            check_outcomes(tuple(texts), n, f"IRT update, model {key[:12]}")
        _need(bool(winner_lora_hash) == bool(new_head_version_id), "a winner and its new head version come together")
        _need(not winner_lora_hash or winner_lora_hash in outcomes, "the winning LoRA has no outcomes")
        # One draw is one round, and a round teaches the table once. The stamp names the draw only,
        # so a second call for the same draw (a replay after a crash) changes nothing, whatever else it carries.
        stamp = digest({"snapshot_id": draw.snapshot_id, "round_seed": draw.round_seed, "task_ids": list(draw.task_ids)})
        old = self._body
        if stamp in old["updates"]:
            return self._id

        # Only resolved rollouts teach anything. A cap hit is not a pass. An infrastructure
        # error or a rollout that was not run says nothing about the task or the model.
        resolved = {PASS: 1.0, FAIL: 0.0, CAP: 0.0}
        tasks = old["tasks"]
        by_model: dict[str, list[tuple[str, float]]] = {}
        by_task: dict[str, list[tuple[str, float]]] = {}
        for key, texts in outcomes.items():
            for text in texts:
                for task, code in zip(draw.task_ids, text):
                    if code in resolved and task in tasks:
                        by_model.setdefault(key, []).append((task, resolved[code]))
                        by_task.setdefault(task, []).append((key, resolved[code]))

        heads = dict(old["heads"])
        known = heads.get(head.version_id) or (heads.get(head.parent_version_id or "") if head.folded else None)
        head_start = (known["ability"], known["information"]) if known else (0.0, PRIOR_PRECISION)
        start = {key: head_start if key == NULL_MODEL else (head_start[0], PRIOR_PRECISION) for key in by_model}
        ability = {key: (mean, precision) for key, (mean, precision) in start.items()}
        difficulty = {task: (tasks[task][0], tasks[task][1]) for task in by_task}
        rho = old["config"]["same_seed_correlation"]
        for _ in range(SWEEPS):
            for key, rows in by_model.items():
                ability[key] = fit(+1, *start[key], [(difficulty[task][0], passed, 1.0) for task, passed in rows])
            for task, rows in by_task.items():
                weight = 1.0 / (1.0 + (len(rows) - 1) * rho)
                difficulty[task] = fit(-1, tasks[task][0], tasks[task][1], [(ability[key][0], passed, weight) for key, passed in rows])

        rounds = old["rounds"] + 1

        def record(key: str, base: Mapping[str, Any] | None = None) -> dict[str, Any]:
            value, information = ability.get(key, head_start)
            rows = by_model.get(key, ())
            return {
                "ability": value,
                "information": information,
                "rollouts": (base or {}).get("rollouts", 0) + len(rows),
                "passes": (base or {}).get("passes", 0) + int(sum(passed for _, passed in rows)),
                "round": rounds,
            }

        if NULL_MODEL in by_model or known:  # a folded head is the same model as its parent: it keeps the record
            heads[head.version_id] = record(NULL_MODEL, known)
        loras = {key: record(key) for key in by_model if key != NULL_MODEL}
        if winner_lora_hash:
            heads[new_head_version_id] = record(winner_lora_hash)
        tagged = new_head_version_id or head.version_id
        newest = sorted(heads, key=lambda key: (key != tagged, -heads[key]["round"], key))[:KEEP_HEADS]
        body = {
            **old,
            "head_version_id": tagged,
            "rounds": rounds,
            "updates": [*old["updates"], stamp][-KEEP_UPDATES:],
            "heads": {key: heads[key] for key in sorted(newest)},
            "loras": loras,
            "tasks": {task: [*difficulty[task], row[2] + 1] if task in difficulty else row[:3] for task, row in tasks.items()},
        }
        self._commit(body)
        return self._id

    def public_snapshot(self) -> Mapping[str, Any]:
        """The published snapshot. Its ``snapshot_id`` is the hash of the ``BODY_KEYS`` fields."""
        body = copy.deepcopy(self._body)
        head = body["heads"].get(body["head_version_id"], {})
        return {
            **body,
            "snapshot_id": self._id,
            "pool_size": len(body["tasks"]),
            "head_ability": head.get("ability", 0.0),
            "mass_total": sum(row[3] for row in body["tasks"].values()),
            "mass_rule": MASS_RULES[body["config"].get("draw_rule", "band")],
            "task_columns": list(TASK_COLUMNS),
            "draw_rule": DRAW_RULE,
            "draw": copy.deepcopy(self._last_draw),  # the last draw made from this snapshot: round seed and n_tasks
        }
