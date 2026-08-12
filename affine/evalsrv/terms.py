"""Per-turn instrumentation: teacher references + forced-logprob calls.

Per turn x, with teacher C and miner A:
  teacher rollouts  R_C = {(z_C^i, y_C^i)}  i = 1..n
  miner rollouts    R_A = {(z_A^j, y_A^j)}  j = 1..n

Pairing (i, j) is diagonal to keep the call count linear. All lp* come from
echo+logprobs teacher forcing, never from tempered sampling logprobs.

Production (Reason-only): each pair records z_A/y_A plus the components the
score needs — lpC(y_C|z_A) and the teacher-ref lpC(y_C|z_C)/lpC(y_C|∅).
Optional full_terms / score_bank keep the retired S* v2 telemetry echoes for
offline replay; they are off in live duels.
"""

from __future__ import annotations

import asyncio

from affine.priors import PRIOR_BANK
from affine.score import eta

from .vllm_client import ModelPool, VllmModel

TeacherClient = VllmModel | ModelPool

EMPTY_THOUGHTS = ""

# Full (legacy telemetry) echo set — unused when reason_only=True.
_FULL_CALLS = [
    ("lpA_yc_za", "miner", "za", "yc"),
    ("lpC_yc_za", "teacher", "za", "yc"),
    ("lpA_yc_zc", "miner", "zc", "yc"),
    ("lpA_yc_e", "miner", "", "yc"),
    ("lpA_ya_za", "miner", "za", "ya"),
    ("lpC_ya_za", "teacher", "za", "ya"),
    ("lpA_ya_zc", "miner", "zc", "ya"),
    ("lpA_ya_e", "miner", "", "ya"),
    ("lpC_ya_e", "teacher", "", "ya"),
    ("lpC_ya_zc", "teacher", "zc", "ya"),
]

# Score path: Reason = lpC(y_C|z_A) − lpC(y_C|∅); empty/own come from refs.
_REASON_CALLS = [
    ("lpC_yc_za", "teacher", "za", "yc"),
]


async def sample_teacher_rollouts(
        teacher: TeacherClient, prefix: list[dict], n: int,
        temperature: float, max_thought: int, max_action: int, *,
        sticky_key: str | None = None) -> list[tuple[str, str]]:
    """Sample teacher (z, y) only — no forced-logprob echoes yet."""
    rollouts = await asyncio.gather(*[
        teacher.sample(prefix, temperature, max_thought + max_action,
                       sticky_key=sticky_key)
        for _ in range(n)
    ])
    return [(z, y) for z, y in rollouts if y]


async def score_teacher_rollouts(
        teacher: TeacherClient, prefix: list[dict],
        rollouts: list[tuple[str, str]], *,
        sticky_key: str | None = None) -> list[dict]:
    """lpC(y|z_C) and lpC(y|∅) for already-sampled teacher rollouts."""
    if not rollouts:
        return []
    scored = await asyncio.gather(*[
        teacher.score_action(prefix, z, y, sticky_key=sticky_key)
        for z, y in rollouts
    ], *[
        teacher.score_action(prefix, EMPTY_THOUGHTS, y, sticky_key=sticky_key)
        for z, y in rollouts
    ])
    k = len(rollouts)
    return [
        {"z": z, "y": y,
         "lp_own": scored[i]["lp_per_byte"],
         "lp_empty": scored[k + i]["lp_per_byte"]}
        for i, (z, y) in enumerate(rollouts)
    ]


async def teacher_reference(teacher: TeacherClient, prefix: list[dict], n: int,
                            temperature: float, max_thought: int,
                            max_action: int, *,
                            sticky_key: str | None = None) -> list[dict]:
    """Sample teacher rollouts once per turn; reused across all miners.

    Returns list of dicts with z, y, lp_own (lpC(y|z_C)) and lp_empty (lpC(y|∅)).
    sticky_key pins all calls for this turn to one teacher replica (prefix cache).
    """
    rollouts = await sample_teacher_rollouts(
        teacher, prefix, n, temperature, max_thought, max_action,
        sticky_key=sticky_key)
    return await score_teacher_rollouts(
        teacher, prefix, rollouts, sticky_key=sticky_key)


async def sample_miner_rollouts(
        miner: VllmModel, prefix: list[dict], n: int,
        temperature: float, max_thought: int, max_action: int
        ) -> list[tuple[str, str]]:
    """Sample miner (z, y) rollouts; filter empty actions."""
    rollouts = await asyncio.gather(*[
        miner.sample(prefix, temperature, max_thought + max_action)
        for _ in range(n)
    ])
    return [(z, y) for z, y in rollouts if y]


async def _bank_lift(teacher: TeacherClient, prefix: list[dict],
                     y_c: str, z_a: str, *,
                     sticky_key: str | None = None) -> float:
    """Λ2_bank = lpC(y_C|z_A) − max_k lpC(y_C|prior_k)."""
    keys = list(PRIOR_BANK)
    zs = [z_a] + [PRIOR_BANK[k] for k in keys]
    scores = await asyncio.gather(*[
        teacher.score_action(prefix, z, y_c, sticky_key=sticky_key) for z in zs
    ])
    return scores[0]["lp_per_byte"] - max(s["lp_per_byte"] for s in scores[1:])


async def miner_terms(teacher: TeacherClient, miner: VllmModel, prefix: list[dict],
                      ref: list[dict], n: int, temperature: float,
                      max_thought: int, max_action: int,
                      score_bank: bool = False,
                      reason_only: bool = True, *,
                      sticky_key: str | None = None,
                      rollouts: list[tuple[str, str]] | None = None) -> dict:
    """Compute the pair record for one miner on one turn.

    reason_only (production): sample the miner, echo lpC(y_C|z_A) on the
    teacher, stamp η from teacher-ref denominators. No lpA / bank GPU work.
    sticky_key pins teacher echoes for this turn to one replica (prefix cache).

    If ``rollouts`` is provided (already sampled), skip miner sampling — used
    when the caller overlapped miner sample with teacher ref scoring.
    """
    if rollouts is None:
        rollouts = await sample_miner_rollouts(
            miner, prefix, n, temperature, max_thought, max_action)
    rollouts = [(z, y) for z, y in rollouts if y]
    if not rollouts or not ref:
        return {"valid": False}

    m = min(len(ref), len(rollouts))
    ref, rollouts = ref[:m], rollouts[:m]

    calls = _REASON_CALLS if reason_only else _FULL_CALLS
    tasks = []
    for i in range(m):
        ctx = {"zc": ref[i]["z"], "yc": ref[i]["y"],
               "za": rollouts[i][0], "ya": rollouts[i][1], "": EMPTY_THOUGHTS}
        for _, model, z_key, y_key in calls:
            if model == "miner":
                tasks.append(miner.score_action(prefix, ctx[z_key], ctx[y_key]))
            else:
                tasks.append(teacher.score_action(
                    prefix, ctx[z_key], ctx[y_key], sticky_key=sticky_key))
    res = await asyncio.gather(*tasks)

    bank_vals = None
    if score_bank:
        bank_vals = await asyncio.gather(*[
            _bank_lift(teacher, prefix, ref[i]["y"], rollouts[i][0],
                       sticky_key=sticky_key)
            for i in range(m)
        ])

    pairs = []
    for i in range(m):
        lp = {name: res[i * len(calls) + j]["lp_per_byte"]
              for j, (name, *_) in enumerate(calls)}
        lp["lpC_yc_zc"] = ref[i]["lp_own"]
        lp["lpC_yc_e"] = ref[i]["lp_empty"]
        lp["z_a"] = rollouts[i][0]
        lp["y_a"] = rollouts[i][1]
        lp["eta"] = eta(lp)
        if bank_vals is not None:
            lp["L2_bank"] = bank_vals[i]
        pairs.append(lp)

    out: dict = {"pairs": pairs, "valid": True, "n_pairs": m}
    if bank_vals is not None:
        out["bank_frac"] = sum(1.0 if v > 0 else 0.0 for v in bank_vals) / m
        out["L2_bank"] = sum(bank_vals) / m
    return out
