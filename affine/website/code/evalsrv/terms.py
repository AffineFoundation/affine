"""Per-turn instrumentation: teacher references + the ten forced-logprob calls.

Per turn x, with teacher C and miner A:
  teacher rollouts  R_C = {(z_C^i, y_C^i)}  i = 1..n
  miner rollouts    R_A = {(z_A^j, y_A^j)}  j = 1..n

Pairing (i, j) is diagonal to keep the call count linear. All lp* come from
echo+logprobs teacher forcing, never from tempered sampling logprobs. The raw
per-pair components are recorded so any scoring rule can be recomputed
offline (audit) without re-running GPU work.
"""

from __future__ import annotations

import asyncio

from affine.priors import PRIOR_BANK
from affine.score import eta

from .vllm_client import VllmModel

EMPTY_THOUGHTS = ""


async def teacher_reference(teacher: VllmModel, prefix: list[dict], n: int,
                            temperature: float, max_thought: int,
                            max_action: int) -> list[dict]:
    """Sample teacher rollouts once per turn; reused across all miners.

    Returns list of dicts with z, y, lp_own (lpC(y|z_C)) and lp_empty (lpC(y|∅)).
    """
    rollouts = await asyncio.gather(*[
        teacher.sample(prefix, temperature, max_thought + max_action)
        for _ in range(n)
    ])
    rollouts = [(z, y) for z, y in rollouts if y]
    scored = await asyncio.gather(*[
        teacher.score_action(prefix, z, y) for z, y in rollouts
    ], *[
        teacher.score_action(prefix, EMPTY_THOUGHTS, y) for z, y in rollouts
    ])
    k = len(rollouts)
    return [
        {"z": z, "y": y,
         "lp_own": scored[i]["lp_per_byte"],
         "lp_empty": scored[k + i]["lp_per_byte"]}
        for i, (z, y) in enumerate(rollouts)
    ]


async def _bank_lift(teacher: VllmModel, prefix: list[dict],
                     y_c: str, z_a: str) -> float:
    """Λ2_bank = lpC(y_C|z_A) − max_k lpC(y_C|prior_k)."""
    keys = list(PRIOR_BANK)
    zs = [z_a] + [PRIOR_BANK[k] for k in keys]
    scores = await asyncio.gather(*[
        teacher.score_action(prefix, z, y_c) for z in zs
    ])
    return scores[0]["lp_per_byte"] - max(s["lp_per_byte"] for s in scores[1:])


async def miner_terms(teacher: VllmModel, miner: VllmModel, prefix: list[dict],
                      ref: list[dict], n: int, temperature: float,
                      max_thought: int, max_action: int,
                      score_bank: bool = True) -> dict:
    """Compute the instrumented pair record for one miner on one turn."""
    rollouts = await asyncio.gather(*[
        miner.sample(prefix, temperature, max_thought + max_action)
        for _ in range(n)
    ])
    rollouts = [(z, y) for z, y in rollouts if y]
    if not rollouts or not ref:
        return {"valid": False}

    m = min(len(ref), len(rollouts))
    ref, rollouts = ref[:m], rollouts[:m]

    calls = [
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
    tasks = []
    for i in range(m):
        ctx = {"zc": ref[i]["z"], "yc": ref[i]["y"],
               "za": rollouts[i][0], "ya": rollouts[i][1], "": EMPTY_THOUGHTS}
        for _, model, z_key, y_key in calls:
            mdl = miner if model == "miner" else teacher
            tasks.append(mdl.score_action(prefix, ctx[z_key], ctx[y_key]))
    res = await asyncio.gather(*tasks)

    bank_vals = None
    if score_bank:
        bank_vals = await asyncio.gather(*[
            _bank_lift(teacher, prefix, ref[i]["y"], rollouts[i][0])
            for i in range(m)
        ])

    pairs = []
    for i in range(m):
        lp = {name: res[i * len(calls) + j]["lp_per_byte"]
              for j, (name, *_) in enumerate(calls)}
        lp["lpC_yc_zc"] = ref[i]["lp_own"]
        lp["lpC_yc_e"] = ref[i]["lp_empty"]
        # Full texts kept: truncation biases offline bank rescoring.
        lp["z_a"] = rollouts[i][0]
        lp["y_a"] = rollouts[i][1]
        # η stamped here so artifacts carry it from day one (denominator is
        # the teacher-ref lp_own already on the pair — no extra echo).
        lp["eta"] = eta(lp)
        if bank_vals is not None:
            lp["L2_bank"] = bank_vals[i]
        pairs.append(lp)

    out: dict = {"pairs": pairs, "valid": True, "n_pairs": m}
    if bank_vals is not None:
        out["bank_frac"] = sum(1.0 if v > 0 else 0.0 for v in bank_vals) / m
        out["L2_bank"] = sum(bank_vals) / m
    return out
