"""The six-term score in conditional cross-scoring form.

Per turn x, with teacher C and miner A:
  teacher rollouts  R_C = {(z_C^i, y_C^i)}  i = 1..n
  miner rollouts    R_A = {(z_A^j, y_A^j)}  j = 1..n

Every term is a difference of teacher-forced per-byte logprobs (higher Δ = worse miner):
  Δ0 (causality, miner-independent): lpC(y_C | z_C) − lpC(y_C | ∅)
  Δ1 behavioral:        lpC(y_C | z_C) − lpA(y_C | z_A)
  Δ2 thought suff.:     lpC(y_C | z_C) − lpC(y_C | z_A)
  Δ3 decoder fidelity:  lpC(y_C | z_C) − lpA(y_C | z_C)
  Δ4 faithfulness:      lpA(y_A | z_A) − lpC(y_A | z_A)   (abs in aggregation)
  Δ5 receptivity:       lpA(y_A | z_A) − lpA(y_A | z_C)   (abs in aggregation)

Pairing (i, j) is diagonal (i-th teacher rollout with i-th miner rollout) to keep the
call count linear. All lp* come from echo+logprobs teacher forcing, never from
tempered sampling logprobs.
"""

import asyncio

from .client import VllmModel
from .priors import PRIOR_BANK

EMPTY_THOUGHTS = ""


async def teacher_reference(teacher: VllmModel, prefix: list[dict], n: int,
                            temperature: float, max_thought: int, max_action: int):
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


async def _bank_lifts(teacher: VllmModel, prefix: list[dict],
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
                      rollouts: list[tuple[str, str]] | None = None,
                      score_bank: bool = False) -> dict:
    """Compute Δ1..Δ5 (per-byte) for one miner on one turn.

    `rollouts` lets callers supply (z, y) pairs directly (e.g. red-team probes
    that rewrite the thought channel of previously sampled rollouts).
    If `score_bank`, also records per-pair L2_bank and miner-level bank_frac
    (fraction of pairs with L2_bank > 0) for the production positivity gate.
    """
    if rollouts is None:
        rollouts = await asyncio.gather(*[
            miner.sample(prefix, temperature, max_thought + max_action)
            for _ in range(n)
        ])
    rollouts = [(z, y) for z, y in rollouts if y]
    if not rollouts or not ref:
        return {"valid": False}

    m = min(len(ref), len(rollouts))
    ref, rollouts = ref[:m], rollouts[:m]

    # Ten forced-logprob measurements per pair. The raw components (not just
    # the Δ aggregates) are recorded so any scoring rule -- raw conditionals,
    # empty-baseline lifts, teacher-anchored action quality -- can be evaluated
    # offline without re-running GPU work (decision after RT-2 showed raw
    # conditionals reward thought suppression).
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
            _bank_lifts(teacher, prefix, ref[i]["y"], rollouts[i][0])
            for i in range(m)
        ])

    d = {f"D{k}": [] for k in range(1, 6)}
    pairs = []
    for i in range(m):
        lp = {name: res[i * len(calls) + j]["lp_per_byte"]
              for j, (name, *_ ) in enumerate(calls)}
        lp["lpC_yc_zc"] = ref[i]["lp_own"]
        lp["lpC_yc_e"] = ref[i]["lp_empty"]
        # Keep full texts — a 2k cap previously truncated 70%+ of late-king
        # thoughts and biased offline Λ2_bank rescored (2026-08-02).
        lp["z_a"] = rollouts[i][0]
        lp["y_a"] = rollouts[i][1]
        if bank_vals is not None:
            lp["L2_bank"] = bank_vals[i]
        pairs.append(lp)
        d["D1"].append(lp["lpC_yc_zc"] - lp["lpA_yc_za"])
        d["D2"].append(lp["lpC_yc_zc"] - lp["lpC_yc_za"])
        d["D3"].append(lp["lpC_yc_zc"] - lp["lpA_yc_zc"])
        d["D4"].append(abs(lp["lpA_ya_za"] - lp["lpC_ya_za"]))
        d["D5"].append(abs(lp["lpA_ya_za"] - lp["lpA_ya_zc"]))

    out = {k: sum(v) / len(v) for k, v in d.items()}
    out["pairs"] = pairs
    out["valid"] = True
    out["n_pairs"] = m
    if bank_vals is not None:
        out["bank_frac"] = sum(1.0 if v > 0 else 0.0 for v in bank_vals) / m
        out["L2_bank"] = sum(bank_vals) / m
    return out
