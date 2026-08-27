"""Per-turn instrumentation: teacher references + forced-logprob calls.

Per turn x, with teacher C and miner A:
  teacher rollouts  R_C = {(z_C^i, y_C^i)}  i = 1..k   (k = n_teacher_samples)
  miner rollouts    R_A = {(z_A^j, y_A^j)}  j = 1..n_miner_samples

Production pairing (Reason-only, v4 2026-08-17): every teacher reference is
kept and scored against the miner rollout (with 1 miner sample per turn that
is k pairs sharing one z_A/y_A) — turn_reason() then takes the tempered
log-mean-exp over the k per-ref Reasons. B-license echoes (lpC(y_A|z_A),
lpC(y_A|∅)) depend only on the miner rollout, so they run once per distinct
rollout, not once per pair. All lp* come from echo+logprobs teacher forcing,
never from tempered sampling logprobs.

Legacy full_terms / score_bank replay keeps the old diagonal
min(refs, rollouts) truncation so archived telemetry stays comparable; it is
off in live duels.
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

# Score path: Reason = lpC(y_i|z_A) − lpC(y_i|∅); empty/own come from refs.
# One echo per (ref, rollout) pair. B echoes are handled separately (once
# per distinct miner rollout) inside miner_terms.
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
        thought_echo: bool = False,
        sticky_key: str | None = None) -> list[dict]:
    """lpC(y|z_C) and lpC(y|∅) for already-sampled teacher rollouts.

    thought_echo (min(R,G) grounding leg): additionally echo each reference
    thought t_i = lpC(z_C^i|x), stamped as ``lp_thought``. The band the
    miner's thought is judged against is built from these — shared by both
    sides via the RefCache, so it costs k echoes per turn per duel.
    """
    if not rollouts:
        return []
    thought_tasks = ([
        teacher.score_thought(prefix, z, sticky_key=sticky_key)
        for z, _ in rollouts
    ] if thought_echo else [])
    scored = await asyncio.gather(*[
        teacher.score_action(prefix, z, y, sticky_key=sticky_key)
        for z, y in rollouts
    ], *[
        teacher.score_action(prefix, EMPTY_THOUGHTS, y, sticky_key=sticky_key)
        for z, y in rollouts
    ], *thought_tasks)
    k = len(rollouts)
    out = []
    for i, (z, y) in enumerate(rollouts):
        rec = {"z": z, "y": y,
               "lp_own": scored[i]["lp_per_byte"],
               "lp_empty": scored[k + i]["lp_per_byte"]}
        if thought_echo:
            rec["lp_thought"] = scored[2 * k + i]["lp_per_byte"]
        out.append(rec)
    return out


async def teacher_reference(teacher: TeacherClient, prefix: list[dict], n: int,
                            temperature: float, max_thought: int,
                            max_action: int, *,
                            thought_echo: bool = False,
                            sticky_key: str | None = None) -> list[dict]:
    """Sample teacher rollouts once per turn; reused across all miners.

    Returns list of dicts with z, y, lp_own (lpC(y|z_C)) and lp_empty
    (lpC(y|∅)) — plus lp_thought (lpC(z_C|x)) when thought_echo is on.
    sticky_key pins all calls for this turn to one teacher replica (prefix cache).
    """
    rollouts = await sample_teacher_rollouts(
        teacher, prefix, n, temperature, max_thought, max_action,
        sticky_key=sticky_key)
    return await score_teacher_rollouts(
        teacher, prefix, rollouts, thought_echo=thought_echo,
        sticky_key=sticky_key)


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
                      reason_only: bool = True,
                      causality_gate: bool = False,
                      thought_echo: bool = False, *,
                      sticky_key: str | None = None,
                      rollouts: list[tuple[str, str]] | None = None) -> dict:
    """Compute the pair record for one miner on one turn.

    reason_only (production, v4): sample the miner, echo lpC(y_i|z_A) on the
    teacher against EVERY teacher ref (k pairs per turn), stamp η from
    teacher-ref denominators. No lpA / bank GPU work. causality_gate adds
    lpC(y_A|z_A) and lpC(y_A|∅) once per distinct miner rollout (the B
    license does not depend on the ref). thought_echo (min(R,G) grounding
    leg, wvk 10) adds m = lpC(z_A|x) once per distinct miner rollout,
    stamped on each pair as ``lpC_za_x``, with the turn's reference band
    values copied from the refs as ``lpC_zc_x``. Legacy full_terms keeps the
    old diagonal min(refs, rollouts) truncation for replay comparability.
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

    if reason_only:
        # v4 pairing: keep all k refs; cycle miner rollouts across them
        # (with the production 1 miner sample, every ref shares one z_A/y_A).
        midx = [i % len(rollouts) for i in range(len(ref))]
        calls = _REASON_CALLS
    else:
        m = min(len(ref), len(rollouts))
        ref, rollouts = ref[:m], rollouts[:m]
        midx = list(range(m))
        calls = _FULL_CALLS
    n_pairs = len(ref)

    tasks = []
    for i in range(n_pairs):
        ctx = {"zc": ref[i]["z"], "yc": ref[i]["y"],
               "za": rollouts[midx[i]][0], "ya": rollouts[midx[i]][1],
               "": EMPTY_THOUGHTS}
        for _, model, z_key, y_key in calls:
            if model == "miner":
                tasks.append(miner.score_action(prefix, ctx[z_key], ctx[y_key]))
            else:
                tasks.append(teacher.score_action(
                    prefix, ctx[z_key], ctx[y_key], sticky_key=sticky_key))
    # B echoes once per distinct miner rollout (ref-independent).
    b_rollouts = sorted(set(midx)) if (reason_only and causality_gate) else []
    for j in b_rollouts:
        tasks.append(teacher.score_action(
            prefix, rollouts[j][0], rollouts[j][1], sticky_key=sticky_key))
        tasks.append(teacher.score_action(
            prefix, EMPTY_THOUGHTS, rollouts[j][1], sticky_key=sticky_key))
    # Grounding echo m = lpC(z_A|x) once per distinct miner rollout.
    g_rollouts = sorted(set(midx)) if thought_echo else []
    for j in g_rollouts:
        tasks.append(teacher.score_thought(
            prefix, rollouts[j][0], sticky_key=sticky_key))
    res = await asyncio.gather(*tasks)

    b_by_rollout: dict[int, dict[str, float]] = {}
    base = n_pairs * len(calls)
    for t, j in enumerate(b_rollouts):
        b_by_rollout[j] = {
            "lpC_ya_za": res[base + 2 * t]["lp_per_byte"],
            "lpC_ya_e": res[base + 2 * t + 1]["lp_per_byte"],
        }
    m_by_rollout: dict[int, float] = {}
    g_base = base + 2 * len(b_rollouts)
    for t, j in enumerate(g_rollouts):
        m_by_rollout[j] = res[g_base + t]["lp_per_byte"]

    bank_vals = None
    if score_bank:
        bank_vals = await asyncio.gather(*[
            _bank_lift(teacher, prefix, ref[i]["y"], rollouts[midx[i]][0],
                       sticky_key=sticky_key)
            for i in range(n_pairs)
        ])

    pairs = []
    for i in range(n_pairs):
        lp = {name: res[i * len(calls) + j]["lp_per_byte"]
              for j, (name, *_) in enumerate(calls)}
        lp.update(b_by_rollout.get(midx[i], {}))
        lp["lpC_yc_zc"] = ref[i]["lp_own"]
        lp["lpC_yc_e"] = ref[i]["lp_empty"]
        if thought_echo:
            lp["lpC_za_x"] = m_by_rollout.get(midx[i])
            lp["lpC_zc_x"] = ref[i].get("lp_thought")
        lp["z_a"] = rollouts[midx[i]][0]
        lp["y_a"] = rollouts[midx[i]][1]
        lp["eta"] = eta(lp)
        if bank_vals is not None:
            lp["L2_bank"] = bank_vals[i]
        pairs.append(lp)

    out: dict = {"pairs": pairs, "valid": True, "n_pairs": n_pairs}
    if bank_vals is not None:
        out["bank_frac"] = (
            sum(1.0 if v > 0 else 0.0 for v in bank_vals) / n_pairs)
        out["L2_bank"] = sum(bank_vals) / n_pairs
    return out
