"""Duel execution on the eval machine: seeded slice, probe, scoring, verdict.

Slice seeding (chain contract): the duel slice is drawn from the public turn
corpus D with

    seed = blake2b(reveal_block_hash || challenger_hotkey, digest_size=8)

so a miner cannot know their slice before revealing (the block hash resolves
after the commit), and any external auditor can re-derive it from public
inputs. Stratified round-robin over repo×phase strata keeps single bug
families from dominating.

Before burning GPU-hours on the full duel, a cheap injectability probe
rejects checkpoints that cannot play the game at all (no parsable actions,
non-finite forced logprobs) — our analogue of a pretraining subnet's
trainability probe: the asset the network buys must remain promptable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import statistics as st
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .corpus import CorpusSync

from affine.corpus.materialize import stratum_key
from affine.score import (
    duel as score_duel,
    score_miner,
)

from .terms import miner_terms, teacher_reference
from .vllm_client import ModelPool, Served, VllmModel

log = logging.getLogger("evalsrv.dueling")

# Back-compat alias for anything that imported _phase_key.
_phase_key = stratum_key


# -- slice ----------------------------------------------------------------------

def turn_id(rec: dict) -> str:
    if rec.get("turn_id"):
        return str(rec["turn_id"])
    return f"{rec['traj_id']}:{rec['turn_idx']}"


def duel_seed(block_hash: str, hotkey: str) -> int:
    material = block_hash.encode() + hotkey.encode()
    return int.from_bytes(
        hashlib.blake2b(material, digest_size=8).digest(), "little")


def sample_slice(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified sample without replacement (round-robin over strata)."""
    if n >= len(rows):
        rng = random.Random(seed)
        out = list(rows)
        rng.shuffle(out)
        return out
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[stratum_key(r)].append(r)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    # Seed-shuffled strata order. With more strata than n, a fixed (sorted)
    # order means every duel draws from the same alphabetically-first strata
    # forever — measured live: a 267-turn reachable pool out of a 9000-turn
    # corpus, 96-99% slice recurrence, fully predictable (and memorizable)
    # by miners. Shuffling the order by the duel seed makes each duel sample
    # a different strata subset, restoring the whole corpus as the pool.
    keys = sorted(by)
    rng.shuffle(keys)
    out: list[dict] = []
    idx = {k: 0 for k in keys}
    while len(out) < n:
        progress = False
        for k in keys:
            i = idx[k]
            if i < len(by[k]):
                out.append(by[k][i])
                idx[k] = i + 1
                progress = True
                if len(out) >= n:
                    break
        if not progress:
            break
    rng.shuffle(out)
    return out


def slice_digest(turns: list[dict]) -> str:
    h = hashlib.sha256()
    for t in turns:
        h.update(turn_id(t).encode())
        h.update(b"\n")
    return h.hexdigest()


# -- probe -----------------------------------------------------------------------

async def probe_injectable(model: VllmModel, turns: list[dict],
                           temperature: float, max_thought: int,
                           max_action: int, n_probe_turns: int = 3) -> str | None:
    """Cheap fail-fast before the full duel. Returns rejection reason or None.

    A checkpoint passes if, across a few turns, it (a) produces at least one
    rollout with a parsable closed ```bash action, and (b) returns finite
    forced logprobs under thought injection.
    """
    any_action = False
    for rec in turns[:n_probe_turns]:
        prefix = rec["prefix"]
        try:
            z, y = await model.sample(prefix, temperature,
                                      max_thought + max_action)
        except Exception as e:
            return f"probe_sample_failed:{type(e).__name__}:{e}"
        if not y:
            continue
        any_action = True
        try:
            scored = await model.score_action(prefix, z, y)
        except Exception as e:
            return f"probe_force_failed:{type(e).__name__}:{e}"
        if not math.isfinite(scored["lp_per_byte"]):
            return f"probe_nonfinite_logprob:{scored['lp_per_byte']}"
        if scored["n_tokens"] == 0:
            # An empty scored span yields a 0.0 sentinel that would pass the
            # finite check while meaning "nothing was actually scored".
            return "probe_empty_action_span"
    if not any_action:
        return f"probe_no_parsable_action_in_{n_probe_turns}_turns"
    return None


# -- duel -------------------------------------------------------------------------

class RefCache:
    """Teacher rollouts per turn, scoped to a single duel.

    Deliberately NOT persisted across duels: a persistent cache froze y_C per
    turn while artifacts publish it, so recurring turns became known targets
    a miner could SFT-memorize for free L1lift (RT-6). Fresh references per
    duel mean memorizing published refs only pays through genuine
    generalization to the teacher's distribution — i.e. distillation, which
    is exactly what S rewards. Within a duel the cache still dedupes teacher
    sampling so both sides score against identical references (that pairing
    is what the verdict needs)."""

    def __init__(self):
        self.cache: dict[str, list[dict]] = {}
        # Per-turn locks: different turns sample teacher references
        # concurrently; only same-turn callers serialize. A single global
        # lock here would collapse the reference phase to sequential.
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def get_or_sample(self, tid: str, teacher: VllmModel,
                            prefix: list[dict], n: int, temperature: float,
                            max_thought: int, max_action: int) -> list[dict]:
        if tid in self.cache:
            return self.cache[tid]
        async with self._locks[tid]:
            if tid in self.cache:
                return self.cache[tid]
            ref = await teacher_reference(teacher, prefix, n, temperature,
                                          max_thought, max_action)
            self.cache[tid] = ref
            return ref


async def score_side(teacher: VllmModel | ModelPool, miner: VllmModel,
                     turns: list[dict], refs: RefCache, duel_cfg: dict,
                     turn_sem: asyncio.Semaphore, on_progress) -> list[dict]:
    rows: list[dict] = []
    done = 0
    total = len(turns)

    async def one(rec: dict) -> None:
        nonlocal done
        tid = turn_id(rec)
        prefix = rec["prefix"]
        async with turn_sem:
            ref = await refs.get_or_sample(
                tid, teacher, prefix,
                int(duel_cfg["n_teacher_samples"]), float(duel_cfg["temperature"]),
                int(duel_cfg["max_thought_tokens"]), int(duel_cfg["max_action_tokens"]))
            if not ref:
                done += 1
                return
            t = await miner_terms(
                teacher, miner, prefix, ref,
                int(duel_cfg["n_miner_samples"]), float(duel_cfg["temperature"]),
                int(duel_cfg["max_thought_tokens"]), int(duel_cfg["max_action_tokens"]),
                score_bank=True)
        t.update({"turn_id": tid, "miner": miner.cfg.name})
        rows.append(t)
        done += 1
        on_progress(miner.cfg.name, done, total)

    await asyncio.gather(*[one(rec) for rec in turns])
    return rows


def _mean_bank(rows: list[dict]) -> float | None:
    vals = [r["bank_frac"] for r in rows if r.get("valid") and "bank_frac" in r]
    return sum(vals) / len(vals) if vals else None


def _miner_summary(rows: list[dict]) -> dict:
    """Per-side summary: the score (reason) plus measured-not-scored telemetry."""
    s = score_miner(rows, bank_frac=_mean_bank(rows))
    return {
        "reason": s.reason if math.isfinite(s.reason) else None,
        "n_turns": s.n_turns, "n_pairs": s.n_pairs,
        # -- telemetry (recorded for study; never affects score or validity) --
        "gate_pass_rate": s.gate_pass_rate, "bank_frac": s.bank_frac,
        "calib_ratio": s.calib_ratio, "baseline_abs": s.baseline_abs,
        "mean_l1lift": s.mean_l1lift,
        "mean_eta": s.mean_eta,
        "mean_len_z": s.mean_len_z, "mean_len_y": s.mean_len_y,
    }


def _teacher_lengths(refs_used: dict[str, list[dict]]) -> dict:
    """Mean char lengths of the teacher rollouts actually used this duel."""
    zs = [len(r["z"]) for ref in refs_used.values() for r in ref]
    ys = [len(r["y"]) for ref in refs_used.values() for r in ref]
    if not zs:
        return {"mean_len_z": None, "mean_len_y": None}
    return {"mean_len_z": st.mean(map(float, zs)),
            "mean_len_y": st.mean(map(float, ys))}


def _len_deltas(side: dict, teacher: dict) -> None:
    """Attach miner − teacher length deltas to a side summary, in place."""
    for key, out in (("mean_len_z", "len_z_delta"), ("mean_len_y", "len_y_delta")):
        if side.get(key) is not None and teacher.get(key) is not None:
            side[out] = side[key] - teacher[key]
        else:
            side[out] = None


async def run_duel(engine_cfg: dict, turns_path: Path | None,
                   king: Served, challenger: Served,
                   teacher: Served | list[Served],
                   block_hash: str, hotkey: str, corpus_info: dict,
                   on_progress,
                   corpus: "CorpusSync | None" = None) -> tuple[dict, dict]:
    """Full duel. Returns (verdict, artifact).

    The verdict is the small audit summary streamed to the validator. The
    artifact is the full training-grade record — sliced turn ids, teacher
    reference rollouts, and both sides' per-turn pair rows (thoughts/actions
    plus every forced-logprob component) — published post-hoc so miners can
    train on exactly what was scored.

    schema_version>=2: sample the Parquet index via ``corpus``, materialize
    only the drawn turns. schema v1 / ``turns_path``: load flat turns.jsonl.
    """
    duel_cfg = engine_cfg["duel"]
    started = time.monotonic()
    seed = duel_seed(block_hash, hotkey)
    n = int(duel_cfg["n_turns"])
    if corpus is not None and corpus.schema_version >= 2:
        rows = corpus.load_index_rows()
        picked = sample_slice(rows, n, seed)
        turns = corpus.materialize_turns(picked)
    else:
        if turns_path is None:
            raise ValueError("turns_path required for schema_version=1")
        with open(turns_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        turns = sample_slice(rows, n, seed)
    # The manifest hash pins exactly which shard set this duel was scored
    # against — replayable even after shards are retired from the window.
    slice_info = {"seed": seed, "n": len(turns),
                  "digest": slice_digest(turns), "block_hash": block_hash,
                  "corpus_epoch": int(corpus_info.get("corpus_epoch", 0)),
                  "manifest_sha256": str(corpus_info.get("manifest_sha256", ""))}
    turn_ids = [turn_id(rec) for rec in turns]

    # Per-engine in-flight budgets. One semaphore shared across all three
    # engines (the old design) couples them: teacher calls starve miner calls
    # and vice versa, and the engines idle in turns. Each vLLM engine bounds
    # its own per-step work via max_num_batched_tokens, so client concurrency
    # only controls queue depth — separate budgets keep every engine fed.
    conc = int(duel_cfg["concurrency"])
    turn_conc = max(4, min(conc, 16))
    async with httpx.AsyncClient() as http:
        teachers = teacher if isinstance(teacher, list) else [teacher]
        teacher_m = ModelPool([
            VllmModel(t, http, asyncio.Semaphore(conc)) for t in teachers
        ])
        king_m = VllmModel(king, http, asyncio.Semaphore(conc))
        chall_m = VllmModel(challenger, http, asyncio.Semaphore(conc))

        rejection = await probe_injectable(
            chall_m, turns, float(duel_cfg["temperature"]),
            int(duel_cfg["max_thought_tokens"]), int(duel_cfg["max_action_tokens"]))
        if rejection:
            verdict = {
                "challenger_wins": False,
                "rejection_reason": f"unpromptable:{rejection}",
                "slice": slice_info,
            }
            return verdict, {"slice": slice_info, "turn_ids": turn_ids,
                             "rejection_reason": verdict["rejection_reason"]}

        # Fresh teacher references every duel (see RefCache docstring): the
        # cache lives and dies inside this call.
        refs = RefCache()
        # Both sides score concurrently: they live on separate vLLM
        # engines on separate GPUs, so interleaving them is pure win.
        # The exact same calls happen — RefCache per-turn locks dedupe
        # teacher reference sampling across the two sides — so scoring
        # semantics are untouched. Per-side turn semaphores bound each
        # side's in-flight turns independently.
        king_rows, chall_rows = await asyncio.gather(
            score_side(teacher_m, king_m, turns, refs, duel_cfg,
                       asyncio.Semaphore(turn_conc), on_progress),
            score_side(teacher_m, chall_m, turns, refs, duel_cfg,
                       asyncio.Semaphore(turn_conc), on_progress),
        )
        # Teacher rollouts actually used this duel (post-hoc: the slice
        # was unpredictable before reveal and the refs are resampled per
        # duel, so publishing them is audit data, not a reusable target).
        refs_used = {tid: refs.cache[tid] for tid in turn_ids
                     if tid in refs.cache}

    result = score_duel(
        chall_rows, king_rows,
        k_sigma=float(duel_cfg["k_sigma"]),
        challenger_bank_frac=_mean_bank(chall_rows),
        king_bank_frac=_mean_bank(king_rows))

    king_sum = _miner_summary(king_rows)
    chall_sum = _miner_summary(chall_rows)
    teacher_sum = _teacher_lengths(refs_used)
    _len_deltas(king_sum, teacher_sum)
    _len_deltas(chall_sum, teacher_sum)

    verdict = {
        "challenger_wins": result.challenger_wins,
        "margin": result.margin if math.isfinite(result.margin) else None,
        "se": result.se if math.isfinite(result.se) else None,
        "z": result.z if math.isfinite(result.z) else None,
        "k_sigma": result.k_sigma,
        "n_paired_turns": result.n_paired_turns,
        "ranking_formula": "Reason = lpC(y_C|z_A) − lpC(y_C|∅)",
        "duel_params": {
            "n_turns": int(duel_cfg["n_turns"]),
            "k_sigma": float(duel_cfg["k_sigma"]),
        },
        "king": king_sum,
        "challenger": chall_sum,
        "teacher": teacher_sum,
        "duel_seconds": time.monotonic() - started,
        "slice": slice_info,
    }
    artifact = {
        "slice": slice_info,
        "turn_ids": turn_ids,
        "teacher_refs": refs_used,
        "king_rows": king_rows,
        "challenger_rows": chall_rows,
    }
    return verdict, artifact
