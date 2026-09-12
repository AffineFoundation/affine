"""Duel execution on the eval machine: seeded slice, probe, scoring, verdict.

Slice seeding (chain contract): the duel slice is drawn from the public turn
corpus D with

    seed = blake2b(reveal_block_hash || challenger_hotkey, digest_size=8)

so a miner cannot know their slice before revealing (the block hash resolves
after the commit), and any external auditor can re-derive it from public
inputs. Stratified round-robin over repo×phase strata keeps single bug
families from dominating.

Sequential near-miss (2026-09-11, `[duel].near_miss_*`): when the first
slice's paired margin lands inside the near-miss window, further slices of
the same size are drawn with

    seed_i = blake2b(reveal_block_hash || challenger_hotkey || "|slice<i>")

from the turns not yet drawn, scored the same way, and the crown is decided
by the unchanged rule on the pooled rows. Every slice's seed and digest is
stamped so the pooled verdict is as re-derivable as a single-slice one.

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

from affine import dialects
from affine.corpus.materialize import stratum_key
from affine.score import (
    DEFAULT_NEAR_MISS_HIGH,
    DEFAULT_NEAR_MISS_LOW,
    NEAR_MISS_WINDOW_MODES,
    DuelResult,
    duel as score_duel,
    near_miss_triggered,
    near_miss_window,
    pooled_margin_stats,
    score_miner,
)

from .terms import (
    miner_terms,
    sample_miner_rollouts,
    sample_teacher_rollouts,
    score_teacher_rollouts,
)
from .protocol_probe import probe_settings, rejection_detail, run_probe
from .vllm_client import EngineUnreachableError, ModelPool, Served, VllmModel

log = logging.getLogger("evalsrv.dueling")

# Back-compat alias for anything that imported _phase_key.
_phase_key = stratum_key


class DuelAborted(RuntimeError):
    """Raised cooperatively when the running duel has been superseded.

    The validator never re-attaches to a running job (dispatch is POST /duel +
    SSE stream), so a new /duel arriving while one runs proves the running
    job's dispatcher is gone (validator restart, crown revert) and its verdict
    can never be consumed. Aborting at the next turn boundary hands the GPUs
    to the live request instead of burning up to a full scoring pass
    (observed 2026-08-14: 47 wasted minutes after the reign-19 revert).
    """


# -- slice ----------------------------------------------------------------------

def turn_id(rec: dict) -> str:
    if rec.get("turn_id"):
        return str(rec["turn_id"])
    return f"{rec['traj_id']}:{rec['turn_idx']}"


def duel_seed(block_hash: str, hotkey: str, slice_index: int = 0) -> int:
    """Slice seed. ``slice_index`` 0 is the contract seed every verdict since
    launch used, bit-for-bit; ``i >= 1`` are the sequential near-miss slices
    (suffix ``|slice<i>`` on the same public material, so they are just as
    unpredictable before reveal and just as re-derivable after)."""
    material = block_hash.encode() + hotkey.encode()
    if slice_index:
        material += f"|slice{slice_index}".encode()
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


def check_dialects(turns: list[dict], allowed: list[str]) -> None:
    """Fail-closed admission tripwire for the drawn slice.

    Every turn's action dialect must be registered (so it can be parsed) AND
    listed in ``[dataset].allowed_action_kinds`` (so the contract admits it).
    The fold enforces the same allowlist before a turn can enter D, so on a
    healthy corpus this never fires. It exists so a corpus-side mistake
    surfaces as a refused duel rather than a slice quietly scored against a
    dialect miners were never told to expect.
    """
    bad: dict[str, int] = {}
    for rec in turns:
        kind = rec.get("action_kind") or dialects.DEFAULT_KIND
        if not dialects.is_registered(kind) or kind not in allowed:
            bad[kind] = bad.get(kind, 0) + 1
    if bad:
        raise RuntimeError(
            f"slice contains inadmissible action_kind(s) {bad}; "
            f"allowed_action_kinds={allowed}")


def near_miss_settings(duel_cfg: dict) -> dict:
    """`[duel].near_miss_*` as one dict: enabled, low, high, extra_slices.

    Absent keys = the rule is off (pre-2026-09-11 behaviour: one slice).
    Fails loudly on a malformed window so a typo cannot silently turn every
    duel into a two-slice duel or none."""
    enabled = bool(duel_cfg.get("near_miss_enabled", False))
    low = float(duel_cfg.get("near_miss_low", DEFAULT_NEAR_MISS_LOW))
    high = float(duel_cfg.get("near_miss_high", DEFAULT_NEAR_MISS_HIGH))
    extra = int(duel_cfg.get("near_miss_extra_slices", 1))
    window_mode = str(duel_cfg.get("near_miss_window_mode", "absolute"))
    if enabled and not (0.0 <= low < high):
        raise ValueError(f"near-miss window needs 0 <= low < high, got "
                         f"low={low} high={high}")
    if enabled and extra < 1:
        raise ValueError(f"near_miss_extra_slices must be >= 1, got {extra}")
    if window_mode not in NEAR_MISS_WINDOW_MODES:
        raise ValueError(f"near_miss_window_mode must be one of "
                         f"{NEAR_MISS_WINDOW_MODES}, got {window_mode!r}")
    return {"enabled": enabled, "low": low, "high": high, "extra_slices": extra,
            "window_mode": window_mode}


# Keys of the validator's margin context that are stamped verbatim under
# duel_params (decaying crown margin, staged 2026-09-12).
MARGIN_STAMP_KEYS = ("min_margin_mode", "min_margin_base", "min_margin_peak",
                     "min_margin_floor", "min_margin_peak_cap",
                     "min_margin_decay_hours", "min_margin_decay_shape",
                     "crown_block", "crown_block_source", "decision_block",
                     "blocks_since_crown")


def confirmation_stamp(confirm: dict, slice_info: dict, result: DuelResult) -> dict:
    """The `confirmation` block of a window-best confirmation verdict: this
    slice's own numbers, the original (pooled) numbers it is pooled with,
    and the exact pooled (n, margin, se, z). `passed` = pooled margin > 0.
    A slice with no finite margin (every turn forfeited on both sides, or a
    gate the original passed now failing) does not pass."""
    base = dict(confirm.get("base") or {})
    n1 = int(base.get("n") or 0)
    m1, se1 = base.get("margin"), base.get("se")
    own = {"index": slice_info.get("index"), "seed": slice_info.get("seed"),
           "n": slice_info.get("n"), "digest": slice_info.get("digest"),
           "n_paired_turns": result.n_paired_turns,
           "n_forfeit_turns": result.n_forfeit_turns,
           "margin": result.margin if math.isfinite(result.margin) else None,
           "se": result.se if math.isfinite(result.se) else None,
           "z": result.z if math.isfinite(result.z) else None,
           "rejection_reason": (
               "thought_too_short" if result.thought_floor_blocked
               else "causality_fail" if result.causality_blocked else None)}
    stamp = {"challenge_id": confirm.get("challenge_id"),
             "slice_index": int(confirm.get("slice_index", 1)),
             "base": {"n": n1, "margin": m1, "se": se1},
             "slice": own, "pooled": None, "passed": False}
    ok = (n1 > 0 and isinstance(m1, (int, float)) and isinstance(se1, (int, float))
          and own["margin"] is not None and own["se"] is not None
          and result.n_paired_turns > 0 and own["rejection_reason"] is None)
    if ok:
        N, M, se, z = pooled_margin_stats(n1, float(m1), float(se1),
                                          result.n_paired_turns, result.margin,
                                          result.se)
        stamp["pooled"] = {"n": N, "margin": M, "se": se,
                           "z": z if math.isfinite(z) else None}
        stamp["passed"] = bool(M > 0.0)
    return stamp


def margin_stamp_for(duel_cfg: dict, margin: dict | None) -> dict:
    """The δ context this duel is decided under, as stamped on the verdict.

    With a validator-supplied ``margin`` carrying ``min_margin_effective``,
    that value is the δ of the crown test and the context keys ride along.
    Without one (older validator, offline harness) the pod's own
    ``[duel].min_margin`` is the δ, stamped as mode "fixed" — so every
    verdict from now on carries ``min_margin_effective`` and
    ``min_margin_mode`` and a replayer never has to guess."""
    base = float(duel_cfg.get("min_margin", 0.0))
    stamp: dict = {"min_margin_mode": "fixed", "min_margin_base": base,
                   "min_margin_effective": base}
    if not margin:
        return stamp
    eff = margin.get("min_margin_effective")
    if eff is None or not math.isfinite(float(eff)) or float(eff) < 0:
        raise ValueError(f"margin.min_margin_effective must be a finite "
                         f"non-negative number, got {eff!r}")
    stamp["min_margin_effective"] = float(eff)
    for key in MARGIN_STAMP_KEYS:
        if key in margin:
            stamp[key] = margin[key]
    return stamp


def _slice_stats(slice_info: dict, result: DuelResult) -> dict:
    """Verdict-sized record of one slice: how it was drawn and what the
    standard rule would have said on that slice alone."""
    return {
        "index": int(slice_info["index"]),
        "seed": slice_info["seed"], "n": slice_info["n"],
        "digest": slice_info["digest"],
        "n_paired_turns": result.n_paired_turns,
        "n_forfeit_turns": result.n_forfeit_turns,
        "margin": result.margin if math.isfinite(result.margin) else None,
        "se": result.se if math.isfinite(result.se) else None,
        "z": result.z if math.isfinite(result.z) else None,
        "challenger_wins": result.challenger_wins,
    }


# -- probe -----------------------------------------------------------------------

def token_caps(duel_cfg: dict):
    """(max_thought, max_action) for a turn's action dialect.

    `[duel].max_thought_tokens` / `max_action_tokens` apply to every kind
    unless `[duel.max_tokens_by_kind.<kind>]` overrides them (`thought`,
    `action`; a missing key keeps the default). Both sides and the teacher
    refs sample under the same cap, so a per-kind cap changes the slice's
    yield, not the pairing — it is still a `[duel]` knob and therefore a
    contract change when set. Empty table = pre-2026-09-04 behaviour exactly.
    Motivation: the gate-closed dry run had `boxed` turns hit finish=length
    at 1024+768 on 15/18 king rollouts (teacher refs 2.55/3), i.e. most math
    turns would forfeit at the flat cap.
    """
    dflt = (int(duel_cfg["max_thought_tokens"]), int(duel_cfg["max_action_tokens"]))
    table = duel_cfg.get("max_tokens_by_kind") or {}
    by_kind = {
        str(kind): (int(v.get("thought", dflt[0])), int(v.get("action", dflt[1])))
        for kind, v in table.items()
    }

    def caps(action_kind: str | None) -> tuple[int, int]:
        return by_kind.get(action_kind or dialects.DEFAULT_KIND, dflt)
    return caps


async def probe_injectable(model: VllmModel | ModelPool, turns: list[dict],
                           temperature: float, max_thought: int,
                           max_action: int, n_probe_turns: int = 3) -> str | None:
    """Cheap fail-fast before the full duel. Returns rejection reason or None.

    A checkpoint passes if, across a few turns, it (a) produces at least one
    rollout with a parsable action in that turn's dialect, and (b) returns
    finite forced logprobs under thought injection.

    Samples run concurrently (each is a ~31k-prefix generate). Echoes stay
    serial: a 16384-token fp32 logprob spike is ~16 GiB, and three at once
    can OOM the miner. Same checks, same score path. Fail-fast on the first
    hard reject after the samples land.
    """
    probe_recs = turns[:n_probe_turns]

    async def sample_one(rec: dict) -> tuple[dict, str, str, str | None]:
        prefix = rec["prefix"]
        try:
            z, y = await model.sample(prefix, temperature,
                                      max_thought + max_action,
                                      action_kind=rec.get("action_kind"))
            return rec, z, y, None
        except EngineUnreachableError:
            raise
        except Exception as e:
            return rec, "", "", f"probe_sample_failed:{type(e).__name__}:{e}"

    sampled = await asyncio.gather(*[sample_one(rec) for rec in probe_recs])
    any_action = False
    for rec, z, y, err in sampled:
        if err:
            return err
        if not y:
            continue
        any_action = True
        try:
            scored = await model.score_action(rec["prefix"], z, y)
        except EngineUnreachableError:
            raise
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
        # (z, y) after sample, before own/empty echoes — lets miner sampling
        # overlap teacher ref scoring on the first side that draws the turn.
        self._raw: dict[str, list[tuple[str, str]]] = {}
        # Per-turn locks: different turns sample teacher references
        # concurrently; only same-turn callers serialize. A single global
        # lock here would collapse the reference phase to sequential.
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def ensure_raw(self, tid: str, teacher: VllmModel | ModelPool,
                         prefix: list[dict], n: int, temperature: float,
                         max_thought: int, max_action: int,
                         action_kind: str | None = None
                         ) -> list[tuple[str, str]]:
        """Teacher (z, y) only — shared across king/challenger for this turn."""
        if tid in self.cache:
            return [(r["z"], r["y"]) for r in self.cache[tid]]
        if tid in self._raw:
            return self._raw[tid]
        async with self._locks[tid]:
            if tid in self.cache:
                return [(r["z"], r["y"]) for r in self.cache[tid]]
            if tid in self._raw:
                return self._raw[tid]
            raw = await sample_teacher_rollouts(
                teacher, prefix, n, temperature, max_thought, max_action,
                sticky_key=tid, action_kind=action_kind)
            self._raw[tid] = raw
            return raw

    async def ensure_scored(self, tid: str, teacher: VllmModel | ModelPool,
                            prefix: list[dict],
                            thought_echo: bool = False) -> list[dict]:
        """lp_own / lp_empty (+ lp_thought under min(R,G)) for the turn's
        raw teacher rollouts. The grounding band echoes are cached here so
        both sides share them — k thought echoes per turn per duel."""
        if tid in self.cache:
            return self.cache[tid]
        async with self._locks[tid]:
            if tid in self.cache:
                return self.cache[tid]
            raw = self._raw.get(tid) or []
            ref = await score_teacher_rollouts(
                teacher, prefix, raw, thought_echo=thought_echo,
                sticky_key=tid)
            self.cache[tid] = ref
            return ref

    async def get_or_sample(self, tid: str, teacher: VllmModel | ModelPool,
                            prefix: list[dict], n: int, temperature: float,
                            max_thought: int, max_action: int,
                            thought_echo: bool = False,
                            action_kind: str | None = None) -> list[dict]:
        if tid in self.cache:
            return self.cache[tid]
        await self.ensure_raw(
            tid, teacher, prefix, n, temperature, max_thought, max_action,
            action_kind)
        return await self.ensure_scored(tid, teacher, prefix, thought_echo)


async def score_side(teacher: VllmModel | ModelPool, miner: VllmModel | ModelPool,
                     turns: list[dict], refs: RefCache, duel_cfg: dict,
                     turn_sem: asyncio.Semaphore, on_progress,
                     abort_event=None) -> list[dict]:
    rows: list[dict] = []
    done = 0
    total = len(turns)
    n_teacher = int(duel_cfg["n_teacher_samples"])
    n_miner = int(duel_cfg["n_miner_samples"])
    temperature = float(duel_cfg["temperature"])
    caps = token_caps(duel_cfg)
    score_bank = bool(duel_cfg.get("score_bank", False))
    reason_only = bool(duel_cfg.get("reason_only", True))
    causality_gate = bool(duel_cfg.get("causality_gate", False))
    # min(R,G) v5: grounding echoes (t_i on refs, m per miner rollout).
    # min(R,G,A) v6 adds the action echoes lpC(y_A|z_C^i) per pair.
    score_mode = str(duel_cfg.get("score_mode", "reason"))
    thought_echo = score_mode in ("min_rg", "min_rga")
    action_echo = score_mode == "min_rga"

    async def one(rec: dict) -> None:
        nonlocal done
        tid = turn_id(rec)
        prefix = rec["prefix"]
        # Per-turn action dialect; absent on pre-dialect corpus records,
        # which means bash (affine.dialects.DEFAULT_KIND).
        action_kind = rec.get("action_kind")
        max_thought, max_action = caps(action_kind)
        async with turn_sem:
            if abort_event is not None and abort_event.is_set():
                raise DuelAborted("superseded by a new duel request")
            # Miner only needs the prefix x. Teacher refs (z_C, y_C) are
            # independent. Running them in series left miner GPUs idle at
            # duel start (chal-00076: all 8 at 0% while 64 turns sat in
            # ensure_raw). Same calls, overlapped. No sticky_key on the
            # miner sample: n_miner=1 cannot reuse a prefix cache, and
            # hash-pinning left one copy idle.
            raw, miner_rollouts = await asyncio.gather(
                refs.ensure_raw(
                    tid, teacher, prefix, n_teacher, temperature,
                    max_thought, max_action, action_kind),
                sample_miner_rollouts(
                    miner, prefix, n_miner, temperature,
                    max_thought, max_action, action_kind=action_kind),
            )
            if not raw:
                done += 1
                return
        # Teacher-only from here: ref echoes, then Reason/B/grounding.
        # Holding turn_sem through these left miner GPUs idle (chal-00075).
        if abort_event is not None and abort_event.is_set():
            raise DuelAborted("superseded by a new duel request")
        ref = await refs.ensure_scored(tid, teacher, prefix, thought_echo)
        if not ref:
            done += 1
            return
        t = await miner_terms(
            teacher, miner, prefix, ref, n_miner, temperature,
            max_thought, max_action,
            score_bank=score_bank, reason_only=reason_only,
            causality_gate=causality_gate,
            thought_echo=thought_echo,
            action_echo=action_echo,
            sticky_key=tid, action_kind=action_kind,
            rollouts=miner_rollouts)
        t.update({"turn_id": tid, "miner": miner.cfg.name})
        rows.append(t)
        done += 1
        on_progress(miner.cfg.name, done, total)

    await asyncio.gather(*[one(rec) for rec in turns])
    return rows


def _mean_bank(rows: list[dict]) -> float | None:
    vals = [r["bank_frac"] for r in rows if r.get("valid") and "bank_frac" in r]
    return sum(vals) / len(vals) if vals else None


def _miner_summary(rows: list[dict], tau: float | None,
                   score_mode: str = "reason",
                   band_c: float = 2.0, band_floor: float = 0.002,
                   forfeit_turn_score: float | None = None,
                   action_norm_bytes: float | None = None) -> dict:
    """Per-side summary: the score (reason) plus measured-not-scored telemetry."""
    s = score_miner(rows, bank_frac=_mean_bank(rows), tau=tau,
                    score_mode=score_mode, band_c=band_c,
                    band_floor=band_floor,
                    forfeit_turn_score=forfeit_turn_score,
                    action_norm_bytes=action_norm_bytes)
    out = {
        "reason": s.reason if math.isfinite(s.reason) else None,
        "n_turns": s.n_turns, "n_pairs": s.n_pairs,
        # Forfeits (v6): turns with no parseable action. Scored at the
        # floor when forfeit_turn_score is set, dropped otherwise.
        "n_forfeits": s.n_forfeits, "forfeit_rate": s.forfeit_rate,
        # -- telemetry (B pass rate is validity only when causality_gate) --
        "gate_pass_rate": s.gate_pass_rate, "bank_frac": s.bank_frac,
        "calib_ratio": s.calib_ratio, "baseline_abs": s.baseline_abs,
        "mean_l1lift": s.mean_l1lift,
        "mean_eta": s.mean_eta,
        "mean_len_z": s.mean_len_z, "median_len_z": s.median_len_z,
        "mean_len_y": s.mean_len_y,
        "mean_b": s.mean_b, "b_gate_pass_rate": s.b_gate_pass_rate,
    }
    if score_mode in ("min_rg", "min_rga"):
        # Which-leg-binds telemetry (post-fork watch item): g_bind_frac
        # near 1.0 means grounding is the binding constraint for this side.
        out["mean_r_leg"] = s.mean_r_leg
        out["mean_g_leg"] = s.mean_g_leg
        out["g_bind_frac"] = s.g_bind_frac
    if score_mode == "min_rga":
        out["mean_a_leg"] = s.mean_a_leg
        out["a_bind_frac"] = s.a_bind_frac
    return out


def _by_dialect(rows: list[dict], kind_by_tid: dict[str, str],
                tau: float | None, score_mode: str,
                band_c: float, band_floor: float,
                forfeit_turn_score: float | None = None,
                action_norm_bytes: float | None = None) -> dict[str, dict]:
    """Per-action_kind telemetry for one side (wvk 11 dialect watch item).

    ``parse_rate`` is the share of this dialect's turns where the side
    produced a parsable action (valid rows); everything else is the same
    leg telemetry as the side summary, restricted to that dialect's turns.
    A bash-only slice yields a single ``bash`` entry equal to the side totals.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        kind = kind_by_tid.get(r["turn_id"], dialects.DEFAULT_KIND)
        groups.setdefault(kind, []).append(r)
    out: dict[str, dict] = {}
    for kind, grp in sorted(groups.items()):
        s = score_miner(grp, tau=tau, score_mode=score_mode,
                        band_c=band_c, band_floor=band_floor,
                        forfeit_turn_score=forfeit_turn_score,
                        action_norm_bytes=action_norm_bytes)
        n_valid = sum(1 for r in grp if r.get("valid") and "pairs" in r)
        out[kind] = {
            "n_turns": len(grp), "n_valid": n_valid,
            "parse_rate": n_valid / len(grp) if grp else None,
            "reason": s.reason if math.isfinite(s.reason) else None,
            "mean_b": s.mean_b, "b_gate_pass_rate": s.b_gate_pass_rate,
            "median_len_z": s.median_len_z if n_valid else None,
            "mean_r_leg": s.mean_r_leg, "mean_g_leg": s.mean_g_leg,
            "g_bind_frac": s.g_bind_frac,
        }
        if score_mode == "min_rga":
            out[kind]["mean_a_leg"] = s.mean_a_leg
            out[kind]["a_bind_frac"] = s.a_bind_frac
    return out


def _teacher_by_dialect(turns: list[dict],
                        refs_used: dict[str, list[dict]]) -> dict[str, dict]:
    """Teacher reference yield per dialect: turns drawn, turns with zero
    parsable refs (unscorable), mean refs per turn."""
    out: dict[str, dict] = {}
    for rec in turns:
        kind = rec.get("action_kind") or dialects.DEFAULT_KIND
        d = out.setdefault(kind, {"n_turns": 0, "zero_ref_turns": 0, "_refs": 0})
        n = len(refs_used.get(turn_id(rec)) or [])
        d["n_turns"] += 1
        d["zero_ref_turns"] += (n == 0)
        d["_refs"] += n
    for d in out.values():
        d["mean_refs"] = d.pop("_refs") / d["n_turns"] if d["n_turns"] else None
    return out


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
                   king: Served | list[Served],
                   challenger: Served | list[Served],
                   teacher: Served | list[Served],
                   block_hash: str, hotkey: str, corpus_info: dict,
                   on_progress,
                   corpus: "CorpusSync | None" = None,
                   abort_event=None,
                   margin: dict | None = None,
                   confirm: dict | None = None) -> tuple[dict, dict]:
    """Full duel. Returns (verdict, artifact).

    ``confirm`` (window-best crown mode, staged 2026-09-12): score ONE fresh
    slice for a window winner instead of a full duel. ``slice_index`` (k ≥
    1) names the draw: slices 0..k−1 are re-derived from the same
    block_hash ‖ hotkey seeds and excluded, exactly as the near-miss extra
    draw does, so the confirmation turns are disjoint from every turn the
    original duel scored. ``base`` = {n, margin, se} of the original
    (pooled) verdict; the verdict gains ``confirmation`` with this slice's
    own numbers and the exact pooled (n, margin, se, z) over both —
    ``passed`` iff the pooled margin is > 0. Probes are skipped (the model
    was admitted by its original duel); the near-miss rule does not apply.

    The verdict is the small audit summary streamed to the validator. The
    artifact is the full training-grade record — sliced turn ids, teacher
    reference rollouts, and both sides' per-turn pair rows (thoughts/actions
    plus every forced-logprob component) — published post-hoc so miners can
    train on exactly what was scored.

    schema_version>=2: sample the Parquet index via ``corpus``, materialize
    only the drawn turns. schema v1 / ``turns_path``: load flat turns.jsonl.

    ``margin`` (decaying crown margin, staged 2026-09-12): the validator's
    δ context for this duel. Its ``min_margin_effective`` replaces
    ``[duel].min_margin`` in the crown test (and in the "bar" near-miss
    window); the whole dict is stamped under ``duel_params``. None = the
    pod's own toml δ, stamped as mode "fixed".
    """
    duel_cfg = dict(engine_cfg["duel"])
    margin_stamp = margin_stamp_for(duel_cfg, margin)
    duel_cfg["min_margin"] = margin_stamp["min_margin_effective"]
    started = time.monotonic()
    n = int(duel_cfg["n_turns"])
    near_miss = near_miss_settings(duel_cfg)
    if corpus is not None and corpus.schema_version >= 2:
        rows = corpus.load_index_rows()
    else:
        if turns_path is None:
            raise ValueError("turns_path required for schema_version=1")
        with open(turns_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    allowed_kinds = [str(k) for k in engine_cfg.get("dataset", {}).get(
        "allowed_action_kinds", [dialects.DEFAULT_KIND])]

    def draw_slice(index: int, exclude: set[str]) -> tuple[list[dict], dict]:
        """Slice `index` (0 = the contract slice): n_turns drawn from the
        corpus minus `exclude`, materialized and admission-checked, plus
        its audit stamp (seed, n, digest, manifest pin)."""
        seed = duel_seed(block_hash, hotkey, index)
        pool = ([r for r in rows if turn_id(r) not in exclude]
                if exclude else rows)
        picked = sample_slice(pool, n, seed)
        turns = (corpus.materialize_turns(picked)
                 if corpus is not None and corpus.schema_version >= 2
                 else picked)
        check_dialects(turns, allowed_kinds)
        # The manifest hash pins exactly which shard set this duel was
        # scored against — replayable even after shards are retired.
        info = {"index": index, "seed": seed, "n": len(turns),
                "digest": slice_digest(turns), "block_hash": block_hash,
                "corpus_epoch": int(corpus_info.get("corpus_epoch", 0)),
                "manifest_sha256": str(corpus_info.get("manifest_sha256", ""))}
        # Schema-3 corpora are a view over traces served from a base URL
        # that may move (Hippius -> data.affine.io); stamp both so a
        # replayer knows which view built these prefixes and where the
        # manifest lived.
        if corpus_info.get("view_spec"):
            info["view_spec"] = str(corpus_info["view_spec"])
            info["corpus_base_url"] = str(corpus_info.get("corpus_base_url", ""))
        return turns, info

    turns, slice_info = draw_slice(0, set())
    turn_ids = [turn_id(rec) for rec in turns]
    if confirm:
        # Re-derive every slice the original duel scored (same seeds) only
        # to exclude their turns; the confirmation slice is draw k.
        k = int(confirm.get("slice_index", 1))
        if k < 1:
            raise ValueError("confirm.slice_index must be >= 1")
        drawn = set(turn_ids)
        prior = [slice_info]
        for idx in range(1, k):
            prev_turns, prev_info = draw_slice(idx, drawn)
            drawn |= {turn_id(r) for r in prev_turns}
            prior.append(prev_info)
        turns, slice_info = draw_slice(k, drawn)
        turn_ids = [turn_id(rec) for rec in turns]
        slice_info["confirmation_of"] = [
            {key: i[key] for key in ("index", "seed", "n", "digest")} for i in prior]
        log.info("confirmation slice %d for %s: n=%d (excluding %d turns of %d "
                 "prior slice(s))", k, confirm.get("challenge_id"), len(turns),
                 len(drawn), len(prior))
    # Every slice actually scored, in order; slice 0 is `slice_info`.
    slices: list[dict] = [{"info": slice_info, "turn_ids": list(turn_ids)}]

    # Per-engine in-flight budgets. One semaphore shared across all three
    # engines (the old design) couples them: teacher calls starve miner calls
    # and vice versa, and the engines idle in turns. Each vLLM engine bounds
    # its own per-step work via max_num_batched_tokens, so client concurrency
    # only controls queue depth — separate budgets keep every engine fed.
    conc = int(duel_cfg["concurrency"])
    # Cap used to be 16 while concurrency=24, which under-fed the dual teacher
    # replicas once sticky routing spread load. Match the client queue depth.
    turn_conc = max(4, conc)
    # Staged v7 knob: miner rollouts without </think> forfeit. Miner sides
    # only — teacher refs keep their semantics so refs (and the G band built
    # from them) are unchanged by the flip.
    require_think_close = bool(duel_cfg.get("require_think_close", False))
    async with httpx.AsyncClient() as http:
        def _pool(served: Served | list[Served],
                  require_close: bool = False) -> ModelPool:
            items = served if isinstance(served, list) else [served]
            return ModelPool([
                VllmModel(s, http, asyncio.Semaphore(conc),
                          require_think_close=require_close) for s in items
            ])
        teacher_m = _pool(teacher)
        king_m = _pool(king, require_think_close)
        chall_m = _pool(challenger, require_think_close)

        rejection = None if confirm else await probe_injectable(
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

        # Chat-protocol conformance (admission rule, staged 2026-09-07):
        # Cursor-shaped prompts through the challenger's own template with
        # thinking on; every reply must close </think> and carry a visible
        # answer. Runs before the 1,300-turn scoring so a reject costs
        # minutes. mode: off (default) | shadow (publish only) | enforce.
        probe_cfg = probe_settings(engine_cfg.get("protocol_probe"))
        protocol = None
        if probe_cfg["mode"] != "off" and not confirm:
            protocol = await run_probe(chall_m, probe_cfg)
            log.info("protocol probe (%s): pass_rate=%.2f think_close=%.2f %s",
                     probe_cfg["mode"], protocol["pass_rate"],
                     protocol["think_close_rate"], protocol["by_reason"])
            if probe_cfg["mode"] == "enforce" and not protocol["passed"]:
                verdict = {
                    "challenger_wins": False,
                    "rejection_reason": f"protocol:{rejection_detail(protocol)}",
                    "protocol_probe": _probe_public(protocol),
                    "slice": slice_info,
                }
                return verdict, {"slice": slice_info, "turn_ids": turn_ids,
                                 "rejection_reason": verdict["rejection_reason"],
                                 "protocol_probe": protocol}

        min_thought = int(duel_cfg.get("min_thought_chars", 0))
        causality_gate = bool(duel_cfg.get("causality_gate", False))
        causality_gamma = (
            float(duel_cfg.get("causality_gamma", 0.30)) if causality_gate else 0.0)
        tau = float(duel_cfg.get("tau", 0.0)) or None  # tau <= 0 → v3 plain mean
        score_mode = str(duel_cfg.get("score_mode", "reason"))
        band_c = float(duel_cfg.get("band_c", 2.0))
        band_floor = float(duel_cfg.get("band_floor", 0.002))
        # v6 forfeit floor: absent/None keeps the legacy drop-from-pairing rule.
        _ff = duel_cfg.get("forfeit_turn_score")
        forfeit_turn_score = float(_ff) if _ff is not None else None
        # A-leg normalizer (min_rga only): summed action lift / this many bytes.
        # Absent = the per-byte design (length-biased; replay of the 09-04 probe).
        _anb = duel_cfg.get("action_norm_bytes")
        action_norm_bytes = float(_anb) if _anb is not None else None
        # Minimum z safeguard (staged 2026-09-12; 0 = off).
        min_z = float(duel_cfg.get("min_z", 0.0) or 0.0)

        def decide(c_rows: list[dict], k_rows: list[dict]) -> DuelResult:
            """The crown rule on a set of paired rows — one slice or the
            pool of all slices, the same call either way. δ is the
            effective margin for this duel (`margin_stamp`)."""
            return score_duel(
                c_rows, k_rows,
                k_sigma=float(duel_cfg["k_sigma"]),
                min_margin=float(duel_cfg.get("min_margin", 0.0)),
                min_thought_chars=min_thought,
                causality_gamma=causality_gamma,
                challenger_bank_frac=_mean_bank(c_rows),
                king_bank_frac=_mean_bank(k_rows),
                tau=tau,
                score_mode=score_mode, band_c=band_c, band_floor=band_floor,
                forfeit_turn_score=forfeit_turn_score,
                action_norm_bytes=action_norm_bytes,
                min_z=min_z)

        # Fresh teacher references every duel (see RefCache docstring): the
        # cache lives and dies inside this call.
        refs = RefCache()

        async def score_slice(slice_turns: list[dict], done_before: int
                              ) -> tuple[list[dict], list[dict]]:
            """Both sides on one slice. They score concurrently: they live
            on separate vLLM engines on separate GPUs, so interleaving them
            is pure win. The exact same calls happen — RefCache per-turn
            locks dedupe teacher reference sampling across the two sides —
            so scoring semantics are untouched. Per-side turn semaphores
            bound each side's in-flight turns independently. Progress is
            reported cumulatively over all slices of the duel."""
            total = done_before + len(slice_turns)

            def progress(miner: str, done: int, _total: int) -> None:
                on_progress(miner, done_before + done, total)
            return await asyncio.gather(
                score_side(teacher_m, king_m, slice_turns, refs, duel_cfg,
                           asyncio.Semaphore(turn_conc), progress,
                           abort_event=abort_event),
                score_side(teacher_m, chall_m, slice_turns, refs, duel_cfg,
                           asyncio.Semaphore(turn_conc), progress,
                           abort_event=abort_event),
            )

        king_rows, chall_rows = await score_slice(turns, 0)
        result = decide(chall_rows, king_rows)
        slice_results = [_slice_stats(slice_info, result)]
        # Sequential near-miss (2026-09-11): a first-slice margin inside the
        # window is one slice's noise away from the bar either way. Draw
        # more slices — different seed, turns the duel has not scored yet,
        # same size and stratification — and let the unchanged rule decide
        # on the pool. Rows are simply concatenated: score.duel pairs by
        # turn_id, so the pooled margin is the mean over every paired turn
        # of every slice, the pooled SE is sd/√n_pooled, forfeits keep
        # their floor, and the gates are read over the pooled challenger
        # rows. Nothing about a single turn's score changes.
        # Window placement: "absolute" = the configured pair; "bar" = around
        # this slice's own crown bar max(k_sigma·SE, δ_effective), so a
        # decayed δ moves the window with it.
        nm_low, nm_high = near_miss_window(
            result, near_miss["window_mode"], near_miss["low"], near_miss["high"])
        triggered = (bool(near_miss["enabled"]) and not confirm
                     and near_miss_triggered(result, nm_low, nm_high))
        if triggered:
            log.info("near-miss: margin=%.5f in (%.4f, %.4f) [%s], z=%.2f — "
                     "drawing %d extra slice(s) of %d turns", result.margin,
                     nm_low, nm_high, near_miss["window_mode"], result.z,
                     near_miss["extra_slices"], n)
            for index in range(1, int(near_miss["extra_slices"]) + 1):
                if abort_event is not None and abort_event.is_set():
                    raise DuelAborted("superseded by a new duel request")
                drawn = {tid for s in slices for tid in s["turn_ids"]}
                extra_turns, extra_info = draw_slice(index, drawn)
                extra_ids = [turn_id(rec) for rec in extra_turns]
                k_extra, c_extra = await score_slice(extra_turns, len(turn_ids))
                slice_results.append(
                    _slice_stats(extra_info, decide(c_extra, k_extra)))
                slices.append({"info": extra_info, "turn_ids": extra_ids})
                turns = turns + extra_turns
                turn_ids = turn_ids + extra_ids
                king_rows = king_rows + k_extra
                chall_rows = chall_rows + c_extra
            result = decide(chall_rows, king_rows)
            log.info("near-miss pooled: n=%d margin=%.5f se=%.5f z=%.2f wins=%s",
                     result.n_paired_turns, result.margin, result.se,
                     result.z, result.challenger_wins)
        # Teacher rollouts actually used this duel (post-hoc: the slice
        # was unpredictable before reveal and the refs are resampled per
        # duel, so publishing them is audit data, not a reusable target).
        refs_used = {tid: refs.cache[tid] for tid in turn_ids
                     if tid in refs.cache}

    if len(slices) > 1:
        # `slice` stays the contract slice (seed/n/digest as every verdict
        # since launch); the extra draws ride along so a reader of `slice`
        # alone sees that the verdict pooled more turns.
        slice_info["extra_slices"] = [
            {k: s["info"][k] for k in ("index", "seed", "n", "digest")}
            for s in slices[1:]]
        slice_info["n_pooled"] = len(turn_ids)

    king_sum = _miner_summary(king_rows, tau, score_mode, band_c, band_floor,
                              forfeit_turn_score, action_norm_bytes)
    chall_sum = _miner_summary(chall_rows, tau, score_mode, band_c, band_floor,
                               forfeit_turn_score, action_norm_bytes)
    teacher_sum = _teacher_lengths(refs_used)
    _len_deltas(king_sum, teacher_sum)
    _len_deltas(chall_sum, teacher_sum)
    # Chat-protocol well-formedness (2026-09-07): share of natural samples
    # that closed </think>. Telemetry whether or not require_think_close is
    # on — the number the flip decision needs. Teacher too, as the reference.
    for summary, pool in ((king_sum, king_m), (chall_sum, chall_m),
                          (teacher_sum, teacher_m)):
        summary["n_samples"] = pool.n_samples
        summary["think_close_rate"] = pool.think_close_rate
    # Per-dialect telemetry (wvk 11 watch item): parse rate and leg means
    # per action_kind on each side; teacher ref yield per dialect.
    kind_by_tid = {turn_id(rec): rec.get("action_kind") or dialects.DEFAULT_KIND
                   for rec in turns}
    king_sum["by_dialect"] = _by_dialect(
        king_rows, kind_by_tid, tau, score_mode, band_c, band_floor,
        forfeit_turn_score, action_norm_bytes)
    chall_sum["by_dialect"] = _by_dialect(
        chall_rows, kind_by_tid, tau, score_mode, band_c, band_floor,
        forfeit_turn_score, action_norm_bytes)
    teacher_sum["by_dialect"] = _teacher_by_dialect(turns, refs_used)
    # Counted over every scored turn (all slices when the near-miss rule
    # pooled), matching the by_dialect telemetry above.
    slice_info["dialects"] = {
        kind: sum(1 for k in kind_by_tid.values() if k == kind)
        for kind in sorted(set(kind_by_tid.values()))}

    _rg_formula = (
        "R = tau·log(mean_i exp(a_i/tau)) − mean_i a_i,"
        " a_i = lpC(y_i|z_A) − lpC(y_i|∅);"
        " G = min(m − (mu − w), (mu + w) − m),"
        " m = lpC(z_A|x), mu/sd over lpC(z_C^i|x),"
        " w = max(band_c·sd, band_floor)")
    if score_mode == "min_rga":
        ranking_formula = (
            "turn = min(R, G, A); " + _rg_formula +
            "; A = tau·log(mean_i exp(b_i/tau)),"
            + (" b_i = Σ_bytes[lpC(y_A|z_C^i) − lpC(y_A|∅)] / action_norm_bytes"
               if action_norm_bytes is not None
               else " b_i = lpC(y_A|z_C^i) − lpC(y_A|∅)"))
    elif score_mode == "min_rg":
        ranking_formula = "turn = min(R, G); " + _rg_formula
    elif tau:
        ranking_formula = (
            "Reason(turn) = tau·log(mean_i exp((lpC(y_i|z_A) − lpC(y_i|∅))/tau))")
    else:
        ranking_formula = "Reason = lpC(y_C|z_A) − lpC(y_C|∅)"
    if forfeit_turn_score is not None:
        ranking_formula += (
            f"; forfeit (no parseable action) scores {forfeit_turn_score:g}")
    if require_think_close:
        ranking_formula += "; a rollout without </think> is a forfeit"
    if near_miss["enabled"]:
        if near_miss["window_mode"] == "bar":
            ranking_formula += (
                f"; first-slice margin in (0.5·bar, 1.5·bar), bar = "
                f"max(k_sigma·SE, δ), draws {near_miss['extra_slices']} more "
                f"slice(s) and the crown is decided on the pooled turns")
        else:
            ranking_formula += (
                f"; first-slice margin in ({near_miss['low']:g}, "
                f"{near_miss['high']:g}) draws {near_miss['extra_slices']} more "
                f"slice(s) and the crown is decided on the pooled turns")
    if margin_stamp["min_margin_mode"] == "decay":
        ranking_formula += (
            f"; δ = {margin_stamp['min_margin_effective']:.6g} for this duel "
            f"(decaying margin: {margin_stamp.get('min_margin_decay_shape', '?')} "
            f"from peak {margin_stamp.get('min_margin_peak')} at crown block "
            f"{margin_stamp.get('crown_block')} to floor "
            f"{margin_stamp.get('min_margin_floor')} over "
            f"{margin_stamp.get('min_margin_decay_hours')} h; "
            f"{margin_stamp.get('blocks_since_crown')} blocks since crown)")
    if min_z > 0:
        ranking_formula += f"; a crown also needs z ≥ {min_z:g}"

    # Sequential near-miss stamp: the window, what each slice said on its
    # own, and (when pooled) the pooled decision — the top-level margin /
    # se / z / n_paired_turns above are the deciding (pooled) numbers.
    near_miss_stamp = {
        "enabled": near_miss["enabled"],
        "low": near_miss["low"], "high": near_miss["high"],
        "window_mode": near_miss["window_mode"],
        # The window actually tested on the first slice ("bar" mode moves it).
        "window": [nm_low, nm_high],
        "extra_slices": near_miss["extra_slices"],
        "triggered": triggered,
        "slices": slice_results,
        "pooled": ({
            "n_turns": len(turn_ids),
            "n_paired_turns": result.n_paired_turns,
            "n_forfeit_turns": result.n_forfeit_turns,
            "margin": result.margin if math.isfinite(result.margin) else None,
            "se": result.se if math.isfinite(result.se) else None,
            "z": result.z if math.isfinite(result.z) else None,
            "challenger_wins": result.challenger_wins,
        } if triggered else None),
    }

    verdict = {
        "challenger_wins": result.challenger_wins,
        "rejection_reason": (
            "thought_too_short" if result.thought_floor_blocked
            else "causality_fail" if result.causality_blocked
            else "z_below_min" if result.min_z_blocked
            else None),
        "margin": result.margin if math.isfinite(result.margin) else None,
        "se": result.se if math.isfinite(result.se) else None,
        "z": result.z if math.isfinite(result.z) else None,
        "k_sigma": result.k_sigma,
        "min_margin": result.min_margin,
        "n_paired_turns": result.n_paired_turns,
        "n_forfeit_turns": result.n_forfeit_turns,
        "ranking_formula": ranking_formula,
        "duel_params": {
            "n_turns": int(duel_cfg["n_turns"]),
            "k_sigma": float(duel_cfg["k_sigma"]),
            "min_margin": float(duel_cfg.get("min_margin", 0.0)),
            "min_thought_chars": min_thought,
            "causality_gate": causality_gate,
            "causality_gamma": causality_gamma,
            "tau": tau,
            "n_teacher_samples": int(duel_cfg["n_teacher_samples"]),
            "n_miner_samples": int(duel_cfg["n_miner_samples"]),
            "score_mode": score_mode,
            "band_c": band_c,
            "band_floor": band_floor,
            "forfeit_turn_score": forfeit_turn_score,
            "action_norm_bytes": action_norm_bytes,
            "require_think_close": require_think_close,
            "allowed_action_kinds": allowed_kinds,
            "max_thought_tokens": int(duel_cfg["max_thought_tokens"]),
            "max_action_tokens": int(duel_cfg["max_action_tokens"]),
            "max_tokens_by_kind": {
                str(k): {str(f): int(n) for f, n in v.items()}
                for k, v in (duel_cfg.get("max_tokens_by_kind") or {}).items()},
            "near_miss_enabled": near_miss["enabled"],
            "near_miss_low": near_miss["low"],
            "near_miss_high": near_miss["high"],
            "near_miss_extra_slices": near_miss["extra_slices"],
            "near_miss_window_mode": near_miss["window_mode"],
            "min_z": min_z,
            # Decaying crown margin (staged 2026-09-12): `min_margin` above
            # is already the effective δ; these say where it came from.
            **margin_stamp,
        },
        "near_miss": near_miss_stamp,
        "king": king_sum,
        "challenger": chall_sum,
        "teacher": teacher_sum,
        "duel_seconds": time.monotonic() - started,
        "slice": slice_info,
    }
    if protocol is not None:
        verdict["protocol_probe"] = _probe_public(protocol)
    if confirm:
        verdict["confirmation"] = confirmation_stamp(confirm, slice_info, result)
        verdict["ranking_formula"] += (
            "; CONFIRMATION SLICE (window-best crown): pooled margin over the "
            "original slice(s) and this one must be > 0")
    artifact = {
        "slice": slice_info,
        # Every turn scored, in slice order (slice 0 first); `slices` splits
        # them per draw with each draw's seed/digest so any one slice — or
        # the pool — can be re-derived from public D.
        "turn_ids": turn_ids,
        "slices": [{**s["info"], "turn_ids": s["turn_ids"]} for s in slices],
        "near_miss": near_miss_stamp,
        "teacher_refs": refs_used,
        "king_rows": king_rows,
        "challenger_rows": chall_rows,
    }
    if protocol is not None:
        artifact["protocol_probe"] = protocol
    return verdict, artifact


def _probe_public(protocol: dict) -> dict:
    """Verdict-sized view of a protocol probe: rates + per-prompt verdicts,
    without the completion text heads (those go to the artifact)."""
    return {
        "mode": protocol["mode"],
        "passed": protocol["passed"],
        "pass_rate": protocol["pass_rate"],
        "min_pass_rate": protocol["min_pass_rate"],
        "think_close_rate": protocol["think_close_rate"],
        "n": protocol["n"],
        "by_reason": protocol["by_reason"],
        "by_prompt": {
            r["id"]: {"ok": r["ok"], "reasons": r["reasons"]}
            for r in sorted(protocol["results"], key=lambda r: (r["id"], -r["ok"]))
        },
        "settings": protocol["settings"],
    }
