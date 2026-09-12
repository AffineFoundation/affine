"""Decaying crown margin (staged 2026-09-12): the schedule, the crown test
with `min_z`, the near-miss window placement, and the history replay.

    source .venv/bin/activate
    python -m unittest discover -s tests -t . -v                  # from affine/
    python -m unittest discover -s affine/tests -t affine         # from the repo root
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from affine.chain import BlockHashUnavailable
from affine.config import load_config
from affine.score import (
    DEFAULT_MIN_MARGIN,
    DuelResult,
    MarginSchedule,
    duel,
    effective_min_margin,
    near_miss_window,
)
from affine.state import King
from affine.validator import Validator
from evalsrv.dueling import margin_stamp_for, near_miss_settings
from scripts.replay_min_margin_decay import (
    Duel,
    load_duels,
    run_replay,
    seed_time,
)

REPO = Path(__file__).resolve().parents[1]
BLOCKS_PER_HOUR = 300  # 3600 s / 12 s


def _rows(diffs: list[float], king_level: float = 0.0,
          z_len: int = 200) -> tuple[list[dict], list[dict]]:
    """Paired challenger/king rows whose per-turn score difference is
    exactly `diffs[i]` under score_mode="reason" with k=1 (turn score =
    lpC_yc_za − lpC_yc_e)."""
    chall, king = [], []
    for i, d in enumerate(diffs):
        tid = f"t{i}"
        z = "x" * z_len
        king.append({"turn_id": tid, "miner": "king", "valid": True, "pairs": [
            {"lpC_yc_za": king_level, "lpC_yc_e": 0.0, "z_a": z, "y_a": "ls"}]})
        chall.append({"turn_id": tid, "miner": "chall", "valid": True, "pairs": [
            {"lpC_yc_za": king_level + d, "lpC_yc_e": 0.0, "z_a": z, "y_a": "ls"}]})
    return chall, king


class ScheduleTests(unittest.TestCase):
    def test_fixed_mode_ignores_the_clock(self):
        s = MarginSchedule(min_margin=0.002, mode="fixed")
        for peak, since in ((None, None), (0.0001, 0), (0.002, 10**9), (0.5, -5)):
            self.assertEqual(s.effective(peak, since), 0.002)
        self.assertEqual(s.next_peak(0.00001), 0.002)

    def test_linear_curve_endpoints_and_midpoint(self):
        s = MarginSchedule(mode="decay", peak_cap=0.002, floor=0.0001,
                           decay_hours=48.0, shape="linear")
        self.assertEqual(s.decay_blocks, 48 * BLOCKS_PER_HOUR)
        self.assertAlmostEqual(s.effective(0.002, 0), 0.002)
        self.assertAlmostEqual(s.effective(0.002, 24 * BLOCKS_PER_HOUR), 0.00105)
        self.assertAlmostEqual(s.effective(0.002, 48 * BLOCKS_PER_HOUR), 0.0001)
        # Past the window it stays on the floor.
        self.assertAlmostEqual(s.effective(0.002, 500 * BLOCKS_PER_HOUR), 0.0001)

    def test_exponential_curve_is_geometric_and_hits_the_floor(self):
        s = MarginSchedule(mode="decay", peak_cap=0.002, floor=0.0001,
                           decay_hours=48.0, shape="exponential")
        self.assertAlmostEqual(s.effective(0.002, 0), 0.002)
        half = s.effective(0.002, 24 * BLOCKS_PER_HOUR)
        self.assertAlmostEqual(half, 0.002 * math.sqrt(0.0001 / 0.002))
        self.assertAlmostEqual(s.effective(0.002, 48 * BLOCKS_PER_HOUR), 0.0001)
        # Same factor per block: ratio over [0, 12h] equals ratio over [12h, 24h].
        a, b, c = (s.effective(0.002, h * BLOCKS_PER_HOUR) for h in (0, 12, 24))
        self.assertAlmostEqual(a / b, b / c)

    def test_never_stricter_than_cap_and_never_below_floor(self):
        s = MarginSchedule(mode="decay", peak_cap=0.002, floor=0.0001)
        self.assertEqual(s.effective(0.01, 0), 0.002)        # stored peak above cap
        self.assertEqual(s.effective(0.00001, 0), 0.0001)    # stored peak below floor
        self.assertEqual(s.effective(None, None), 0.002)     # no crown known → cap
        self.assertEqual(s.effective(0.002, -100), 0.002)    # clock skew → peak

    def test_doubling_rule(self):
        s = MarginSchedule(mode="decay", peak_cap=0.002, floor=0.0001)
        self.assertAlmostEqual(s.next_peak(0.0007), 0.0014)
        self.assertAlmostEqual(s.next_peak(0.0015), 0.002)   # capped
        self.assertAlmostEqual(s.next_peak(0.0001), 0.0002)  # crown at the floor
        reset = MarginSchedule(mode="decay", peak_cap=0.002, floor=0.0001,
                               double_on_crown=False)
        self.assertEqual(reset.next_peak(0.0001), 0.002)

    def test_doubling_from_the_floor_never_recovers_the_cap(self):
        """The literal spec: a crown at the floor starts the next cycle at
        2·floor, so back-to-back floor crowns pin δ near the floor."""
        s = MarginSchedule(mode="decay", peak_cap=0.002, floor=0.0001,
                           decay_hours=24.0)
        peak = s.peak_cap
        for _ in range(5):
            delta = s.effective(peak, 30 * BLOCKS_PER_HOUR)  # crown 30 h in → at floor
            peak = s.next_peak(delta)
        self.assertAlmostEqual(peak, 0.0002)

    def test_effective_min_margin_helper(self):
        s = MarginSchedule(mode="decay", peak_cap=0.002, floor=0.0001,
                           decay_hours=48.0, shape="linear")
        eff, since = effective_min_margin(s, 0.002, 1000, 1000 + 24 * BLOCKS_PER_HOUR)
        self.assertEqual(since, 24 * BLOCKS_PER_HOUR)
        self.assertAlmostEqual(eff, 0.00105)
        eff, since = effective_min_margin(s, None, None, 5)
        self.assertIsNone(since)
        self.assertEqual(eff, 0.002)

    def test_validation(self):
        with self.assertRaises(ValueError):
            MarginSchedule(mode="sometimes")
        with self.assertRaises(ValueError):
            MarginSchedule(mode="decay", shape="steps")
        with self.assertRaises(ValueError):
            MarginSchedule(mode="decay", floor=0.003, peak_cap=0.002)
        with self.assertRaises(ValueError):
            MarginSchedule(mode="decay", decay_hours=0)
        with self.assertRaises(ValueError):
            MarginSchedule(mode="decay", double_factor=1.0)
        # Fixed mode does not validate decay knobs it will never use.
        MarginSchedule(mode="fixed", floor=0.003, peak_cap=0.002)


class CrownTestTests(unittest.TestCase):
    def test_min_z_off_is_todays_rule(self):
        chall, king = _rows([0.003] * 50 + [0.001] * 50)
        r = duel(chall, king, k_sigma=2.0, min_margin=0.0, min_thought_chars=0,
                 tau=None, score_mode="reason")
        self.assertTrue(r.challenger_wins)
        self.assertFalse(r.min_z_blocked)
        self.assertEqual(r.min_z, 0.0)

    def test_min_z_blocks_a_bar_clearing_tie(self):
        # margin 0.002 with SE ≈ 0.0009 → z ≈ 2.2: clears max(2·SE, δ=0)
        # under today's rule, blocked by min_z = 2.5.
        chall, king = _rows([0.011] * 50 + [-0.007] * 50)
        r0 = duel(chall, king, k_sigma=2.0, min_margin=0.0, min_thought_chars=0,
                  tau=None, score_mode="reason", min_z=0.0)
        self.assertAlmostEqual(r0.margin, 0.002)
        self.assertTrue(2.0 < r0.z < 2.5)
        self.assertTrue(r0.challenger_wins)
        r = duel(chall, king, k_sigma=2.0, min_margin=0.0, min_thought_chars=0,
                 tau=None, score_mode="reason", min_z=2.5)
        self.assertFalse(r.challenger_wins)
        self.assertTrue(r.min_z_blocked)
        # min_z never flips a loss into a win and does not mark a plain loss.
        chall, king = _rows([-0.001] * 100)
        r = duel(chall, king, k_sigma=2.0, min_margin=0.0, min_thought_chars=0,
                 tau=None, score_mode="reason", min_z=2.5)
        self.assertFalse(r.challenger_wins)
        self.assertFalse(r.min_z_blocked)

    def test_effective_delta_is_just_min_margin_to_the_crown_test(self):
        chall, king = _rows([0.0015] * 100)  # se = 0 → z = inf
        self.assertTrue(duel(chall, king, min_margin=0.001, min_thought_chars=0,
                             tau=None, score_mode="reason").challenger_wins)
        self.assertFalse(duel(chall, king, min_margin=0.002, min_thought_chars=0,
                              tau=None, score_mode="reason").challenger_wins)


class NearMissWindowTests(unittest.TestCase):
    def _result(self, se: float, min_margin: float, k_sigma: float = 2.0) -> DuelResult:
        return DuelResult("c", "k", 0.0015, se, 0.0015 / se if se else 0.0,
                          k_sigma, False, 1300, min_margin=min_margin)

    def test_absolute_is_the_configured_pair(self):
        self.assertEqual(near_miss_window(self._result(0.0007, 0.002), "absolute",
                                          0.001, 0.003), (0.001, 0.003))

    def test_bar_mode_matches_absolute_at_todays_delta(self):
        lo, hi = near_miss_window(self._result(0.0007, 0.002), "bar")
        self.assertAlmostEqual(lo, 0.001)
        self.assertAlmostEqual(hi, 0.003)

    def test_bar_mode_follows_the_binding_leg(self):
        # δ decayed to 0.0001: bar = 2·SE = 0.0014 → window (0.0007, 0.0021).
        lo, hi = near_miss_window(self._result(0.0007, 0.0001), "bar")
        self.assertAlmostEqual(lo, 0.0007)
        self.assertAlmostEqual(hi, 0.0021)

    def test_bar_mode_without_se_falls_back(self):
        r = DuelResult("c", "k", 0.0, float("inf"), 0.0, 2.0, False, 1)
        self.assertEqual(near_miss_window(r, "bar", 0.001, 0.003), (0.001, 0.003))
        with self.assertRaises(ValueError):
            near_miss_window(r, "sideways")


class ConfigTests(unittest.TestCase):
    def test_shipped_toml_is_the_live_contract(self):
        """Pre-flip (wvk ≤ 14) the shipped toml is today's fixed δ. From the
        wvk-15 flip (operator directive 2026-09-12 15:27 UTC) it is the
        decaying margin: reset-to-cap, 48 h linear, block clock, "bar"
        near-miss window, plus one tie safeguard (min_z or a δ floor of
        half the cap). Either state must be exactly one of these two."""
        cfg = load_config(REPO / "affine.toml")
        d = cfg.duel
        self.assertEqual(d.min_margin, DEFAULT_MIN_MARGIN)
        self.assertEqual(d.min_margin_peak_cap, d.min_margin)
        sched = d.margin_schedule()
        if d.min_margin_mode == "fixed":
            self.assertLessEqual(cfg.weight_version_key, 14)
            self.assertEqual(d.min_z, 0.0)
            self.assertEqual(d.near_miss_window_mode, "absolute")
            self.assertEqual(sched.effective(0.0001, 10**6), d.min_margin)
            self.assertEqual(sched.next_peak(0.0), d.min_margin)
            return
        self.assertEqual(d.min_margin_mode, "decay")
        self.assertGreaterEqual(cfg.weight_version_key, 15)
        self.assertFalse(d.min_margin_double_on_crown)
        self.assertEqual(d.min_margin_decay_hours, 48.0)
        self.assertEqual(d.min_margin_decay_shape, "linear")
        self.assertEqual(d.near_miss_window_mode, "bar")
        self.assertTrue(d.min_z >= 2.5 or d.min_margin_floor >= 0.001,
                        "wvk 15 ships with a tie safeguard: min_z ≥ 2.5 or floor ≥ 0.001")
        self.assertEqual(sched.effective(None, 0), d.min_margin_peak_cap)
        self.assertAlmostEqual(sched.effective(None, 48 * BLOCKS_PER_HOUR), d.min_margin_floor)
        self.assertEqual(sched.next_peak(d.min_margin_floor), d.min_margin_peak_cap)

    def test_decay_knobs_parse_and_validate(self):
        src = (REPO / "affine.toml").read_text()
        fixed = src.replace('min_margin_mode = "decay"', 'min_margin_mode = "fixed"')
        flipped = fixed.replace('min_margin_mode = "fixed"', 'min_margin_mode = "decay"')
        self.assertNotEqual(fixed, flipped)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "affine.toml"
            p.write_text(flipped)
            d = load_config(p).duel
            self.assertEqual(d.margin_schedule().mode, "decay")
            self.assertAlmostEqual(d.margin_schedule().effective(
                0.002, 48 * BLOCKS_PER_HOUR), d.min_margin_floor)
            floor_line = f"min_margin_floor = {d.min_margin_floor:g}"
            self.assertIn(floor_line, flipped)
            p.write_text(flipped.replace(floor_line, "min_margin_floor = 0.5"))
            with self.assertRaises(ValueError):
                load_config(p)
            minz_line = f"min_z = {load_config(REPO / 'affine.toml').duel.min_z:g}"
            if minz_line == "min_z = 0":
                minz_line = "min_z = 0.0"
            self.assertIn(minz_line, src)
            p.write_text(src.replace(minz_line, "min_z = -1"))
            with self.assertRaises(ValueError):
                load_config(p)


class PodStampTests(unittest.TestCase):
    DUEL_CFG = {"min_margin": 0.002, "near_miss_enabled": True,
                "near_miss_low": 0.001, "near_miss_high": 0.003,
                "near_miss_extra_slices": 1}

    def test_no_context_stamps_fixed_toml_delta(self):
        stamp = margin_stamp_for(self.DUEL_CFG, None)
        self.assertEqual(stamp, {"min_margin_mode": "fixed", "min_margin_base": 0.002,
                                 "min_margin_effective": 0.002})

    def test_context_overrides_delta_and_carries_the_clock(self):
        ctx = {"min_margin_mode": "decay", "min_margin_base": 0.002,
               "min_margin_effective": 0.00105, "min_margin_peak": 0.002,
               "crown_block": 9_000_000, "decision_block": 9_007_200,
               "blocks_since_crown": 7_200, "crown_block_source": "king",
               "not_a_stamp_key": "dropped"}
        stamp = margin_stamp_for(self.DUEL_CFG, ctx)
        self.assertEqual(stamp["min_margin_effective"], 0.00105)
        self.assertEqual(stamp["min_margin_mode"], "decay")
        self.assertEqual(stamp["blocks_since_crown"], 7_200)
        self.assertNotIn("not_a_stamp_key", stamp)
        for bad in ({"min_margin_effective": None}, {"min_margin_effective": -1},
                    {"min_margin_effective": float("nan")}):
            with self.assertRaises(ValueError):
                margin_stamp_for(self.DUEL_CFG, bad)

    def test_near_miss_settings_window_mode(self):
        self.assertEqual(near_miss_settings(self.DUEL_CFG)["window_mode"], "absolute")
        self.assertEqual(near_miss_settings({**self.DUEL_CFG, "near_miss_window_mode": "bar"})
                         ["window_mode"], "bar")
        with self.assertRaises(ValueError):
            near_miss_settings({**self.DUEL_CFG, "near_miss_window_mode": "nope"})


class ValidatorSideTests(unittest.TestCase):
    """The validator's δ context and post-verdict guard, driven through a
    stub `self` (no chain, no wallet)."""

    def _stub(self, mode: str, block: int, min_z: float = 0.0) -> SimpleNamespace:
        # Pin the knobs these tests reason about, so they hold before and
        # after the shipped toml flips to decay mode.
        dcfg = replace(load_config(REPO / "affine.toml").duel,
                       min_margin_mode=mode, min_z=min_z,
                       min_margin_peak_cap=0.002, min_margin_floor=0.0001,
                       min_margin_decay_hours=48.0, min_margin_decay_shape="linear",
                       min_margin_double_on_crown=True, min_margin_double_factor=2.0)
        return SimpleNamespace(cfg=SimpleNamespace(duel=dcfg),
                               subtensor=SimpleNamespace(block=block))

    def _king(self, **kw) -> King:
        base = dict(hotkey="hk", repo="r", revision="v", block=8_990_000,
                    challenge_id="chal-1", reign_number=3, crowned_at="")
        return King(**{**base, **kw})

    def test_fixed_mode_context_is_todays_delta(self):
        ctx = Validator._margin_context(self._stub("fixed", 9_000_000), self._king())
        self.assertEqual(ctx["min_margin_mode"], "fixed")
        self.assertEqual(ctx["min_margin_effective"], 0.002)
        self.assertEqual(ctx["crown_block_source"], "reveal_block")
        self.assertNotIn("min_margin_peak", ctx)

    def test_decay_mode_reads_the_kings_cycle(self):
        king = self._king(crown_block=9_000_000, min_margin_peak=0.002)
        stub = self._stub("decay", 9_000_000 + 24 * BLOCKS_PER_HOUR)
        ctx = Validator._margin_context(stub, king)
        self.assertEqual(ctx["crown_block_source"], "king")
        self.assertEqual(ctx["blocks_since_crown"], 24 * BLOCKS_PER_HOUR)
        self.assertAlmostEqual(ctx["min_margin_effective"], 0.00105)  # 48 h linear
        cycle = Validator._crown_cycle(stub, ctx)
        self.assertEqual(cycle["crown_block"], 9_000_000 + 24 * BLOCKS_PER_HOUR)
        self.assertAlmostEqual(cycle["min_margin_peak"], 0.002)       # 2·0.00105 capped

    def test_decay_mode_fails_closed_without_a_block(self):
        with self.assertRaises(BlockHashUnavailable):
            Validator._margin_context(self._stub("decay", 0), self._king())
        # Fixed mode does not need the clock.
        ctx = Validator._margin_context(self._stub("fixed", 0), self._king())
        self.assertIsNone(ctx["decision_block"])

    def test_crown_bar_guard_denies_but_never_grants(self):
        stub = self._stub("fixed", 9_000_000)
        ctx = {"min_margin_effective": 0.002}
        # Pod said win at margin 0.0015 (a stale pod with δ=0.001): denied.
        v = {"challenger_wins": True, "margin": 0.0015, "se": 0.0005, "z": 3.0,
             "duel_params": {"min_margin": 0.001}}
        Validator._apply_crown_bar(stub, v, ctx)
        self.assertFalse(v["challenger_wins"])
        self.assertEqual(v["rejection_reason"], "margin_below_bar")
        # Pod said loss at a margin that clears the effective bar: stays a loss.
        v = {"challenger_wins": False, "margin": 0.0030, "se": 0.0005, "z": 6.0,
             "duel_params": {"min_margin": 0.002}}
        Validator._apply_crown_bar(stub, v, ctx)
        self.assertFalse(v["challenger_wins"])
        self.assertNotIn("rejection_reason", v)
        # min_z guard.
        stub = self._stub("fixed", 9_000_000, min_z=2.5)
        v = {"challenger_wins": True, "margin": 0.0025, "se": 0.0011, "z": 2.27,
             "duel_params": {"min_margin": 0.002}}
        Validator._apply_crown_bar(stub, v, ctx)
        self.assertFalse(v["challenger_wins"])
        self.assertEqual(v["rejection_reason"], "z_below_min")


class ReplayTests(unittest.TestCase):
    T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)

    def _duel(self, hours: float, margin: float, se: float, cid: str,
              actual_win: bool = False, gate: bool = False) -> Duel:
        return Duel(challenge_id=cid, at=self.T0 + timedelta(hours=hours),
                    hotkey=f"hk-{cid}", margin=margin, se=se, z=margin / se,
                    n_paired_turns=1300, gate_blocked=gate, actual_win=actual_win)

    def test_fixed_replay_reproduces_actual_crowns(self):
        duels = [self._duel(1, 0.0015, 0.0006, "a"),             # z 2.5, < δ
                 self._duel(5, 0.0025, 0.0007, "b", actual_win=True),
                 self._duel(9, 0.0030, 0.0007, "c", gate=True)]  # gate-blocked
        rep = run_replay(duels, MarginSchedule(min_margin=0.002, mode="fixed"),
                         origin=self.T0)
        self.assertEqual([c.challenge_id for c in rep.crowns], ["b"])
        self.assertEqual(rep.n_actual_lost, 0)
        self.assertEqual(rep.n_delta_binding, 3)

    def test_decay_crowns_a_near_miss_once_delta_has_fallen(self):
        sched = MarginSchedule(mode="decay", peak_cap=0.002, floor=0.0001,
                               decay_hours=24.0, shape="linear")
        # margin 0.0015, z 2.5 — blocked while δ > 0.0015 (first ~6.3 h), not after.
        duels = [self._duel(2, 0.0015, 0.0006, "early"),
                 self._duel(12, 0.0015, 0.0006, "late")]
        rep = run_replay(duels, sched, origin=self.T0, mode="chained")
        self.assertEqual([c.challenge_id for c in rep.crowns], ["late"])
        crown = rep.crowns[0]
        self.assertAlmostEqual(crown.delta, 0.002 - 0.0019 * 12 / 24)
        self.assertAlmostEqual(crown.hours_since_crown, 12.0)
        self.assertAlmostEqual(crown.peak_after, min(0.002, 2 * crown.delta))

    def test_chained_restarts_clock_but_actual_clock_does_not(self):
        sched = MarginSchedule(mode="decay", peak_cap=0.002, floor=0.0001,
                               decay_hours=24.0, shape="linear",
                               double_on_crown=False)
        # A replay-only crown at 20 h, then a 0.0016 margin at 21 h: with the
        # clock restarted (chained) δ is back at the cap and it loses; on the
        # actual clock δ is near the floor and it wins.
        duels = [self._duel(20, 0.0016, 0.0006, "first"),
                 self._duel(21, 0.0016, 0.0006, "second")]
        chained = run_replay(duels, sched, origin=self.T0, mode="chained")
        actual = run_replay(duels, sched, origin=self.T0, mode="actual-clock")
        self.assertEqual([c.challenge_id for c in chained.crowns], ["first"])
        self.assertEqual([c.challenge_id for c in actual.crowns], ["first", "second"])

    def test_min_z_counts_blocked_ties_and_lost_actual_crowns(self):
        sched = MarginSchedule(mode="decay", peak_cap=0.002, floor=0.0001,
                               decay_hours=24.0)
        duels = [self._duel(30, 0.0016, 0.0007, "tie"),                 # z 2.29
                 self._duel(31, 0.0025, 0.0009, "real", actual_win=True)]  # z 2.78
        rep = run_replay(duels, sched, origin=self.T0, min_z=3.0)
        self.assertEqual(rep.crowns, [])
        self.assertEqual(rep.n_blocked_by_min_z, 2)
        self.assertEqual(rep.n_actual_lost, 1)

    def test_load_duels_reads_history_rows(self):
        rows = [
            {"event": "crowned", "at": "2026-08-27T20:00:00+00:00",
             "challenge_id": "seed", "hotkey": "", "reign_number": 0},
            {"event": "verdict", "at": "2026-08-28T01:00:00+00:00",
             "challenge_id": "chal-00001", "hotkey": "hk1",
             "verdict": {"margin": 0.0012, "se": 0.0006, "z": 2.0,
                         "n_paired_turns": 1290, "rejection_reason": None}},
            {"event": "verdict", "at": "2026-08-28T02:00:00+00:00",
             "challenge_id": "chal-00002", "hotkey": "hk2",
             "verdict": {"challenger_wins": False,
                         "rejection_reason": "protocol_probe"}},   # no margin
            {"event": "failed", "at": "2026-08-28T03:00:00+00:00",
             "challenge_id": "chal-00003", "hotkey": "hk3"},
            {"event": "crowned", "at": "2026-08-28T04:00:00+00:00",
             "challenge_id": "chal-00004", "hotkey": "hk4", "reign_number": 1,
             "verdict": {"margin": 0.0025, "se": 0.0004, "z": 6.25,
                         "n_paired_turns": 1300, "rejection_reason": None}},
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "history.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            duels = load_duels(p)
            self.assertEqual([d.challenge_id for d in duels],
                             ["chal-00001", "chal-00004"])
            self.assertTrue(duels[1].actual_win)
            self.assertTrue(duels[1].near_copy)   # se 0.0004 < 0.0005
            self.assertEqual(duels[1].actual_reign, 1)
            self.assertEqual(seed_time(p),
                             datetime(2026, 8, 27, 20, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
