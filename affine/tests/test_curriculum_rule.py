"""Rule v1 of the adaptive curriculum (ops/curriculum/rule.py): shrinkage
chain, weight formula, group floors (per-group half-static + joint
coding+terminal), cap, +-0.05 clamp, multiplicity and determinism.

    source .venv/bin/activate
    python -m unittest tests.test_curriculum_rule -v                # from affine/
    python -m unittest discover -s affine/tests -t affine           # from the repo root
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

RULE_PATH = Path(__file__).resolve().parents[2] / "ops" / "curriculum" / "rule.py"
_spec = importlib.util.spec_from_file_location("curriculum_rule", RULE_PATH)
rule = importlib.util.module_from_spec(_spec)
sys.modules["curriculum_rule"] = rule
_spec.loader.exec_module(rule)

KNOBS = dict(floor_frac=0.5, floor_ct=0.40, cap=0.60, max_shift=0.05)
STATIC = {"coding": 0.36, "terminal": 0.22, "math": 0.02, "tool_use": 0.02, "nl2repo": 0.01,
          "king_fail": 0.09, "completion": 0.10, "general": 0.05, "king_loop_onset": 0.04,
          "king_pivot": 0.02, "king_recoverable": 0.02, "king_done": 0.01, "king_tooluse": 0.02,
          "completion_pre": 0.02}


def _strata():
    return {
        "a": {"group": "coding", "cell": "c1", "n_w": 10.0, "M": 0.5, "S": 0.9, "w": 0.0, "n_turns": 5},
        "b": {"group": "coding", "cell": "c1", "n_w": 0.0, "M": None, "S": None, "w": 0.0, "n_turns": 1},
        "c": {"group": "king_fail", "cell": "k1", "n_w": 2.0, "M": 1.0, "S": 1.0, "w": 0.0, "n_turns": 9},
    }


class ShrinkageTests(unittest.TestCase):
    def test_prior_when_unobserved(self):
        self.assertEqual(rule.shrink(0, None, 0.3, 8), 0.3)
        self.assertEqual(rule.shrink(0, 0.9, 0.3, 8), 0.3)

    def test_moves_toward_value_with_n(self):
        self.assertAlmostEqual(rule.shrink(8, 1.0, 0.0, 8), 0.5)
        self.assertAlmostEqual(rule.shrink(24, 1.0, 0.0, 8), 0.75)

    def test_chain_stratum_cell_group_corpus(self):
        strata = _strata()
        cells = {"c1": {"group": "coding", "n_w": 10.0, "M": 0.5, "S": 0.9},
                 "k1": {"group": "king_fail", "n_w": 2.0, "M": 1.0, "S": 1.0}}
        groups = {"coding": {"n_w": 10.0, "M": 0.5, "S": 0.9}, "king_fail": {"n_w": 2.0, "M": 1.0, "S": 1.0}}
        rule.shrunk_rates(strata, cells, groups, n0=8, corpus_s=0.85)
        # unobserved stratum b takes exactly its cell's shrunk value
        g_m = rule.shrink(10, 0.5, 0.25, 8)
        c_m = rule.shrink(10, 0.5, g_m, 8)
        self.assertAlmostEqual(strata["b"]["M_t"], c_m)
        self.assertAlmostEqual(strata["a"]["M_t"], rule.shrink(10, 0.5, c_m, 8))
        # a thin king stratum is pulled toward its group, not left at 1.0
        self.assertLess(strata["c"]["M_t"], 1.0)
        self.assertGreater(strata["c"]["M_t"], 0.5)

    def test_probe_prior_for_unscored_stratum_only(self):
        strata = _strata()
        rule.shrunk_rates(strata, {}, {}, n0=8, corpus_s=0.85, probe_s={"b": 0.2, "a": 0.2})
        self.assertAlmostEqual(strata["b"]["S_t"], 0.2)          # unscored -> probe prior
        self.assertNotAlmostEqual(strata["a"]["S_t"], 0.2)       # scored -> ignores the probe


class WeightTests(unittest.TestCase):
    def test_formula(self):
        self.assertAlmostEqual(rule.stratum_weight(0.5, 0.8, eps=0.02, gamma=1.0), 0.52 * 0.8)
        self.assertAlmostEqual(rule.stratum_weight(0.5, 0.8, eps=0.02, gamma=2.0), 0.52 ** 2 * 0.8)

    def test_dead_meter_gets_no_weight(self):
        self.assertEqual(rule.stratum_weight(0.9, 0.0, eps=0.02, gamma=1.0), 0.0)


class RuleV12Tests(unittest.TestCase):
    def test_forfeit_plus_scaled_gap_with_gate(self):
        # corpus mean forfeit 0.03, corpus mean Dbar+ 0.006 -> scale 5: units match
        m12, w = rule.stratum_weight_v12(0.10, 0.010, 0.9, dplus_scale=5.0, eps=0.02, gamma=1.0, s_gate=0.5)
        self.assertAlmostEqual(m12, 0.10 + 0.05)
        self.assertAlmostEqual(w, 0.17)
        # ref-dead stratum: gated out, the deficit is not observable there
        _, w_dead = rule.stratum_weight_v12(0.10, 0.010, 0.2, dplus_scale=5.0, eps=0.02, gamma=1.0, s_gate=0.5)
        self.assertEqual(w_dead, 0.0)
        # a negative mean gap never subtracts
        m12n, _ = rule.stratum_weight_v12(0.0, -0.4, 0.9, dplus_scale=5.0, eps=0.02, gamma=1.0, s_gate=0.5)
        self.assertEqual(m12n, 0.0)

    def test_shrink_field_chain(self):
        strata = {"a": {"group": "g", "cell": "c", "n_w": 0.0, "forfeit_rate": None},
                  "b": {"group": "g", "cell": "c", "n_w": 8.0, "forfeit_rate": 0.5}}
        cells = {"c": {"group": "g", "n_w": 8.0, "forfeit_rate": 0.5}}
        groups = {"g": {"n_w": 8.0, "forfeit_rate": 0.5}}
        rule.shrink_field(strata, cells, groups, field="forfeit_rate", n_field="n_w", out="F_t",
                          corpus_prior=0.1, n0=8)
        g_t = rule.shrink(8, 0.5, 0.1, 8)          # 0.3
        c_t = rule.shrink(8, 0.5, g_t, 8)          # 0.4
        self.assertAlmostEqual(strata["a"]["F_t"], c_t)
        self.assertAlmostEqual(strata["b"]["F_t"], rule.shrink(8, 0.5, c_t, 8))


class GroupVectorTests(unittest.TestCase):
    def _vec(self, raw, current, static=STATIC, **over):
        return rule.group_vector(raw, static, current, **{**KNOBS, **over})

    def test_floor_half_static(self):
        raw = {g: 0.0 for g in STATIC}
        raw["coding"] = 0.9
        raw["math"] = 0.1
        # math is at raw 0.1; every other group has supply (current > 0) but raw 0
        current = {g: 1.0 / len(STATIC) for g in STATIC}
        v = self._vec(raw, current, max_shift=1.0)
        for g, s in STATIC.items():
            self.assertGreaterEqual(v["after_floor"][g], 0.5 * s - 1e-9, g)
        self.assertAlmostEqual(sum(v["after_floor"].values()), 1.0)
        self.assertEqual(v["reasons"]["king_fail"], "floor")

    def test_joint_coding_terminal_floor(self):
        raw = {g: 0.0 for g in STATIC}
        raw.update({"coding": 0.10, "terminal": 0.05, "king_fail": 0.85})
        current = {g: 1.0 / len(STATIC) for g in STATIC}
        v = self._vec(raw, current, max_shift=1.0)
        ct = v["after_floor"]["coding"] + v["after_floor"]["terminal"]
        self.assertGreaterEqual(ct, 0.40 - 1e-9)
        self.assertTrue(v["joint_floor_applied"])
        self.assertEqual(v["reasons"]["coding"], "joint_floor")
        self.assertLessEqual(v["after_floor"]["king_fail"], 0.60 + 1e-9)
        self.assertAlmostEqual(sum(v["after_floor"].values()), 1.0)

    def test_cap(self):
        raw = {g: 0.0 for g in STATIC}
        raw["coding"] = 1.0
        current = {g: 1.0 / len(STATIC) for g in STATIC}
        v = self._vec(raw, current, max_shift=1.0)
        self.assertAlmostEqual(v["after_floor"]["coding"], 0.60)
        self.assertEqual(v["reasons"]["coding"], "capped")
        self.assertAlmostEqual(sum(v["after_clamp"].values()), 1.0)

    def test_clamp_five_points_per_fold(self):
        raw = {g: 0.0 for g in STATIC}
        raw.update({"coding": 0.20, "terminal": 0.20, "king_fail": 0.60})
        current = {"coding": 0.36, "terminal": 0.22, "king_fail": 0.09, "completion": 0.10, "general": 0.05,
                   "math": 0.02, "tool_use": 0.02, "nl2repo": 0.01, "king_loop_onset": 0.04, "king_pivot": 0.02,
                   "king_recoverable": 0.02, "king_done": 0.01, "king_tooluse": 0.02, "completion_pre": 0.02}
        v = self._vec(raw, current)
        for g in current:
            self.assertLessEqual(abs(v["after_clamp"][g] - current[g]), 0.05 + 1e-9, g)
        self.assertAlmostEqual(sum(v["after_clamp"].values()), 1.0)
        self.assertAlmostEqual(v["after_clamp"]["king_fail"], 0.14)
        self.assertEqual(v["reasons"]["king_fail"], "clamped_up")
        # the mass king_fail could not take stays with the groups that wanted it least badly
        self.assertGreaterEqual(v["after_clamp"]["coding"], 0.31 - 1e-9)

    def test_clamp_down_reason(self):
        raw = {"coding": 0.10, "terminal": 0.90}
        current = {"coding": 0.70, "terminal": 0.30}
        v = self._vec(raw, current, static={"coding": 0.5, "terminal": 0.5})
        self.assertAlmostEqual(v["after_clamp"]["coding"], 0.65)
        self.assertEqual(v["reasons"]["coding"], "clamped_down")
        self.assertEqual(v["reasons"]["terminal"], "clamped_up")

    def test_no_supply_stays_zero(self):
        raw = {"coding": 0.7, "terminal": 0.3}
        current = {"coding": 0.7, "terminal": 0.3, "nl2repo": 0.0}
        v = self._vec(raw, current, static={"coding": 0.5, "terminal": 0.3, "nl2repo": 0.2})
        self.assertEqual(v["after_clamp"]["nl2repo"], 0.0)
        self.assertEqual(v["reasons"]["nl2repo"], "no_supply")

    def test_floors_check(self):
        good = {"coding": 0.4, "terminal": 0.2, "math": 0.4}
        st = {"coding": 0.5, "terminal": 0.3, "math": 0.2}
        self.assertTrue(rule.check_floors(good, st, floor_frac=0.5, floor_ct=0.40, cap=0.60)["ok"])
        bad = {"coding": 0.7, "terminal": 0.1, "math": 0.2}
        r = rule.check_floors(bad, st, floor_frac=0.5, floor_ct=0.40, cap=0.60)
        self.assertFalse(r["ok"])
        self.assertEqual(r["above_cap"], ["coding"])
        self.assertEqual(r["below_floor"], ["terminal"])


class SliceKeyAggregationTests(unittest.TestCase):
    def test_bucket_weighs_mean_of_its_base_strata(self):
        # coding: 3 base strata merged into ONE bucket; king_fail: one stratum = one key
        strata = {"r1|pr": {"group": "coding", "w": 0.1}, "r2|pr": {"group": "coding", "w": 0.3},
                  "r3|pr": {"group": "coding", "w": 0.2}, "king_fail:0001": {"group": "king_fail", "w": 0.4}}
        index_rows = [{"stratum": "coding:b00001", "stratum_src": "r1|pr"},
                      {"stratum": "coding:b00001", "stratum_src": "r2|pr"},
                      {"stratum": "coding:b00001", "stratum_src": "r3|pr"},
                      {"stratum": "king_fail:0001#0", "stratum_src": "king_fail:0001"},
                      {"stratum": "king_fail:0001#1", "stratum_src": "king_fail:0001"}]
        by_key = rule.raw_group_shares_by_slice_key(strata, index_rows)
        by_base = rule.raw_group_shares(strata)
        # slice keys: coding 0.2 (one bucket, mean w) vs king_fail 0.4 + 0.4 (two sub-strata)
        self.assertAlmostEqual(by_key["coding"], 0.2 / 1.0)
        self.assertAlmostEqual(by_key["king_fail"], 0.8 / 1.0)
        # base strata: coding 0.6 vs king_fail 0.4 -- the count re-inflates coding
        self.assertAlmostEqual(by_base["coding"], 0.6)
        self.assertGreater(by_base["coding"], by_key["coding"])


class MultiplicityTests(unittest.TestCase):
    def test_rank_to_m_and_caps(self):
        strata = {f"s{i}": {"group": "g", "w": float(i), "n_turns": 9} for i in range(5)}
        strata["s4"]["n_turns"] = 1          # top weight but a single turn
        rule.multiplicity(strata, m_max=3)
        self.assertEqual(strata["s0"]["m"], 1)
        self.assertEqual(strata["s2"]["m"], 2)
        self.assertEqual(strata["s3"]["m"], 3)
        self.assertEqual(strata["s4"]["m"], 1)  # m <= n_turns
        rule.multiplicity(strata, m_max=2)
        self.assertEqual(strata["s3"]["m"], 2)

    def test_recurrence_projection(self):
        strata = {"a": {"group": "g", "w": 1.0, "n_turns": 10, "m": 1},
                  "b": {"group": "g", "w": 2.0, "n_turns": 10, "m": 3}}
        p = rule.recurrence_projection(strata, {"g": 0.5}, slice_n=1300)
        self.assertAlmostEqual(p["groups"]["g"]["expected_draws_per_turn_per_duel"], 650 / 20)
        self.assertAlmostEqual(p["per_stratum"]["b"], 650 * 3 / 4 / 10)
        self.assertEqual(p["max_turn_stratum"], "b")


class DeterminismTests(unittest.TestCase):
    def test_same_inputs_same_outputs(self):
        strata = _strata()
        cells = {"c1": {"group": "coding", "n_w": 10.0, "M": 0.5, "S": 0.9},
                 "k1": {"group": "king_fail", "n_w": 2.0, "M": 1.0, "S": 1.0}}
        groups = {"coding": {"n_w": 10.0, "M": 0.5, "S": 0.9}, "king_fail": {"n_w": 2.0, "M": 1.0, "S": 1.0}}
        outs = []
        for _ in range(2):
            s = copy.deepcopy(strata)
            rule.shrunk_rates(s, copy.deepcopy(cells), copy.deepcopy(groups), n0=8, corpus_s=0.85)
            for rec in s.values():
                rec["w"] = rule.stratum_weight(rec["M_t"], rec["S_t"], eps=0.02, gamma=1.0)
            rule.multiplicity(s, m_max=3)
            raw = rule.raw_group_shares(s)
            v = rule.group_vector(raw, STATIC, {"coding": 0.7, "king_fail": 0.3}, **KNOBS)
            outs.append((s, v))
        self.assertEqual(outs[0], outs[1])

    def test_tie_break_by_key(self):
        strata = {k: {"group": "g", "w": 1.0, "n_turns": 9} for k in ("z", "a", "m")}
        rule.multiplicity(strata, m_max=3)
        self.assertEqual([strata[k]["m"] for k in ("a", "m", "z")], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
