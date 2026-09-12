"""Window-best crown mode (operator rule 2026-09-12 16:39 UTC): the window
clock, candidate ranking, exact pooling, the pod's confirmation stamp, the
state rows, and the validator's close procedure with a stubbed eval pod.

    python -m unittest discover -s affine/tests -t affine
"""

from __future__ import annotations

import json
import statistics
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from affine.config import load_config
from affine.score import (
    DuelResult,
    pooled_margin_stats,
    rank_window_candidates,
    window_candidate_reason,
    window_id_of,
)
from affine.state import King, QueueEntry, State
from affine.validator import Validator
from evalsrv.dueling import confirmation_stamp

REPO = Path(__file__).resolve().parents[1]


class WindowRuleTests(unittest.TestCase):
    def test_window_id_is_block_floor_div(self):
        self.assertEqual(window_id_of(0), 0)
        self.assertEqual(window_id_of(3599), 0)
        self.assertEqual(window_id_of(3600), 1)
        self.assertEqual(window_id_of(9052470), 2514)
        self.assertEqual(window_id_of(9052470, 7200), 1257)
        with self.assertRaises(ValueError):
            window_id_of(1, 0)

    def test_candidate_reasons(self):
        self.assertEqual(window_candidate_reason({"margin": None}), "no_margin")
        self.assertEqual(window_candidate_reason({"margin": float("nan")}), "no_margin")
        self.assertEqual(window_candidate_reason({"margin": 0.001,
                                                  "rejection_reason": "causality_fail"}),
                         "gate:causality_fail")
        self.assertEqual(window_candidate_reason({"margin": 0.0}), "margin_not_positive")
        self.assertEqual(window_candidate_reason({"margin": -0.01}), "margin_not_positive")
        self.assertIsNone(window_candidate_reason({"margin": 0.0004, "rejection_reason": None}))

    def test_ranking_filters_sorts_and_dedupes_hotkeys(self):
        vs = [
            {"challenge_id": "chal-1", "hotkey": "A", "margin": 0.0012, "z": 1.5},
            {"challenge_id": "chal-2", "hotkey": "B", "margin": 0.0030, "z": 3.1},
            {"challenge_id": "chal-3", "hotkey": "A", "margin": 0.0025, "z": 2.9},
            {"challenge_id": "chal-4", "hotkey": "C", "margin": -0.002, "z": -2.0},
            {"challenge_id": "chal-5", "hotkey": "D", "margin": None},
            {"challenge_id": "chal-6", "hotkey": "E", "margin": 0.004,
             "rejection_reason": "thought_too_short"},
        ]
        ranked, dropped = rank_window_candidates(vs, True)
        self.assertEqual([r["challenge_id"] for r in ranked], ["chal-2", "chal-3"])
        reasons = {d["challenge_id"]: d["reason"] for d in dropped}
        self.assertEqual(reasons, {"chal-4": "margin_not_positive", "chal-5": "no_margin",
                                   "chal-6": "gate:thought_too_short",
                                   "chal-1": "hotkey_duplicate"})
        ranked_all, _ = rank_window_candidates(vs, False)
        self.assertEqual([r["challenge_id"] for r in ranked_all],
                         ["chal-2", "chal-3", "chal-1"])

    def test_ranking_ties_are_deterministic(self):
        vs = [{"challenge_id": "chal-9", "hotkey": "A", "margin": 0.002, "z": 2.0},
              {"challenge_id": "chal-8", "hotkey": "B", "margin": 0.002, "z": 2.0},
              {"challenge_id": "chal-7", "hotkey": "C", "margin": 0.002, "z": 2.5}]
        ranked, _ = rank_window_candidates(vs)
        self.assertEqual([r["challenge_id"] for r in ranked], ["chal-7", "chal-8", "chal-9"])

    def test_pooled_stats_match_the_concatenated_rows(self):
        a = [0.001 * ((i * 7) % 11) - 0.003 for i in range(1300)]
        b = [0.0005 * ((i * 13) % 9) - 0.001 for i in range(1300)]
        n1, m1, se1 = len(a), statistics.mean(a), statistics.stdev(a) / len(a) ** 0.5
        n2, m2, se2 = len(b), statistics.mean(b), statistics.stdev(b) / len(b) ** 0.5
        N, M, se, z = pooled_margin_stats(n1, m1, se1, n2, m2, se2)
        both = a + b
        self.assertEqual(N, 2600)
        self.assertAlmostEqual(M, statistics.mean(both), places=12)
        self.assertAlmostEqual(se, statistics.stdev(both) / len(both) ** 0.5, places=12)
        self.assertAlmostEqual(z, M / se, places=9)
        with self.assertRaises(ValueError):
            pooled_margin_stats(0, 0.0, 0.0, 10, 0.0, 0.0)


def _result(margin: float, se: float, n: int = 1300, **kw) -> DuelResult:
    z = margin / se if se else 0.0
    base = dict(challenger="c", king="k", margin=margin, se=se, z=z,
                k_sigma=2.0, challenger_wins=False, n_paired_turns=n)
    base.update(kw)
    return DuelResult(**base)


class ConfirmationStampTests(unittest.TestCase):
    SLICE = {"index": 1, "seed": 42, "n": 1300, "digest": "abc"}

    def test_pooled_positive_passes(self):
        conf = {"challenge_id": "chal-1", "slice_index": 1,
                "base": {"n": 1300, "margin": 0.0020, "se": 0.0008}}
        stamp = confirmation_stamp(conf, self.SLICE, _result(0.0004, 0.0009))
        self.assertTrue(stamp["passed"])
        self.assertEqual(stamp["pooled"]["n"], 2600)
        self.assertAlmostEqual(stamp["pooled"]["margin"], 0.0012)
        self.assertEqual(stamp["slice"]["digest"], "abc")

    def test_pooled_negative_fails(self):
        conf = {"challenge_id": "chal-1", "slice_index": 1,
                "base": {"n": 1300, "margin": 0.0004, "se": 0.0008}}
        stamp = confirmation_stamp(conf, self.SLICE, _result(-0.0010, 0.0009))
        self.assertFalse(stamp["passed"])
        self.assertLess(stamp["pooled"]["margin"], 0)

    def test_gate_or_missing_numbers_fail(self):
        conf = {"challenge_id": "chal-1", "slice_index": 1,
                "base": {"n": 1300, "margin": 0.003, "se": 0.0008}}
        stamp = confirmation_stamp(conf, self.SLICE,
                                   _result(0.003, 0.0008, causality_blocked=True))
        self.assertFalse(stamp["passed"])
        self.assertIsNone(stamp["pooled"])
        stamp = confirmation_stamp({"challenge_id": "x", "base": {}}, self.SLICE,
                                   _result(0.003, 0.0008))
        self.assertFalse(stamp["passed"])


class ConfigTests(unittest.TestCase):
    def test_shipped_knobs(self):
        d = load_config(REPO / "affine.toml").duel
        self.assertIn(d.crown_mode, ("duel", "window_best"))
        self.assertEqual(d.crown_window_blocks, 3600)
        self.assertTrue(d.crown_confirm_slice)
        self.assertEqual(d.crown_confirm_max, 2)
        self.assertTrue(d.crown_one_entry_per_hotkey)

    def test_validation(self):
        src = (REPO / "affine.toml").read_text()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "affine.toml"
            for bad in ('crown_mode = "duel"\n', 'crown_mode = "window_best"\n'):
                if bad in src:
                    p.write_text(src.replace(bad, 'crown_mode = "lottery"\n'))
                    with self.assertRaises(ValueError):
                        load_config(p)
            p.write_text(src.replace("crown_window_blocks = 3600", "crown_window_blocks = 0"))
            with self.assertRaises(ValueError):
                load_config(p)
            p.write_text(src.replace("crown_confirm_max = 2", "crown_confirm_max = 0"))
            with self.assertRaises(ValueError):
                load_config(p)


class StateTests(unittest.TestCase):
    def _state(self, td: str) -> State:
        s = State(Path(td))
        s.king = King(hotkey="K", repo="r", revision="v", block=9_000_000,
                      challenge_id="chal-0", reign_number=11, crowned_at="")
        return s

    def test_window_verdict_rows_and_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            s = self._state(td)
            s.open_crown_window(2514, 3600, 9_052_470)
            e = QueueEntry("chal-5", "H", "r2://b/p/", "digest", 9_052_000, "")
            v = {"margin": 0.0015, "se": 0.0007, "z": 2.1, "n_paired_turns": 1290,
                 "rejection_reason": None, "duel_rule_wins": False,
                 "crown_mode": "window_best", "window_id": 2514,
                 "decision_block": 9_052_470, "challenger_wins": False,
                 "challenger": {"reason": 0.0123}, "slice": {}}
            s.record_window_verdict(e, v, uid=7, duration_s=1800)
            self.assertIsNone(s.king.previous or None)  # no crown happened
            self.assertEqual(s.king.reign_number, 11)
            cw = s.crown_window
            self.assertEqual(len(cw["verdicts"]), 1)
            self.assertEqual(cw["verdicts"][0]["score"], 0.0123)
            self.assertEqual(cw["verdicts"][0]["n_slices"], 1)
            s.flush()
            s2 = State(Path(td)); s2.load()
            self.assertEqual(s2.crown_window["window_id"], 2514)
            self.assertEqual(s2.crown_window["verdicts"][0]["challenge_id"], "chal-5")
            rows = [json.loads(l) for l in open(s.history_path)]
            self.assertEqual([r["event"] for r in rows], ["verdict"])
            self.assertFalse(rows[0]["accepted"])
            self.assertEqual(rows[0]["verdict"]["crown_mode"], "window_best")
            s.record_window_close({"window_id": 2514, "outcome": "king_stays_no_candidates"})
            rows = [json.loads(l) for l in open(s.history_path)]
            self.assertEqual(rows[-1]["event"], "window_close")


class _FakeEval:
    """Stub eval pod: returns a confirmation stamp per challenge id."""

    def __init__(self, outcomes: dict[str, bool]):
        self.outcomes = outcomes
        self.calls: list[dict] = []

    async def run_duel(self, **kw):
        self.calls.append(kw)
        cid = kw["confirm"]["challenge_id"]
        passed = self.outcomes[cid]
        base = kw["confirm"]["base"]
        return {"challenger_wins": False, "rejection_reason": None,
                "margin": 0.0003 if passed else -0.004, "se": 0.0009,
                "job_id": f"job-{cid}",
                "confirmation": {"challenge_id": cid, "slice_index": kw["confirm"]["slice_index"],
                                 "base": base, "slice": {"index": 1},
                                 "pooled": {"n": 2600, "margin": 0.001 if passed else -0.001,
                                            "se": 0.0006, "z": 1.6 if passed else -1.6},
                                 "passed": passed}}

    async def fetch_artifact(self, job_id):
        return None


class _Stub(SimpleNamespace):
    """Just enough of `Validator` for `_close_window`; the real methods are
    bound onto it from the class."""

    async def _maybe_set_weights(self, force=False):
        self.weights_set = True

    async def _publish_eval_artifact(self, entry, verdict):
        self.published.append(entry.challenge_id)

    def _promote_or_none(self, entry):
        return f"r2://public/models/sha256/{entry.revision}/"

    def _margin_context(self, king):
        return {"min_margin_mode": "fixed", "min_margin_base": 0.002,
                "min_margin_effective": 0.002, "decision_block": self.subtensor.block,
                "crown_block": king.block, "crown_block_source": "reveal_block",
                "blocks_since_crown": self.subtensor.block - king.block}


def _stub(td: str, outcomes: dict[str, bool], block: int, **cfg_over) -> _Stub:
    cfg = load_config(REPO / "affine.toml")
    over = dict(crown_mode="window_best", crown_window_blocks=3600,
                crown_confirm_slice=True, crown_confirm_max=2,
                crown_one_entry_per_hotkey=True)
    over.update(cfg_over)
    dcfg = replace(cfg.duel, **over)
    state = State(Path(td))
    state.king = King(hotkey="K", repo="r2://public/models/sha256/king/", revision="king",
                      block=9_033_530, challenge_id="chal-00409", reign_number=11,
                      crowned_at="")
    stub = _Stub(cfg=SimpleNamespace(duel=dcfg, secrets=SimpleNamespace(hf_token="")),
                 state=state, subtensor=SimpleNamespace(block=block),
                 eval_client=_FakeEval(outcomes), r2_reader=None,
                 watchdog=SimpleNamespace(beat=lambda: None),
                 dashboard=SimpleNamespace(flush=lambda force=False: None),
                 bench=SimpleNamespace(enqueue_for=lambda *a, **k: None),
                 published=[], weights_set=False)
    for name in ("_window_due", "_close_window", "_close_window_safely", "_finalize_window",
                 "_confirm_candidate", "_window_row", "_stamp_window_verdict"):
        setattr(stub, name, getattr(Validator, name).__get__(stub))
    stub.WINDOW_CLOSE_MAX_ATTEMPTS = Validator.WINDOW_CLOSE_MAX_ATTEMPTS
    return stub


def _file(stub: _Stub, cid: str, hotkey: str, margin: float, se: float = 0.0008,
          block: int = 9_052_000, reason=None, n_slices: int = 1) -> None:
    e = QueueEntry(cid, hotkey, f"r2://private/models/registrations/{cid}/", f"rev-{cid}",
                   block, "")
    v = {"challenger_wins": margin > 0.002, "rejection_reason": reason, "margin": margin,
         "se": se, "z": margin / se, "n_paired_turns": 1300, "block_hash": "0xabc",
         "slice": ({"extra_slices": [{"index": 1}]} if n_slices == 2 else {}),
         "challenger": {"reason": 0.01}}
    stub._stamp_window_verdict(v, {"decision_block": block})
    stub.state.record_window_verdict(e, v, uid=1)


class ValidatorWindowTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_tick_opens_window_not_due(self):
        with tempfile.TemporaryDirectory() as td:
            stub = _stub(td, {}, 9_052_470)
            due, block = stub._window_due()
            self.assertFalse(due)
            self.assertEqual(stub.state.crown_window["window_id"], 2514)
            self.assertEqual(stub.state.crown_window["king_reign"], 11)
            stub.subtensor.block = 2515 * 3600
            due, _ = stub._window_due()
            self.assertTrue(due)

    async def test_stamp_marks_candidate_and_forces_no_crown(self):
        with tempfile.TemporaryDirectory() as td:
            stub = _stub(td, {}, 9_052_470)
            stub._window_due()
            v = {"challenger_wins": True, "margin": 0.003, "se": 0.0008, "z": 3.75,
                 "rejection_reason": None}
            stub._stamp_window_verdict(v, {"decision_block": 9_052_470})
            self.assertFalse(v["challenger_wins"])
            self.assertTrue(v["duel_rule_wins"])
            self.assertEqual(v["window_id"], 2514)
            self.assertEqual(v["crown_decision"], "window_candidate")
            v2 = {"challenger_wins": False, "margin": -0.01, "se": 0.0008, "z": -12,
                  "rejection_reason": None}
            stub._stamp_window_verdict(v2, {"decision_block": 9_052_470})
            self.assertEqual(v2["crown_decision"], "not_candidate:margin_not_positive")

    async def test_close_crowns_best_confirmed_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            stub = _stub(td, {"chal-2": True, "chal-3": False}, 9_052_470)
            stub._window_due()
            _file(stub, "chal-1", "A", 0.0012)
            _file(stub, "chal-2", "B", 0.0030, n_slices=2)     # best, near-miss pooled
            _file(stub, "chal-3", "A", 0.0025)                 # A's best; chal-1 dropped
            _file(stub, "chal-4", "C", -0.002)
            _file(stub, "chal-5", "D", 0.004, reason="causality_fail")
            with mock.patch("affine.validator.model_store.fetch_repo_info",
                            return_value=SimpleNamespace(total_safetensors_bytes=1)):
                stub.subtensor.block = 2515 * 3600 + 5
                await stub._close_window(stub.subtensor.block)
            king = stub.state.king
            self.assertEqual(king.reign_number, 12)
            self.assertEqual(king.challenge_id, "chal-2")
            self.assertEqual(king.repo, "r2://public/models/sha256/rev-chal-2/")
            self.assertEqual(king.crown_block, 2515 * 3600 + 5)
            self.assertTrue(stub.weights_set)
            # Only the best candidate was confirmed (it passed).
            self.assertEqual([c["confirm"]["challenge_id"] for c in stub.eval_client.calls],
                             ["chal-2"])
            self.assertEqual(stub.eval_client.calls[0]["confirm"]["slice_index"], 2)
            self.assertEqual(stub.eval_client.calls[0]["king_revision"], "king")
            rows = [json.loads(l) for l in open(stub.state.history_path)]
            close = [r for r in rows if r["event"] == "window_close"][-1]
            self.assertEqual(close["outcome"], "crowned")
            self.assertEqual(close["winner"]["challenge_id"], "chal-2")
            self.assertEqual([c["challenge_id"] for c in close["candidates"]],
                             ["chal-2", "chal-3"])
            self.assertEqual({d["challenge_id"]: d["reason"] for d in close["dropped"]},
                             {"chal-4": "margin_not_positive",
                              "chal-5": "gate:causality_fail", "chal-1": "hotkey_duplicate"})
            self.assertEqual(len(close["verdicts_considered"]), 5)
            self.assertTrue(close["confirmations"][0]["passed"])
            crowned = [r for r in rows if r["event"] == "crowned"][-1]
            self.assertEqual(crowned["challenge_id"], "chal-2")
            self.assertEqual(crowned["verdict"]["crown_mode"], "window_best")
            self.assertEqual(crowned["verdict"]["window_id"], 2514)
            self.assertEqual(crowned["verdict"]["private_repo"],
                             "r2://private/models/registrations/chal-2/")
            # Next window is open for the new king.
            self.assertEqual(stub.state.crown_window["window_id"], 2515)
            self.assertEqual(stub.state.crown_window["king_reign"], 12)
            self.assertEqual(stub.state.crown_window["verdicts"], [])

    async def test_close_falls_through_to_next_candidate_then_king_stays(self):
        with tempfile.TemporaryDirectory() as td:
            stub = _stub(td, {"chal-2": False, "chal-3": False, "chal-1": True}, 9_052_470)
            stub._window_due()
            _file(stub, "chal-1", "A", 0.0012)
            _file(stub, "chal-2", "B", 0.0030)
            _file(stub, "chal-3", "C", 0.0025)
            with mock.patch("affine.validator.model_store.fetch_repo_info",
                            return_value=SimpleNamespace(total_safetensors_bytes=1)):
                await stub._close_window(2515 * 3600 + 5)
            self.assertEqual(stub.state.king.reign_number, 11)
            # crown_confirm_max = 2: chal-1 (third) is never tried.
            self.assertEqual([c["confirm"]["challenge_id"] for c in stub.eval_client.calls],
                             ["chal-2", "chal-3"])
            rows = [json.loads(l) for l in open(stub.state.history_path)]
            close = rows[-1]
            self.assertEqual(close["event"], "window_close")
            self.assertEqual(close["outcome"], "king_stays_none_confirmed")
            self.assertIsNone(close["winner"])
            self.assertEqual(len(close["confirmations"]), 2)
            self.assertEqual(stub.state.crown_window["window_id"], 2515)

    async def test_close_without_candidates_needs_no_pod(self):
        with tempfile.TemporaryDirectory() as td:
            stub = _stub(td, {}, 9_052_470)
            stub._window_due()
            _file(stub, "chal-4", "C", -0.002)
            await stub._close_window(2515 * 3600)
            self.assertEqual(stub.eval_client.calls, [])
            rows = [json.loads(l) for l in open(stub.state.history_path)]
            self.assertEqual(rows[-1]["outcome"], "king_stays_no_candidates")
            self.assertEqual(stub.state.king.reign_number, 11)

    async def test_no_confirmation_mode_crowns_best_directly(self):
        with tempfile.TemporaryDirectory() as td:
            stub = _stub(td, {}, 9_052_470, crown_confirm_slice=False)
            stub._window_due()
            _file(stub, "chal-2", "B", 0.0030)
            await stub._close_window(2515 * 3600)
            self.assertEqual(stub.state.king.challenge_id, "chal-2")
            self.assertEqual(stub.eval_client.calls, [])

    async def test_infra_failure_retries_then_king_stays(self):
        with tempfile.TemporaryDirectory() as td:
            stub = _stub(td, {}, 9_052_470)
            stub._window_due()
            _file(stub, "chal-2", "B", 0.0030)

            async def boom(**kw):
                raise RuntimeError("pod down")
            stub.eval_client.run_duel = boom
            with mock.patch("affine.validator.model_store.fetch_repo_info",
                            return_value=SimpleNamespace(total_safetensors_bytes=1)):
                for _ in range(Validator.WINDOW_CLOSE_MAX_ATTEMPTS - 1):
                    await stub._close_window_safely(2515 * 3600)
                    self.assertEqual(stub.state.crown_window["window_id"], 2514)
                await stub._close_window_safely(2515 * 3600)
            self.assertEqual(stub.state.crown_window["window_id"], 2515)
            rows = [json.loads(l) for l in open(stub.state.history_path)]
            self.assertEqual(rows[-1]["outcome"], "king_stays_confirmation_unavailable")
            self.assertEqual(stub.state.king.reign_number, 11)


if __name__ == "__main__":
    unittest.main()
