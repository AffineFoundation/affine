"""King payout window (operator directive 2026-09-14 11:09 UTC): the share
computation for 0 / 1 / 2 / 3 paid crowns, the 72 h expiry boundary, revoked
and genesis rows excluded, inaccessible crowns forfeiting, one share per
crown for a hotkey with several crowns, and the uid mapping incl. the burn
fallback. Also the State-level view and the revert-keeps-crowned_at rule.

    source .venv/bin/activate
    python -m unittest tests.test_king_payout -v                    # from affine/
    python -m unittest discover -s affine/tests -t affine           # from the repo root
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from affine import payout
from affine.config import load_config
from affine.state import State

WINDOW_S = 72 * 3600.0
NOW = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _crown(reign: int, hotkey: str, age_h: float, **extra) -> dict:
    """A lineage row crowned `age_h` hours before NOW."""
    row = {"reign_number": reign, "hotkey": hotkey, "repo": f"r2://m/{reign}/",
           "revision": f"{reign:064x}", "block": 1000 + reign,
           "crowned_at": _iso(NOW - timedelta(hours=age_h)),
           "current": False}
    row.update(extra)
    return row


def _paid(rows: list[dict]) -> list[tuple[int, float]]:
    ann = payout.annotate_lineage(rows, window_s=WINDOW_S, now=NOW)
    return [(m["reign_number"], m["share"]) for m in payout.paid_crowns(ann)]


class ShareTests(unittest.TestCase):
    def test_zero_kings_is_burn(self):
        ann = payout.annotate_lineage([], window_s=WINDOW_S, now=NOW)
        self.assertEqual(payout.paid_crowns(ann), [])
        self.assertEqual(payout.shares_by_hotkey(ann), {})
        self.assertEqual(payout.uid_weights({}, {}, burn_uid=0), ([0], [1.0]))
        self.assertTrue(payout.describe(ann).startswith("burn"))

    def test_all_expired_is_burn(self):
        rows = [_crown(12, "A", 100, current=True), _crown(11, "B", 140)]
        self.assertEqual(_paid(rows), [])
        ann = payout.annotate_lineage(rows, window_s=WINDOW_S, now=NOW)
        self.assertTrue(all(m["expired"] for m in ann))
        self.assertEqual([m["weight_bps"] for m in ann], [0, 0])
        self.assertEqual(payout.uid_weights(payout.shares_by_hotkey(ann),
                                            {"A": 1, "B": 2}, burn_uid=0),
                         ([0], [1.0]))

    def test_one_king_takes_everything(self):
        rows = [_crown(12, "A", 40, current=True), _crown(11, "B", 100)]
        self.assertEqual(_paid(rows), [(12, 1.0)])
        ann = payout.annotate_lineage(rows, window_s=WINDOW_S, now=NOW)
        self.assertEqual([m["weight_bps"] for m in ann], [10000, 0])
        self.assertEqual(payout.uid_weights(payout.shares_by_hotkey(ann),
                                            {"A": 156, "B": 83}, burn_uid=0),
                         ([156], [1.0]))

    def test_two_kings_split_fifty_fifty(self):
        rows = [_crown(13, "A", 5, current=True), _crown(12, "B", 40),
                _crown(11, "C", 100)]
        self.assertEqual(_paid(rows), [(13, 0.5), (12, 0.5)])
        uids, w = payout.uid_weights(
            payout.shares_by_hotkey(
                payout.annotate_lineage(rows, window_s=WINDOW_S, now=NOW)),
            {"A": 1, "B": 2, "C": 3}, burn_uid=0)
        self.assertEqual(uids, [1, 2])
        self.assertEqual(w, [0.5, 0.5])

    def test_three_kings_split_in_thirds(self):
        rows = [_crown(14, "A", 1, current=True), _crown(13, "B", 30),
                _crown(12, "C", 71), _crown(11, "D", 73)]
        paid = _paid(rows)
        self.assertEqual([r for r, _ in paid], [14, 13, 12])
        for _, share in paid:
            self.assertAlmostEqual(share, 1 / 3)
        ann = payout.annotate_lineage(rows, window_s=WINDOW_S, now=NOW)
        self.assertEqual([m["weight_bps"] for m in ann], [3333, 3333, 3333, 0])
        self.assertAlmostEqual(sum(payout.shares_by_hotkey(ann).values()), 1.0)

    def test_expiry_boundary_is_exclusive(self):
        # One second inside the window: paid. Exactly 72 h: not paid.
        inside = [_crown(12, "A", 0, current=True,
                         crowned_at=_iso(NOW - timedelta(hours=72) + timedelta(seconds=1)))]
        self.assertEqual(_paid(inside), [(12, 1.0)])
        edge = [_crown(12, "A", 72, current=True)]
        self.assertEqual(_paid(edge), [])
        ann = payout.annotate_lineage(edge, window_s=WINDOW_S, now=NOW)
        self.assertEqual(ann[0]["paid_until"], _iso(NOW))
        self.assertTrue(ann[0]["expired"])

    def test_sitting_king_stops_earning_after_window_but_keeps_throne(self):
        # The current king is 80 h old: it stays `current` (throne) but earns 0.
        rows = [_crown(12, "A", 80, current=True), _crown(11, "B", 10)]
        ann = payout.annotate_lineage(rows, window_s=WINDOW_S, now=NOW)
        self.assertTrue(ann[0]["current"])
        self.assertFalse(ann[0]["earning"])
        self.assertEqual(_paid(rows), [(11, 1.0)])

    def test_revoked_and_genesis_rows_never_pay(self):
        rows = [_crown(13, "A", 1, current=True),
                _crown(12, "B", 2, revoked=True),
                _crown(0, "", 3)]  # genesis: empty hotkey
        self.assertEqual(_paid(rows), [(13, 1.0)])

    def test_missing_or_bad_crowned_at_is_not_paid(self):
        rows = [_crown(12, "A", 1, current=True, crowned_at=None),
                _crown(11, "B", 1, crowned_at="not a date"),
                _crown(10, "C", 1)]
        self.assertEqual(_paid(rows), [(10, 1.0)])

    def test_inaccessible_crown_forfeits_its_share(self):
        rows = [_crown(13, "A", 1, current=True), _crown(12, "B", 2)]
        ann = payout.annotate_lineage(rows, window_s=WINDOW_S, now=NOW,
                                      inaccessible={"B"})
        self.assertEqual([(m["reign_number"], m["share"])
                          for m in payout.paid_crowns(ann)], [(13, 1.0)])
        self.assertTrue(ann[1]["inaccessible"])
        self.assertFalse(ann[1]["expired"])

    def test_one_share_per_crown_sums_per_hotkey(self):
        # Hotkey A dethroned its own king: two crowns inside the window.
        rows = [_crown(14, "A", 1, current=True), _crown(13, "B", 20),
                _crown(12, "A", 40)]
        ann = payout.annotate_lineage(rows, window_s=WINDOW_S, now=NOW)
        self.assertEqual([m["share"] for m in ann], [1 / 3, 1 / 3, 1 / 3])
        shares = payout.shares_by_hotkey(ann)
        self.assertAlmostEqual(shares["A"], 2 / 3)
        self.assertAlmostEqual(shares["B"], 1 / 3)
        uids, w = payout.uid_weights(shares, {"A": 7, "B": 9}, burn_uid=0)
        self.assertEqual(uids, [7, 9])
        self.assertAlmostEqual(w[0], 2 / 3)
        self.assertAlmostEqual(w[1], 1 / 3)

    def test_unregistered_hotkey_is_skipped_and_rest_renormalised(self):
        shares = {"A": 0.5, "B": 0.5}
        self.assertEqual(payout.uid_weights(shares, {"A": 4}, burn_uid=0),
                         ([4], [1.0]))
        self.assertEqual(payout.uid_weights(shares, {}, burn_uid=0),
                         ([0], [1.0]))

    def test_iso_parsing_accepts_z_and_naive_utc(self):
        self.assertEqual(payout.parse_iso("2026-09-14T12:00:00Z"), NOW)
        self.assertEqual(payout.parse_iso("2026-09-14T12:00:00"), NOW)
        self.assertIsNone(payout.parse_iso(""))
        self.assertIsNone(payout.parse_iso("nope"))

    def test_rule_text_and_contract_block(self):
        self.assertIn("72 hours", payout.rule_text(72))
        blk = payout.contract_block(72, "2026-09-14T13:00:00+00:00", 0)
        self.assertEqual(blk["window_hours"], 72)
        self.assertEqual(blk["burn_uid"], 0)
        self.assertIsNone(payout.contract_block(72, "", 0)["effective_at"])


class StateViewTests(unittest.TestCase):
    """The State-level view feeds both the weight sweep and the dashboard."""

    def _state(self, td: str) -> State:
        s = State(Path(td))
        s.set_king(hotkey="A", repo="r2://m/a/", revision="a" * 64,
                   block=1, challenge_id="chal-1")
        s.set_king(hotkey="B", repo="r2://m/b/", revision="b" * 64,
                   block=2, challenge_id="chal-2")
        s.set_king(hotkey="A", repo="r2://m/a2/", revision="c" * 64,
                   block=3, challenge_id="chal-3")
        return s

    def test_fresh_lineage_all_paid_one_row_per_reign(self):
        with tempfile.TemporaryDirectory() as td:
            s = self._state(td)
            rows = s.king_lineage_members(WINDOW_S)
            self.assertEqual([m["reign_number"] for m in rows], [2, 1, 0])
            self.assertEqual([m["hotkey"] for m in rows], ["A", "B", "A"])
            self.assertTrue(all(m["earning"] for m in rows))
            self.assertEqual(s.king_chain_hotkeys(WINDOW_S), ["A", "B"])
            shares = s.king_payout_shares(WINDOW_S)
            self.assertAlmostEqual(shares["A"], 2 / 3)
            self.assertAlmostEqual(shares["B"], 1 / 3)

    def test_old_crowns_expire_from_the_state_view(self):
        with tempfile.TemporaryDirectory() as td:
            s = self._state(td)
            later = datetime.now(timezone.utc) + timedelta(hours=73)
            self.assertEqual(s.king_chain_members(WINDOW_S, now=later), [])
            self.assertEqual(s.king_payout_shares(WINDOW_S, now=later), {})

    def test_inaccessible_sweep_result_applies(self):
        with tempfile.TemporaryDirectory() as td:
            s = self._state(td)
            s.inaccessible_hotkeys = {"B"}
            paid = s.king_chain_members(WINDOW_S)
            self.assertEqual([m["hotkey"] for m in paid], ["A", "A"])

    def test_revert_keeps_original_crowned_at_and_drops_dead_king(self):
        with tempfile.TemporaryDirectory() as td:
            s = self._state(td)
            original = s.king.previous[0]["crowned_at"]  # reign 1 (B)
            restored = s.revert_king("unservable")
            self.assertEqual(restored.hotkey, "B")
            self.assertEqual(restored.reign_number, 3)
            self.assertEqual(restored.crowned_at, original)
            rows = s.king_lineage_members(WINDOW_S)
            # The dead reign 2 (A's second crown) is gone from the lineage.
            self.assertEqual([m["reign_number"] for m in rows], [3, 0])
            # A crash restart replays the revert row and keeps the same time.
            s2 = State(Path(td))
            s2.king = None
            s2._reconcile_from_history()
            self.assertEqual(s2.king.crowned_at, original)


class ConfigTests(unittest.TestCase):
    def test_toml_knobs(self):
        cfg = load_config()
        self.assertEqual(cfg.king_payout_window_s, 72 * 3600.0)
        self.assertIsInstance(cfg.king_payout_rule_effective_at, str)


if __name__ == "__main__":
    unittest.main()
