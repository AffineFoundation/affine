"""Unit test for flip_scorer credit_mode="symmetric" (SN120, 2026-10-26).

Builds a synthetic probe ledger + DISPATCH tree in a temp dir and runs the real
flip_scorer.compute(). Checks:
  1. noise only (true pass rate unchanged) nets ~0 per flip event in expectation under
     "symmetric" (and is biased positive under the legacy "surprise_asymmetric" mode);
  2. a real improvement nets positive;
  3. harm nets negative, the negative net is kept in flip_score (not clipped) and carries
     across epochs, and the harmful miner gets share 0;
  4. shares are never all-zero (nonzero floor), even when every miner nets <= 0.
Run: python -m pytest -q test_flip_symmetric.py   (or: python test_flip_symmetric.py)
"""
from __future__ import annotations

import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flip_scorer as fs  # noqa: E402

NOISE, GOOD, BAD = "a" * 64, "b" * 64, "c" * 64


def build(root: Path, *, epochs: int, n_tasks: int, shift: dict, seed: int, owner=None):
    """Each task belongs to one miner (owner(t)) and is dispatched to it every epoch. A k=1 probe
    samples every task every epoch at p(e, t) = clip(base[t] + shift[m] * e): shift 0 = pure noise
    (true pass rate never changes), > 0 = the miner's batches keep improving the task, < 0 = harm.
    A float shift of +-1.0 or more forces pass/fail deterministically from epoch 1 on. Returns cfg."""
    rng = random.Random(seed)
    base = {t: rng.uniform(0.05, 0.45) for t in range(n_tasks)}
    miners = list(shift)
    owner = owner or (lambda t: miners[t % len(miners)])
    (root / "results").mkdir(parents=True)
    ledger = {}
    for e in range(epochs):
        per_task = {}
        for t in range(n_tasks):
            p = min(0.99, max(0.01, base[t] + shift[owner(t)] * e))
            if abs(shift[owner(t)]) >= 1.0:
                p = 1.0 if (e == 0) == (shift[owner(t)] < 0) else 0.0
            ok = rng.random() < p
            per_task[str(t)] = dict(c=1, n=1) if ok else dict(c=0, w=1, n=1)
        ck = "ck%03d" % e
        (root / "results" / (ck + ".private.json")).write_text(
            json.dumps(dict(envs=dict(affine_math=dict(per_task=per_task)))))
        ledger[ck] = dict(state="published", published=dict(epoch_index=e, checkpoint_id=ck, finished_at=e))
        sel = [dict(index=t, miner=owner(t)) for t in range(n_tasks)]
        d = root / "epochs" / str(e) / "coordinator"
        d.mkdir(parents=True)
        (d / "DISPATCH.json").write_text(json.dumps(dict(payload=dict(selected=sel))))
    (root / "ledger.json").write_text(json.dumps(ledger))
    return dict(
        env="affine_math",
        paths=dict(audit_out=str(root / "out"), epochs_root=str(root / "epochs"),
                   live_scores=str(root / "none-scores.json"), probe_ledger=str(root / "ledger.json"),
                   probe_results=str(root / "results"), verify_ledger=str(root / "none-verify.json"),
                   website=str(root / "site")),
        formula=dict(credit_mode="symmetric", symmetric_weight=1.0, ema_alpha=0.3, eps=1e-4, min_epoch=0,
                     eligibility="hard_fail", surprise="neglog", share_mode="relu_normalize",
                     credit_epoch_rule="legacy_pair_first_epoch",
                     sampling_verify_gate=dict(enabled=True, credit_requires_verify_pass=True,
                                               unknown_policy="pass_through",
                                               uid_ban_path=str(root / "none-ban.json"))),
        uid_map=dict(source="legacy"),
    )


def run(cfg, mode=None):
    if mode:
        cfg = json.loads(json.dumps(cfg))
        cfg["formula"]["credit_mode"] = mode
    doc, detail = fs.compute(cfg)
    by = {m["miner_hex"]: m for m in doc["miners"]}
    events = [ev for p in detail["pairs"] for ev in p["events"]]
    return doc, by, events


class SymmetricCredit(unittest.TestCase):
    def test_weights_equal_magnitude(self):
        for p in (0.01, 0.1, 0.38, 0.9):
            g, l = fs.flip_weights(p, "symmetric", "neglog", 1e-4, 1.0)
            self.assertEqual(g, l)
            self.assertEqual(g, 1.0)
        g, l = fs.flip_weights(0.1, "surprise_asymmetric", "neglog", 1e-4)
        self.assertAlmostEqual(g / l, math.log(0.1) / math.log(0.9), places=9)  # legacy 22:1
        with self.assertRaises(ValueError):
            fs.flip_weights(0.5, "bogus", "neglog", 1e-4)

    def test_noise_nets_zero_improvement_positive_harm_negative(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = build(Path(td), epochs=12, n_tasks=1500, seed=7,
                        shift={NOISE: 0.0, GOOD: +0.04, BAD: -0.04})
            doc, by, ev = run(cfg)
            # every event is +-1 in symmetric mode
            self.assertTrue(all(e["delta"] in (1.0, -1.0) for e in ev if e["kind"] != "fail_to_fail"))
            noise = [e["delta"] for e in ev if e["miner"] == NOISE]
            n = len(noise)
            mean = sum(noise) / n
            sd = math.sqrt(sum((x - mean) ** 2 for x in noise) / (n - 1))
            self.assertLess(abs(mean), 4 * sd / math.sqrt(n), "noise must net ~0 in expectation")
            self.assertGreater(by[GOOD]["flip_score"], 0)
            self.assertLess(by[BAD]["flip_score"], 0)  # negative net kept, not clipped
            self.assertEqual(by[BAD]["share_now"], 0.0)
            self.assertGreater(by[GOOD]["share_now"], by[NOISE]["share_now"])
            self.assertAlmostEqual(sum(m["share_now"] for m in doc["miners"]), 1.0, places=9)
            # legacy mode pays noise a positive expected credit (the bug being fixed)
            _, _, ev_old = run(cfg, "surprise_asymmetric")
            old = [e["delta"] for e in ev_old if e["miner"] == NOISE]
            mo = sum(old) / len(old)
            so = math.sqrt(sum((x - mo) ** 2 for x in old) / (len(old) - 1))
            self.assertGreater(mo, 3 * so / math.sqrt(len(old)))

    def test_negative_net_carries_forward(self):
        # BAD's tasks decay to ~0.01: an early debt that later noise gains cannot clip away.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = build(root, epochs=10, n_tasks=800, seed=11, shift={NOISE: 0.0, GOOD: 0.04, BAD: -0.06})
            doc, by, ev = run(cfg)
            per_epoch = by[BAD]["by_epoch"]
            running, seen_neg = 0.0, False
            for k in sorted(per_epoch, key=int):
                running += per_epoch[k]["credit"]
                seen_neg |= running < 0
            self.assertTrue(seen_neg)
            self.assertAlmostEqual(running, by[BAD]["flip_score"], places=9)  # net = sum of epochs, no per-epoch clip
            self.assertTrue(any(v["credit"] < 0 for v in per_epoch.values()))

    def test_never_all_zero(self):
        # every miner only harms (pass at e0, fail after) -> floor must still give a positive share vector
        with tempfile.TemporaryDirectory() as td:
            cfg = build(Path(td), epochs=4, n_tasks=200, seed=3, shift={BAD: -1.0, NOISE: -1.0})
            doc, by, _ = run(cfg)
            self.assertTrue(all(m["flip_score"] <= 0 for m in doc["miners"]))
            shares = [m["share_now"] for m in doc["miners"]]
            self.assertGreater(sum(shares), 0.999)
            self.assertTrue(any(s > 0 for s in shares))
            self.assertTrue(doc["formula"]["nonzero_weight_floor"]["floor_applied"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
