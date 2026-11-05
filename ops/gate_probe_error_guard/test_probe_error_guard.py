#!/usr/bin/env python3
import json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import probe_error_guard as G

def cells(n, c=0, x=0):
    return {str(i): dict(n=1, c=int(i < c), w=0, u=0, b=0, x=int(i < x)) for i in range(n)}

class TestGuard(unittest.TestCase):
    def test_clean_under_threshold(self):
        # 1 error in 6376 ~ 0.00016
        pt = {str(i): dict(n=1, c=1, w=0, u=0, b=0, x=0) for i in range(100)}
        pt['0'] = dict(n=1, c=0, w=0, u=0, b=0, x=1)
        s = G.cell_error_rate(pt)
        self.assertAlmostEqual(s['rate'], 0.01, places=4)  # 1/100 = 0.01 boundary
        # strictly greater than threshold blocks; equal is ok
        self.assertFalse(G.is_over_threshold(pt))  # 0.01 is NOT > 0.01

    def test_clean_gate_candidate(self):
        pt = {str(i): dict(n=8, c=8, w=0, u=0, b=0, x=0) for i in range(510)}
        self.assertFalse(G.is_over_threshold(pt))
        ok, s = G.should_judge(pt, side='candidate', epoch_dir=Path(tempfile.mkdtemp()), attempt=0)
        self.assertTrue(ok); self.assertEqual(s['x'], 0)

    def test_catastrophic_e004_pattern(self):
        pt = {str(i): dict(n=1, c=0, w=0, u=0, b=0, x=1) for i in range(7496)}
        pt['0'] = dict(n=1, c=0, w=0, u=0, b=0, x=0)  # 7495/7496
        self.assertTrue(G.is_over_threshold(pt))
        s = G.cell_error_rate(pt)
        self.assertGreater(s['rate'], 0.99)

    def test_errors_not_counted_as_wrong(self):
        # schema: x is separate from w; rate uses x/n only
        pt = {'1': dict(n=10, c=5, w=3, u=0, b=0, x=2)}
        s = G.cell_error_rate(pt)
        self.assertEqual(s['x'], 2); self.assertEqual(s['n'], 10)
        self.assertAlmostEqual(s['rate'], 0.2)

    def test_hold_and_alert_after_max(self):
        d = Path(tempfile.mkdtemp())
        pt = {str(i): dict(n=1, c=0, w=0, u=0, b=0, x=1) for i in range(100)}
        ok, s = G.should_judge(pt, side='head', epoch_dir=d, attempt=G.REPROBE_MAX)
        self.assertFalse(ok)
        held = json.loads((d / 'HELD.json').read_text())
        self.assertTrue(held['active'])
        self.assertEqual(held['holds'][-1]['side'], 'head')

    def test_threshold_constant(self):
        self.assertEqual(G.THRESHOLD, 0.01)
        self.assertEqual(G.REPROBE_MAX, 2)

if __name__ == '__main__':
    unittest.main()
