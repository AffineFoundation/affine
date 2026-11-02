import sys
p = sys.argv[1]; s = open(p).read()
MARK = "<!-- gain-mult-20261102 -->"
if MARK in s: print("already"); sys.exit(0)
old = """  are never all zero (there's a nonzero floor), and they're set hourly. Current payees are in https://affine.io/k1.json (`payees`).
"""
assert s.count(old) == 1
new = old + """- King gain multiplier """ + MARK + """ (from E581, 2026-11-02): a king crowned after E580 has its 1/N share multiplied
  by m, then the payees' shares are rescaled so their total stays the same. m measures how much the king actually improved
  on its parent across the whole gate pool. The prior sampler runs both heads on the same tasks at the same token cap, and
  gain = the king's mean pass rate minus the parent's on those shared tasks. cons = max(0, gain - 1 SE). Then
  m = clamp(cons / median, 0.5, 2) (lambda = 1). The median is taken over recent positive cons values (the last 10; it is about
  0.004 today). m stays 1 until the pair shares at least 2,000 tasks (`gain_settled` false). Kings crowned at or before E580
  keep m = 1 for good. Per-king m and settled: https://affine.io/mining.json `training.k1.king_gain_multiplier`, and
  k1.json `payees[].gain_m` / `gain_settled`. Per-task data stays private.
"""
s = s.replace(old, new)
s = s.replace("Last updated: 2026-11-02 14:51 CR (UTC-6).", "Last updated: 2026-11-02 16:50 CR (UTC-6).")
open(p, "w").write(s); print("patched")
