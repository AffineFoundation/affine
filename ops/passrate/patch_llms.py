#!/usr/bin/env python3
"""llms.txt: performance.json section -> passrate.json section (pt1130u). Idempotent; asserts each anchor once."""
import sys, re
p = sys.argv[1]; s = open(p).read()
if 'passrate-20261102' in s: print('already patched'); sys.exit(0)
a = s.index('<!-- perf-eval-20261031 -->'); b = s.index('<!-- /perf-eval-20261031 -->') + len('<!-- /perf-eval-20261031 -->')
new = '''<!-- passrate-20261102 -->
## Pass rate across all envs (informational)

- https://affine.io/passrate.json: pass rate of every king (head) over the WHOLE gate pool, all envs (gate-pool-v1, 19,305
  tasks: 7,492 affine_math + 11,813 dapo), measured from the prior. Shown at the top of https://affine.io/ (first tile).
- Prior: a separate pod runs the current head on every pool task, 4 rollouts per task per pass, pass after pass. A king's
  number = mean over all 19,305 tasks of correct/graded over its complete passes (a pass cut short by a head change is
  dropped, since its finished tasks skew easy). Every task is measured, so the SE is rollout noise only (~0.1 points).
- Until a king has a complete pass (about an hour after it is crowned), its number is the gate's head baseline for that
  round (~500 tasks drawn with unequal inclusion probabilities pi, 12 rollouts each) re-weighted to the whole pool by 1/pi
  (Hajek, per env, combined by env size), with a 95% CI (about +-3 points). Inclusion probabilities stay private.
- Same prompt, sampling (temperature 1.0, top_p 0.95, top_k 20) and grader as the gate; capped replies count as fails;
  8,192 output tokens before the 12K switch (E567), 12,288 after (each row says its max_tokens).
- Informational only: it never affects promotions, the gate, weights or training.
<!-- /passrate-20261102 -->'''
s = s[:a] + new + s[b:]
old = 'on the full pool (performance.json), pass went from'
assert s.count(old) == 1; s = s.replace(old, 'on the full affine_math pool (the performance eval), pass went from')
old = '- Performance: https://affine.io/performance.json   This file: https://affine.io/llms.txt'
assert s.count(old) == 1; s = s.replace(old, '- Pass rate across envs: https://affine.io/passrate.json   This file: https://affine.io/llms.txt')
assert 'performance.json' not in s, [m.start() for m in re.finditer('performance.json', s)]
open(p, 'w').write(s); print('patched')
