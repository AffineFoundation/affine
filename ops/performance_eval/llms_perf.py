#!/usr/bin/env python3
"""Insert (or remove with --remove) the Performance section in an llms.txt, before '## 7. Privacy and transparency' (fallback '## Links').
Marked with <!-- perf-eval-20261031 --> ... <!-- /perf-eval-20261031 --> so it can be removed exactly."""
import sys, re
p = sys.argv[1]; s = open(p).read(); A, B = '<!-- perf-eval-20261031 -->', '<!-- /perf-eval-20261031 -->'
s = re.sub(re.escape(A) + r'.*?' + re.escape(B) + r'\n\n?', '', s, flags=re.S) if A in s else s
if '--remove' not in sys.argv:
    sec = (A + '\n## Performance over time (informational)\n\n'
           '- https://affine.io/performance.json: pass rate of every king (head) on two FIXED sets, scored on a separate machine:\n'
           '  the full public affine_math pool (all 7,496 tasks) and a private holdout (4,976 MATH test problems, minus any that\n'
           '  match a pool task; only its size and sha256 are published: 08d994bab2da775b807b23ebaf098c35db95d14a12af7ced2a46d6df4ab6c734).\n'
           '- Same prompt, sampling (temperature 1.0, top_p 0.95, top_k 20, 8192 output tokens) and grader as the gate probe; fixed\n'
           '  seeds; 1 sample per task; task-clustered SE. Shown at the top of https://affine.io/ (two lines, one per set).\n'
           '- Informational only: it never affects promotions, the gate, weights or training. The gate itself scores a few hundred\n'
           '  head-mixed tasks per round, so its pass rates are not comparable across rounds; these are.\n' + B + '\n\n')
    for anchor in ('## 7. Privacy and transparency', '## Links'):
        if anchor in s: s = s.replace(anchor, sec + anchor, 1); break
    else: s = s.rstrip('\n') + '\n\n' + sec
open(p, 'w').write(s); print('ok', p)
