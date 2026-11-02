#!/usr/bin/env python3
"""usage: patch_texts_irt.py apply|revert LLMS_TXT "<switch time CR>" - llms.txt section 6/8 text for the irt-info draw. Exact, reversible
string replacements (apply asserts every old string is present once; revert swaps back whatever is present)."""
import sys, os
mode, path = sys.argv[1], sys.argv[2]; T = sys.argv[3] if len(sys.argv) > 3 else ''
OLD_DRAW = ("- The draw: inclusion probability pi_i proportional to sqrt(E[p_i(1-p_i)]) under Beta(s_i+1, n_i-s_i+1) (tasks the head\n"
            "  sometimes solves get the most weight), mixed with a 10 % uniform floor (every task can be drawn), capped at 0.10, scaled\n")
NEW_DRAW = ("- The draw (IRT information-weighted, live from " + T + "): inclusion probability pi_i proportional to sqrt(I_i),\n"
            "  I_i = a_i'^2 P_i (1 - P_i) = the Fisher information of task i at the head's ability theta_h under a 2PL item response\n"
            "  model P_i(theta) = 1 / (1 + exp(-a_i (theta - b_i))). a_i (discrimination) and b_i (difficulty) are fitted (MAP) on every\n"
            "  recorded checkpoint x gate-pool task pass/fail (first fit: 365 checkpoints, 3.06M rollouts; refit daily); theta_h is the\n"
            "  MAP of the head's own gate-pool rollouts (its winning candidate probe + its head runs), recomputed when each round opens.\n"
            "  a_i' = a_i shrunk halfway (log scale) toward the pool median and clipped to [0.5, 2] x median, so a weakly determined a_i\n"
            "  can't give a task extreme weight. Tasks the head almost always (or never) solves get little weight; tasks near the head's\n"
            "  ability that separate stronger from weaker checkpoints get the most. a_i, b_i, theta_h and pi_i are private (like the prior\n"
            "  table); the method and code are public (priors/baseline_v1.py irt_score / theta_map, priors/irt/). Before the switch\n"
            "  (prior-v1, E497 on): pi_i proportional to sqrt(E[p_i(1-p_i)]) under Beta(s_i+1, n_i-s_i+1) from the prior table.\n"
            "  Both: mixed with a 10 % uniform floor (every task can be drawn), capped at 0.10, scaled\n")
PAIRS = [
    ("## 6. The promotion gate (held-out, prior-v1)\n", "## 6. The promotion gate (held-out, prior-v1, IRT information-weighted draw)\n"),
    (OLD_DRAW, NEW_DRAW),
    ("- Borderline difficulty: the draw favours tasks the head sometimes solves (pi ~ sqrt(p(1-p)) from the prior), and every\n",
     "- Borderline difficulty: the draw favours tasks near the head's ability that discriminate (pi ~ sqrt of the task's IRT\n  information at the head), and every\n"),
]
s = open(path).read()
if mode == 'apply':
    for a, b in PAIRS:
        if b in s: continue
        assert s.count(a) == 1, 'anchor not found once: %r' % a[:70]
        s = s.replace(a, b)
else:
    for a, b in PAIRS:
        if b.split('live from ')[0] in s and a not in s:
            # NEW_DRAW carries the switch time: locate it by prefix
            i = s.find(b.split('live from ')[0]) if 'live from' in b else s.find(b)
            j = i + len(b) if 'live from' not in b else s.find("  Both: mixed with a 10 % uniform floor (every task can be drawn), capped at 0.10, scaled\n", i) + len("  Both: mixed with a 10 % uniform floor (every task can be drawn), capped at 0.10, scaled\n")
            s = s[:i] + a + s[j:]
t = path + '.tmp.irt'; open(t, 'w').write(s); os.chmod(t, os.stat(path).st_mode & 0o777); os.replace(t, path)
print('llms.txt %s ok' % mode)
