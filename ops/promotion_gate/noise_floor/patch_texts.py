#!/usr/bin/env python3
"""Gate noise floor (Jacob pt900u, 2026-10-31): public SE wording. usage: patch_texts.py apply|revert N [file ...]
Anchored, idempotent text patches (reverse patch on revert, so concurrent edits by other jobs survive).
Files: training_params.py (mining.json / training.json 'heldout.se'), gate_ledger.py (gate.json rule), llms.txt (section 6)."""
import os, sys
mode, N = sys.argv[1], int(sys.argv[2])
CODE = "/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/code"
FILES = sys.argv[3:] or [CODE + "/training_params.py", CODE + "/gate_ledger.py", "/home/const/subnet120/affine/website/llms.txt"]
TP_DEF = '''def heldout_se(h):
    # gate noise floor (Jacob pt900u, 2026-10-31): how the held-out SE is built, read from PROMOTION-GATE.json "heldout"
    return dict(formula='weighted_z = gain / SE ; SE = sqrt(se_task^2 + sigma_run^2 * (1 + 1/3))',
                se_task='empirical per-task standard error of the weighted gain: sqrt(sum w^2 g^2) / sum w',
                sigma_run='run-to-run noise of the head, measured each round from its 3 replicate head runs on the kept tasks '
                          '(between-run variance of the weighted mean minus within-run sampling variance); used as max(sigma_run_floor, measured)',
                sigma_run_floor=h.get('sigma_run', 0.022), sigma_run_history=h.get('sigma_run_history'))


def heldout_public():'''
TP_RET_OLD = "    return dict(from_epoch=h.get('from_epoch'), bar_z=h.get('bar_z', 1.0), z_inclusive=h.get('z_inclusive', False),\n"
TP_RET_NEW = "    return dict(from_epoch=h.get('from_epoch'), bar_z=h.get('bar_z', 1.0), z_inclusive=h.get('z_inclusive', False), se=heldout_se(h),\n"
TP_RULE_OLD = "z = gain/SE > bar_z and cap-hit < '"
TP_RULE_NEW = "z = gain/SE > bar_z (SE: heldout.se) and cap-hit < '"
GL_OLD = "weighted_z = gain / its standard error. From E296 (free-file '"
GL_NEW = ("weighted_z = gain / SE, SE = sqrt(se_task^2 + sigma_run^2 (1 + 1/3)): se_task is the per-task standard error of the gain and '\n"
          "                'sigma_run the head\\'s run-to-run noise measured on the round\\'s 3 replicate head runs (E271-E%d: sigma_run floored '\n"
          "                'at 0.022 = 2.2 points; from E%d no fixed floor, the measured value is used). From E296 (free-file '") % (N - 1, N)
LL_OLD = "  weighted_z = gain / its standard error.\n"
LL_NEW = ("  weighted_z = gain / SE, SE = sqrt(se_task^2 + sigma_run^2 * (1 + 1/3)): se_task = per-task standard error of\n"
          "  the gain, sigma_run = the head's run-to-run noise measured on the round's 3 replicate head runs. From E%d there\n"
          "  is no fixed floor on sigma_run (E271-E%d: floor 0.022), so the bar is about +1.5 to +2 points of weighted gain.\n") % (N, N - 1)
PAIRS = {"training_params.py": [("def heldout_public():", TP_DEF), (TP_RET_OLD, TP_RET_NEW), (TP_RULE_OLD, TP_RULE_NEW)],
         "gate_ledger.py": [(GL_OLD, GL_NEW)], "llms.txt": [(LL_OLD, LL_NEW)]}
rc = 0
for p in FILES:
    s = open(p).read(); o = s
    for old, new in next(v for k, v in PAIRS.items() if os.path.basename(p).endswith(k)):
        a, b = (old, new) if mode == "apply" else (new, old)
        if b in s and (a not in s or a in b):
            continue            # already in target state
        if s.count(a) != 1:
            print("ANCHOR %s: %r found %d times" % (p, a[:60], s.count(a))); rc = 2; continue
        s = s.replace(a, b)
    if s != o:
        t = p + ".tmp-nf"; open(t, "w").write(s); os.chmod(t, os.stat(p).st_mode & 0o777)
        if p.endswith(".py"):
            import py_compile; py_compile.compile(t, doraise=True)
        os.replace(t, p); print("%s %s" % (mode, p))
    else:
        print("unchanged %s" % p)
sys.exit(rc)
