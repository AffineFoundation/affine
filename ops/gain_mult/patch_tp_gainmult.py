#!/usr/bin/env python3
"""Patch training_params.py: mining.json training.k1.king_gain_multiplier = RA/20261102-gain-mult/state/GAIN-MULT.public.json
(per-king m rounded + settled only; null if the file is missing). Idempotent. usage: patch_tp_gainmult.py FILE"""
import sys
P = sys.argv[1]
s = open(P).read()
MARK = '# gain-mult 20261102'
if MARK in s:
    print('already patched'); sys.exit(0)
old = "                  transparency=TRANSPARENCY)"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, "                  transparency=TRANSPARENCY,\n"
                   "                  king_gain_multiplier=_gain_mult_public())   " + MARK)
old2 = "def lifecycle_config():\n"
assert s.count(old2) == 1
s = s.replace(old2, "def _gain_mult_public():   " + MARK + "\n"
                    "    try: return json.loads((RA / '20261102-gain-mult/state/GAIN-MULT.public.json').read_bytes())\n"
                    "    except Exception: return None\n\n\n" + old2)
open(P, 'w').write(s)
print('patched')
