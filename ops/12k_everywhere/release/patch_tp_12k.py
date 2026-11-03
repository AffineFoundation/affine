#!/usr/bin/env python3
"""RA/20261102-12k-everywhere: training_params.py (mining.json) -> 12,288 output tokens for batch files and training.
usage: patch_tp_12k.py SRC OUT"""
import sys
from pathlib import Path
t = Path(sys.argv[1]).read_text()
def rep(t, old, new):
    assert t.count(old) == 1, (old[:80], t.count(old)); return t.replace(old, new)
t = rep(t, "rollouts_per_prompt='any number >= 1', max_output_tokens=8192, max_context_tokens=12288,",
        "rollouts_per_prompt='any number >= 1', max_output_tokens=12288, max_context_tokens=16384, legacy_capped_output_tokens=8192,")
t = rep(t, '"(old-format batches until the cutoff: 1[correct] instead of the miner reward). Rollouts at exactly 8192 output tokens are "\n'
           '            "length-capped (no EOS) and treated as unresolved for the derail mask.")\n',
        '"(old-format batches until the cutoff: 1[correct] instead of the miner reward). Rollouts at exactly 12288 output tokens (or exactly "\n'
        '            "8192: 8K-generated files queued before the 12K switch) are length-capped (no EOS) and treated as unresolved for the derail mask.")\n'
        "    if lo.get('overlong_max_tokens') == 8192:   # RA/20261102-12k-everywhere: trainer v8.3.3m free-file window (12K everywhere)\n"
        "        lo.update(overlong_soft_start=10096, overlong_max_tokens=12288,\n"
        "                  overlong_note='12K everywhere (2026-11-02, trainer v8.3.3m): free-file rollouts use the window 10096..12288 (the same 2,192-token '\n"
        "                                'ramp, now ending at the 12,288 cap); the signed job settings still list 6000/8192.')\n")
t = rep(t, "                                 enable_thinking=r['harness'].get('enable_thinking')) for r in m['environments']}\n",
        "                                 enable_thinking=r['harness'].get('enable_thinking')) for r in m['environments']}\n"
        "    for _h in harness.values():   # RA/20261102-12k-everywhere: free files and training use 12,288 / 16,384 (signed manifest still 8192 / 12288)\n"
        "        if _h.get('max_output_tokens') == 8192:\n"
        "            _h.update(max_output_tokens=12288, max_context_tokens=16384, signed_manifest_max_output_tokens=8192, signed_manifest_max_context_tokens=12288)\n")
Path(sys.argv[2]).write_text(t)
print("ok")
