"""RA/20261102-12k-everywhere: mining.json top-level max_output_tokens follows FF-PUBLIC.json max_output_tokens (12288) for free files;
FF-PUBLIC notice gets the 12K sentence. usage: patch_observer.py OBSERVER.py FF-PUBLIC.json"""
import json, sys
from pathlib import Path
o, f = Path(sys.argv[1]), Path(sys.argv[2])
s = o.read_text()
a = "        result['notice'] = ff['notice_after_cutoff']\n"
n = a + ("        if type(ff.get('max_output_tokens')) is int:   # RA/20261102-12k-everywhere: free-file cap (signed manifest still lists 8192)\n"
         "            result['max_output_tokens'] = ff['max_output_tokens']\n")
if 'RA/20261102-12k-everywhere' not in s:
    assert s.count(a) == 1; o.write_text(s.replace(a, n))
d = json.loads(f.read_text())
d['max_output_tokens'] = 12288
t = (' New (2026-11-02, from epoch 590): 12,288 output tokens everywhere (generation, batch files, training, gate); '
     'generate with max_tokens 12,288. Files generated at 8K still count.')
if t not in d['notice_after_cutoff']: d['notice_after_cutoff'] += t
f.write_text(json.dumps(d, indent=1, ensure_ascii=False) + '\n')
print('ok')
