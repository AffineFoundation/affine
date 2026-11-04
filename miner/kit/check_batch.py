#!/usr/bin/env python3
"""The validator's FULL batch check, locally: validator/batch/ff_format.check with the head tokenizer (verbatim live code).

  python3 check_batch.py --hotkey 5F... --tokenizer ./head batch.json

../affine_batch.py check does the structural part without a tokenizer; this one also tokenizes exactly like the trainer:
control tokens 248044-248046 forbidden, output <= 12288 tokens incl. EOS (exactly 12288 = capped, no EOS; any shorter output gets the EOS), prompt+output <= 16384 (caps: https://affine.io/caps.json),
then the caps (first 125 prompts / 1000 rollouts, whole prompts dropped). Prints the same summary the lifecycle logs.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kitpath  # noqa: E402,F401
import ff_format  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('file'); ap.add_argument('--hotkey', required=True); ap.add_argument('--tokenizer', required=True)
a = ap.parse_args()
ok, summary, _ = ff_format.check(Path(a.file).read_bytes(), a.hotkey, ff_format.tokenizer(a.tokenizer), 125, 1000, v2=True)
print(json.dumps(dict(ok=ok, summary=summary) if ok else dict(ok=False, reason=summary)))
sys.exit(0 if ok else 1)
