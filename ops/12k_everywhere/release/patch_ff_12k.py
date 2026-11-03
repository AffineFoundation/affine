#!/usr/bin/env python3
"""RA/20261102-12k-everywhere: lifecycle / site files for the 12,288-token output cap (Jacob 2026-11-02 18:48 CR "12 k should be everywhere").
  ff_format.py   MAX_OUTPUT 8192 -> 12288 (incl. the appended EOS), MAX_CONTEXT 12288 -> 16384 (= the gate), LEGACY_CAP 8192: an output of
                 exactly 8192 tokens still counts as length-capped (8K-generated files queued before the switch stay valid and keep their
                 meaning; nothing that was valid becomes invalid).
  affine_batch.py  docstring of the local check.
  FF-PUBLIC.json   batch_file.rules EOS text.
usage: patch_ff_12k.py BASE_DIR OUT_DIR  (BASE_DIR has ff_format.py affine_batch.py FF-PUBLIC.json)"""
import json, sys
from pathlib import Path
base, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
def rep(t, old, new, name):
    assert t.count(old) == 1, (name, old[:70], t.count(old)); return t.replace(old, new)
t = (base / "ff_format.py").read_text()
t = rep(t, "    a length-capped rollout (stop \"max_tokens\", no EOS); longer is invalid. prompt + output <= MAX_CONTEXT tokens.\n",
        "    a length-capped rollout (stop \"max_tokens\", no EOS); longer is invalid. prompt + output <= MAX_CONTEXT tokens.\n"
        "  * 12K everywhere (RA/20261102-12k-everywhere, Jacob 2026-11-02 18:48 CR): MAX_OUTPUT 12288 (was 8192), MAX_CONTEXT 16384 (was 12288).\n"
        "    Transition: an output of exactly LEGACY_CAP (8192) tokens is still length-capped (an 8K-generated file queued before the switch);\n"
        "    every file valid before stays valid with the same EOS/capped meaning.\n", "ff_format doc")
t = rep(t, "MAX_OUTPUT = 8192          # output tokens incl. the appended EOS\nMAX_CONTEXT = 12288        # prompt + output tokens\n",
        "MAX_OUTPUT = 12288         # output tokens incl. the appended EOS (12K everywhere, 2026-11-02; was 8192)\n"
        "MAX_CONTEXT = 16384        # prompt + output tokens (= the gate's 16,384 context; was 12288)\n"
        "LEGACY_CAP = 8192          # exactly 8192 output tokens = length-capped too (8K-generated files queued before the switch)\n", "ff_format consts")
t = rep(t, "            stop = 'max_tokens' if len(ot) == MAX_OUTPUT else 'eos'\n",
        "            stop = 'max_tokens' if len(ot) in (MAX_OUTPUT, LEGACY_CAP) else 'eos'\n", "ff_format stop")
(out / "ff_format.py").write_text(t)
a = (base / "affine_batch.py").read_text()
a = rep(a, "(the validator also tokenizes: <= 8192 output tokens, <= 12288 prompt+output, no chat",
        "(the validator also tokenizes: <= 12288 output tokens, <= 16384 prompt+output, no chat", "affine_batch doc")
(out / "affine_batch.py").write_text(a)
d = json.loads((base / "FF-PUBLIC.json").read_text())
r = d["batch_file"]["rules"]
old = ("EOS: an output of up to 8191 tokens gets <|im_end|> appended (it ended). Exactly 8192 tokens = cut at the length cap (no EOS, treated as "
       "unresolved/capped). More than 8192 tokens is invalid. Prompt + output must be at most 12288 tokens.")
assert r[5] == old, r[5]
r[5] = ("EOS: an output of up to 12287 tokens gets <|im_end|> appended (it ended). Exactly 12288 tokens = cut at the length cap (no EOS, treated as "
        "unresolved/capped). More than 12288 tokens is invalid. Prompt + output must be at most 16384 tokens. Generate with max_tokens 12288 "
        "(12K everywhere since 2026-11-02: generation, batch files, training and the gate). Transition: an output of exactly 8192 tokens "
        "(an 8K-generated file) also counts as capped.")
(out / "FF-PUBLIC.json").write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
print("ok", out)
