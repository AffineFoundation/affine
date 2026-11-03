#!/usr/bin/env python3
"""SN120 token caps loader: the one canonical source is caps.json next to this file.

Only max_output_tokens (and the two small constants next to it) are stored; everything else is derived here, so a cap
change is ONE edit of caps.json plus `tools/caps/caps_apply.py` (renders every consumer, rolls out at an epoch boundary).

    import caps; C = caps.load()
    C['max_output_tokens']    12288   generation max_tokens / batch-file output cap / gate task cap
    C['max_context_tokens']   16384   prompt + output (= max_output_tokens + prompt_headroom_tokens)
    C['max_model_len']        16384   vLLM --max-model-len / MAX_MODEL_LEN (= max_context_tokens)
    C['overlong_soft_start']  10096   overlong penalty ramp start (= max_output_tokens - overlong_ramp_tokens)
    C['overlong_max_tokens']  12288   overlong penalty ramp end (= max_output_tokens)
    C['last_eos_output_tokens'] 12287 the longest output that still gets an EOS appended
    C['legacy_capped_output_tokens'] 8192  an output of exactly this many tokens also counts as capped (queued old-cap files)

CLI:  python3 caps.py                 -> all values (JSON)
      python3 caps.py max_model_len   -> 16384   (for shell scripts)
Stdlib only, Python >= 3.6. Path override: SN120_CAPS=/path/to/caps.json.
"""
import json, os, sys
from pathlib import Path

PATH = Path(os.environ.get("SN120_CAPS") or Path(__file__).resolve().parent / "caps.json")
DERIVED = ("max_context_tokens", "max_model_len", "overlong_soft_start", "overlong_max_tokens", "overlong_window", "last_eos_output_tokens")


def derive(raw):
    """raw caps.json dict -> full caps dict (stored + derived values). Raises ValueError on an inconsistent file."""
    o, h, ramp = int(raw["max_output_tokens"]), int(raw["prompt_headroom_tokens"]), int(raw["overlong_ramp_tokens"])
    legacy = int(raw["legacy_capped_output_tokens"])
    if not (o > ramp > 0 and h > 0 and 0 < legacy and float(raw["overlong_penalty"]) >= 0):
        raise ValueError("caps.json: inconsistent caps %r" % dict(o=o, h=h, ramp=ramp, legacy=legacy))
    c = dict(raw)
    c.update(max_output_tokens=o, prompt_headroom_tokens=h, overlong_ramp_tokens=ramp, legacy_capped_output_tokens=legacy,
             overlong_penalty=float(raw["overlong_penalty"]), max_context_tokens=o + h, max_model_len=o + h,
             overlong_soft_start=o - ramp, overlong_max_tokens=o, overlong_window=[o - ramp, o], last_eos_output_tokens=o - 1)
    return c


def load(path=None):
    return derive(json.loads(Path(path or PATH).read_text()))


def at_epoch(epoch, c=None):
    """The caps in force for an epoch (from caps.json history; the current caps for epochs after the last entry)."""
    c = c or load(); cur = None
    for h in sorted(c.get("history") or [], key=lambda h: h["from_epoch"]):
        if epoch is not None and int(epoch) >= int(h["from_epoch"]): cur = h
    return dict(cur or {}, max_output_tokens=(cur or c)["max_output_tokens"])


def public(c=None):
    """The block published in mining.json (training.caps) and training.json."""
    c = c or load()
    keys = ("max_output_tokens", "max_context_tokens", "max_model_len", "overlong_soft_start", "overlong_max_tokens", "overlong_penalty",
            "legacy_capped_output_tokens", "last_eos_output_tokens", "history")
    out = {k: c[k] for k in keys}
    out.update(schema=c["schema"], source="https://github.com/AffineFoundation/affine/blob/main/caps.json (loader caps.py)")
    return out


if __name__ == "__main__":
    C = load()
    if len(sys.argv) > 1:
        v = C[sys.argv[1]]
        print(json.dumps(v) if isinstance(v, (list, dict)) else v)
    else:
        print(json.dumps(C, indent=1, sort_keys=True))
