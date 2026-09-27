#!/usr/bin/env python3
"""Parity harness for the tool baker on real D trace envelopes.

For each harness (policy id pattern) sample N tool-using envelopes from the
fold's trace cache, bake every sampled path with the given teacher's
template and check `parity_ok` (glm: role-marker exempt for tool results;
everything else byte-exact). Prints ok / fail per harness and the first
differing bytes of the first failure. Used for the wvk-25 D1 gate
(GLM-5.3-Flash) and as the Qwen regression check.

  python ops/v20/toolbake_parity.py --repo zai-org/GLM-5.3-Flash --n 60
  python ops/v20/toolbake_parity.py --repo Qwen/Qwen3.8-27B --n 60 --strict
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))

from affine.corpus.trace import ToolParityError, has_tool_traffic, sampled_paths  # noqa: E402
from affine.toolbake import ToolBaker  # noqa: E402

HARNESSES = {
    "bash_tool": ("teacher_bashtool", "king_bashtool", "glm_bashtool"),
    "pi": ("teacher_pi", "king_pi", "glm_pi"),
    "claude_code": ("teacher_claude_code", "king_claude_code"),
    "kimi_code": ("teacher_kimi_code", "king_kimi_code", "teacher_kimi", "king_kimi"),
    "hermes": ("teacher_hermes_agent", "king_hermes_agent", "teacher_hermes", "king_hermes"),
    "terminus": ("teacher_terminus", "king_terminus"),
    "toolcall_null": ("teacher_toolcall", "king_toolcall"),
}


def sample_envelopes(cache: Path, per_harness: int, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    files = sorted(glob.glob(str(cache / "chunks" / "*.jsonl.gz")))
    rng.shuffle(files)
    out: dict[str, list[dict]] = {h: [] for h in HARNESSES}
    for p in files:
        if all(len(v) >= per_harness for v in out.values()):
            break
        try:
            for line in gzip.open(p, "rt"):
                e = json.loads(line)
                pid = str((e.get("policy") or {}).get("id") or "")
                for h, pids in HARNESSES.items():
                    if pid in pids and len(out[h]) < per_harness:
                        out[h].append(e)
        except (OSError, ValueError):
            continue
    return out


def check(baker: ToolBaker, env: dict, strict: bool) -> tuple[int, int, int, tuple | None]:
    """(paths ok, paths ok byte-exact without the role exemption, paths failed, first diff)."""
    trace = env["trace"]
    tools = trace.get("tools") or []
    ok = exact = fail = 0
    first = None
    for msgs in sampled_paths(trace):
        if not has_tool_traffic(trace, msgs):
            continue
        try:
            baked = baker.bake(msgs, tools)
            strict_ok = baker.parity_ok(msgs, tools, baked, role_marker_exempt=False)
            good = strict_ok or (not strict and baker.parity_ok(msgs, tools, baked, role_marker_exempt=True))
        except Exception as e:  # noqa: BLE001
            fail += 1
            first = first or (-1, f"{type(e).__name__}: {e}"[:160], "")
            continue
        if good:
            ok += 1
            exact += strict_ok
        else:
            fail += 1
            first = first or baker.parity_diff(msgs, tools, baked)
    return ok, exact, fail, first


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="teacher repo (default: affine.toml [teacher].repo)")
    ap.add_argument("--n", type=int, default=40, help="envelopes per harness")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--cache", default=str(REPO / "ops" / "corpus_build" / "cache" / "traces"))
    ap.add_argument("--strict", action="store_true", help="no role-marker exemption")
    ap.add_argument("--json", default=None, help="write the summary here")
    a = ap.parse_args()
    baker = ToolBaker.from_pretrained(a.repo)
    print(f"template style: {baker.style.id} (preamble {baker.preamble!r})")
    envs = sample_envelopes(Path(a.cache), a.n, a.seed)
    summary: dict[str, dict] = {}
    for h, lst in envs.items():
        ok = exact = fail = 0
        first = None
        for e in lst:
            o, x, f, d = check(baker, e, a.strict)
            ok += o
            exact += x
            fail += f
            first = first or d
        summary[h] = {"envelopes": len(lst), "paths_ok": ok, "paths_ok_byte_exact": exact, "paths_fail": fail,
                      "parity_rate": round(ok / (ok + fail), 4) if ok + fail else None,
                      "first_diff": list(first) if first else None}
        print(f"{h:14s} envelopes {len(lst):3d}  paths ok {ok:5d} (byte-exact {exact:5d})  fail {fail:4d}  "
              f"rate {summary[h]['parity_rate']}" + (f"  first diff @{first[0]}: want {first[1]!r} got {first[2]!r}" if first else ""))
    tot_ok = sum(v["paths_ok"] for v in summary.values()); tot_f = sum(v["paths_fail"] for v in summary.values())
    print(f"TOTAL paths ok {tot_ok} fail {tot_f} rate {tot_ok / (tot_ok + tot_f):.4f}" if tot_ok + tot_f else "no tool paths")
    if a.json:
        Path(a.json).write_text(json.dumps({"repo": a.repo, "style": baker.style.id, "strict": a.strict,
                                            "n_per_harness": a.n, "seed": a.seed, "harnesses": summary}, indent=1))


if __name__ == "__main__":
    main()
