"""Read the protocol-probe shadow (code-fence) cases on the verdicts since the shadow deploy
and put them next to the offline HumanEval read of the same digest when a bench run exists.

  python ops/v21/probe_shadow_read.py [--since 2026-09-27T09:30] [--runs ~/benchsuite/runs]

Live shadow = verdict.protocol_probe.shadow (pass_rate over the shadow prompts, by_reason).
Offline = check_code_block(fence_required=False) over the 164 stored HumanEval replies of
the model's bench run (king/humaneval__t0/traces.jsonl), i.e. the code_only rule, plus
the HumanEval pass rate. Promotion criterion (operator 2026-09-27 09:40 UTC): after ~10
verdicts the live shadow rates agree with the offline table (reign-22-like ≈ 0.6–0.8,
genesis-like ≈ 1.0)."""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))
from evalsrv import protocol_probe as pp  # noqa: E402


def offline(run_dir: str) -> tuple[int, float, float, dict] | None:
    p = f"{run_dir}/king/humaneval__t0/traces.jsonl"
    if not os.path.exists(p):
        p += ".gz"
        if not os.path.exists(p):
            return None
    op = gzip.open if p.endswith(".gz") else open
    n = ok_fence = ok_he = 0
    by: collections.Counter = collections.Counter()
    for line in op(p, "rt"):
        d = json.loads(line)
        for t in d.get("traces") or []:
            for node in t.get("nodes") or []:
                m = node.get("message") or {}
                if m.get("role") != "assistant":
                    continue
                n += 1
                r = pp.check_code_block(m.get("content") or "", fence_required=False)
                ok_fence += r["ok"]
                by.update(r["reasons"])
                ok_he += bool((t.get("rewards") or {}).get("passed", {}).get("score"))
    return (n, ok_fence / n, ok_he / n, dict(by)) if n else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-27T09:30")
    ap.add_argument("--runs", default=os.path.expanduser("~/benchsuite/runs"))
    ap.add_argument("--history", default=str(REPO / "affine/state/history.jsonl"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.history) if l.strip()]
    runs_by_digest: dict[str, list[str]] = collections.defaultdict(list)
    runs_by_cid: dict[str, list[str]] = collections.defaultdict(list)
    for run in glob.glob(f"{a.runs}/*"):
        name = os.path.basename(run)
        parts = name.split("-")
        if len(parts) >= 2:
            runs_by_digest[parts[1]].append(run)
        if "chal-" in name:
            runs_by_cid["chal-" + name.split("chal-")[1][:5]].append(run)
    print(f"{'duel':10} {'at':16} {'uid':>4} {'digest':12} {'enforced':>8} {'shadow':>7} {'n':>3} shadow reasons | offline code_only  HE pass  (run)")
    live_rates: list[tuple[float, float | None]] = []
    for r in rows:
        if r.get("at", "") < a.since or r.get("event") not in ("verdict", "crowned", "failed"):
            continue
        v = r.get("verdict") or {}
        probe = v.get("protocol_probe") or {}
        if not probe:
            continue
        sh = probe.get("shadow") or {}
        digest = (r.get("revision") or "")[:12]
        runs = runs_by_digest.get(digest) or runs_by_cid.get(r["challenge_id"]) or []
        off = None
        for run in sorted(runs, reverse=True):
            off = offline(run)
            if off:
                break
        off_s = (f"{off[1]:.2f}  {off[2]:.3f}  ({os.path.basename(sorted(runs, reverse=True)[0])})"
                 if off else "—")
        shadow_rate = sh.get("pass_rate")
        live_rates.append((shadow_rate, off[1] if off else None))
        print(f"{r['challenge_id']:10} {r['at'][:16]} {str(r.get('uid')):>4} {digest:12} "
              f"{probe.get('pass_rate', float('nan')):8.2f} {shadow_rate if shadow_rate is not None else float('nan'):7.2f} "
              f"{sh.get('n', 0):3d} {sh.get('by_reason')} | {off_s}")
    pairs = [(l, o) for l, o in live_rates if l is not None and o is not None]
    if pairs:
        diffs = [l - o for l, o in pairs]
        print(f"\n{len(pairs)} verdicts with both reads: mean(live − offline) {sum(diffs)/len(diffs):+.3f}, "
              f"max |diff| {max(abs(d) for d in diffs):.3f}")
    print(f"{len(live_rates)} verdicts with a shadow block since {a.since}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
