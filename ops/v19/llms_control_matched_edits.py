"""llms.txt: describe control_matched (fully matched, 2-ref both sides) and make it the rollback signal. Idempotent."""
from __future__ import annotations
import argparse
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--builder", type=Path, default=REPO / "affine" / "scripts" / "build_llms_txt.py"); a = ap.parse_args()
    s = a.builder.read_text()
    if "control_matched" in s: print("already"); return 0
    old = """teacher's lead on A is larger (+0.29 sd, z +11). The sentence "kings' thoughts \\
predict the teacher's actions better than the teacher's own" that stood here \\
from 09-23 to 09-25 was wrong. The fully matched (2-reference) form of \\
`control_kmatched` replaces the shipped one at the next telemetry deploy; the \\
rollback signal for this fork is read on the fully matched values. Telemetry \\
only.
"""
    if s.count(old) != 1: raise SystemExit("anchor")
    new = """teacher's lead on A is larger (+0.29 sd, z +11). The sentence "kings' thoughts \\
predict the teacher's actions better than the teacher's own" that stood here \\
from 09-23 to 09-25 was wrong. **Since the telemetry deploy of 2026-09-25 every \\
verdict also carries `shadow.sd_meter.by_anchor.loo.control_matched`** (`all`, \\
`R`, `Gc`, `A`: margin, SE, z, n): for each turn and each left-out reference j, \\
the held-out reference and the king are both scored over the *same* k−1 \\
references — the king's R and A recomputed as tempered log-mean-exps over those \\
k−1 pairs — against the mean of those k−1 references, with king forfeits and \\
content-floor turns excluded. `control_kmatched` (the king's R and A over all k \\
references) stays for continuity. **`control_matched` is the rollback signal from \\
2026-09-25 on** (a sign flip against its pre-deploy values: overall mixed, R at \\
parity, A strongly positive, typicality mixed). Telemetry only.
"""
    a.builder.write_text(s.replace(old, new)); print("patched", a.builder); return 0
if __name__ == "__main__":
    raise SystemExit(main())
