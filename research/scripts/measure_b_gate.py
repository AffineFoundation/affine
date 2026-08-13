"""Measure teacher-side B pass rates on stored duel artifacts.

B = lpC(y_A|z_A) − lpC(y_A|∅). The gate is B ≥ τ=0.02 and no leakage,
with miner valid if pass rate ≥ γ=0.30. Score stays Reason.

s4-era artifacts have the B echoes (full telemetry). Live Reason-only
n80 artifacts (thermo, legend) do not: those rows report missing B,
leakage, and empty-thought share. Empty z ⇒ B = 0 by construction.

Run from repo root:
    source .venv/bin/activate && python research/scripts/measure_b_gate.py
"""

from __future__ import annotations

import json
import statistics as st
from collections import Counter
from pathlib import Path

from affine.score import (
    DEFAULT_CAUSALITY_TAU,
    b_gate_pass,
    leakage,
    reason,
    teacher_causality,
)

ROOT = Path(__file__).resolve().parents[2]
TAU = DEFAULT_CAUSALITY_TAU
GAMMA = 0.30

# Full-telemetry (has lpC_ya_*). One representative per family.
S4_ARTIFACTS = [
    ROOT / "mining/experiments/s4-h30-m7-king-self/results/h30_sim_result_artifact.json",
    ROOT / "mining/experiments/s4-h29-king-self-clip-l1/results/h29_sim_result_artifact.json",
    ROOT / "mining/experiments/s4-h117-f22-raw-everest12/results/h117_sim_result_artifact.json",
    ROOT / "mining/experiments/s4-h44-m7-winner-za-clip08/results/h44_sim_result_artifact.json",
    ROOT / "mining/experiments/s4-h43-m7-winner-za-a64/results/h43_sim_result_artifact.json",
    ROOT / "mining/experiments/s4-h42-m7-winner-za-lr5e6/results/h42_sim_result_artifact.json",
    ROOT / "mining/experiments/s4-h41-m7-winner-za-r32/results/h41_sim_result_artifact.json",
    ROOT / "mining/experiments/s4-h39-m7-winner-za-lr3e5/results/h39_sim_result_artifact.json",
    ROOT / "mining/experiments/s4-h1v2-sft/results/h1v2_sim_result_artifact.json",
    ROOT / "mining/experiments/s4-h1-sft/results/h1_sim_result_n40_artifact.json",
    ROOT / "mining/experiments/s4-h2-merge/results/h2_kp65_sim_result_artifact.json",
    ROOT / "mining/experiments/s4-h5b-talentpigs-distill/results/h5b_sim_result_artifact.json",
]

# Reason-only n80 (no B echoes). Cue / empty kings.
REASON_ONLY = [
    ROOT / "mining/experiments/r160-thermopylae-grpo/artifacts/r160_mid50_sim_result_artifact.json",
    ROOT / "mining/experiments/r160-thermopylae-grpo/artifacts/r160_mid150_sim_result_artifact.json",
    ROOT / "mining/experiments/r68-hirank-hilr/artifacts/r68_final_sim_result_artifact.json",
    ROOT / "mining/experiments/r159-legend-grpo/artifacts/r159_final_sim_artifact_p2448.json",
    ROOT / "mining/experiments/r17-coder-rl/artifacts/h135_sim_result_artifact_guass_p2210.json",
]


def _pairs(rows: list[dict]) -> list[dict]:
    return [p for r in rows if r.get("valid") and "pairs" in r for p in r["pairs"]]


def _side_stats(rows: list[dict]) -> dict | None:
    pairs = _pairs(rows)
    if not pairs:
        return None
    miner = rows[0].get("miner", "?")
    zs = [(p.get("z_a") or "") for p in pairs]
    stripped = [z.strip() for z in zs]
    lens = [len(s) for s in stripped]
    empty = sum(1 for s in stripped if not s)
    leak = sum(1 for p in pairs
               if leakage(p.get("z_a") or "", p.get("y_a") or ""))
    b_vals = [teacher_causality(p) for p in pairs]
    have_b = [v for v in b_vals if v is not None]
    flags = [b_gate_pass(p, tau=TAU) for p in pairs]
    flags_have = [g for g in flags if g is not None]
    reasons = [reason(p) for p in pairs]
    top_z = Counter(stripped).most_common(3)
    return {
        "miner": miner,
        "n_pairs": len(pairs),
        "median_len_z": float(st.median(lens)),
        "mean_len_z": float(st.mean(lens)),
        "empty_frac": empty / len(pairs),
        "leak_frac": leak / len(pairs),
        "mean_reason": float(st.mean(reasons)),
        "has_b": bool(have_b),
        "n_b": len(have_b),
        "mean_b": (float(st.mean(have_b)) if have_b else None),
        "b_pass_rate": (float(st.mean(1.0 if g else 0.0 for g in flags_have))
                        if flags_have else None),
        "would_pass_gamma": (
            None if not flags_have
            else st.mean(1.0 if g else 0.0 for g in flags_have) >= GAMMA),
        "top_z": [(z[:40], n) for z, n in top_z],
    }


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def _fmt_rate(v: float | None) -> str:
    return "  n/a" if v is None else f"{v:5.1%}"


def _fmt_num(v: float | None) -> str:
    return "   n/a" if v is None else f"{v:+6.4f}"


def main() -> None:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("B gate measurement  τ=0.02  γ=0.30")
    emit("B = lpC(y_A|z_A) − lpC(y_A|∅)  + no leakage")
    emit("Gate is OFF live (causality_gate=false). This is a try, not a fork.")
    emit()
    hdr = (f"{'artifact':<28} {'side':<12} {'n':>4} {'med_z':>6} "
           f"{'empty':>6} {'leak':>6} {'mean_B':>8} {'pass':>6} "
           f"{'γ':>4} {'Reason':>8}")
    emit(hdr)
    emit("-" * len(hdr))

    def row(label: str, side: str, s: dict | None) -> None:
        if s is None:
            emit(f"{label:<28} {side:<12}  (no pairs)")
            return
        gamma = ("ok" if s["would_pass_gamma"] else "NO"
                 if s["would_pass_gamma"] is not None else "—")
        emit(
            f"{label:<28} {side:<12} {s['n_pairs']:>4} "
            f"{s['median_len_z']:>6.0f} {_fmt_rate(s['empty_frac'])} "
            f"{_fmt_rate(s['leak_frac'])} {_fmt_num(s['mean_b'])} "
            f"{_fmt_rate(s['b_pass_rate'])} {gamma:>4} "
            f"{s['mean_reason']:+8.4f}"
        )
        for z, n in s["top_z"]:
            shown = z if z else "(empty)"
            emit(f"{'':28}   z×{n:<4} {shown!r}")

    emit("=== full telemetry (B measurable) ===")
    for path in S4_ARTIFACTS:
        obj = _load(path)
        label = path.parent.parent.name[:28]
        if obj is None:
            emit(f"{label:<28} MISSING {path}")
            continue
        if "challenger_rows" not in obj:
            emit(f"{label:<28} skipped ({obj.get('rejection_reason', 'no rows')})")
            continue
        row(label, "challenger", _side_stats(obj["challenger_rows"]))
        row("", "king", _side_stats(obj["king_rows"]))

    emit()
    emit("=== Reason-only (B echoes omitted) ===")
    emit("Empty z ⇒ B=0 by construction. Cue phrases are unmeasured until")
    emit("a GLM echo of y_A with that z. Leakage 0% does not pass B.")
    for path in REASON_ONLY:
        obj = _load(path)
        label = path.parent.parent.name[:28]
        if obj is None:
            emit(f"{label:<28} MISSING {path}")
            continue
        if "challenger_rows" not in obj:
            emit(f"{label:<28} skipped ({obj.get('rejection_reason', 'no rows')})")
            continue
        row(label, "challenger", _side_stats(obj["challenger_rows"]))
        row("", "king", _side_stats(obj["king_rows"]))

    out = ROOT / "research/results/b_gate_measure.txt"
    out.write_text("\n".join(lines) + "\n")
    emit()
    emit(f"wrote {out}")


if __name__ == "__main__":
    main()
