"""Phase 1 of the min(R, G) upgrade test: centered-Reason replay.

Replays every stored v4-era duel artifact (k=3 tempered refs) under the
proposed centered reason leg:

    R_turn = LME_tau(R_i) - mean_i(R_i)

where R_i = lpC(y_i|z_A) - lpC(y_i|0) are the stored per-ref reasons. The
LME is shift-equivariant, so this equals tempering the ref-centered R_i.
A flat "insurance" thought profile (same lift on every ref) centers to ~0;
committing to the right ref keeps its spread. Pure re-aggregation of stored
logprobs -- no GPU, no re-sampling, licenses (length floor / B) untouched.

Reads:  affine/state/evals/chal-*.json.gz  (full duel records)
Writes: research/results/centered_reason_replay.{json,txt}

Run from repo root:
    source .venv/bin/activate
    python research/scripts/centered_reason_replay.py
"""

from __future__ import annotations

import gzip
import json
import math
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))

from affine.score import reason, turn_reason  # noqa: E402

EVALS_DIR = REPO / "affine" / "state" / "evals"
OUT_JSON = REPO / "research" / "results" / "centered_reason_replay.json"
OUT_TXT = REPO / "research" / "results" / "centered_reason_replay.txt"

MIN_MARGIN = 0.002
K_SIGMA = 2.0


def turn_scores(rows: list[dict], tau: float) -> dict[str, tuple[float, float]]:
    """turn_id -> (old tempered score, centered score) for valid turns."""
    out = {}
    for r in rows:
        if not (r.get("valid") and r.get("pairs")):
            continue
        a = [reason(p) for p in r["pairs"]]
        old = turn_reason(r["pairs"], tau)
        out[r["turn_id"]] = (old, old - st.mean(a))
    return out


def paired(cs: dict, ks: dict, idx: int) -> dict:
    diffs = [cs[t][idx] - ks[t][idx] for t in sorted(set(cs) & set(ks))]
    n = len(diffs)
    if n < 2:
        return {"n": n, "margin": None, "se": None, "z": None, "wins": False}
    mean = st.mean(diffs)
    se = st.stdev(diffs) / math.sqrt(n)
    z = mean / se if se > 0 else math.inf
    return {"n": n, "margin": mean, "se": se, "z": z,
            "wins_2sigma": mean > K_SIGMA * se,
            "wins": mean > max(K_SIGMA * se, MIN_MARGIN)}


def side_mean(scores: dict, idx: int) -> float | None:
    vals = [v[idx] for v in scores.values()]
    return st.mean(vals) if vals else None


def main() -> None:
    results = []
    for path in sorted(EVALS_DIR.glob("chal-*.json.gz")):
        try:
            d = json.loads(gzip.decompress(path.read_bytes()))
        except Exception as e:  # noqa: BLE001 -- corrupt file should not stop the sweep
            print(f"skip {path.name}: {e}", file=sys.stderr)
            continue
        v = d.get("verdict") or {}
        dp = v.get("duel_params") or {}
        if int(dp.get("n_teacher_samples") or 1) < 2:
            continue  # centering needs k>=2; skip v3-era artifacts
        tau = float(dp.get("tau") or 0.03)
        req = d.get("request") or {}
        cs = turn_scores(d.get("challenger_rows") or [], tau)
        ks = turn_scores(d.get("king_rows") or [], tau)
        if not cs or not ks:
            continue
        old = paired(cs, ks, 0)
        cen = paired(cs, ks, 1)
        results.append({
            "challenge_id": path.stem.replace(".json", ""),
            "challenger": req.get("challenger_repo") or req.get("repo"),
            "king": req.get("king_repo"),
            "stored_margin": v.get("margin"),
            "stored_wins": v.get("challenger_wins"),
            "rejection": v.get("rejection_reason"),
            "old": old,
            "centered": cen,
            "chall_mean_old": side_mean(cs, 0),
            "king_mean_old": side_mean(ks, 0),
            "chall_mean_cen": side_mean(cs, 1),
            "king_mean_cen": side_mean(ks, 1),
        })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=1))

    def short(r: str | None) -> str:
        return (r or "?").split("/")[0][:22]

    lines = [
        f"centered-Reason replay over {len(results)} v4 duels "
        f"(k=3, LME-mean centering; delta={MIN_MARGIN})",
        "",
        f"{'challenge':<11} {'challenger':<23} {'old margin':>11} "
        f"{'old z':>7} {'W':>2} {'cen margin':>11} {'cen z':>7} {'W':>2} "
        f"{'chall cenR':>10} {'king cenR':>10}",
    ]
    for r in results:
        o, c = r["old"], r["centered"]
        lines.append(
            f"{r['challenge_id']:<11} {short(r['challenger']):<23} "
            f"{o['margin']:>11.5f} {o['z']:>7.2f} "
            f"{'Y' if r['stored_wins'] else '.':>2} "
            f"{c['margin']:>11.5f} {c['z']:>7.2f} "
            f"{'Y' if c['wins'] else '.':>2} "
            f"{r['chall_mean_cen']:>10.5f} {r['king_mean_cen']:>10.5f}")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
