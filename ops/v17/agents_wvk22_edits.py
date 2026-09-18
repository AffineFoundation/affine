"""AGENTS.md snapshot for wvk 22 (run AFTER the flip, with the first wvk-22
verdict id). Idempotent.

  python ops/v17/agents_wvk22_edits.py --flip-time "2026-09-18 23:1x UTC" --first-verdict chal-00589 --first-z 0.4 --first-se 0.028
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AGENTS = REPO / "AGENTS.md"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip-time", required=True)
    ap.add_argument("--first-verdict", required=True)
    ap.add_argument("--first-margin", required=True)
    ap.add_argument("--first-se", required=True)
    ap.add_argument("--first-z", required=True)
    ap.add_argument("--first-seconds", required=True)
    a = ap.parse_args()
    s = AGENTS.read_text()
    if "weight_version_key = 22" in s:
        print("already patched")
        return 0

    def sub(old: str, new: str) -> None:
        nonlocal s
        if s.count(old) != 1:
            raise SystemExit(f"anchor not unique/missing: {old[:70]!r}")
        s = s.replace(old, new)

    sub("- `weight_version_key = 21` (2026-09-17 10:52 UTC, explicit dated operator\n",
        f"""- `weight_version_key = 22` ({a.flip_time}, explicit dated operator
  directive 2026-09-18 10:04 UTC "I like it. And I want to ship it" / 10:25
  "lets reduce the turns to 1000" / go 15:11 "only when the current queued
  models have run"): **the sd-meter is the rule** — `score_mode =
  "sd_min_rga"`, `n_turns = 1000`, `[duel.sd_meter]` `min_margin_sd = 0.07`,
  `k_sigma = 2.0`, `forfeit_sd = -4.5`, `content_lift_nats = 1.0`,
  `content_min_tokens = 10`, `typicality_width = 2.0`, `a_norm_bytes = 1.0`,
  `anchor = "loo"`. turn = min(z_R, typ_c, z_A): the largest standardised
  deviation of the reply from the teacher's own k = 3 samples across thought
  typicality on content tokens (|lpC(tok|x) − lpC(tok|∅)| > 1 nat),
  thought→action (the live centred R) and action←thought (summed A leg), in
  teacher-sd units — μ per turn from the refs' leave-one-out values (6 cross
  echoes, shared), σ pooled per dialect over the duel. `band_c` /
  `band_floor` / `min_margin` / `forfeit_turn_score` stay in the toml for
  wvk ≤ 21 replay only. Code: `affine/evalsrv/sdmeter.py` (+ `terms.py` /
  `vllm_client.py` cache-aware echoes with per-tag cost accounting,
  `dueling.py` `sd_min_rga` decide path), `affine/affine/config.py` /
  `score.py`; ops `ops/v17/` (shadow deploy, notice, flip, rollback,
  `ops/sd-meter/refresh_frozen.py`). Shadow read 10:41–23:xx UTC on every
  verdict (`verdict.shadow.sd_meter`, both anchors): 3 duels all sane — LOO
  and frozen agreed 3/3 with the live decision, positive control teacher vs
  king z +3.4 / +5.9 / +3.6, no leg bound > 45 %, σ per dialect within 1.2× of
  phase-2; cost +80 % echo requests / +57 % prompt tokens / +60–63 % computed
  tokens, wall +15–19 % at n = 1300. Calibration: δ = 0.082·sd_diff (last 40
  live verdicts) × measured sd_diff 0.89 → 0.07; the live −0.1 floor sat at
  −4.7…−5.3 sd of live valid turns (the phase-2 −2.4 was a wider panel) → −4.5
  (a 2 % forfeit gap ≈ one δ, as under wvk 12). Cutoff: the queue as of
  15:11 UTC (`chal-00582`…`00588`) ran under wvk 21 first; reign 15
  (`chal-00581`, crowned 14:59 UTC under wvk 21, sd-meter agreed: +0.139 sd,
  z 4.73) stands; forward-only. First wvk-22 verdict `{a.first_verdict}`:
  margin {a.first_margin} sd, SE {a.first_se}, z {a.first_z}, {a.first_seconds} s.
  Rollback rule (first 3 verdicts; `ops/v17/rollback_wvk22.sh`): SE > 2× the
  shadow's, teacher-vs-king z ≤ −2, any leg binds > 80 %, a leg dropped on
  > 5 % of valid turns. Public claim unchanged: a better distillation meter,
  not benchmark alignment. Plan + status: project store `docs/wvk22-plan.md`.
  21 = 2026-09-17 10:52 UTC, explicit dated operator
""")
    sub("## 2. Frozen production scoring — min(R,G) v5: centered Reason + banded Grounding + δ + length floor + B gate (2026-08-27, `weight_version_key = 10`, genesis reset)\n",
        """## 2. Frozen production scoring — sd-meter min(z_R, typ_c, z_A) since wvk 22 (2026-09-18); min(R,G) v5 below is the wvk 10–21 rule

**Live rule since wvk 22 (2026-09-18):** `score_mode = "sd_min_rga"`, `n_turns =
1000` — turn = min(z_R, typ_c, z_A) in teacher-sd units (definition, knobs and
calibration in §4 under `weight_version_key = 22`; module
`affine/evalsrv/sdmeter.py`). R below is still the R leg; G (the band) is
replaced by content-token typicality; the A leg (summed) is live. Gates
(thought-length floor, B licence, protocol probe) unchanged.

### History — min(R,G) v5 (wvk 10–21): centered Reason + banded Grounding + δ + length floor + B gate (2026-08-27, `weight_version_key = 10`, genesis reset)
""")
    sub("> Affine SN120: teacher-anchored thought-injection duels. Since 2026-08-27\n",
        """> **Since wvk 22 (2026-09-18) the live rule is the sd-meter:** turn =
> min(z_R, typ_c, z_A) — the largest standardised deviation of the reply from
> the teacher's own three samples across content-token thought typicality,
> thought→action (centred R) and action←thought (summed A), in teacher-sd
> units (leave-one-out μ per turn, σ pooled per dialect); 1,000-turn slices;
> crown iff paired mean > max(2·SE, 0.07 sd) + gates; forfeit −4.5 sd. The
> paragraph below describes the wvk 10–21 rule it replaced.
>
> Affine SN120: teacher-anchored thought-injection duels. Since 2026-08-27
""")
    AGENTS.write_text(s)
    print("patched", AGENTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
