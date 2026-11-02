# Task-level cap rule (from E507, 2026-11-01)

Jacob 2026-11-01 19:42-19:43 CR: one flat cap rule over the whole baseline, no env-specific rules.

- **Rule:** RED if >= 33% of the candidate's baseline answers hit the token cap (8,192 tokens, no EOS).
  cap-hit = capped / all candidate rollouts on the scored baseline tasks, every task counted alike (`CAP_HIT_MAX = 0.33`
  in `priors/baseline_v1.py`, `heldout.prior_v1.cap_max` in the gate flag). It replaces the affine_math-only rule
  (Hajek capped share on affine_math < 0.20), which applied before E507.
- The head's cap-hit is computed the same way and recorded in each decision (`h_cap_rate`), not part of the rule.
- max_tokens, the draw, the grader and the rest of the GREEN rule (Delta > 0, z > 1.00, >= 100 tasks scored) are unchanged.
- Offline replay of E497-E505 with the new rule: every outcome unchanged (all RED on gain/z; candidate task-level cap-hit
  0.188-0.216, head 0.189-0.192).
- Files: `activate_cap.sh` (install at a quiet epoch gap, health check, `MODE=revert`), `watchdog_cap.py` (auto-revert on gate
  errors, stalls or a decision without the rule), `setflag_cap.py`, `patch_texts_cap.py` (llms.txt), `replay_cap.py`.
