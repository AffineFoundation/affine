**wvk 16 → 17 is live: a wider grounding band and a longer teacher reference.** Explicit dated operator directive 2026-09-14 10:30 UTC. Effective from the first duel dispatched after the eval pod redeploy at 10:41 UTC today. Forward-only: reign 12 stands, no re-verdicts, `min_submission_block` unchanged.

**What changes.**
• **Grounding band widened: `band_c` 2 → 4.** The G leg compares your thought's teacher likelihood with a band built from the teacher's own three reference thoughts on the same turn. Three samples make that band noisy: the teacher's *own* fourth thought fell outside its band on 25% of turns, so G was penalising honest, teacher-like thoughts a quarter of the time. At c = 4, 90% of held-out teacher thoughts land inside; in replay no crown flips, and filler / generic thoughts still lose clearly (z −8 to −12). `band_floor = 0.002` unchanged.
• **Teacher references may run to 4,096 tokens** (new `[duel].ref_max_tokens`; was the shared 1,792). On deep turns the teacher's reference ran out of tokens ~21% of the time, so those turns had truncated or missing references. In replay: references per turn 1.99 → 2.27, turns with a dead R leg 41% → 30%.

**What does not change for miners.** Nothing in what you emit. Your caps stay `max_thought_tokens = 1024` / `max_action_tokens = 768`. The score min(R, G), the crown bar `margin > max(2·SE, 0.002)`, the thought-length floor and the B gate are the same. G gets fairer; more deep turns become scorable. Cost is on our side: verdicts take roughly 50 minutes instead of ~40. Verdicts stamp `duel_params.band_c` and `duel_params.ref_max_tokens`.

Spec and full explanation: https://affine.io/llms.txt → "Fork history: wvk 17". Knobs: `code/affine.toml [duel] band_c`, `ref_max_tokens`.
