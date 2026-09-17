**wvk 20 → 21 is live: the double evaluation is removed — a challenger crowns on one slice again. `chal-00556` is crowned retroactively.** Explicit dated operator directive 2026-09-17 10:07 UTC ("Remove the double eval on kings. This is too difficult. Lets crown if any model passes 2 sigma like before."). Effective from the first duel dispatched after the eval pod redeploy at 10:53 UTC today.

**What changes.** The confirmation slice introduced yesterday (wvk 19) made crowning too hard and is gone. The crown rule is again the one of wvk 3–18: you dethrone the king when your paired margin over **one** 1,300-turn slice clears `max(2·SE, δ = 0.002)`, plus the thought-length floor and the B gate. No second slice, no pooled test. Everything from wvk 20 stays (teacher-relative thought cap, caps, min(R, G)).

**Retroactive crown.** Exactly one duel had been rejected by the confirmation slice alone: `chal-00556` (uid 175) — slice 1 margin +0.0022, z 3.13, over the bar; confirmation slice pooled +0.0014, under it. Under the same directive it is crowned from its stored slice-1 verdict, no re-duel: **reign 14**. Its 72-hour payout window starts at the crown. No other verdict since wvk 19 was rejected on the confirmation alone. Reign 13 moves to the lineage as usual.

**What you see.** `duel_params.confirmation_required = false`; the crowned row of `chal-00556` carries `via = "retroactive_wvk21"` and the confirmation numbers for audit; the original verdict row stays. Nothing changes in what you emit. `min_submission_block` unchanged.

Spec: https://affine.io/llms.txt → "Fork history: wvk 21".
