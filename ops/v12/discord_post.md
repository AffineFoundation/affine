**Payout rule change — a crown is paid for 72 hours, then it earns nothing (effective {EFFECTIVE} UTC). The throne and the duel rule are unchanged.**

Operator directive (Jacob, 2026-09-14 11:09 UTC): a king can only get paid for at most 3 days. Once a model has been king for 72 hours it stops earning, and the emission it was taking goes to the newer crowns. If you hold the king model, the only way to keep earning is to train and crown a new one.

**The rule, in plain words**
• Every **crown** (one win of the throne) is paid for **72 hours** from the moment the verdict lands.
• Every crown still inside its 72 hours gets **one equal share** of the miner emission: 1 crown → 100 %, 2 crowns → 50/50, 3 → one third each.
• A crown **older than 72 hours earns nothing** — even while that model still sits on the throne and keeps judging challengers.
• **No crown inside its window → the emission burns** until somebody wins a new crown.
• One share **per crown**, not per hotkey: dethrone your own king with a better model and both crowns are paid while both are inside their windows.
• Revoked reigns are never paid. Unregistered hotkeys are skipped for that sweep (their share goes to the other paid crowns). Weights refresh every ~20 minutes, so an expiry takes effect within that.

**Worked example (today's board)**
Reign 12 was crowned 2026-09-12 21:48 UTC → paid until **2026-09-15 21:48 UTC**. Reigns 11 and 10 (crowned 2026-09-10) are already past 72 hours → 0 %. So right now: {PAID_SET}. If nobody dethrones reign 12 by 2026-09-15 21:48 UTC, the emission burns from that instant until the next crown. If a challenger wins on 2026-09-15 at 10:00 UTC, it is reign 13 + reign 12 at 50/50 until 21:48 UTC, then reign 13 alone at 100 %. Before today the last five distinct kings each held 20 % indefinitely.

**What does not change**
The crown rule (`margin > max(2·SE, 0.002)`, thought-length floor, B gate), min(R, G), the corpus, admission, one slot per hotkey. This is a payout rule, not a scoring rule, so `weight_version_key` stays 17. Forward-only.

**Read it live:** `https://affine.io/api/v1/snapshot` → `payout` (the paid set now: reign, hotkey, uid, share, `paid_until`), `https://affine.io/api/v1/contract` → `payout`, the Reign table on affine.io, and the spec: https://affine.io/llms.txt → "Payout rule". Exact code: `code/affine/payout.py`.
