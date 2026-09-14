**Payout rule change — a crown is paid for 72 hours, then it earns nothing (effective {EFFECTIVE} UTC). The throne and the duel rule are unchanged.**

Operator directive (Jacob, 2026-09-14 11:09 UTC): a king can only get paid for at most 3 days. After 72 hours as king a model stops earning and its share goes to the newer crowns. If you hold the king, the way to keep earning is to train and crown a new model.

**The rule**
• Every **crown** (one win of the throne) is paid for **72 hours** from the verdict.
• Every crown inside its 72 hours gets **one equal share**: 1 crown → 100 %, 2 → 50/50, 3 → thirds.
• A crown **older than 72 hours earns nothing**, even while it still holds the throne and judges challengers.
• **No crown inside its window → the emission burns** until the next crown.
• One share **per crown**, not per hotkey: dethrone your own king with a better model and both crowns are paid while both are in their windows.
• Revoked reigns are never paid; unregistered hotkeys are skipped for that sweep. Weights refresh every ~20 min.

**Example (today's board)**
Reign 12 was crowned 2026-09-12 21:48 UTC → paid until **2026-09-15 21:48 UTC**. Reigns 11 and 10 (2026-09-10) are past 72 h → 0 %. Now: {PAID_SET}. If nobody dethrones reign 12 by 2026-09-15 21:48 UTC, the emission burns until the next crown. A win on 2026-09-15 10:00 UTC gives reign 13 + reign 12 at 50/50 until 21:48, then reign 13 alone. Before today the last five distinct kings held 20 % each indefinitely.

**Unchanged:** crown rule `margin > max(2·SE, 0.002)`, thought-length floor, B gate, min(R, G), corpus, admission, one slot per hotkey. Payout rule, not scoring: `weight_version_key` stays 17. Forward-only.

**Live:** https://affine.io/api/v1/snapshot → `payout` (paid set, share, `paid_until`); `/api/v1/contract` → `payout`; Reign table on affine.io; spec https://affine.io/llms.txt → "Payout rule"; code `code/affine/payout.py`.
