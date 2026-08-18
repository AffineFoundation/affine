# LEDGER — money in / money out

**Cap: 40 lines.** Totals + last 10 movements. Older → archive/.
Floor ≥ **$10,000**. Mining B300 burn floor **>$20,000/day = $833/h**.
Live burn = Σ $/h over `mine-*` pods every pass.

**Funding (operator 2026-08-16):** free to convert **SN120 Alpha → TAO → Lium**
(`lium fund -w miner`) to scale mining. Keep ≥ τ50 free coldkey for regs.
**SN120 α cannot pay Lium directly** — CLI only takes TAO or **SN51** α
(`lium fund --alpha …`). Optional SN51 probe OK; log every Alpha/TAO/Lium move.

## Totals

| item | value | as of |
|---|---|---|
| $UTC | 86639.43180152056 | p3816 |
| Lium balance | **$86639.43180152056** | p3816 |
| cumulative mining spend | ~$119,489 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$76639** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** · B300×8=0 | p3816 |
| miner coldkey free | τ1260.384 | after α→TAO→Lium (kept ≥τ50) |
| miner stake | **~59.04α ≈τ3.35** on r252 (below τ5 sweep) | p3815 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T06:40:53Z | 86639.43180152056 | p3816 **R759 N80 LIVE** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T06:35:20Z | 86721.17406911239 | p3815 **API BL fix** fbb1135f (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T06:29:54Z | 86762.78364134047 | p3814 **brave idle→R767+R768 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T06:24:30Z | 86802.95725626276 | p3813 **R337 REAPED** SSH-dead (−$45.60/h); burn **~$331.45/h**; B300×8=0 |
| 2026-08-18T06:18:22Z | 86844.13102679935 | p3812 **R754 REFUTE→R766 TRAIN** (no rent; B300×8=0); burn **~$377.05/h** |
| 2026-08-18T06:12:42Z | 86871.91920548776 | p3811 **R752 REFUTE→R765 TRAIN** (no rent; B300×8=0); burn **~$377.05/h** |
| 2026-08-18T06:06:22Z | 86916.45803607823 | p3810 **R337 rented** golden-lion-72 8×B200 +$45.60/h (waiter 05:53Z); burn **~$377.05/h** |
| 2026-08-18T05:55:23Z | 87005.70014165033 | p3809 **R753+R749 REFUTE→R763+R764 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T05:47:40Z | 87088.3678286418 | p3808 **brave idle→R761+R762 TRAIN** SoftCtx Mega (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T05:41:12Z | 87128.67186157257 | p3807 **R751 REFUTE** → **R760 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
