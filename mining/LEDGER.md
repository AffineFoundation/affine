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
| $UTC | 86871.91920548776 | p3811 |
| Lium balance | **$86871.91920548776** | p3811 |
| cumulative mining spend | ~$119,257 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$76872** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$377.05/h** · vs floor $833/h · gap **−$455.95/h** · B300×8=0 | p3811 |
| miner coldkey free | τ1260.384 | after α→TAO→Lium (kept ≥τ50) |
| miner stake | **~29.52α ≈τ1.68** on r252 (below τ5 sweep) | p3811 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T06:12:42Z | 86871.91920548776 | p3811 **R752 REFUTE→R765 TRAIN** (no rent; B300×8=0); burn **~$377.05/h** |
| 2026-08-18T06:06:22Z | 86916.45803607823 | p3810 **R337 rented** golden-lion-72 8×B200 +$45.60/h (waiter 05:53Z); burn **~$377.05/h** |
| 2026-08-18T05:55:23Z | 87005.70014165033 | p3809 **R753+R749 REFUTE→R763+R764 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T05:47:40Z | 87088.3678286418 | p3808 **brave idle→R761+R762 TRAIN** SoftCtx Mega (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T05:41:12Z | 87128.67186157257 | p3807 **R751 REFUTE** → **R760 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T05:36:22Z | 87170.06851375397 | p3806 **R749+R752+R753 wait→n80 ARMED** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T05:31:52Z | 87209.36908915383 | p3805 **R750 REFUTE** → **R759 TRAIN** + **R751 n80 LIVE** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T05:23:55Z | 87291.25110727787 | p3804 **R748+R716 REFUTE** → **R757+R758 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T05:18:26Z | 87332.50039745866 | p3803 **R750 MERGE idle→n80 LIVE** R252 4,5 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T05:12:14Z | 87413.93831595869 | p3802 **R748 MERGE idle→n80 LIVE** zesty 4,5 (no rent; B300×8=0); burn **~$331.45/h** |
