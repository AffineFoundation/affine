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
| $UTC | 89418.86732571699 | p3687 |
| Lium balance | **$89418.86732571699** | p3687 |
| cumulative mining spend | ~$113,856 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$79459** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** | $UTC |
| miner coldkey free | τ1261.907 | after unstake+fund (kept ≥τ50) |
| miner stake | **~29.52 α** on r252 (~τ1.68; below τ5 sweep) | 2026-08-17 |
| registrations / submissions | **9** / **9** (… **r252** r33; **r596** chal-00822; **r637** chal-00829) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-17T15:49:45Z | 89418.86732571699 | p3687 **R633 SCP** brave→golden (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T15:45:08Z | 89458.69272416751 | p3686 **R651 REPAIR** shard4 trunc (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T15:41:51Z | 89458.69272416751 | p3685 **R669 TRAIN** brave 0,1 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T15:39:37Z | 89499.30125277331 | p3684 **R647 REFUTE** + **R668 TRAIN** zesty 6,7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T15:34:06Z | 89538.95342149827 | p3683 **R667 TRAIN** R252 6,7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T15:30:08Z | 89580.34088845953 | p3682 **R665/R666 TRAIN** crown 4–7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T15:24:45Z | 89620.56896854432 | p3681 **R634 REFUTE** + **R664 TRAIN** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T15:18:40Z | 89660.83348912947 | p3680 **R663 TRAIN** crown 0,1 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T15:14:14Z | 89701.95981428088 | p3679 **R634 SCP_READY→chall** + **R647 SCP** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:56:55Z | 89823.04036438931 | p3678 **R634 REPAIR** missing shard4 (no rent; B300 empty); burn **~$331.45/h** |
