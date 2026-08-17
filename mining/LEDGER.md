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
| $UTC | 88757.17006429855 | p3700 |
| Lium balance | **$88757.17006429855** | p3700 |
| cumulative mining spend | ~$114,546 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$78757** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** | $UTC |
| miner coldkey free | τ1261.907 | after unstake+fund (kept ≥τ50) |
| miner stake | **~29.52 α** on r252 (~τ1.68; below τ5 sweep) | 2026-08-17 |
| registrations / submissions | **9** / **9** (… **r252** r33; **r596** chal-00822; **r637** chal-00829) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-17T17:15:14Z | 88757.17006429855 | p3700 **R677+R678 TRAIN** crown 2–5 UltraExtra (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T17:11:30Z | 88798.04014378539 | p3699 **R653 REFUTE** + **R663 SCP** crown→golden (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T17:05:37Z | 88798.04014378539 | p3698 **R676 TRAIN** crown 0,1 UltraExtra Long HiRank LoBeta (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T17:00:34Z | 88879.64636172207 | p3697 **R653 SCP_READY→CHALL** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:54:51Z | 88891.92650988192 | p3696 **R653+R655 REPAIR** armed (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:51:10Z | 88891.92650988192 | p3695 **R675 TRAIN** R252 6,7 Soft MidRank MidBeta UltraExtra (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:47:51Z | 88932.60495805209 | p3694 **R674 TRAIN** zesty 4,5 UltraExtra Long MidRank LoBeta (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:43:34Z | 88972.87103187863 | p3693 **R652 REFUTE** + **R653 SCP** brave→golden (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:36:51Z | 89014.5473246602 | p3692 **R673 TRAIN** crown 6,7 + API waiter p3692 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:31:56Z | 89053.83944436345 | p3691 **R652 SCP_READY→CHALL** + API waiter p3691 (no rent; B300 empty); burn **~$331.45/h** |
