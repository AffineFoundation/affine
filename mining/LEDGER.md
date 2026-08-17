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
| $UTC | 88879.64636172207 | p3697 |
| Lium balance | **$88879.64636172207** | p3697 |
| cumulative mining spend | ~$114,423 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$78852** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** | $UTC |
| miner coldkey free | τ1261.907 | after unstake+fund (kept ≥τ50) |
| miner stake | **~29.52 α** on r252 (~τ1.68; below τ5 sweep) | 2026-08-17 |
| registrations / submissions | **9** / **9** (… **r252** r33; **r596** chal-00822; **r637** chal-00829) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-17T17:00:34Z | 88879.64636172207 | p3697 **R653 SCP_READY→CHALL** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:54:51Z | 88891.92650988192 | p3696 **R653+R655 REPAIR** armed (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:51:10Z | 88891.92650988192 | p3695 **R675 TRAIN** R252 6,7 Soft MidRank MidBeta UltraExtra (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:47:51Z | 88932.60495805209 | p3694 **R674 TRAIN** zesty 4,5 UltraExtra Long MidRank LoBeta (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:43:34Z | 88972.87103187863 | p3693 **R652 REFUTE** + **R653 SCP** brave→golden (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:36:51Z | 89014.5473246602 | p3692 **R673 TRAIN** crown 6,7 + API waiter p3692 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:31:56Z | 89053.83944436345 | p3691 **R652 SCP_READY→CHALL** + API waiter p3691 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:17:35Z | 89176.17995128914 | p3690 **R633 REFUTE** + **R652 SCP** brave→golden (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T16:05:31Z | 89257.27615641385 | p3689 **R670/R671/R672 TRAIN** brave 2–7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T15:58:49Z | 89338.21618472386 | p3688 **R651 REFUTE** + **R655 SCP** R252→lunar (no rent; B300 empty); burn **~$331.45/h** |
