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
| $UTC | 89701.95981428088 | p3679 |
| Lium balance | **$89701.96** | p3679 |
| cumulative mining spend | ~$113,613 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$79702** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** | $UTC |
| miner coldkey free | τ1261.907 | after unstake+fund (kept ≥τ50) |
| miner stake | **~29.52 α** on r252 (~τ1.68; below τ5 sweep) | 2026-08-17 |
| registrations / submissions | **9** / **9** (… **r252** r33; **r596** chal-00822; **r637** chal-00829) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-17T15:14:14Z | 89701.95981428088 | p3679 **R634 SCP_READY→chall** + **R647 SCP** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:56:55Z | 89823.04036438931 | p3678 **R634 REPAIR** missing shard4 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:53:05Z | 89862.89758784987 | p3677 **R662 TRAIN** zesty 4,5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:48:33Z | 89903.7512585819 | p3676 **R661 TRAIN** brave 6,7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:45:10Z | 89944.11225839579 | p3675 **R660 TRAIN** brave 4,5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:40:48Z | 89944.11225839579 | p3674 **R659 TRAIN** brave 2,3 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:36:50Z | 89984.86832843628 | p3673 **R634 v4 ARM** lean+wait (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:34:05Z | 90024.43853546199 | p3672 **R658 TRAIN** crown 6,7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:31:17Z | 90024.43853546199 | p3671 **R657 TRAIN** crown 4,5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:27:35Z | 90065.92614516531 | p3670 **R656 TRAIN** crown 2,3 (no rent; B300 empty); burn **~$331.45/h** |
