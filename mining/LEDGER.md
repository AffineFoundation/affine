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
| $UTC | 88309.64973851852 | p3707 |
| Lium balance | **$88309.64973851852** | p3707 |
| cumulative mining spend | ~$114,994 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$78310** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** | $UTC |
| miner coldkey free | τ1261.907 | after unstake+fund (kept ≥τ50) |
| miner stake | **~29.52 α** on r252 (~τ1.68; below τ5 sweep) | 2026-08-17 |
| registrations / submissions | **9** / **9** (… **r252** r33; **r596** chal-00822; **r637** chal-00829) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-17T18:07:06Z | 88309.64973851852 | p3707 **R684 TRAIN** R252 6,7 UltraExtra (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T18:03:36Z | 88348.46163591537 | p3706 **R655 REFUTE** + **R675 SCP** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T17:57:30Z | 88390.81176785109 | p3705 **R655 N80** + **R682/R683 TRAIN** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T17:50:53Z | 88431.73763177953 | p3704 **R655 SCP_READY→CHALL** lunar 4,5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T17:27:45Z | 88634.85122193607 | p3703 **R681 TRAIN** brave 4,5 UltraExtra (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T17:24:44Z | 88675.56647358203 | p3702 **R680 TRAIN** zesty 6,7 UltraExtra (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T17:20:22Z | 88716.21528076984 | p3701 **R679 TRAIN** brave 2,3 UltraExtra (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T17:15:14Z | 88757.17006429855 | p3700 **R677+R678 TRAIN** crown 2–5 UltraExtra (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T17:11:30Z | 88798.04014378539 | p3699 **R653 REFUTE** + **R663 SCP** crown→golden (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T17:05:37Z | 88798.04014378539 | p3698 **R676 TRAIN** crown 0,1 UltraExtra Long HiRank LoBeta (no rent; B300 empty); burn **~$331.45/h** |
