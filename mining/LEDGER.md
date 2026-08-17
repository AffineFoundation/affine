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
| $UTC | 87902.7516655513 | p3714 |
| Lium balance | **$87902.7516655513** | p3714 |
| cumulative mining spend | ~$115,400 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$77903** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** | $UTC |
| miner coldkey free | τ1261.907 | after unstake+fund (kept ≥τ50) |
| miner stake | **~29.52 α** on r252 (~τ1.68; below τ5 sweep) | 2026-08-17 |
| registrations / submissions | **9** / **9** (… **r252** r33; **r596** chal-00822; **r637** chal-00829) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-17T18:57:49Z | 87902.7516655513 | p3714 **R681 CHALL** + **R679 SCP** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T18:51:17Z | 87943.3856544509 | p3713 **R687+R688 TRAIN** brave 2–5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T18:47:03Z | 87984.17573676819 | p3712 **R686 TRAIN** zesty 4,5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T18:41:08Z | 88024.3553383761 | p3711 **R663 REFUTE** + **R681 SCP** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T18:33:32Z | 88105.67977134137 | p3710 **R674 MERGE→CHALL** zesty (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T18:29:01Z | 88146.74284328493 | p3709 **R663 SCP→CHALL** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T18:13:37Z | 88268.48564071755 | p3708 **R685 TRAIN** brave 0,1 UltraExtra (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T18:07:06Z | 88309.64973851852 | p3707 **R684 TRAIN** R252 6,7 UltraExtra (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T18:03:36Z | 88348.46163591537 | p3706 **R655 REFUTE** + **R675 SCP** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T17:57:30Z | 88390.81176785109 | p3705 **R655 N80** + **R682/R683 TRAIN** (no rent; B300 empty); burn **~$331.45/h** |
