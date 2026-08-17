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
| $UTC | 87007.57450046793 | p3733 |
| Lium balance | **$87007.57450046793** | p3733 |
| cumulative mining spend | ~$116,292 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$77008** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** | $UTC |
| miner coldkey free | τ1261.907 | after unstake+fund (kept ≥τ50) |
| miner stake | **~29.52 α** on r252 (~τ1.68; below τ5 sweep) | 2026-08-17 |
| registrations / submissions | **9** / **9** (… **r252** r33; **r596** chal-00822; **r637** chal-00829) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-17T20:49:33Z | 87007.57450046793 | p3733 **R691 REFUTE** + **R694 CHALL→N80** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T20:44:46Z | 87047.62726802021 | p3732 **R701 TRAIN** lunar 4,5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T20:39:44Z | 87088.57212436352 | p3731 **R691 CHALL→N80** zesty 6,7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T20:34:21Z | 87129.12053348684 | p3730 **R700 TRAIN** brave 2,3 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T20:30:07Z | 87169.80599688251 | p3729 **R699 TRAIN** brave 0,1 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T20:25:21Z | 87169.80599688251 | p3728 **R675 REFUTE** ~0.97× (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T20:20:06Z | 87251.23687742968 | p3727 **R698 TRAIN** crown 6,7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T20:14:30Z | 87291.75608994192 | p3726 **R697 TRAIN** R252 6,7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T20:07:52Z | 87332.8407842355 | p3725 **R696 TRAIN** brave 4,5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T20:02:15Z | 87371.47604053513 | p3724 **R695 TRAIN** R252 4,5 (no rent; B300 empty); burn **~$331.45/h** |
