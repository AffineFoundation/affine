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
| $UTC | 86763.36460387707 | p3738 |
| Lium balance | **$86763.36460387707** | p3738 |
| cumulative mining spend | ~$116,537 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$76763** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** | $UTC |
| miner coldkey free | τ1261.907 | after unstake+fund (kept ≥τ50) |
| miner stake | **~29.52 α** on r252 (~τ1.68; below τ5 sweep) | 2026-08-17 |
| registrations / submissions | **9** / **9** (… **r252** r33; **r596** chal-00822; **r637** chal-00829) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-17T21:17:48Z | 86763.36460387707 | p3738 **R707 TRAIN** + **R683 WIN** harvest (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T21:13:08Z | 86803.6346418794 | p3737 **R705+R706 TRAIN** crown/R252 4,5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T21:08:36Z | 86844.27069591514 | p3736 **R704 TRAIN** brave 6,7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T21:03:39Z | 86883.20978941604 | p3735 **R680 REFUTE** + **R703 TRAIN** zesty 4,5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T20:57:42Z | 86925.78324557716 | p3734 **R694 REFUTE** + **R702 TRAIN** zesty 6,7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T20:49:33Z | 87007.57450046793 | p3733 **R691 REFUTE** + **R694 CHALL→N80** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T20:44:46Z | 87047.62726802021 | p3732 **R701 TRAIN** lunar 4,5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T20:39:44Z | 87088.57212436352 | p3731 **R691 CHALL→N80** zesty 6,7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T20:34:21Z | 87129.12053348684 | p3730 **R700 TRAIN** brave 2,3 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T20:30:07Z | 87169.80599688251 | p3729 **R699 TRAIN** brave 0,1 (no rent; B300 empty); burn **~$331.45/h** |
