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
| $UTC | 87332.8407842355 | p3725 |
| Lium balance | **$87332.8407842355** | p3725 |
| cumulative mining spend | ~$115,967 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$77333** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** | $UTC |
| miner coldkey free | τ1261.907 | after unstake+fund (kept ≥τ50) |
| miner stake | **~29.52 α** on r252 (~τ1.68; below τ5 sweep) | 2026-08-17 |
| registrations / submissions | **9** / **9** (… **r252** r33; **r596** chal-00822; **r637** chal-00829) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-17T20:07:52Z | 87332.8407842355 | p3725 **R696 TRAIN** brave 4,5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T20:02:15Z | 87371.47604053513 | p3724 **R695 TRAIN** R252 4,5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T19:56:50Z | 87413.77000699901 | p3723 **R683 SCP_QUEUED** gate R680 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T19:51:44Z | 87454.8892824822 | p3722 **R686 REFUTE** + **R694 TRAIN** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T19:41:30Z | 87569.12166206757 | p3721 **R686 CHALL** zesty local (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T19:36:59Z | 87569.12166206757 | p3720 **R693 TRAIN** crown 4,5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T19:32:04Z | 87616.9897861287 | p3719 **R680 SCP** zesty→golden (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T19:28:13Z | 87658.41023194946 | p3718 **R679+R673 REFUTE** + **R692 TRAIN** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T19:22:04Z | 87698.72802150971 | p3717 **R690+R691 TRAIN** brave/zesty 6,7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T19:14:26Z | 87780.64946210934 | p3716 **R689 TRAIN** R252 6,7 (no rent; B300 empty); burn **~$331.45/h** |
