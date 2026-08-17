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
| $UTC | 87569.12166206757 | p3721 |
| Lium balance | **$87569.12166206757** | p3721 |
| cumulative mining spend | ~$115,731 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$77569** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** | $UTC |
| miner coldkey free | τ1261.907 | after unstake+fund (kept ≥τ50) |
| miner stake | **~29.52 α** on r252 (~τ1.68; below τ5 sweep) | 2026-08-17 |
| registrations / submissions | **9** / **9** (… **r252** r33; **r596** chal-00822; **r637** chal-00829) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-17T19:41:30Z | 87569.12166206757 | p3721 **R686 CHALL** zesty local (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T19:36:59Z | 87569.12166206757 | p3720 **R693 TRAIN** crown 4,5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T19:32:04Z | 87616.9897861287 | p3719 **R680 SCP** zesty→golden (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T19:28:13Z | 87658.41023194946 | p3718 **R679+R673 REFUTE** + **R692 TRAIN** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T19:22:04Z | 87698.72802150971 | p3717 **R690+R691 TRAIN** brave/zesty 6,7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T19:14:26Z | 87780.64946210934 | p3716 **R689 TRAIN** R252 6,7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T19:07:40Z | 87821.36483452916 | p3715 **TTL+24h** + **R681 REFUTE** + **R673 TK** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T18:57:49Z | 87902.7516655513 | p3714 **R681 CHALL** + **R679 SCP** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T18:51:17Z | 87943.3856544509 | p3713 **R687+R688 TRAIN** brave 2–5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T18:47:03Z | 87984.17573676819 | p3712 **R686 TRAIN** zesty 4,5 (no rent; B300 empty); burn **~$331.45/h** |
