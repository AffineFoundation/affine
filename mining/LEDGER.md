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
| $UTC | 85745.25562401353 | p3752 |
| Lium balance | **$85745.25562401353** | p3752 |
| cumulative mining spend | ~$117,556 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$75745** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** | $UTC |
| miner coldkey free | τ1260.382 | after R683 reg burn (~τ1.53; kept ≥τ50) |
| miner stake | **~29.52 α** on r252 (~τ1.68; below τ5 sweep) | 2026-08-17 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-17T23:22:29Z | 85745.25562401353 | p3752 **R715+R716 TRAIN** brave 4–7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-17T23:17:07Z | 85786.49413103018 | p3751 **R693 N80 LIVE** crown 4,5 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-17T23:12:57Z | 85826.5515667286 | p3750 **R713+R714 TRAIN** brave 0–3 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-17T22:44:00Z | 86071.169199442 | p3749 **R705 REFUTE** + **R706/R707** R252 n80/chall (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T22:34:00Z | 86152.07244358957 | p3748 **R705 N80 LIVE** crown 4,5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T22:25:16Z | 86233.67342036712 | p3747 **R703 REFUTE** + **R709 TRAIN** zesty 4,5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T22:20:49Z | 86233.67342036712 | p3746 **R708 TRAIN** zesty 6,7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T22:15:57Z | 86274.26724677208 | p3745 **R703 N80 LIVE** + **R702 REFUTE** (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T22:10:35Z | 86355.76560593078 | p3744 **R702 N80 LIVE** zesty 6,7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T22:02:38Z | 86394.18774338797 | p3743 **R696 SCP** brave→crown + R683 LOST + R698 REFUTE (no rent; B300 empty); burn **~$331.45/h** |
