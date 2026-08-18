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
| $UTC | 87849.86205615434 | p3775 |
| Lium balance | **$87849.86205615434** | p3775 |
| cumulative mining spend | ~$118,813 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$77850** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** · B300×8=0 | $UTC |
| miner coldkey free | τ1260.384 | after α→TAO→Lium (kept ≥τ50) |
| miner stake | **0** (r252 swept p3763) | p3763 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T02:09:00Z | 87849.86205615434 | p3775 **R727 REFUTE** + **R737 TRAIN** crown 6,7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T02:01:52Z | 87887.57828641804 | p3774 **R726 REFUTE** + **R736 TRAIN** crown 4,5 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T01:56:59Z | 87931.00583067711 | p3773 **brave cold TK+R715/R716** armed (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T01:50:09Z | 88012.08748004797 | p3772 **R726+R727 MERGE→CHALL→N80 LIVE** crown 4–7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T01:42:16Z | 88053.65090549592 | p3771 **R722–R725 REFUTE** + **R732–R735 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T01:32:14Z | 88133.29748882844 | p3770 **R720+R721 REFUTE** + **R730+R731 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T01:25:30Z | 88215.69490882389 | p3769 **R719 REFUTE** + **R722–R725 N80** + **R729 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T01:15:45Z | 88296.85158235152 | p3768 **R712 REFUTE** + **R719/R720/R721 N80** + **R728 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T01:00:50Z | 88376.44652397608 | p3767 **R718 REFUTE** + **R727 TRAIN** + **R712 N80** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T00:52:21Z | 88459.51878187543 | p3766 **R717 REFUTE** + **R726 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
