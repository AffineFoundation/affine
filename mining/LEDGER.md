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
| $UTC | 86750.52887741415 | p3793 |
| Lium balance | **$86750.52887741415** | p3793 |
| cumulative mining spend | ~$119,827 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$76751** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** · B300×8=0 | p3793 |
| miner coldkey free | τ1260.384 | after α→TAO→Lium (kept ≥τ50) |
| miner stake | **~60.1α / τ3.41** r252 (below ~τ5 sweep) | p3781 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T04:21:38Z | 86750.52887741415 | p3793 **R742+R743 REFUTE** → **R750+R751 TRAIN** R252 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T04:12:30Z | 86873.14562863595 | p3792 **R738 REFUTE** → **R749 TRAIN** lunar 6,7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T04:06:09Z | 86873.14562863595 | p3791 **R742 MERGE→N80 LIVE** R252 6,7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T03:59:40Z | 86954.00486330794 | p3790 **R738 MERGE→N80 LIVE** lunar 6,7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T03:55:36Z | 86954.00486330794 | p3789 **R740 REFUTE** → **R748 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T03:50:41Z | 86994.53446327084 | p3788 **R739 REFUTE** → **R747 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T03:45:43Z | 87075.80785219138 | p3787 **R715 REFUTE** + **R740 N80** + **R716 RELAY** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T03:40:08Z | 87116.53062567447 | p3786 **R739 MERGE→N80 LIVE** zesty 6,7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T03:36:13Z | 87116.53062567447 | p3785 **R715 SCP→N80 LIVE** crown (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T03:17:30Z | 87279.71714707588 | p3784 **R734/R735/R737/R729/R730 REFUTE** → **R742–R746 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
