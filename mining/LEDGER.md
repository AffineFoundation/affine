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
| $UTC | 87576.51574428375 | p3798 |
| Lium balance | **$87576.51574428375** | p3798 |
| cumulative mining spend | ~$119,000 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$77577** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** · B300×8=0 | p3798 |
| miner coldkey free | τ1260.384 | after α→TAO→Lium (kept ≥τ50) |
| miner stake | **0** (r252 swept p3795) | p3795 |
| registrations / submissions | **10** / **10** (… **r637** chal-00829; **r683** chal-00860 **LOST**) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-18T04:45:54Z | 87576.51574428375 | p3798 **R745 MERGE→n80 ARMED** golden 4,5 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T04:42:11Z | 87617.13271789509 | p3797 **R747 MERGE→N80 LIVE** zesty 6,7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T04:36:31Z | 87697.93050158377 | p3796 **R744 REFUTE** → **R753 TRAIN** crown 6,7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T04:32:00Z | 87697.93050158377 | p3795 **α→TAO→Lium** r252 **89.96α → τ5.1067** (`lium fund` fail → `btcli transfer` to Lium ck); bal **$86668→$87698**; burn **~$331.45/h** |
| 2026-08-18T04:27:49Z | 86709.91904113453 | p3794 **R741 REFUTE** → **R752 TRAIN** + **R744 N80** (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T04:21:38Z | 86750.52887741415 | p3793 **R742+R743 REFUTE** → **R750+R751 TRAIN** R252 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T04:12:30Z | 86873.14562863595 | p3792 **R738 REFUTE** → **R749 TRAIN** lunar 6,7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T04:06:09Z | 86873.14562863595 | p3791 **R742 MERGE→N80 LIVE** R252 6,7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T03:59:40Z | 86954.00486330794 | p3790 **R738 MERGE→N80 LIVE** lunar 6,7 (no rent; B300×8=0); burn **~$331.45/h** |
| 2026-08-18T03:55:36Z | 86954.00486330794 | p3789 **R740 REFUTE** → **R748 TRAIN** (no rent; B300×8=0); burn **~$331.45/h** |
