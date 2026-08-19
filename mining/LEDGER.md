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
| $UTC | 82552.81158770059 | p3998 |
| Lium balance | **$82552.81158770059** | p3998 |
| cumulative mining spend | ~$129,196 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$72553** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (8 pods) | **~$405.70/h** · vs floor $833/h · gap **−$427.30/h** · B300/8×B200 stock=0 | p3998 |
| miner coldkey free | τ1257.618 | kept ≥τ50 |
| miner stake | **~59α / τ3.25** (under τ5 sweep) | p3985 |
| registrations / submissions | **11** / **11** (… **r861** **LOST** chal-00934) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-19T06:49:00Z | 82552.81158770059 | p3998 R891+R874 REFUTE→R896+R897 TRAIN (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T06:36:00Z | 82598.05862308426 | p3997 R891+R874 n80 + R835/R796 REFUTE→R894/R895 TRAIN (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T06:24:44Z | 82696.62365024321 | p3996 R852+R853 REFUTE→R835+R796 n80 (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T06:09:38Z | 82845.3017869402 | p3995 R880+R881 REFUTE→R852+R853 n80 (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T05:56:07Z | 82989.53496329929 | p3994 R879+R869 REFUTE→R880+R881 n80 (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T05:46:32Z | 83036.45406233937 | p3993 R870 REFUTE→R879 n80 + R869 Triton reload n80 (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T05:29:42Z | 83227.21664753786 | p3992 R854 REFUTE→R874 MERGE+wait n80 (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T05:24:47Z | 83273.06867031682 | p3991 R872/R873 REFUTE→R892/R893 TRAIN + R337 vera DL/wait R869/R870 (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T05:14:36Z | 83369.55475127233 | p3990 R854 SIZE_OK→chall+n80 lunar 6,7 (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T05:08:49Z | 83417.38974895934 | p3989 R871 REFUTE + R873 MERGE→n80 + R872 relaunch (no rent; stock 0); burn **~$405.70/h** |
