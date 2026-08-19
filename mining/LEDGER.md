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
| $UTC | 83273.06867031682 | p3991 |
| Lium balance | **$83273.06867031682** | p3991 |
| cumulative mining spend | ~$128,474 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$73273** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (8 pods) | **~$405.70/h** · vs floor $833/h · gap **−$427.30/h** · B300/8×B200 stock=0 | p3991 |
| miner coldkey free | τ1257.618 | kept ≥τ50 |
| miner stake | **~59α / τ3.25** (under τ5 sweep) | p3985 |
| registrations / submissions | **11** / **11** (… **r861** **LOST** chal-00934) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-19T05:24:47Z | 83273.06867031682 | p3991 R872/R873 REFUTE→R892/R893 TRAIN + R337 vera DL/wait R869/R870 (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T05:14:36Z | 83369.55475127233 | p3990 R854 SIZE_OK→chall+n80 lunar 6,7 (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T05:08:49Z | 83417.38974895934 | p3989 R871 REFUTE + R873 MERGE→n80 + R872 relaunch (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T05:00:43Z | 83462.04019233573 | p3988 R872 MERGE→chall+n80 LIVE R252 (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T04:53:27Z | 83559.0238706292 | p3987 R871 SIZE_OK→chall+n80 LIVE (affine_pkg sync; no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T04:38:17Z | 83701.36494035226 | p3986 R875/R876/R865 REFUTE→R889/R890/R891 TRAIN (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T04:28:18Z | 83796.4606360361 | p3985 R865 SIZE_OK→n80 LIVE lunar (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T04:19:59Z | 83890.22387945114 | p3984 R875+R876 MERGE brave (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T04:12:52Z | 83938.99680950548 | p3983 R888 king TP1 + R871 host-relay (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T04:03:52Z | 84032.521883819 | p3982 R864 REFUTE→R865 relay + R888 GRPO TRAIN (no rent; stock 0); burn **~$405.70/h** |
