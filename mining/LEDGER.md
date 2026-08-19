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
| $UTC | 81851.87347801791 | p4006 |
| Lium balance | **$81851.87347801791** | p4006 |
| cumulative mining spend | ~$129,895 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$71852** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (8 pods) | **~$405.70/h** · vs floor $833/h · gap **−$427.30/h** · B300/8×B200 stock=0 | p4006 |
| miner coldkey free | τ1257.618 | kept ≥τ50 |
| miner stake | **~59α / τ3.25** (under τ5 sweep) | p3985 |
| registrations / submissions | **11** / **11** (… **r861** **LOST** chal-00934) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-19T07:51:26Z | 81851.87347801791 | p4006 R890+R887+R889 REFUTE→R905+R906+R907 TRAIN (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T07:42:04Z | 81902.83733737194 | p4005 R877+R878 REFUTE→R903+R904 TRAIN (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T07:32:04Z | 82003.1950291492 | p4004 R885+R886 REFUTE→R901+R902 TRAIN + R887 n80 (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T07:24:10Z | 82106.76996975813 | p4003 R877+R878+R889+R890 MERGE idle→dual n80 (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T07:17:35Z | 82158.64292318153 | p4002 R885+R886 MERGE idle→dual n80 crown (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T07:12:46Z | 82209.8471142238 | p4001 R892+R893 REFUTE→R899+R900 TRAIN (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T07:03:03Z | 82303.32840314256 | p4000 R898 TRAIN R888 idle 5,6 (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T06:55:40Z | 82357.59590326155 | p3999 R892+R893 TRAIN_DONE→MERGE+wait n80 (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T06:49:00Z | 82552.81158770059 | p3998 R891+R874 REFUTE→R896+R897 TRAIN (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T06:36:00Z | 82598.05862308426 | p3997 R891+R874 n80 + R835/R796 REFUTE→R894/R895 TRAIN (no rent; stock 0); burn **~$405.70/h** |
