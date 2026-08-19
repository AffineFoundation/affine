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
| $UTC | 81010.52235174661 | p4016 |
| Lium balance | **$81010.52235174661** | p4016 |
| cumulative mining spend | ~$130,735 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$71011** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$306.66/h** · vs floor $833/h · gap **−$526.34/h** · B300/8×B200 stock=0 | p4016 |
| miner coldkey free | τ1257.618 | kept ≥τ50 |
| miner stake | **~59α / τ3.25** (under τ5 sweep) | p3985 |
| registrations / submissions | **11** / **11** (… **r861** **LOST** chal-00934) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-19T09:15:30Z | 81010.52235174661 | p4016 lunar R896+R897 + R888 R898 MERGE idle→n80 LOADING (no rent; stock 0); burn **~$306.66/h** |
| 2026-08-19T09:08:42Z | 81051.33014496622 | p4015 crown R901+R902 MERGE idle→n80 LOADING (no rent; stock 0); burn **~$306.66/h** |
| 2026-08-19T09:00:30Z | 81088.7874554946 | p4014 R899+R900 REFUTE→R910+R911 TRAIN; tore R337+R338 (−$99.04/h); burn **~$306.66/h** |
| 2026-08-19T08:52:27Z | 81190.11588804063 | p4013 R900 bad-LAUNCHED→chall :8003 (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T08:37:31Z | 81344.29018578386 | p4012 R899 MERGE→n80 relaunch (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T08:31:09Z | 81393.21198145708 | p4010 R338 king rsync crown→unblocked R882/R883 n80 (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T08:18:10Z | 81545.39811919727 | p4009 R894+R895 REFUTE→R908+R909 TRAIN (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T08:05:57Z | 81648.62234334697 | p4008 R338 MERGE idle→local king+R882/R883 n80 arm (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T07:59:11Z | 81750.59551374718 | p4007 R894 MERGE idle→n80 + R895 merge-repair→n80 R337 (no rent; stock 0); burn **~$405.70/h** |
| 2026-08-19T07:51:26Z | 81851.87347801791 | p4006 R890+R887+R889 REFUTE→R905+R906+R907 TRAIN (no rent; stock 0); burn **~$405.70/h** |
